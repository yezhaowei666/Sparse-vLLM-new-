"""Persistent CNN history with independent per-layer offloaded KV residency."""
import torch
import triton as tr
from torch import nn
from sparsevllm.kernels.triton import attnpredict as k
from sparsevllm.kernels.triton import leasesparse as tail
from .omnikv.manager import OmniKVCacheManager
from sparsevllm.kernels.triton.omnikv_lru import plan_lru
from sparsevllm.operators.indexed_host_copy import gather_rows


class AttnPredictCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,3,padding=1)
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.conv3=nn.Conv1d(32,1,1)

    def forward(self,x,valid):
        x=x.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        x=self.conv1(x).relu_()
        x.mul_(valid[:,None,None,:])
        x=self.conv2(x).relu_().mean(dim=2).contiguous()
        return self.conv3(x).squeeze(1)


class AttnPredictCacheManager(OmniKVCacheManager):
    independent_layer_cache = True
    extra_selection_tokens = 1

    @staticmethod
    def offload_setting(config):
        return True

    def __init__(self,config,parallel_context,*,allocation_budget_bytes=None):
        super().__init__(config,parallel_context,allocation_budget_bytes=allocation_budget_bytes)
        L,R,H,N=self.num_layers-2,self.max_buffer_rows,self.hf_config.num_attention_heads,tr.cdiv(self.max_model_len,16)
        self.blocks=N
        self.columns=max(3,N-8)
        self.hot_count=min(config.decode_keep_tokens//16,self.columns)
        self.history=torch.zeros((L,R,H,64,N),dtype=torch.float16,device=self.device)
        self.cursor=torch.zeros((L,R),dtype=torch.int32,device=self.device)
        self.keep=torch.zeros((L,R,4096),dtype=torch.int32,device=self.device)
        self.keep_lens=torch.zeros((L,R),dtype=torch.int32,device=self.device)
        self.positions=torch.zeros((L,R,4097),dtype=torch.int32,device=self.device)
        self.slots=torch.zeros_like(self.positions)
        self.view_lens=torch.zeros((L,R),dtype=torch.int32,device=self.device)
        self.scores=torch.empty((L,R,H,4097),dtype=torch.float32,device=self.device)
        self.pool=torch.zeros((L,R,H,N),dtype=torch.float32,device=self.device)
        self.cnn=AttnPredictCNN().to(device=self.device,dtype=torch.float16,memory_format=torch.channels_last).eval()
        self.cnn.load_state_dict(torch.load(config.attnpredict_model_path,map_location=self.device,weights_only=True))
        self.cnn.forward=torch.compile(self.cnn.forward,fullgraph=True,options={"triton.cudagraphs":False})
        # Target 32 MB for each layer's largest FP16 activation; process at least one head.
        self.cnn_head_chunk=max(1,32*1024*1024//(32*64*self.columns*2))
        self.prediction_streams=[torch.cuda.Stream(device=self.device) for _ in range(L)]
        self.prediction_ready=[torch.cuda.Event(external=True) for _ in range(L)]
        self.attention_ready=[torch.cuda.Event(external=True) for _ in range(L)]
        self.prediction_meta=torch.zeros((L,3,R),dtype=torch.int32,device=self.device)
        self.prediction_graphs={}
        self.prediction_replays=0
        self.predictions_submitted=set()
        for ready in self.prediction_ready:ready.record(torch.cuda.current_stream(self.device))
        self.prefetch_positions=torch.empty((L,R,4097),dtype=torch.int32,device=self.device)
        self.prefetch_slots=torch.empty_like(self.prefetch_positions)
        self.prefetch_lens=torch.empty((L,R),dtype=torch.int32,device=self.device)
        self.prefetch_view=torch.empty((L,R,self.selected_capacity),dtype=torch.int32,device=self.device)
        self._prefill_seqs=[]

    def _prepare_prefill(self,seqs):
        self._prefill_seqs=seqs
        return super()._prepare_prefill(seqs)

    @torch.inference_mode()
    def predict(self,layer,rows,lens,writes):
        B,H,C=rows.numel(),self.hf_config.num_attention_heads,self.columns
        x=torch.empty((B*H,64,C),device=self.device,dtype=torch.float16)
        valid=torch.empty((B*H,C),device=self.device,dtype=torch.bool)
        k.pack[(B,H,64)](self.history[layer-2],self.cursor[layer-2],rows,lens,x,valid,self.blocks,H,C,tr.next_power_of_2(C))
        predicted=torch.empty((B*H,C),device=self.device,dtype=torch.float16)
        # Heads are independent; bound each layer's concurrent convolution workspace.
        for h in range(0,B*H,self.cnn_head_chunk):
            end=min(h+self.cnn_head_chunk,B*H)
            predicted[h:end].copy_(self.cnn(x[h:end],valid[h:end]))
        scores=predicted.view(B,H,C).float().amax(dim=1)
        scores.masked_fill_(~valid.view(B,H,C)[:,0],-float('inf'))
        top=scores.topk(self.hot_count,dim=-1,sorted=False).indices
        k.select_positions[(B,)](top,rows,lens,writes,self.keep[layer-2],self.keep_lens[layer-2],self.hot_count,4096,4096,num_warps=8)

    @torch.inference_mode()
    def collect_prefill_attention_score(self,layer_idx,q,view,*,b_start_loc,chunk_lens,attention_lse=None):
        if layer_idx<2:return
        H,N=q.shape[1],self.blocks
        for b,seq in enumerate(self._prefill_seqs):
            if seq.num_prefilled_tokens+seq.current_chunk_size<seq.num_tokens:continue
            torch.cuda.current_stream(self.device).wait_event(self.prediction_ready[layer_idx-2])
            rows=view.meta.req_indices[b:b+1];lens=view.meta.context_lens[b:b+1];chunks=chunk_lens[b:b+1]
            maxima=torch.empty((1,H,64,N),device=self.device,dtype=torch.float32)
            lses=torch.empty_like(maxima);norm=torch.empty((H*64,),device=self.device,dtype=torch.float32)
            keys=view.payload.k_cache
            tail.tail_logits[(1,H,tr.cdiv(N,8))](q,keys,view.meta.active_slots,rows,lens,b_start_loc[b:b+1],chunks,maxima,lses,
                N,view.meta.active_slots.shape[1],H,self.num_kv_heads,self.head_dim,*q.stride()[:2],*keys.stride()[:2],self.head_dim**-.5)
            tail.tail_norm[(H*64,)](lses,norm,N,tr.next_power_of_2(N))
            k.seed[(1,H,64)](self.history[layer_idx-2],maxima,norm,rows,lens,chunks,self.cursor[layer_idx-2],N,H,tr.next_power_of_2(N))
            self.snapshot_prediction(layer_idx,rows,lens,rows)
            stream=self.prediction_streams[layer_idx-2]
            with torch.cuda.stream(stream):
                stream.wait_event(self.attention_ready[layer_idx-2])
                r,n,w=self.prediction_meta[layer_idx-2,:,:1]
                self.predict(layer_idx,r,n,w)
                self.prefetch_selected(layer_idx,r,n,w)
                self.prediction_ready[layer_idx-2].record(stream)

    def prefetch_selected(self,layer,rows,lens,writes):
        B=rows.numel();g=layer-2
        k.view[(B,)](self.keep[layer-2],self.keep_lens[layer-2],rows,lens,writes,
            self.buffer_req_to_token_slots,self.prefetch_positions[g],self.prefetch_slots[g],
            self.prefetch_lens[g],self.max_model_len,4097,8192,APPEND_CURRENT=False)
        data=self.lru.metadata[self.lru.layer_groups[layer]]
        plan_lru(*data,self.prefetch_slots[g],self.selected_rows[:B],rows,self.prefetch_lens[g,:B],writes,self.prefetch_view[g])
        for component,dest in enumerate(self.selected_staging[layer]):
            gather_rows(self.attention_cache_storage.pointers[layer],dest,self.prefetch_slots[g],
                self.selected_rows[:B],self.prefetch_lens[g,:B],capacity=self.selected_capacity,
                component=component,plan=data[4],miss_tokens=data[5],miss_counts=data[6],
                slot_map=self.attention_cache_storage.host_slot_map,block_budget=32)

    def snapshot_prediction(self,layer,rows,lens,writes):
        for dest,source in zip(self.prediction_meta[layer-2],(rows,lens,writes)):
            dest[:rows.numel()].copy_(source)
        self.attention_ready[layer-2].record(torch.cuda.current_stream(self.device))

    def prediction_step(self,layer,batch):
        g=layer-2;H=self.hf_config.num_attention_heads;N=self.blocks
        rows,lens,writes=self.prediction_meta[g,:,:batch]
        self.pool[g].zero_()
        k.feedback[(batch,H)](self.scores[g],self.positions[g],self.view_lens[g],self.pool[g],rows,writes,N,H,4097,8192,self.head_dim**-.5)
        k.append[(batch,H,tr.cdiv(N,256))](self.pool[g],self.history[g],self.cursor[g],rows,writes,N,H,256)
        k.advance[(1,)](self.cursor[g],rows,writes,batch,tr.next_power_of_2(batch))
        self.predict(layer,rows,lens,writes)
        self.prefetch_selected(layer,rows,lens,writes)

    def submit_prediction(self,layer,batch):
        g=layer-2;stream=self.prediction_streams[g]
        with torch.cuda.stream(stream):
            if self.config.decode_graph:
                key=(layer,batch)
                if key not in self.prediction_graphs:
                    stream.wait_event(self.attention_ready[g])
                    self.prediction_step(layer,batch)
                    stream.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        stream.wait_event(self.attention_ready[g])
                        self.prediction_step(layer,batch)
                        self.prediction_ready[g].record(stream)
                    self.prediction_graphs[key]=graph
                self.prediction_graphs[key].replay()
                self.prediction_replays+=1
            else:
                stream.wait_event(self.attention_ready[g])
                self.prediction_step(layer,batch)
                self.prediction_ready[g].record(stream)
        self.predictions_submitted.add(layer)

    def _prepare_decode_graph_buffers(self,seqs,**kwargs):
        self.predictions_submitted.clear()
        return super()._prepare_decode_graph_buffers(seqs,**kwargs)

    def on_forward_end(self,seqs,is_prefill):
        if not is_prefill:
            for layer in range(2,self.num_layers):
                if layer not in self.predictions_submitted:
                    self.submit_prediction(layer,self.layer_batch_state.req_indices.numel())
        return super().on_forward_end(seqs,is_prefill)

    def free_seq(self,seq_id):
        for ready in self.prediction_ready:torch.cuda.current_stream(self.device).wait_event(ready)
        row=self.seq_id_to_row[seq_id]
        self.history[:,row].zero_();self.cursor[:,row].zero_();self.keep_lens[:,row].zero_()
        return super().free_seq(seq_id)

    def decode_graph_keepalive_tensors(self):
        return super().decode_graph_keepalive_tensors()+[v for v in vars(self).values() if isinstance(v,torch.Tensor)]+list(self.cnn.parameters())

    def memory_accounting(self):
        result=super().memory_accounting()
        result['attnpredict_async']={'streams':len(self.prediction_streams),
            'prediction_graphs':len(self.prediction_graphs),'prediction_replays':self.prediction_replays,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device)}
        return result
