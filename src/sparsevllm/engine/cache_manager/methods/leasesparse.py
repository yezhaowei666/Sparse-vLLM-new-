"""Lease-owned GPU state with CPU history and a shared full-prefill view."""
from pathlib import Path
import torch
import triton as tr
from sparsevllm.utils.context import get_context

from sparsevllm.kernels.triton import leasesparse as kernels
from .omnikv.manager import OmniKVCacheManager
from .omnikv.storage import OmniKVStorage, payload_tensors
from .omnikv.capacity import fit_omnikv_host_slots
from ..standard import StandardCacheManager


def lease_pool_bytes(layers, rows, per_layer):
    return layers*rows*(4096*(per_layer+16)), per_layer+layers*rows*4+8


class LeaseSparseCacheManager(OmniKVCacheManager):
    @staticmethod
    def offload_setting(config):
        return config.enable_leasesparse_offload

    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        super().__init__(config, parallel_context, allocation_budget_bytes=allocation_budget_bytes)
        self.sources = tuple(config.leasesparse_sources)
        self.ends = self.sources[1:] + (self.num_layers,)
        self.groups = {l: g for g,(a,z) in enumerate(zip(self.sources,self.ends)) for l in range(a,z)}
        self.blocks = tr.cdiv(self.max_model_len,16)
        G,R,H,N = len(self.sources),self.max_buffer_rows,self.hf_config.num_attention_heads,self.blocks
        def zeros(shape,dtype=torch.int32):
            return torch.zeros(shape,dtype=dtype,device=self.device)
        self.predictor = None
        if config.leasesparse_predictor_path:
            from sparsevllm.engine.sparse_methods.leasesparse import MassQueryPredictor
            from sparsevllm.models.rope import resolve_rope_theta
            checkpoint=torch.load(config.leasesparse_predictor_path,map_location='cpu',weights_only=True)
            trained=checkpoint['config']
            if (trained['architecture'],trained['history'],trained['horizon'],trained['rope_theta']) != ('mass',8,4,resolve_rope_theta(self.hf_config)):
                raise ValueError('LeaseSparse requires the mass H8/F4 predictor with matching RoPE.')
            self.predictor=MassQueryPredictor(rope_theta=trained['rope_theta']).to(device=self.device,dtype=torch.float32).eval()
            self.predictor.load_state_dict(checkpoint['model'])
            self.predictor.requires_grad_(False)
            self.predictor.predict_with_mass=torch.compile(self.predictor.predict_with_mass,fullgraph=True,
                options={"triton.cudagraphs":False})
            self.query_history=zeros((G,R,H,8,self.head_dim),torch.bfloat16)
            self.key_sums=zeros((G,R,self.num_kv_heads,N,self.head_dim),torch.float32)
            self.key_tail=zeros((G,R,self.num_kv_heads,80,self.head_dim),torch.bfloat16)
            self.key_positions=torch.full((G,R,self.num_kv_heads),-1,device=self.device,dtype=torch.int32)
            self.predict_this_step=False
            self.prediction_commit=False
            self.predictor_layer_ids=torch.arange(G,device=self.device)
            self.next_prediction={}
        else:
            self.ema = zeros((G,R,H,N),torch.float32)
            self.seen = zeros((G,R,N))
        # Two logical selections, sharing a single physical KV pool.
        self.hot = torch.full((G,R,2,248),-1,dtype=torch.int32,device=self.device)
        self.starts=zeros((G,R))
        self.pending_starts=zeros((G,R))
        self.pending=zeros((G,R))
        self.bank=zeros((G,R))
        self.refresh=zeros((G,))
        self.lease_counts=zeros((G,2))
        self.positions=zeros((G,R,4096))
        self.slots=zeros((G,R,4096))
        self.lengths=zeros((G,R))
        self.next_positions=zeros((G,R,4096))
        self.next_slots=zeros((G,R,4096))
        self.next_lengths=zeros((G,R))
        if self.predictor is None:
            self.pool=zeros((R,H,N),torch.float32)
            self.observed=zeros((R,N))
        self.reduced=zeros((G,R,N) if self.predictor is not None else (R,N),torch.float32)
        if self.predictor is None:
            self.score_buffer=torch.empty((R,H,4096),dtype=torch.float32,device=self.device)
        self.lease_stream=torch.cuda.Stream(device=self.device)
        self.lease_ready=[torch.cuda.Event() for _ in self.sources]
        if config.leasesparse_trace:
            self.layer_uses=torch.full((self.num_layers,R,2),-1,dtype=torch.int32,device=self.device)
            self.completion_counter=zeros((1,))
            self.completed=zeros((G,))
        self.trace_events=[]
        self.trace=[]
        self._trace_previous={}
        self._prefill_seqs=[]

    def allocate_kv_cache(self):
        if not self.offload_enabled:
            return StandardCacheManager.allocate_kv_cache(self)
        original=self.attention_cache_storage
        available,per_layer=self._get_available_slots_info()
        R,L,P=self.max_buffer_rows,self.num_kv_layers,4096
        # One physical pool; lease generations only own position indices.
        fixed,per_slot=lease_pool_bytes(L,R,per_layer)
        slots=min((available-fixed)//per_slot,self.max_model_len*R)
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        slots=fit_omnikv_host_slots(slots,int(mem['MemAvailable'].split()[0])*1024//2,
                                 [h*d*original.dtype.itemsize for h,d in OmniKVStorage.payload_shapes(original)],L,0)
        if slots<=0 or (getattr(self.config,'startup_cache_phase','production')!='profiling' and slots<self.max_model_len):
            raise MemoryError('LeaseSparse CPU/GPU cache capacity cannot hold one request.')
        self.config.num_kvcache_slots=int(slots)
        storage=OmniKVStorage(original,num_layers=L,num_slots=slots,full_layers=(),device=self.device)
        self.attention_cache_storage=storage
        self.kv_cache=None
        def allocate(count):
            return tuple(torch.empty(count,*s,dtype=storage.dtype,device=self.device) for s in storage.shapes)
        self.prefill_staging=allocate(slots)
        self.selected_staging={l:allocate(R*P) for l in range(L)}
        self.cache_keys=torch.full((L,2,R*P),-1,dtype=torch.int32,device=self.device)
        self.directory=torch.full((L,R*slots),-1,dtype=torch.int32,device=self.device)
        self.resident_slots=torch.zeros((L,R,P),dtype=torch.int32,device=self.device)
        self.free_workspace=torch.empty((L,R,P),dtype=torch.int32,device=self.device)
        self.selected_rows=torch.arange(R,dtype=torch.int32,device=self.device)
        self.selected_slots=self.resident_slots[0]
        self.prefetch_stream=torch.cuda.Stream(device=self.device)
        self.layer_ready={l:torch.cuda.Event() for l in range(L)}
        self.selection_done=torch.cuda.Event()
        self._selection_pending=False
        self._prefill_next_layer={l:l+1 for l in range(L-1)}

    def _prepare_prefill(self,seqs):
        self._prefill_seqs=seqs
        return super()._prepare_prefill(seqs)

    def collect_prefill_attention_score(self,layer_idx,q,view,*,b_start_loc,chunk_lens,attention_lse=None):
        from sparsevllm.engine.sparse_methods.leasesparse import seed_prefill_lease

        if self.predictor is None:
            seed_prefill_lease(self,layer_idx,q,view,b_start_loc=b_start_loc,chunk_lens=chunk_lens)
        else:
            offset=0
            for b,seq in enumerate(self._prefill_seqs):
                start,count=seq.num_prefilled_tokens,seq.current_chunk_size
                rows=view.meta.req_indices[b:b+1];lens=view.meta.context_lens[b:b+1]
                row=self.seq_id_to_row[seq.seq_id]
                kernels.predictor_queries[(1,q.shape[1])](q[offset:],self.query_history[layer_idx],rows,lens,rows,
                    q.shape[1],self.head_dim,*q.stride()[:2],True,row,start,count)
                if start+count==seq.num_tokens:
                    self.submit_prediction(layer_idx,rows,lens,start+count,initial=True)
                    self.next_prediction[seq.seq_id]=start+count+4
                offset+=count

    def on_kv_stored(self,layer_idx,k,slot_mapping):
        if self.predictor is None:return
        s=self.layer_batch_state
        if get_context().is_prefill:
            offset=0
            for seq in self._prefill_seqs:
                start,count=seq.num_prefilled_tokens,seq.current_chunk_size
                row=self.seq_id_to_row[seq.seq_id]
                kernels.predictor_keys_prefill[(tr.cdiv(start%16+count,16),self.num_kv_heads)](
                    k[offset:],self.key_sums[layer_idx],self.key_tail[layer_idx],self.key_positions[layer_idx],row,start,count,
                    self.blocks,self.num_kv_heads,self.head_dim,*k.stride()[:2])
                offset+=count
        else:
            kernels.predictor_keys_decode[(s.req_indices.numel(),self.num_kv_heads)](
                k,self.key_sums[layer_idx],self.key_tail[layer_idx],self.key_positions[layer_idx],s.req_indices,s.context_lens,slot_mapping,
                self.blocks,self.num_kv_heads,self.head_dim,*k.stride()[:2])

    def record_decode_query(self,layer_idx,q):
        if self.predictor is None:return
        s=self.layer_batch_state
        kernels.predictor_queries[(s.req_indices.numel(),q.shape[1])](q,self.query_history[layer_idx],
            s.req_indices,s.context_lens,s.slot_mapping,q.shape[1],self.head_dim,*q.stride()[:2])

    @torch.inference_mode()
    def score_prediction(self,layers,rows,lens,columns):
        batch=rows.numel()
        H,D=self.hf_config.num_attention_heads,self.head_dim
        q=torch.empty((len(layers)*batch,H,8,D),device=self.device,dtype=torch.float32)
        partial=torch.empty((len(layers)*batch,self.num_kv_heads,D),device=self.device,dtype=torch.float32)
        for i,layer in enumerate(layers):
            kernels.predictor_pack[(batch,H)](self.query_history[layer],self.key_tail[layer],rows,lens,
                q[i*batch:],partial[i*batch:],H,self.num_kv_heads,D)
        positions=(lens[:,None].long()+torch.arange(-8,0,device=self.device)).repeat(len(layers),1)
        layer_ids=self.predictor_layer_ids[layers[0]:layers[-1]+1].repeat_interleave(batch)
        predicted,mass=self.predictor.predict_with_mass(q,positions,layer_ids)
        for i,layer in enumerate(layers):
            query=predicted[i*batch:(i+1)*batch]
            scores=torch.empty((batch,4,H,columns),device=self.device,dtype=torch.float32)
            kernels.predictor_logits[(batch,self.num_kv_heads,tr.cdiv(columns,32))](query,self.key_sums[layer],partial[i*batch:],
                rows,lens,scores,self.blocks,columns,H,self.num_kv_heads,D,*query.stride())
            kernels.predictor_probability[(batch,4*H)](scores,mass[i*batch:],lens,columns,H,tr.next_power_of_2(columns))
            kernels.predictor_reduce[(batch,tr.cdiv(columns,32))](scores,self.reduced[layer],columns,self.blocks,H)

    @torch.inference_mode()
    def submit_prediction(self,layer,rows,lens,max_length,*,initial=False):
        # Prefill initialization; decode prediction is part of the main graph.
        self.score_prediction((layer,),rows,lens,min(self.blocks,tr.cdiv(max_length,4096)*256))
        self.refresh[layer:layer+1].fill_(1)
        kernels.select[(rows.numel(),)](self.reduced[layer],self.hot[layer],self.bank[layer],self.pending[layer],self.pending_starts[layer],
            rows,lens,rows,self.refresh[layer:layer+1],self.lease_counts[layer],self.blocks,248,
            tr.next_power_of_2(self.blocks),initial,num_warps=8)

    def decode_graph_path_id(self,seqs=()):
        path=super().decode_graph_path_id(seqs)
        if self.predictor is None:return path
        self.predict_this_step=any(seq.decode_input_position+1>=self.next_prediction[seq.seq_id] for seq in seqs)
        return path+(':predict' if self.predict_this_step else '')

    def decode_graph_capture_paths(self,seqs):
        path=super().decode_graph_path_id(seqs)
        return (path,path+':predict') if self.predictor is not None else (path,)

    def prepare_decode_graph_in(self,state):
        super().prepare_decode_graph_in(state)
        if self.predictor is not None:
            self.predict_this_step=state.contract.topology_path_id.endswith(':predict')
            # Warmup computes the network but must not publish a future lease.
            self.prediction_commit=not state.capture_warmup

    def make_view(self,g,*,next_lease=False):
        s=self.layer_batch_state
        B=s.req_indices.numel()
        p=self.next_positions if next_lease else self.positions
        slots=self.next_slots if next_lease else self.slots
        lens=self.next_lengths if next_lease else self.lengths
        kernels.view[(B,)](self.hot[g],self.bank[g],self.starts[g],self.pending_starts[g],s.req_indices,s.context_lens,s.slot_mapping,
                          self.buffer_req_to_token_slots,p[g],slots[g],lens[g],self.max_model_len,4096,248,
                          next_lease,self.refresh[g:g+1],4096)
        return p[g,:B],slots[g,:B],lens[g,:B]

    def recall(self,layer,*,next_lease=False):
        g=self.groups[layer]
        s=self.layer_batch_state
        B=s.req_indices.numel()
        p=self.next_positions if next_lease else self.positions
        lens=self.next_lengths if next_lease else self.lengths
        current=payload_tensors(self._current_writes[layer]) if not next_lease else self.selected_staging[layer]
        kernels.plan_pool[(B,)](self.cache_keys[layer,0],self.directory[layer],self.buffer_req_to_token_slots,
            s.req_indices,s.context_lens,s.slot_mapping,p[g],lens[g],
            self.resident_slots[layer],self.free_workspace[layer],self.refresh[g:g+1],
            self.config.num_kvcache_slots,self.max_model_len,4096,next_lease,num_warps=8)
        for c,(dest,cur) in enumerate(zip(self.selected_staging[layer],current)):
            kernels.recall[(B,min(256,max(1,32//B)) if next_lease else 256)](self.attention_cache_storage.pointers[layer],dest,self.cache_keys[layer,c],
                self.buffer_req_to_token_slots,s.req_indices,s.context_lens,s.slot_mapping,
                p[g],lens[g],self.resident_slots[layer],cur,self.refresh[g:g+1],
                self.max_model_len,4096,self.num_kv_heads*self.head_dim,next_lease,c,
                tr.next_power_of_2(self.num_kv_heads*self.head_dim),cur.stride(0))

    def get_layer_compute_payload(self,layer_idx,active_slots,req_indices,context_lens,selection=None):
        if not self.offload_enabled:
            return StandardCacheManager.get_layer_compute_payload(self,layer_idx,active_slots,req_indices,context_lens,selection)
        if layer_idx not in self.sources:
            torch.cuda.current_stream(self.device).wait_event(self.lease_ready[self.groups[layer_idx]])
        self.recall(layer_idx)
        return (self.attention_cache_storage.make_payload(self.selected_staging[layer_idx]),
                self.resident_slots[layer_idx,:req_indices.numel()],self.selected_rows[:req_indices.numel()],context_lens)

    def on_forward_end(self,seqs,is_prefill):
        if self.predictor is not None and not is_prefill and self.predict_this_step:
            for seq in seqs:self.next_prediction[seq.seq_id]=seq.decode_input_position+5
        torch.cuda.current_stream(self.device).wait_stream(self.lease_stream)
        if self.config.leasesparse_trace and not is_prefill:
            completed=torch.cuda.Event()
            completed.record(torch.cuda.current_stream(self.device))
            completed.synchronize()
            starts=self.starts.cpu().tolist()
            pending=self.pending.cpu().tolist()
            next_starts=self.pending_starts.cpu().tolist()
            flags=self.refresh.cpu().tolist()
            positions=self.positions.cpu()
            lengths=self.lengths.cpu().tolist()
            uses=self.layer_uses.cpu().tolist()
            serials=self.completed.cpu().tolist()
            for g,source in enumerate(self.sources):
                if flags[g]:
                    self.trace_events.append(dict(event=serials[g],source=source,kind="preparation_complete",
                        selection_steps={seq.seq_id:seq.decode_input_position-seq.num_prompt_tokens+1 for seq in seqs}))
            for b,seq in enumerate(seqs):
                row=self.seq_id_to_row[seq.seq_id]
                step=seq.decode_input_position-seq.num_prompt_tokens+1
                for g,source in enumerate(self.sources):
                    start=starts[g][row]
                    key=(seq.seq_id,g)
                    previous=self._trace_previous.get(key,(-1,-1))
                    switched=previous[0]!=start
                    lease_id=previous[1]+int(switched)
                    self._trace_previous[key]=(start,lease_id)
                    self.trace.append(dict(request=seq.seq_id,source=source,step=step,
                        lease_id=lease_id, trigger_step=step if flags[g] else None,
                        selection_step=step if flags[g] else None,
                        submit_step=step if flags[g] else None,
                        commit_step=step if switched else None,
                        switch_step=step if switched else None,
                        layer_first_use={l:uses[l][row][1]-seq.num_prompt_tokens+1 for l in range(source,self.ends[g])},
                        lease_start=start,pending_start=next_starts[g][row] if pending[g][row] else None,
                        layers=list(range(source,self.ends[g])),
                        positions=positions[g,b,:lengths[g][b]].tolist(),
                        completion_event=serials[g] if flags[g] else None, preparation_complete=True))
        return super().on_forward_end(seqs,is_prefill)

    def free_seq(self,seq_id):
        row=self.seq_id_to_row[seq_id]
        for g in range(len(self.sources)):self._trace_previous.pop((seq_id,g),None)
        if self.config.leasesparse_trace:self.layer_uses[:,row].fill_(-1)
        if self.predictor is None:
            self.ema[:,row].zero_()
            self.seen[:,row].zero_()
        else:
            self.query_history[:,row].zero_()
            self.key_tail[:,row].zero_()
            self.key_positions[:,row].fill_(-1)
            self.next_prediction.pop(seq_id,None)
            # Every new block starts by overwriting its sum; old blocks are masked by length.
        self.hot[:,row].fill_(-1)
        for x in (self.starts,self.pending_starts,self.pending,self.bank):x[:,row].zero_()
        if self.offload_enabled:
            self.cache_keys[:,:,row*4096:(row+1)*4096].fill_(-1)
            n=self.config.num_kvcache_slots
            self.directory[:,row*n:(row+1)*n].fill_(-1)
        return super().free_seq(seq_id)

    def decode_graph_keepalive_tensors(self):
        return super().decode_graph_keepalive_tensors()+[v for v in vars(self).values() if isinstance(v,torch.Tensor)]+(list(self.predictor.parameters()) if self.predictor is not None else [])

    def memory_accounting(self):
        result=super().memory_accounting()
        if self.offload_enabled:
            result["leasesparse_gpu_only_kv_bytes"]=result.pop("omnikv_gpu_only_kv_bytes")
        counts=self.lease_counts.cpu().tolist()
        result['leasesparse_counts']={'decode_refresh_per_source':[v[0] for v in counts],
                                     'initial_selection_per_source':[v[1] for v in counts]}
        return result
