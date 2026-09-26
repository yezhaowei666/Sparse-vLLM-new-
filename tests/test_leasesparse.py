"""Independent probability/EMA, lease timing, and recall reference checks."""
import unittest
import ast
from types import MethodType
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import triton as tr
from sparsevllm.kernels.triton import leasesparse as k
from sparsevllm.operators.indexed_host_copy import make_pointer_table
from sparsevllm.engine.cache_manager.base import ExplicitKVPayload


class LeaseKernels(unittest.TestCase):
    def test_selection_topk_matches_stable_sort_and_graph_replay(self):
        # Partial selection must preserve ties, short histories, masked rows,
        # and pending-bank publication; the legacy random oracle lacks these.
        torch.manual_seed(17)
        for N in (7, 257, 7563):
            B,K=3,248
            ints=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
            scores=torch.randint(0,4,(B,N),device='cuda').float()
            hot=torch.full((B,2,K),-9,device='cuda',dtype=torch.int32)
            rows=ints([2,0,1]);bank=ints([1,0,0]);pending=ints([0,0,0]);starts=ints([0,0,0])
            lengths=ints([N*16,N*16-13,N*16]);writes=ints([0,1,-1]);refresh=ints([1]);counts=ints([0,0])
            def run():
                k.select[(B,)](scores,hot,bank,pending,starts,rows,lengths,writes,refresh,counts,
                    N,K,tr.next_power_of_2(N),False,num_warps=8)
            run()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
            for value in (0,3):
                scores[0].fill_(value)
                graph.replay()
                for b,row in enumerate([2,0]):
                    end=tr.cdiv(int(lengths[b]),16)-4
                    expected=torch.full((K,),-1,device='cuda',dtype=torch.int32)
                    chosen=torch.argsort(scores[b,4:max(4,end)],descending=True,stable=True)[:K]+4
                    expected[:chosen.numel()]=chosen.sort().values.int()
                    torch.testing.assert_close(hot[row,1-int(bank[row])],expected)
                    self.assertEqual(int(pending[row]),1)
                    self.assertEqual(int(starts[row]),int(lengths[b]))
                self.assertTrue(bool((hot[1]==-9).all()))
            before=hot.clone();refresh.zero_();graph.replay()
            torch.testing.assert_close(hot,before)

    def test_prefill_probability_and_ema(self):
        torch.manual_seed(31)
        for n,take in ((31,7),(129,64),(513,19)):
            H,KH,D=4,2,32;N=tr.cdiv(n,16)
            q=torch.randn(take,H,D,device='cuda',dtype=torch.bfloat16)
            keys=torch.randn(n,KH,D,device='cuda',dtype=torch.bfloat16)
            table=torch.randperm(n,device='cuda',dtype=torch.int32)[None]
            ints=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
            rows=ints([0]);lens=ints([n]);starts=ints([0]);chunks=ints([take])
            maxima=torch.empty(1,H,64,N,device='cuda');lse=torch.empty_like(maxima);norm=torch.empty(H*64,device='cuda')
            state=torch.zeros(1,H,N,device='cuda');seen=torch.zeros(1,N,device='cuda',dtype=torch.int32)
            k.tail_logits[(1,H,tr.cdiv(N,8))](q,keys,table,rows,lens,starts,chunks,maxima,lse,N,n,H,KH,D,*q.stride()[:2],*keys.stride()[:2],D**-.5)
            k.tail_norm[(H*64,)](lse,norm,N,tr.next_power_of_2(N))
            k.tail_ema[(1,H,tr.cdiv(N,128))](state,seen,maxima,norm,rows,lens,chunks,N,H,128,enable_fp_fusion=False)
            expanded=keys[table[0].long()].repeat_interleave(H//KH,dim=1)
            logits=torch.einsum('thd,nhd->htn',q.float(),expanded.float())*D**-.5
            logits.masked_fill_(torch.arange(n,device='cuda')[None,None,:]>torch.arange(n-take,n,device='cuda')[None,:,None],float('-inf'))
            p=logits.softmax(-1);p=torch.nn.functional.pad(p,(0,N*16-n)).reshape(H,take,N,16).amax(-1)
            expected=torch.zeros(H,N,device='cuda');observed=torch.zeros(N,device='cuda',dtype=torch.bool)
            for t in range(take):
                mask=torch.arange(N,device='cuda')*16<=n-take+t
                expected=torch.where(mask,torch.where(observed,expected*.8+p[:,t]*.2,p[:,t]),expected)
                observed|=mask
            torch.testing.assert_close(state[0],expected,atol=2e-5,rtol=2e-3)
            self.assertTrue(torch.equal(seen[0].bool(),observed))

    def test_decode_pool_and_update(self):
        torch.manual_seed(4);B,H,N,P=2,4,37,512
        scores=torch.randn(B,H,P,device='cuda');positions=torch.stack([torch.randperm(N*16,device='cuda')[:P] for _ in range(B)]).int()
        lens=torch.tensor([P,301],device='cuda',dtype=torch.int32);rows=torch.tensor([1,0],device='cuda',dtype=torch.int32)
        refresh=torch.ones(1,device='cuda',dtype=torch.int32);pool=torch.zeros(B,H,N,device='cuda');obs=torch.zeros(B,N,device='cuda',dtype=torch.int32)
        state=torch.rand(B,H,N,device='cuda');seen=torch.ones(B,N,device='cuda',dtype=torch.int32);expected=state.clone()
        k.pool_decode[(B,H)](scores,positions,lens,pool,obs,refresh,P,N,H,*scores.stride()[:2],P,.125)
        for b,l in enumerate([P,301]):
            prob=(scores[b,:,:l]*.125).softmax(-1);v=torch.zeros(H,N,device='cuda')
            v.scatter_reduce_(1,(positions[b,:l]//16).long()[None].expand(H,-1),prob,reduce='amax')
            mask=torch.zeros(N,device='cuda',dtype=torch.bool);mask[positions[b,:l]//16]=True
            expected[1-b,:,mask]=expected[1-b,:,mask]*.8+v[:,mask]*.2
        k.ema[(B,H,1)](state,seen,pool,obs,rows,refresh,N,H,128,enable_fp_fusion=False)
        torch.testing.assert_close(state,expected,atol=2e-7,rtol=2e-5)
        refresh.zero_();old=state.clone();k.ema[(B,H,1)](state,seen,pool,obs,rows,refresh,N,H,128,enable_fp_fusion=False)
        self.assertTrue(torch.equal(old,state))

    def test_lease_boundaries_and_view(self):
        for period in (1,4,16):
            B,N,P=2,8200,4096;blocks=tr.cdiv(N,16)
            ints=lambda shape,value=0:torch.full(shape,value,device='cuda',dtype=torch.int32)
            rows=ints((B,));rows.copy_(torch.arange(B,device='cuda'));writes=rows.clone()
            start=ints((B,));pendingstart=ints((B,),8001);pending=ints((B,),1);bank=ints((B,));refresh=ints((1,))
            hot=ints((B,2,248),-1);hot[:,1]=torch.arange(4,252,device='cuda')
            lens=ints((B,),8002);table=torch.arange(B*N,device='cuda',dtype=torch.int32).reshape(B,N)
            pos=ints((B,P));slots=ints((B,P));vl=ints((B,))
            trigger=[];switch=[]
            for step in range(1,50):
                lens.fill_(8001+step)
                waspending=pending.clone()
                k.begin[(1,)](start,pendingstart,pending,bank,refresh,rows,lens,writes,B,B,B,period)
                if waspending.any():switch.append(step)
                if refresh.item():
                    trigger.append(step);pending.fill_(1);pendingstart.copy_(lens)
                k.view[(B,)](hot,bank,start,pendingstart,rows,lens,writes,table,pos,slots,vl,N,P,248,False,refresh,4096)
                if step==1:
                    expected=torch.cat((torch.arange(64),torch.arange(64,4032),torch.arange(7938,8002))).cuda()
                    self.assertTrue(torch.equal(pos[0,:4096].long(),expected))
            self.assertEqual(trigger,list(range(period,50,period)))
            self.assertEqual(switch,[1]+list(range(period+1,50,period)))

    def test_indexed_recall(self):
        # After consumption, replace disjoint selections in a full fixed pool;
        # intersecting selections retain physical slots, including after eviction.
        B,N,P,H,D=1,20000,4096,2,16;W=H*D
        host=torch.randn(N,H,D,dtype=torch.bfloat16,pin_memory=True)
        ptr=make_pointer_table((host,host),device=torch.device('cuda'))
        gpu=torch.empty(P,H,D,device='cuda',dtype=host.dtype)
        keys=torch.full((P,),-1,device='cuda',dtype=torch.int32)
        directory=torch.full((N,),-1,device='cuda',dtype=torch.int32)
        ints=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
        table=torch.randperm(N,device='cuda',dtype=torch.int32)[None]
        rows=ints([0]);lens=ints([N]);write=ints([N-1]);flag=ints([1]);length=ints([P])
        pos=torch.arange(P,device='cuda',dtype=torch.int32)[None];out=torch.zeros_like(pos)
        free=torch.empty(1,P,device='cuda',dtype=torch.int32)
        current=torch.randn(1,H,D,device='cuda',dtype=host.dtype)
        def run(target,next_lease=False):
            k.plan_pool[(B,)](keys,directory,table,rows,lens,write,target,length,out,free,flag,N,N,P,next_lease,num_warps=8)
            k.recall[(B,2)](ptr,gpu,keys,table,rows,lens,write,target,length,out,current,flag,N,P,W,next_lease,0,tr.next_power_of_2(W),current.stride(0))
        def check_directory():
            # Eviction must invalidate the old mapping, not merely reject it on lookup.
            expected=torch.full_like(directory,-1)
            occupied=keys>=0
            expected[keys[occupied].long()]=torch.arange(P,device='cuda',dtype=torch.int32)[occupied]
            torch.testing.assert_close(directory,expected,atol=0,rtol=0)
        run(pos)
        original=out.clone()
        run(pos+1)
        check_directory()
        torch.testing.assert_close(out[0,:P-1],original[0,1:],atol=0,rtol=0)
        torch.testing.assert_close(gpu[out[0].long()],host[table[0,(pos+1)[0]].cpu().long()].cuda(),atol=0,rtol=0)
        for start in (P,P//2,2*P,0):
            target=torch.arange(start,start+P,device='cuda',dtype=torch.int32)[None]
            old=directory[table[0,target[0]].long()].clone()
            hit=(old>=0)&(keys[old.clamp_min(0).long()]==table[0,target[0]])
            run(target,True)
            torch.testing.assert_close(gpu[out[0].long()],host[table[0,target[0]].cpu().long()].cuda(),atol=0,rtol=0)
            torch.testing.assert_close(out[0,hit],old[hit],atol=0,rtol=0)
            self.assertEqual(out.unique().numel(),P)
            check_directory()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run(pos,True)
        for start in (P,2*P,P//2,0):
            pos.copy_(torch.arange(start,start+P,device='cuda',dtype=torch.int32)[None])
            graph.replay()
            check_directory()
            torch.testing.assert_close(gpu[out[0].long()],host[table[0,pos[0]].cpu().long()].cuda(),atol=0,rtol=0)


    @unittest.skipUnless(os.environ.get("SPARSEVLLM_LEGACY_ROOT"), "set SPARSEVLLM_LEGACY_ROOT for the old implementation oracle")
    def test_original_ema_selection_and_positions(self):
        root=Path(os.environ["SPARSEVLLM_LEGACY_ROOT"])/"src/sparsevllm"
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module
        original=load("lease_original_update",root/"triton_kernel/lease_ema_update.py")
        selection=load("lease_original_selection",root/"engine/cache_manager/_predictive_offload/selection.py")
        torch.manual_seed(79)
        B,H,N=2,28,8008
        state=torch.rand(B,H,N,device="cuda");seen=torch.randint(0,2,(B,N),device="cuda",dtype=torch.int32)
        scores=torch.rand(B,H,N,device="cuda");observed=torch.randint(0,2,(B,N),device="cuda",dtype=torch.int32)
        expected=[original.update_lease_ema(state[b],seen[b].bool(),scores[b,:,None],observed[b,None].bool(),.2) for b in range(B)]
        rows=torch.arange(B,device="cuda",dtype=torch.int32);flag=torch.ones(1,device="cuda",dtype=torch.int32)
        k.ema[(B,H,tr.cdiv(N,128))](state,seen,scores,observed,rows,flag,N,H,128,enable_fp_fusion=False)
        k.publish_seen[(B,tr.cdiv(N,128))](seen,observed,rows,flag,N,128)
        for b in range(B):
            torch.testing.assert_close(state[b],expected[b][0],atol=0,rtol=0)
            self.assertTrue(torch.equal(seen[b].bool(),expected[b][1]))
        reduced=torch.zeros(B,N,device="cuda");hot=torch.full((B,2,248),-1,device="cuda",dtype=torch.int32)
        bank=torch.zeros(B,device="cuda",dtype=torch.int32);pending=bank.clone();starts=bank.clone()
        lens=torch.tensor([128001,127973],device="cuda",dtype=torch.int32);counts=bank.clone()
        k.reduce_heads[(B,tr.cdiv(N,128))](state,reduced,rows,flag,N,H,128,32)
        k.select[(B,)](reduced,hot,bank,pending,starts,rows,lens,rows,flag,counts,N,248,tr.next_power_of_2(N),False,num_warps=8)
        manager=SimpleNamespace(_selection_outputs={},pooling_block_size=16,sink_token=64,local_token=64)
        for b,n in enumerate(lens.tolist()):
            end=tr.cdiv(n,16)-4
            selected=selection.select_with_workspace(manager,0,b,state[b,:,4:end],248).cpu().numpy()
            expected_positions=selection.expand_block_ids(manager,selected,n,4)
            actual=hot[b,1].cpu().numpy()
            self.assertTrue(np.array_equal(actual,np.sort(selected+4)))
            actual_positions=selection.expand_block_ids(manager,actual,n,0)
            self.assertTrue(np.array_equal(actual_positions,expected_positions))

    @unittest.skipUnless(os.environ.get("SPARSEVLLM_LEGACY_ROOT"), "set SPARSEVLLM_LEGACY_ROOT for lease timing oracle")
    def test_original_mixed_request_trigger(self):
        path=Path(os.environ["SPARSEVLLM_LEGACY_ROOT"])/"src/sparsevllm/engine/cache_manager/predictive_offload.py"
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="PredictiveOffloadCacheManager")
        functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {"_lease_age_reached","should_collect_decode_attn_score"}]
        namespace={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),"exec"),namespace)
        old=SimpleNamespace(_lease_start_positions=[{0:100,1:113}],_decode_rows=np.array([0,1]),
            _prefetch_futures=[None],_collect_decode_positions=[False],_reuse_steps=16,
            _is_layer_reuse_source=lambda layer:True,_lease_missing_decode_rows=lambda layer:False)
        old._lease_age_reached=MethodType(namespace["_lease_age_reached"],old)
        collect=MethodType(namespace["should_collect_decode_attn_score"],old)
        ints=lambda x:torch.tensor(x,device="cuda",dtype=torch.int32)
        starts=ints([100,113]);next_starts=starts.clone();pending=ints([0,0]);bank=ints([0,0])
        flag=ints([0]);rows=ints([0,1]);writes=rows.clone();lengths=ints([101,111])
        for step in range(1,50):
            old._decode_current_positions=np.array([99+step,109+step])
            expected=collect(0)
            lengths.copy_(ints((old._decode_current_positions+1).tolist()))
            prior_pending=bool(pending.any())
            k.begin[(1,)](starts,next_starts,pending,bank,flag,rows,lengths,writes,2,2,2)
            self.assertEqual(bool(flag),expected)
            if prior_pending:
                old._lease_start_positions[0]=dict(enumerate(next_starts.cpu().tolist()))
                old._prefetch_futures[0]=None
            if expected:
                pending.fill_(1);next_starts.copy_(lengths);old._prefetch_futures[0]=object()

    def test_recall_strided_current_and_graph_replay(self):
        B,N,P,H,D=2,4096,4096,4,128;W=H*D
        host=torch.randn(B*N,H,D,dtype=torch.bfloat16,pin_memory=True)
        ptr=make_pointer_table((host,host),device=torch.device("cuda"))
        gpu=torch.empty(B*P,H,D,device="cuda",dtype=host.dtype)
        keys=torch.full((B*P,),-1,device="cuda",dtype=torch.int32)
        directory=torch.full((B*B*N,),-1,device="cuda",dtype=torch.int32)
        free=torch.empty(B,P,device="cuda",dtype=torch.int32)
        table=torch.randperm(B*N,device="cuda",dtype=torch.int32).reshape(B,N)
        rows=torch.arange(B,device="cuda",dtype=torch.int32);lens=torch.tensor([N,N-7],device="cuda",dtype=torch.int32)
        writes=rows.clone();flag=torch.ones(1,device="cuda",dtype=torch.int32)
        pos=torch.arange(P,device="cuda",dtype=torch.int32).expand(B,-1).contiguous();out=torch.zeros_like(pos)
        current=torch.randn(B,3*H,D,device="cuda",dtype=host.dtype)[:,2*H:]
        def run():
            k.plan_pool[(B,)](keys,directory,table,rows,lens,writes,pos,lens,out,free,flag,B*N,N,P,False,num_warps=8)
            k.recall[(B,2)](ptr,gpu,keys,table,rows,lens,writes,pos,lens,out,current,flag,N,P,W,False,0,W,current.stride(0))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):run()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        keys.fill_(-1);directory.fill_(-1)
        for step in range(24):
            if step%3==0:keys.fill_(-1)
            rows.copy_(torch.tensor([step%2,1-step%2],device="cuda",dtype=torch.int32))
            current.normal_()
            graph.replay()
            for b,n in enumerate(lens.tolist()):
                expected=host[table[int(rows[b]),:n].cpu().long()].cuda();expected[-1]=current[b]
                torch.testing.assert_close(gpu[out[b,:n].long()],expected,atol=0,rtol=0)
                host[int(table[int(rows[b]),n-1])].copy_(current[b].cpu())

        saved_keys=keys.clone();saved_gpu=gpu.clone();saved_directory=directory.clone()
        writes.fill_(-1);graph.replay()
        torch.testing.assert_close(keys,saved_keys,atol=0,rtol=0)
        torch.testing.assert_close(gpu,saved_gpu,atol=0,rtol=0,equal_nan=True)
        torch.testing.assert_close(directory,saved_directory,atol=0,rtol=0)

    def test_attention_against_full_precision_oracle(self):
        from sparsevllm.kernels.triton.paged_flash_decoding import paged_flash_decode
        from sparsevllm.layers.attention_backend import TritonAttentionBackend
        torch.manual_seed(45)
        B,H,KH,D,N=2,28,4,128,4096
        q=torch.randn(B,H,D,device="cuda",dtype=torch.bfloat16)
        keys=torch.randn(B*N,KH,D,device="cuda",dtype=q.dtype);values=torch.randn_like(keys)
        slots=torch.randperm(B*N,device="cuda",dtype=torch.int32).reshape(B,N)
        rows=torch.arange(B,device="cuda",dtype=torch.int32);lengths=torch.tensor([N,N-31],device="cuda",dtype=torch.int32)
        mid=torch.empty(B,H,16,D,device="cuda");lse=torch.empty(B,H,16,device="cuda")
        scores=torch.empty(B,H,N,device="cuda")
        view=SimpleNamespace(payload=ExplicitKVPayload(keys,values),
            meta=SimpleNamespace(active_slots=slots,req_indices=rows,context_lens=lengths,attn_score=scores))
        eager=TritonAttentionBackend().run_decode(q,view,mid_o=mid,mid_o_logexpsum=lse,max_len_in_batch=N,block_seq=256,num_heads=H,num_kv_heads=KH)
        output=torch.empty_like(q)
        def run():
            paged_flash_decode(q,keys,values,slots,rows,lengths,mid,lse,attn_score=scores,target_tokens_per_split=256,output=output)
        run();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        graph.replay()
        for b,n in enumerate(lengths.tolist()):
            kb=keys[slots[b,:n].long()].repeat_interleave(H//KH,1).float()
            vb=values[slots[b,:n].long()].repeat_interleave(H//KH,1).float()
            logits=torch.einsum("hd,nhd->hn",q[b].float(),kb)
            expected=torch.einsum("hn,nhd->hd",(logits*D**-.5).softmax(-1),vb).to(q.dtype)
            torch.testing.assert_close(scores[b,:,:n],logits,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(eager[b],expected,atol=2e-3,rtol=2e-2)
            torch.testing.assert_close(output[b],expected,atol=2e-3,rtol=2e-2)

    @unittest.skipUnless(os.environ.get("LEASE_TEST_MODEL"), "set LEASE_TEST_MODEL for model lifecycle validation")
    @torch.inference_mode()
    def test_model_lifecycle_and_physical_kv(self):
        import json
        from sparsevllm import LLM,SamplingParams
        offload=bool(int(os.environ["LEASE_TEST_OFFLOAD"]));graph=bool(int(os.environ["LEASE_TEST_GRAPH"]))
        llm=LLM(os.environ["LEASE_TEST_MODEL"],sparse_method="leasesparse",enable_leasesparse_offload=offload,
            decode_graph=graph,leasesparse_trace=True,
            **json.loads(os.environ.get("LEASE_TEST_PARAMS", "{}")),max_model_len=8600,max_num_seqs_in_batch=2,max_decoding_seqs=2,
            engine_prefill_chunk_size=4096,max_num_batched_tokens=8192)
        # Disable replay only after provider preparation to isolate graph
        # execution from the different default eager attention implementation.
        replay=bool(int(os.environ.get("LEASE_TEST_REPLAY",str(int(graph)))))
        llm.model_runner.config.decode_graph=replay
        m=llm.model_runner.cache_manager;m.trace.clear();m.lease_counts.zero_()
        original=m.on_forward_end;checks=0
        def check(seqs,is_prefill):
            nonlocal checks
            original(seqs,is_prefill)
            if is_prefill:return
            torch.cuda.synchronize()
            for b,seq in enumerate(seqs):
                row=m.seq_id_to_row[seq.seq_id]
                for layer in (0,7,27):
                    g=m.groups[layer]
                    # After refresh, the fixed pool holds the pending selection.
                    prepared=offload and bool(m.refresh[g])
                    lengths=m.next_lengths if prepared else m.lengths
                    positions=m.next_positions if prepared else m.positions
                    slots=m.next_slots if prepared else m.slots
                    count=int(lengths[g,b]);positions=positions[g,b,:count].long()
                    physical=m.buffer_req_to_token_slots[row,positions].long()
                    self.assertTrue(torch.equal(slots[g,b,:count].long(),physical))
                    if offload:
                        active=m.directory[layer,row*m.config.num_kvcache_slots+physical].long()
                        self.assertTrue(bool(((active>=row*4096)&(active<(row+1)*4096)).all()))
                        self.assertTrue(torch.equal(m.cache_keys[layer,0,active].long(),physical))
                        host=m.attention_cache_storage.layer_payload(layer)
                        for gpu,cpu in zip(m.selected_staging[layer],(host.k_cache,host.v_cache)):
                            torch.testing.assert_close(gpu[active],cpu[physical.cpu()].cuda(),atol=0,rtol=0)
                    checks+=1
        m.on_forward_end=check
        outputs=[]
        for prompts,counts in (([[100]*8193,[101]*8225],[40,23]),([[102]*8177],[35])):
            for prompt,count in zip(prompts,counts):llm.add_request(prompt,SamplingParams(temperature=0,max_tokens=count,ignore_eos=True))
            finished={}
            while not llm.is_finished():
                result,_=llm.step()
                for request,tokens,_,_ in result:finished[request]=tokens
            outputs.append([finished[r] for r in sorted(finished)])
        for item in m.trace:
            if item["switch_step"] is not None:
                self.assertTrue(all(v==item["switch_step"] for v in item["layer_first_use"].values()))
        runner=llm.model_runner.decode_graph_runner
        result=dict(outputs=outputs,trace=m.trace,completion_events=m.trace_events,kv_checks=checks,
            replay_count=runner.replay_count if runner else 0,counts=m.lease_counts.cpu().tolist())
        if replay:self.assertGreater(result["replay_count"],70)
        with open(os.environ["LEASE_TEST_OUTPUT"],"w") as f:json.dump(result,f)
        llm.exit()




class PredictorKernels(unittest.TestCase):
    @torch.inference_mode()
    def test_incremental_keys_queries_and_partial_boundary(self):
        # Unaligned prefill chunks, decode append, and reordered/padded rows.
        H,KH,D,N,R=4,2,32,32,3
        sums=torch.zeros(R,KH,N,D,device='cuda')
        tail=torch.zeros(R,KH,80,D,device='cuda',dtype=torch.bfloat16)
        history=torch.zeros(R,H,8,D,device='cuda',dtype=torch.bfloat16)
        keys=torch.randn(337,KH,D,device='cuda',dtype=torch.bfloat16)
        queries=torch.randn(337,H,D,device='cuda',dtype=torch.bfloat16)
        ints=lambda v:torch.tensor(v,device='cuda',dtype=torch.int32)
        rows=ints([2]);lens=ints([0])
        last=torch.full((R,KH),-1,device='cuda',dtype=torch.int32)
        for start,end in ((0,117),(117,119),(119,279)):
            count=end-start;lens.fill_(end)
            k.predictor_keys_prefill[(tr.cdiv(start%16+count,16),KH)](keys[start:end],sums,tail,last,2,start,count,N,KH,D,*keys.stride()[:2])
            k.predictor_queries[(1,H)](queries[start:end],history,rows,lens,rows,H,D,*queries.stride()[:2],True,2,start,count)
        # Cross both 16-token block and 80-token ring boundaries. Repeated
        # writes model graph warmup at the first and interior tokens of a block.
        for pos in range(279,337):
            lens.fill_(pos+1)
            k.predictor_keys_decode[(1,KH)](keys[pos:pos+1],sums,tail,last,rows,lens,rows,N,KH,D,*keys.stride()[:2])
            k.predictor_keys_decode[(1,KH)](keys[pos:pos+1],sums,tail,last,rows,lens,rows,N,KH,D,*keys.stride()[:2])
            k.predictor_queries[(1,H)](queries[pos:pos+1],history,rows,lens,rows,H,D,*queries.stride()[:2])
        expected=torch.nn.functional.pad(keys.float(),(0,0,0,0,0,(-337)%16)).reshape(-1,16,KH,D).sum(1).permute(1,0,2)
        torch.testing.assert_close(sums[2,:,:expected.shape[1]],expected,rtol=0,atol=0)
        packed=torch.empty(1,H,8,D,device='cuda');partial=torch.empty(1,KH,D,device='cuda')
        k.predictor_pack[(1,H)](history,tail,rows,lens,packed,partial,H,KH,D)
        torch.testing.assert_close(packed[0],queries[-8:].float().permute(1,0,2))
        cutoff=337-64
        torch.testing.assert_close(partial[0],keys[(cutoff-1)//16*16:cutoff].float().mean(0))
        before=sums.clone();old_history=history.clone()
        k.predictor_keys_decode[(1,KH)](keys[:1],sums,tail,last,rows,lens,ints([-1]),N,KH,D,*keys.stride()[:2])
        k.predictor_queries[(1,H)](queries[:1],history,rows,lens,ints([-1]),H,D,*queries.stride()[:2])
        torch.testing.assert_close(sums,before);torch.testing.assert_close(history,old_history)
        # Reusing a request row must overwrite block zero, not accumulate old keys.
        k.predictor_keys_prefill[(1,KH)](keys[:7],sums,tail,last,2,0,7,N,KH,D,*keys.stride()[:2])
        torch.testing.assert_close(sums[2,:,0],keys[:7].float().sum(0))

    @torch.inference_mode()
    def test_fused_prediction_ranking_matches_full_key_reference_and_graph(self):
        torch.manual_seed(143)
        B,H,KH,D,N=2,28,4,128,32
        keys=torch.randn(B,512,KH,D,device='cuda',dtype=torch.bfloat16)
        sums=keys.float().reshape(B,N,16,KH,D).sum(2).permute(0,2,1,3).contiguous()
        ints=lambda v:torch.tensor(v,device='cuda',dtype=torch.int32)
        rows=ints([1,0]);lens=ints([279,127])
        partial=torch.empty(B,KH,D,device='cuda')
        def update_partial():
            for b,n in enumerate(lens.tolist()):
                cutoff=max(n-64,0)
                partial[b]=keys[int(rows[b]),(cutoff-1)//16*16:cutoff].float().mean(0)
        update_partial()
        # Inductor emits [B,H,D,F] storage viewed as [B,F,H,D]. The feature
        # stride is F, not 1; contiguous synthetic inputs missed this regression.
        q=torch.randn(B,H,D,4,device='cuda').permute(0,3,1,2);mass=torch.randn(B,4,H,device='cuda')
        scores=torch.empty(B,4,H,N,device='cuda');reduced=torch.empty(B,N,device='cuda')
        def run():
            k.predictor_logits[(B,KH,1)](q,sums,partial,rows,lens,scores,N,N,H,KH,D,*q.stride())
            k.predictor_probability[(B,4*H)](scores,mass,lens,N,H,N)
            k.predictor_reduce[(B,1)](scores,reduced,N,N,H)
        run()
        def reference():
            means=sums[rows.long()]/16
            for b,n in enumerate(lens.tolist()):
                means[b,:,(n-65)//16]=partial[b]
            ex=torch.einsum('bfhd,bhnd->bfhn',q,means.repeat_interleave(H//KH,1))/D**.5
            for b,n in enumerate(lens.tolist()):
                ex[b,:,:,(n-65)//16]+=torch.tensor(((n-65)%16+1)/16,device='cuda').log()
            valid=(torch.arange(N,device='cuda')[None]>=4)&(torch.arange(N,device='cuda')[None]*16<lens[:,None]-64)
            ex.masked_fill_(~valid[:,None,None],-torch.inf)
            expected=torch.zeros_like(scores)
            for b,n in enumerate(lens.tolist()):
                if n>128:expected[b]=ex[b].softmax(-1)*mass[b].sigmoid()[...,None]
            return expected
        torch.testing.assert_close(scores,reference(),rtol=2e-5,atol=2e-7)
        torch.testing.assert_close(reduced,reference().mean((1,2)),rtol=2e-5,atol=2e-7)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run();stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
        torch.cuda.current_stream().wait_stream(stream)
        # Replay crosses the empty-history boundary and every partial-block size,
        # with both rows active and their request mapping swapped.
        for offset in range(17):
            lens.copy_(ints([128+offset,272+offset]));rows.copy_(ints([offset%2,1-offset%2]))
            update_partial();q.add_(.02);mass.add_(.03);graph.replay()
            expected=reference()
            torch.testing.assert_close(scores,expected,rtol=2e-5,atol=2e-7)
            torch.testing.assert_close(reduced,expected.mean((1,2)),rtol=2e-5,atol=2e-7)


    @torch.inference_mode()
    def test_prediction_graphs_share_scratch_without_stale_rows_or_layers(self):
        from sparsevllm.engine.cache_manager.methods.leasesparse import LeaseSparseCacheManager
        from sparsevllm.engine.sparse_methods.leasesparse import MassQueryPredictor
        m=LeaseSparseCacheManager.__new__(LeaseSparseCacheManager)
        m.device=torch.device('cuda');m.blocks=64;m.num_kv_heads=4;m.head_dim=128
        m.hf_config=SimpleNamespace(num_attention_heads=28)
        m.predictor=MassQueryPredictor().cuda().eval()
        m.predictor.out.weight.normal_(std=.005);m.predictor.mass_head.weight.normal_(std=.01)
        eager_prediction=m.predictor.predict_with_mass
        m.predictor.predict_with_mass=torch.compile(eager_prediction,fullgraph=True,options={"triton.cudagraphs":False})
        m.query_history=torch.randn(2,2,28,8,128,device='cuda',dtype=torch.bfloat16)
        m.key_sums=torch.randn(2,2,4,64,128,device='cuda')
        m.key_tail=torch.randn(2,2,4,80,128,device='cuda',dtype=torch.bfloat16)
        m.prediction_meta=torch.tensor([[1,0],[700,515]],device='cuda',dtype=torch.int32)
        m.predictor_layer_ids=torch.arange(2,device='cuda')
        m.reduced=torch.empty(2,2,64,device='cuda')
        def reference(layer,batch):
            rows,lens=m.prediction_meta[:,:batch]
            positions=lens[:,None].long()+torch.arange(-8,0,device='cuda')
            history=m.query_history[layer,rows.long()]
            q=history.gather(2,(positions%8)[:,None,:,None].expand(-1,28,-1,128)).float()
            predicted,mass=eager_prediction(q,positions,m.predictor_layer_ids[layer:layer+1].expand(batch))
            means=m.key_sums[layer,rows.long()]/16
            for b,n in enumerate(lens.tolist()):
                cutoff=n-64;start=(cutoff-1)//16*16
                means[b,:,start//16]=m.key_tail[layer,int(rows[b]),:,torch.arange(start,cutoff,device='cuda')%80].float().mean(1)
            scores=torch.einsum('bfhd,bhnd->bfhn',predicted,means.repeat_interleave(7,1))/128**.5
            for b,n in enumerate(lens.tolist()):
                scores[b,:,:,(n-65)//16]+=torch.tensor(((n-65)%16+1)/16,device='cuda').log()
            valid=(torch.arange(64,device='cuda')>=4)&(torch.arange(64,device='cuda')[None]*16<lens[:,None]-64)
            scores.masked_fill_(~valid[:,None,None],-torch.inf)
            return (scores.softmax(-1)*mass.sigmoid()[...,None]).mean((1,2))
        m.score_prediction((0,),*m.prediction_meta,64);first=m.reduced[0].clone()
        m.score_prediction((1,),*m.prediction_meta,64);second=m.reduced[1].clone()
        m.score_prediction((0,1),*m.prediction_meta,64)
        torch.testing.assert_close(m.reduced,torch.stack((first,second)),rtol=1e-5,atol=2e-7)
        pool=torch.cuda.graph_pool_handle();stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs=[]
        with torch.cuda.stream(stream):
            for layer,batch in ((0,1),(1,2)):
                m.score_prediction((layer,),*m.prediction_meta[:,:batch],64);stream.synchronize()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream,pool=pool):m.score_prediction((layer,),*m.prediction_meta[:,:batch],64)
                graphs.append(g)
        torch.cuda.current_stream().wait_stream(stream)
        for index in (1,0,1):
            layer,batch=((0,1),(1,2))[index]
            m.query_history.add_(.125);m.prediction_meta[1].add_(1)
            graphs[index].replay();actual=m.reduced[layer,:batch].clone()
            torch.testing.assert_close(actual,reference(layer,batch),rtol=2e-4,atol=2e-7)
            m.score_prediction((layer,),*m.prediction_meta[:,:batch],64)
            torch.testing.assert_close(actual,m.reduced[layer,:batch],rtol=1e-5,atol=2e-7)


if __name__=='__main__':unittest.main()
