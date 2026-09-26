"""Independent history, prediction, physical KV and request-lifetime checks."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import triton as tr
from torch import nn
from sparsevllm.kernels.triton import attnpredict as k
from sparsevllm.engine.cache_manager.methods.attnpredict import AttnPredictCNN


def legacy_cnn():
    path=Path(os.environ['SPARSEVLLM_LEGACY_ROOT'])/'src/sparsevllm/engine/cache_manager/attnpredict_offload.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AttnPredictCNN')
    scope={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    model=scope['AttnPredictCNN']().cuda().half().eval()
    model.load_state_dict(torch.load(os.environ['AP_TEST_WEIGHTS'],map_location='cuda',weights_only=True))
    return model


def reference_positions(scores,n):
    end=max(tr.cdiv(n,16)-8,0);count=min(248,end)
    blocks=scores[:end].topk(count,sorted=False).indices+4
    positions=torch.cat((torch.arange(min(64,n),device='cuda'),
        (blocks[:,None]*16+torch.arange(16,device='cuda')).flatten(),
        torch.arange(max(0,n-64),n,device='cuda')))
    return positions.unique(sorted=True)


@pytest.mark.parametrize('graph',[False,True])
@torch.inference_mode()
def test_history_feedback_and_masked_cnn(graph):
    torch.manual_seed(131)
    B,H,N,P=2,4,270,4097;C=N-8
    rows=torch.tensor([1,0],device='cuda',dtype=torch.int32)
    lens=torch.tensor([4301,129],device='cuda',dtype=torch.int32);writes=rows.clone()
    hist=torch.rand(B,H,64,N,device='cuda',dtype=torch.float16)
    cursor=torch.tensor([63,7],device='cuda',dtype=torch.int32)
    pool=torch.zeros(B,H,N,device='cuda');positions=torch.zeros(B,P,device='cuda',dtype=torch.int32)
    positions[:,:128]=torch.arange(128,device='cuda');vlens=torch.full((B,),128,device='cuda',dtype=torch.int32)
    logits=torch.randn(B,H,P,device='cuda');x=torch.empty(B*H,64,C,device='cuda',dtype=torch.float16)
    valid=torch.empty(B*H,C,device='cuda',dtype=torch.bool)
    def update():
        pool.zero_()
        k.feedback[(B,H)](logits,positions,vlens,pool,rows,writes,N,H,P,8192,.125)
        k.append[(B,H,tr.cdiv(N,256))](pool,hist,cursor,rows,writes,N,H,256)
        k.advance[(1,)](cursor,rows,writes,B,2)
        k.pack[(B,H,64)](hist,cursor,rows,lens,x,valid,N,H,C,512)
    update();torch.cuda.synchronize()
    if graph:
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):update()
    expected=hist.clone();expected_cursor=cursor.clone()
    for step in range(67):
        writes[1]=-1 if step%3==0 else 0
        logits.normal_()
        for b,row in enumerate(rows.tolist()):
            if writes[b]<0:continue
            prob=(logits[b,:,:128]*.125).softmax(-1).bfloat16().float()
            pooled=prob.reshape(H,8,16).amax(-1)
            t=int(expected_cursor[row]);expected[row,:,t].zero_();expected[row,:,t,:8]=pooled
            expected_cursor[row]=(t+1)%64
        g.replay() if graph else update()
        torch.testing.assert_close(hist,expected,atol=0,rtol=0)
        assert torch.equal(cursor,expected_cursor)
    cnn=AttnPredictCNN().cuda().half().eval();original=legacy_cnn();cnn.load_state_dict(original.state_dict())
    out=cnn(x,valid).view(B,H,C)
    tiled=torch.cat([cnn(x[h:h+1],valid[h:h+1]) for h in range(B*H)]).view(B,H,C)
    torch.testing.assert_close(tiled,out,atol=2e-3,rtol=2e-3)
    for b,row in enumerate(rows.tolist()):
        width=max(0,tr.cdiv(int(lens[b]),16)-8)
        refhist=hist[row].roll(-int(cursor[row]),dims=1)[:,:,4:4+width]
        torch.testing.assert_close(x.view(B,H,64,C)[b,:,:,:width],refhist,atol=0,rtol=0)
        ref=original(refhist)
        torch.testing.assert_close(out[b,:,:width],ref,atol=2e-3,rtol=2e-3)
    shared=out.float().amax(1).masked_fill(~valid.view(B,H,C)[:,0],-float('inf'))
    top=shared.topk(248,sorted=False).indices;keep=torch.zeros(B,4096,device='cuda',dtype=torch.int32);counts=torch.zeros(B,device='cuda',dtype=torch.int32)
    k.select_positions[(B,)](top,rows,lens,rows,keep,counts,248,4096,4096,num_warps=8)
    for b,row in enumerate(rows.tolist()):
        assert torch.equal(keep[row,:counts[row]].long(),reference_positions(shared[b],int(lens[b])))


@pytest.mark.skipif(not os.environ.get('AP_TEST_MODEL'),reason='requires Qwen model')
@torch.inference_mode()
def test_model_lifecycle():
    from sparsevllm import LLM,SamplingParams
    graph=bool(int(os.environ['AP_TEST_GRAPH']))
    llm=LLM(os.environ['AP_TEST_MODEL'],sparse_method='attnpredict',attnpredict_model_path=os.environ['AP_TEST_WEIGHTS'],
        decode_graph=graph,max_model_len=8400,max_num_seqs_in_batch=2,max_decoding_seqs=2,
        engine_prefill_chunk_size=4096,max_num_batched_tokens=8192)
    static=bool(os.environ.get('AP_TEST_EAGER_STATIC'))
    if static:
        runner=llm.model_runner.decode_graph_runner
        runner.run=lambda seqs,**kwargs: (runner.run_eager_static(seqs),None)
    asynchronous=bool(os.environ.get("AP_TEST_ASYNC_ONLY"))
    saved_logits=[]; latest_logits=[]
    if os.environ.get('AP_TEST_FORCE_TOKEN'):
        llm.model_runner._sample_model_outputs=lambda logits,seqs,**kwargs: torch.full((logits.shape[0],),100,device=logits.device,dtype=torch.int64)
        llm.model_runner._record_debug_logits=lambda logits: latest_logits.__setitem__(slice(None),[logits.float().clone() if asynchronous else logits.float().cpu()])
    m=llm.model_runner.cache_manager;original=m.on_forward_end;oracle=legacy_cnn();checks=0;prior={};outputs=[]
    assert m.attention_cache_storage.full_layers==frozenset({0,1}) or set(m.attention_cache_storage.full_layers)=={0,1}
    assert len(set(m.lru.layer_groups.values()))==26
    assert len({stream.cuda_stream for stream in m.prediction_streams})==26
    def check(seqs,is_prefill):
        nonlocal checks
        original(seqs,is_prefill)
        if latest_logits:saved_logits.append(latest_logits[0])
        if asynchronous:return
        torch.cuda.synchronize()
        for b,seq in enumerate(seqs):
            row=m.seq_id_to_row[seq.seq_id];n=seq.num_prefilled_tokens+seq.current_chunk_size if is_prefill else seq.decode_input_position+1
            if is_prefill and n<seq.num_tokens:continue
            if not is_prefill:
                count=int(m.view_lens[25,b]);pos=m.positions[25,b,:count].long()
                assert torch.equal(pos,torch.cat((prior[seq.seq_id],torch.tensor([n-1],device='cuda'))))
                physical=m.buffer_req_to_token_slots[row,pos].long()
                host=m.attention_cache_storage.layer_payload(27)
                slots=m.selected_slots[b,:count].long()
                for gpu,cpu in zip(m.selected_staging[27],(host.k_cache,host.v_cache)):
                    actual_kv=gpu[slots];expected_kv=cpu[physical.cpu()].cuda()
                    torch.testing.assert_close(actual_kv,expected_kv,atol=0,rtol=0)
                prob=(m.scores[25,b,:,:count]*m.head_dim**-.5).softmax(-1).bfloat16().float()
                dense=torch.zeros(28,m.blocks*16,device='cuda');dense[:,pos]=prob
                pooled=dense.view(28,m.blocks,16).amax(-1).half()
                t=(int(m.cursor[25,row])-1)%64
                torch.testing.assert_close(m.history[25,row,:,t],pooled,atol=6e-8,rtol=0.008)
            end=max(tr.cdiv(n,16)-4,4)
            history=m.history[25,row].roll(-int(m.cursor[25,row]),dims=1)[:,:,4:end]
            predicted=oracle(history).float().amax(0) if history.shape[-1]>=3 else torch.ones(history.shape[-1],device='cuda')
            expected=reference_positions(predicted,n)
            actual=m.keep[25,row,:m.keep_lens[25,row]].long()
            # FP16 convolution shape changes may perturb tied cutoffs; compare score mass separately.
            assert len(actual)==len(expected) and actual.unique().numel()==len(actual)
            selected=((actual[(actual>=64)&(actual<n-64)]//16).unique()-4)
            selected=selected[(selected>=0)&(selected<predicted.numel())]
            if predicted.numel()>248:
                assert predicted[selected].min()>=predicted.topk(248).values[-1]-2e-3
            # Verify the next selection is already resident before its next consumer.
            physical=m.buffer_req_to_token_slots[row,actual].long()
            directory=m.lru.metadata[m.lru.layer_groups[27]][0]
            resident=directory[row,physical].long()
            assert (resident>=0).all()
            host=m.attention_cache_storage.layer_payload(27)
            offset=row*m.lru.metadata[m.lru.layer_groups[27]][1].shape[1]
            for gpu,cpu in zip(m.selected_staging[27],(host.k_cache,host.v_cache)):
                torch.testing.assert_close(gpu[offset+resident],cpu[physical.cpu()].cuda(),atol=0,rtol=0)
            prior[seq.seq_id]=actual.clone();checks+=1
    m.on_forward_end=check
    for prompts,counts in (([[100]*8193,[101]*8225],[19,9]),([[102]*8177],[5])):
        for prompt,count in zip(prompts,counts):llm.add_request(prompt,SamplingParams(temperature=0,max_tokens=count,ignore_eos=True))
        finished={}
        while not llm.is_finished():
            result,_=llm.step()
            for request,tokens,_,_ in result:finished[request]=tokens
        outputs.append([finished[r] for r in sorted(finished)])
    assert not m.history.any() and not m.keep_lens.any()
    runner=llm.model_runner.decode_graph_runner
    replay=runner.replay_count if runner else 0
    if graph and not static:assert replay>=22
    Path(os.environ['AP_TEST_OUTPUT']).write_text(json.dumps(dict(outputs=outputs,checks=checks,replay_count=replay,async_state=m.memory_accounting()["attnpredict_async"])))
    if saved_logits:torch.save(saved_logits,os.environ['AP_TEST_OUTPUT']+'.pt')
    llm.exit()


@torch.inference_mode()
def test_prefill_tail_history_with_short_last_chunk():
    from sparsevllm.kernels.triton import leasesparse as tail
    torch.manual_seed(731)
    H,KH,D,N=4,2,128,20
    lens=torch.tensor([257,319],device='cuda',dtype=torch.int32)
    chunks=torch.tensor([1,33],device='cuda',dtype=torch.int32)
    starts=torch.tensor([0,1],device='cuda',dtype=torch.int32)
    rows=torch.tensor([1,0],device='cuda',dtype=torch.int32)
    q=torch.randn(34,H,D,device='cuda',dtype=torch.bfloat16)
    keys=torch.randn(640,KH,D,device='cuda',dtype=torch.bfloat16)
    table=torch.randperm(640,device='cuda').int().view(2,320)
    maximum=torch.empty(2,H,64,N,device='cuda');lses=torch.empty_like(maximum);norm=torch.empty(2*H*64,device='cuda')
    history=torch.zeros(2,H,64,N,device='cuda',dtype=torch.float16);cursor=torch.ones(2,device='cuda',dtype=torch.int32)
    tail.tail_logits[(2,H,tr.cdiv(N,8))](q,keys,table,rows,lens,starts,chunks,maximum,lses,N,320,H,KH,D,*q.stride()[:2],*keys.stride()[:2],D**-.5)
    tail.tail_norm[(2*H*64,)](lses,norm,N,32)
    k.seed[(2,H,64)](history,maximum,norm,rows,lens,chunks,cursor,N,H,32)
    for b,row in enumerate(rows.tolist()):
        n,t=int(lens[b]),int(chunks[b]);query=q[int(starts[b]):int(starts[b])+t].float()
        key=keys[table[row,:n].long()].float().repeat_interleave(H//KH,1)
        logits=torch.einsum('thd,lhd->htl',query,key)*D**-.5
        logits.masked_fill_(torch.arange(n,device='cuda')[None,None,:]>torch.arange(n-t,n,device='cuda')[None,:,None],-float('inf'))
        p=logits.softmax(-1).bfloat16().float()
        expected=torch.nn.functional.pad(p,(0,N*16-n)).view(H,t,N,16).amax(-1).half()
        torch.testing.assert_close(history[row,:,-t:],expected,atol=6e-8,rtol=.008)
        assert not history[row,:,:64-t].any() and cursor[row]==0


@pytest.mark.parametrize('change',[
    {'world_size':2}, {'enable_prefix_caching':True}, {'prefill_sparse_method':'snapkv'},
    {'full_attention_layers':[0,2]}, {'attnpredict_model_path':'/nonexistent/predictor.pth'}])
def test_reject_unsupported_contract(change):
    from sparsevllm.configs.groups import SparseMethodConfig
    from sparsevllm.configs.sparse import finalize_sparse_layout
    config=SparseMethodConfig(sparse_method='attnpredict',decode_keep_tokens=3968,recent_keep_tokens=64)
    config.hf_config=SimpleNamespace(model_type='qwen2',num_hidden_layers=28,dtype=torch.bfloat16)
    config.runtime_layout=SimpleNamespace(kv_idx_to_layer_idx=tuple(range(28)))
    config.world_size=1;config.enable_prefix_caching=False;config.prefill_sparse_method=''
    config.full_attention_layers=[];config.attnpredict_model_path=os.environ['AP_TEST_WEIGHTS']
    for name,value in change.items():setattr(config,name,value)
    with pytest.raises((ValueError,FileNotFoundError)):finalize_sparse_layout(config)


@torch.inference_mode()
def test_attention_eager_and_graph_against_fp32():
    from sparsevllm.engine.cache_manager import ExplicitKVPayload
    from sparsevllm.kernels.triton.paged_flash_decoding import paged_flash_decode
    from sparsevllm.layers.attention_backend import TritonAttentionBackend
    torch.manual_seed(121)
    B,H,KH,D,N=2,28,4,128,4097
    q=torch.randn(B,H,D,device='cuda',dtype=torch.bfloat16)
    key=torch.randn(B*N,KH,D,device='cuda',dtype=q.dtype);value=torch.randn_like(key)
    slots=torch.randperm(B*N,device='cuda',dtype=torch.int32).reshape(B,N)
    rows=torch.arange(B,device='cuda',dtype=torch.int32);lengths=torch.tensor([4097,129],device='cuda',dtype=torch.int32)
    mid=torch.empty(B,H,32,D,device='cuda');lse=torch.empty(B,H,32,device='cuda');scores=torch.empty(B,H,N,device='cuda')
    view=SimpleNamespace(payload=ExplicitKVPayload(key,value),meta=SimpleNamespace(active_slots=slots,req_indices=rows,context_lens=lengths,attn_score=scores))
    eager=TritonAttentionBackend().run_decode(q,view,mid_o=mid,mid_o_logexpsum=lse,max_len_in_batch=N,block_seq=256,num_heads=H,num_kv_heads=KH)
    output=torch.empty_like(q)
    def run():paged_flash_decode(q,key,value,slots,rows,lengths,mid,lse,attn_score=scores,target_tokens_per_split=256,output=output)
    run();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    graph.replay()
    for b,n in enumerate(lengths.tolist()):
        keys=key[slots[b,:n].long()].repeat_interleave(H//KH,1).float()
        values=value[slots[b,:n].long()].repeat_interleave(H//KH,1).float()
        logits=torch.einsum('hd,nhd->hn',q[b].float(),keys)
        expected=torch.einsum('hn,nhd->hd',(logits*D**-.5).softmax(-1),values).to(q.dtype)
        torch.testing.assert_close(scores[b,:,:n],logits,atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(eager[b],expected,atol=2e-3,rtol=2e-2)
        torch.testing.assert_close(output[b],expected,atol=2e-3,rtol=2e-2)


@pytest.mark.parametrize('append_current',[False,True])
@torch.inference_mode()
def test_current_and_prefetch_views(append_current):
    rows=torch.tensor([1,0],device='cuda',dtype=torch.int32)
    lens=torch.tensor([129,65],device='cuda',dtype=torch.int32)
    writes=torch.tensor([128,-1],device='cuda',dtype=torch.int32)
    keep=torch.zeros((2,4096),device='cuda',dtype=torch.int32)
    keep[1,:3]=torch.tensor([0,64,127],device='cuda')
    counts=torch.tensor([1,3],device='cuda',dtype=torch.int32)
    table=torch.randperm(512,device='cuda').int().view(2,256)
    positions=torch.empty((2,4097),device='cuda',dtype=torch.int32)
    slots=torch.empty_like(positions);lengths=torch.empty(2,device='cuda',dtype=torch.int32)
    def run():k.view[(2,)](keep,counts,rows,lens,writes,table,positions,slots,lengths,256,4097,8192,APPEND_CURRENT=append_current)
    run();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for n in (129,130,145):
        lens[0]=n;graph.replay()
        expected=torch.tensor([0,64,127]+([n-1] if append_current else []),device='cuda')
        assert lengths.tolist()==[len(expected),0]
        assert torch.equal(positions[0,:len(expected)].long(),expected)
        assert torch.equal(slots[0,:len(expected)],table[1,expected])


@pytest.mark.parametrize('columns',[263,2000,8000])
@torch.inference_mode()
def test_compiled_cnn_head_groups_preserve_masked_scores(columns):
    # Catch fusion/grouping errors at a partial head group and padded context edge.
    torch.manual_seed(932)
    cnn=AttnPredictCNN().cuda().half().to(memory_format=torch.channels_last).eval()
    oracle=legacy_cnn();cnn.load_state_dict(oracle.state_dict())
    cnn.forward=torch.compile(cnn.forward,fullgraph=True,options={"triton.cudagraphs":False})
    x=torch.rand(7,64,columns,device='cuda',dtype=torch.float16)*.02
    valid=torch.ones(7,columns,device='cuda',dtype=torch.bool)
    width=columns//2
    valid[-1,width:]=False;x[-1,:,width:]=0
    expected=torch.cat((oracle(x[:-1]),torch.nn.functional.pad(oracle(x[-1:,:,:width]),(0,columns-width))))
    out=torch.empty_like(expected)
    chunk=max(1,32*1024*1024//(32*64*columns*2))
    def run():
        for h in range(0,len(x),chunk):out[h:h+chunk].copy_(cnn(x[h:h+chunk],valid[h:h+chunk]))
    run();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    graph.replay()
    torch.testing.assert_close(out[:-1],expected[:-1],atol=2e-3,rtol=2e-3)
    torch.testing.assert_close(out[-1,:width],expected[-1,:width],atol=2e-3,rtol=2e-3)
