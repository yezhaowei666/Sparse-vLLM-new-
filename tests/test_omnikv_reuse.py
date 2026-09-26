"""Selection leases preserve per-request history and move the current window."""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from sparsevllm.engine.cache_manager.methods.omnikv.manager import OmniKVCacheManager
from sparsevllm.kernels.triton.omnikv_fused import build_omnikv_keep_and_slots
from sparsevllm.operators.omnikv_selection import OmniKVSelectionSpec, prepare_omnikv_selection


@pytest.mark.parametrize('value', [0,-1,True,1.5])
def test_invalid_reuse_period(value):
    from sparsevllm.configs.groups import SparseMethodConfig
    from sparsevllm.configs.sparse import normalize_sparse_method_name
    with pytest.raises(ValueError,match='positive integer'):
        normalize_sparse_method_name(SparseMethodConfig(sparse_method='omnikv',omnikv_reuse_steps=value))


def test_reuse_requires_offload():
    from sparsevllm.configs.groups import SparseMethodConfig
    from sparsevllm.configs.sparse import normalize_sparse_method_name
    with pytest.raises(ValueError,match='requires offload'):
        normalize_sparse_method_name(SparseMethodConfig(sparse_method='omnikv',omnikv_reuse_steps=4))


@pytest.mark.parametrize('period', [4,16])
@pytest.mark.parametrize('graph', [False,True])
def test_reuse_selection_rows_padding_short_history_and_replay(period, graph):
    m = object.__new__(OmniKVCacheManager)
    m.config = SimpleNamespace(obs_layer_ids=[0],decode_keep_tokens=7,sink_keep_tokens=2,
                              recent_keep_tokens=3,omnikv_reuse_steps=period)
    m.device=torch.device('cuda');m.max_buffer_rows=3
    m.initialize_selection_reuse()
    ints=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
    rows=ints([0,1,0]);lengths=ints([20,7,1]);writes=ints([0,1,-1])
    m.layer_batch_state=SimpleNamespace(slot_mapping=writes)
    scores=torch.randn(3,128,device='cuda')
    provider=prepare_omnikv_selection(OmniKVSelectionSpec(2,7),device=m.device)
    table=torch.arange(3*128,device='cuda',dtype=torch.int32).reshape(3,128)
    outputs=[]
    def run():
        m.prepare_selection_reuse(0,rows,lengths)
        hist=(lengths-3).clamp_min(2);candidates=(hist-2).clamp_min(0)
        chosen=provider.select(scores,torch.where(m.reuse_refresh,candidates,0),7)
        chosen=m.commit_selection_reuse(0,rows,lengths,chosen)
        outputs[:]=build_omnikv_keep_and_slots(chosen,candidates.clamp_max(7),hist,lengths-hist,
                                             table,rows,2,max_s=12,context_lens=lengths)
    run()
    if graph:
        captured=torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):run()
    m.reuse_starts.fill_(-1);m.reuse_counts.zero_()
    expected={};starts={}
    for step in range(1,39):
        order=[0,1,0] if step%2 else [1,0,0]
        rows.copy_(ints(order));lengths.copy_(ints([20+step if r==0 else 7+step for r in order]))
        scores.normal_()
        if step==22:  # Freed row is reused by a new request.
            m.reuse_starts[:,1]=-1;expected.pop(1,None);starts.pop(1,None)
        old=m.reuse_indices.clone()
        if graph:captured.replay()
        else:run()
        positions,slots,visible=outputs
        for b,row in enumerate(order[:2]):
            n=int(lengths[b]);refresh=row not in starts or n-starts[row]>=period or starts[row]<12
            if refresh:
                h=max(n-3,2);count=min(7,h-2)
                expected[row]=set((scores[b,2:h].topk(count).indices+2).tolist())
                starts[row]=n
            assert int(m.reuse_starts[0,row])==starts[row]
            assert set(m.reuse_indices[0,row][m.reuse_indices[0,row]>=0].tolist())==expected[row]
            p=positions[b,:int(visible[b])].tolist()
            assert set(p)==set(range(2))|expected[row]|set(range(n-3,n))
            assert len(p)==len(set(p))
            torch.testing.assert_close(slots[b,:len(p)],table[row,positions[b,:len(p)].long()])
        assert torch.equal(m.reuse_indices[0,2],old[0,2])


@pytest.mark.skipif(not os.environ.get('OMNI_TEST_MODEL'), reason='requires local model')
def test_model_reuse_lifecycle():
    from sparsevllm import LLM,SamplingParams
    period=int(os.environ['OMNI_TEST_PERIOD'])
    graph=bool(int(os.environ['OMNI_TEST_GRAPH']))
    llm=LLM(os.environ['OMNI_TEST_MODEL'],sparse_method='omnikv',enable_omnikv_offload=True,
            omnikv_reuse_steps=period,decode_graph=graph,max_model_len=8600,
            max_num_seqs_in_batch=2,max_decoding_seqs=2,engine_prefill_chunk_size=4096,
            max_num_batched_tokens=8192)
    m=llm.model_runner.cache_manager
    original=m.on_forward_end;trace=[];previous={}
    def inspect(seqs,prefill):
        original(seqs,prefill)
        if prefill or period==1:return
        torch.cuda.synchronize()
        for seq in seqs:
            row=m.seq_id_to_row[seq.seq_id];n=seq.decode_input_position+1
            for layer,g in m.reuse_groups.items():
                start=int(m.reuse_starts[g,row]);chosen=m.reuse_indices[g,row].cpu().tolist()
                key=(seq.seq_id,layer);old=previous.get(key)
                fresh=old is None or n-old[0]>=period
                assert start==(n if fresh else old[0])
                if not fresh:assert chosen==old[1]
                assert len(set(chosen))==m.config.decode_keep_tokens
                assert min(chosen)>=m.config.sink_keep_tokens
                assert max(chosen)<start-m.config.recent_keep_tokens
                previous[key]=(start,chosen)
                trace.append(dict(request=seq.seq_id,layer=layer,step=n-seq.num_prompt_tokens,
                                  refreshed=fresh,start=start))
    m.on_forward_end=inspect
    outputs=[]
    for prompts,counts in (([[100]*8193,[101]*8225],[40,23]),([[102]*8177],[35])):
        for prompt,count in zip(prompts,counts):llm.add_request(prompt,SamplingParams(temperature=0,max_tokens=count,ignore_eos=True))
        finished={}
        while not llm.is_finished():
            result,_=llm.step()
            for request,tokens,_,_ in result:finished[request]=tokens
        outputs.append([finished[k] for k in sorted(finished)])
    runner=llm.model_runner.decode_graph_runner
    r=dict(outputs=outputs,trace=trace,replay_count=runner.replay_count if runner else 0,
           force_eager=runner.force_eager_count if runner else 0)
    if graph:assert r['replay_count']>70 and r['force_eager']==0
    with open(os.environ['OMNI_TEST_OUTPUT'],'w') as f:json.dump(r,f)
    llm.exit()
