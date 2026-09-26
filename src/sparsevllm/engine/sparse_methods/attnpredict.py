"""Per-layer prediction after attention; consume it on the following step."""
import torch
from sparsevllm.engine.cache_manager import SparseSelection
from sparsevllm.kernels.triton import attnpredict as k
from .base import SparseMethodRuntime


class AttnPredictRuntime(SparseMethodRuntime):
    def _begin_prepare_step(self,step):
        self.cache_manager.begin_selection_step()
        self.cache_manager.predictions_submitted.clear()

    def reset_decode_attn_scores_for_graph(self,refs):
        # Raw per-head logits overwrite valid entries after the layer's consumer wait.
        return True

    def needs_attention_score(self,layer_idx,step):
        return not step.is_prefill and layer_idx>=2

    def _prepare_decode_attention_score(self,layer_idx,state,batch_size,num_heads,max_len):
        state.attn_score=self.cache_manager.scores[layer_idx-2,:batch_size]

    def build_prefill_selection(self,request):
        return self._full_selection(request.layer_idx)

    def build_decode_selection(self,request):
        layer=request.layer_idx
        if layer<2:return self._full_selection(layer)
        m=self.cache_manager;s=m.layer_batch_state;B=s.req_indices.numel()
        torch.cuda.current_stream(self.device).wait_event(m.prediction_ready[layer-2])
        k.view[(B,)](m.keep[layer-2],m.keep_lens[layer-2],s.req_indices,s.context_lens,s.slot_mapping,
            m.buffer_req_to_token_slots,m.positions[layer-2],m.slots[layer-2],m.view_lens[layer-2],m.max_model_len,4097,8192)
        return SparseSelection(kind='slots',req_indices=m.selected_rows[:B],context_lens=m.view_lens[layer-2,:B],
            active_slots=m.slots[layer-2,:B],max_context_len=4097,attn_score=m.scores[layer-2,:B],global_req_indices=s.req_indices)

    def on_attention_end(self,event):
        if event.forward_context.is_prefill or event.layer_idx<2:return
        m=self.cache_manager;s=m.layer_batch_state
        m.snapshot_prediction(event.layer_idx,s.req_indices,s.context_lens,s.slot_mapping)
        if not torch.cuda.is_current_stream_capturing():
            m.submit_prediction(event.layer_idx,s.req_indices.numel())
