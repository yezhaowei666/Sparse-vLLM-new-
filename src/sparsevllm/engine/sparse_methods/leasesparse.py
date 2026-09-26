"""LeaseSparse logical selection; all persistent state belongs to the cache."""
import torch
from torch import nn
import torch.nn.functional as F
import triton as tr
from sparsevllm.engine.cache_manager import SparseSelection
from sparsevllm.kernels.triton import leasesparse as kernels
from .base import SparseMethodRuntime


class FutureTCN(nn.Module):
    def __init__(self, heads=28, dim=128, hidden=64, horizon=4, history=8):
        super().__init__()
        self.heads, self.dim, self.horizon = heads, dim, horizon
        if history not in (8, 32, 64):
            raise ValueError('Supported history lengths: 8, 32, 64')
        self.history = history
        self.in_proj = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        # Left padding only; receptive field = 1 + 3 + 2*2 = 8 queries.
        self.conv1 = nn.Conv1d(hidden, hidden, 4)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=2)
        if history >= 32:
            # Receptive field: 8 + 3*8 = 32, with no missing positions.
            self.conv3 = nn.Conv1d(hidden, hidden, 4, dilation=8)
        if history == 64:
            self.conv4 = nn.Conv1d(hidden, hidden, 3, dilation=16)
        self.out = nn.Linear(hidden, horizon * dim)

    def encode(self, q):
        b, h, t, _ = q.shape
        x = self.norm(self.in_proj(q)).reshape(b * h, t, -1).transpose(1, 2)
        x = F.gelu(self.conv1(F.pad(x, (3, 0))))
        x = F.gelu(self.conv2(F.pad(x, (4, 0))))
        if self.history >= 32:
            x = F.gelu(self.conv3(F.pad(x, (24, 0))))
        if self.history == 64:
            x = F.gelu(self.conv4(F.pad(x, (32, 0))))
        return x.transpose(1, 2).reshape(b, h, t, -1)

    def forward(self, q):
        x = self.encode(q)[:, :, -1]
        b, h, _ = x.shape
        return self.out(x).reshape(b, h, self.horizon, self.dim).permute(0, 2, 1, 3)


class ResidualQueryPredictor(FutureTCN):
    """Predict content changes, then apply the known future Qwen2 RoPE positions."""
    def __init__(self, heads=28, dim=128, hidden=64, horizon=4, history=8, rope_theta=10000000.):
        if history != 8 or dim % 2:
            raise ValueError('Residual predictor requires history=8 and even head dimension')
        super().__init__(heads, dim, hidden, horizon, history)
        self.temporal = nn.Linear(history, horizon, bias=False)
        self.register_buffer('inv_freq', 1. / (rope_theta ** (torch.arange(0, dim, 2).float() / dim)))
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def rotate(self, q, positions, inverse=False):
        # q: [batch, heads, time, dim]; Qwen2 uses split-half rotation.
        phase = positions.float()[..., None] * self.inv_freq
        phase = torch.cat((phase, phase), -1)[:, None]
        rotated = torch.cat((-q[..., self.dim // 2:], q[..., :self.dim // 2]), -1)
        return q * phase.cos() + rotated * phase.sin() * (-1 if inverse else 1)

    def encode(self, changes):
        b, h, t, _ = changes.shape
        x = self.norm(self.in_proj(changes)).reshape(b * h, t, -1).transpose(1, 2)
        x = x + F.gelu(self.conv1(F.pad(x, (3, 0))))
        x = x + F.gelu(self.conv2(F.pad(x, (4, 0))))
        return x.transpose(1, 2).reshape(b, h, t, -1)

    def forward(self, q, positions):
        content = self.rotate(q, positions, inverse=True)
        latest = content[:, :, -1:]
        changes = content - latest
        linear = self.temporal(changes.transpose(-1, -2)).transpose(-1, -2)
        encoded = self.encode(changes)[:, :, -1]
        correction = self.out(encoded).reshape(q.shape[0], q.shape[1], self.horizon, self.dim)
        future = positions[:, -1:] + torch.arange(1, self.horizon + 1, device=q.device)
        return self.rotate(latest + linear + correction, future).permute(0, 2, 1, 3)


class ConditionedQueryPredictor(ResidualQueryPredictor):
    """Horizon-specific history retrieval conditioned on layer/head identity."""
    def __init__(self, heads=28, dim=128, hidden=64, horizon=4, history=8, rope_theta=10000000.):
        super().__init__(heads, dim, hidden, horizon, history, rope_theta)
        self.layer_embedding = nn.Embedding(28, 16)
        self.head_embedding = nn.Embedding(heads, 16)
        self.condition = nn.Linear(16, 2 * hidden)
        self.horizon_embedding = nn.Parameter(torch.randn(horizon, hidden) * .02)
        self.query_proj = nn.Linear(hidden, hidden)
        self.key_proj = nn.Linear(hidden, hidden)
        self.value_proj = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q, positions, layer, return_features=False):
        if layer is None or isinstance(layer, int) and not 0 <= layer < self.layer_embedding.num_embeddings:
            raise ValueError('Conditioned predictor requires the actual layer index')
        content = self.rotate(q, positions, inverse=True)
        latest = content[:, :, -1:]
        changes = content - latest
        encoded = self.encode(changes)
        layer_features = self.layer_embedding.weight[layer:layer+1] if isinstance(layer, int) else self.layer_embedding(layer)
        identity = layer_features[:, None] + self.head_embedding.weight[None]
        scale, shift = self.condition(identity).chunk(2, -1)
        memory = encoded * (1 + scale[:, :, None]) + shift[:, :, None]
        # Each future step reads all eight history states; current content is
        # available explicitly, rather than only through centered differences.
        context = self.norm(self.in_proj(latest))
        queries = memory[:, :, -1:] + context + self.horizon_embedding[None, None]
        weights = (self.query_proj(queries) @ self.key_proj(memory).transpose(-1, -2)) / encoded.shape[-1] ** .5
        decoded = queries + weights.softmax(-1) @ self.value_proj(memory)
        correction = self.out(decoded)
        linear = self.temporal(changes.transpose(-1, -2)).transpose(-1, -2)
        future = positions[:, -1:] + torch.arange(1, self.horizon + 1, device=q.device)
        prediction = self.rotate(latest + linear + correction, future).permute(0, 2, 1, 3)
        return (prediction, decoded.permute(0, 2, 1, 3)) if return_features else prediction


class MassQueryPredictor(ConditionedQueryPredictor):
    """Predict eligible historical mass alongside the conditional block probes."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mass_head = nn.Linear(self.out.in_features, 1)
        # Equal initial mass preserves the previous zero-training block ranking.
        nn.init.zeros_(self.mass_head.weight)
        nn.init.zeros_(self.mass_head.bias)

    def predict_with_mass(self, q, positions, layer):
        query, features = super().forward(q, positions, layer, return_features=True)
        return query, self.mass_head(features).squeeze(-1)


def select_lease(m,g,rows,lengths,writes,*,initial=False):
    B=rows.numel()
    N=m.blocks
    H=m.hf_config.num_attention_heads
    kernels.reduce_heads[(B,tr.cdiv(N,128))](m.ema[g],m.reduced,rows,m.refresh[g:g+1],N,H,128,tr.next_power_of_2(H))
    kernels.select[(B,)](m.reduced,m.hot[g],m.bank[g],m.pending[g],m.pending_starts[g],
                        rows,lengths,writes,m.refresh[g:g+1],m.lease_counts[g],N,248,tr.next_power_of_2(N),initial,num_warps=8)

def seed_prefill_lease(m,layer_idx,q,view,*,b_start_loc,chunk_lens,attention_lse=None):
    if layer_idx not in m.sources:
        return
    g=m.groups[layer_idx]
    H=q.shape[1]
    N=m.blocks
    for b,seq in enumerate(m._prefill_seqs):
        if seq.num_prefilled_tokens+seq.current_chunk_size<seq.num_tokens:
            continue
        rows=view.meta.req_indices[b:b+1]
        lens=view.meta.context_lens[b:b+1]
        chunks=chunk_lens[b:b+1]
        starts=b_start_loc[b:b+1]
        maxima=torch.empty((1,H,64,N),dtype=torch.float32,device=m.device)
        lses=torch.empty_like(maxima)
        norm=torch.empty((H*64,),dtype=torch.float32,device=m.device)
        keys=view.payload.k_cache
        kernels.tail_logits[(1,H,tr.cdiv(N,8))](q,keys,view.meta.active_slots,rows,lens,starts,chunks,maxima,lses,
            N,view.meta.active_slots.shape[1],H,m.num_kv_heads,m.head_dim,*q.stride()[:2],*keys.stride()[:2],m.head_dim**-.5)
        kernels.tail_norm[(H*64,)](lses,norm,N,tr.next_power_of_2(N))
        kernels.tail_ema[(1,H,tr.cdiv(N,128))](m.ema[g],m.seen[g],maxima,norm,rows,lens,chunks,N,H,128,enable_fp_fusion=False)
        m.refresh[g:g+1].fill_(1)
        # A nonnegative write marks a live row, independently of the prompt's last slot.
        select_lease(m,g,rows,lens,rows,initial=True)


class LeaseSparseRuntime(SparseMethodRuntime):
    def needs_attention_score(self,layer_idx,step):
        return not self.config.leasesparse_predictor_path and not step.is_prefill and layer_idx in self.config.leasesparse_sources

    def _prepare_decode_attention_score(self,layer_idx,state,batch_size,num_heads,max_len):
        state.attn_score=self.cache_manager.score_buffer[:batch_size]

    def build_prefill_selection(self,request):
        return self._full_selection(request.layer_idx)

    def build_decode_selection(self,request):
        m=self.cache_manager
        layer=request.layer_idx
        g=m.groups[layer]
        s=m.layer_batch_state
        B=s.req_indices.numel()
        if layer in m.sources:
            if layer == 0:
                m.lease_stream.wait_stream(torch.cuda.current_stream(self.device))
            if m.predictor is None:
                torch.cuda.current_stream(self.device).wait_stream(m.lease_stream)
            # Decide refresh before consuming the previous step's pending lease.
            kernels.begin[(1,)](m.starts[g],m.pending_starts[g],m.pending[g],m.bank[g],m.refresh[g:g+1],
                s.req_indices,s.context_lens,s.slot_mapping,m.max_buffer_rows,tr.next_power_of_2(B),B,self.config.leasesparse_reuse_steps,
                int(m.predict_this_step and m.prediction_commit) if m.predictor is not None else -1)
            m.make_view(g)
        if self.config.leasesparse_trace:
            kernels.mark_use[(B,)](m.layer_uses[layer],m.starts[g],s.req_indices,s.context_lens,s.slot_mapping,B)
        state=self.layer_batch_sparse_states[layer]
        return SparseSelection(kind='slots',req_indices=m.selected_rows[:B] if m.offload_enabled else torch.arange(B,device=self.device,dtype=torch.int32),
            context_lens=m.lengths[g,:B],max_context_len=4096,attn_score=state.attn_score,
            active_slots=m.slots[g,:B],global_req_indices=s.req_indices)

    def on_attention_end(self,event):
        m=self.cache_manager
        layer=event.layer_idx
        if event.forward_context.is_prefill:
            return
        if layer not in m.sources:
            if m.offload_enabled:
                m.lease_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(m.lease_stream):m.recall(layer,next_lease=True)
            return
        if m.predictor is not None and not m.predict_this_step:return
        g=m.groups[layer]
        s=m.layer_batch_state
        B=s.req_indices.numel()
        H=self.config.hf_config.num_attention_heads
        N=m.blocks
        m.lease_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(m.lease_stream):
            if m.predictor is not None:
                m.score_prediction((layer,),s.req_indices,s.context_lens,N)
                kernels.select[(B,)](m.reduced[layer],m.hot[layer],m.bank[layer],m.pending[layer],m.pending_starts[layer],
                    s.req_indices,s.context_lens,s.slot_mapping,m.refresh[layer:layer+1],m.lease_counts[layer],N,248,
                    tr.next_power_of_2(N),False,num_warps=8)
            else:
                kernels.clear_pool[(tr.cdiv(B*H*N,256),)](m.pool,m.observed,m.refresh[g:g+1],B*N,H,256)
                score=self.layer_batch_sparse_states[layer].attn_score
                kernels.pool_decode[(B,H)](score,m.positions[g],m.lengths[g],m.pool,m.observed,m.refresh[g:g+1],4096,N,H,*score.stride()[:2],4096,self.attn_softmax_scale)
                kernels.ema[(B,H,tr.cdiv(N,128))](m.ema[g],m.seen[g],m.pool,m.observed,s.req_indices,m.refresh[g:g+1],N,H,128,enable_fp_fusion=False)
                kernels.publish_seen[(B,tr.cdiv(N,128))](m.seen[g],m.observed,s.req_indices,m.refresh[g:g+1],N,128)
                select_lease(m,g,s.req_indices,s.context_lens,s.slot_mapping)
            if m.offload_enabled:
                m.make_view(g,next_lease=True)
                m.recall(layer,next_lease=True)
            m.lease_ready[g].record(m.lease_stream)


    def on_layer_end(self,event):
        m=self.cache_manager
        g=m.groups[event.layer_idx]
        if not event.forward_context.is_prefill and self.config.leasesparse_trace and event.layer_idx==m.ends[g]-1:
            with torch.cuda.stream(m.lease_stream):
                kernels.record_completion[(1,)](m.refresh[g:g+1],m.completion_counter,m.completed[g:g+1])
        # Close the side-stream fork before graph replay returns.
        if not event.forward_context.is_prefill and event.layer_idx==self.num_layers-1:
            torch.cuda.current_stream(self.device).wait_stream(self.cache_manager.lease_stream)
