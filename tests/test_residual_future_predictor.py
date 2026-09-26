"""Position, identity-path, temporal dependence and checkpoint regression checks."""
import io

import pytest
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb

from scripts.data.train_future_predictor import ConditionedQueryPredictor, ResidualQueryPredictor, build_predictor


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rotation_and_initial_forecast_match_teacher(device):
    """Catch split-half/adjacent-pair confusion and future-position off-by-one."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(11)
    model = ResidualQueryPredictor(dim=128).to(device)
    config = Qwen2Config(hidden_size=256, num_attention_heads=2, rope_theta=10000000.)
    rope = Qwen2RotaryEmbedding(config).to(device)
    positions = torch.tensor([list(range(127992,128000)), list(range(31992,32000))], device=device)
    content = torch.randn(2,2,8,128,device=device)
    cos,sin = rope(content, positions)
    q,_ = apply_rotary_pos_emb(content,content,cos,sin)
    torch.testing.assert_close(model.rotate(content,positions), q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model.rotate(q,positions,inverse=True),content, rtol=1e-5,atol=1e-5)
    future=positions[:,-1:]+torch.arange(1,5,device=device)
    latest=content[:,:,-1:].expand(-1,-1,4,-1)
    cos,sin=rope(latest,future)
    expected,_=apply_rotary_pos_emb(latest,latest,cos,sin)
    torch.testing.assert_close(model(q,positions),expected.permute(0,2,1,3),rtol=1e-5,atol=1e-5)


def test_linear_branch_can_extrapolate_known_content_trend():
    """A temporal map must act on time, not mix heads or feature coordinates."""
    model=ResidualQueryPredictor(heads=2,dim=6,hidden=8)
    positions=torch.arange(20,28)[None]
    trend=torch.tensor([.1,.2,-.1,.3,-.2,.4])
    content=positions[None,...,None].float()*trend
    content=content.expand(1,2,8,6)
    with torch.no_grad():
        model.temporal.weight[:, -2] = -torch.arange(1,5).float()
    actual=model(model.rotate(content,positions),positions)
    future=torch.arange(28,32)[None]
    expected_content=future[None,...,None].float()*trend
    expected=model.rotate(expected_content.expand(1,2,4,6),future).permute(0,2,1,3)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)


def test_residual_encoder_is_causal_and_keeps_identity_path():
    """Catch accidental right padding or omission of the internal skip connections."""
    torch.manual_seed(13)
    model=ResidualQueryPredictor(heads=2,dim=6,hidden=8)
    x=torch.randn(1,2,8,6)
    a=model.encode(x)
    changed=x.clone();changed[:,:,4:]+=10
    torch.testing.assert_close(a[:,:,:4],model.encode(changed)[:,:,:4])
    with torch.no_grad():
        for conv in (model.conv1,model.conv2):
            conv.weight.zero_();conv.bias.zero_()
    torch.testing.assert_close(model.encode(x),model.norm(model.in_proj(x)))


def test_nonlinear_branch_uses_all_history_and_can_train_after_zero_init():
    """Zero output initialization must not permanently block encoder gradients."""
    torch.manual_seed(19)
    model=ResidualQueryPredictor(heads=2,dim=6,hidden=8)
    q=torch.randn(2,2,8,6)
    pos=torch.arange(8)[None].expand(2,-1)
    target=torch.randn(2,4,2,6)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    loss=(model(q,pos)-target).square().mean()
    loss.backward()
    assert model.out.weight.grad.abs().sum()>0
    assert model.temporal.weight.grad.abs().sum()>0
    opt.step();opt.zero_grad()
    q.requires_grad_(True)
    # Isolate the nonlinear path; the temporal linear weights cannot hide a dead convolution.
    with torch.no_grad():model.temporal.weight.zero_()
    (model(q,pos)-target).square().mean().backward()
    assert (q.grad.abs().sum((0,1,3))>0).all()
    for conv in (model.conv1,model.conv2):
        assert conv.weight.grad.abs().sum()>0


def test_checkpoint_factory_preserves_residual_predictions():
    """Evaluation must reload the selected architecture and its RoPE state."""
    cfg=dict(architecture='residual',history=8,horizon=4,rope_theta=10000000.)
    model=build_predictor(cfg)
    buf=io.BytesIO();torch.save(dict(config=cfg,model=model.state_dict()),buf);buf.seek(0)
    ckpt=torch.load(buf,weights_only=True)
    restored=build_predictor(ckpt['config']);restored.load_state_dict(ckpt['model'])
    q=torch.randn(1,28,8,128);pos=torch.arange(8)[None]
    torch.testing.assert_close(model(q,pos),restored(q,pos))


def test_conditioned_model_keeps_initial_baseline_and_requires_layer():
    """Metadata must be explicit and new conditioning must not destroy initialization."""
    torch.manual_seed(29)
    m=ConditionedQueryPredictor(heads=2,dim=6,hidden=8)
    baseline=ResidualQueryPredictor(heads=2,dim=6,hidden=8)
    q=torch.randn(2,2,8,6);pos=torch.arange(8)[None].expand(2,-1)
    torch.testing.assert_close(m(q,pos,0),baseline(q,pos))
    torch.testing.assert_close(m(q,pos,27),baseline(q,pos))
    with pytest.raises(ValueError,match='actual layer'):
        m(q,pos,None)


def test_conditioned_decoder_learns_and_distinguishes_heads_layers_horizons():
    """Catch unused identity inputs, duplicated future steps and dead attention paths."""
    torch.manual_seed(31)
    m=ConditionedQueryPredictor(heads=2,dim=6,hidden=8)
    # Identical head inputs make differences attributable to head identity.
    q=torch.randn(2,1,8,6).expand(-1,2,-1,-1).clone()
    pos=torch.arange(8)[None].expand(2,-1)
    target=torch.randn(2,4,2,6)
    opt=torch.optim.AdamW(m.parameters(),lr=.01)
    (m(q,pos,3)-target).square().mean().backward();opt.step();opt.zero_grad()
    q.requires_grad_(True)
    (m(q,pos,3)-target).square().mean().backward()
    for name in ('conv1.weight','conv2.weight','query_proj.weight','key_proj.weight',
                 'value_proj.weight','condition.weight','horizon_embedding','head_embedding.weight'):
        grad=dict(m.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    assert m.layer_embedding.weight.grad[3].abs().sum()>0
    assert (q.grad.abs().sum((0,1,3))>0).all()
    with torch.no_grad():
        a=m(q,pos,3);b=m(q,pos,4)
        assert not torch.allclose(a,b)
        assert not torch.allclose(a[:,:,0],a[:,:,1])
        assert not torch.allclose(a[:,0],a[:,3])


def test_conditioned_checkpoint_reload():
    """Conditioned checkpoints must reload without reverting to the old decoder."""
    cfg=dict(architecture='conditioned',history=8,horizon=4,rope_theta=10000000.)
    m=build_predictor(cfg)
    with torch.no_grad():m.out.weight.normal_(std=.01)
    buf=io.BytesIO();torch.save(dict(config=cfg,model=m.state_dict()),buf);buf.seek(0)
    ckpt=torch.load(buf,weights_only=True);restored=build_predictor(ckpt['config'])
    restored.load_state_dict(ckpt['model'])
    q=torch.randn(1,28,8,128);pos=torch.arange(8)[None]
    torch.testing.assert_close(m(q,pos,27),restored(q,pos,27))
