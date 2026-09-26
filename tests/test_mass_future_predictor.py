"""Independent checks of full-mass ranking, objective and causal supervision."""
import io

import pytest
import torch

from scripts.data.train_future_predictor import (
    build_predictor, mass_objective, selection_probabilities,
)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full_mass_ranking_and_shared_kl_match_manual_probability(device):
    """Conditional normalization must not give a 1%-history head equal influence."""
    p = torch.tensor([[[[.99, .01, 0.], [.1, .9, 0.]]]], device=device)
    m = torch.tensor([[[.01, .8]]], device=device)
    logits = p.log().requires_grad_()
    mass_logits = torch.logit(m).requires_grad_()
    truth = torch.tensor([[[[.002, .008, 0.], [.56, .24, 0.]]]], device=device)
    target = truth / truth.sum(-1, keepdim=True)
    rank = selection_probabilities(logits, mass_logits)
    torch.testing.assert_close(rank, (p * m[..., None]).sum((1, 2)) / 2)
    assert selection_probabilities(logits).argmax(-1).item() == 0
    assert rank.argmax(-1).item() == 1
    terms = mass_objective(logits, mass_logits, target, truth)
    expected_cond = (target[..., :2] * (target[..., :2].log() - p[..., :2].log())).sum(-1).mean()
    expected_mass = -(m * m.log() + (1-m) * (1-m).log()).mean()
    t = truth.sum((1, 2))[:, :2]; t = t / t.sum(-1, keepdim=True)
    predicted = rank[:, :2] / rank[:, :2].sum(-1, keepdim=True)
    expected_shared = (t * (t.log() - predicted.log())).sum()
    for actual, expected in zip(terms, (expected_cond, expected_mass, expected_shared)):
        torch.testing.assert_close(actual, expected)
    sum(terms).backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(mass_logits.grad).all()


def test_zero_history_and_short_labels_have_finite_gradients():
    """No-history heads still supervise mass; nonexistent future steps get no labels."""
    scores = torch.randn(2, 4, 2, 3, requires_grad=True)
    mass_logits = torch.randn(2, 4, 2, requires_grad=True)
    truth = torch.zeros(2, 1, 2, 3)
    terms = mass_objective(scores, mass_logits, truth, truth)
    sum(terms).backward()
    assert terms[0] == 0 and terms[2] == 0
    assert torch.isfinite(scores.grad).all()
    assert mass_logits.grad[:, :1].abs().sum() > 0
    assert mass_logits.grad[:, 1:].count_nonzero() == 0
    assert (mass_logits.grad[:, :1] > 0).all()


def test_mass_checkpoint_preserves_nonconstant_inference_and_initial_ranking():
    """A restored model must carry learned mass, not silently use equal head weights."""
    cfg = dict(architecture='mass', history=8, horizon=4, rope_theta=10000000.)
    model = build_predictor(cfg)
    q = torch.randn(1, 28, 8, 128)
    positions = torch.arange(8)[None]
    _, initial = model.predict_with_mass(q, positions, 7)
    logits = torch.randn(1, 4, 28, 5)
    torch.testing.assert_close(selection_probabilities(logits, initial), selection_probabilities(logits) * .5)
    with torch.no_grad():
        model.mass_head.weight.normal_(std=.05)
    expected = model.predict_with_mass(q, positions, 7)
    assert expected[1].std() > 0
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf); buf.seek(0)
    restored = build_predictor(cfg)
    restored.load_state_dict(torch.load(buf, weights_only=True))
    actual = restored.predict_with_mass(q, positions, 7)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
