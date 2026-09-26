"""Independent causal, sampling, loss, and shared-selection regression oracles."""

import itertools
import math

import pytest
import torch

from scripts.data.train_future_predictor import (
    FutureTCN,
    ResidualQueryPredictor,
    ConditionedQueryPredictor,
    MassQueryPredictor,
    forward_loss,
    prepare_layer,
    kl_loss,
    metrics,
    predict_scores,
    sample_tensor,
)


def _trajectory(prompt_tokens=151, decode_steps=100, keys=None, queries=None):
    """Build saved fields from a full token-level causal softmax, independently."""
    generator = torch.Generator().manual_seed(37)
    heads, kv_heads, dim = 4, 2, 6
    length = prompt_tokens + decode_steps
    if keys is None:
        keys = torch.randn(kv_heads, length, dim, generator=generator)
    if queries is None:
        queries = torch.randn(heads, 64 + decode_steps, dim, generator=generator)
    positions = torch.arange(prompt_tokens - 64, length)
    probabilities = []
    for head in range(heads):
        logits = queries[head] @ keys[head // 2].T / math.sqrt(dim)
        logits[torch.arange(length)[None] > positions[:, None]] = -torch.inf
        probabilities.append(logits.softmax(-1))
    probabilities = torch.stack(probabilities)
    sums = torch.stack([probabilities[..., start:start + 16].sum(-1)
                        for start in range(0, length, 16)], dim=-1)
    means = torch.stack([keys[:, start:min(start + 16, prompt_tokens)].mean(1)
                         for start in range(0, prompt_tokens, 16)], dim=1)
    x = dict(queries=queries, sums=sums, prompt_key_mean=means,
             row_positions=positions, prompt_tokens=prompt_tokens,
             decode_keys=keys[:, prompt_tokens:])
    tail_start = (prompt_tokens - 64) // 16 * 16
    boundary = dict(tail_start=tail_start, prompt_tail_keys=keys[:, tail_start:prompt_tokens])
    return x, boundary, probabilities, keys


def test_all_eight_history_positions_influence_prediction():
    """Catch the observed padding bug and a receptive field shorter than history."""
    torch.manual_seed(13)
    model = FutureTCN(heads=2, dim=6, hidden=8, horizon=4)
    q = torch.randn(2, 2, 8, 6, requires_grad=True)
    prediction = model(q)
    assert prediction.shape == (2, 4, 2, 6)
    prediction.square().sum().backward()
    influence = q.grad.abs().sum(dim=(0, 1, 3))
    assert torch.isfinite(influence).all()
    assert (influence > 0).all(), influence


def test_encoder_never_reads_future_history_positions():
    """Causal output at a position must be invariant to all later input changes."""
    torch.manual_seed(17)
    model = FutureTCN(heads=2, dim=6, hidden=8, horizon=4).eval()
    q = torch.randn(1, 2, 8, 6)
    with torch.no_grad():
        original = model.encode(q)
        assert original.shape == (1, 2, 8, 8)
        for stop in (1, 4, 7):
            changed = q.clone()
            changed[:, :, stop:] = torch.randn_like(changed[:, :, stop:]) * 30
            actual = model.encode(changed)
            torch.testing.assert_close(actual[:, :, :stop], original[:, :, :stop], rtol=0, atol=0)


@pytest.mark.parametrize("prompt_tokens,anchors", [(151, [0, 4, 65, 96]), (160, [0, 1])])
def test_dynamic_candidates_and_boundaries_match_full_token_attention(prompt_tokens, anchors):
    """Catch label offsets, omitted generated history, and partial-block rounding."""
    _check_dynamic_candidates(prompt_tokens, anchors, "cpu")


def _check_dynamic_candidates(prompt_tokens, anchors, device):
    x, boundary, probabilities, keys = _trajectory(prompt_tokens=prompt_tokens)
    batch = sample_tensor(prepare_layer(x, boundary, device), anchors)
    mass = torch.zeros_like(batch["mass"], device="cpu")
    protected = torch.zeros_like(batch["protected"], device="cpu")
    for item, anchor in enumerate(anchors):
        current = prompt_tokens + anchor
        cutoff = current - 64
        torch.testing.assert_close(batch["q"][item].cpu(), x["queries"][:, 56 + anchor:64 + anchor])
        for block in range(mass.shape[-1]):
            indices = list(range(max(64, block * 16), min(cutoff, (block + 1) * 16)))
            assert bool(batch["valid"][item, block]) == bool(indices)
            if not indices:
                continue
            expected_key = keys[:, indices].mean(1)
            actual_key = batch["partial_key"][item] if block == int(batch["partial_index"][item]) else batch["key"][:, block]
            torch.testing.assert_close(actual_key.cpu(), expected_key, rtol=2e-5, atol=2e-6)
            for step in range(4):
                mass[item, step, :, block] = probabilities[:, 64 + anchor + step, indices].sum(-1)
        for step in range(4):
            end = current + step + 1
            probs = probabilities[:, 64 + anchor + step]
            protected[item, step] = probs[:, :64].sum(-1) + probs[:, end - 64:end].sum(-1)
        assert int(batch["partial_count"][item]) == cutoff - int(batch["partial_index"][item]) * 16
        if anchor >= 65:
            assert batch["valid"][item, prompt_tokens // 16]
            assert cutoff > prompt_tokens
    torch.testing.assert_close(batch["mass"].cpu(), mass, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(batch["protected"].cpu(), protected, rtol=2e-5, atol=2e-6)
    target = mass / mass.sum(-1, keepdim=True)
    torch.testing.assert_close(batch["target"].cpu(), target, rtol=2e-5, atol=2e-6)
    actual = metrics(batch["target"].log(), batch["mass"], 999, batch["protected"])
    per_step = mass.sum(-1).mean(-1) + protected.mean(-1)
    expected = (per_step.mean().item(), per_step.mean().item(), 1., per_step.min(-1).values.mean().item())
    assert actual == pytest.approx(expected, abs=2e-6)


def test_future_keys_queries_and_partial_tail_cannot_leak_into_prediction():
    """Changing excluded keys in the same partial block cannot change its score."""
    anchor, prompt_tokens = 66, 151
    x, boundary, _, keys = _trajectory(prompt_tokens=prompt_tokens)
    expected = sample_tensor(prepare_layer(x, boundary), [anchor])
    changed_keys = keys.clone()
    changed_keys[:, prompt_tokens + anchor - 64:] += 3
    changed_queries = x["queries"].clone()
    changed_queries[:, 64 + anchor:] *= -2
    other, other_boundary, _, _ = _trajectory(prompt_tokens=prompt_tokens, keys=changed_keys, queries=changed_queries)
    actual = sample_tensor(prepare_layer(other, other_boundary), [anchor])
    for field in ("q", "valid", "partial_index", "partial_count"):
        torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
    torch.testing.assert_close(actual["partial_key"], expected["partial_key"], rtol=2e-5, atol=2e-6)
    torch.manual_seed(41)
    model = FutureTCN(heads=4, dim=6, hidden=8, horizon=4).eval()
    with torch.no_grad():
        _, scores = forward_loss(model, actual)
        _, old_scores = forward_loss(model, expected)
    torch.testing.assert_close(scores, old_scores, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU mathematical reference")
def test_cuda_dynamic_boundaries_match_independent_token_reference():
    """Exercise GPU indexing/reductions against CPU full-attention probabilities."""
    _check_dynamic_candidates(151, [0, 4, 65, 96], "cuda")


def test_gqa_scores_match_independent_head_mapping_and_dot_products():
    """Catch interleaved KV mapping and hardcoded Qwen head/dimension constants."""
    torch.manual_seed(23)
    qhat = torch.randn(2, 4, 6, 5)
    key = torch.randn(2, 3, 5)
    actual = predict_scores(qhat, key)
    expected = torch.empty(2, 4, 6, 3)
    for batch, step, head, block in itertools.product(range(2), range(4), range(6), range(3)):
        kv_head = 0 if head < 3 else 1
        expected[batch, step, head, block] = sum(
            float(qhat[batch, step, head, d]) * float(key[kv_head, block, d])
            for d in range(5)
        ) / math.sqrt(5)
    torch.testing.assert_close(actual, expected)


def test_kl_matches_probability_definition_including_zero_target_mass():
    """Catch non-KL weighting and wrong reduction across heads/horizon/batch."""
    target = torch.tensor([0.75, 0.25, 0.0]).expand(2, 4, 3, 3).clone()
    target[1, 1, 2] = torch.tensor([0.0, 0.0, 1.0])
    pred = torch.tensor([0.5, 0.25, 0.25]).expand_as(target)
    expected_terms = []
    for p, prediction in zip(target.reshape(-1, 3).tolist(), pred.reshape(-1, 3).tolist()):
        expected_terms.append(sum(a * math.log(a / b) for a, b in zip(p, prediction) if a))
    loss = kl_loss(pred.log(), target)
    assert float(loss.detach()) == pytest.approx(sum(expected_terms) / len(expected_terms), abs=1e-6)
    assert torch.isfinite(loss)


def test_identical_block_distributions_have_zero_kl_and_zero_gradient():
    """A fitted distribution must remain a stationary minimum of the loss."""
    target = torch.tensor([0.6, 0.3, 0.1]).expand(2, 4, 3, 3)
    logits = target.log().clone().requires_grad_()
    loss = kl_loss(logits, target)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0, abs=1e-6)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-7)


def _exhaustive_metrics(pred, target, budget):
    """Enumerate all shared sets; no tensor Top-K or production selection helper."""
    batches, steps, heads, blocks = target.shape
    coverage, oracle, overlap, worst = [], [], [], []
    for batch in range(batches):
        scores = [float(pred[batch, :, :, b].mean()) for b in range(blocks)]
        selected = tuple(sorted(range(blocks), key=lambda b: (-scores[b], b))[:budget])

        def step_coverage(chosen, step):
            return sum(float(target[batch, step, h, b]) for h in range(heads) for b in chosen) / heads

        step_values = [step_coverage(selected, step) for step in range(steps)]
        coverage.append(sum(step_values) / steps)
        worst.append(min(step_values))
        best_value, best_set = -1, None
        for chosen in itertools.combinations(range(blocks), budget):
            value = sum(step_coverage(chosen, step) for step in range(steps)) / steps
            if value > best_value:
                best_value, best_set = value, chosen
        oracle.append(best_value)
        overlap.append(len(set(selected) & set(best_set)) / budget)
    return tuple(sum(values) / batches for values in (coverage, oracle, overlap, worst))


def test_shared_set_metrics_match_exhaustive_fixed_budget_oracle():
    """Catch head/time reduction errors and per-step selection disguised as reuse."""
    torch.manual_seed(29)
    pred = torch.rand(3, 4, 2, 5) + 0.1
    pred /= pred.sum(-1, keepdim=True)
    target = torch.rand_like(pred) + 0.1
    target /= target.sum(-1, keepdim=True)
    actual = metrics(pred.log(), target, budget=2)
    expected = _exhaustive_metrics(pred, target, budget=2)
    assert actual == pytest.approx(expected, abs=1e-6)
    assert actual[0] <= actual[1] + 1e-6


def test_switching_attention_cannot_reselect_a_block_each_future_step():
    """The old per-step metric reports 0.95, while one shared block covers 0.5."""
    target = torch.tensor([[0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95]])[None, :, None, :]
    coverage, oracle, overlap, worst = metrics(target.log(), target, budget=1)
    assert (coverage, oracle, overlap, worst) == pytest.approx((0.5, 0.5, 1.0, 0.05), abs=1e-6)


def test_shared_selection_optimizes_mean_head_mass_not_max_head_score():
    """Different heads favor opposite blocks; the middle block maximizes coverage."""
    target = torch.tensor([[0.55, 0.44, 0.01], [0.01, 0.44, 0.55]])[None, None].expand(1, 4, 2, 3)
    actual = metrics(target.log(), target, budget=1)
    assert actual == pytest.approx((0.44, 0.44, 1.0, 0.44), abs=1e-6)


def test_tied_selection_is_deterministic_by_block_index():
    """Tied logits must not change overlap/coverage between devices or runs."""
    logits = torch.zeros(1, 4, 2, 4)
    target = torch.zeros_like(logits)
    target[..., 0] = 1
    assert metrics(logits, target, budget=1) == pytest.approx((1, 1, 1, 1), abs=1e-6)


def test_checkpoint_roundtrip_preserves_all_horizon_predictions(tmp_path):
    """Saved best-model weights must reproduce predictions after loading."""
    torch.manual_seed(31)
    model = FutureTCN(heads=2, dim=6, hidden=8, horizon=4).eval()
    q = torch.randn(1, 2, 8, 6)
    path = tmp_path / "predictor.pt"
    torch.save(model.state_dict(), path)
    restored = FutureTCN(heads=2, dim=6, hidden=8, horizon=4).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(q), model(q), rtol=0, atol=0)


def test_zero_history_rows_do_not_change_supervised_kl_or_gradient():
    """Heads with no historical mass supply no conditional label or KL gradient."""
    logits = torch.tensor([[[[0.2, -0.1, 0.3], [100., 3., -2.]]]], requires_grad=True)
    target = torch.tensor([[[[0.5, 0.3, 0.2], [0., 0., 0.]]]])
    actual = kl_loss(logits, target)
    reference_logits = logits[:, :, :1].detach().clone().requires_grad_()
    expected = kl_loss(reference_logits, target[:, :, :1])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad[:, :, :1], reference_logits.grad)
    torch.testing.assert_close(logits.grad[:, :, 1:], torch.zeros_like(logits.grad[:, :, 1:]))


def test_completely_unsupervised_batch_fails_explicitly():
    """Do not invent uniform labels or silently optimize an empty loss."""
    with pytest.raises(ValueError, match="no positive historical probability"):
        kl_loss(torch.zeros(1, 4, 2, 3), torch.zeros(1, 4, 2, 3))


def test_masked_zero_probability_candidates_have_finite_kl_gradient():
    """Negative-infinity candidate masks must contribute exactly zero loss."""
    target = torch.tensor([[[[0.6, 0.4, 0.]]]])
    logits = target.log().clone().requires_grad_()
    loss = kl_loss(logits, target)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0, abs=1e-6)
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-7)


def _ema_fixture():
    x, boundary, probs, _ = _trajectory(prompt_tokens=151, decode_steps=21)
    x['maxima'] = torch.stack([probs[..., s:s+16].amax(-1) for s in range(0, probs.shape[-1], 16)], -1)
    return x, boundary, probs


def _reference_ema(probs, prompt, reuse, phase=0, budget=2, include_tail=False):
    """Token-by-token sparse support reference; no production feedback helpers."""
    heads, rows, length = probs.shape
    blocks = (length + 15) // 16
    state = [[0.] * blocks for _ in range(heads)]
    seen = [False] * blocks

    def update(row, support):
        for block in range(blocks):
            indices = [i for i in support if i // 16 == block]
            if not indices:
                continue
            for head in range(heads):
                den = sum(float(probs[head, row, i]) for i in support)
                value = max(float(probs[head, row, i]) for i in indices) / den
                state[head][block] = .8 * state[head][block] + .2 * value if seen[block] else value
            seen[block] = True

    def select(anchor):
        eligible = range(4, (prompt + anchor - 64 + 15) // 16)
        return sorted(eligible, key=lambda b: (-max(state[h][b] for h in range(heads)), b))[:budget]

    def support_at(hot, lease, row):
        end = prompt - 64 + row + 1
        return [i for i in range(end) if i < 64 or i >= end - 64
                or (i < prompt + lease - 64 and i // 16 in hot)]

    def coverage(hot, lease, row):
        indices = support_at(hot, lease, row)
        return sum(float(probs[h, row, i]) for h in range(heads) for i in indices) / heads

    for row in range(64):
        update(row, list(range(prompt - 64 + row + 1)))
    hot, lease = select(0), 0
    if reuse == 1:
        values = []
        for anchor in range(rows - 64):
            if anchor:
                update(63 + anchor, support_at(hot, lease, 63 + anchor))
            hot, lease = select(anchor), anchor
            values.append(coverage(hot, lease, 64 + anchor))
        return torch.tensor(values) if include_tail else torch.tensor(values).unfold(0, 4, 1)
    values = {}
    for anchor in range(phase, rows - 64 - (0 if include_tail else 3), 4):
        if anchor:
            update(63 + anchor, support_at(hot, lease, 63 + anchor))
        hot, lease = select(anchor), anchor
        values[anchor] = [coverage(hot, lease, 64 + anchor + j) for j in range(min(4, rows - 64 - anchor))]
    return values


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_ema_replay_matches_independent_token_sparse_softmax(device):
    """Catch head-max/EMA timing errors, phase resets, stale boundaries and dense leakage."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from scripts.data.evaluate_future_ema import feedback_data, replay_ema_every_step, replay_ema_four_steps
    source, boundary, probs = _ema_fixture()
    x = feedback_data(source, boundary, device)
    n = probs.shape[1] - 64 - 3
    batch = sample_tensor(x, list(range(n)))
    actual1 = replay_ema_every_step(x, budget=2).cpu()
    expected1 = _reference_ema(probs, 151, 1)
    torch.testing.assert_close(actual1, expected1, atol=3e-6, rtol=1e-5)
    actual4 = replay_ema_four_steps(x, batch, budget=2).cpu()
    expected4 = torch.empty_like(actual4)
    for phase in range(4):
        for anchor, values in _reference_ema(probs, 151, 4, phase).items():
            expected4[anchor] = torch.tensor(values)
    torch.testing.assert_close(actual4, expected4, atol=3e-6, rtol=1e-5)


def test_ema_leaves_unobserved_blocks_unchanged_and_initializes_new_blocks():
    """Unobserved history must not decay or accept full-attention oracle feedback."""
    from scripts.data.evaluate_future_ema import update_ema
    state = torch.tensor([[.6, .3, 0.], [.1, .9, 0.]])
    seen = torch.tensor([True, True, False])
    observed = torch.tensor([True, False, True])
    score = torch.tensor([[.2, 99., .4], [.3, 99., .8]])
    actual, flags = update_ema(state, seen, score, observed)
    torch.testing.assert_close(actual, torch.tensor([[.52, .3, .4], [.14, .9, .8]]))
    assert flags.all()
    torch.testing.assert_close(state, torch.tensor([[.6, .3, 0.], [.1, .9, 0.]]))


@pytest.mark.parametrize('steps', [1, 3, 5, 21])
@pytest.mark.parametrize('network', [FutureTCN, ResidualQueryPredictor, ConditionedQueryPredictor, MassQueryPredictor])
def test_longbench_replay_preserves_short_answers_and_final_partial_lease(steps, network):
    """Count every natural decode step once, including outputs shorter than four."""
    from scripts.data.evaluate_longbench_future import replay, forecast
    from scripts.data.evaluate_future_ema import feedback_data
    source, boundary, probs, _ = _trajectory(prompt_tokens=151, decode_steps=steps)
    source['maxima'] = torch.stack([probs[...,s:s+16].amax(-1) for s in range(0,probs.shape[-1],16)],-1)
    torch.manual_seed(51)
    model=network(heads=4,dim=6,hidden=8,horizon=4).eval()
    if isinstance(model, MassQueryPredictor):
        with torch.no_grad():
            model.mass_head.weight.normal_(std=.1)
    actual=replay(model,source,boundary,device='cpu',budget=2,layer=7)
    one=_reference_ema(probs,151,1,include_tail=True)
    four_dict=_reference_ema(probs,151,4,include_tail=True)
    four=torch.tensor([v for values in four_dict.values() for v in values])
    x=feedback_data(source,boundary,'cpu')
    x['layer']=7
    predicted=[];oracle=[]
    for anchor in range(0,steps,4):
        batch=sample_tensor(x,[anchor],horizon=min(4,steps-anchor))
        hot=forecast(model,batch,budget=2).tolist()
        cutoff=151+anchor-64
        eligible=list(range(4,(cutoff+15)//16))
        possibilities=[]
        for selection in itertools.combinations(eligible,2):
            scores=[]
            for t in range(anchor,min(anchor+4,steps)):
                end=152+t
                positions=[i for i in range(end) if i<64 or i>=end-64 or (i<cutoff and i//16 in selection)]
                scores.append(float(probs[:,64+t,positions].sum(-1).mean()))
            possibilities.append(scores)
        oracle.extend(max(possibilities,key=lambda values:sum(values)/len(values)))
        for t in range(anchor,min(anchor+4,steps)):
            end=152+t
            positions=[i for i in range(end) if i<64 or i>=end-64 or (i<cutoff and i//16 in hot)]
            predicted.append(float(probs[:,64+t,positions].sum(-1).mean()))
    values=torch.stack((one,four,torch.tensor(predicted),torch.tensor(oracle)),1)
    expected=torch.stack((values.mean(0),torch.stack([values[a:a+4].amin(0) for a in range(0,steps,4)]).mean(0)),-1)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=1e-5)


@pytest.mark.parametrize('network', [FutureTCN, ResidualQueryPredictor, ConditionedQueryPredictor, MassQueryPredictor])
def test_longbench_forecast_does_not_use_available_future_label_length(network):
    """EOS must not reduce the four predicted horizons used to choose blocks."""
    from scripts.data.evaluate_longbench_future import forecast
    source,boundary,_,_=_trajectory()
    x=prepare_layer(source,boundary)
    x['layer']=7
    torch.manual_seed(53)
    model=network(heads=4,dim=6,hidden=8,horizon=4).eval()
    if isinstance(model, MassQueryPredictor):
        with torch.no_grad():
            model.mass_head.weight.normal_(std=.1)
    full=sample_tensor(x,[0],horizon=4)
    short=sample_tensor(x,[0],horizon=1)
    short['target'].zero_()
    torch.testing.assert_close(forecast(model,full,budget=2),forecast(model,short,budget=2))
    _, scores=forward_loss(model,full)
    if isinstance(scores, tuple):
        logits, mass_logits = scores
        probabilities = logits.softmax(-1) * mass_logits.sigmoid()[..., None]
    else:
        probabilities = scores.softmax(-1)
    expected=probabilities.mean((1,2))[0].argsort(descending=True,stable=True)[:2]
    torch.testing.assert_close(forecast(model,full,budget=2),expected)


@pytest.mark.parametrize('steps', [1, 2, 3])
def test_short_training_only_supervises_real_future_but_selects_four_forecasts(steps):
    """EOS must neither fabricate labels nor reveal the future length to selection."""
    from scripts.data.evaluate_longbench_future import forecast
    source, boundary, _, _ = _trajectory(decode_steps=steps)
    x = prepare_layer(source, boundary)
    model = FutureTCN(heads=4, dim=6, hidden=8, horizon=4)
    batch = sample_tensor(x, [0], horizon=steps)
    loss, scores = forward_loss(model, batch)
    assert scores.shape[1] == 4
    p = scores[:, :steps].softmax(-1)
    y = batch['target']
    valid = y > 0
    expected = torch.where(valid, y * (y.clamp_min(1e-30).log() - p.clamp_min(1e-30).log()), 0).sum(-1).mean()
    torch.testing.assert_close(loss.detach(), expected)
    loss.backward()
    gradient = model.out.weight.grad.reshape(4, 6, -1)
    assert gradient[:steps].abs().sum() > 0
    assert gradient[steps:].count_nonzero() == 0
    selected = scores.softmax(-1).mean((1, 2))[0].argsort(descending=True, stable=True)[:2]
    torch.testing.assert_close(selected, forecast(model, batch, budget=2))


def test_document_sampling_keeps_short_trajectories_and_early_windows():
    """A long document must not displace short task examples from training."""
    import random
    from scripts.data.train_future_predictor import balanced_anchors
    short = balanced_anchors([0], 128, random.Random(1))
    long = balanced_anchors(list(range(511)), 128, random.Random(1))
    assert len(short) == len(long) == 128
    assert short == [0] * 128
    assert all(0 <= a < 32 for a in long[:64])
    assert all(0 <= a < 511 for a in long)


@pytest.mark.parametrize('anchor', [0, 1, 9, 17])
def test_true_query_key_summaries_match_token_reference(anchor):
    """Catch sink/recent leakage, partial-block padding, and negative-Q max errors."""
    from scripts.data.evaluate_longbench_future import true_query_scores
    source, boundary, _, keys = _trajectory()
    x = prepare_layer(source, boundary)
    batch = sample_tensor(x, [anchor])
    p = source['prompt_tokens']
    bounds = tuple(torch.stack([op(keys[:, a:min(a+16,p)],dim=1).values
                                for a in range(0,p,16)],1) for op in (torch.min,torch.max))
    actual = true_query_scores(x, source, boundary, batch, anchor, 4, bounds)
    cutoff = p + anchor - 64
    for block in range((cutoff+15)//16):
        if block < 4:
            assert torch.isneginf(actual[...,block]).all()
            continue
        ks = keys[:,block*16:min(block*16+16,cutoff)].repeat_interleave(2,0)
        q = source['queries'][:,64+anchor:68+anchor].permute(1,0,2)
        mean = (q*ks.mean(1)).sum(-1)
        maximum = (q*ks.amax(1)).sum(-1)
        upper = (q.clamp_min(0)*ks.amax(1)+q.clamp_max(0)*ks.amin(1)).sum(-1)
        expected = torch.stack((mean,maximum,upper))/math.sqrt(6) + math.log(ks.shape[1]/16)
        torch.testing.assert_close(actual[...,block],expected,atol=1e-6,rtol=1e-5)
        token_max = torch.einsum('thd,hnd->thn',q,ks).amax(-1)
        assert (upper >= token_max-1e-6).all()
    assert torch.isneginf(actual[...,(cutoff+15)//16:]).all()


@pytest.mark.parametrize('horizon', [1, 4])
@pytest.mark.parametrize('history', [32, 64])
def test_long_history_has_complete_causal_receptive_field(horizon, history):
    """A longer input must influence prediction at every position without future leakage."""
    torch.manual_seed(13)
    model = FutureTCN(heads=2, dim=6, hidden=8, horizon=horizon, history=history)
    q = torch.randn(2, 2, history, 6, requires_grad=True)
    result = model(q)
    assert result.shape == (2, horizon, 2, 6)
    result.square().sum().backward()
    assert (q.grad.abs().sum((0,1,3)) > 0).all()
    with torch.no_grad():
        a = model.encode(q)
        changed = q.detach().clone(); changed[:,:,history//2:] += 3
        torch.testing.assert_close(a[:,:,:history//2], model.encode(changed)[:,:,:history//2])


def test_single_step_predictor_replay_matches_independent_token_coverage():
    """One-step mode must refresh each step and use that step's moving history boundary."""
    from scripts.data.evaluate_longbench_future import replay, forecast
    source,boundary,probs,_ = _trajectory(decode_steps=7)
    source['maxima'] = torch.stack([probs[...,a:a+16].amax(-1) for a in range(0,probs.shape[-1],16)],-1)
    model=FutureTCN(heads=4,dim=6,hidden=8,horizon=1,history=64).eval()
    actual=replay(model,source,boundary,device='cpu',budget=1)
    x=prepare_layer(source,boundary)
    predicted=[]; optimal=[]
    for step in range(7):
        batch=sample_tensor(x,[step],horizon=1,history=64)
        block=int(forecast(model,batch,budget=1)[0])
        cutoff=151+step-64; end=152+step
        def coverage(b):
            ids=[i for i in range(end) if i<64 or i>=end-64 or (i<cutoff and i//16==b)]
            return float(probs[:,64+step,ids].sum(-1).mean())
        predicted.append(coverage(block))
        optimal.append(max(coverage(b) for b in range(4,(cutoff+15)//16)))
    for index,values in [(2,predicted),(3,optimal)]:
        expected=torch.tensor([sum(values)/7,(min(values[:4])+min(values[4:]))/2])
        torch.testing.assert_close(actual[index],expected,atol=2e-6,rtol=1e-5)


def test_one_and_four_step_labels_align_at_same_forecast_origin():
    """Changing output horizon must not shift the first target or input history."""
    source,boundary,_,_=_trajectory()
    x=prepare_layer(source,boundary)
    one=sample_tensor(x,[0,3,17],horizon=1,history=64)
    four=sample_tensor(x,[0,3,17],horizon=4,history=64)
    for key in ('q','key','valid','partial_index','partial_key','partial_count'):
        torch.testing.assert_close(one[key],four[key],rtol=0,atol=0)
    for key in ('target','mass','protected','supervised'):
        torch.testing.assert_close(one[key],four[key][:,:1],rtol=0,atol=0)
    for i,anchor in enumerate((0,3,17)):
        torch.testing.assert_close(one['q'][i],source['queries'][:,anchor:64+anchor])
