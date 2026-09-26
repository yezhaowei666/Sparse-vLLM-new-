"""Train shared four-step block selection with exact sink/recent token boundaries."""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from sparsevllm.engine.sparse_methods.leasesparse import (
    FutureTCN, ResidualQueryPredictor, ConditionedQueryPredictor, MassQueryPredictor,
)


def build_predictor(config):
    # Checkpoints before architecture selection contain only the original TCN.
    architecture = config.get('architecture', 'tcn')
    kwargs = dict(history=config['history'], horizon=config['horizon'])
    if architecture == 'tcn':
        return FutureTCN(**kwargs)
    if architecture == 'residual':
        return ResidualQueryPredictor(**kwargs, rope_theta=config['rope_theta'])
    if architecture == 'mass':
        return MassQueryPredictor(**kwargs, rope_theta=config['rope_theta'])
    if architecture == 'conditioned':
        return ConditionedQueryPredictor(**kwargs, rope_theta=config['rope_theta'])
    raise ValueError(f'Unknown predictor architecture: {architecture}')


def predict_queries(model, batch):
    if isinstance(model, ConditionedQueryPredictor):
        return model(batch['q'], batch['positions'], batch['layer'])
    if isinstance(model, ResidualQueryPredictor):
        return model(batch['q'], batch['positions'])
    return model(batch['q'])


def predict_scores(qhat, key):
    heads, dim = qhat.shape[-2:]
    if heads % key.shape[0]:
        raise ValueError('Query heads must be divisible by KV heads')
    mapped = key.repeat_interleave(heads // key.shape[0], dim=0)
    return torch.einsum('bthd,hnd->bthn', qhat, mapped) * dim ** -.5


def kl_loss(scores, target):
    logp = scores.log_softmax(-1)
    # Zero-probability, masked candidates contribute exactly zero, including -inf logits.
    observed = target.sum(-1) > 0
    if not observed.any():
        raise ValueError('Batch has no positive historical probability for KL supervision')
    terms = F.kl_div(logp.masked_fill(target == 0, 0), target, reduction='none').sum(-1)
    return terms[observed].mean()


def prepare_layer(x, boundary, device='cpu'):
    """Precompute suffix token masses; no per-token probabilities are stored on disk."""
    p = int(x['prompt_tokens'])
    start = (p - 64) // 16 * 16
    q = x['queries'].to(device=device, dtype=torch.float32)
    raw = torch.cat((boundary['prompt_tail_keys'], x['decode_keys']), dim=1).to(device=device, dtype=torch.float32)
    if int(boundary['tail_start']) != start or raw.shape[1] != p + q.shape[1] - 64 - start:
        raise ValueError('Boundary supplement does not match trajectory length')
    h, rows, dim = q.shape
    if not torch.equal(x['row_positions'], torch.arange(p - 64, p + rows - 64)):
        raise ValueError('Non-contiguous trajectory query positions')
    kv, length, _ = raw.shape
    logits = torch.bmm(q.reshape(kv, -1, dim), raw.transpose(1, 2)).reshape(h, rows, length) * dim ** -.5
    positions = x['row_positions'].to(device)
    visible = torch.arange(start, start + length, device=device)[None, :] <= positions[:, None]
    padded = F.pad(logits.masked_fill(~visible[None], -torch.inf), (0, (-length) % 16), value=-torch.inf).reshape(h, rows, -1, 16)
    mask = F.pad(visible, (0, (-length) % 16)).reshape(rows, -1, 16)
    # The full-attention normalizer cancels inside each block. Redistribute its
    # saved mass using exact token logits, preserving FP16 block mass identically.
    within = padded.masked_fill(~mask.any(-1)[None, :, :, None], 0).softmax(-1).masked_fill(~mask[None], 0)
    sums = x['sums'].to(device=device, dtype=torch.float32)
    probs = (within * sums[:, :, start // 16:, None]).flatten(-2)[..., :length]
    suffix_cum = F.pad(probs.cumsum(-1), (1, 0))
    key_cum = F.pad(raw.cumsum(1), (0, 0, 1, 0))
    starts = torch.arange(0, length, 16, device=device)
    ends = (starts + 16).clamp_max(length)
    suffix_mean = (key_cum[:, ends] - key_cum[:, starts]) / (ends - starts)[None, :, None]
    key = torch.cat((x['prompt_key_mean'][:, :start // 16].to(device=device, dtype=torch.float32), suffix_mean), 1)
    if sums.shape != (h, rows, key.shape[1]) or not torch.isfinite(sums).all() or (sums < 0).any():
        raise ValueError('Invalid block probability tensor')
    totals = sums.sum(-1)
    if (totals - 1).abs().max() > .001:
        raise ValueError('Stored dense attention rows do not sum to one')
    return dict(q=q, key=key, sums=sums, totals=totals, suffix_cum=suffix_cum,
                key_cum=key_cum, tail_start=start, prompt_tokens=p)


def sample_tensor(x, anchors, horizon=4, history=8):
    """Freeze historical candidates at refresh; sink and recent are retained separately."""
    device = x['q'].device
    a = torch.as_tensor(anchors, device=device, dtype=torch.long)
    current = x['prompt_tokens'] + a
    cutoff = current - 64
    if history > 64 or history < 1 or cutoff.min() <= 64 or a.min() < 0 or a.max() + 64 + horizon > x['q'].shape[1]:
        raise ValueError('Invalid query history, horizon, anchor or historical region')
    qrows = 64 + a[:, None] + torch.arange(-history, 0, device=device)
    future = 64 + a[:, None] + torch.arange(horizon, device=device)
    q = x['q'][:, qrows].permute(1, 0, 2, 3)
    mass = x['sums'][:, future].permute(1, 2, 0, 3).clone()
    total = x['totals'][:, future].permute(1, 2, 0)
    block = torch.arange(x['key'].shape[1], device=device)
    valid = (block[None] >= 4) & (16 * block[None] < cutoff[:, None])
    # Pick the last eligible block, including a full 16-token final block.
    last = (cutoff - 1) // 16
    count = cutoff - last * 16
    lo, hi = (last * 16 - x['tail_start']).clamp_min(0), cutoff - x['tail_start']
    partial_key = ((x['key_cum'][:, hi] - x['key_cum'][:, lo]) / count[None, :, None]).permute(1, 0, 2)
    partial_key = torch.where((count == 16)[:, None, None], x['key'][:, last].permute(1, 0, 2), partial_key)
    cum = x['suffix_cum']
    partial_mass = (cum[:, future, hi[:, None]] - cum[:, future, lo[:, None]]).permute(1, 2, 0)
    full_mass = mass.gather(-1, last[:, None, None, None].expand(*mass.shape[:-1], 1)).squeeze(-1)
    partial_mass = torch.where((count == 16)[:, None, None], full_mass, partial_mass)
    mass.scatter_(-1, last[:, None, None, None].expand(*mass.shape[:-1], 1), partial_mass[..., None])
    mass.masked_fill_(~valid[:, None, None, :], 0)
    # For future step j, recent is [current+j-64, current+j); it never overlaps frozen history.
    end = current[:, None] + torch.arange(1, horizon + 1, device=device)
    recent = (cum[:, future, end - x['tail_start']] - cum[:, future, end - 64 - x['tail_start']]).permute(1, 2, 0)
    sink = x['sums'][:, future, :4].sum(-1).permute(1, 2, 0)
    mass = mass / total[..., None]
    protected = (sink + recent) / total
    norm = mass.sum(-1, keepdim=True)
    if not torch.isfinite(partial_key).all() or (mass < 0).any():
        raise ValueError('Invalid historical probability or key summary')
    # Some stored heads put zero mass on history. They have no conditional
    # historical distribution; retain them for coverage, omit their KL term.
    target = mass / torch.where(norm > 0, norm, 1)
    return dict(q=q, positions=current[:, None] + torch.arange(-history, 0, device=device), layer=x.get('layer'),
                key=x['key'], target=target, mass=mass, protected=protected, supervised=norm.squeeze(-1) > 0,
                valid=valid, partial_index=last, partial_key=partial_key, partial_count=count)


def scored_predictions(model, batch):
    if isinstance(model, MassQueryPredictor):
        qhat, mass_logits = model.predict_with_mass(batch['q'], batch['positions'], batch['layer'])
    else:
        qhat, mass_logits = predict_queries(model, batch), None
    scores = predict_scores(qhat, batch['key'])
    heads, dim = qhat.shape[-2:]
    mapped = batch['partial_key'].repeat_interleave(heads // batch['partial_key'].shape[1], dim=1)
    partial = torch.einsum('bthd,bhd->bth', qhat, mapped) * dim ** -.5
    # A key mean represents count tokens: partial block mass is scaled by count/16.
    partial = partial + (batch['partial_count'].float() / 16).log()[:, None, None]
    scores = scores.scatter(-1, batch['partial_index'][:, None, None, None].expand(*scores.shape[:-1], 1), partial[..., None])
    scores = scores.masked_fill(~batch['valid'][:, None, None, :], -torch.inf)
    return scores, mass_logits


def selection_probabilities(scores, mass_logits=None):
    probabilities = scores.softmax(-1)
    if mass_logits is not None:
        probabilities = probabilities * mass_logits.sigmoid()[..., None]
    return probabilities.mean((1, 2))


def mass_objective(scores, mass_logits, target, mass):
    # Labels cover actual future steps only; inference always averages all forecasts.
    scores, mass_logits = scores[:, :target.shape[1]], mass_logits[:, :target.shape[1]]
    observed = target.sum(-1) > 0
    conditional = kl_loss(scores, target) if observed.any() else scores[torch.isfinite(scores)].sum() * 0
    true_mass = mass.sum(-1)
    mass_loss = F.binary_cross_entropy_with_logits(mass_logits, true_mass)
    # Log-space aggregation avoids underflow. This is a standard KL between
    # normalized four-step/head-aggregated historical distributions.
    log_joint = scores.log_softmax(-1) + F.logsigmoid(mass_logits)[..., None]
    valid = torch.isfinite(scores[:, 0, 0])
    safe_joint = log_joint.masked_fill(~valid[:, None, None], 0)
    aggregate = safe_joint.logsumexp((1, 2)).masked_fill(~valid, -torch.inf)
    truth = mass.mean((1, 2))
    total = truth.sum(-1, keepdim=True)
    shared_target = truth / torch.where(total > 0, total, 1)
    shared = kl_loss(aggregate, shared_target) if (total > 0).any() else conditional * 0
    return conditional, mass_loss, shared


def forward_loss(model, batch, return_terms=False):
    scores, mass_logits = scored_predictions(model, batch)
    if mass_logits is not None:
        terms = mass_objective(scores, mass_logits, batch['target'], batch['mass'])
        result = (sum(terms), (scores.detach(), mass_logits.detach()))
    else:
        terms = (kl_loss(scores[:, :batch['target'].shape[1]], batch['target']),)
        result = (terms[0], scores.detach())
    return (*result, terms) if return_terms else result


def metrics(logits, target, budget, protected=None):
    """Full dense-probability coverage; one head-shared set reused across the horizon."""
    if isinstance(logits, tuple):
        logits, mass_logits = logits
    else:
        mass_logits = None
    p = selection_probabilities(logits, mass_logits)
    y = target.mean(2)
    valid = torch.isfinite(logits[:, 0, 0])
    k = min(budget, p.shape[-1])
    # Stable sorting gives the same lower-index tie rule to prediction and oracle.
    idx = p.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)[:, :k]
    oracle_idx = y.mean(1).masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)[:, :k]
    chosen = torch.zeros_like(valid).scatter_(-1, idx, valid.gather(-1, idx))
    oracle_set = torch.zeros_like(valid).scatter_(-1, oracle_idx, valid.gather(-1, oracle_idx))
    base = torch.zeros_like(y[..., 0]) if protected is None else protected.mean(2)
    per_step = (y * chosen[:, None]).sum(-1) + base
    oracle_per_step = (y * oracle_set[:, None]).sum(-1) + base
    overlap = (chosen & oracle_set).sum(-1) / chosen.sum(-1)
    return per_step.mean().item(), oracle_per_step.mean().item(), overlap.mean().item(), per_step.min(-1).values.mean().item()


def load_ids(root, split):
    return json.loads((root / 'splits.json').read_text())[split]


def make_items(root, split, horizon, history, limit=0, include_tail=False):
    items = []
    for doc in load_ids(root, split):
        record = json.loads((root / 'records' / f'{doc}.json').read_text())
        if record['status'] != 'success' or record['split'] != split or record['decode_steps'] < (1 if include_tail else horizon):
            raise ValueError(f'Invalid {split} trajectory: {doc}')
        for layer in range(28):
            items.extend((doc, layer, a) for a in range(record['decode_steps'] if include_tail else record['decode_steps'] - horizon + 1))
    if limit and limit < len(items):
        # A smoke must sample multiple documents, layers and decode positions.
        items = random.Random(20260923).sample(items, limit)
    if not items:
        raise ValueError(f'Empty {split} split')
    return items


def groups_for(items):
    groups = defaultdict(list)
    for doc, layer, anchor in items:
        groups[(doc, layer)].append(anchor)
    return groups


def balanced_anchors(anchors, count, rng):
    """Equal windows per document/layer; half emphasize the first 32 decode steps."""
    early = [a for a in anchors if a < 32]
    return rng.choices(early, k=count // 2) + rng.choices(anchors, k=count - count // 2)


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(root, boundary_root):
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, value in dict(history=64, block_size=16, query_heads=28, kv_heads=4, head_dim=128, layers=28).items():
        if protocol[name] != value:
            raise ValueError(f'Unsupported dataset {name}={protocol[name]}')
    seen, contexts = set(), set()
    for split in ('train', 'val', 'test'):
        for doc in load_ids(root, split):
            record = json.loads((root / 'records' / f'{doc}.json').read_text())
            if doc in seen or record['context_sha256'] in contexts:
                raise ValueError(f'Duplicate document/context across splits: {doc}')
            if record['status'] != 'success' or record['split'] != split:
                raise ValueError(f'Invalid split record: {doc}')
            seen.add(doc)
            contexts.add(record['context_sha256'])
    supplemental = json.loads((boundary_root / 'protocol.json').read_text())
    if supplemental['source_protocol_sha256'] != file_sha(root / 'protocol.json'):
        raise ValueError('Boundary supplement belongs to another dataset')


def run(args):
    if min(args.epochs, args.batch_size, args.log_every, args.lr, args.threads) <= 0 or min(args.limit_items, args.limit_val_items) < 0:
        raise ValueError('Invalid training counts, learning rate or limits')
    if args.windows_per_layer < 0:
        raise ValueError('Negative window quota')
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260923)
    random.seed(20260923)
    device = torch.device(args.device)
    root, out = args.data_root, args.output
    boundary_root = args.boundaries or root / 'boundaries_v1'
    validate_dataset(root, boundary_root)
    train = groups_for(make_items(root, 'train', args.horizon, args.history, args.limit_items, args.include_tail))
    val = groups_for(make_items(root, 'val', args.horizon, args.history, args.limit_val_items, args.include_tail))
    # Check all required supplements before creating a run or updating weights.
    for doc, layer in train.keys() | val.keys():
        if not (boundary_root / 'data' / doc / f'layer_{layer:02d}.pt').is_file():
            raise FileNotFoundError(f'Missing boundary supplement: {doc}/layer_{layer:02d}.pt')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.architecture in ('residual', 'conditioned', 'mass'):
        if args.model_config is None:
            raise ValueError('Residual predictor requires --model-config for exact RoPE')
        teacher = json.loads(args.model_config.read_text())
        protocol = json.loads((root / 'protocol.json').read_text())
        if file_sha(args.model_config) != protocol['model_config_sha256']:
            raise ValueError('RoPE config differs from data collection teacher')
        if teacher['model_type'] != 'qwen2' or teacher['rope_scaling'] is not None:
            raise ValueError('Residual predictor supports Qwen2 unscaled RoPE only')
        config.update(rope_theta=teacher['rope_theta'], model_config_sha256=file_sha(args.model_config))
    model = build_predictor(config).to(device)
    out.mkdir(parents=True, exist_ok=False)
    config.update(version=5, history=args.history, horizon=args.horizon, sink_tokens=64, recent_tokens=64,
                  block_size=16, selected_blocks=248, token_budget=4096, loss='standard KL on eligible historical blocks',
                  coverage='sink + sliding recent + frozen shared historical selection / full dense mass',
                  aggregation='mean over windows, layers and heads', seed=20260923,
                  zero_history_policy='zero-mass head/step has no KL label; keep it in all coverage metrics',
                  source_protocol_sha256=file_sha(root / 'protocol.json'), script_sha256=file_sha(Path(__file__)),
                  boundary_protocol_sha256=file_sha(boundary_root / 'protocol.json'), torch=str(torch.__version__))
    if args.architecture == 'mass':
        config.update(loss='conditional standard KL + historical mass BCE + shared standard KL; weights 1,1,1', checkpoint_selection='maximum validation coverage', initial_evaluation=True)
    if args.windows_per_layer:
        config.update(sampling='equal windows per document/layer; half first32, half full trajectory, with replacement',
                      optimizer_step='one per document/layer, accumulate horizon groups weighted by window count',
                      initialization='from scratch with original seed; no previous weights loaded')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_count = len(train) * args.windows_per_layer if args.windows_per_layer else sum(map(len, train.values()))
    val_count = len(val) * args.windows_per_layer if args.windows_per_layer else sum(map(len, val.values()))
    batches_per_epoch = len(train) if args.windows_per_layer else sum((len(v) + args.batch_size - 1) // args.batch_size for v in train.values())
    print(f'CONFIG device={device} train_items={train_count} val_items={val_count} batches_per_epoch={batches_per_epoch} params={sum(p.numel() for p in model.parameters())} loss={config['loss']} sink=64 recent=64 dynamic_history=true', flush=True)
    step, best = 0, (-float('inf') if args.architecture == 'mass' else float('inf'))
    start = time.perf_counter()
    if args.report:
        with args.report.open('a') as stream:
            stream.write(f'\n## 训练：{out.name}\n\n每轮{train_count}个训练窗口，{val_count}个验证窗口。\n\n'
                         '| Epoch | Train loss | Val loss | Coverage | Worst step | Oracle | Top-K overlap | 秒 | Val conditional KL | Val mass BCE | Val shared KL |\n'
                         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for epoch in range(0 if args.architecture == 'mass' else 1, args.epochs + 1):
        results = {}
        components = {}
        for split, groups in ((('val', val),) if epoch == 0 else (('train', train), ('val', val))):
            training = split == 'train'
            model.train(training)
            group_list = list(groups.items())
            if training:
                random.shuffle(group_list)
            totals = torch.zeros(5, dtype=torch.float64)
            component_totals = torch.zeros(3, dtype=torch.float64)
            samples = supervised_rows = zero_mass_rows = 0
            with torch.set_grad_enabled(training):
                for (doc, layer), anchors in group_list:
                    filename = f'layer_{layer:02d}.pt'
                    source = torch.load(root / 'data' / doc / filename, map_location='cpu', weights_only=True)
                    boundary = torch.load(boundary_root / 'data' / doc / filename, map_location='cpu', weights_only=True)
                    expected_sha = json.loads((root / 'records' / f'{doc}.json').read_text())['files'][filename]['sha256']
                    if boundary['source_sha256'] != expected_sha:
                        raise ValueError(f'Boundary source hash mismatch: {doc}/{filename}')
                    x = prepare_layer(source, boundary, device)
                    x['layer'] = layer
                    del source, boundary
                    anchors = list(anchors)
                    if args.windows_per_layer:
                        rng = random if training else random.Random(f'20260923/{doc}/{layer}')
                        anchors = balanced_anchors(anchors, args.windows_per_layer, rng)
                    if training:
                        random.shuffle(anchors)
                    by_horizon = defaultdict(list)
                    for anchor in anchors:
                        by_horizon[min(args.horizon, x['q'].shape[1] - 64 - anchor)].append(anchor)
                    batches = [(h, group[i:i + args.batch_size]) for h, group in by_horizon.items()
                               for i in range(0, len(group), args.batch_size)]
                    if training and args.windows_per_layer:
                        opt.zero_grad(set_to_none=True)
                    for horizon, selected in batches:
                        batch = sample_tensor(x, selected, horizon=horizon, history=args.history)
                        loss, scores, terms = forward_loss(model, batch, return_terms=True)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Non-finite {split} loss: {doc}/{layer}')
                        if training:
                            if not args.windows_per_layer:
                                opt.zero_grad(set_to_none=True)
                            (loss * (len(selected) / len(anchors) if args.windows_per_layer else 1)).backward()
                            # Check without modifying finite gradients.
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                            if not args.windows_per_layer:
                                opt.step()
                                step += 1
                        c, o, z, w = metrics(scores, batch['mass'], 248, batch['protected'])
                        observed = batch['supervised'].sum().item()
                        supervised_rows += observed
                        zero_mass_rows += batch['supervised'].numel() - observed
                        loss_weight = len(selected) if args.architecture == 'mass' else observed
                        totals += torch.tensor([loss.item() * loss_weight, c * len(selected), o * len(selected), z * len(selected), w * len(selected)], dtype=torch.float64)
                        component_totals[:len(terms)] += torch.tensor([v.item() for v in terms], dtype=torch.float64) * loss_weight
                        samples += len(selected)
                        if training and not args.windows_per_layer and step % args.log_every == 0:
                            print(f'epoch={epoch}/{args.epochs} step={step} samples={samples}/{train_count} loss={loss.item():.6f} coverage={c:.4f} oracle={o:.4f} topk_overlap={z:.4f}', flush=True)
                    if training and args.windows_per_layer:
                        opt.step()
                        step += 1
                        if step % args.log_every == 0:
                            print(f'epoch={epoch}/{args.epochs} step={step} samples={samples}/{train_count} loss={loss.item():.6f} coverage={c:.4f} oracle={o:.4f} topk_overlap={z:.4f}', flush=True)
                    del x, batch, scores, loss
            results[split] = (totals / torch.tensor([samples if args.architecture == 'mass' else supervised_rows, samples, samples, samples, samples])).tolist()
            components[split] = (component_totals / (samples if args.architecture == 'mass' else supervised_rows)).tolist()
            print(f'{split}_rows supervised={supervised_rows} zero_mass_rows={zero_mass_rows} samples={samples}', flush=True)
        va = results['val']
        train_loss_text = f"{results['train'][0]:.6f}" if epoch else 'not_run'
        print(f'epoch={epoch}/{args.epochs} train_loss={train_loss_text} val_loss={va[0]:.6f} coverage={va[1]:.4f} worst_step_coverage={va[4]:.4f} oracle={va[2]:.4f} topk_overlap={va[3]:.4f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        print(f'val_components conditional_KL={components["val"][0]:.6f} mass_BCE={components["val"][1]:.6f} shared_KL={components["val"][2]:.6f}', flush=True)
        checkpoint = dict(model=model.state_dict(), epoch=epoch, step=step, val_loss=va[0],
                          metrics=dict(zip(('loss' if args.architecture == 'mass' else 'kl', 'coverage', 'oracle', 'topk_overlap', 'worst_step_coverage'), va)),
                          config=config, loss_components=components)
        if args.report:
            with args.report.open('a') as stream:
                stream.write(f'| {epoch} | {train_loss_text} | {va[0]:.6f} | {va[1]:.6%} | {va[4]:.6%} | {va[2]:.6%} | {va[3]:.6%} | {time.perf_counter()-start:.1f} | {components["val"][0]:.6f} | {components["val"][1]:.6f} | {components["val"][2]:.6f} |\n')
        if epoch == 0:
            torch.save(checkpoint, out / 'initial.pt')
            continue
        torch.save(checkpoint, out / 'last.pt')
        improved = va[1] > best if args.architecture == 'mass' else va[0] < best
        if improved:
            best = va[1] if args.architecture == 'mass' else va[0]
            torch.save(checkpoint, out / 'best.pt')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--boundaries', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--history', type=int, choices=(8,32,64), default=8)
    p.add_argument('--architecture', choices=('tcn', 'residual', 'conditioned', 'mass'), default='tcn')
    p.add_argument('--model-config', type=Path)
    p.add_argument('--horizon', type=int, choices=(1,4), default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--limit-items', type=int, default=0)
    p.add_argument('--limit-val-items', type=int, default=0)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--include-tail', action='store_true')
    p.add_argument('--windows-per-layer', type=int, default=0,
                   help='Equal sampled windows per document/layer; half from first 32 steps')
    p.add_argument('--report', type=Path, help='Append epoch metrics to a Markdown experiment record')
    run(p.parse_args())


if __name__ == '__main__':
    main()
