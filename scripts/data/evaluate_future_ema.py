"""Matched dense-trajectory coverage of native ordinary EMA and future prediction."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.data.train_future_predictor import (
    FutureTCN, file_sha, forward_loss, load_ids, metrics, prepare_layer,
    sample_tensor, validate_dataset,
)


def feedback_data(source, boundary, device):
    """Restore suffix token probabilities without exposing unselected scores to EMA."""
    x = prepare_layer(source, boundary, device)
    raw = torch.cat((boundary['prompt_tail_keys'], source['decode_keys']), 1).to(device).float()
    h, rows, dim = x['q'].shape
    length = raw.shape[1]
    logits = torch.bmm(x['q'].reshape(raw.shape[0], -1, dim), raw.transpose(1, 2)).reshape(h, rows, length) * dim ** -.5
    positions = torch.arange(x['tail_start'], x['tail_start'] + length, device=device)
    mask = positions[None] <= source['row_positions'].to(device)[:, None]
    padded_mask = F.pad(mask, (0, (-length) % 16)).reshape(rows, -1, 16)
    logits = F.pad(logits.masked_fill(~mask[None], -torch.inf), (0, (-length) % 16), value=-torch.inf).reshape(h, rows, -1, 16)
    within = logits.masked_fill(~padded_mask.any(-1)[None, :, :, None], 0).softmax(-1).masked_fill(~padded_mask[None], 0)
    token_probs = (within * x['sums'][:, :, x['tail_start'] // 16:, None]).flatten(-2)[..., :length]
    x.update(token_probs=token_probs, token_positions=positions,
             maxima=source['maxima'].to(device).float(), blocks=torch.arange(x['key'].shape[1], device=device))
    return x


def update_ema(state, seen, score, observed, alpha=.2):
    updated = torch.where(seen[None], (1 - alpha) * state + alpha * score, score)
    return torch.where(observed[None], updated, state), seen | observed


def initialize_ema(x):
    state = torch.zeros_like(x['maxima'][:, 0])
    seen = torch.zeros_like(x['blocks'], dtype=torch.bool)
    for row in range(64):
        visible = x['blocks'] * 16 <= x['prompt_tokens'] - 64 + row
        state, seen = update_ema(state, seen, x['maxima'][:, row], visible)
    return state, seen


def choose(state, x, anchor, budget):
    cutoff = x['prompt_tokens'] + anchor - 64
    valid = (x['blocks'] >= 4) & (x['blocks'] * 16 < cutoff)
    scores = state.amax(0).masked_fill(~valid, -torch.inf)
    # Match native head-max EMA ranking and lower-index ties, not predictor head-mean.
    order = scores.argsort(descending=True, stable=True)
    return order[:min(budget, int(valid.sum()))]


def observe(x, hot, lease_anchor, row):
    """Only the currently consumed sparse support updates EMA, with sparse Softmax."""
    p = x['prompt_tokens']
    current_length = p - 64 + row + 1
    cutoff = p + lease_anchor - 64
    selected = torch.zeros_like(x['blocks'], dtype=torch.bool).scatter_(0, hot, True)
    split = x['tail_start'] // 16
    prefix_observed = selected[:split] | (x['blocks'][:split] < 4)
    pos = x['token_positions']
    tail_observed = (pos < current_length) & ((pos >= current_length - 64) | ((pos < cutoff) & selected[pos // 16]))
    tail = x['token_probs'][:, row] * tail_observed[None]
    denominator = (x['sums'][:, row, :split] * prefix_observed).sum(-1) + tail.sum(-1)
    if (denominator <= 0).any():
        raise ValueError('Sparse support has zero probability')
    suffix_max = F.pad(tail, (0, (-tail.shape[-1]) % 16)).reshape(tail.shape[0], -1, 16).amax(-1)
    suffix_observed = F.pad(tail_observed, (0, (-tail.shape[-1]) % 16)).reshape(-1, 16).any(-1)
    observed = torch.cat((prefix_observed, suffix_observed))
    maxima = torch.cat((x['maxima'][:, row, :split] * prefix_observed, suffix_max), -1)
    feedback = maxima / denominator[:, None]
    coverage = (denominator / x['totals'][:, row]).mean()
    return feedback, observed, coverage


def replay_ema_every_step(x, budget=248):
    state, seen = initialize_ema(x)
    steps = x['q'].shape[1] - 64
    coverage = []
    for anchor in range(steps):
        hot = choose(state, x, anchor, budget)
        score, observed, c = observe(x, hot, anchor, 64 + anchor)
        coverage.append(c)
        state, seen = update_ema(state, seen, score, observed)
    # Same four-step windows as the predictor; EMA may refresh within the window.
    return torch.stack(coverage).unfold(0, 4, 1)


def replay_ema_four_steps(x, batch, budget=248):
    windows = batch['q'].shape[0]
    coverage = torch.empty((windows, 4), device=x['q'].device)
    initial, initial_seen = initialize_ema(x)
    initial_hot = choose(initial, x, 0, budget)
    mass = batch['mass'].mean(2)
    protected = batch['protected'].mean(2)
    # Four independent causal schedules cover every original sliding-window anchor.
    for phase in range(4):
        state, seen = initial.clone(), initial_seen.clone()
        hot, lease_anchor = initial_hot, 0
        for anchor in range(phase, windows, 4):
            if anchor:
                score, observed, _ = observe(x, hot, lease_anchor, 63 + anchor)
                state, seen = update_ema(state, seen, score, observed)
            hot = choose(state, x, anchor, budget)
            lease_anchor = anchor
            coverage[anchor] = mass[anchor, :, hot].sum(-1) + protected[anchor]
    return coverage


def summarize(per_step):
    return [per_step.mean().item(), per_step.min(-1).values.mean().item()]


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.data_root
    boundaries = root / 'boundaries_v1'
    validate_dataset(root, boundaries)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['config']['source_protocol_sha256'] != file_sha(root / 'protocol.json'):
        raise ValueError('Predictor checkpoint and dataset differ')
    if checkpoint['config']['version'] != 3:
        raise ValueError('Expected corrected v3 predictor checkpoint')
    if args.report.exists():
        raise FileExistsError(f'Report already exists: {args.report}')
    model = FutureTCN().to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    start = time.perf_counter()
    totals = torch.zeros((4, 2), dtype=torch.float64)
    samples = 0
    docs = []
    for doc in load_ids(root, 'val'):
        record = json.loads((root / 'records' / f'{doc}.json').read_text())
        doc_total = torch.zeros_like(totals)
        doc_samples = 0
        for layer in range(28):
            filename = f'layer_{layer:02d}.pt'
            source = torch.load(root / 'data' / doc / filename, map_location='cpu', weights_only=True)
            boundary = torch.load(boundaries / 'data' / doc / filename, map_location='cpu', weights_only=True)
            if boundary['source_sha256'] != record['files'][filename]['sha256']:
                raise ValueError(f'Boundary source mismatch: {doc}/{filename}')
            x = feedback_data(source, boundary, args.device)
            n = x['q'].shape[1] - 64 - 3
            batch = sample_tensor(x, list(range(n)))
            _, scores = forward_loss(model, batch)
            pred, oracle, _, worst = metrics(scores, batch['mass'], 248, batch['protected'])
            pred_per_step = (batch['mass'].mean(2) * torch.zeros_like(batch['valid']).scatter_(
                -1, scores.softmax(-1).mean(2).mean(1).argsort(dim=-1, descending=True, stable=True)[:, :248], True)[:, None]).sum(-1) + batch['protected'].mean(2)
            oracle_idx = batch['mass'].mean((1, 2)).masked_fill(~batch['valid'], -torch.inf).argsort(dim=-1, descending=True, stable=True)[:, :248]
            oracle_per_step = batch['mass'].mean(2).gather(-1, oracle_idx[:, None].expand(-1, 4, -1)).sum(-1) + batch['protected'].mean(2)
            ema1 = replay_ema_every_step(x)
            ema4 = replay_ema_four_steps(x, batch)
            if not (torch.isfinite(ema1).all() and torch.isfinite(ema4).all()):
                raise ValueError(f'Invalid EMA coverage: {doc}/{layer}')
            if min(ema1.min(), ema4.min()) < -1e-6 or max(ema1.max(), ema4.max()) > 1 + 1e-5:
                raise ValueError('Coverage outside probability range')
            if (ema4.mean(1) > oracle_per_step.mean(1) + 2e-6).any():
                raise ValueError('EMA4 exceeds matched shared-set oracle')
            torch.testing.assert_close(torch.tensor(summarize(pred_per_step)), torch.tensor([pred, worst]), atol=1e-6, rtol=0)
            values = torch.tensor([summarize(ema1), summarize(ema4), [pred, worst], summarize(oracle_per_step)], dtype=torch.float64)
            doc_total += values * n
            doc_samples += n
            del source, boundary, x, batch, scores, ema1, ema4, pred_per_step, oracle_per_step
        totals += doc_total
        samples += doc_samples
        result = doc_total / doc_samples
        docs.append((doc, doc_samples, result.tolist()))
        print('DOC', doc, 'status=success', 'windows_layers=', doc_samples, 'metrics=', result.tolist(), flush=True)
    overall = totals / samples
    if abs(overall[2, 0].item() - checkpoint['metrics']['coverage']) > 1e-6:
        raise ValueError('Predictor coverage did not reproduce the trained validation metric')
    elapsed = time.perf_counter() - start
    names = ['普通EMA：每步刷新', '普通EMA：4步复用', '未来预测器：4步复用', '未来4步共享集合oracle']
    lines = ['# 同数据普通EMA覆盖率对照', '', '## 目的和配置', '',
             '在预测器同一验证集上比较普通EMA每步刷新、普通EMA4步复用与冻结预测器，不重新生成答案，不修改引擎。',
             f'- 数据：`{root}`，4篇val、28层，共{samples}个层/窗口；不使用test。',
             '- 模型：Qwen2.5-7B-Instruct-1M，源轨迹BF16；源块质量FP16、块最大值BF16，回放FP32。',
             '- 固定4096预算、sink64/recent64、块16、最多248历史块、逐层独立；生成历史和部分块处理完全相同。',
             '- 普通EMA沿用原生alpha=0.2、块内max、头间max；最近64个预填充Query初始化。观测过的块按0.8旧值+0.2当前值更新，首次观测直接赋值，未观测保留原值。',
             '- Decode反馈只来自各方法自身实际使用的稀疏集合，按各头可见概率总质量重新归一化，模拟稀疏Softmax；不从池外或未来标签更新EMA。',
             '- EMA4仅刷新时更新EMA，使用刚计算完成的前一步反馈，新集合从下一步使用。复用期间recent逐步滑动，冻结历史Token边界。',
             '- 为精确覆盖预测器所有重叠4步窗口，EMA4独立重放刷新起点0/1/2/3的四条因果轨迹，每个窗口只计一次；非零起点的首段先用共同预填充集合。未逐窗口重置EMA。',
             '- EMA1按自身连续轨迹逐步更新选块，再按相同4步窗口计算平均及最差步。它在窗口内可换块，不受4步共同集合oracle约束。',
             '- coverage=(sink+recent+所选历史的完整注意力质量)/完整可见注意力总质量；窗口和层等权，头取均值。最差步为每个4步窗口内最低覆盖率再平均。',
             '- 普通EMA原生max排序与预测器mean概率排序各自保留；相同约束和评价目标不等于改写EMA为另一估计器。',
             '- 后缀Token概率按原块质量乘块内条件Softmax重建，保护部分块边界；前缀最大值采用原保存BF16统计。两方法共用同一稠密参考Q/K与生成轨迹，本实验是离线选择覆盖率，不是稀疏自由生成质量。',
             f'- 检查点：`{args.checkpoint}`，epoch={checkpoint["epoch"]}，SHA256={file_sha(args.checkpoint)}。',
             f'- 脚本SHA256：{file_sha(Path(__file__))}。', '', '## 命令', '', '```bash',
             f'.venv/bin/python -u -m scripts.data.evaluate_future_ema --data-root {root} --checkpoint {args.checkpoint} --device {args.device} --report {args.report}',
             '```', '', '## 总体结果', '', '| 方法 | 平均覆盖率 | 周期最差步覆盖率 |', '|---|---:|---:|']
    for name, values in zip(names, overall):
        lines.append(f'| {name} | {values[0]*100:.6f}% | {values[1]*100:.6f}% |')
    lines += ['', f'预测器减EMA4：{100*(overall[2,0]-overall[1,0]):+.6f}个百分点；预测器减EMA1：{100*(overall[2,0]-overall[0,0]):+.6f}个百分点。',
              '', '## 逐文档结果', '', '| 文档 | 层/窗口数 | EMA1 | EMA4 | 预测器 | oracle |', '|---|---:|---:|---:|---:|---:|']
    for doc, count, values in docs:
        lines.append(f'| {doc} | {count} | ' + ' | '.join(f'{v[0]*100:.6f}%' for v in values) + ' |')
    lines += ['', f'验证：全部文档status=success；概率范围、EMA4共同集合上限检查通过；预测器覆盖率复现训练验证指标，误差≤1e-6。总耗时{elapsed:.1f}秒。',
              '结果为4篇验证文档的描述性对照，不把重叠窗口当作独立文档进行显著性推断。']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines) + '\n')
    print('FINAL', overall.tolist(), 'elapsed=', elapsed, 'report=', args.report, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())


if __name__ == '__main__':
    main()
