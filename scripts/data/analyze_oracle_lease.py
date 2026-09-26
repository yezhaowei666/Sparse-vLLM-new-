"""Stateful, future-informed greedy EMA leases on saved dense trajectories."""
import argparse
import json
import shlex
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from scripts.data.evaluate_future_ema import feedback_data, initialize_ema, choose, observe, update_ema
from scripts.data.train_future_predictor import file_sha, load_ids

LENGTHS = (4, 8, 12, 16)
TOLERANCES = (0., .005, .01)


def pick_length(coverage, reference, tolerance, lengths):
    losses = [(reference[:n] - coverage[:n]).mean().item() for n in lengths]
    valid = [n for n, loss in zip(lengths, losses) if loss <= tolerance + 1e-7]
    # Explicit minimum-lease action when past state divergence makes all candidates infeasible.
    return (max(valid), False) if valid else (min(lengths), True)


def replay(x, lease=4, reference=None, tolerance=0.):
    steps = x['q'].shape[1] - 64
    state, seen = initialize_ema(x)
    coverage = torch.empty(steps, device=x['q'].device)
    anchor, hot, old_anchor = 0, None, 0
    leases, violations = [], []
    while anchor < steps:
        if anchor:
            score, observed, _ = observe(x, hot, old_anchor, 63 + anchor)
            state, seen = update_ema(state, seen, score, observed)
        hot = choose(state, x, anchor, 248)
        cap = min(lease, steps - anchor)
        future = torch.stack([observe(x, hot, anchor, 64 + anchor + j)[2] for j in range(cap)])
        if reference is None:
            length, violated = cap, False
        else:
            lengths = sorted({min(n, steps - anchor) for n in LENGTHS if n <= lease})
            length, violated = pick_length(future, reference[anchor:anchor+cap], tolerance, lengths)
        coverage[anchor:anchor+length] = future[:length]
        leases.append(length)
        violations.append(violated)
        old_anchor, anchor = anchor, anchor + length
    if not torch.isfinite(coverage).all() or coverage.min() < -1e-6 or coverage.max() > 1 + 1e-5:
        raise ValueError('Coverage outside probability range')
    return coverage.cpu().double(), leases, violations


def summary(cov, ref, leases, violations):
    loss = ref - cov
    return dict(steps=len(cov), coverage_sum=cov.sum().item(), loss_sum=loss.sum().item(),
                refreshes=len(leases)-1, leases=len(leases), violated_leases=sum(violations),
                worst_step_loss=loss.max().item(), step_loss_over_1pp=int((loss > .01).sum()),
                lease_counts=dict(Counter(leases)))


def combine(rows):
    out = {k: sum(r[k] for r in rows) for k in ('steps', 'coverage_sum', 'loss_sum', 'refreshes', 'leases', 'violated_leases', 'step_loss_over_1pp')}
    out['worst_step_loss'] = max(r['worst_step_loss'] for r in rows)
    hist = Counter()
    for row in rows:
        hist.update(row['lease_counts'])
    out['lease_counts'] = dict(sorted(hist.items()))
    return out


def table(results):
    base = results['固定4步']['refreshes']
    lines = ['| 方法 | 覆盖率 | 相对4步下降(百分点) | 刷新减少 | 平均租期 | 门槛违约租期 | 单步下降>1百分点占比 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, s in results.items():
        lines.append(f'| {name} | {100*s["coverage_sum"]/s["steps"]:.4f}% | {100*s["loss_sum"]/s["steps"]:+.4f} | {100*(1-s["refreshes"]/base):.2f}% | {s["steps"]/s["leases"]:.2f} | {s["violated_leases"]}/{s["leases"]} | {100*s["step_loss_over_1pp"]/s["steps"]:.2f}% |')
    return lines


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Independent handcrafted choices: longest admissible and explicit infeasibility.
    assert pick_length(torch.tensor([1.]*4+[0.]*4), torch.ones(8), 0., [4,8]) == (4, False)
    assert pick_length(torch.ones(8), torch.ones(8), 0., [4,8]) == (8, False)
    assert pick_length(torch.zeros(8), torch.ones(8), 0., [4,8]) == (4, True)
    started = time.monotonic()
    protocol = json.loads((args.data_root/'protocol.json').read_text())
    totals, docs = {}, []
    for doc in load_ids(args.data_root, 'val'):
        per_doc = {}
        record = json.loads((args.data_root/'records'/f'{doc}.json').read_text())
        for layer in range(protocol['layers']):
            filename = f'layer_{layer:02d}.pt'
            source = torch.load(args.data_root/'data'/doc/filename, weights_only=True, map_location='cpu')
            boundary = torch.load(args.data_root/'boundaries_v1/data'/doc/filename, weights_only=True, map_location='cpu')
            if boundary['source_sha256'] != record['files'][filename]['sha256']:
                raise ValueError('Boundary source mismatch')
            x = feedback_data(source, boundary, 'cuda')
            base, leases, violations = replay(x)
            outputs = {'固定4步': summary(base, base, leases, violations)}
            ref = base.to(device='cuda', dtype=torch.float32)
            if layer == 0:
                same, same_leases, same_violations = replay(x, lease=4, reference=ref)
                torch.testing.assert_close(same, base, atol=0., rtol=0.)
                assert leases == same_leases and not any(same_violations)
            for lease in (8,16):
                cov, ls, vs = replay(x, lease=lease)
                outputs[f'固定{lease}步'] = summary(cov, base, ls, vs)
            for eps in TOLERANCES:
                cov, ls, vs = replay(x, lease=16, reference=ref, tolerance=eps)
                outputs[f'未来已知，容许{100*eps:g}个百分点'] = summary(cov, base, ls, vs)
            for name, s in outputs.items():
                per_doc.setdefault(name, []).append(s)
                totals.setdefault(name, []).append(s)
            del x, source, boundary
        report = {name: combine(rows) for name, rows in per_doc.items()}
        docs.append((doc, report))
        print('DOC', doc, 'status=success', flush=True)
        print('\n'.join(table(report)), flush=True)
    overall = {name: combine(rows) for name, rows in totals.items()}
    lines = ['# EMA未来已知自适应租期离线实验', '', '## 目的', '',
        '以固定4步EMA覆盖率为参照，评估使用未来标签选择4/8/12/16步租期能减少多少刷新；不训练模型，不修改推理系统。', '',
        '## 配置与定义', '',
        '- Qwen2.5-7B-Instruct-1M原摘要验证集四篇，所有28层独立；稠密生成轨迹离线回放，未使用训练集和test。',
        '- alpha=0.2、块内max/头间max；4096预算，sink64/recent64，最多248历史块。头平均、层/生成步等权覆盖率，包含生成历史及部分块。',
        '- 固定4/8/16各沿自身连续EMA状态运行，仅在刷新边界使用前一步实际可见集合的稀疏归一化反馈更新，未观测块保持旧值。预填充末64个Q初始化。',
        '- 自适应也维护自身EMA状态。每个租期开始时选择一次集合，离线评估持有该集合未来4/8/12/16步的覆盖率；选平均覆盖率相对同区间固定4步下降不超过门槛的最长租期。尾段按实际剩余步数截断。',
        '- 这是知道未来的贪心理想调度，不是全局最优上界：各次决策会改变后续EMA状态，不免费借用固定4步的中间更新。',
        '- 如果所有候选都违反门槛，明确记录不可行事件并执行最短4步；不隐藏该事件，不将有违约的策略宣称为严格满足局部门槛。',
        '- 门槛约束每个选中租期的平均覆盖率，不约束每个单步；另报单步下降超过1个百分点比例及最大单步下降。',
        '- 刷新次数=租期数减1，排除共同初始化且不计算输出结束后无用刷新；平均租期=总生成步/租期数。刷新次数减少不等于吞吐等比例增加。',
        '- 本实验只判断摘要轨迹上的覆盖率/刷新权衡，任务质量和吞吐仍是后续独立验收项。',
        f'- 数据：`{args.data_root}`；protocol SHA256：`{file_sha(args.data_root/"protocol.json")}`。',
        f'- 脚本SHA256：`{file_sha(Path(__file__))}`。数据加载/数值检查失败直接退出。', '',
        '## 验证', '', '- 手工标签检查最长可行租期和不可行情形；每篇第0层限制只选4步时与固定4步覆盖率及租期逐项完全相同；全部覆盖率有限且在[0,1]内。', '',
        '## 命令', '', '```bash', '.venv/bin/python -u -m scripts.data.analyze_oracle_lease '+shlex.join(sys.argv[1:]), '```', '',
        '## 总体结果', ''] + table(overall)
    lines += ['', '## 租期分布与绝对计数', '', '```json', json.dumps(overall, ensure_ascii=False, indent=2), '```']
    for doc, results in docs:
        lines += ['', f'## {doc}', ''] + table(results)
    lines += ['', f'耗时{time.monotonic()-started:.1f}秒；全部4篇status=success；逐样本数据仅驻内存，实验目录只保存Markdown。']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print('OVERALL\n'+'\n'.join(table(overall)), flush=True)


if __name__ == '__main__':
    main()
