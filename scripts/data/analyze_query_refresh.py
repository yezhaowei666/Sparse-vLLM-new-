"""Compare causal Query drift signals against one-step EMA refresh benefit."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scripts.data.evaluate_future_ema import feedback_data, initialize_ema, choose, observe, update_ema
from scripts.data.train_future_predictor import load_ids, file_sha


def ranks(x):
    values, order = x.sort()
    _, inverse, counts = values.unique_consecutive(return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    mid = ends - (counts.double() - 1) / 2
    out = torch.empty_like(x, dtype=torch.float64)
    out[order] = mid[inverse]
    return out


def stats(rows, column):
    score, gain = rows[:, column], rows[:, 2]
    a, b = ranks(score), ranks(gain)
    corr = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
    label = gain > .01
    pos, neg = int(label.sum()), int((~label).sum())
    auc = ((a[label].sum() - pos * (pos + 1) / 2) / (pos * neg)).item() if pos and neg else None
    k = max(1, len(score) // 4)
    picked = score.argsort(descending=True)[:k]
    return dict(n=len(score), event_rate=label.double().mean().item(), spearman=corr, auc=auc,
                top25_recall=(label[picked].sum() / pos).item() if pos else None,
                top25_gain=gain[picked].mean().item(), mean_gain=gain.mean().item())


@torch.inference_mode()
def current_step(args):
    """Paired one-step interventions on one fixed-16 EMA trajectory."""
    from scripts.data.train_future_predictor import sample_tensor
    started = time.monotonic()
    protocol = json.loads((args.data_root / 'protocol.json').read_text())
    signals = ('层均值Q余弦/租期起点', '逐头余弦均值/租期起点',
               'KV组最小相似度/前一步', 'KV组最小相似度/租期起点')
    all_rows, docs = [], []
    for doc in load_ids(args.data_root, 'val'):
        record = json.loads((args.data_root / 'records' / f'{doc}.json').read_text())
        doc_rows = []
        for layer in range(protocol['layers']):
            name = f'layer_{layer:02d}.pt'
            source = torch.load(args.data_root / 'data' / doc / name, map_location='cpu', weights_only=True)
            boundary = torch.load(args.data_root / 'boundaries_v1/data' / doc / name, map_location='cpu', weights_only=True)
            if boundary['source_sha256'] != record['files'][name]['sha256']:
                raise ValueError('Boundary source mismatch')
            x = feedback_data(source, boundary, 'cuda')
            q, kv = x['q'], source['decode_keys'].shape[0]
            steps = q.shape[1] - 64
            batch = sample_tensor(x, list(range(steps)), horizon=1)
            mass = batch['mass'][:, 0].mean(1)
            protected = batch['protected'][:, 0].mean(1)
            order = mass.masked_fill(~batch['valid'], -torch.inf).argsort(descending=True, stable=True)[:, :248]
            oracle = mass.gather(1, order).sum(1) + protected
            state, seen = initialize_ema(x)
            hot, anchor = choose(state, x, 0, 248), 0
            rows = []
            for t in range(steps):
                if t:
                    score, observed, _ = observe(x, hot, anchor, 63+t)
                    trial, trial_seen = update_ema(state, seen, score, observed)
                    new_hot = choose(trial, x, t, 248)
                    if t % 16 == 0:
                        state, seen, hot, anchor = trial, trial_seen, new_hot, t
                if t == anchor:
                    continue  # Refresh is mandatory at the reference trajectory's boundaries.
                row, ref = 64+t, 64+anchor
                _, _, held = observe(x, hot, anchor, row)
                _, _, refreshed = observe(x, new_hot, t, row)
                # Independent block-sum coverage cross-check of the token-boundary observer.
                direct = mass[t, new_hot].sum() + protected[t]
                torch.testing.assert_close(refreshed, direct, atol=2e-6, rtol=0)
                if oracle[t] + 2e-6 < max(held, refreshed):
                    raise ValueError('Shared-block oracle below feasible selection')
                cos_anchor = F.cosine_similarity(q[:, row], q[:, ref], dim=-1)
                cos_previous = F.cosine_similarity(q[:, row], q[:, row-1], dim=-1)
                similarity = torch.stack((
                    F.cosine_similarity(q[:, row].mean(0), q[:, ref].mean(0), dim=0),
                    cos_anchor.mean(), cos_previous.reshape(kv, -1).mean(-1).min(),
                    cos_anchor.reshape(kv, -1).mean(-1).min()))
                rows.append(torch.cat((1-similarity, torch.stack((held, refreshed, oracle[t], held.new_tensor(t-anchor))))))
            values = torch.stack(rows).cpu().double()
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite current-step diagnostic')
            doc_rows.append(values)
            del x, source, boundary, batch, mass, q
        values = torch.cat(doc_rows)
        all_rows.append(values)
        docs.append((doc, len(values), values[:, 4:7].mean(0).tolist()))
        print('DOC', docs[-1], 'status=success', flush=True)
    values = torch.cat(all_rows)
    lines = ['# 当前Q判断当前步：检测信号与刷新能力', '', '## 目的和配置', '',
        '- 原摘要val四篇、28层，4096预算、sink/recent各64、248历史块，alpha=0.2；固定16步EMA参考轨迹。只统计租期内第1至15个偏移步，跳过必刷的第0步。',
        '- 不同信号和刷新操作使用完全相同的层/步样本、当前稠密Q及历史KV、生成历史和部分块边界；覆盖率含sink/recent。',
        '- 当前Q保留RoPE，在当前注意力之前已可计算；信号不读当前注意力标签或未来步。租期起点Q指本次集合开始使用的那一步Q。',
        '- 层级信号：先对28个Q头的向量取均值，再与租期起点的均值Q计算余弦（RefreshKV式）。另列逐头余弦再取均值的旧聚合方式。',
        '- 组级信号：按共享KV的4组，每组7个Q头余弦取平均；任一组低于阈值就报警，以4组最小相似度代表。分别比较前一步Q（FreeKV式）及租期起点Q。',
        '- 原论文的组级纠错会更换对应组KV；本实验仍保持LeaseSparse整层共享248块，只借鉴分组检测，不称为完整复现FreeKV。',
        '- EMA刷新：使用前一步、当前稀疏集合实际可见的反馈，做一次EMA更新并重新选块，然后评价当前步；没有使用当前完整注意力更新EMA。',
        '- 全历史最优：直接用当前完整真实注意力概率，按跨头平均块质量选择当前允许的248块；它是当前步共享块的覆盖率上限，不是可部署的免费刷新操作。',
        '- 全历史最优与EMA也存在max/mean排序及累计/当前评分差别，其差距不能全部归因于只观察缓存内。',
        '- 两种反事实刷新只用于本步评价，不改变固定16步参考轨迹的后续状态；本实验隔离检测与动作效果，不测闭环自适应质量或吞吐。',
        '- 事件定义为刷新比继续复用提高超过1个百分点。分别用EMA收益、全历史最优收益定义事件；AUROC=0.5是随机区分，Top25召回是信号最大25%捕获的事件比例。',
        '- 固定阈值0.8/0.85/0.9/0.95仅作描述性扫描，未用test调参。报警后覆盖率为只替换该步选择的反事实平均，不表示闭环运行结果。',
        '- 校验：EMA刷新覆盖率逐步与独立块概率求和一致（容差2e-6），最优覆盖不低于两种合法集合，所有数值有限。',
        f'- 数据`{args.data_root}`；protocol SHA256 `{file_sha(args.data_root/"protocol.json")}`；脚本SHA256 `{file_sha(Path(__file__))}`。', '',
        '## 命令', '', '```bash', '.venv/bin/python -u -m scripts.data.analyze_query_refresh ' + ' '.join(sys.argv[1:]), '```', '',
        f'## 整体覆盖率（{len(values)}个层/步）', '', '| 继续复用 | EMA刷新 | 全历史最优 |', '|---:|---:|---:|',
        '| '+' | '.join(f'{v*100:.4f}%' for v in values[:,4:7].mean(0).tolist())+' |', '',
        '## 信号区分能力', '', '| 范围 | 信号 | 收益标签 | 事件比例 | AUROC | Top25事件召回 | Spearman |', '|---|---|---|---:|---:|---:|---:|']
    groups = [('总体', values)] + [(f'偏移{age}', values[values[:,7]==age]) for age in (1,4,8,15)]
    for scope, group in groups:
        for col, name in enumerate(signals):
            for target, label in ((5, 'EMA'), (6, '全历史最优')):
                s = stats(torch.stack((group[:,col], group[:,col], group[:,target]-group[:,4]), 1), 0)
                lines.append(f'| {scope} | {name} | {label} | {s["event_rate"]:.2%} | {s["auc"]:.4f} | {s["top25_recall"]:.2%} | {s["spearman"]:.4f} |')
    lines += ['', '## 固定阈值扫描', '', '| 信号 | 相似度阈值 | 报警比例 | EMA事件召回 | 最优事件召回 | 报警后EMA覆盖率 | 报警后最优覆盖率 |', '|---|---:|---:|---:|---:|---:|---:|']
    for col, name in enumerate(signals):
        for threshold in (.8,.85,.9,.95):
            hit = values[:,col] > 1-threshold
            recalls = [((hit & (values[:,j]-values[:,4]>.01)).sum()/(values[:,j]-values[:,4]>.01).sum()).item() for j in (5,6)]
            coverages = [torch.where(hit, values[:,j], values[:,4]).mean().item() for j in (5,6)]
            lines.append(f'| {name} | {threshold} | {hit.double().mean():.2%} | {recalls[0]:.2%} | {recalls[1]:.2%} | {coverages[0]:.4%} | {coverages[1]:.4%} |')
    lines += ['', '## 逐文档覆盖率', '', '| 文档 | 层/步 | 复用 | EMA刷新 | 全历史最优 |', '|---|---:|---:|---:|---:|']
    for doc, n, cov in docs:
        lines.append(f'| {doc} | {n} | '+ ' | '.join(f'{v:.4%}' for v in cov)+' |')
    lines += ['', '## 状态', '', f'全部4篇status=success；耗时{time.monotonic()-started:.1f}秒。只保存Markdown，原始逐步表留在内存后释放。', '',
        '论文依据：[RefreshKV正文2.2节](https://aclanthology.org/2025.acl-long.1211.pdf)、[FreeKV正文3.3节](https://arxiv.org/html/2505.13109v5#S3.S3)。']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print('\n'.join(line for line in lines if line.startswith('| 总体') or line.startswith('| 继续') or line.startswith('| '+signals[0]+' |')), flush=True)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--model-config', type=Path)
    ap.add_argument('--mode', choices=('next', 'current'), default='next')
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.mode == 'current':
        return current_step(args)
    cfg = json.loads(args.model_config.read_text())
    if cfg['rope_scaling'] is not None:
        raise ValueError('This diagnostic requires unscaled RoPE')
    if args.report.exists():
        raise FileExistsError(args.report)
    start = time.monotonic()
    all_rows, docs = [], []
    for doc in load_ids(args.data_root, 'val'):
        record = json.loads((args.data_root / 'records' / f'{doc}.json').read_text())
        doc_rows = []
        for layer in range(cfg['num_hidden_layers']):
            name = f'layer_{layer:02d}.pt'
            source = torch.load(args.data_root / 'data' / doc / name, weights_only=True, map_location='cpu')
            boundary = torch.load(args.data_root / 'boundaries_v1/data' / doc / name, weights_only=True, map_location='cpu')
            if boundary['source_sha256'] != record['files'][name]['sha256']:
                raise ValueError('Boundary source mismatch')
            x = feedback_data(source, boundary, 'cuda')
            q = x['q']
            dim = q.shape[-1]
            inv = cfg['rope_theta'] ** (-torch.arange(0, dim, 2, device='cuda').float() / dim)
            phase = source['row_positions'].cuda().float()[:, None] * inv
            phase = torch.cat((phase, phase), -1)[None]
            rotated = torch.cat((-q[..., dim//2:], q[..., :dim//2]), -1)
            content = q * phase.cos() - rotated * phase.sin()
            # Independent rotation round-trip and norm invariants.
            cr = torch.cat((-content[..., dim//2:], content[..., :dim//2]), -1)
            torch.testing.assert_close(content * phase.cos() + cr * phase.sin(), q, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(content.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
            state, seen = initialize_ema(x)
            hot, lease_anchor, ref = choose(state, x, 0, 248), 0, 63
            rows = []
            for t in range(q.shape[1] - 65):
                row = 64 + t
                feedback, observed, _ = observe(x, hot, lease_anchor, row)
                trial, trial_seen = update_ema(state, seen, feedback, observed)
                new_hot = choose(trial, x, t + 1, 248)
                _, _, held_cov = observe(x, hot, lease_anchor, row + 1)
                _, _, new_cov = observe(x, new_hot, t + 1, row + 1)
                drift = 1 - F.cosine_similarity(q[:, row], q[:, ref], dim=-1).mean()
                content_drift = 1 - F.cosine_similarity(content[:, row], content[:, ref], dim=-1).mean()
                rows.append(torch.stack((drift, content_drift, new_cov-held_cov,
                                         drift.new_tensor(t + 1 - lease_anchor))))
                # The hypothetical refresh never changes the fixed-16 reference trajectory.
                if (t + 1) % 16 == 0:
                    state, seen, hot, lease_anchor, ref = trial, trial_seen, new_hot, t + 1, row
            values = torch.stack(rows).cpu().double()
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite diagnostic values')
            doc_rows.append(values)
            del source, boundary, x, q, content
        values = torch.cat(doc_rows)
        all_rows.append(values)
        docs.append((doc, stats(values, 0), stats(values, 1)))
        print('DOC', doc, docs[-1][1:], flush=True)
    values = torch.cat(all_rows)
    lines = ['# Query变化与EMA刷新收益离线比较', '',
        '## 目的与口径', '',
        '- 仅使用原摘要数据val四篇、全部28层；不训练、不读取test、不生成新轨迹。',
        '- 固定16步EMA轨迹，alpha=0.2，4096预算，sink64/recent64，248历史块；只有第16步更新EMA并切换集合。',
        '- 每步完成后，用当前实际可见块反馈假设执行一次EMA更新和选块；比较下一步刷新集合与继续持有集合的完整注意力覆盖率。假设刷新不改变参考轨迹。',
        '- 包含生成历史、部分块和移动recent；复用时保持旧历史边界，刷新使用新边界。覆盖率含sink+recent。',
        '- 信号=1减逐头余弦相似度的头均值；比较当前Q与本租期刷新反馈Q，首租期参考提示词最后Q。',
        '- 两种信号：实际post-RoPE Q；按真实位置逆旋转的内容Q。FP32计算，通过旋转往返及范数一致性检查。',
        '- 标签是下一步刷新覆盖率减复用覆盖率；收益超过1个百分点定义为需要刷新事件。标签只用于评价，信号不看未来。',
        '- Spearman是信号与收益的秩相关；AUROC衡量事件区分，0.5为随机。Top25召回为信号最高25%样本覆盖的事件比例，不是已部署阈值。',
        '- 分租期年龄1/4/8/16单独统计，排除信号仅随位置距离增长的混淆。样本是相关的层/步观测，不作为独立统计重复。',
        '- 这是稠密生成轨迹上的离线稀疏反馈回放，不是任务质量或端到端吞吐测量；当前结果限于摘要验证数据。',
        f'- 数据：`{args.data_root}`；protocol SHA256：`{file_sha(args.data_root / "protocol.json")}`。',
        f'- 模型配置：`{args.model_config}`；脚本SHA256：`{file_sha(Path(__file__))}`。', '',
        '## 命令', '', '```bash', '.venv/bin/python -u -m scripts.data.analyze_query_refresh ' + ' '.join(sys.argv[1:]), '```', '',
        '## 结果', '', '| 范围 | 信号 | 样本数 | 事件比例 | Spearman | AUROC | Top25事件召回 | Top25平均收益(百分点) |', '|---|---|---:|---:|---:|---:|---:|---:|']
    groups = [('总体', values)] + [(f'年龄{a}', values[values[:, 3] == a]) for a in (1, 4, 8, 16)]
    for name, group in groups:
        for col, signal in enumerate(('保留RoPE', '去除RoPE')):
            s = stats(group, col)
            lines.append(f'| {name} | {signal} | {s["n"]} | {s["event_rate"]:.4%} | {s["spearman"]:.4f} | {s["auc"]} | {s["top25_recall"]} | {s["top25_gain"]*100:.4f} |')
    lines += ['', '## 逐文档结果', '', '```json', json.dumps(docs, ensure_ascii=False), '```', '', f'耗时：{time.monotonic()-start:.1f}秒。全部样本status=success。']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[-16:-5]), flush=True)


if __name__ == '__main__':
    main()
