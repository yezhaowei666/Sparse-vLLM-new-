"""Frozen future predictor vs native EMA on natural LongBench dense trajectories.

Only a Markdown report is persisted; attention tensors live for one document.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from scripts.data.collect_future_attention import context_hash, pool
from scripts.data.evaluate_future_ema import feedback_data, initialize_ema, choose, observe, update_ema
from scripts.data.train_future_predictor import build_predictor, predict_queries, file_sha, sample_tensor, predict_scores, scored_predictions, selection_probabilities


def true_query_scores(x, source, boundary, batch, anchor, horizon, bounds):
    """Score frozen historical blocks with actual future Q, not predicted vectors."""
    q = x['q'][:, 64+anchor:64+anchor+horizon].permute(1, 0, 2)
    raw = torch.cat((boundary['prompt_tail_keys'], source['decode_keys']), 1).to(q)
    start = x['tail_start'] // 16
    minimum, maximum = [v.to(q).clone() for v in bounds]
    pad = (-raw.shape[1]) % 16
    shape = (raw.shape[0], -1, 16, raw.shape[-1])
    lo = F.pad(raw, (0,0,0,pad), value=torch.inf).reshape(shape).amin(2)
    hi = F.pad(raw, (0,0,0,pad), value=-torch.inf).reshape(shape).amax(2)
    minimum = torch.cat((minimum[:, :start], lo), 1)
    maximum = torch.cat((maximum[:, :start], hi), 1)
    mean = x['key'].clone()
    last = int(batch['partial_index'][0])
    count = int(batch['partial_count'][0])
    cutoff = x['prompt_tokens'] + anchor - 64
    if count != 16:
        part = raw[:, last*16-x['tail_start']:cutoff-x['tail_start']]
        minimum[:, last] = part.amin(1)
        maximum[:, last] = part.amax(1)
    mean[:, last] = batch['partial_key'][0]
    repeat = q.shape[1] // mean.shape[0]
    mean, minimum, maximum = [v.repeat_interleave(repeat, 0) for v in (mean, minimum, maximum)]
    scores = [torch.einsum('thd,hnd->thn', q, mean),
              torch.einsum('thd,hnd->thn', q, maximum),
              torch.einsum('thd,hnd->thn', q.clamp_min(0), maximum)
              + torch.einsum('thd,hnd->thn', q.clamp_max(0), minimum)]
    scores = torch.stack(scores) * q.shape[-1] ** -.5
    scores[..., last] += torch.tensor(count/16, device=q.device).log()
    return scores.masked_fill(~batch['valid'][0], -torch.inf)


@torch.inference_mode()
def forecast(model, batch, budget=248):
    scores, mass_logits = scored_predictions(model, batch)
    # Always use every forecast, without reading the future EOS or labels.
    ranking = selection_probabilities(scores, mass_logits).masked_fill(~batch['valid'], -torch.inf)
    count = min(budget, int(batch['valid'][0].sum()))
    return ranking[0].argsort(descending=True, stable=True)[:count]


@torch.inference_mode()
def replay(model, source, boundary, device='cuda', budget=248, key_bounds=None, layer=None):
    x = feedback_data(source, boundary, device)
    x['layer'] = layer
    steps = x['q'].shape[1] - 64
    if steps <= 0:
        raise ValueError('No decode query to evaluate')
    values = torch.empty((steps, 7 if key_bounds is not None else 4), device=device)
    state1, seen1 = initialize_ema(x)
    state4, seen4 = state1.clone(), seen1.clone()
    hot4 = choose(state4, x, 0, budget)
    previous = 0
    for anchor in range(0, steps, 4):
        if anchor:
            score, observed, _ = observe(x, hot4, previous, 63 + anchor)
            state4, seen4 = update_ema(state4, seen4, score, observed)
        hot4 = choose(state4, x, anchor, budget)
        previous = anchor
        horizon = min(4, steps - anchor)
        batch = sample_tensor(x, [anchor], horizon=horizon, history=model.history)
        mass = batch['mass'][0].mean(1)
        protected = batch['protected'][0].mean(1)
        hot_oracle = mass.mean(0).masked_fill(~batch['valid'][0], -torch.inf).argsort(descending=True, stable=True)[:len(hot4)]
        values[anchor:anchor+horizon, 1] = mass[:, hot4].sum(-1) + protected
        if model.horizon == 4:
            hot_pred = forecast(model, batch, budget)
            values[anchor:anchor+horizon, 2] = mass[:, hot_pred].sum(-1) + protected
        values[anchor:anchor+horizon, 3] = mass[:, hot_oracle].sum(-1) + protected
        if key_bounds is not None:
            scores = true_query_scores(x, source, boundary, batch, anchor, horizon, key_bounds)
            ranks = scores.softmax(-1).mean((1, 2)).masked_fill(~batch['valid'][0], -torch.inf)
            for column, ranking in enumerate(ranks, 4):
                hot = ranking.argsort(descending=True, stable=True)[:len(hot4)]
                values[anchor:anchor+horizon, column] = mass[:, hot].sum(-1) + protected
            if (values[anchor:anchor+horizon, 4:].mean(0) > values[anchor:anchor+horizon, 3].mean() + 2e-6).any():
                raise ValueError('True-Q shared selection exceeds oracle')
        for step in range(anchor, anchor + horizon):
            hot1 = choose(state1, x, step, budget)
            score, observed, cov = observe(x, hot1, step, 64 + step)
            values[step, 0] = cov
            state1, seen1 = update_ema(state1, seen1, score, observed)
            if model.horizon == 1:
                single = sample_tensor(x, [step], horizon=1, history=model.history)
                chosen = forecast(model, single, budget)
                actual = single['mass'][0,0].mean(0)
                base = single['protected'][0,0].mean()
                optimal = actual.masked_fill(~single['valid'][0], -torch.inf).argsort(descending=True, stable=True)[:len(chosen)]
                values[step,2] = actual[chosen].sum() + base
                values[step,3] = actual[optimal].sum() + base
        if (values[anchor:anchor+horizon, 1:3].mean(0) > values[anchor:anchor+horizon, 3].mean() + 2e-6).any():
            raise ValueError('Shared selection exceeds matched oracle')
    if not torch.isfinite(values).all() or values.min() < -1e-6 or values.max() > 1 + 1e-5:
        raise ValueError('Invalid coverage')
    worst = torch.stack([values[a:a+4].amin(0) for a in range(0, steps, 4)]).mean(0)
    return torch.stack((values.mean(0), worst), -1).cpu()


def validate_cohort(args):
    rows = json.loads(args.manifest.read_text())
    if len(rows) != 48 or sorted(Counter(r['category'] for r in rows).values()) != [8] * 6:
        raise ValueError('Expected original 48-document, six-category cohort')
    training = {json.loads(p.read_text())['context_sha256'] for p in (args.training_data / 'records').glob('*.json')
                if json.loads(p.read_text())['status'] == 'success'}
    contexts = set()
    task_data = {task: [json.loads(line) for line in (args.longbench / f'{task}.jsonl').read_text().splitlines()]
                 for task in {r['task'] for r in rows}}
    for row in rows:
        original = task_data[row['task']][row['index']]
        raw_hash = hashlib.sha256(original['context'].encode()).hexdigest()
        record_hash = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        norm_hash = context_hash(original['context'])
        ids_hash = hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()
        if raw_hash != row['context_sha256'] or record_hash != row['record_sha256'] or ids_hash != row['input_ids_sha256']:
            raise ValueError(f'LongBench source or prompt checksum differs: {row["id"]}')
        if norm_hash in training or norm_hash in contexts:
            raise ValueError(f'Duplicate evaluation/training context: {row["id"]}')
        contexts.add(norm_hash)
    return rows


def report(args, rows, results, metadata):
    successes = [r for r in results if r['status'] == 'success']
    history, horizon = metadata['history'], metadata['horizon']
    lines = ['# 冻结未来预测器LongBench注意力覆盖率', '', '## 目的与固定配置', '',
             '冻结指定检查点预测器，在旧实验48篇真实LongBench提示上检验跨任务选块覆盖率，与原生普通EMA同轨迹对照。不计算LongBench答案分数。',
             f'- 模型：`{args.model}`；BF16、SDPA、prefill按4096分块；greedy、自然EOS、max_new_tokens={args.max_new_tokens}。',
             '- 沿用旧48篇验证清单，六类各8篇，共16个任务；原输入Token完全不变、不截断、不改写为摘要任务。',
             '- 原始LongBench行、原文、input_ids哈希核对通过；按与训练数据相同的去空白小写哈希核对，48篇与预测器数据目录中的全部成功样本无重复。该清单是历史实验用过的集合，不称为全新的最终测试集。',
             '- 固定4096总预算、sink/recent各64、block16、最多248个中间块，28层独立，无跨层共享。生成历史与部分块按实际Token处理。',
             '- 普通EMA alpha=.2、块内和头间max、最后64个prefill Q初始化；仅更新自己观测的块，未观测保留；反馈按稀疏可见质量重新归一化。EMA4只在刷新时更新，下一步使用新集合。',
             f'- 预测器历史{history}个Q、输出未来{horizon}步，不读取未来Q/标签或EOS；每{horizon}步刷新。所有方法共用同一稠密Q/K和自然生成轨迹。',
             f'- 从第一个decode query起连续回放，预测器刷新起点0、{horizon}、{2*horizon}…；所有decode步各计一次。末尾不足预测长度也保留，评分只取实际步，选块仍平均全部{horizon}个输出。自然短回答不剔除，零decode记为skipped_by_policy。',
             '- 覆盖率包含sink/recent，分母是完整可见注意力。每文档先对全部层/实际decode步平均，再类别内文档等权、六类等权（沿用旧实验汇总方式）。周期最差步为每个连续4步观察窗口最低值，再窗口/层/文档/类别平均。',
             f'- oracle在每个实际{horizon}步租期内选一组共享块；EOS尾部按实际未来步计算。周期最差步统一按4步窗口统计，便于跨版本对照；不是单步版本自身的租期最差步。',
             '- 这是稠密参考轨迹上的因果选块覆盖率实验，不是各稀疏方法自由生成的隐藏状态传播测量，也不是引擎吞吐测试。',
             '- 只写本Markdown；Q/K、注意力张量、答案Token只保留在内存，逐文档释放。', '', '## 溯源', '',
             *[f'- {k}: `{v}`' for k, v in metadata.items()], '', '## 命令', '', '```bash',
             f'.venv/bin/python -u -m scripts.data.evaluate_longbench_future --model {args.model} --manifest {args.manifest} --longbench {args.longbench} --training-data {args.training_data} --checkpoint {args.checkpoint} --report {args.report} --max-new-tokens {args.max_new_tokens}' + (' --key-summary-probe' if getattr(args, 'key_summary_probe', False) else ''),
             '```', '', f'## 进度：{len(results)}/{len(rows)}，成功{len(successes)}', '']
    groups = defaultdict(list)
    for r in successes:
        groups[r['category']].append(torch.tensor(r['metrics'], dtype=torch.float64))
    names = ['普通EMA每步', '普通EMA4步', f'预测器{horizon}步', '共享集合oracle' if horizon==4 else '逐步oracle']
    if getattr(args, 'key_summary_probe', False):
        names += ['真实Q+Key均值', '真实Q+Key逐维最大值', '真实Q+Key正负上下界']
        lines += ['真实未来Query诊断：均值、逐维最大值、按Query正负选择Key最大/最小值的上下界三种评分；'
                  '各头先对候选块Softmax，再跨头和实际未来步平均，固定选248块。部分块统一加log(实际数量/16)。'
                  '全部仅用刷新时可见历史Key，真实未来Query仅作为诊断输入。', '']
    if groups:
        category_means = {c: torch.stack(v).mean(0) for c,v in groups.items()}
        overall = torch.stack(list(category_means.values())).mean(0)
        lines += ['| 方法 | 平均覆盖率 | 周期最差步覆盖率 |', '|---|---:|---:|']
        for name, pair in zip(names, overall):
            lines.append(f'| {name} | {100*pair[0]:.6f}% | {100*pair[1]:.6f}% |')
        lines += ['', '| 类别 | 文档数 | ' + ' | '.join(names) + ' | 预测器−EMA4/百分点 |', '|---|---:|' + '---:|'*(len(names)+1)]
        for c, vals in category_means.items():
            lines.append(f'| {c} | {len(groups[c])} | ' + ' | '.join(f'{100*v[0]:.6f}%' for v in vals) + f' | {100*(vals[2,0]-vals[1,0]):+.6f} |')
        tasks = defaultdict(list)
        for r in successes:
            tasks[r['task']].append(torch.tensor(r['metrics']))
        lines += ['', '| 任务 | 文档数 | ' + ' | '.join(names) + ' |', '|---|---:|' + '---:|'*len(names)]
        for task, vals in tasks.items():
            means = torch.stack(vals).mean(0)
            lines.append(f'| {task} | {len(vals)} | ' + ' | '.join(f'{100*v[0]:.6f}%' for v in means) + ' |')
    lines += ['', '## 逐文档记录', '', '| 文档 | 状态 | 输入/输出/decode | ' + ' | '.join(names) + ' | 秒数 |', '|---|---|---|' + '---:|'*(len(names)+1)]
    for r in results:
        scores = ' | '.join(f'{100*v[0]:.6f}%' for v in r['metrics']) if r['status'] == 'success' else ' | '.join(['—'] * len(names))
        lines.append(f'| {r["id"]} | {r["status"]} | {r["input"]}/{r.get("output",0)}/{r.get("decode",0)} | {scores} | {r["seconds"]:.1f} |')
        if 'error' in r:
            lines.append(f'\n失败：`{r["error"]}`\n')
    lines += ['', '## 生成与数值核查', '', '| 文档 | 结束原因 | 输出Token SHA256 | 最大行概率误差 |', '|---|---|---|---:|']
    for r in successes:
        lines.append(f'| {r["id"]} | {r["ended_by"]} | {r["tokens_sha256"]} | {r["row_error"]:.8f} |')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.max_new_tokens < 2:
        raise ValueError('Need at least two output tokens to observe a decode query')
    if args.report.exists():
        raise FileExistsError(args.report)
    rows = validate_cohort(args)
    protocol = json.loads((args.training_data/'protocol.json').read_text())
    if file_sha(args.model/'config.json') != protocol['model_config_sha256']:
        raise ValueError('Teacher model config differs from training')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt['config']['source_protocol_sha256'] != file_sha(args.training_data/'protocol.json'):
        raise ValueError('Predictor provenance mismatch')
    predictor = build_predictor(ckpt['config']).cuda().eval()
    if args.key_summary_probe and predictor.horizon != 4:
        raise ValueError('Key summary diagnostic requires four-step comparison')
    predictor.load_state_dict(ckpt['model'])
    metadata = dict(checkpoint=str(args.checkpoint), epoch=ckpt['epoch'], architecture=ckpt['config'].get('architecture', 'tcn'), history=predictor.history, horizon=predictor.horizon, checkpoint_sha256=file_sha(args.checkpoint),
                    manifest_sha256=file_sha(args.manifest), script_sha256=file_sha(Path(__file__)),
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), model_config_sha256=protocol['model_config_sha256'])
    if metadata['architecture'] == 'mass':
        metadata.update(selection='mean over all predicted steps and heads of sigmoid(mass_logit) * conditional_block_probability',
                        checkpoint_selection=ckpt['config']['checkpoint_selection'],
                        predictor_script_sha256=file_sha(Path(__file__).with_name('train_future_predictor.py')))
    results = []
    report(args, rows, results, metadata)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    state = {}

    def capture(module, q, k, v, mask, **kwargs):
        result = original(module, q, k, v, mask, **kwargs)
        layer = module.layer_idx
        n = k.shape[-2]
        count = min(q.shape[-2], max(0, n - (state['prompt'] - 64)))
        if count:
            maximum, mass = pool(q, k, count)
            state['maxima'][layer].append(maximum.cpu())
            state['sums'][layer].append(mass.cpu())
            state['queries'][layer].append(q[0, :, -count:].cpu())
        if n == state['prompt']:
            means = F.pad(k[0].float(), (0,0,0,(-n)%16)).reshape(4,-1,16,128).sum(2)
            counts = torch.full((means.shape[1],),16,device=k.device)
            counts[-1] = (n-1)%16+1
            state['means'][layer] = (means/counts[None,:,None]).to(torch.bfloat16).cpu()
            state['tail'][layer] = k[0,:,((n-64)//16)*16:].cpu()
            if args.key_summary_probe:
                shape=(k.shape[1],-1,16,k.shape[-1])
                state['bounds'][layer] = (
                    F.pad(k[0],(0,0,0,(-n)%16),value=torch.inf).reshape(shape).amin(2).cpu(),
                    F.pad(k[0],(0,0,0,(-n)%16),value=-torch.inf).reshape(shape).amax(2).cpu())
        if n > state['prompt']:
            state['keys'][layer].append(k[0,:,-1:].cpu())
        return result

    ALL_ATTENTION_FUNCTIONS['sdpa'] = capture
    try:
        for index, row in enumerate(rows,1):
            started = time.perf_counter()
            rec = dict(id=row['id'],task=row['task'],category=row['category'],input=row['input_tokens'],status='model_failed')
            print(f'START {index}/48 {row["id"]} input={row["input_tokens"]}',flush=True)
            state.clear()
            state.update(prompt=row['input_tokens'],means=[None]*28,tail=[None]*28,bounds=[None]*28,
                         **{key:[[] for _ in range(28)] for key in ('maxima','sums','queries','keys')})
            try:
                ids = torch.tensor([row['input_ids']],device='cuda')
                cache = None
                for offset in range(0,ids.shape[1],4096):
                    result = model(input_ids=ids[:,offset:offset+4096],past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache = result.past_key_values
                token = result.logits[:,-1].argmax(-1,keepdim=True)
                output = [token.item()]
                for _ in range(args.max_new_tokens-1):
                    if output[-1] in eos:
                        break
                    result = model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache = result.past_key_values
                    token = result.logits[:,-1].argmax(-1,keepdim=True)
                    output.append(token.item())
                del result,cache,ids,token
                steps = len(output)-1
                rec.update(output=len(output),decode=steps,ended_by='eos' if output[-1] in eos else 'length_cap',
                           tokens_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest())
                if not steps:
                    rec.update(status='skipped_by_policy',error='No decode query after prefill')
                else:
                    rec['status'] = 'metric_failed'
                    values=[]; max_error=0.
                    for layer in range(28):
                        width=max(t.shape[-1] for t in state['sums'][layer])
                        sums=torch.cat([F.pad(t,(0,width-t.shape[-1])) for t in state['sums'][layer]],1)
                        maxima=torch.cat([F.pad(t,(0,width-t.shape[-1])) for t in state['maxima'][layer]],1)
                        q=torch.cat(state['queries'][layer],1)
                        keys=torch.cat(state['keys'][layer],1)
                        if q.shape != (28,64+steps,128) or keys.shape != (4,steps,128):
                            raise ValueError('Collected trajectory shape mismatch')
                        error=(sums.float().sum(-1)-1).abs().max().item()
                        max_error=max(max_error,error)
                        if error>.001:
                            raise ValueError(f'Dense attention mass mismatch: {error}')
                        source=dict(queries=q,sums=sums,maxima=maxima,prompt_key_mean=state['means'][layer],
                                    decode_keys=keys,prompt_tokens=row['input_tokens'],
                                    row_positions=torch.arange(row['input_tokens']-64,row['input_tokens']+steps))
                        boundary=dict(prompt_tail_keys=state['tail'][layer],tail_start=(row['input_tokens']-64)//16*16)
                        values.append(replay(predictor,source,boundary,key_bounds=state['bounds'][layer],layer=layer))
                    rec.update(status='success',metrics=torch.stack(values).mean(0).tolist(),row_error=max_error)
                    del source,boundary,sums,maxima,q,keys,values
                state.clear()
                rec['seconds']=time.perf_counter()-started
                results.append(rec)
                report(args,rows,results,metadata)
                print('DONE',index,'/48',rec['id'],rec['status'],'decode=',steps,'metrics=',rec.get('metrics'),'seconds=',round(rec['seconds'],1),flush=True)
            except Exception as error:
                rec.update(error=f'{type(error).__name__}: {error}',seconds=time.perf_counter()-started)
                results.append(rec)
                report(args,rows,results,metadata)
                raise
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa']=original
    print('COMPLETE',len(results),'report=',args.report,flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ('model','manifest','longbench','training-data','checkpoint','report'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--key-summary-probe',action='store_true')
    run(p.parse_args())


if __name__=='__main__':
    main()
