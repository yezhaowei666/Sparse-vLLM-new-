"""Collect full-attention Qwen trajectories for multi-step block prediction."""
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def context_hash(text):
    return hashlib.sha256(re.sub(r'\s+', '', text.lower()).encode()).hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Dataset already prepared; use collect to continue it')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    excluded = set()
    for path in args.longbench.glob('*.jsonl'):
        for line in path.open():
            excluded.add(context_hash(json.loads(line)['context']))
    sources = out / 'sources'
    source_rows = []
    for path in sorted(sources.glob('govreport_train_*.json')):
        for r in json.loads(path.read_text())['rows']:
            assert not r['truncated_cells']
            source_rows.append(('govreport', r['row_idx'], r['row']['report']))
    for i, line in enumerate((sources / 'qmsum_train.jsonl').open()):
        r = json.loads(line)
        text = '\n'.join(f"{x['speaker']}: {x['content']}" for x in r['meeting_transcripts'])
        source_rows.append(('qmsum', i, text))
    rng = random.Random(20260922)
    rng.shuffle(source_rows)
    candidates = {'govreport': [], 'qmsum': []}
    seen = set(excluded)
    rejected = Counter()
    for task, index, context in source_rows:
        ch = context_hash(context)
        if ch in seen:
            rejected['duplicate_or_longbench'] += 1
            continue
        seen.add(ch)
        instruction = (
            'Write a detailed, factual summary of the following government report. '
            'Cover the main issues, evidence, findings, recommendations, and limitations. '
            if task == 'govreport' else
            'Write a detailed, factual summary of the following meeting. '
            'Cover the main topics, participants\' positions, reasons, decisions, and action items. '
        )
        prompt = instruction + 'Use 350-450 words, organized into coherent paragraphs. Do not invent details.\n\n' + context + '\n\nDetailed summary:'
        ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, return_dict=False)
        if not 8000 <= len(ids) <= 12000:
            rejected['outside_8000_12000'] += 1
            continue
        candidates[task].append(dict(id=f'{task}_train_{index}', source_task=task,
            source_split='train', source_index=index, context_sha256=ch,
            prompt=prompt, input_ids=ids, prompt_tokens=len(ids)))
    rows = []
    schedule = ['train'] * 6 + ['val', 'test']
    for task, cs in candidates.items():
        assert len(cs) >= 24, (task, len(cs))
        for j, r in enumerate(cs):
            r['split'] = schedule[j % 8]
    # Alternate sources; each source contributes 12/2/2 successful documents.
    for j in range(max(map(len, candidates.values()))):
        for task in candidates:
            if j < len(candidates[task]):
                rows.append(candidates[task][j])
    protocol = dict(version=1, model=str(args.model.resolve()), model_config_sha256=digest(args.model/'config.json'),
        script_sha256=digest(Path(__file__)), seed=20260922, target_per_source={'train':12,'val':2,'test':2},
        min_input_tokens=8000, max_input_tokens=12000, max_new_tokens=512, min_decode_steps=256,
        history=64, block_size=16, layers=28, query_heads=28, kv_heads=4, head_dim=128,
        prefill_chunk=4096, attention='dense SDPA', model_dtype='BF16', qk_product='BF16 inputs FP32 accumulation/output',
        stored_maxima='BF16', stored_sums='FP16', stored_queries='BF16 post-RoPE',
        key_summary='BF16 post-RoPE prompt-block mean; final partial block has explicit count',
        decode_keys='BF16 post-RoPE, one newly appended key per decode step',
        generation='greedy; natural EOS; no min-length suppression; length-capped answers marked',
        max_attempts=120, source_files={p.name:digest(p) for p in sources.iterdir() if p.is_file()},
        excluded_longbench_contexts=len(excluded), candidate_counts={k:len(v) for k,v in candidates.items()},
        preparation_skips=dict(rejected), torch=torch.__version__)
    # Worst case: 575 recorded rows and 782 blocks, all 32 documents at max length.
    t, b = 64+511, math.ceil((12000+511)/16)
    upper = 32*28*(28*t*b*4 + 28*t*128*2 + 4*math.ceil(12000/16)*128*2 + 4*511*128*2)
    protocol['tensor_upper_bytes'] = upper
    if shutil.disk_usage(out).free < upper + 3*2**30:
        raise RuntimeError(f'Insufficient space: need {upper/2**30:.2f} GB plus 3 GB')
    save_json(out/'manifest.json', rows)
    save_json(out/'protocol.json', protocol)
    print('PREPARED', protocol['candidate_counts'], 'upper_GB',upper/2**30, flush=True)


def pool(q, k, count):
    # [1,H,T,D] and [1,KV,N,D]; collect only causal rows requested at the tail.
    q = q[0, :, -count:]
    k = k[0]
    h, _, dim = q.shape
    kh, length, _ = k.shape
    logits = torch.bmm(q.reshape(kh, -1, dim), k.transpose(1,2), out_dtype=torch.float32)
    logits = logits.reshape(h, count, length) * dim**-.5
    mask = torch.arange(length, device=q.device)[None, :] > torch.arange(length-count, length, device=q.device)[:, None]
    probs = logits.masked_fill(mask[None], -torch.inf).softmax(-1)
    blocks = F.pad(probs, (0, (-length)%16)).reshape(h, count, -1, 16)
    return blocks.amax(-1).to(torch.bfloat16), blocks.sum(-1).to(torch.float16)


def check_pool():
    torch.manual_seed(20260922)
    q = torch.randn(1, 4, 7, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 2, 35, 128, device='cuda', dtype=torch.bfloat16)
    mx, mass = pool(q, k, 7)
    logits = q.float() @ k.float().repeat_interleave(2, dim=1).transpose(-1,-2) / 128**.5
    mask = torch.arange(35,device='cuda')[None,:] > torch.arange(28,35,device='cuda')[:,None]
    p = logits.masked_fill(mask,-torch.inf).softmax(-1)[0]
    p = F.pad(p,(0,13)).reshape(4,7,3,16)
    torch.testing.assert_close(mx.float(), p.amax(-1), atol=.002, rtol=.005)
    torch.testing.assert_close(mass.float(), p.sum(-1), atol=.0005, rtol=.001)
    assert (mass.float().sum(-1)-1).abs().max()<.001
    print('CHECK PASS: grouped causal block scores match explicit FP32 reference',flush=True)


@torch.inference_mode()
def collect(args):
    out = args.output
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['script_sha256'] == digest(Path(__file__))
    assert str(args.model.resolve()) == protocol['model']
    rows = json.loads((out/'manifest.json').read_text())
    (out/'records').mkdir(exist_ok=True)
    (out/'data').mkdir(exist_ok=True)
    done = {p.stem:json.loads(p.read_text()) for p in (out/'records').glob('*.json')}
    counts = Counter((r['source_task'],r['split']) for r in done.values() if r['status']=='success')
    targets = protocol['target_per_source']
    check_pool()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos,int) else set(eos)
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    state = {}

    def capture(module, q, k, v, mask, **kwargs):
        result = original(module,q,k,v,mask,**kwargs)
        layer = module.layer_idx
        n = k.shape[-2]
        count = min(q.shape[-2], max(0,n-(state['prompt']-64)))
        if count:
            mx, mass = pool(q,k,count)
            state['maxima'][layer].append(mx.cpu())
            state['sums'][layer].append(mass.cpu())
            state['queries'][layer].append(q[0,:,-count:].cpu())
        if n == state['prompt']:
            pad = (-n)%16
            values = F.pad(k[0].float(),(0,0,0,pad)).reshape(4,-1,16,128).sum(2)
            sizes = torch.full((values.shape[1],),16,device=k.device)
            sizes[-1] = 16-pad
            state['key_mean'][layer] = (values/sizes[None,:,None]).to(torch.bfloat16).cpu()
            if protocol.get('collect_boundaries'):
                state['tail'][layer] = k[0,:,((n-64)//16)*16:].cpu()
        if n > state['prompt']:
            state['decode_keys'][layer].append(k[0,:,-1:].cpu())
        return result

    ALL_ATTENTION_FUNCTIONS['sdpa'] = capture
    accepted = 0
    try:
        for row in rows:
            key = (row['source_task'],row['split'])
            if row['id'] in done or counts[key] >= targets[row['split']]:
                continue
            if len(done) >= protocol['max_attempts']:
                raise RuntimeError('Maximum collection attempts reached before quota')
            start = time.perf_counter()
            state.clear()
            state.update(prompt=row['prompt_tokens'], key_mean=[None]*28, tail=[None]*28,
                **{name:[[] for _ in range(28)] for name in ('maxima','sums','queries','decode_keys')})
            record = {k:v for k,v in row.items() if k not in ('input_ids','prompt')}
            record['status'] = 'model_failed'
            try:
                torch.cuda.reset_peak_memory_stats()
                ids = torch.tensor([row['input_ids']],device='cuda')
                cache = None
                for s in range(0,ids.shape[1],4096):
                    result = model(input_ids=ids[:,s:s+4096],past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache = result.past_key_values
                token = result.logits[:,-1].argmax(-1,keepdim=True)
                output = [token.item()]
                for _ in range(protocol['max_new_tokens'] - 1):
                    if output[-1] in eos:
                        break
                    result = model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache = result.past_key_values
                    token = result.logits[:,-1].argmax(-1,keepdim=True)
                    output.append(token.item())
                steps = len(output)-1
                record.update(generated_ids=output,generated_text=tok.decode(output,skip_special_tokens=True),
                    decode_steps=steps,ended_by='eos' if output[-1] in eos else 'length_cap',
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),generation_seconds=time.perf_counter()-start)
                del cache,result,ids,token
                if steps < protocol['min_decode_steps']:
                    record.update(status='skipped_by_policy',reason=f"fewer_than_{protocol['min_decode_steps']}_decode_steps")
                else:
                    folder = out/'data'/row['id']
                    folder.mkdir()
                    files = {}
                    max_error = 0.
                    for layer in range(28):
                        width = max(x.shape[-1] for x in state['maxima'][layer])
                        mx = torch.cat([F.pad(x,(0,width-x.shape[-1])) for x in state['maxima'][layer]],1)
                        mass = torch.cat([F.pad(x,(0,width-x.shape[-1])) for x in state['sums'][layer]],1)
                        queries = torch.cat(state['queries'][layer],1)
                        keys = torch.cat(state['decode_keys'][layer],1)
                        assert mx.shape == mass.shape == (28,64+steps,width)
                        assert queries.shape == (28,64+steps,128) and keys.shape == (4,steps,128)
                        for x in (mx,mass,queries,keys,state['key_mean'][layer]):
                            assert torch.isfinite(x).all()
                        error = (mass.float().sum(-1)-1).abs().max().item()
                        assert error < .001, error
                        max_error = max(max_error,error)
                        payload = dict(maxima=mx,sums=mass,queries=queries,
                            prompt_key_mean=state['key_mean'][layer],decode_keys=keys,
                            prompt_tokens=row['prompt_tokens'],split=row['split'],
                            row_positions=torch.arange(row['prompt_tokens']-64,row['prompt_tokens']+steps),
                            prompt_last_block_count=(row['prompt_tokens']-1)%16+1)
                        path = folder/f'layer_{layer:02d}.pt'
                        torch.save(payload,path)
                        files[path.name] = dict(sha256=digest(path),bytes=path.stat().st_size)
                        if protocol.get('collect_boundaries'):
                            tail_folder=out/'boundaries_v1'/'data'/row['id']
                            tail_folder.mkdir(parents=True,exist_ok=True)
                            torch.save(dict(prompt_tail_keys=state['tail'][layer],
                                tail_start=(row['prompt_tokens']-64)//16*16,
                                source_sha256=files[path.name]['sha256']),tail_folder/path.name)
                    record.update(status='success',files=files,max_probability_row_error=max_error,
                        tensor_bytes=sum(x['bytes'] for x in files.values()))
                    counts[key] += 1
                    accepted += 1
                record['seconds'] = time.perf_counter()-start
            except Exception as e:
                record.update(error=f'{type(e).__name__}: {e}',seconds=time.perf_counter()-start)
                save_json(out/'records'/f"{row['id']}.json",record)
                raise
            save_json(out/'records'/f"{row['id']}.json",record)
            done[row['id']] = record
            state.clear()
            print(json.dumps({k:record[k] for k in ('id','split','status','decode_steps','seconds')},ensure_ascii=False),flush=True)
            save_json(out/'progress.json',dict(success=sum(counts.values()),attempts=len(done),
                counts={f'{k[0]}/{k[1]}':v for k,v in counts.items()},
                tensor_bytes=sum(r.get('tensor_bytes',0) for r in done.values()),
                seconds=sum(r['seconds'] for r in done.values())))
            if args.limit and accepted>=args.limit:
                return
            if all(counts[(task,split)]==n for task in {r['source_task'] for r in rows} for split,n in targets.items()):
                save_json(out/'complete.json',json.loads((out/'progress.json').read_text()))
                return
        if not all(counts[(task,split)]==n for task in {r['source_task'] for r in rows} for split,n in targets.items()):
            raise RuntimeError(f'Candidate pool exhausted: {counts}')
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = original


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','collect','check'])
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--longbench',type=Path)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='prepare':
        prepare(args)
    elif args.stage=='check':
        check_pool()
    else:
        collect(args)


if __name__=='__main__':
    main()
