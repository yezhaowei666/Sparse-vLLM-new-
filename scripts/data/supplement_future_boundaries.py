"""Recover prompt-tail keys for exact token boundaries inside pooled blocks.

Replays only the original prompts. Saved decode queries/keys determine every
decode label; generation and the original trajectory files remain unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


@torch.inference_mode()
def recover_layer(source, prompt_keys, prompt_queries, source_sha, protocol_sha):
    """Verify replay identity before retaining the unpooled prompt-tail keys."""
    p = source['prompt_tokens']
    h, rows, dim = source['queries'].shape
    kh, steps, _ = source['decode_keys'].shape
    if rows != 64 + steps or p <= 128 or h % kh:
        raise ValueError('Invalid trajectory shape or prompt length')
    if prompt_keys.shape != (kh, p, dim) or prompt_queries.shape != (h, 64, dim):
        raise ValueError('Replayed prompt key/query shape differs from the source')
    if source['queries'].dtype != torch.bfloat16 or source['decode_keys'].dtype != torch.bfloat16:
        raise ValueError('Expected BF16 source queries and decode keys')
    if not torch.equal(source['row_positions'], torch.arange(p - 64, p + steps)):
        raise ValueError('Source row_positions do not match the causal trajectory')
    q = source['queries'][:, :64].to(prompt_keys.device)
    query_error = (prompt_queries.float() - q.float()).abs().max().item()
    if query_error != 0:
        raise ValueError(f'Replayed prompt Q differs: max_abs_error={query_error}')
    pad = (-p) % 16
    key_sum = F.pad(prompt_keys.float(), (0, 0, 0, pad)).reshape(kh, -1, 16, dim).sum(2)
    counts = torch.full((key_sum.shape[1],), 16, device=q.device)
    counts[-1] = 16 - pad
    means = (key_sum / counts[None, :, None]).to(torch.bfloat16)
    key_error = (means.float() - source['prompt_key_mean'].to(q.device).float()).abs().max().item()
    if key_error != 0:
        raise ValueError(f'Replayed prompt key mean differs: max_abs_error={key_error}')
    tail_start = ((p - 64) // 16) * 16
    payload = dict(version=1, prompt_tokens=p, tail_start=tail_start,
                   prompt_tail_keys=prompt_keys[:, tail_start:].cpu(),
                   source_sha256=source_sha, source_protocol_sha256=protocol_sha)
    errors = dict(query_max_abs_error=query_error, key_mean_max_abs_error=key_error)
    return payload, errors


def validate_source(root, row, protocol):
    path = root / 'records' / f"{row['id']}.json"
    record = json.loads(path.read_text())
    if record['status'] != 'success' or record['prompt_tokens'] != len(row['input_ids']):
        raise ValueError(f"Invalid source record: {row['id']}")
    if row['split'] != record['split'] or row['context_sha256'] != record['context_sha256']:
        raise ValueError(f"Source manifest and record disagree: {row['id']}")
    expected = {f'layer_{i:02d}.pt' for i in range(protocol['layers'])}
    if set(record['files']) != expected:
        raise ValueError(f"Incomplete source layers: {row['id']}")
    for name, info in record['files'].items():
        source = root / 'data' / row['id'] / name
        if source.stat().st_size != info['bytes'] or digest(source) != info['sha256']:
            raise ValueError(f'Source file checksum mismatch: {source}')
    return record, digest(path)


@torch.inference_mode()
def supplement(args):
    root, out = args.data_root.resolve(), args.output.resolve()
    if out == root:
        raise ValueError('--output must be a separate sidecar directory')
    source_protocol_path = root / 'protocol.json'
    protocol = json.loads(source_protocol_path.read_text())
    if str(args.model.resolve()) != protocol['model']:
        raise ValueError('Model path differs from the original collection protocol')
    if digest(args.model / 'config.json') != protocol['model_config_sha256']:
        raise ValueError('Model config checksum differs from the original dataset')
    required = dict(history=64, block_size=16, layers=28, query_heads=28, kv_heads=4, head_dim=128,
                    prefill_chunk=4096, model_dtype='BF16', attention='dense SDPA')
    if any(protocol[k] != v for k, v in required.items()):
        raise ValueError('Unsupported source collection protocol')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to reproduce BF16 attention')
    if torch.__version__ != protocol['torch']:
        raise ValueError('Use the same PyTorch version as the original collection')
    source_protocol_sha = digest(source_protocol_path)
    side_protocol = dict(version=1, source_protocol_sha256=source_protocol_sha,
                         source_manifest_sha256=digest(root / 'manifest.json'),
                         source_splits_sha256=digest(root / 'splits.json'),
                         model=protocol['model'], model_config_sha256=protocol['model_config_sha256'],
                         script_sha256=digest(Path(__file__)), torch=torch.__version__,
                         tail_start='floor((prompt_tokens-64)/16)*16',
                         boundary_probability='stored block mass times within-block causal softmax(QK)',
                         prompt_replay_tolerance=0)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(exist_ok=True)
    (out / 'records').mkdir(exist_ok=True)
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != side_protocol:
            raise ValueError('Existing sidecar protocol differs; choose a new output directory')
    else:
        if any((out / 'data').iterdir()) or any((out / 'records').iterdir()):
            raise ValueError('Sidecar files exist without a protocol')
        save_json(out / 'protocol.json', side_protocol)
    splits = json.loads((root / 'splits.json').read_text())
    ids = [doc for split in ('train', 'val', 'test') for doc in splits[split]]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate document across source splits')
    manifest = {r['id']: r for r in json.loads((root / 'manifest.json').read_text())}
    if args.limit_docs:
        ids = ids[:args.limit_docs]
    model = None
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    state = {}

    def capture(module, q, k, v, mask, **kwargs):
        result = original(module, q, k, v, mask, **kwargs)
        layer = module.layer_idx
        name = f'layer_{layer:02d}.pt'
        if name in state['record']['files']:
            return result
        length = k.shape[-2]
        count = min(q.shape[-2], max(0, length - (state['prompt'] - 64)))
        if count:
            state['queries'][layer].append(q[0, :, -count:].clone())
        if length == state['prompt']:
            source_path = root / 'data' / state['doc'] / name
            source = torch.load(source_path, map_location='cpu', weights_only=True)
            if source['split'] != state['source_record']['split']:
                raise ValueError(f'Source tensor split mismatch: {source_path}')
            payload, errors = recover_layer(source, k[0], torch.cat(state['queries'][layer], dim=1),
                                            state['source_record']['files'][name]['sha256'], source_protocol_sha)
            path = out / 'data' / state['doc'] / name
            tmp = path.with_suffix('.tmp')
            torch.save(payload, tmp)
            tmp.replace(path)
            state['record']['files'][name] = dict(sha256=digest(path), bytes=path.stat().st_size,
                                                 source_sha256=payload['source_sha256'], **errors)
            state['queries'][layer].clear()
            save_json(out / 'records' / f"{state['doc']}.json", state['record'])
        return result

    ALL_ATTENTION_FUNCTIONS['sdpa'] = capture
    try:
        for index, doc in enumerate(ids, 1):
            started = time.perf_counter()
            row = manifest[doc]
            source_record, record_sha = validate_source(root, row, protocol)
            record_path = out / 'records' / f'{doc}.json'
            record = dict(id=doc, status='incomplete', source_record_sha256=record_sha, files={})
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['source_record_sha256'] != record_sha or record['id'] != doc:
                    raise ValueError(f'Sidecar source record differs: {doc}')
                for name, info in record['files'].items():
                    path = out / 'data' / doc / name
                    if (info['source_sha256'] != source_record['files'][name]['sha256']
                            or path.stat().st_size != info['bytes'] or digest(path) != info['sha256']):
                        raise ValueError(f'Sidecar checksum mismatch: {path}')
            folder = out / 'data' / doc
            folder.mkdir(exist_ok=True)
            if any(path.name not in record['files'] for path in folder.glob('*.pt')):
                raise ValueError(f'Unindexed sidecar layer exists: {doc}')
            if len(record['files']) < protocol['layers']:
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model, dtype=torch.bfloat16, attn_implementation='sdpa',
                        local_files_only=True).cuda().eval()
                    model.requires_grad_(False)
                state.clear()
                state.update(doc=doc, prompt=row['prompt_tokens'], source_record=source_record,
                             record=record, queries=[[] for _ in range(protocol['layers'])])
                input_ids = torch.tensor([row['input_ids']], device='cuda')
                cache = None
                for start in range(0, input_ids.shape[1], protocol['prefill_chunk']):
                    result = model(input_ids=input_ids[:, start:start + protocol['prefill_chunk']],
                                   past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = result.past_key_values
                del result, cache, input_ids
                state.clear()
            if len(record['files']) != protocol['layers']:
                raise RuntimeError(f'Incomplete replay: {doc}')
            record['status'] = 'success'
            record['seconds'] = time.perf_counter() - started
            save_json(record_path, record)
            max_error = max(max(x['query_max_abs_error'], x['key_mean_max_abs_error'])
                            for x in record['files'].values())
            print(f"DOC {index}/{len(ids)} id={doc} status=success layers={len(record['files'])} "
                  f"replay_max_abs_error={max_error:.8f} seconds={record['seconds']:.1f}", flush=True)
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit-docs', type=int, default=0)
    args = parser.parse_args()
    if args.limit_docs < 0:
        parser.error('--limit-docs must be nonnegative')
    if args.output is None:
        args.output = args.data_root / 'boundaries_v1'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    supplement(args)


if __name__ == '__main__':
    main()
