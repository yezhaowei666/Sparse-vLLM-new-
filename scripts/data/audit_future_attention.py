"""Verify saved trajectory integrity, causality, split isolation, and block mass."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    root = args.root
    torch.set_num_threads(4)
    complete = json.loads((root/'complete.json').read_text())
    records = [json.loads(p.read_text()) for p in (root/'records').glob('*.json')]
    good = [r for r in records if r['status']=='success']
    manifest = {r['id']:r for r in json.loads((root/'manifest.json').read_text())}
    assert len(good)==complete['success']==32
    assert len({r['context_sha256'] for r in good})==32
    counts=Counter((r['source_task'],r['split']) for r in good)
    assert all(counts[(task,split)]==n for task in ('govreport','qmsum') for split,n in [('train',12),('val',2),('test',2)])
    files=0
    max_error=0.
    for r in good:
        assert r['split']==manifest[r['id']]['split']
        p,d=r['prompt_tokens'],r['decode_steps']
        assert 8000<=p<=12000 and 256<=d<=511
        assert len(manifest[r['id']]['input_ids'])==p and len(r['generated_ids'])==d+1
        positions=torch.arange(p-64,p+d)
        width=(p+d+15)//16
        causal=torch.arange(width)[None,:]>positions[:,None]//16
        for layer in range(28):
            path=root/'data'/r['id']/f'layer_{layer:02d}.pt'
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):
                    h.update(chunk)
            assert h.hexdigest()==r['files'][path.name]['sha256']
            x=torch.load(path,map_location='cpu',weights_only=True)
            assert x['maxima'].shape==x['sums'].shape==(28,64+d,width)
            assert x['queries'].shape==(28,64+d,128)
            assert x['decode_keys'].shape==(4,d,128)
            assert x['prompt_key_mean'].shape==(4,(p+15)//16,128)
            assert x['prompt_last_block_count']==(p-1)%16+1
            assert torch.equal(x['row_positions'],positions)
            assert all(torch.isfinite(x[k]).all() for k in ('maxima','sums','queries','decode_keys','prompt_key_mean'))
            mx,mass=x['maxima'].float(),x['sums'].float()
            assert (mx>=0).all() and (mass>=0).all()
            assert (mass.masked_select(causal[None].expand_as(mass))==0).all()
            assert (mx.masked_select(causal[None].expand_as(mx))==0).all()
            # Independent block invariant, allowing BF16/FP16 rounding.
            assert (mass+1e-6>=mx*.995).all()
            assert (mass<=mx*16*1.005+1e-6).all()
            error=(mass.sum(-1)-1).abs().max().item()
            assert error<.001
            max_error=max(max_error,error)
            files+=1
        print('AUDIT PASS',r['id'],flush=True)
    report=dict(status='success',documents=len(good),layer_files=files,
        split_counts={s:sum(r['split']==s for r in good) for s in ('train','val','test')},
        input_min=min(r['prompt_tokens'] for r in good),input_max=max(r['prompt_tokens'] for r in good),
        decode_min=min(r['decode_steps'] for r in good),decode_max=max(r['decode_steps'] for r in good),
        decode_total=sum(r['decode_steps'] for r in good),
        windows_4_by_split={s:sum(r['decode_steps']-3 for r in good if r['split']==s) for s in ('train','val','test')},
        stopped= dict(Counter(r['ended_by'] for r in good)),
        rejected=dict(Counter(r['status'] for r in records if r['status']!='success')),
        max_probability_row_error=max_error,tensor_bytes=sum(r['tensor_bytes'] for r in good),
        collection_seconds=sum(r['seconds'] for r in records),
        peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in good))
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'splits.json').write_text(json.dumps({s:sorted(r['id'] for r in good if r['split']==s)
        for s in ('train','val','test')},indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
