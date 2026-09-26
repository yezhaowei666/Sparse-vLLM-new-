"""Build disjoint mixed-task prompts from original training sources and synthetic records."""
import argparse
import json
import math
import random
import re
import shutil
from pathlib import Path

from transformers import AutoTokenizer
from scripts.data.collect_future_attention import context_hash, save_json, digest


def prepare(args):
    root=args.output
    if (root/'protocol.json').exists():
        raise FileExistsError(root/'protocol.json')
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rng=random.Random(20260923)
    evaluation=[json.loads(line) for p in args.longbench.glob('*.jsonl') for line in p.open()]
    excluded={context_hash(r['context']) for r in evaluation}
    # Short exact fragments catch partial source reuse, not just whole-prompt copies.
    eval_fragments=set()
    for r in evaluation:
        words=re.sub(r'\s+',' ',r['context'].lower()).split()
        for i in range(0,len(words)-39,20):
            eval_fragments.add(' '.join(words[i:i+40]))
    def overlap(text):
        words=re.sub(r'\s+',' ',text.lower()).split()
        return any(' '.join(words[i:i+40]) in eval_fragments for i in range(len(words)-39))
    rows=[]; seen=set(excluded)
    def add(category, prompt, context, provenance):
        ids=tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,return_dict=False)
        if not 5000<=len(ids)<=16000 or context_hash(context) in seen:
            return False
        j=sum(r['category']==category for r in rows)
        if j>=10:return False
        rows.append(dict(id=f'mixed_{category}_{j:02d}',category=category,source_task=category,
                         split='train' if j<8 else 'val',source_split='train',provenance=provenance,
                         prompt=prompt,input_ids=ids,prompt_tokens=len(ids),context_sha256=context_hash(context)))
        seen.add(context_hash(context));return True
    def source(name):return [r['row'] for r in json.loads((root/'sources'/f'{name}.json').read_text())]
    qasper=source('qasper');rng.shuffle(qasper)
    for r in qasper:
        context=r['title']+'\n'+r['abstract']+'\n'+'\n'.join('\n'.join(p) for p in r['full_text']['paragraphs'])
        if overlap(context):continue
        add('SQA','Read the paper and answer the question concisely using only the paper.\n\n'+context+'\n\nQuestion: '+r['qas']['question'][0]+'\nAnswer:',context,dict(dataset='allenai/qasper',id=r['id']))
    hotpot=source('hotpot');rng.shuffle(hotpot)
    eligible=[]
    for r in hotpot:
        text='\n'.join(t+': '+''.join(ss) for t,ss in zip(r['context']['title'],r['context']['sentences']))
        if not overlap(text):eligible.append((r,text))
    used_titles=set(); cursor=0
    while cursor<len(eligible) and sum(r['category']=='MQA' for r in rows)<10:
        context=[];provenance=[]; question=None; n=0
        while cursor<len(eligible) and n<7500:
            r,text=eligible[cursor];cursor+=1
            if used_titles.intersection(r['context']['title']):continue
            used_titles.update(r['context']['title'])
            context.append(text);provenance.append(r['id']);n+=len(tok.encode(text))
            if question is None:question=r['question']
        if n<5000:break
        rng.shuffle(context);context='\n\n'.join(context)
        add('MQA','Answer the question using the following articles. Give only the answer.\n\n'+context+'\n\nQuestion: '+question+'\nAnswer:',context,dict(dataset='hotpotqa/hotpot_qa',ids=provenance))
    trec=source('trec'); rng.shuffle(trec)
    # Each episode uses separate original training examples, including demonstrations.
    for j in range(10):
        episode=trec[j*500:(j+1)*500]
        context='\n\n'.join('Question: '+r['text']+'\nType: '+r['label_text'] for r in episode[:-1])
        prompt='Classify the final question using the same fine-grained question types as the examples. Output only its type.\n\n'+context+'\n\nQuestion: '+episode[-1]['text']+'\nType:'
        add('FewShot',prompt,context,dict(dataset='SetFit/TREC-QC',questions=[r['text'] for r in episode]))
    code=source('code'); rng.shuffle(code);code.sort(key=lambda r:r['repository_name']);cursor=0;used_repos=set()
    while cursor<len(code) and sum(r['category']=='Code' for r in rows)<10:
        parts=[];provenance=[];n=0;repos=set()
        while cursor<len(code) and n<7500:
            r=code[cursor];cursor+=1
            if r['repository_name'] in used_repos or overlap(r['whole_func_string']):continue
            text=r['whole_func_string'];parts.append(text);n+=len(tok.encode(text));repos.add(r['repository_name']);provenance.append(r['func_code_url'])
        if n<5000:break
        used_repos.update(repos)
        last=parts[-1];cut=max(1,int(len(last)*.65)); context='\n\n'.join(parts[:-1])+ '\n\n'+last[:cut]
        add('Code','Continue the Python code below. Output only the continuation.\n\n'+context,context,dict(dataset='code-search-net/code_search_net',urls=provenance))
    for j in range(10):
        records=[]
        for i in range(220):
            records.append(f'Entry {i+1}: reference {rng.getrandbits(64):016x}; city {rng.choice(["Paris","Lima","Tokyo","Oslo"])}; quantity {rng.randrange(10,1000)}; status {rng.choice(["open","closed","pending"])}.')
        target=rng.randrange(220);context='\n'.join(records)
        question=(f'What is the reference in Entry {target+1}? Output only the reference.' if j%2==0 else 'How many entries have status pending? Output only the number.')
        add('Synthetic',context+'\n\n'+question,context,dict(dataset='locally_generated_records',seed=20260923,index=j))
    # Preserve the original summary train/validation document split.
    summary=[]
    manifest=json.loads((args.summary/'manifest.json').read_text())
    split_ids=json.loads((args.summary/'splits.json').read_text())
    for split,per_source in [('train',4),('val',1)]:
        for task in ['govreport','qmsum']:
            chosen=[r for r in manifest if r['id'] in split_ids[split] and r['source_task']==task][:per_source]
            for r in chosen:
                summary.append(dict(r,category='Summary',source_task='Summary',reuse_root=str(args.summary)))
    rows.extend(summary)
    from collections import Counter
    counts=Counter((r['category'],r['split']) for r in rows)
    print('COUNTS',counts,flush=True)
    if len(rows)!=60 or sorted(counts.values())!=[2]*6+[8]*6:raise ValueError('Incomplete balanced cohort')
    if len({r['context_sha256'] for r in rows})!=60:raise ValueError('Duplicate contexts')
    protocol=json.loads((args.summary/'protocol.json').read_text())
    upper=sum(28*(28*191*math.ceil((r['prompt_tokens']+127)/16)*4 + 28*191*128*2
                  + 4*math.ceil(r['prompt_tokens']/16)*128*2 + 4*127*128*2)
              for r in rows if 'reuse_root' not in r)
    if shutil.disk_usage(root).free < upper + 2*2**30:
        raise RuntimeError('Not enough space for collected trajectories plus 2 GB')
    for key in ('source_files','candidate_counts','preparation_skips'):
        protocol.pop(key,None)
    protocol.update(version=4,seed=20260923,script_sha256=digest(Path(__file__).with_name('collect_future_attention.py')),
                    preparation_script_sha256=digest(Path(__file__)),max_new_tokens=128,min_decode_steps=1,
                    collect_boundaries=True,target_per_source={'train':8,'val':2},
                    min_input_tokens=5000,max_input_tokens=16000,tensor_upper_bytes=upper,max_attempts=60,
                    source_description='Original training splits plus new synthetic records; no LongBench examples used for training',
                    categories={c:dict(train=8,val=2) for c,_ in counts},sources={p.name:digest(p) for p in (root/'sources').glob('*.json')},
                    excluded_longbench_contexts=len(excluded),generation='greedy natural EOS, retain all nonempty decode trajectories',
                    input_bounds=[5000,16000],summary_reuse=str(args.summary))
    save_json(root/'manifest.json',rows);save_json(root/'protocol.json',protocol)
    save_json(root/'splits.json',{s:[r['id'] for r in rows if r['split']==s] for s in ('train','val','test')})
    (root/'records').mkdir();(root/'data').mkdir()
    (root/'boundaries_v1'/'data').mkdir(parents=True)
    save_json(root/'boundaries_v1'/'protocol.json',dict(source_protocol_sha256=digest(root/'protocol.json'),method='inline exact post-RoPE prompt tail'))
    for row in summary:
        doc=row['id']; src=args.summary
        record=json.loads((src/'records'/f'{doc}.json').read_text())
        record.update(category='Summary',source_task='Summary',reuse_root=str(src))
        save_json(root/'records'/f'{doc}.json',record)
        (root/'data'/doc).symlink_to(src/'data'/doc,target_is_directory=True)
        (root/'boundaries_v1'/'data'/doc).symlink_to(src/'boundaries_v1'/'data'/doc,target_is_directory=True)
    print('PREPARED',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['output','model','longbench','summary']:p.add_argument('--'+key,type=Path,required=True)
    prepare(p.parse_args())
