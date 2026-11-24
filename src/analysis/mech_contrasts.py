"""Recompute predeclared mechanism contrasts with paired graph-bootstrap intervals.

Usage: python mech_contrasts.py RESULTS OUT.json
Each graph is one sample; all its seeds and all four questions travel together.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


CONTRASTS={
    'ablation':[('learned',base) for base in ('uniform','random','shuffled','old','direction')],
    'groups':[('four','one')],
    'joint':[('cue64','matched_plain'),('cue64','learned'),('cue64','first'),('cue64','pool')],
    'readout':[('read_four','read_four_random_labels'),('read_one','read_one_random_labels'),('read_four','read_one')],
    'geometry':[('oracle','direction'),('ceiling','direction')],
}


def is_hit(row,method):
    if method.startswith('read_'):
        return int(row['h1']['probes'][method]['pair']==row['first_relation'])
    if method=='oracle': return int(row['geometry']['oracle_fact']==row['gold'])
    if method=='ceiling': return int(row['geometry']['possible'])
    return int(row['predictions'][method]==row['gold'])


def contrast_ci(cells,seed):
    rng=np.random.default_rng(seed)
    estimates=[]
    draws=[]
    for matrix in cells.values():
        x=np.array(matrix,dtype=np.float64)
        estimates.append(x.mean(0))
        parts=[]
        for off in range(0,10000,100):
            sample=rng.integers(len(x),size=(100,len(x)))
            parts.append(x[sample].mean(1))
        draws.append(np.concatenate(parts))
    boot=np.sort(np.mean(draws,0),axis=0)
    return (np.mean(estimates,axis=0),boot[249],boot[9749],boot[499])


def main(args):
    p=args.results
    metric=json.loads((p/'metrics.json').read_bytes())
    stage=metric['round']
    specs=CONTRASTS[stage]
    methods=sorted(set(t for pair in specs for t in pair))
    pergraph=defaultdict(dict); hashes={}
    for s in metric['seeds']:
        path=p/f'predictions-{s}.jsonl.gz'
        with path.open('rb') as f: hashes[path.name]=hashlib.file_digest(f,'sha256').hexdigest()
        with gzip.open(path,'rt',encoding='utf8') as f:
            for raw in f:
                r=json.loads(raw)
                key=(r['split'],r['kind'],r['graph_id'])
                nested=pergraph[key].setdefault(s,{})
                if r['variant'] in nested: raise ValueError('Duplicate observation')
                nested[r['variant']]=[is_hit(r,m) for m in methods]
    by_view=defaultdict(lambda:defaultdict(list))
    for (sp,kind,g),all_seeds in pergraph.items():
        if set(all_seeds)!=set(metric['seeds']):raise ValueError('Missing seed')
        for view in ('original','paraphrase','quartet'):
            hits=[]
            for obs in all_seeds.values():
                if view=='quartet':
                    if 'q0' not in obs: continue
                    vals=np.prod([obs['q'+str(i)] for i in range(4)],axis=0)
                else:
                    vals=np.array(obs[view])
                hits.append([vals[methods.index(a)]-vals[methods.index(b)] for a,b in specs])
            if hits:
                if len(hits)!=len(all_seeds):raise ValueError('Missing per-seed quartet')
                by_view[(sp,view)][kind].append(np.mean(hits,0))
    out={}
    for (sp,view),strata in sorted(by_view.items()):
        mean,lo,hi,lcb=contrast_ci(strata,args.seed)
        out.setdefault(sp,{})[view]={
            'graphs':{k:len(v) for k,v in strata.items()},
            'contrasts':{a+' - '+b:{'delta':float(mean[i]),'two_sided_95':[float(lo[i]),float(hi[i])],
                        'one_sided_lcb95':float(lcb[i])} for i,(a,b) in enumerate(specs)}}
    with args.out.open('x',encoding='utf8') as f:
        json.dump({'round':stage,'seeds':metric['seeds'],'diagnostic_only':True,'bootstrap_seed':args.seed,
          'bootstrap_draws':10000,'correction':'Descriptive paired intervals, no adjustment across multiple comparisons.',
          'score':('Two-visible-relation identification' if stage=='readout' else 'First-hop accuracy'),
          'predictions_sha256':hashes,'results':out},f,indent=2,ensure_ascii=False,allow_nan=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--seed',type=int,default=819432)
    main(parser.parse_args())
