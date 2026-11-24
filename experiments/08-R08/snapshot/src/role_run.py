"""Run R08 nested mask, representation and objective controls on development graphs."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import torch
from mech_core import correct_index
from mech_data import make_splits
from mech_metrics import group_summary,average_dict,confidence
from mech_run import balanced,save,now,digest,line
import role_core as rc


def train_controls(bundle,train,rels,cfg,seed,log):
    import model
    ds,ys=train
    f=model.extract(bundle,ds,cfg['feature_batch'])
    y=correct_index(ds,ys)
    real,x=rc.inputs(ds,f.mask.shape[1],rels)
    pos=torch.zeros_like(f.mask); rnd=torch.zeros_like(f.mask)
    order=[]
    for i,(d,g) in enumerate(zip(ds,ys,strict=True)):
        target=d.nodes[g][1]
        terms=list(dict.fromkeys(t for t in d.question if t in rels))
        order.append(terms.index(target))
        pos[i,:len(d.question)]=torch.tensor([t==target for t in d.question])
    eligible=(f.active & (y>=0)).nonzero().flatten()
    rng=torch.Generator().manual_seed(seed+7943)
    perm=eligible[torch.randperm(len(eligible),generator=rng)]
    shuffled=torch.tensor(order)
    shuffled[eligible]=torch.tensor(order)[perm]
    for i,d in enumerate(ds):
        terms=list(dict.fromkeys(t for t in d.question if t in rels))
        word=terms[int(shuffled[i])]
        rnd[i,:len(d.question)]=torch.tensor([t==word for t in d.question])
    params={}
    for k,values,mask,target,objective in [
        ('hidden_all_CE',f.hidden,f.mask,pos,'candidate'),('hidden_rel_CE',f.hidden,real,pos,'candidate'),
        ('cue_candidate',x,real,pos,'candidate'),('cue_role',x,real,pos,'relation'),
        ('cue_random_role',x,real,rnd,'relation')]:
        params[k],history=rc.train(f,y,values,mask,target,objective,cfg,seed)
        log('trained',method=k,seed=seed,eligible=len(eligible),history=history,trainable_parameters=64)
    return params


def score_view(b,ds,ys,rels,params,cfg,seed,split,variant):
    import model
    f=model.extract(b,ds,cfg['feature_batch'])
    mask,x=rc.inputs(ds,f.mask.shape[1],rels)
    y=correct_index(ds,ys)
    scores={'pool':f.pooled,'direction':f.base}
    raw={}
    for k,w in params.items():
        feat=f.hidden if k in ('hidden_all_CE','hidden_rel_CE','prior') else x
        valid=f.mask if k in ('prior','hidden_all_CE') else mask
        scores[k],p=rc.logits(f,feat,valid,w)
        raw[k]=p.exp()
    scores['rel_uniform'],p=rc.logits(f,x,mask,torch.zeros(64))
    raw['rel_uniform']=p.exp()
    role_words=[list(dict.fromkeys(t for t in d.question if t in rels)) for d in ds]
    for tag,col in [('surface_first',0),('surface_last',1)]:
        m=torch.zeros_like(mask)
        for i,d in enumerate(ds):
            m[i,:len(d.question)]=torch.tensor([w==role_words[i][col] for w in d.question])
        scores[tag],p=rc.logits(f,x,m,torch.zeros(64))
        raw[tag]=p.exp()
    # A real candidate-order intervention: shortlist membership fixed, full
    # feature path rerun, check question-only H1 and H2 score equivariance.
    flipped=[replace(d,candidates=tuple(reversed(d.candidates))) for d in ds]
    swapped=model.extract(b,flipped,cfg['feature_batch'])
    mt,xt=rc.inputs(flipped,swapped.mask.shape[1],rels)
    flip,p=rc.logits(swapped,xt,mt,params['cue_role'])
    delta=(flip-scores['cue_role'].flip(1)).abs().amax(1)
    if float(delta.max())>2e-5 or not torch.equal(p.exp(),raw['cue_role']):
        raise ValueError('Candidate swap changed H1 or permuted scores')
    if not torch.equal(scores['cue_role'][~f.active],f.base[~f.active]):
        raise ValueError('Closed gate identity failed')
    positions=mask.nonzero()
    for i,(d,g,terms) in enumerate(zip(ds,ys,role_words,strict=True)):
        true=d.nodes[g][1]
        displayed=[d.question.index(t) for t in terms]
        # Retain all mentions in case a later question repeats one relation.
        occurrence=[[j for j,t in enumerate(d.question) if t==term] for term in terms]
        role={}
        for k,ps in raw.items():
            masses=[float(ps[i,idx].sum()) for idx in occurrence]
            choice=terms[int(masses[1]>masses[0])] if masses[0]!=masses[1] else None
            role[k]={'masses_in_surface_order':masses,'choice':choice,
                     'hit_tie_half':0.5 if choice is None else float(choice==true)}
        preds={k:d.candidates[int(v[i].argmax())] for k,v in scores.items()}
        preds['first']=d.candidates[0]
        yield dict(split=split,variant=variant,seed=seed,kind=d.stratum,graph_id=d.doc_id,gate=bool(f.active[i]),
            covered=bool(y[i]>=0),gold=g,candidates=d.candidates,predictions=preds,
            terms_in_surface_order=terms,first_relation=true,first_mentioned_earlier=true==terms[0],
            h1=role,swap_error=float(delta[i]),logits={k:v[i].tolist() for k,v in scores.items()})


def semantic_stats(rows):
    grouped=defaultdict(list);index={}
    for r in rows:
        grouped[r['variant']].append(r);index[(r['seed'],r['graph_id'],r['variant'])]=r
    out={}
    for k in rows[0]['h1']:
        def choice(r): return r['h1'][k]['choice']
        def hit(r):return r['h1'][k]['hit_tie_half']
        def mate(r,v):return index[(r['seed'],r['graph_id'],v)]
        first=grouped['original'];q0=grouped['q0']
        out[k]={'original_role':float(np.mean([hit(r) for r in first])),
                'paraphrase_role':float(np.mean([hit(r) for r in grouped['paraphrase']])),
                'original_tie_rate':float(np.mean([choice(r) is None for r in first])),
                'surface_first_role':float(np.mean([hit(r) for r in first if r['first_mentioned_earlier']])),
                'surface_last_role':float(np.mean([hit(r) for r in first if not r['first_mentioned_earlier']])),
                'rewrite_agreement':float(np.mean([choice(r) is not None and choice(r)==choice(mate(r,'paraphrase')) for r in first])),
                'reversal_both':float(np.mean([all(choice(mate(r,v))==mate(r,v)['first_relation'] for v in pair)
                                             for r in q0 for pair in [('q0','q1'),('q2','q3')]])),
                'four_roles_correct':float(np.mean([all(choice(mate(r,'q'+str(j)))==mate(r,'q'+str(j))['first_relation'] for j in range(4)) for r in q0]))}
    return out


def objective_intervals(rows,cfg):
    """Paired accuracy and role-readout gains over every nontrivial nested control."""
    rs=[r for r in rows if r['variant']=='original']
    rank_bases=('hidden_all_CE','hidden_rel_CE','cue_candidate','cue_random_role','rel_uniform','surface_first','surface_last')
    metrics=[(task,k) for task in ('rank','role') for k in rank_bases]
    strata=defaultdict(lambda:defaultdict(list))
    for r in rs:
        def val(method,task):
            return int(r['predictions'][method]==r['gold']) if task=='rank' else r['h1'][method]['hit_tie_half']
        strata[r['kind']][r['graph_id']].append([val('cue_role',t)-val(b,t) for t,b in metrics])
    rng=np.random.default_rng(cfg['bootstrap_seed']); draws=cfg['bootstrap_draws']
    point=np.zeros(len(metrics));boot=np.zeros((draws,len(metrics)))
    for graphs in strata.values():
        x=np.array([np.mean(s,0) for s in graphs.values()])
        point+=x.mean(0)/len(strata)
        for j in range(0,draws,100):
            n=min(100,draws-j);ix=rng.integers(len(x),size=(n,len(x)))
            boot[j:j+n]+=x[ix].mean(1)/len(strata)
    band=np.sort(boot,axis=0)
    return {task+'_vs_'+base:{'delta':float(point[i]),'lower95':float(band[499,i]),
                'two_sided_95':[float(band[249,i]),float(band[9749,i])]} for i,(task,base) in enumerate(metrics)}


def run(a):
    import model
    cfg=json.loads(a.config.read_bytes());a.out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic()
    def log(e,**kw):
        with (a.out/'events.jsonl').open('a',encoding='utf8') as f: line(f,dict(utc=now(),event=e,**kw))
    save(a.out/'configuration.json',dict(protocol=cfg,protocol_sha256=digest(a.config),argv=sys.argv,utc=now(),
        python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform(),seeds=a.seeds,
        source_sha256={p.name:digest(p) for p in list(Path(__file__).parent.glob('role_*.py'))+
                        list(Path(__file__).parent.glob('mech_*.py'))+[a.code/'model.py',a.code/'data_views.py']}))
    try:
        b=model.load_bundle(a.b13,a.seeds[0]);train,sub,splits,rels,manifest=make_splits(a.b13,a.b16,b,cfg)
        vocab=sorted({r for d in train[0] for _,r,_ in d.nodes})
        if vocab!=rels: raise ValueError('Declared relation alphabet differs from training facts')
        save(a.out/'dataset.json',manifest)
        sub=balanced(sub,a.smoke)
        if a.smoke: splits={k:{v:balanced(pair,a.smoke) for v,pair in views.items()} for k,views in splits.items()}
        # Graphs and questions are reused across seeds. Preserve exact fixed input.
        with gzip.open(a.out/'development-inputs.jsonl.gz','wt',encoding='utf8') as f:
            for split,views in splits.items():
                for variant,(ds,ys) in views.items():
                    for d,g in zip(ds,ys,strict=True):
                        obj=dict(split=split,variant=variant,kind=d.stratum,graph_id=d.doc_id,question=d.question,gold=g,candidates=d.candidates)
                        if variant=='original':obj['nodes']=d.nodes
                        line(f,obj)
        per={};allrows=defaultdict(list);time_seed={}
        for seed in a.seeds:
            t=time.monotonic()
            if seed!=a.seeds[0]:b=model.load_bundle(a.b13,seed)
            log('begin_seed',seed=seed)
            params=train_controls(b,sub,vocab,cfg,seed,log)
            path=a.weights/f'w-{seed}.json';params['prior']=torch.tensor(json.loads(path.read_bytes())['w'])
            checkpoint=dict(model='cue_role',seed=seed,w=params['cue_role'].tolist(),feature_version=cfg['version'],
                 markers=[list(a) for a in rc.MARKERS],relations=vocab)
            save(a.out/f'checkpoint-{seed}.json',checkpoint)
            save(a.out/f'parameters-{seed}.json',dict(seed=seed,prior_sha256=digest(path),weights={k:v.tolist() for k,v in params.items()}))
            per[seed]={}
            with gzip.open(a.out/f'predictions-{seed}.jsonl.gz','wt',encoding='utf8') as out,gzip.open(
                        a.out/f'errors-{seed}.jsonl.gz','wt',encoding='utf8') as err:
                for split,views in splits.items():
                    rows=[]
                    for variant,(ds,ys) in views.items():
                        for r in score_view(b,ds,ys,vocab,params,cfg,seed,split,variant):
                            line(out,r)
                            if r['predictions']['cue_role']!=r['gold']:line(err,r)
                            r.pop('logits')
                            for stats in r['h1'].values():stats.pop('masses_in_surface_order')
                            rows.append(r)
                    per[seed][split]=dict(ranking=group_summary(rows),h1=semantic_stats(rows))
                    allrows[split].extend(rows)
                    log('scored',seed=seed,split=split,rows=len(rows))
            time_seed[seed]=time.monotonic()-t
        agg={sp:average_dict([per[s][sp]['ranking'] for s in a.seeds]) for sp in splits}
        h1={sp:semantic_stats(rows) for sp,rows in allrows.items()}
        h3={sp:confidence(rows,'cue_role',cfg['bootstrap_draws'],cfg['bootstrap_seed']) for sp,rows in allrows.items()}
        ci={sp:objective_intervals(rows,cfg) for sp,rows in allrows.items()}
        fresh=agg['fresh']; main=fresh['cue_role']
        check=(all(main['original_top1']-fresh[b]['original_top1']>=.05-1e-12 and main['paraphrase_top1']>fresh[b]['paraphrase_top1'] for b in ('first','pool')) and
               main['quartet_all']>fresh['first']['quartet_all'] and main['quartet_all']>=fresh['pool']['quartet_all'])
        support=(main['original_top1']>fresh['cue_candidate']['original_top1'] and
                h1['fresh']['cue_role']['original_role']>h1['fresh']['cue_candidate']['original_role'])
        eligible=bool(check and a.seeds==cfg['seeds'] and not a.smoke and not a.diagnostic_only)
        save(a.out/'metrics.json',dict(round='role_objective',primary_model='cue_role',seeds=a.seeds,aggregate=agg,h1=h1,
              per_seed=per,controls=list(fresh),paired_intervals=h3,objective_intervals=ci,
              criterion_met=support,development_gate={'criteria_met':check,'eligible':eligible},
              diagnostic_only=a.diagnostic_only,smoke=bool(a.smoke),protocol_sha256=digest(a.config),
              elapsed_seconds=time.monotonic()-tick,seconds_by_seed=time_seed,
              max_candidate_swap_error=max(r['swap_error'] for rs in allrows.values() for r in rs),
              hypothesis=cfg['hypothesis'],finished_utc=now()))
        log('finished',gate=eligible,hypothesis_supported=support)
    except BaseException:
        log('failed',trace=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('b13','b16','prior','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code',type=Path);p.add_argument('--weights',type=Path)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/role_protocol.json')
    p.add_argument('--seeds',nargs='+',type=int,default=[1701,2701,3701]);p.add_argument('--smoke',type=int,default=0)
    p.add_argument('--diagnostic-only',action='store_true',help='Ineligible if an earlier candidate has already frozen.')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or not set(a.seeds)<={1701,2701,3701}:p.error('invalid seeds')
    a.b13=a.b13.resolve();a.b16=a.b16.resolve();a.code=(a.code or a.prior/'code').resolve()
    a.weights=(a.weights or a.prior/'runs/本地开发-01').resolve()
    sys.path.insert(0,str(a.code));sys.dont_write_bytecode=True;run(a)
