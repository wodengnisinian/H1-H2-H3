"""Execute one registered CPU mechanism experiment; output directory must be new."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import runpy
import sys
import time
import traceback

import numpy as np
import torch
from mech_data import augment, make_splits
import mech_core as core
import mech_probe as probe
from mech_metrics import group_summary, average_dict, confidence


def now(): return datetime.now(timezone.utc).isoformat()

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def line(f,row):
    f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def balanced(view,limit):
    if not limit: return view
    ds,ys=view
    c=defaultdict(int); keep=[]
    for i,d in enumerate(ds):
        if c[d.stratum]<limit:
            c[d.stratum]+=1;keep.append(i)
    return [ds[i] for i in keep],[ys[i] for i in keep]


def fit_round(args,cfg,bundle,sub,prior,seed,log):
    import model
    stage=args.round
    w=torch.tensor(prior['w'],dtype=torch.float32)
    state={'w':w,'histories':{},'proof':{}}
    if stage not in ('groups','joint','readout'): return state
    ds,ys=sub
    if stage=='joint':
        families=runpy.run_path(str(args.b13/'inputs/source/generate.py'))['QUESTION_FAMILIES']['train']
        aug,_=augment(sub,families,bundle.evaluator,seed)
        ds,ys=ds+aug,ys+ys
    feats=model.extract(bundle,ds,cfg['feature_batch'])
    y=core.correct_index(ds,ys)
    if stage=='readout':
        state['probes']=probe.probe_fit(bundle,feats,ds,ys,cfg['_relations'],seed)
    elif stage=='joint':
        for key,with_cues in [('matched_plain',False),('cue64',True)]:
            vector,hist=core.fit_joint(feats,ds,y,seed,cfg,with_cues=with_cues)
            state[key]=vector;state['histories'][key]=hist
            log('trained',seed=seed,method=key,eligible=int((feats.active&(y>=0)).sum()),history=hist)
    elif stage=='groups':
        base,amp=core.one_direction(bundle,ds,feats)
        for key,features in [('four',feats),('one',replace(feats,base=base,amplitude=amp))]:
            vector,hist=model.fit(features,y,epochs=cfg['epochs'],lr=cfg['lr'],batch_size=cfg['batch'],seed=seed)
            state[key]=vector;state['histories'][key]=hist
            log('trained',seed=seed,method=key,eligible=int((feats.active&(y>=0)).sum()),history=hist)
    return state


def run_view(args,cfg,b,docs,gold,st,split,variant,seed):
    import model
    f=model.extract(b,docs,cfg['feature_batch'])
    y=core.correct_index(docs,gold)
    w=st['w']
    weights=core.weights(f,w)
    learned=core.apply_logp(f,weights)
    scores={'pool':f.pooled,'direction':f.base}
    extra={}
    bounds=None; oracle=None; read=None; mass=None; swaperr=None
    if args.round=='geometry':
        bounds=model.correct_margin_bounds(f,y)
        gold_mentions=torch.zeros_like(f.mask)
        for i,(d,g) in enumerate(zip(docs,gold,strict=True)):
            rel=d.nodes[g][1]
            gold_mentions[i,:len(d.question)]=torch.tensor([t==rel for t in d.question])
        if not bool(gold_mentions.any(1).all()): raise ValueError('Gold relation missing')
        logp=torch.zeros_like(f.mask,dtype=torch.float32).masked_fill(~gold_mentions,-torch.inf).log_softmax(-1)
        oracle=core.apply_logp(f,logp)
    elif args.round=='ablation':
        rng=torch.Generator().manual_seed(seed+442)
        rnd=torch.randn(w.shape,generator=rng);rnd=rnd*w.norm()/rnd.norm()
        scores.update(old=f.old,uniform=core.apply_logp(f,core.weights(f,w*0)),learned=learned,
            random=core.apply_logp(f,core.weights(f,rnd)),
            shuffled=core.apply_logp(f,core.shuffled_logp(weights,f.mask,docs,seed)))
        rev=[replace(d,candidates=tuple(reversed(d.candidates))) for d in docs]
        rf=model.extract(b,rev,cfg['feature_batch'])
        rw=core.weights(rf,w)
        swapped=core.apply_logp(rf,rw)
        swaperr=(swapped-learned.flip(1)).abs().amax(1)
        if swaperr.max()>2e-5 or not torch.allclose(weights.exp(),rw.exp(),atol=2e-5,rtol=0):
            raise ValueError('Candidate swap audit failed')
    elif args.round=='readout':
        scores['learned']=learned
        read=probe.read_probe(b,f,docs,cfg['_relations'],st['probes'])
        mass=probe.masses(docs,gold,weights,cfg['_relations'])
    elif args.round=='groups':
        base,amp=core.one_direction(b,docs,f)
        scores['four']=core.apply_logp(f,core.weights(f,st['four']))
        scores['one']=core.apply_logp(f,core.weights(f,st['one']),base,amp)
    elif args.round=='joint':
        scores['learned']=learned
        for tag in ('matched_plain','cue64'):
            mode='learned' if tag=='matched_plain' else tag
            scores[tag]=core.from_state(f,docs,{'model':mode,'w':st[tag].tolist()})
    if any(not v.isfinite().all() for v in scores.values()): raise ValueError('Nonfinite scores')
    for i,(d,g) in enumerate(zip(docs,gold,strict=True)):
        preds={tag:d.candidates[int(v[i].argmax())] for tag,v in scores.items()}
        preds['first']=d.candidates[0]
        correctrel=d.nodes[g][1]
        other=next(t for t in d.question if t in cfg['_relations'] and t!=correctrel)
        r=dict(graph_id=d.doc_id,kind=d.stratum,variant=variant,seed=seed,split=split,
            gold=g,candidates=d.candidates,covered=bool(y[i]>=0),gate=bool(f.active[i]),
            first_relation=correctrel,first_mentioned_earlier=d.question.index(correctrel)<d.question.index(other),
            predictions=preds,logits={tag:v[i].tolist() for tag,v in scores.items()})
        if bounds is not None:
            lo,hi=bounds[i].tolist()
            possible=bool(y[i]>=0 and (hi>0 or (hi==0 and y[i]==0)))
            r['geometry']={'min':lo if y[i]>=0 else None,'max':hi if y[i]>=0 else None,
               'possible':possible,'oracle_fact':d.candidates[int(oracle[i].argmax())],
               'oracle_logits':oracle[i].tolist(),'direction_wrong':preds['direction']!=g}
        if swaperr is not None: r['swap_logits_error']=float(swaperr[i])
        if read is not None:
            a,bmass,hit,first,second=mass[i]
            r['h1']={'mass_first':a,'mass_second':bmass,'mass_correct_tie_half':hit,
                     'mass_choice':first if a>bmass else second if bmass>a else None,
                     'first':first,'second':second,
                     'probes':{k:{'pair':cfg['_relations'][a[i]],'all':cfg['_relations'][c[i]]} for k,(a,c) in read.items()}}
        yield r


def diagnostics(rows, stage):
    subsets=defaultdict(list)
    for r in rows:
        subsets[(r['split'],r['variant'])].append(r)
    outcome={}
    for (sp,var),rs in subsets.items():
        if var not in ('original','paraphrase'): continue
        def avg(values): return float(np.mean(values))
        d={'n':len(rs),'coverage':avg([r['covered'] for r in rs])}
        if stage=='readout':
            d['mass_first_accuracy']=avg([r['h1']['mass_correct_tie_half'] for r in rs])
            d['mass_tie_rate']=avg([r['h1']['mass_choice'] is None for r in rs])
            for k in rs[0]['h1']['probes']:
                for dim in ['all','pair']:
                    d[k+'_'+dim]=avg([r['h1']['probes'][k][dim]==r['first_relation'] for r in rs])
            if var=='original':
                qs={v:{(r['graph_id'],r['seed']):r for r in subsets[(sp,v)]} for v in ['q0','q1','q2','q3','paraphrase']}
                for method in ['mass',*rs[0]['h1']['probes']]:
                    def picked(r):
                        return r['h1']['mass_choice'] if method=='mass' else r['h1']['probes'][method]['pair']
                    keys=qs['q0'].keys()
                    if keys:
                        d[method+'_reverse_both']=avg([all(picked(qs[v][key])==qs[v][key]['first_relation'] for v in pair)
                                               for key in keys for pair in [('q0','q1'),('q2','q3')]])
                    d[method+'_rewrite_stability']=avg([picked(r) is not None and picked(r)==picked(qs['paraphrase'][(r['graph_id'],r['seed'])]) for r in rs])
        if stage=='geometry':
            for gate,sel in [('all',rs),('open',[r for r in rs if r['gate']]),('closed',[r for r in rs if not r['gate']])]:
                d[gate]={'rows':len(sel),'attainable_ceiling':avg([r['geometry']['possible'] for r in sel]),
                    'label_oracle_accuracy':avg([r['geometry']['oracle_fact']==r['gold'] for r in sel]),
                    'direction_errors':sum(r['geometry']['direction_wrong'] for r in sel),
                    'structurally_uncorrectable':sum(r['geometry']['direction_wrong'] and not r['geometry']['possible'] for r in sel)}
        if stage=='ablation': d['max_swap_error']=max(r['swap_logits_error'] for r in rs)
        outcome.setdefault(sp,{})[var]=d
    return outcome


def run(args):
    import model
    cfg=json.loads(args.config.read_text(encoding='utf8'))
    args.out.mkdir(parents=True,exist_ok=False)
    def log(event,**kw):
        with (args.out/'events.jsonl').open('a',encoding='utf8') as fo: line(fo,dict(utc=now(),event=event,**kw))
    initial=now(); start=time.monotonic()
    save(args.out/'configuration.json',dict(config=cfg,round=args.round,argv=sys.argv,started=initial,
        python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform(),
        seeds=args.seeds,smoke_per_stratum=args.smoke,code_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('mech_*.py')},
        prior_code_sha256={n:digest(args.code/n) for n in ['model.py','data_views.py']},protocol_sha256=digest(args.config)))
    try:
        b=model.load_bundle(args.b13,args.seeds[0])
        train,sub,splits,relations,manifest=make_splits(args.b13,args.b16,b,cfg)
        cfg['_relations']=relations
        save(args.out/'dataset.json',manifest)
        sub=balanced(sub,args.smoke)
        if args.smoke:
            splits={k:{v:balanced(data,args.smoke) for v,data in views.items()} for k,views in splits.items()}
        log('data_ready',pair_filtered_train=len(sub[0]),historical_train=len(train[0]))
        primary={'readout':'learned','geometry':'direction','ablation':'learned','groups':'four','joint':'cue64'}[args.round]
        perseed={};allrows=defaultdict(list);seed_time={}
        # Store each fact graph once and every question and gold candidate order, so reported mistakes can be inspected.
        with gzip.open(args.out/'development-inputs.jsonl.gz','wt',encoding='utf8') as fh:
            for split,views in splits.items():
                for variant,(ds,ys) in views.items():
                    for d,g in zip(ds,ys,strict=True):
                        obj=dict(split=split,variant=variant,graph_id=d.doc_id,kind=d.stratum,question=d.question,
                                 candidates=d.candidates,gold=g)
                        if variant=='original': obj['nodes']=d.nodes
                        line(fh,obj)
        for seed in args.seeds:
            tick=time.monotonic()
            if seed!=args.seeds[0]: b=model.load_bundle(args.b13,seed)
            weights=args.weights/f'w-{seed}.json'
            prior=json.loads(weights.read_bytes())
            if prior['seed']!=seed: raise ValueError('Checkpoint seed mismatch')
            log('begin_seed',seed=seed,prior_sha256=digest(weights))
            st=fit_round(args,cfg,b,sub,prior,seed,log)
            state_json={k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in st.items() if k!='probes'}
            if 'probes' in st: state_json['probes']={k:v.tolist() for k,v in st['probes'].items()}
            save(args.out/f'parameters-{seed}.json',state_json)
            if args.round=='joint':
                save(args.out/f'checkpoint-{seed}.json',{'model':'cue64','seed':seed,'w':st['cue64'].tolist()})
            perseed[seed]={}
            with gzip.open(args.out/f'predictions-{seed}.jsonl.gz','wt',encoding='utf8') as fo, gzip.open(
                args.out/f'errors-{seed}.jsonl.gz','wt',encoding='utf8') as err:
                for split,views in splits.items():
                    slim=[]
                    for variant,(docs,ys) in views.items():
                        for r in run_view(args,cfg,b,docs,ys,st,split,variant,seed):
                            line(fo,r)
                            if r['predictions'][primary]!=r['gold']: line(err,r)
                            r.pop('logits'); slim.append(r)
                    perseed[seed][split]=group_summary(slim)
                    allrows[split].extend(slim)
                    log('scored',seed=seed,split=split,rows=len(slim))
            seed_time[seed]=time.monotonic()-tick
        aggregate={sp:average_dict([perseed[s][sp] for s in args.seeds]) for sp in splits}
        intervals={sp:confidence(rs,primary,cfg['bootstrap_draws'],cfg['bootstrap_seed']) for sp,rs in allrows.items()}
        diags=diagnostics([r for rs in allrows.values() for r in rs],args.round)
        for s in args.seeds:
            perseed[s]['diagnostics']=diagnostics([r for rs in allrows.values() for r in rs if r['seed']==s],args.round)
        fresh=aggregate['fresh']; main=fresh[primary]
        check=(all(main['original_top1']-fresh[b]['original_top1']>=.05-1e-12 and
                   main['paraphrase_top1']>fresh[b]['paraphrase_top1'] for b in ('first','pool'))
               and main['quartet_all']>fresh['first']['quartet_all'] and main['quartet_all']>=fresh['pool']['quartet_all'])
        if args.round=='readout':
            support=all(diags[sp]['original']['read_four_pair']-diags[sp]['original']['read_four_random_labels_pair']>=.05 for sp in ('fresh','template','combination'))
        elif args.round=='geometry':
            support=diags['fresh']['original']['all']['attainable_ceiling']>=max(fresh[b]['original_top1'] for b in ('first','pool'))+.05
        elif args.round=='ablation':
            support=all(main[k]>fresh[c][k] for c in ('random','shuffled','uniform') for k in ('original_top1','quartet_all'))
        elif args.round=='groups':
            support=all(main[k]>fresh['one'][k] for k in ('original_top1','quartet_all'))
        else: support=check
        gate={'criteria_met':check,'eligible':args.round=='joint' and check and not args.smoke and args.seeds==cfg['seeds']}
        save(args.out/'metrics.json',dict(round=args.round,seeds=args.seeds,primary_model=primary,aggregate=aggregate,
            per_seed=perseed,controls=list(fresh),paired_intervals=intervals,diagnostics=diags,
            development_gate=gate,criterion_met=bool(support),smoke=bool(args.smoke),
            protocol_sha256=digest(args.config),elapsed_seconds=time.monotonic()-start,seconds_by_seed=seed_time,
            started_utc=initial,finished_utc=now(),hypothesis=cfg['rounds'][args.round]))
        log('finished',criterion_met=bool(support),gate=gate)
    except BaseException:
        log('failed',trace=traceback.format_exc())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round',choices=['readout','geometry','ablation','groups','joint'],required=True)
    for name in ['prior','b13','b16','out']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code',type=Path)
    p.add_argument('--weights',type=Path,help='directory containing prior w-{seed}.json')
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/mech_protocol.json')
    p.add_argument('--seeds',type=int,nargs='+',default=[1701,2701,3701])
    p.add_argument('--smoke',type=int,default=0,help='maximum graphs per stratum per view, excludes formal qualification')
    a=p.parse_args()
    a.b13=a.b13.resolve();a.b16=a.b16.resolve()
    a.code=(a.code or a.prior/'code').resolve()
    a.weights=(a.weights or a.prior/'runs/本地开发-01').resolve()
    if len(set(a.seeds))!=len(a.seeds) or not set(a.seeds)<={1701,2701,3701}: p.error('invalid seeds')
    sys.path.insert(0,str(a.code))
    sys.dont_write_bytecode=True
    run(a)
