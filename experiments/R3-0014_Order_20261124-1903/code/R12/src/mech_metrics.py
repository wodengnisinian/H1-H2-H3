"""Graph-paired performance on first hop and within-graph behavior."""
from collections import defaultdict
import numpy as np


def group_summary(rows):
    methods=rows[0]['predictions'].keys()
    index={(r['graph_id'],r['variant']):r for r in rows}
    values={}
    for method in methods:
        hits=defaultdict(list)
        strat=defaultdict(list)
        gate=defaultdict(list)
        form=defaultdict(list)
        for r in rows:
            ok=r['predictions'][method]==r['gold']
            hits[r['variant']].append(ok)
            if r['variant']=='original':
                strat[r['kind']].append(ok)
                gate[str(r['gate'])].append(ok)
                form[str(r['first_mentioned_earlier'])].append(ok)
        all4=[];pair=[];reversal=[];paraphrase=[];qrewrite=[]
        for r in rows:
            if r['variant']=='original':
                p=index[(r['graph_id'],'paraphrase')]
                paraphrase.append(r['predictions'][method]==p['predictions'][method])
            if r['variant']!='q0':
                continue
            four=[index[(r['graph_id'],f'q{i}')] for i in range(4)]
            picks=[v['predictions'][method] for v in four]
            ok=[p==v['gold'] for p,v in zip(picks,four,strict=True)]
            all4.append(all(ok));pair.extend([all(ok[:2]),all(ok[2:])])
            reversal.extend([picks[0]!=picks[1],picks[2]!=picks[3]])
            qrewrite.extend([picks[0]==picks[2],picks[1]==picks[3]])
        avg=lambda x:float(np.mean(x)) if x else None
        d={'original_top1':avg(hits['original']),'paraphrase_top1':avg(hits['paraphrase']),
           'quartet_all':avg(all4),'reversal_both':avg(pair),'reversal_changed':avg(reversal),
           'rewrite_agreement':avg(paraphrase),'quartet_rewrite_agreement':avg(qrewrite),
           'graph_count':len(hits['original']),'quartet_graphs':len(all4),
           'by_kind':{k:avg(a) for k,a in strat.items()},'by_gate':{k:avg(a) for k,a in gate.items()},
           'by_relation_mention_order':{k:avg(a) for k,a in form.items()}}
        values[method]=d
    return values


def average_dict(values):
    if isinstance(values[0],dict):
        return {key:average_dict([a[key] for a in values]) for key in values[0]}
    return float(np.mean(values)) if values[0] is not None else None


def confidence(rows,model,draws,seed):
    out={}
    for variant in ('original','paraphrase'):
        graph=defaultdict(list)
        for r in rows:
            if r['variant']==variant:
                p=r['predictions']; g=r['gold']
                graph[(r['kind'],r['graph_id'])].append([int(p[model]==g)-int(p[b]==g) for b in ('first','pool')])
        kinds=sorted({s for s,_ in graph})
        rng=np.random.default_rng(seed)
        boot=np.zeros((draws,2)); point=np.zeros(2)
        for s in kinds:
            matrix=np.array([np.mean(h,0) for (k,_),h in graph.items() if k==s])
            point+=matrix.mean(0)/len(kinds)
            for j in range(0,draws,128):
                n=min(128,draws-j)
                idx=rng.integers(len(matrix),size=(n,len(matrix)))
                boot[j:j+n]+=matrix[idx].mean(1)/len(kinds)
        band=np.sort(boot,axis=0)
        out[variant]={b:{'delta':float(point[i]),'lcb95':float(band[int(draws*.05)-1,i]),
                         'two_sided_95':[float(band[int(draws*.025)-1,i]),float(band[int(draws*.975)-1,i])]}
                      for i,b in enumerate(('first','pool'))}
    return out
