"""Question-only, fixed-capacity probes and first-relation diagnostics."""
import torch
import torch.nn.functional as F
from mech_core import question_groups, single_group


def representation(bundle,f):
    groups=question_groups(bundle,f)
    # Both expose 64 values. Fixed coordinate blocks of identical size retain
    # group identity; one-control performs exactly one attention read.
    return {'read_one':single_group(bundle,f),'read_four':torch.cat([groups[:,i,16*i:16*(i+1)] for i in range(4)],dim=1)}


def probe_fit(bundle, f, docs, labels, relations, seed):
    y=torch.tensor([relations.index(d.nodes[g][1]) for d,g in zip(docs,labels,strict=True)])
    target=F.one_hot(y,len(relations)).double()
    rng=torch.Generator().manual_seed(seed)
    random_order=torch.randperm(len(y),generator=rng)
    out={}
    for name,x in representation(bundle,f).items():
        x=torch.cat([x,torch.ones(len(x),1)],1).double()
        gram=x.T@x+torch.eye(x.shape[1],dtype=x.dtype)
        out[name]=torch.linalg.solve(gram,x.T@target).float()
        out[name+'_random_labels']=torch.linalg.solve(gram,x.T@target[random_order]).float()
    return out


def read_probe(bundle,f,docs,relations,state):
    xs=representation(bundle,f)
    visible=torch.tensor([[r in d.question for r in relations] for d in docs])
    results={}
    for name,vector in state.items():
        x=xs[name.split('_random_labels')[0]]
        x=torch.cat([x,torch.ones(len(x),1)],1)
        scores=x@vector
        predictions=scores.masked_fill(~visible,-torch.inf).argmax(1)
        results[name]=(predictions.tolist(),scores.argmax(1).tolist())
    return results


def masses(docs,gold,logp,relations):
    p=logp.exp()
    output=[]
    for i,(d,g) in enumerate(zip(docs,gold,strict=True)):
        first=d.nodes[g][1]
        pair=sorted(set(d.question).intersection(relations))
        other=next(r for r in pair if r!=first)
        a=sum(float(p[i,j]) for j,w in enumerate(d.question) if w==first)
        b=sum(float(p[i,j]) for j,w in enumerate(d.question) if w==other)
        output.append((a,b,0.5 if a==b else float(a>b),first,other))
    return output
