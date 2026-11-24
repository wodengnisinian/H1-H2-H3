"""Feature-only R08 role scorer with a fixed H2; no generator/answer code in inference."""
import torch
import torch.nn.functional as F


# Human-specified temporal and linking word classes. They are task priors,
# held fixed for all seeds. Neither argument of an input fact is consulted.
MARKERS = (
    ('先','首先','此前'), ('第一步','开头'),
    ('再','随后','之后'), ('第二','最后','结束'),
    ('从','以','起点','出发','开始'), ('其',),
    ('此人',), ('按','沿','追踪','找'),
)
BOUNDARY={'，','。','？','：'}
FAMILY={token:i for i,words in enumerate(MARKERS) for token in words}


def inputs(docs,width,vocabulary):
    """Return role lexicon mask and token features, shape N×T and N×T×64.

    For each query relation token and each marker family, use its nearest
    preceding occurrence. Four distance bins within the current fragment and
    four across punctuation delimiters. Markers on the right never contribute.
    """
    import numpy as np
    vocab=set(vocabulary)
    mask=np.zeros((len(docs),width),dtype=np.bool_)
    x=np.zeros((len(docs),width,64),dtype=np.float32)
    for i,d in enumerate(docs):
        seen=[None]*8
        fragment=0
        for j,token in enumerate(d.question):
            if token in vocab:
                mask[i,j]=True
                for family,entry in enumerate(seen):
                    if entry is None: continue
                    at,seg=entry
                    gap=j-at
                    bin_id=int(gap>2)+int(gap>5)+int(gap>10)
                    across=seg!=fragment
                    x[i,j,8*family+4*int(across)+bin_id]=1.0 if not across else 0.5
            if token in BOUNDARY: fragment+=1
            if token in FAMILY: seen[FAMILY[token]]=(j,fragment)
        if len({t for t in d.question if t in vocab})!=2:
            raise ValueError(f"Expected two distinct in-vocabulary relations in question {d.doc_id}")
    return torch.from_numpy(mask),torch.from_numpy(x)


def logits(f, x, valid, w):
    t=(x@w).masked_fill(~valid,-torch.inf)
    logp=t.log_softmax(-1)
    B=(logp[:,None,:]+f.relation).logsumexp(-1)
    return f.base+f.amplitude[:,None]*B,logp


def predict(bundle,docs,state):
    import model
    if state['model']!='cue_role' or any(d.gold is not None for d in docs):
        raise ValueError('Expected frozen cue_role state and feature-only examples')
    if state.get('feature_version')!='r08-previous-markers-v1' or state.get('markers')!=[list(a) for a in MARKERS]:
        raise ValueError('R08 feature checkpoint version mismatch')
    f=model.extract(bundle,docs)
    valid,x=inputs(docs,f.mask.shape[1],state['relations'])
    scores,_=logits(f,x,valid,torch.tensor(state['w'],dtype=torch.float32))
    return scores


def train(f, y, x, valid, role_target, which, cfg, seed):
    """role_target may contain multiple positions if the target word repeats."""
    eligible=(f.active & (y>=0)).nonzero().flatten()
    rng=torch.Generator().manual_seed(seed)
    w=torch.zeros(64,requires_grad=True)
    opt=torch.optim.Adam([w],lr=cfg['lr'])
    history=[]
    for ep in range(cfg['epochs']):
        total=0.; grad=0.
        for ix in eligible[torch.randperm(len(eligible),generator=rng)].split(cfg['batch']):
            t=(x[ix]@w).masked_fill(~valid[ix],-torch.inf)
            p=t.log_softmax(-1)
            if which=='relation':
                loss=-p.masked_fill(~role_target[ix],-torch.inf).logsumexp(-1).mean()
            elif which=='candidate':
                b=(p[:,None,:]+f.relation[ix]).logsumexp(-1)
                z=f.base[ix]+f.amplitude[ix,None]*b
                loss=F.cross_entropy(z,y[ix])
            else: raise ValueError(which)
            opt.zero_grad()
            loss.backward()
            grad+=float(w.grad.norm())*len(ix)
            opt.step()
            total+=float(loss.detach())*len(ix)
        history.append(dict(epoch=ep+1,examples=len(eligible),loss=total/len(eligible),
                            mean_grad_norm=grad/len(eligible),w_norm=float(w.detach().norm())))
    return w.detach(),history
