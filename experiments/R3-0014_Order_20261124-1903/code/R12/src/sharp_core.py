"""R12 feature-only hard relation decoding of a frozen R08 role reader."""
import torch
from role_core import MARKERS, inputs as role_inputs


def features(docs,width,state):
    if state.get('feature_version')!='r08-previous-markers-v1' or state.get('markers')!=[list(a) for a in MARKERS]:
        raise ValueError('Frozen R08 feature map mismatch')
    mask,x=role_inputs(docs,width,state['relations'])
    # Sort by Unicode code point for deterministic ties independent of candidates
    # and surface order. Collect every mention, including repeated mentions.
    classes=torch.zeros((len(docs),2,width),dtype=torch.bool)
    vocab=set(state['relations'])
    names=[]
    for i,d in enumerate(docs):
        words=sorted(set(d.question).intersection(vocab))
        if len(words)!=2:raise ValueError('Expected two distinct question relation words')
        names.append(words)
        for j,t in enumerate(d.question):
            if t in vocab:classes[i,words.index(t),j]=True
    return mask,x,classes,names


def decode(x,mask,classes,w):
    t=(x@w).masked_fill(~mask,-torch.inf)
    logp=t.log_softmax(-1)
    # Normalizer of softmax(t) cancels in the pairwise comparison of masses.
    masses=t[:,None,:].masked_fill(~classes,-torch.inf).logsumexp(-1)
    choice=masses.argmax(1)
    selected=classes[torch.arange(len(t)),choice]
    hard=torch.zeros_like(t).masked_fill(~selected,-torch.inf).log_softmax(-1)
    return logp,hard,choice,masses.softmax(-1)


def rank(f,logp):
    return f.base+f.amplitude[:,None]*(logp[:,None,:]+f.relation).logsumexp(-1)


def scope_inputs(bundle,docs,state):
    """Expose the exact mask, pre-linear features and relation grouping.

    H1 consumes question tokens and the frozen R08 relation vocabulary. The
    bundle's encoder supplies the separate H2 directional and matching terms.
    """
    if any(d.gold is not None for d in docs):raise ValueError('Expected feature-only examples')
    mask,x,classes,_=features(docs,max(len(d.question) for d in docs),state)
    return mask,x,classes


def predict(bundle,docs,state):
    import model
    if state.get('model')!='sharp_role' or state.get('decoder')!='max-relation-mass-uniform-v1' or any(d.gold is not None for d in docs):
        raise ValueError('Expected R12 sharp_role state and feature-only examples')
    f=model.extract(bundle,docs)
    mask,x,classes,_=features(docs,f.mask.shape[1],state)
    _,hard,_,_=decode(x,mask,classes,torch.tensor(state['w'],dtype=torch.float32))
    return rank(f,hard)
