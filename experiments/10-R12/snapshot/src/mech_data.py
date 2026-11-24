"""Development-only graph populations, including graph-disjoint template and pair challenges."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import random
import runpy


NEW_TEMPLATES = (
 "请 确定 第一步 的 记录 。 从 {anchor} 出发 ， 首先 追踪 {first} 关系 ， 之后 沿 {second} 关系 找 一人 。",
 "最后 一段 按 {second} 关系 。 此前 从 {anchor} 开始 ， 首先 沿 {first} 关系 找 一人 。 应 先 使用 哪条 记录 ？",
 "以 {anchor} 为 起点 ， 第一步 沿 {first} 关系 ， 随后 按 {second} 关系 找 一人 。 哪条 记录 对应 开头 一段 ？",
 "第二 段 沿 {second} 关系 找 一人 。 第一步 以 {anchor} 为 起点 ， 追踪 {first} 关系 。 应 先 使用 哪条 记录 ？",
)

# Lexicographically enumerate all 28 unordered pairs and choose seven before
# observing development labels. Both ordered orientations are withheld.
def pair_holdout(relations):
    import itertools
    return {pair for i,pair in enumerate(itertools.combinations(sorted(relations),2)) if i%4==0}


def pair_of(d,relations):
    values = sorted(set(d.question).intersection(relations))
    if len(values)!=2:
        raise ValueError(f"Expected two relations on {d.doc_id}")
    return tuple(values)


def fingerprint(d):
    return frozenset(person for h,_,t in d.nodes for person in (h,t))


def augment(train, templates, evaluator, seed):
    """Change training wording, preserving graph, gold and fixed candidates."""
    import data_views as dv
    token_templates = [t.split() for t in templates]
    docs, gold = train
    result = []
    rng = random.Random(seed)
    for d in docs:
        roles = dv.roles_for(d,token_templates)
        options = [t for t in templates if tuple(t.format(anchor=roles[0],first=roles[1],second=roles[2]).split())!=d.question]
        result.append(dv.make_question(d,roles,rng.choice(options),evaluator))
    return result, gold


def make_splits(b13,b16,bundle,cfg):
    import data_views as dv
    views, historical = dv.build_views(b13,b16,bundle.trainer,bundle.evaluator)
    train = views.pop('train')
    fresh, fresh_meta = dv.build_fresh_views(b13,bundle.trainer,bundle.evaluator)
    old = runpy.run_path(str(b13/'inputs/source/generate.py'))
    relations = sorted(old['RELATIONS'])
    held = pair_holdout(relations)
    train_pairs = [pair_of(d,relations) for d in train[0]]
    sub_idx = [i for i,pair in enumerate(train_pairs) if pair not in held]
    sub = ([train[0][i] for i in sub_idx],[train[1][i] for i in sub_idx])
    excluded = {fingerprint(d) for d in train[0]+views['original'][0]+fresh['original'][0]}
    excluded.update(frozenset(n for h,_,t in r['nodes'] for n in (h,t))
                    for r in dv.rows(b13/'cohort/features.jsonl'))
    styles = set(NEW_TEMPLATES)
    existing = set().union(*old['QUESTION_FAMILIES'].values(),dv.PARAPHRASE,dv.QUARTET,dv.FRESH_TEMPLATES,
                           runpy.run_path(str(b13/'code/generate_cohort.py'))['TEMPLATES'])
    # Include clause rewrites used by the archived frozen model.
    train_roles = {d.doc_id:dv.roles_for(d,[s.split() for s in old['QUESTION_FAMILIES']['train']]) for d in train[0]}
    for row in dv.rows(b16/dv.DATA_RELATIVE/'train-clause-views.jsonl'):
        a,b,c = train_roles[row['doc_id']]
        for key,r1,r2 in [('original',b,c),('swapped',c,b)]:
            rename = {a:'{anchor}',r1:'{first}',r2:'{second}'}
            for pairs in row[key] or []:
                for item in pairs:
                    existing.add(' '.join(rename.get(w,w) for w in item['question']))
    if styles.intersection(existing):
        raise ValueError('Challenge question template overlap')
    out = {'fresh':fresh,'archive':views}
    for si,split in enumerate(('template','combination')):
        rng = random.Random(cfg['challenge_seed']+si)
        raw=[]
        per=cfg['challenge_per_stratum']
        for k,stratum in enumerate(dv.STRATA):
            positions = [(i+k*per)%12 for i in range(per)]
            rng.shuffle(positions)
            for i,pos in enumerate(positions):
                while True:
                    index = rng.randrange(16800,70_000_000)
                    people = frozenset(old['ranked_people'](index))
                    if people in excluded:
                        continue
                    t = NEW_TEMPLATES[i%4] if split=='template' else dv.FRESH_TEMPLATES[i%4]
                    r = old['document'](index,stratum,pos,t,rng)
                    p = tuple(sorted(set(r['question']).intersection(relations)))
                    if split!='combination' or p in held:
                        break
                excluded.add(people)
                raw.append(r)
        main = dv.load_view(raw,bundle.trainer,bundle.evaluator)
        out[split] = dv.make_eval_views(main,NEW_TEMPLATES if split=='template' else dv.FRESH_TEMPLATES,bundle.evaluator)
    manifest = dict(roles=relations,held_pairs=sorted(held),frozen_train_pairs=len(set(train_pairs)),
        head_train_pairs=sorted({train_pairs[i] for i in sub_idx}),head_train_graphs=len(sub_idx),
        challenge_templates=NEW_TEMPLATES,checked_existing_templates=len(existing),
        archived_train_hash=historical['train_graph_set_sha256'],archived_dev_hash=historical['dev_graph_set_sha256'],
        fresh_index_hash=fresh_meta['person_indices_sha256'],dataset={})
    for split, variants in out.items():
        ds = variants['original'][0]
        manifest['dataset'][split] = {'size':len(ds),'ids_sha256':dv.canonical_hash([d.doc_id for d in ds]),
            'graphs_sha256':dv.canonical_hash(sorted(dv.graph_hash(d) for d in ds)),
            'candidates_sha256':dv.canonical_hash([list(d.candidates) for d in ds]),
            'gold_sha256':dv.canonical_hash(variants['original'][1]),
            'question_sha256':dv.canonical_hash([d.question for d in ds]),
            'strata':dict(Counter(d.stratum for d in ds))}
    return train,sub,out,relations,manifest
