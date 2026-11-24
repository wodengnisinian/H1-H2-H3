"""Audit an already opened H3 result from its sealed feature/prediction receipts."""
import argparse
from collections import Counter,defaultdict
import gzip,hashlib,json
from pathlib import Path
import numpy as np

def read(path):return json.loads(path.read_bytes())
def rows(path):
    with path.open(encoding="utf8") as f:yield from map(json.loads,f)
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def people(nodes):return frozenset(t for h,_,v in nodes for t in (h,v))

def audit(root,run):
    out=run/"results"; candidate=Path(read(out/"freeze.json")["candidate"])
    assert (out/"opening.json").is_file() and (out/"metrics.json").is_file()
    metrics=read(out/"metrics.json");cohort=read(out/"cohort.json");seal=read(out/"prediction-seal.json")
    assert sha(out/"prediction-seal.json")==read(out/"opening.json")["seal"]
    assert sha(out/"cohort.json")==seal["cohort"] and sha(out/"features.jsonl")==cohort["features_sha256"]
    assert all(sha(out/n)==h for n,h in seal["sha256"].items())
    exp=next((p for p in sorted((root/"experiments").glob("*-R03")) if (p/"results/metrics.json").is_file()))
    with gzip.open(exp/"results/development-inputs.jsonl.gz","rt",encoding="utf8") as f:
        exposed={people(r["nodes"]) for r in map(json.loads,f) if r["variant"]=="original"}
    sources=root.parent/"inputs";exposed.update(people(r["nodes"]) for r in rows(sources/"b13/cohort/features.jsonl"))
    for name in ("train.jsonl","dev.jsonl"):
        exposed.update(people(r["nodes"]) for r in rows(sources/"b16/inputs/b15/inputs/b14/inputs/server-run/data"/name))
    vocab=read(root/"configs/formal_templates.json")["original"]
    seen=set(); templates=Counter();positions=Counter()
    raw={r["doc_id"]:r for r in rows(out/"features.jsonl") if r["variant"]=="original"}
    assert len(raw)==4800
    frame=list(rows(out/"scored-predictions.jsonl"))
    gold={r["graph_id"]:int(r["gold"]) for r in frame if r["variant"]=="original"}
    for id_,r in raw.items():
        nodes=r["nodes"];key=people(nodes)
        assert key not in seen and key not in exposed
        seen.add(key)
        q=r["question"];matches=[]
        for i,style in enumerate(vocab):
            t=style.split()
            if len(t)==len(q) and all(a.startswith("{") or a==b for a,b in zip(t,q)):
                matches.append((i,{a:b for a,b in zip(t,q) if a.startswith("{")}))
        assert len(matches)==1
        i,roles=matches[0];templates[r["stratum"],i]+=1
        legal=[idx for idx,(h,rel,v) in enumerate(nodes) if h==roles["{anchor}"] and rel==roles["{first}"]
               and any(h2==v and r2==roles["{second}"] and len({h,v,v2})==3 for h2,r2,v2 in nodes)]
        assert legal==[gold[id_]];positions[gold[id_]]+=1
    assert len(templates)==12 and set(templates.values())=={400} and list(sorted(positions.values()))==[400]*12
    bytype=defaultdict(lambda:defaultdict(list))
    reverse=defaultdict(dict);correct=Counter()
    for r in frame:
        variant=r["variant"];p=r["predictions"]
        if variant=="original":
            bytype[r["kind"]][r["graph_id"]].append([int(p["learned"]==r["gold"])-int(p[b]==r["gold"]) for b in ("first","pool")])
            for s,v in p.items():correct[s]+=v==r["gold"]
        if variant.startswith("q"):reverse[r["graph_id"],r["seed"]][variant]=p["learned"]==r["gold"]
    n=4800*3
    for k,v in correct.items():
        score=metrics["question_views"]["original"]["overall"]["accuracy"][k]
        assert v==score["hits"] and score["denominator"]==n
    four=sum(all(v.values()) for v in reverse.values())
    assert all(len(v)==4 for v in reverse.values()) and four==metrics["four_question"]["overall"]["all_four"]["learned"]["hits"]
    rng=np.random.default_rng(913651);boot=np.zeros((10000,2));points=np.zeros(2)
    for ds in bytype.values():
        assert len(ds)==1600 and all(len(v)==3 for v in ds.values())
        m=np.mean(list(ds.values()),axis=1);points+=m.mean(0)/3
        for i in range(0,len(boot),100):
            boot[i:i+100]+=m[rng.integers(len(m),size=(100,len(m)))].mean(1)/3
    reference=metrics["paired_comparisons"]["original"]["contrasts_learned_minus"]
    assert np.allclose(points*100,[reference[b]["delta_pp"] for b in ("first","pool")])
    return dict(status="verified",graphs=len(seen),shared_person_sets=0,original_graph_seed=n,all_four=four,
                original_correct=correct,templates_per_type=list(templates.values()),gold_positions=positions,
                independent_graph_bootstrap_lcb_pp=(100*np.sort(boot,axis=0)[499]).tolist(),baselines=["first","pool"],
                authoritative_metrics_sha256=sha(out/"metrics.json"))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    a=p.parse_args();result=audit(a.root.resolve(),a.run.resolve())
    (a.root/"analysis/正式验收复核.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False))
