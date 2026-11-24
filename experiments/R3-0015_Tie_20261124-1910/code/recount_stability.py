"""Regroup predictions by canonical graph and seed, verifying count and strict decision."""
import collections,gzip,json
from pathlib import Path
here=Path(__file__).resolve().parents[1]
def rows(path):
    with gzip.open(here/path,"rt",encoding="utf8") as f:return list(map(json.loads,f))
def compute():
    raw={r["graph_id"]:r for r in rows("data/fresh-original.jsonl.gz")}
    graphs={r["graph_id"]:r for r in rows("results/graphs.jsonl.gz")}
    predictions=rows("results/predictions.jsonl.gz")
    assert len(raw)==len(graphs)==600 and len(predictions)==1800
    hits=collections.Counter(); strict=collections.defaultdict(list);examples={}; seen=set()
    for r in predictions:
        g=raw[r["graph_id"]];v=graphs[r["graph_id"]]; gold=g["gold"]
        assert r["gold"]==gold==v["gold"] and r["kind"]==g["kind"]
        key=(r["graph_id"],r["seed"]);assert key not in seen;seen.add(key)
        for rule,orders in r["rules"].items():
            cands=set();choices=set()
            for order,entry in orders.items():
                perm=v["presentations"][order]["local_to_canonical"]
                original_ids=tuple(perm[i] for i in entry["local_candidates"])
                assert original_ids==tuple(entry["candidates"])
                cands.add(original_ids);choices.add(entry["predictions"]["sharp_role"])
                assert entry["covered"]==(gold in original_ids)
                for m,pred in entry["predictions"].items():
                    assert pred==perm[entry["local_predictions"][m]] and pred in original_ids
                    hits[rule,order,m]+=pred==gold
            strict[rule,r["graph_id"]].append((len(cands)==1,len(choices)==1))
        before=r["rules"]["source_index"]["original"]["predictions"]["sharp_role"]
        after=r["rules"]["content_sha256"]["original"]["predictions"]["sharp_role"]
        if before!=after:
            label="corrected" if after==gold else "broken" if before==gold else "both_wrong"
            examples.setdefault(label,dict(graph_id=g["graph_id"],seed=r["seed"],gold=gold,before=before,after=after))
    results={}
    for rule in ("source_index","content_sha256"):
        sub=[vs for (m,g),vs in strict.items() if m==rule];assert len(sub)==600 and all(len(v)==3 for v in sub)
        results[rule]=dict(strict_joint=sum(all(all(pair) for pair in vs) for vs in sub),
                           prediction_only=sum(all(p for _,p in vs) for vs in sub))
    cloud=json.loads((here/"results/summary.json").read_bytes())
    assert all(results[k]["strict_joint"]==v for k,v in cloud["primary_passing_graphs"].items())
    return dict(graphs=len(graphs),graph_seeds=len(seen),strict=results,correct={"|".join(k):v for k,v in hits.items()},examples=examples)
if __name__=="__main__":print(json.dumps(compute(),ensure_ascii=False,indent=2))
