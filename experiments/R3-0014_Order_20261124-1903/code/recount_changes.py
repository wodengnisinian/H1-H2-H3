"""Independent cross-tabulation of a completed development intervention, no model invocation."""
import collections,csv,gzip,json
from pathlib import Path
here=Path(__file__).resolve().parents[1]
def read_rows(name):
    with gzip.open(here/name,"rt",encoding="utf8") as f:return list(map(json.loads,f))
def count():
    rows=read_rows("results/predictions.jsonl.gz")
    raw={r["graph_id"]:r for r in read_rows("data/fresh-original.jsonl.gz")}
    assert len(rows)==len({(r["seed"],r["graph_id"]) for r in rows})==1800 and len(raw)==600
    assert collections.Counter(r["seed"] for r in rows)=={1701:600,2701:600,3701:600}
    tally=collections.defaultdict(collections.Counter);examples={}
    for r in rows:
        g=raw[r["graph_id"]];assert g["gold"]==r["gold"] and g["kind"]==r["kind"]
        base=r["views"]["identity"];assert base["ordered_candidates"]==g["candidates"]
        for v,x in r["views"].items():
            changed_set=set(x["ordered_candidates"])!=set(base["ordered_candidates"])
            assert changed_set==x["set_changed"] and x["covered"]==(g["gold"] in x["ordered_candidates"])
            for m,a in x["predictions"].items():
                b=base["predictions"][m];c=tally[v,m]
                assert a in x["ordered_candidates"]
                c["correct"]+=a==g["gold"];c["changed"]+=a!=b
                if a!=b and m=="sharp_role":
                    c["newly_correct"]+=a==g["gold"];c["newly_wrong"]+=b==g["gold"]
                    c["both_wrong"]+=g["gold"] not in (a,b)
                    c["changed_set"]+=changed_set;c["competition"]+=g["kind"]=="competing_chain"
                    key=(v,"corrected" if a==g["gold"] else "broken" if b==g["gold"] else "both_wrong")
                    if key not in examples:examples[key]=dict(graph_id=g["graph_id"],seed=r["seed"],question=g["question"],gold=g["gold"],before=b,after=a,original_candidates=base["ordered_candidates"],modified_candidates=x["ordered_candidates"])
    with (here/"results/accuracy.csv").open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["stratum"]=="all":
                c=tally[row["view"],row["method"]]
                assert c["correct"]==int(row["correct"]) and c["changed"]==int(row["changed_predictions"])
    return {"unique_graphs":len(raw),"graph_seeds":len(rows),"counts":{"|".join(k):dict(v) for k,v in tally.items()},"examples":{"|".join(k):v for k,v in examples.items()}}
if __name__=="__main__":print(json.dumps(count(),ensure_ascii=False,indent=2))
