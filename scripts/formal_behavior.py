"""Read already-opened predictions to score relation reversal and paraphrase at graph level."""
import argparse,collections,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
a=p.parse_args();out=a.run/"results";assert (out/"metrics.json").exists() and (out/"opening.json").exists()
panels=collections.defaultdict(dict)
with (out/"scored-predictions.jsonl").open(encoding="utf8") as f:
    for line in f:
        r=json.loads(line);panels[(r["kind"],r["graph_id"],r["seed"])][r["variant"]]=r
methods=list(next(iter(panels.values()))["original"]["predictions"])
ans={}
for method in methods:
    def hit(v):return int(v["predictions"][method]==v["gold"])
    def pick(v):return v["predictions"][method]
    per=collections.defaultdict(lambda:collections.defaultdict(list))
    for (kind,g,seed),panel in panels.items():
        scores={"rewrite_same_choice":int(pick(panel["original"])==pick(panel["paraphrase"])),
                "original_and_rewrite_correct":hit(panel["original"])*hit(panel["paraphrase"])}
        if "q0" in panel:
            q=[panel["q"+str(i)] for i in range(4)]
            scores.update(reversal_both_correct=.5*(hit(q[0])*hit(q[1])+hit(q[2])*hit(q[3])),
                          reversal_changed_choice=.5*((pick(q[0])!=pick(q[1]))+(pick(q[2])!=pick(q[3]))),
                          quartet_paraphrase_agreement=.5*((pick(q[0])==pick(q[2]))+(pick(q[1])==pick(q[3]))))
        for k,v in scores.items():per[k][kind,g].append(v)
    res={}
    for name,by_graph in per.items():
        bytype=collections.defaultdict(list)
        for (kind,g),v in by_graph.items():
            assert len(v)==3
            bytype[kind].append(np.mean(v))
        rng=np.random.default_rng(25013);boot=np.zeros(10000);point=0
        for values in bytype.values():
            ar=np.array(values);point+=ar.mean()/len(bytype)
            for i in range(0,len(boot),100):
                boot[i:i+100]+=ar[rng.integers(len(ar),size=(100,len(ar)))].mean(1)/len(bytype)
        res[name]=dict(graphs=len(by_graph),seeds=3,percent=100*point,
                       two_sided_95_percent=(100*np.sort(boot)[[249,9749]]).tolist())
    ans[method]=res
(a.root/"analysis/H3行为检验.json").write_text(json.dumps(ans,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print(json.dumps(ans["learned"]))
