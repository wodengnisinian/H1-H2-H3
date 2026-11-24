"""Replay a frozen H1/H2 version and its controls on a supplied development-only cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import traceback

import torch


SEEDS = (1701, 2701, 3701)
RELATIONS = sorted(("拜访", "联系", "协助", "邀请", "推荐", "提醒", "指导", "跟随"))
VERSIONS = {"original": "learned", "r12": "sharp_role"}
STRATA = {"role_reversal", "relation_order", "competing_chain"}
VIEWS = {"original", "paraphrase", "unseen_template", "q0", "q1", "q2", "q3"}
TOLERANCE = 2e-5
WEIGHTS = ("a3cf4167344c14eb0d0e281e9e43ce379289fe40157e39de9a230074c211fa59",
           "1d3a0e23cdb417f229a7231cec039b8d9b7ec803c08b539a4f38ce4fc6f183a9",
           "9217b8edda35fad73650fbe8bdf354e26df0a4f0f8530b6549d62773f93f670e")
PROBES = ("e3ea1d8e7c43620bc661880eff3f5b5447da9f5819fc0b13ec42856e8dd64bfd",
          "a1bc12ee4b806c7ac4b6b345c131ffe264b93feef8497015bc4ffe9faf01b50e",
          "6c44e08bfeba8f0bd6db4e061bb7cb5131ba57292bbc877a0b0fa07047bfaeec")
GROUPS = ("37ee3fcc9da697e3a419631ef41f3348470043c994ab6b270ed03326d0c51355",
          "ad101a3b8533e9cf738da41806913dea738acedcf895b572ad664bee681fb027",
          "561a5fe7d04d8511d9a78271721b7318c7d5a3ac3ec8816c59c32276e81ee036")
SHARP = ("98f1c920d07cb74a4409c4aaab32c8f21069e6e8cfcbefc8a5344dcd25a2e33d",
         "3e1653c209aee3161b1431fb7029d1b9464c25195c2136cafd0e9ab107012dd7",
         "2bd1a3da698ed71c9d816843c75cfd3d3a9299a11e5d743faed421f039f792e8")
CUE_STATES = ("5b5483916092e5d1e0949ac597e523a12c6d9385b3d8aa518986550f60cd000e",
              "86ee5be6bdf6f2efcb4e17cbbb30d8e5833d311a5e38ecf2dd411ef90fa5e48d",
              "145306da24c5ac3f39fa57fbd6b1078c2d5c61d66d63139d83e752b7cd141c13")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_file(path, value):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def event(out, name, **details):
    entry = dict(utc=datetime.now(timezone.utc).isoformat(), event=name, **details)
    with (out / "events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def check_file(path, expected, receipt):
    digest = sha(path)
    require(digest == expected, f"Frozen file digest mismatch: {path}")
    receipt[str(path)] = digest
    return path


def vector(x):
    v = torch.tensor(x, dtype=torch.float32)
    require(v.shape == (64,) and v.isfinite().all(), "Expected a finite frozen 64-vector")
    return v


def load_state(args, seed, receipt):
    i = SEEDS.index(seed)
    exp = args.r3 / "experiments"

    def load(path, expected):
        return json.loads(check_file(path, expected, receipt).read_bytes())

    prior = load(args.inputs / f"prior/runs/本地开发-01/w-{seed}.json", WEIGHTS[i])
    probe = load(exp / f"03-R03/results/parameters-{seed}.json", PROBES[i])
    group = load(exp / f"06-R06/results/parameters-{seed}.json", GROUPS[i])
    require(prior["seed"] == seed and prior["w"] == probe["w"] == group["w"], "Original weight alias mismatch")
    for history, expected in [(prior["training"], 8000), (group["histories"]["one"], 5992),
                              (group["histories"]["four"], 5992)]:
        require([r["epoch"] for r in history] == list(range(1, 13)) and
                all(r["examples"] == expected for r in history), "Frozen training budget mismatch")
    st = dict(w=vector(prior["w"]), probes=probe["probes"], one=vector(group["one"]), four=vector(group["four"]))
    if args.version == "r12":
        ck = load(exp / f"10-R12/results/checkpoint-{seed}.json", SHARP[i])
        r08 = load(exp / f"08-R08/results/parameters-{seed}.json", CUE_STATES[i])
        require((ck["model"], ck["decoder"], ck["seed"]) == ("sharp_role", "max-relation-mass-uniform-v1", seed)
                and ck["w"] == r08["weights"]["cue_role"] and list(ck["relations"]) == RELATIONS,
                "R12 checkpoint/lexicon mismatch")
        st.update(ck=ck, role_w=vector(ck["w"]), random_label_w=vector(r08["weights"]["cue_random_role"]))
    return st


def load_cohort(path):
    read = gzip.open if path.suffix == ".gz" else open
    with read(path, "rt", encoding="utf8") as stream:
        rows = [json.loads(line) for line in stream]
    keys = {(r["graph_id"], r["view"]) for r in rows}
    require(rows and len(rows) == len(keys), "Repeated graph/view key")
    allowed = {(v,k) for v in VIEWS for k in STRATA}
    valid = set(RELATIONS)
    for r in rows:
        require((r["view"], r["stratum"]) in allowed and len(r["nodes"]) == 12
                and len(set(map(tuple,r["nodes"]))) == 12 and len(set(r["candidates"])) == len(r["candidates"]) == 2
                and all(type(j) is int and 0 <= j < 12 for j in [r["gold"], *r["candidates"]]),
                f"Malformed development row {r['graph_id']}/{r['view']}")
        mentioned = [t for t in r["question"] if t in valid]
        require(len(set(mentioned)) == 2 and r["first_mentioned"] == mentioned[0]
                and r["first_relation"] in mentioned and r["nodes"][r["gold"]][1] == r["first_relation"],
                "Invalid relation labels")
    return rows


def uniform_log(mask):
    return torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf).log_softmax(-1)


def masks_for(docs, width):
    pairs = [sorted(set(d.question).intersection(RELATIONS)) for d in docs]
    masks = torch.zeros(len(docs), 2, width, dtype=torch.bool)
    for i, (d, pair) in enumerate(zip(docs, pairs, strict=True)):
        for j, t in enumerate(d.question):
            if t in pair:
                masks[i,pair.index(t),j] = True
    return pairs, masks


def norm_matched_random(w, seed):
    v = torch.randn(w.shape, generator=torch.Generator().manual_seed(seed + 442))
    return v * (w.norm() / v.norm())


def permute_restricted(logp, mask, docs, seed):
    changed = logp.clone()
    for i, d in enumerate(docs):
        positions = mask[i].nonzero().flatten()
        key = f"{seed}|{d.doc_id}|{' '.join(d.question)}".encode("utf8")
        rng = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        changed[i, positions] = logp[i, positions[torch.randperm(len(positions), generator=rng)]]
    return changed


def masses(logp, masks):
    return (logp.exp().double()[:, None, :] * masks).sum(-1)


def hard_from_log(logp, masks):
    group = logp[:,None,:].masked_fill(~masks, -torch.inf).logsumexp(-1)
    return uniform_log(masks[torch.arange(len(logp)), group.argmax(1)])


@torch.inference_mode()
def score_batch(version, bundle, st, docs, seed, model, core, probe, sharp):
    require(all(d.gold is None for d in docs), "Labels in inference document")
    f = model.extract(bundle, docs, batch_size=64 if version == "original" else 128)
    pairs, masks = masks_for(docs, f.mask.shape[1])
    mentioned = masks.any(1)
    first_positions = torch.tensor([pair.index(next(t for t in d.question if t in pair))
                                    for pair,d in zip(pairs,docs,strict=True)])
    first_p = uniform_log(masks[torch.arange(len(docs)),first_positions])
    uniform = uniform_log(f.mask)
    mention_uniform = uniform_log(mentioned)
    # (first-hop logits, question-only relation masses, effective H2 weights)
    arms = {}

    def weighted(name, h1_weights, h2_weights=None, base=None, amplitude=None):
        actual = h1_weights if h2_weights is None else h2_weights
        arms[name] = (core.apply_logp(f,actual,base,amplitude), masses(h1_weights,masks), actual)

    weighted("uniform", uniform)
    weighted("uniform_mentions", mention_uniform)
    weighted("first_mentioned", first_p)
    arms.update(first=(None,None,None), pool=(f.pooled,None,None), direction=(f.base,None,None), old=(f.old,None,None))
    one_base, one_amp = core.one_direction(bundle,docs,f)
    weighted("one", core.weights(f,st["one"]), base=one_base, amplitude=one_amp)
    weighted("four", core.weights(f,st["four"]))
    representations = probe.representation(bundle,f)
    name_index = torch.tensor([[RELATIONS.index(t) for t in pair] for pair in pairs])
    for tag, matrix in st["probes"].items():
        mode = tag.split("_random_labels")[0]
        read = representations[mode]
        projected = torch.cat([read,torch.ones(len(read),1)], dim=1) @ torch.tensor(matrix,dtype=torch.float32)
        # Here the two values are conditional probe probabilities, separate from the H1 token-mass endpoint.
        pair_probability = projected.gather(1,name_index).double().softmax(-1)
        name = "probe_" + mode.removeprefix("read_") + ("_shuffled_labels" if tag.endswith("random_labels") else "")
        arms[name] = (None,pair_probability,None)
    if version == "original":
        learned = core.weights(f, st["w"])
        weighted("learned", learned)
        weighted("random", core.weights(f, norm_matched_random(st["w"], seed)))
        weighted("shuffled", core.shuffled_logp(learned, f.mask, docs, seed))
        canonical = model.score(f, st["w"])
        zero_score = model.score(f, torch.zeros(64))
        require((zero_score - arms["uniform"][0]).abs().max() <= TOLERANCE, "Uniform formula mismatch")
        require((canonical - arms["learned"][0]).abs().max() <= TOLERANCE, "Original formula mismatch")
        # Use the exact original factorization on both sides of the primary comparison.
        arms["learned"] = (canonical, arms["learned"][1], learned)
        arms["uniform"] = (zero_score, arms["uniform"][1], uniform)
        primary_p = learned
    else:
        valid, x, classes, names = sharp.features(docs,f.mask.shape[1],st["ck"])
        require(names == pairs and torch.equal(classes,masks) and torch.equal(valid,mentioned), "R12 token support mismatch")
        soft, hard, role_choice, _ = sharp.decode(x,valid,classes,st["role_w"])
        weighted("sharp_role",soft,hard)
        weighted("soft_role",soft)
        weighted("uniform_mentions_hard",mention_uniform,hard_from_log(mention_uniform,masks))
        for name, w in [("random",norm_matched_random(st["role_w"],seed)), ("random_role_labels",st["random_label_w"])]:
            random_soft, random_hard, _, _ = sharp.decode(x,valid,classes,w)
            weighted(name,random_soft,random_hard)
        shuffled = permute_restricted(soft,valid,docs,seed)
        weighted("shuffled",shuffled,hard_from_log(shuffled,masks))
        effective_choice = [p.index(d.question[int(j)]) for p,d,j in
                            zip(pairs,docs,hard.argmax(-1),strict=True)]
        require(role_choice.tolist() == effective_choice, "Hard relation decoder inconsistent with its weights")
        require(torch.equal(sharp.rank(f,hard),arms["sharp_role"][0]), "R12 H2 implementation mismatch")
        primary_p = soft
    for name, (s, hm, _) in arms.items():
        if s is not None:
            require(s.isfinite().all(), f"Nonfinite logits in {name}")
        if hm is not None:
            require(hm.isfinite().all(), f"Nonfinite H1 readout in {name}")
    closed = ~f.active
    for name in (VERSIONS[version],"uniform","uniform_mentions","random","shuffled","old"):
        require(torch.equal(arms[name][0][closed], f.base[closed]), f"Closed gate discrepancy in {name}")
    return arms, pairs, f.active.tolist(), primary_p


def save_rows(stream, version, rows, docs, seed, arm_data):
    arms, pairs, gates, primary_logp = arm_data
    hits = Counter()
    for i,(d,r) in enumerate(zip(docs,rows,strict=True)):
        def selected(name):
            return d.candidates[int(arms[name][0][i].argmax())]

        common = dict(version=version, graph_id=r["graph_id"], graph_index=r["graph_index"], stratum=r["stratum"],
                      view=r["view"], seed=seed, gold=r["gold"], candidates=list(d.candidates), covered=r["gold"] in d.candidates,
                      gate=gates[i], first_relation=r["first_relation"], first_mentioned=r["first_mentioned"],
                      relation_names=pairs[i], uniform_predicted=selected("uniform"), first_predicted=d.candidates[0],
                      pool_predicted=selected("pool"))
        for name,(s,hm,eff) in arms.items():
            values = hm[i].tolist() if hm is not None else None
            read = None
            h1_hit = None
            if values is not None:
                if values[0] != values[1]:
                    read = pairs[i][int(values[1] > values[0])]
                h1_hit = .5 if read is None else float(read == r["first_relation"])
            predicted = selected(name) if s is not None else (d.candidates[0] if name == "first" else None)
            record = dict(**common, variant=name, relation_masses=values, selected_relation=read, h1_hit=h1_hit,
                          h1_readout_kind="probe_probability" if name.startswith("probe_") else (
                              "summed_token_weight" if hm is not None else None),
                          logits=s[i].tolist() if s is not None else None, predicted=predicted)
            if name == VERSIONS[version]:
                record.update(question_weights=primary_logp[i,:len(d.question)].exp().tolist(),
                              effective_h2_weights=eff[i,:len(d.question)].exp().tolist())
                hits["primary_rows"] += 1
                hits["primary_hits"] += int(predicted == r["gold"])
            stream.write(json.dumps(record,ensure_ascii=False,separators=(",",":"),allow_nan=False) + "\n")
            hits["prediction_rows"] += 1
    return hits


def main(a):
    a.r3 = a.r3.resolve()
    a.inputs = a.inputs.resolve()
    event(a.out,"start",version=a.version)
    files = {}
    check_file(a.inputs / "prior/code/model.py","f85636ffa19443880e48d953c7d552af3d6bda5238c60ad424f445202c2c2b0f",files)
    check_file(a.inputs / "prior/实验协议.json","22c42d26fe59b8bc2a16c13ed5b5a7a9b737263b3dc02431d2f21a6e432904ce",files)
    probe_src = a.r3 / "experiments/03-R03/snapshot/src"
    check_file(probe_src / "mech_core.py","8a2ae2cf12883285ca2c42499210ba406cb5bdbe598964f6c192d4d5859f1795",files)
    check_file(probe_src / "mech_probe.py","514a3bd1d7c586ae6c70e296efc588439a6f00ce4330f41be6e72dd3aff3ac23",files)
    sys.path[:0] = [str(probe_src),str(a.inputs/"prior/code")]
    import mech_core as core
    import mech_probe as probe
    import model

    sharp = None
    if a.version == "r12":
        src = a.r3 / "experiments/10-R12/snapshot/src"
        check_file(src/"sharp_core.py","f56c15184956476784b4922b453d4898178412e4c709f0f5dc632ab5dba86c54",files)
        check_file(src/"role_core.py","713909ab7c417c2b7f950938e291498c51651adfef3922a169067749a509ab8f",files)
        sys.path.insert(0,str(src))
        import sharp_core as sharp
    rows = load_cohort(a.cohort)
    data_hash = sha(a.cohort)
    code_hash = sha(Path(__file__))
    event(a.out,"inputs_verified",cohort_sha256=data_hash,rows=len(rows),graphs=len({r["graph_id"] for r in rows}))
    receipts = []
    for seed in SEEDS:
        event(a.out,"seed_started",seed=seed)
        seed_files = {}
        st = load_state(a,seed,seed_files)
        bundle = model.load_bundle(a.inputs/"b13",seed)
        for kind in ("serial","pooled"):
            folder = bundle.evaluator.checkpoint_dir(a.inputs/"b13/inputs/server-run",kind,seed)
            check_file(folder/"best.pt",bundle.evaluator.CHECKPOINTS[kind][seed],seed_files)
        output = a.out/f"predictions-{seed}.jsonl.gz"
        temp = a.out/f"predictions-{seed}.partial.gz"
        collected = Counter()
        with gzip.open(temp,"wt",encoding="utf8") as stream:
            for start in range(0,len(rows),128):
                batch = rows[start:start+128]
                docs = [bundle.trainer.Example(r["graph_id"],r["stratum"],tuple(r["question"]),tuple(map(tuple,r["nodes"])),
                                              tuple(r["candidates"]),None) for r in batch]
                reranked = bundle.evaluator.shortlist(docs,"direction")
                require(all(d.candidates == e.candidates for d,e in zip(docs,reranked,strict=True)),
                        "Supplied candidates differ from historical direction shortlist")
                calculated = score_batch(a.version,bundle,st,docs,seed,model,core,probe,sharp)
                collected.update(save_rows(stream,a.version,batch,docs,seed,calculated))
        temp.replace(output)
        receipt = dict(version=a.version,seed=seed,cohort_sha256=data_hash,script_sha256=code_hash,
                       frozen_files_sha256=seed_files,counts=collected,predictions_sha256=sha(output))
        json_file(a.out/f"receipt-{seed}.json",receipt)
        receipts.append(receipt)
        event(a.out,"seed_saved",seed=seed,**collected)
    combined = a.out/"predictions.jsonl.gz"
    pending = a.out/"combined.partial.gz"
    # Gzip supports concatenated members and preserves exact successful per-seed streams.
    with pending.open("wb") as dst:
        for seed in SEEDS:
            with (a.out/f"predictions-{seed}.jsonl.gz").open("rb") as src:
                shutil.copyfileobj(src,dst)
    pending.replace(combined)
    metadata = dict(version=a.version,primary_variant=VERSIONS[a.version],seeds=SEEDS,
                    cohort_sha256=data_hash,source_sha256=files,script_sha256=code_hash,receipts=receipts,
                    predictions_sha256=sha(combined),historic_training_unchanged=True,new_training_updates=0,
                    primary_uniform="All valid question positions, identical candidate scorer and H2 gate.",
                    restricted_uniform="uniform_mentions assigns equal weight to each visible relation-token occurrence.",
                    r12_h1="Sum soft question-only relation masses before hard decoding. Exact ties receive half credit.",
                    r12_h2="Choose highest mass relation, Unicode lexical tie break, uniform weight on its mentions.",
                    original_h1="Softmax of frozen h_j dot w over every valid token; sum by mentioned relation.",
                    original_h2="Frozen direction plus gated weighted relation logmeanexp; archived L2 disabled.",
                    random_controls="Random normal vector with learned norm. Original shuffles over all valid positions; R12 shuffles only relation-mention positions and redecodes with its frozen hard policy.",
                    matched_groups="one/four are paired 64-parameter rankers trained on identical filtered rows and 5,992 eligible slots/epoch for 12 epochs. They are a separate controlled comparison.",
                    probe_groups="Same 9,007 training graphs and ridge-1 readout. One vector versus 16 coordinates from each of four readouts gives 64 dimensions for both. Random-label controls share the original training-label permutation.",
                    labels="Used only for cohort checks and post-inference hit counting. Supplied feature-only Example.gold is None.",
                    views=dict(Counter(r["view"] for r in rows)),python=platform.python_version(),torch=str(torch.__version__))
    json_file(a.out/"inference-receipt.json",metadata)
    event(a.out,"complete",predictions_sha256=metadata["predictions_sha256"])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version",choices=VERSIONS,required=True)
    for label in ("r3","inputs","cohort","out"):
        p.add_argument("--"+label,type=Path,required=True)
    args = p.parse_args()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error("--out requires a new or empty directory")
    args.out.mkdir(parents=True,exist_ok=True)
    try:
        main(args)
    except Exception:
        json_file(args.out/"failure.json",{"status":"incomplete","traceback":traceback.format_exc()})
        event(args.out,"failed")
        raise
