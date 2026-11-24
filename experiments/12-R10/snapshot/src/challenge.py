"""Frozen two-hop challenge. Usage: python challenge.py --config challenge.json --out NEW_DIR."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import random
import shutil

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPLITS = ("train", "dev", "seen_pair_test", "challenge")
MODELS = ("first", "onehop", "twohop", "symbolic")
VIEWS = ("AB-normal", "BA-normal", "AB-reordered", "BA-reordered")
FEATURE_FIELDS = {"question", "facts"}


def check(test, detail):
    if not test:
        raise ValueError(detail)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def line(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_log(out, event, **details):
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line(row))
    print(line(row), flush=True, end="")


def relation_pairs(config):
    relations = config["relations"]
    check(len(relations) == len(set(relations)) == 8, "Eight distinct relations required")
    splits = {}
    for split, distances in config["pair_split_ring_distances"].items():
        splits[split] = [(relations[i], relations[j]) for i, j in combinations(range(8), 2)
                         if min(j - i, 8 - (j - i)) in distances]
    tr, dv, ch = (set(splits[k]) for k in ("train", "dev", "challenge"))
    check(not (tr & dv or tr & ch or dv & ch) and len(tr | dv | ch) == 28, "Ordered/reversed pair leak")
    check(set(splits["seen_pair_test"]) == tr, "Seen-pair reference mismatch")
    for pairs in (tr, dv, ch):
        check({r for pair in pairs for r in pair} == set(relations), "Single relation OOV")
    return splits


def question(anchor, first, second, style, reverse_clause):
    a, b = (s.format(r=r) for s, r in zip(style, (first, second), strict=True))
    first_clause, second_clause = (b, a) if reverse_clause else (a, b)
    return f"从 {anchor} 出发 。 {first_clause} 。 {second_clause} 。 选择 首条 记录 。".split()


def make_twins(split, group_number, pair, pair_inventory, rng):
    """Flip only the assignment of two pairs of continuation labels; keep the whole fact order fixed."""
    nodes = [f"R3NODE_{split}_{group_number:04d}_{i:02d}" for i in range(13)]
    rng.shuffle(nodes)
    anchor, m0, m1, m2, m3, t0, t1, t2, t3, d0, d1, d2, d3 = nodes
    a, b = pair
    neighbors = {r: sorted({other for p in pair_inventory if r in p for other in p if other not in pair})
                 for r in pair}
    c, d = rng.choice(neighbors[a]), rng.choice(neighbors[b])
    mids, tails = [m0, m1, m2, m3], [t0, t1, t2, t3]
    prefix = [[anchor, rel, mid] for rel, mid in zip((a, a, b, b), mids, strict=True)]
    # Include the two queried relations on irrelevant edges too.
    extras = [[d0, a, d1], [d0, b, d2], [d0, c, d3], [d0, d, d1]]
    permutation = rng.sample(range(12), 12)
    true_a, true_b = rng.randrange(2), rng.randrange(2)
    twins = []
    for twin in range(2):
        ia, ib = true_a ^ twin, true_b ^ twin
        branch_relations = [c, c, d, d]
        branch_relations[ia], branch_relations[2 + ib] = b, a
        continuations = [[m, r, t] for m, r, t in zip(mids, branch_relations, tails, strict=True)]
        facts = prefix + continuations + extras
        twins.append(([facts[i] for i in permutation], [permutation.index(ia), permutation.index(2 + ib)]))
    return anchor, twins, {a: c, b: d}


def observed(features, relations, vocab):
    """Pure inference feature function; accepts only question tokens and observed fact triples."""
    check(set(features) == FEATURE_FIELDS, "Unapproved inference field")
    words, facts = features["question"], features["facts"]
    entities = {x for head, _, tail in facts for x in (head, tail)}
    anchors = entities.intersection(words)
    check(len(anchors) == 1, "Question does not name a unique anchor")
    anchor = next(iter(anchors))
    mention_pos = [i for i, w in enumerate(words) if w in relations]
    mentions = [words[i] for i in mention_pos]
    check(len(set(mentions)) == len(mentions) == 2, "Two distinct question relations required")
    candidates = [i for i, (head, _, _) in enumerate(facts) if head == anchor]
    check(len(candidates) == 4, "Four anchor-outgoing candidates required")
    clean = ["<entity>" if w in entities else "<relation>" if w in relations else w for w in words]
    contexts = [[vocab[clean[j]] if 0 <= j < len(clean) else vocab["<pad>"]
                 for j in range(i - 2, i + 3)] for i in mention_pos]
    first_match, successor = [], []
    for ci in candidates:
        _, relation, mid = facts[ci]
        first_match.append([int(relation == mention) for mention in mentions])
        reachable = {r for head, r, tail in facts if head == mid and len({anchor, mid, tail}) == 3}
        successor.append([int(mention in reachable) for mention in mentions])
    return candidates, contexts, first_match, successor


def symbolic_choice(features, candidates, relations):
    """Read the order markers from the visible question, then enumerate a unique legal directed path."""
    words, facts = features["question"], features["facts"]
    roles = {}
    for pos, word in enumerate(words):
        if word in relations:
            marker = words[pos - 2]
            key = {"先": 1, "第一步": 1, "再": 2, "第二步": 2}[marker]
            check(key not in roles, "Ambiguous order marker")
            roles[key] = word
    valid = [k for k, i in enumerate(candidates) for head, rel, tail in facts
             if facts[i][1] == roles[1] and facts[i][2] == head and rel == roles[2]
             and len({facts[i][0], head, tail}) == 3]
    check(len(valid) == 1, "Full graph does not uniquely resolve the question")
    return valid[0]


def vocabulary(config):
    words = {"<entity>", "<relation>", "<pad>", "从", "出发", "。", "选择", "首条", "记录"}
    for style in config["styles"]["train"]:
        for phrase in style:
            words.update(tok for tok in phrase.split() if tok != "{r}")
    return {word: i for i, word in enumerate(sorted(words))}


def make_data(config, out):
    pairs = relation_pairs(config)
    vocab = vocabulary(config)
    rels = set(config["relations"])
    style_sets = {}
    for k, styles in config["styles"].items():
        style_sets[k] = {tuple(s) for s in styles}
    check(not (style_sets["train"] & style_sets["dev"] or style_sets["train"] & style_sets["challenge"]
               or style_sets["dev"] & style_sets["challenge"]), "Whole query template leak")
    data, manifest = {}, {}
    alternate = []
    for split_i, split in enumerate(SPLITS):
        rng = random.Random(config["generator_seed"] + split_i)
        style_key = "challenge" if split == "seen_pair_test" else split
        groups_per_pair = config["twin_groups_per_pair"][split]
        meta, contexts, firsts, seconds, labels, oracle = [], [], [], [], [], []
        graph_keys, graph_people, answer_positions = set(), set(), Counter()
        fp, gp = out / f"{split}-features.jsonl", out / f"{split}-gold.jsonl"
        with fp.open("x", encoding="utf-8") as f, gp.open("x", encoding="utf-8") as g:
            for pair_i, pair in enumerate(pairs[split]):
                for j in range(groups_per_pair):
                    number = pair_i * groups_per_pair + j
                    group = f"{split}-{number:04d}"
                    style = config["styles"][style_key][j % len(config["styles"][style_key])]
                    anchor, twins, foil = make_twins(split, number, pair, pairs[split], rng)
                    group_people = frozenset(p for h, _, t in twins[0][0] for p in (h, t))
                    check(len(group_people) == 13 and group_people not in graph_people, "Repeated entity group")
                    graph_people.add(group_people)
                    first_twin = {}
                    for twin, (facts, target_facts) in enumerate(twins):
                        graph = f"{group}-{twin}"
                        graph_key = sha256(line(sorted(facts)).encode()).hexdigest()
                        check(graph_key not in graph_keys, "Repeated graph within split")
                        graph_keys.add(graph_key)
                        connected_pairs = {frozenset((r, s)) for h, r, mid in facts for mid2, s, t in facts
                                           if mid == mid2 and len({h, mid, t}) == 3}
                        check(connected_pairs <= {frozenset(p) for p in pairs[split]},
                              "A directed two-edge path crosses a relation-combination split")
                        bag = None
                        fixed_candidates = None
                        for view, tag in enumerate(VIEWS):
                            reverse = view % 2
                            order = pair[::-1] if reverse else pair
                            words = question(anchor, *order, style, view >= 2)
                            feature = {"question": words, "facts": facts}
                            cand, ctx, first, second = observed(feature, rels, vocab)
                            answer = cand.index(target_facts[reverse])
                            symbol = symbolic_choice(feature, cand, rels)
                            check(answer == symbol, "Generated label disagrees with independent parser/path oracle")
                            check(all(v == 2 for v in map(sum, zip(*first))), "Same-relation ambiguity lost")
                            check(fixed_candidates is None or cand == fixed_candidates, "Quartet candidate drift")
                            check(bag is None or bag == Counter(words), "Quartet word multiset drift")
                            bag, fixed_candidates = Counter(words), cand
                            same = (ctx, first, cand, words, [facts[i] for i in cand])
                            if not twin:
                                first_twin[view] = same, answer
                            else:
                                previous, previous_gold = first_twin[view]
                                check(previous == same and previous_gold != answer, "Counterfactual twin violation")
                            qid = f"{graph}-{view}"
                            info = {"id": qid, "group": group, "graph": graph, "pair": "/".join(pair),
                                    "twin": twin, "view": tag, "candidates": cand}
                            # Keys for joining metadata never enter observed() or either neural model.
                            f.write(line({"id": qid, **feature}))
                            g.write(line({"id": qid, "gold_fact": target_facts[reverse]}))
                            if split == "challenge":
                                alt_words = question(anchor, order[0], foil[order[0]], style, view >= 2)
                                check(sum(x != y for x, y in zip(words, alt_words, strict=True)) == 1,
                                      "Second-relation intervention changed more than one token")
                                alt = {"question": alt_words, "facts": facts}
                                ca, cx, fi, se = observed(alt, rels, vocab)
                                alt_ans = symbolic_choice(alt, ca, rels)
                                same_first = [k for k, idx in enumerate(cand) if facts[idx][1] == order[0]]
                                check(ca == cand and sorted((alt_ans, answer)) == same_first,
                                      "Second-only target does not switch within the original first-relation pair")
                                alt_info = {**info, "id": qid + "-second", "base_id": qid}
                                alternate.append((alt_info, alt, cx, fi, se, alt_ans))
                            meta.append(info)
                            contexts.append(ctx)
                            firsts.append(first)
                            seconds.append(second)
                            labels.append(answer)
                            oracle.append(symbol)
                            answer_positions[cand[answer]] += 1
        data[split] = {"inputs": (torch.tensor(contexts), torch.tensor(firsts, dtype=torch.float32),
                                  torch.tensor(seconds, dtype=torch.float32)), "labels": torch.tensor(labels),
                       "meta": meta, "oracle": np.asarray(oracle), "groups_per_pair": groups_per_pair}
        manifest[split] = {"graphs": len(graph_keys), "groups": len(graph_people), "questions": len(labels),
                           "relation_pairs": pairs[split], "template_family": style_key,
                           "gold_fact_position_counts": dict(answer_positions),
                           "features_sha256": digest(fp), "gold_sha256": digest(gp),
                           "canonical_graph_set_sha256": sha256(line(sorted(graph_keys)).encode()).hexdigest(),
                           "namespace": f"R3NODE_{split}_"}
    af, ag = out / "challenge_second-features.jsonl", out / "challenge_second-gold.jsonl"
    with af.open("x", encoding="utf-8") as f, ag.open("x", encoding="utf-8") as g:
        for info, features, _, _, _, y in alternate:
            f.write(line({"id": info["id"], **features}))
            g.write(line({"id": info["id"], "gold_fact": info["candidates"][y]}))
    info, _, cx, fi, se, alt_labels = zip(*alternate)
    data["challenge_second"] = {"inputs": (torch.tensor(cx), torch.tensor(fi, dtype=torch.float32),
                                          torch.tensor(se, dtype=torch.float32)), "labels": torch.tensor(alt_labels),
                                "meta": info, "oracle": np.asarray(alt_labels)}
    manifest["challenge_second"] = {"graphs": manifest["challenge"]["graphs"], "questions": len(info),
                                    "paired_to": "challenge", "features_sha256": digest(af), "gold_sha256": digest(ag)}
    return data, vocab, manifest


class Readout(nn.Module):
    def __init__(self, vocabulary_size, dim, use_next):
        super().__init__()
        self.use_next = use_next
        self.embedding = nn.Embedding(vocabulary_size, dim)
        self.roles = nn.Linear(5 * dim, 2)
        self.log_scales = nn.Parameter(torch.zeros(2))

    def forward(self, context, first, second):
        # In one-hop mode, second is never read. Directly tested against twin equality at prediction time.
        encoding = self.embedding(context).flatten(2)
        weights = self.roles(encoding).softmax(dim=1)
        scale = F.softplus(self.log_scales)
        scores = scale[0] * (first * weights[:, None, :, 0]).sum(-1)
        if self.use_next:
            scores = scores + scale[1] * (second * weights[:, None, :, 1]).sum(-1)
        return scores


def train_predict(name, seed, data, vocab, config, device, out):
    opt = config["training"]
    torch.manual_seed(seed)
    model = Readout(len(vocab), opt["embedding_dim"], name == "twohop").to(device)
    optim = torch.optim.Adam(model.parameters(), lr=opt["learning_rate"])
    batch = opt["batch_size"]
    train = data["train"]
    features = tuple(t.to(device) for t in train["inputs"])
    labels = train["labels"].to(device)
    shuffle = torch.Generator().manual_seed(seed)
    trace = []
    for epoch in range(1, opt["epochs"] + 1):
        indices = torch.randperm(len(labels), generator=shuffle)
        total_loss = 0.0
        hits = 0
        for ids in indices.split(batch):
            ids = ids.to(device)
            optim.zero_grad(set_to_none=True)
            scores = model(*(t[ids] for t in features))
            loss = F.cross_entropy(scores, labels[ids])
            loss.backward()
            optim.step()
            total_loss += loss.detach().item() * len(ids)
            hits += (scores.detach().argmax(-1) == labels[ids]).sum().item()
        record = {"seed": seed, "model": name, "epoch": epoch, "mean_cross_entropy": total_loss / len(labels),
                  "training_online_top1": hits / len(labels)}
        trace.append(record)
        with (out / "losses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line(record))
    torch.save({"model": model.cpu().state_dict(), "name": name, "seed": seed, "epoch": opt["epochs"],
                "vocabulary": vocab, "embedding_dim": opt["embedding_dim"]}, out / f"{name}-{seed}.pt")
    model.to(device).eval()
    predicted = {}
    with torch.inference_mode():
        for split, rows in data.items():
            if split == "train":
                continue
            xs, ys = rows["inputs"], rows["labels"].numpy()
            logits = torch.cat([model(*(t[i:i+batch].to(device) for t in xs)).cpu()
                                for i in range(0, len(ys), batch)]).numpy()
            preds = logits.argmax(axis=1)
            if name == "onehop":
                paired = logits.reshape(-1, 2, 4, 4)
                check(np.allclose(paired[:, 0], paired[:, 1], rtol=1e-6, atol=1e-6)
                      and np.array_equal(paired[:, 0].argmax(-1), paired[:, 1].argmax(-1)),
                      f"One-hop predictions differ between twin graphs: {split}")
            predicted[split] = preds
    return predicted


def summarize(split, rows, predictions, config, out):
    seeds = config["model_seeds"]
    metadata, labels = rows["meta"], rows["labels"].numpy()
    per_seed, errors = {}, Counter()
    hits = {method: [] for method in MODELS}
    with (out / f"{split}-predictions.jsonl").open("x", encoding="utf-8") as pred_file, \
         (out / f"{split}-errors.jsonl").open("x", encoding="utf-8") as error_file:
        for seed in seeds:
            choices = {"first": np.zeros_like(labels), "symbolic": rows["oracle"],
                       **{m: predictions[(m, seed)][split] for m in ("onehop", "twohop")}}
            current = {m: (choices[m] == labels) for m in MODELS}
            for m, h in current.items():
                hits[m].append(h.reshape(-1, 2, 4))
            per_seed[str(seed)] = {m: float(h.mean()) for m, h in current.items()}
            for i, info in enumerate(metadata):
                ci = info["candidates"]
                failed = [m for m in MODELS if not current[m][i]]
                prediction = {**info, "seed": seed, "gold_fact": ci[labels[i]],
                              "prediction_fact": {m: ci[choices[m][i]] for m in MODELS}}
                pred_file.write(line(prediction))
                if failed:
                    error_file.write(line({**prediction, "wrong_models": failed}))
                    errors.update(failed)
    success = config["success"]
    result = {"split": split, "per_seed_top1": per_seed, "questions_per_seed": len(labels),
              "error_judgments": dict(errors), "methods": {}, "gain_vs_onehop": {}}
    stacked = {m: np.stack(h) for m, h in hits.items()}  # seed × twin groups × twin × four views
    for m, h in stacked.items():
        result["methods"][m] = {
            "top1": float(h.mean()), "four_question_accuracy": float(h.all(axis=3).mean()),
            "semantic_swap_both_correct": float((h[..., ::2] & h[..., 1::2]).mean()),
            "both_clause_orders_correct": float((h[..., :2] & h[..., 2:]).mean()),
            "both_counterfactual_twins_correct": float(h.all(axis=2).mean()),
            "by_view": {k: float(h[..., i].mean()) for i, k in enumerate(VIEWS)},
        }
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    # Resample whole twin groups within each unordered relation pair; reverse queries never split.
    delta = (stacked["twohop"].astype(float) - stacked["onehop"]).mean(axis=(0, 2, 3))
    groups = rows["groups_per_pair"]
    strata = delta.reshape(-1, groups)
    boot = np.zeros(config["bootstrap"]["replicates"])
    for values in strata:
        draw = rng.integers(0, groups, size=(len(boot), groups))
        boot += values[draw].mean(axis=1) / len(strata)
    low, high = np.percentile(boot * 100, [2.5, 97.5])
    result["gain_vs_onehop"] = {"estimate_pp": float(delta.mean()*100), "two_sided_95_ci_pp": [float(low), float(high)],
                                "replicates": len(boot), "twin_groups": len(delta),
                                "resampling_unit": "counterfactual twin graph group, stratified by unordered relation pair"}
    two, one, sym = (result["methods"][m] for m in ("twohop", "onehop", "symbolic"))
    result["bounds"] = {"uniform_candidate_expected": 0.25, "any_first_only_upper": 0.5, "full_graph_symbolic": sym["top1"]}
    check(one["top1"] <= success["onehop_ceiling"] and sym["top1"] == success["symbolic_top1_required"],
          "Theoretical control bound or oracle violation")
    result["meets_preregistered_challenge_rule"] = None if split != "challenge" else bool(
        two["top1"] >= success["challenge_top1_min"] and two["four_question_accuracy"] >= success["challenge_quartet_min"]
        and min(v["twohop"] for v in per_seed.values()) >= success["each_seed_top1_min"]
        and low > success["gain_over_onehop_ci_lower_pp_min_strict"])
    return result


def score_second_switch(data, predictions, config, out):
    base, alt = (data[k] for k in ("challenge", "challenge_second"))
    labels, targets = base["labels"].numpy(), alt["labels"].numpy()
    check(np.all(labels != targets), "Second-only pair has an unchanged answer")
    method_means = {m: [] for m in MODELS}
    with (out / "second-switch-predictions.jsonl").open("x", encoding="utf-8") as stream:
        for seed in config["model_seeds"]:
            for name in MODELS:
                if name == "symbolic":
                    av, bv = base["oracle"], alt["oracle"]
                elif name == "first":
                    av = bv = np.zeros_like(labels)
                else:
                    av, bv = (predictions[name, seed][k] for k in ("challenge", "challenge_second"))
                both = (av == labels) & (bv == targets)
                method_means[name].append({"seed": seed, "both_correct": float(both.mean()),
                                           "alternate_top1": float(np.mean(bv == targets)),
                                           "changed_prediction": float(np.mean(av != bv))})
                for i, info in enumerate(alt["meta"]):
                    cand = info["candidates"]
                    stream.write(line({"id": info["id"], "base_id": info["base_id"], "seed": seed, "model": name,
                                       "base_gold": cand[labels[i]], "new_gold": cand[targets[i]],
                                       "base_pred": cand[av[i]], "new_pred": cand[bv[i]], "both_correct": bool(both[i])}))
    return {m: {"per_seed": values, "both_correct": float(np.mean([s["both_correct"] for s in values]))}
            for m, values in method_means.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--prepare-only", action="store_true", help="Check generated datasets; skip training and scoring")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=True, parents=True)
    check(not any(out.iterdir()), f"Output directory is not empty: {out}")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    check(config["schema_version"] == 1, "Unsupported configuration")
    shutil.copyfile(args.config, out / "frozen-config.json")
    torch.set_num_threads(config["training"]["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    check(args.device != "cuda" or torch.cuda.is_available(), "Requested CUDA unavailable")
    write_log(out, "start", config_sha256=digest(args.config), code_sha256=digest(Path(__file__)),
              torch=str(torch.__version__), numpy=np.__version__, device=args.device)
    data, vocab, manifest = make_data(config, out)
    dump(out / "dataset-manifest.json", manifest)
    write_log(out, "validated", questions={k: v["questions"] for k, v in manifest.items()})
    if args.prepare_only:
        return
    predictions = {}
    for seed in config["model_seeds"]:
        for name in ("onehop", "twohop"):
            predictions[name, seed] = train_predict(name, seed, data, vocab, config, args.device, out)
            write_log(out, "checkpoint", model=name, seed=seed, sha256=digest(out / f"{name}-{seed}.pt"))
    scores = {k: summarize(k, data[k], predictions, config, out) for k in SPLITS if k != "train"}
    second = score_second_switch(data, predictions, config, out)
    scores["challenge"]["second_relation_only"] = second
    scores["challenge"]["meets_preregistered_challenge_rule"] &= (
        second["twohop"]["both_correct"] >= config["success"]["second_only_pair_both_correct_min"])
    dump(out / "scores.json", scores)
    write_log(out, "complete", challenge_pass=scores["challenge"]["meets_preregistered_challenge_rule"])
    dump(out / "sha256.json", {f.name: digest(f) for f in sorted(out.iterdir()) if f.is_file()})


if __name__ == "__main__":
    main()
