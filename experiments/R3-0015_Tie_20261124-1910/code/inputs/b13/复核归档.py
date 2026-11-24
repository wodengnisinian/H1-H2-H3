"""Check recorded B13 artifacts without invoking the models or opening a new assessment."""
from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_bytes())


def digest(name):
    with (ROOT / name).open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def check():
    freeze = read("冻结清单.json")
    for n, expected in freeze["sha256"].items():
        assert digest(n) == expected, n
    cohort = read("cohort/seal.json")
    pred = read("predictions/prediction-seal.json")
    opening = read("assessment/opening.json")
    receipt = read("assessment/assessment-receipt.json")
    result = read("assessment/score.json")
    terminal = read("ledger/terminal.json")
    start = read("ledger/started.json")
    timestamps = [freeze["frozen_utc"], start["started_utc"], read("cohort/entropy.json")["created_utc"],
                  cohort["sealed_utc"], pred["started_utc"], pred["sealed_utc"], opening["opened_utc"],
                  receipt["finished_utc"], terminal["finished_utc"]]
    times = [datetime.fromisoformat(t) for t in timestamps]
    assert all(x <= y for x, y in zip(times, times[1:])) and times[-1].timestamp() <= start["deadline_unix"]
    assert len(list((ROOT / "assessment").glob("opening*.json"))) == 1
    assert opening["attempt"] == opening["maximum_attempts"] == receipt["attempt"] == terminal["formal_openings"] == 1
    for item in (cohort, pred, opening, receipt):
        assert item["protocol_sha256"] == digest("实验协议.json")
        assert item["freeze_sha256"] == digest("冻结清单.json")
    assert digest("cohort/seal.json") == pred["cohort_seal_sha256"] == opening["cohort_seal_sha256"] == receipt["cohort_seal_sha256"]
    assert digest("cohort/features.jsonl") == cohort["features_sha256"] == pred["features_sha256"]
    assert digest("cohort/gold.jsonl") == cohort["gold_sha256"] == opening["gold_sha256"] == receipt["gold_sha256"]
    assert digest("cohort/entropy.json") == cohort["entropy_sha256"]
    assert digest("predictions/prediction-seal.json") == opening["prediction_seal_sha256"] == receipt["prediction_seal_sha256"]
    assert digest("assessment/opening.json") == receipt["opening_sha256"]
    assert digest("assessment/score.json") == receipt["score_sha256"] == terminal["score_sha256"]
    assert len(pred["prediction_sha256"]) == 7 and pred["prediction_sha256"] == opening["prediction_sha256"] == receipt["prediction_sha256"]
    for name, checksum in pred["prediction_sha256"].items():
        assert digest("predictions/" + name) == checksum, name
    assert pred["mode"] == "direction" and pred["documents"] == cohort["documents"] == 3000
    assert cohort["distinct_person_sets"] == 3000 and cohort["person_preimage_range"][0] == 16800
    assert set(cohort["strata_counts"].values()) == {1000}
    assert all(len(v) == 8 and set(v.values()) == {125} for v in cohort["template_counts_by_stratum"].values())
    assert receipt["returncode"] == 0 and terminal["study_completed"]
    assert all(read(f"ledger/{stage}/001/finish.json")["returncode"] == 0 for stage in ("generation", "prediction", "assessment"))
    assert all(len(list((ROOT / "ledger" / stage).glob("*/start.json"))) == 1 for stage in ("generation", "prediction", "assessment"))
    assert result["input_sha256"]["gold"] == receipt["canonical_scoring_labels_sha256"]
    assert result["input_sha256"]["lexical"] == pred["prediction_sha256"]["direct.jsonl"]
    for k, name in (("candidate", "serial"), ("pooled", "pooled")):
        assert all(result["input_sha256"][k][str(s)] == pred["prediction_sha256"][f"{name}-{s}.jsonl"] for s in result["seeds"])
    assert result["documents"] == 3000 and result["seeds"] == pred["seeds"] == [1701, 2701, 3701]
    assert result["minimum_point_gain"] == .05 and result["bootstrap"]["replicates"] == 10000
    assert result["bootstrap"]["seed"] == 619482 and result["bootstrap"]["lower_rank"] == 500
    hits = {k: round(v*9000) for k,v in result["accuracy"].items()}
    passes = []
    shortfalls = {}
    for key in ("lexical","pooled"):
        c = result["contrasts"]["candidate_minus_"+key]
        net = hits["candidate"]-hits[key]
        assert round(c["effect"]*9000)==net
        assert c["point_passed"] == (net >= 450) and c["interval_passed"] == (c["lower_95"] > 0)
        passes.append(c["point_passed"] and c["interval_passed"])
        shortfalls[key] = max(0,450-net)
    assert result["passed"] == all(passes) and receipt["status"] == ("passed" if all(passes) else "failed")
    return {"scope":"existing files and aggregate arithmetic; no rescore",
            "frozen_files":len(freeze["sha256"]), "sealed_predictions":len(pred["prediction_sha256"]),
            "documents":3000, "attempt":1, "hits_per_9000":hits, "point_shortfalls":shortfalls,
            "status":receipt["status"],"score_sha256":terminal["score_sha256"],"utc_sequence":timestamps}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
