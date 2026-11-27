"""Print the R32 variant table from results/stats/metrics.json (read-only)."""
import json
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
metrics = json.loads((ROUND / "results/stats/metrics.json").read_text(encoding="utf8"))


def num(value):
    return None if value is None else round(value * 100, 2)


print(f"{'variant':20} {'H1':>7} {'H1pass':>6} {'H2':>7} {'uH1':>7} {'uH2':>7} {'4q':>6}")
for arm, v in metrics["variants"].items():
    primary = v.get("primary", {})
    unseen = v.get("unseen_template", {})
    behavior = v.get("behavior", {})
    print(f"{arm:20} {str(num(primary.get('h1_accuracy'))):>7} "
          f"{str(primary.get('h1_pass')):>6} "
          f"{str(num(primary.get('h2_accuracy'))):>7} "
          f"{str(num(unseen.get('h1_accuracy'))):>7} "
          f"{str(num(unseen.get('h2_accuracy'))):>7} "
          f"{str(num(behavior.get('quartet_all_correct'))):>6}")
print("\nprimary per-seed H1:", {k: round(v["h1_accuracy"] * 100, 2) for k, v in metrics["primary_per_seed"].items()})
print("primary per-stratum H1:", {k: round(v["h1_accuracy"] * 100, 2) for k, v in metrics["primary_per_stratum"].items()})
