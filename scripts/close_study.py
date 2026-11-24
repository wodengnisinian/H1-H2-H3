"""Close this bounded study after required experiments and one scored independent cohort."""
from pathlib import Path
import hashlib,json,shutil,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import run_loop as loop
reg=loop.events()
fin=[r for r in reg if r["event"]=="end" and r["code"]==0]
byid={r["id"]:r for r in fin}
assert set(byid)=={f"R{i:02}" for i in range(1,14)}
for row in fin:loop.verify_finished(root/row["path"])
p=root/byid["R09"]["path"]/"results"
check=json.loads((p/"metrics.json").read_bytes())
comp=check["paired_comparisons"]["original"]
assert comp["both_thresholds_met"] and check["h3"]["reference_panel_three_fixed_seeds_and_three_kinds"]
assert len(list((root/"experiments").rglob("results/opening.json")))==1
aud=loop.read(root/"analysis/正式验收复核.json")
assert aud["status"]=="verified" and aud["authoritative_metrics_sha256"]==loop.sha(p/"metrics.json")
selected=root/byid["R12"]["path"]/"results"
frozen=loop.read(p/"freeze.json")
for seed in (1701,2701,3701):
    source=selected/f"checkpoint-{seed}.json";target=root/"checkpoints"/source.name
    assert loop.sha(source)==frozen["sha256"][str(source)]
    if target.exists():
        assert loop.sha(source)==loop.sha(target)
    else:shutil.copyfile(source,target)
decision=dict(stage="completed",reason="完成预设开发、一次性独立H3和隔离的两跳及语义范围挑战。冻结输入在方案状态诊断上存在已证实的不可区分性。",
            used_formal_openings=1,max_formal_openings=1,h3_passed=True,model=frozen["module"],graphs=comp["graphs"],
            original_accuracy=check["question_views"]["original"]["overall"]["accuracy"]["learned"],contrasts=comp["contrasts_learned_minus"],
            freeze_sha256=loop.sha(p/"freeze.json"),prediction_seal_sha256=loop.sha(p/"prediction-seal.json"),
            scoring_sha256=loop.sha(p/"metrics.json"),opened_utc=loop.read(p/"opening.json")["utc"],
            completed_experiments=[r["id"] for r in fin],all_attempts=len([r for r in reg if r["event"]=="start"]),
            numerical_cpu_seconds=sum(r["cpu_seconds"] for r in fin),wall_seconds_summed=sum(r["wall_seconds"] for r in fin))
loop.save(root/"analysis/H3验收判定.json",decision)
state=loop.read(loop.STATE)
state.update(stage="completed",stop_reason=decision["reason"],checkpoints="checkpoints/checkpoint-{1701,2701,3701}.json",
             decision="analysis/H3验收判定.json",next_task="本轮封存。后续范围扩展见 next_experiment.md。",budget=loop.budget(),updated=loop.now())
loop.save(loop.STATE,state)
loop.logmsg("全部 13 项已归档。正式模型 sharp_role 符合事前 H3 双基线标准，正式队列已开封一次。")
print(json.dumps(decision,ensure_ascii=False))
