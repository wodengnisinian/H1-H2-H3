"""Write the per-round report from a completed run, leaving source predictions untouched."""
import json,sys
from pathlib import Path
def pct(x):
    return "—" if x is None else f"{x*100:.2f}%"
def pp(x):return "—" if x is None else f"{x*100:+.2f}"
def passed(x):return "达到" if x else "未达到"
def generate(r):
    def j(n):return json.loads((r/n).read_bytes())
    cfg=j("configs/launch.json");co=j("data/cohort/manifest.json");m=j("results/stats/metrics.json");p=m["primary"]
    v=cfg["version"];rid=cfg["id"]; arm=p["variant"];label="论文原始共享词权重" if v=="original" else "R12 冻结关系读头与硬选择"
    b=m["variants"][arm]["behavior"];elapsed=j("logs/resources.json")["elapsed_seconds"]
    full="本轮两个开发门槛均达到。" if p["h1_pass"] and p["h2_pass"] else "本轮未通过 H1、H2 的联合开发门槛。"
    model=("每一题计算上下文向量与 64 维共享权重的内积，在全部有效词位上归一化。按两个问题关系分别累加词位权重，比较两项总质量。H2 在冻结方向分上叠加门控幅度乘以带该权重的关系匹配 log-sum-exp。复用原先从零学习的三个权重，每个种子使用 12,000 张训练图中的 8,000 个门控开启且金标入围样本，Adam 学习率 0.003，批量 128，固定 12 轮。"
           if v=="original" else "复用三个已冻结的 R12 检查点。问题读头使用关系词表和八组人工时序线索。按问题预测的关系总质量作硬选择，把有效权重均分给所选关系的各次提及，再送入同一冻结 H2、同一候选和同一门控。H1 在硬选择前比较两个关系的总质量。额外的软读出和只在关系词上均分的消融单独报告。")
    report=f"""# {rid} {label}在新开发图上的验证

2026 年 11 月 24 日以单 CPU 线程在既有云实例完成推理和统计，耗时 {elapsed:.2f} 秒。{full} 本轮新增训练更新数为 0。三个种子的检查点均在产生本轮新开发图前已经存在。

## 假设与实验设计

{j('configs/protocol.json')['hypothesis']} {model}

在 600 张合成图上构造 2,600 个问法，三种题型各 200 张。每图包括原问和同义改写，200 张可逆图另有两种关系方向与两种句序构成的四问。先在每图内平均预定问法与三个种子，然后按图、题型等权平均。另有每图一个未见句式，用于单列的泛化检验，不进入主指标。新图在生成器 6,000 万至 6,500 万的半开区间内抽取，与原正式区间分离。与审查过的 {co['archived_personnel_sets']:,} 个历史人员集合重叠为零；新句式按实体与关系归一化后与已查模板比较。[数据清单](data/cohort/manifest.json) · [事前协议](configs/protocol.json)

H1 对同一关系的提及汇总质量；两关系完全平分计半次正确。词面基线选最先提及的关系。H2 对所有有效题词均匀加权作为主基线；候选得分平分选首位，未覆盖计错。按题型在图上配对重抽 10,000 次，第 500 个有序差值作为单侧 95% 下界。总体通过规则在各模型运行前固定，各种子完整列出。

## 主指标

| 模块 | 模型 | 比较基线 | 配对增益（百分点） | 单侧 95% 下界（百分点） | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| H1 首步关系 | {pct(p['h1_accuracy'])} | {pct(p['first_mentioned_accuracy'])} | {pp(p['h1_delta'])} | {pp(p['h1_lcb95'])} | {passed(p['h1_pass'])} |
| H2 首跳证据 | {pct(p['h2_accuracy'])} | {pct(p['uniform_accuracy'])} | {pp(p['h2_delta'])} | {pp(p['h2_lcb95'])} | {passed(p['h2_pass'])} |

H1 的绝对门槛为 80%，还要求相对词面顺序的下界大于零。H2 要求相对均匀权重增益至少 2 个百分点且下界大于零。新增主要证据来自多视图图均值；原问、改写和未见句式均可单独复核。

| 种子 | H1 | H1 增益下界（百分点） | H2 | H2 对均匀增益（百分点） | H2 增益下界（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for seed,item in m["primary_per_seed"].items():
        report+=f"| {seed} | {pct(item['h1_accuracy'])} | {pp(item['h1_lcb95'])} | {pct(item['h2_accuracy'])} | {pp(item['h2_delta'])} | {pp(item['h2_lcb95'])} |\n"
    report+="\n## 受控比较与行为证据\n\n| 方法 | H1 识别率 | H2 首跳 Top-1 |\n| --- | ---: | ---: |\n"
    cn={"uniform":"均匀权重（全部有效词位）","uniform_mentions":"只在关系提及处均分","first_mentioned":"首个提及的关系","shuffled":"打乱学习权重","one":"单组表示（匹配预算）","four":"四组表示（匹配预算）","first":"候选首位","pool":"独立池化模型","old":"原第一层","direction":"纯方向分","random":"等范数随机权重","learned":"原论文权重","sharp_role":"R12 硬选择","soft_role":"同一 R08 权重的软读出","random_role_labels":"打乱角色训练标签","uniform_mentions_hard":"关系位置均匀后硬选择"}
    for name,row in m["variants"].items():
        item=row["primary"];report+=f"| {cn.get(name,name)} | {pct(item['h1_accuracy'])} | {pct(item['h2_accuracy'])} |\n"
    a=m["primary_sensitivity"]["unseen_template"];o=m["primary_sensitivity"]["original"]
    report+=f"""
单组和四组的受控评分头共享每轮 5,992 个有效训练样本及预算；另列的线性探针使用同一 9,007 张训练图，含随机标签对照。探针的可读性、词权重对关系的选择和最终首跳命中分别计量。门控开关的分组结果、纯方向分作为强制关闭校正项的干预，以及实际错例见[门控与错误分析](results/gates_and_errors.json)。

| 原问 H1 / H2 | 未见句式 H1 / H2 | 同图四问全对 | 语义反转两问都对 | 同义改写选中同一事实 | 候选覆盖 |
| --- | --- | ---: | ---: | ---: | ---: |
| {pct(o['h1_accuracy'])} / {pct(o['h2_accuracy'])} | {pct(a['h1_accuracy'])} / {pct(a['h2_accuracy'])} | {pct(b['quartet_all_correct'])} | {pct(b['semantic_reversal_both_correct'])} | {pct(b['paraphrase_fact_stability'])} | {pct(p['coverage'])} |

在封存的同一批开发题上，H2 相对直接首位为 {pp(p['h2_minus_direct'])} 个百分点、相对池化为 {pp(p['h2_minus_pool'])} 个百分点。这些是开发对照。正式 H3 队列已使用一次，本轮正式开封数为零。

## 证据边界与后续判断

"""
    if v=="original":
        report+=f"""H2 的增益与 H1 的绝对识别水平需要独立判断。首步关系识别率距 80% 门槛尚差 {max(0,80-100*p['h1_accuracy']):.2f} 个百分点；四问和未见样式的结果也说明不同问法表现有差异。错误不能仅归因于参数不足或门控。下一轮沿用同一开发图和所有基线，对已冻结的 R12 另行计分，以检验更改关系读头与硬选择后的版本。该联合比较不能拆分两项改动各自的因果贡献。
"""
    else:
        report+="改进版本与原论文的表示机制分别编号。本文结果只涉及本轮两候选合成题、三个固定种子和声明的句式。历史 R09 的正式成绩适用于当时冻结的 R12，早于此次新 H1/H2 开发判据。本轮不构成新增正式验收。\n"
    report+="\n[全量指标](results/stats/metrics.json) · [逐图统计](results/stats/graph_metrics.csv) · [全部逐题预测](results/prediction/predictions.jsonl.gz) · [输入、训练来源和权重摘要](configs/frozen-dependencies.json) · [运行命令](configs/launch.json) · [退出和资源](logs/resources.json)\n"
    with (r/"report.md").open("x",encoding="utf8") as f:f.write(report)
if __name__=="__main__":generate(Path(sys.argv[1]).resolve())
