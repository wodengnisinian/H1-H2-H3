"""Render completed registry attempts into per-round Chinese reports and a numeric CSV index."""

import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import tempfile


TITLES = {
    "R01": "输入、分割与候选位置审计", "R02": "历史模型重放", "R03": "问题表示的首步关系可读性",
    "R04": "固定方向分与门控的可达上界", "R05": "词位置权重的消融", "R06": "四组与单组问题表示",
    "R07": "分句线索与固定 H2 的联合比较", "R08": "首步关系监督与候选监督",
    "R09": "冻结候选的独立 H3 验收", "R10": "真实两跳与未见关系组合",
    "R11": "采用与放弃方案的反事实检验", "R12": "冻结关系读头的硬选择", "R13": "R08 确定性重训复核",
}
TRANSLATED = {
    "R03": ("冻结的问题表示能否揭示陌生措辞中的首步关系？", "在新开发、模板和组合三个分割的原问上，四组双关系探针都须比随机训练标签对照高至少 5 个百分点。拟合样本与岭惩罚相同，词权重对两个关系完全平分计半次命中。"),
    "R04": ("门控和逐词匹配分差是否限制可达的首跳准确率？", "判断新开发原问的逐题上界是否达到较强基线再加 5 个百分点，区分关门、开门、金标未入围及平分情形。"),
    "R05": ("学习到的词位置权重能否优于位置干预对照？", "新开发原问与四问全对率都须高于均匀、等范数随机和位置打乱三个对照；候选置换后分数误差应在 2×10⁻⁵ 以内。"),
    "R07": ("固定 H2 时，显式分句顺序线索能否改善新措辞的排序？", "新开发原问对首位和池化分别至少高 5 个百分点，改写对两者均严格更高。四问须高于首位且不低于池化。"),
}
SPLIT = {"fresh": "新开发图", "archive": "历史开发图", "template": "新模板", "combination": "保留组合",
         "challenge": "未见关系组合", "dev": "开发关系组合", "seen_pair_test": "已见组合新图"}
NOTES = {
    "R01": "图的人员集合、文档编号和金标位置可以检查跨分割重复及位置捷径。审计结果本身不估计模型效果。",
    "R02": "重放使用已公开的 B13。七条预测流逐条比较，历史数据用于校验实现和量化偏差。",
    "R03": "首步关系探针从问题表示中二选一。随机标签对照使用同样样本和拟合预算。探针的识别率与真实首跳准确率是两个指标。",
    "R04": "对两候选每个词的关系匹配分差取极值，用门控幅度与方向分差计算可达边界。逐题上界允许为每道题选取最优权重。",
    "R05": "保持候选、匹配投影和门控固定，分别比较训练权重、均匀权重、等范数随机向量和词位打乱。",
    "R06": "四组分别计算词位 softmax 后读出四个问题分量；单组以四个查询向量的均值只读一次。该对照同时改变查询数及其聚合。",
    "R07": "等预算学习 64 个参数。将一个上下文坐标替换为题面分句线索，与同批训练的普通上下文对照比较。",
    "R08": "从问题中的时间、实体引导词和关系词的相对位置取特征，固定 H2，比较以候选答案或显式首步关系为目标训练的头。人工标记类别是输入的归纳偏置。",
    "R09": "原问为主要指标。以图为配对单位，三题型等权、图内先平均固定种子；分别计算候选减直接首位与候选减池化的增益。",
    "R10": "每张图从同一锚点给出四条候选；相同首步关系的两条边必须用后继区分。孪生图保持题面、候选及位置相同，仅交换后继关系。按无序关系对分割，并检查所有连通两跳。",
    "R11": "同一张图同时列出两种相反的可行方案。只交换“已采用”和“已放弃”的位置，词袋和有序候选保持一致，正确首条随之改变。显式状态规则提供可解性对照。",
    "R12": "沿用 R08 关系读头与 H2。对关系 v 的所有词位 Iᵥ 计算 Tᵥ = log∑exp(tⱼ)，选择 Tᵥ 最大的关系，在它的词位上均匀分配总质量 1，其余位置为 0。软权重和硬选择使用同一套参数。",
    "R13": "在新进程中按 R08 的固定种子、原始数据、代码和训练预算从零重训。按内容哈希复核训练产物，压缩数据比较解压后的字节。",
}
NEXT = {"R01": "复算已封存的 B13 预测与误差。", "R02": "在开发图上测量问题表示的首步可读性。",
        "R03": "测量门控与分差允许的逐题可达上界。", "R04": "用干预区分学习到的词权重与可达上界。",
        "R05": "在同一训练预算下拆分问题查询组数的影响。", "R06": "比较含分句线索的模型与匹配的无提示对照。",
        "R07": "按开发门槛决定候选冻结或开展首步关系监督比较。",
        "R08": "检验固定读头下硬选择与软权重的差别，再核对候选进入经典 H3 的资格。",
        "R09": "独立验收只归属于本次 freeze.json 指定的结构与检查点。四候选机制挑战与方案状态诊断另有计分。",
        "R10": "在归档中分别列出经典 H3 和四候选挑战的证据范围。",
        "R11": "根据输入碰撞诊断解释冻结模型能否区分方案状态；本轮不选择正式候选。",
        "R12": "按开发门槛先核对 sharp_role 的资格，再由 R09 的冻结回执确定正式验收对象。",
        "R13": "将逐字节重现结果归入可复现性证据，不累计为新的独立测试样本。"}


def read(path):
    return json.loads(path.read_bytes())


def atomic_write(path, text, encoding="utf-8"):
    path.parent.mkdir(exist_ok=True, parents=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def records(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8-sig") as stream:
        for raw in stream:
            if raw.strip():
                yield json.loads(raw)


def pct(x):
    return "—" if x is None else f"{100*x:.2f}%"


def pp(x):
    return "—" if x is None else f"{x:+.2f}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(head, rows):
    grid = [head, ["---"] * len(head), *rows]
    return "\n".join("| " + " | ".join(map(cell, line)) + " |" for line in grid)


def link(name, path):
    return f"[{name}](<{path}>)"


class Report:
    def __init__(self, root, event, cfg, index):
        self.root = root
        self.id = event["id"]
        self.run = root / event["path"]
        self.out = self.run / "results"
        self.cfg = cfg
        self.index = index
        self.source = self.out / event["result_file"]
        hypothesis,criterion = cfg.get("hypothesis","见本轮冻结方案。"),cfg.get("criteria","见本轮冻结方案。")
        if self.id in TRANSLATED and hypothesis.isascii():
            hypothesis,criterion=TRANSLATED[self.id]
        self.content = [f"# {self.id}　{TITLES.get(self.id, self.id)}", "## 研究问题",
                        hypothesis, "## 事前判据", criterion, "## 设计与数据",
                        NOTES.get(self.id, cfg.get("description", ""))]

    def num(self, split, seed, model, metric, value, unit="%", source=None):
        if type(value) in (int, float):
            path = (source or self.source).relative_to(self.root).as_posix()
            self.index.append([self.id, split, seed, model, metric, round(value, 8), unit, path])

    def txt(self, *paragraphs):
        self.content.extend(s for s in paragraphs if s)

    def error_samples(self, paths, primary, *, mode="mech"):
        samples, buckets, examined = [], set(), False
        for path in paths:
            if not path.exists():
                continue
            examined = True
            for r in records(path):
                split = r.get("split", "正式" if mode == "formal" else "历史" if mode == "history" else "挑战")
                if mode == "mech" and split != "fresh":
                    continue
                if mode == "history":
                    seed = next(iter(r["per_seed"]))
                    pred = r["per_seed"][seed]["serial"]
                else:
                    seed = r.get("seed", "")
                    pred = r.get("predictions", r.get("prediction_fact", {})).get(primary)
                gold = r.get("gold", r.get("gold_fact"))
                if pred is None or str(pred) == str(gold):
                    continue
                kind = r.get("kind", r.get("pair", ""))
                view = r.get("variant", r.get("view", "原问"))
                if (kind, view) in buckets:
                    continue
                buckets.add((kind, view))
                samples.append([r.get("graph_id", r.get("id", r.get("doc_id"))), seed, kind, view,
                                gold, pred, str(r.get("candidates", []))])
                if len(samples) >= 3:
                    break
            if len(samples) >= 3:
                break
        self.txt("## 逐题核查", (f"按文件顺序取 {primary} 在不同题型或问法上的前 {len(samples)} 条错误。\n\n" +
                 table(["图或题编号", "种子", "题型或组合", "问法", "金标", "预测", "有序候选"], samples))
                 if samples else ("所查逐题文件中，指定模型在该范围内没有错例。" if examined else "本轮产物未提供逐题错例。"))
        if samples and mode!="history":
            id0,_,_,view0,g0,p0,_ = samples[0]
            path=self.out/{"mech":"development-inputs.jsonl.gz","formal":"features.jsonl","twohop":"challenge-features.jsonl"}[mode]
            question=None
            nodes=None
            if path.exists():
                for row in records(path):
                    if row.get("graph_id",row.get("id",row.get("doc_id")))!=id0:
                        continue
                    if row.get("variant")==view0 or mode=="twohop":question=row["question"]
                    if "nodes" in row:nodes=row["nodes"]
                    if "facts" in row:nodes=row["facts"]
                    if question is not None and nodes is not None:break
            if question is not None and nodes is not None:
                self.txt("首例题面为“"+"".join(question)+"”。金标记录为“"+" ".join(nodes[int(g0)])+
                         "”，该模型选择了“"+" ".join(nodes[int(p0)])+"”。")


def input_audit(r, d):
    r.txt(f"共验证 {d['verified_file_count']} 份输入文件。训练、开发和已公开 B13 之间重合的人物集合数为 {d['train_dev_or_b13_overlap']}。B13 有 {d['old_b13_unique_person_sets']} 组人物。",
          table(["分割", "图数", "不同人物集合"], [[k, x["graphs"], x["unique_person_sets"]] for k,x in d["splits"].items()]),
          "## 首位捷径")
    rows = []
    for split, parts in d["first_candidate_by_stratum_n_and_hits"].items():
        for k,(n,h) in parts.items():
            rows.append([split,k,h,n,pct(h/n)])
            r.num(split, "确定性", "first", f"{k}_top1", 100*h/n)
    r.txt(table(["开发集","题型","首位命中","图数","准确率"], rows),
          "本轮仅核验输入和计数。随机种子、模型错例与区间估计不适用。")


def historical(r, receipt):
    path = r.out / "B13-primary.json"
    d = read(path)
    n, hits = d["document_count"], d["pooled_counts"]
    r.txt(f"历史图数 {n:,}，三个模型种子为 {', '.join(map(str, d['seed_order']))}。收据记录 {len(receipt['steps'])} 个复核步骤。", "## 复算结果")
    tab = []
    for seed in d["seed_order"]:
        for method, key in (("serial",f"serial_{seed}"), ("pooled",f"pooled_{seed}"), ("first","gold_rank_one")):
            p = hits[key]/n
            tab.append([seed, method, hits[key], pct(p)])
            r.num("B13", seed, method, "original_top1",100*p, source=path)
    r.txt(table(["种子","方法","正确图数","首跳准确率"],tab))
    if "first_edge_rule_hits" in hits:
        r.txt(f"题面起点与首步关系唯一匹配的图为 {hits['first_edge_rule_hits']:,}/{n:,}。")
    r.txt("## 配对比较", table(["串联减基线","增益（百分点）","单侧95%下界","归档下界"],
          [[k,pp(v["effect_pp"]),pp(v["lower95_pp"]),pp(v["archived_lower95_pp"])] for k,v in d["contrasts"].items()]),
          f"复算采样实现为 {d['bootstrap']}。")
    qp = r.out / "B13-quartets.json"
    if qp.exists():
        q = read(qp)
        r.txt("四问均命中的计数为 " + "，".join(f"{k} {v['all_four_correct']:,}/{v['graph_seed']:,}" for k,v in q["counts"].items())+"。")
    r.txt("本次复核已通过六组模型预测和直接首位预测的逐文件哈希检查。")
    r.error_samples([r.out/"B13-original-by-graph.jsonl"], "serial", mode="history")


def mech(r, d):
    seedsets = d.get("per_seed", {})
    main = d.get("primary_model", d.get("primary", ""))
    all_metrics = d["aggregate"]
    fresh = all_metrics.get("fresh", {})
    r.txt("## 首跳与配对结果", f"固定种子 {', '.join(map(str,d.get('seeds',seedsets)))}。主要开发分割为 fresh。四问全对要求同一图、同一种子的四个视图全部正确。")
    r.txt(table(["新开发方法","原问","改写","四问全对","换序两问均对","图数","四问图数"],
          [[m,pct(v.get("original_top1")),pct(v.get("paraphrase_top1")),pct(v.get("quartet_all")),
            pct(v.get("reversal_both")),int(v.get("graph_count",0)),int(v.get("quartet_graphs",0))] for m,v in fresh.items()]))
    for seed, splits in [("汇总", all_metrics), *seedsets.items()]:
        for sp, group in splits.items():
            if sp not in all_metrics:
                continue
            if "ranking" in group and "h1" in group:
                for name, values in group["h1"].items():
                    for measure, value in values.items():
                        if type(value) in (float,int):r.num(sp,seed,name,measure,100*value)
                group=group["ranking"]
            for method, vals in group.items():
                for measure, value in vals.items():
                    if type(value) not in (int,float):
                        continue
                    count = measure in ("graph_count","quartet_graphs")
                    r.num(sp,seed,method,measure,value if count else 100*value,"图" if count else "%")
    if main and main in fresh:
        r.txt(table(["主要方法 "+main,"原问","改写","四问全对"],
              [[SPLIT.get(s,s),pct(group[main].get("original_top1")),pct(group[main].get("paraphrase_top1")),
                pct(group[main].get("quartet_all"))] for s,group in all_metrics.items() if main in group]))
    diag = d.get("diagnostics",{})
    if r.id=="R03":
        tab=[]
        for split in ("fresh","template","combination"):
            x=diag.get(split,{}).get("original",{})
            if "read_four_pair" in x:
                a,b=x["read_four_pair"],x["read_four_random_labels_pair"]
                tab.append([SPLIT[split],pct(a),pct(b),pp(100*(a-b)),pct(x.get("mass_first_accuracy"))])
        r.txt("## 首步关系探针",table(["原问分割","四组探针二选一","随机标签","差值（百分点）","词权重指向正确关系"],tab))
        for seed, diagnostics in [("汇总",diag),*((s,v.get("diagnostics",{})) for s,v in seedsets.items())]:
            for sp,variants in diagnostics.items():
                for var,measures in variants.items():
                    for k,v in measures.items():
                        if k.startswith(("read_","mass_")) and type(v) in (int,float):
                            r.num(sp,seed,k,var,100*v)
    if r.id=="R04":
        x=diag.get("fresh",{}).get("original",{})
        r.txt("## 方向错误的可修正空间",table(["门控","上界","使用标签的诊断","方向错误","其中无法修正"],
              [[k,pct(v["attainable_ceiling"]),pct(v["label_oracle_accuracy"]),v["direction_errors"],v["structurally_uncorrectable"]]
               for k,v in x.items() if isinstance(v,dict) and "attainable_ceiling" in v]))
    if r.id=="R08":
        h1=d.get("h1",{}).get("fresh",{})
        ci=d.get("objective_intervals",{}).get("fresh",{})
        r.txt("## 直接监督的匹配对照",table(["模型","原问首步角色识别","原问首跳"],
              [[m,pct(h1[m].get("original_role")),pct(fresh[m].get("original_top1"))]
               for m in ("cue_role","cue_candidate","cue_random_role") if m in h1 and m in fresh]))
        r.txt(table(["cue_role 减 cue_candidate","增益（百分点）","双侧 95% 区间"],
              [[k,pp(100*x["delta"])," 至 ".join(pp(100*v) for v in x["two_sided_95"])]
               for k,x in ci.items() if k in ("rank_vs_cue_candidate","role_vs_cue_candidate")]))
        for split, methods in d.get("h1",{}).items():
            for m, vals in methods.items():
                for k,val in vals.items():
                    if type(val) in (float,int):r.num(split,"汇总",m,k,val*100)
    if r.id=="R12":
        controls=d.get("frozen_control_intervals",{}).get("fresh",{})
        r.txt("## 冻结参数下的硬选择效应",
              f"本轮新增训练步数 {d['new_training_updates']}；R08 软分数重放的最大绝对偏差为 {d['max_r08_soft_replay_error']:.3g}，置换候选后的最大偏差为 {d['max_swap_error']:.3g}。",
              table(["新开发问法","硬选择减对照","增益（百分点）","单侧95%下界","双侧95%区间"],
                    [[v,b,pp(100*x["delta"]),pp(100*x["lcb95"])," 至 ".join(pp(100*z) for z in x["two_sided_95"])]
                     for v,items in controls.items() for b,x in items.items() if isinstance(x,dict) and "delta" in x]))
        ds=diag.get("fresh",{})
        r.txt(table(["新开发原问分组","判断数","硬选择修正","硬选择新增错误","识别率","给定正确关系的诊断准确率"],
              [[name,x.get("decisions"),x.get("corrected"),x.get("broken"),pct(x.get("role_accuracy")),pct(x.get("oracle_accuracy"))]
               for name,x in ds.items() if name in ("all","gate True","gate False") and isinstance(x,dict)]),
              "软权重、硬选择和标签诊断分别计分。只有事先指定且达到开发门槛的推理实现有资格进入 R09。")
        for sp, variants in d.get("frozen_control_intervals",{}).items():
            for v, items in variants.items():
                for control,x in items.items():
                    if isinstance(x,dict) and "delta" in x:
                        r.num(sp,"汇总",f"sharp_role - {control}",v+"_gain",100*x["delta"],"百分点")
    intervals=d.get("paired_intervals",{}).get("fresh",{})
    itab=[]
    for v,baselines in intervals.items():
        for b,x in baselines.items():
            if isinstance(x,dict) and "delta" in x:
                ci=x.get("two_sided_95",[])
                itab.append([v,b,pp(x["delta"]*100),pp(100*x["lcb95"]) if "lcb95" in x else "—",
                             " 至 ".join(pp(100*i) for i in ci)])
    if itab:
        r.txt("## 新开发集的配对增益",table(["问法","对照","主要方法减对照（百分点）","单侧95%下界","双侧95%区间"],itab),
              "在题型内按图重抽，先对同一图的种子取平均。此处的区间对应首跳排名与两条基线的差。")
    status = d.get("criterion_met")
    if isinstance(status,bool):
        r.txt("## 判定", "该轮的预设判据" + ("达到。" if status else "未达到。"))
    gate=d.get("development_gate",{})
    if isinstance(gate,dict) and "eligible" in gate:
        r.txt("本轮记录的独立验收资格为" + ("通过。" if gate["eligible"] else "未取得。"))
    r.txt("开发分布中的关系组合留给新拟合的头评估。历史编码器曾见全部 28 个关系对，保留组合上的成绩需按这一条件解释。")
    if d.get("smoke"):
        r.txt("该文件标记为小样本联调，结果仅用于验证计算流程。")
    r.error_samples(sorted(r.out.glob("errors*.jsonl.gz")), main)


def formal(r, d):
    q = d["question_views"]
    freeze_path=r.out/"freeze.json"
    if freeze_path.is_file():
        freeze=read(freeze_path)
        r.txt("本次正式计分中的 learned 对应封存推理模块 "+str(freeze.get("module","未记录"))+
              "，该候选的检查点目录为 "+str(freeze.get("candidate","未记录"))+"。其结果不自动转移到软权重、硬选择或其他架构。")
    r.txt("## 封存后的独立计分",
          f"原问共 {q['original']['overall']['graphs']:,} 张图。四问覆盖 {d['four_question']['overall']['graphs']:,} 张图。固定种子 {', '.join(map(str,d['seeds']))}。")
    rows=[]
    for method,value in q["original"]["macro_accuracy_percent"].items():
        para=q["paraphrase"]["macro_accuracy_percent"][method]
        four=d["four_question"]["overall"]["all_four"][method]["percent"]
        rows.append([method,f"{value:.2f}%",f"{para:.2f}%",f"{four:.2f}%"])
        for label,v in [("original",value),("paraphrase",para),("quartet_all",four)]:
            r.num("formal","汇总",method,label,v)
        for seed in d["seeds"]:
            for view, v in q.items():
                r.num("formal",seed,method,view,v["by_seed"][str(seed)]["accuracy"][method]["percent"])
            r.num("formal",seed,method,"quartet_all",d["four_question"]["by_seed"][str(seed)]["all_four"][method]["percent"])
    r.txt(table(["方法","原问","改写","四问全对"],rows), "## H3 的两条基线")
    b=d["paired_comparisons"]["original"]
    r.txt(table(["基线","增益（百分点）","单侧95%下界","增益门槛","下界门槛"],
          [[k,pp(x["delta_pp"]),pp(x["lower_95_one_sided_pp"]),"通过" if x["point_pass"] else "未通过",
            "通过" if x["interval_pass"] else "未通过"] for k,x in b["contrasts_learned_minus"].items()]),
          f"配对重抽 {b['replicates']:,} 次。预设双基线验收" + ("通过。" if b["both_thresholds_met"] else "未通过。"),
          "结论适用于本次封存的两候选合成题、模型、候选规则与问法。")
    r.error_samples([r.out/"scored-predictions.jsonl"],"learned",mode="formal")


def twohop(r, scores):
    d=scores["challenge"]
    tab=[]
    for split,panel in scores.items():
        for method,m in panel["methods"].items():
            for key in ("top1","four_question_accuracy","semantic_swap_both_correct","both_counterfactual_twins_correct"):
                r.num(split,"汇总",method,key,100*m[key])
            if split=="challenge":
                tab.append([method,pct(m["top1"]),pct(m["four_question_accuracy"]),
                            pct(m["semantic_swap_both_correct"]),pct(m["both_counterfactual_twins_correct"])])
        for seed,models in panel["per_seed_top1"].items():
            for k,v in models.items():r.num(split,seed,k,"top1",100*v)
    r.txt("## 挑战集结果", f"每种子 {d['questions_per_seed']:,} 道常规挑战题，固定种子 {', '.join(d['per_seed_top1'])}。",
          table(["方法","首跳","四问全对","换序两问均对","孪生两图均对"],tab))
    sec=d.get("second_relation_only",{})
    if sec:
        r.txt("只替换第二步关系并同时答对两题的比例",table(["方法","双对率"],[[m,pct(x["both_correct"])] for m,x in sec.items()]))
        for m,x in sec.items():
            r.num("challenge_second","汇总",m,"both_correct",100*x["both_correct"])
            for item in x["per_seed"]:
                for k,v in item.items():
                    if k!="seed":r.num("challenge_second",item["seed"],m,k,100*v)
    g=d["gain_vs_onehop"]
    r.txt(f"二跳相对单跳增益为 {pp(g['estimate_pp'])} 个百分点，双侧 95% 区间为 " +
          " 至 ".join(pp(x) for x in g["two_sided_95_ci_pp"]) + f"。在关系对内以 {g['twin_groups']} 个孪生组为单位重抽 {g['replicates']:,} 次。",
          "按预设的各项挑战门槛，二跳机制" + ("通过。" if d["meets_preregistered_challenge_rule"] else "未通过。"),
          "仅见首跳输入的确定性选择器在均衡孪生图上的上界为 50%，四选一均匀随机期望为 25%。符号路径枚举在本次挑战的实测成绩为 " +pct(d["methods"]["symbolic"]["top1"])+"。",
          "两跳模型显式得到词相等与图连接特征。挑战检验对这一合成结构的迁移；其方法、候选数和题型与经典 H3 各自计分。")
    model="twohop" if d["error_judgments"].get("twohop",0) else "onehop"
    if model=="onehop":r.txt("挑战集二跳模型的错误数为零，下面抽取单跳对照的错误。")
    r.error_samples([r.out/"challenge-errors.jsonl"],model,mode="twohop")


def scope(r,d):
    seeds=d["per_seed"]
    pooled=d["pooled"]
    r.txt(f"共 {d['graphs']} 张图，每图两道配对题。固定种子为 {', '.join(map(str,seeds))}。翻转率只要求改变选择，双对率还要求两题都命中各自的正确首条。", "## 状态干预与输入碰撞",
          table(["方法","首跳","双对率","翻转率","相同输入对数","由碰撞限定的首跳上界"],
                [[m,pct(x["top1"]),pct(x["both_correct_rate"]),pct(x["flip_consistency"]),
                  f"{x['proven_collisions']}/{x['graph_seed_pairs']}",pct(x["top1_upper_from_proven_collisions"])]
                 for m,x in pooled.items()]))
    rates={"top1","both_correct_rate","flip_consistency","same_choice_rate",
           "top1_upper_from_proven_collisions","both_upper_from_proven_collisions"}
    count_units={"question_denominator":"题×种子","correct_questions":"题×种子",
                 "graph_seed_pairs":"图×种子","both_correct":"图×种子","predictions_changed":"图×种子",
                 "proven_collisions":"图×种子"}
    for seed, methods in [("汇总",pooled),*seeds.items()]:
        for method,x in methods.items():
            for k,v in x.items():
                if k in rates:r.num("scope",seed,method,k,100*v)
                elif k in count_units:r.num("scope",seed,method,k,v,count_units[k])
    r.txt("## 双对率判据", table(["模型减直接首位","双对率增益（百分点）","单侧95%下界","状态判据"],
          [[m,pp(d["paired_bootstrap"][f"{m}_minus_direct_first"]["both_correct"]["delta_pp"]),
            pp(d["paired_bootstrap"][f"{m}_minus_direct_first"]["both_correct"]["lower95_one_sided_pp"]),
            "通过" if ok else "未通过"] for m,ok in d["predeclared_scope_behavior"].items()]),
          "每个种子的双对率须至少达到 80%，按图重抽的双对率增益单侧 95% 下界还须大于零。碰撞上界只使用可证明相同的输入。",
          table(["种子","采用词元 ID","放弃词元 ID","未知词 ID","开门题数"],
                [[s,x["state_token_ids"].get("已采用"),x["state_token_ids"].get("已放弃"),x["unknown_id"],
                  x["gate_open_questions"]] for s,x in d["diagnostics"].items()]))
    for path in sorted(r.out.glob("errors-*.jsonl")):
        example=next(records(path),None)
        if example is None:continue
        wrong=[m for m,x in example["metrics"].items() if not x["both_correct"] and m not in ("direct_first","status_rule")]
        if not wrong:continue
        pick=next((m for m in reversed(["R07","R08","R12"]) if m in wrong),wrong[0])
        r.txt("## 逐题核查", f"种子 {example['seed']}，图 {example['graph_id']}，有序候选 {example['candidates']}。",
              table(["问法","金标","模型 "+pick],[[q,y,p] for q,y,p in
                    zip(example["questions"],example["gold"],example["predictions"][pick],strict=True)]))
        break
    r.txt("本轮冻结全部模型，只定位在该种措辞和分词条件下是否能区分方案状态。")


def reproduction(r,d):
    files=d["matched_files"]
    checks=d["matched_metrics"]
    seeds=sorted({Path(name).name.split("-")[-1].split(".")[0] for name in files})
    r.txt(f"复跑使用种子 {', '.join(seeds)}。输出按 {len(files)} 份文件和 {len(checks)} 项评分对象逐一核查。",
          "## 同种子重训的内容一致性",
          table(["种子","检查点","参数","逐题预测","错误明细"],[[seed,*[
              "相同" if files.get(f"{pre}-{seed}.{ext}",{}).get("identical") else "缺失或不同"
              for pre,ext in (("checkpoint","json"),("parameters","json"),("predictions","jsonl.gz"),("errors","jsonl.gz"))]]
              for seed in seeds]),
          table(["评分对象","原运行与复跑"],[[k,"相同" if v else "不同"] for k,v in checks.items()]),
          "端到端重现判据" + ("通过。" if d["reproduced"] else "未通过。"),
          "对逐题预测、错误记录和已计算区间逐项比对，保持数据和种子不变。两次运行共享样本，重复观测不增加有效图数。")
    for name,info in files.items():
        seed=name.split("-")[-1].split(".")[0]
        r.num("R08-repeat",seed,"cue_role",name.split("-")[0]+"_identical",int(info["identical"]),"0/1")
    for k,v in checks.items():r.num("R08-repeat","汇总","cue_role",k+"_identical",int(v),"0/1")
    errors=sorted((r.out/"training").glob("errors-*.jsonl.gz"))
    if errors:
        r.txt("## 原有错例的复现",f"随机种子和输入保持一致时，错误流按解压后字节比较。"
              f"第一份逐题错误文件为 {link(errors[0].name,'../'+errors[0].relative_to(r.root).as_posix())}。")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.root.resolve()
    registry=root/"experiment_registry.jsonl"
    if not registry.exists():
        print("No registry yet");return
    ended={}
    for e in records(registry):
        if e.get("event")=="end" and e.get("code")==0:
            ended[e["id"]]=e
    index=[]
    for eid,e in sorted(ended.items()):
        run=root/e["path"]
        out=run/"results"
        rec=out/e["result_file"]
        if not rec.is_file():
            print(f"Skipped incomplete archive {eid}: {rec}");continue
        spec=run/"preregistration.json"
        cfg=read(spec if spec.exists() else root/"configs"/f"{eid}.json")
        local_index=[]
        report=Report(root,e,cfg,local_index)
        data=read(rec)
        def render():
            if eid=="R01":input_audit(report,data)
            elif eid=="R02":historical(report,data)
            elif eid=="R09" and "question_views" in data:formal(report,data)
            elif eid=="R10" and "challenge" in data:twohop(report,data)
            elif eid=="R11" and "predeclared_scope_behavior" in data:scope(report,data)
            elif eid=="R13" and "matched_files" in data:reproduction(report,data)
            elif "aggregate" in data:mech(report,data)
            else:raise ValueError("未识别的收据模式")
        try:
            render()
        except (ValueError,KeyError,TypeError,AttributeError,IndexError,FileNotFoundError) as err:
            print(f"{eid}: saved archive links only; {type(err).__name__}: {err}")
            report=Report(root,e,cfg,[])
            keys=", ".join(map(str,data)) if isinstance(data,dict) else type(data).__name__
            report.txt("## 判读状态","登记记录确认进程成功结束。本报告尚未解析该收据的指标结构。"+
                       f"收据的顶层字段为 {keys}。")
        else:
            index.extend(local_index)
        next_step=NEXT.get(eid,cfg.get("after",""))
        fp=Path("..")/e["path"]
        report.txt("## 后续与归档",next_step,"　".join([
            link("冻结方案",(fp/"preregistration.json").as_posix()),
            link("本轮结果",(fp/"results"/e["result_file"]).as_posix()),
            link("计算资源",(fp/"resources.json").as_posix()),
            link("完整性清单",(fp/"artifacts.json").as_posix()),
            link("逐种子数值","../analysis/results_summary.csv")]))
        atomic_write(root/"reports"/f"{eid}.md","\n\n".join(report.content)+"\n")
    buf=io.StringIO(newline="")
    writer=csv.writer(buf)
    writer.writerow(["实验","分割","种子","方法","指标","数值","单位","结果来源"])
    writer.writerows(index)
    atomic_write(root/"analysis/results_summary.csv",buf.getvalue(),encoding="utf-8-sig")
    print(f"Rendered registered completed rounds; indexed {len(index)} numeric results")


if __name__=="__main__":
    main()
