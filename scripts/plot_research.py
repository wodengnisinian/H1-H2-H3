"""Plot final R12 development, R09 sealed acceptance and the distinct R10 four-candidate challenge."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.patches import Patch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def completed(root, rid, filename):
    registry = root / "experiment_registry.jsonl"
    if registry.is_file():
        entries = [json.loads(s) for s in registry.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        complete = [e for e in entries if e.get("id") == rid and e.get("event") == "end" and e.get("code") == 0]
        require(bool(complete), f"{rid} 尚无已完成的登记；请同步登记和评分产物后绘图。")
        directory = root / complete[-1]["path"] / "results"
    else:
        candidates = [p/"results" for p in (root/"experiments").glob(f"*-{rid}")
                      if (p/"artifacts.json").is_file() and (p/"results"/filename).is_file()]
        require(len(candidates) == 1, f"{rid} 缺少唯一可用的完整归档")
        directory = candidates[0]
    return directory, read(directory / filename)


@dataclass
class Panel:
    title: str
    kind: str
    single_graphs: int
    quartet_graphs: int
    seeds: tuple
    labels: list
    single: list
    quartet: list


def development(root):
    _, score = completed(root, "R12", "metrics.json")
    require(not score["smoke"] and score["primary_model"] == "sharp_role", "需要 R12 完整开发成绩")
    fresh = score["aggregate"]["fresh"]
    methods = {"first":"直接\n首位", "pool":"池化\n模型", "soft_role":"关系读头\n软权重", "sharp_role":"关系读头\n硬选择"}
    g = int(fresh["sharp_role"]["graph_count"])
    q = int(fresh["sharp_role"]["quartet_graphs"])
    require(all((fresh[m]["graph_count"],fresh[m]["quartet_graphs"]) == (g,q) for m in methods), "R12 分母不一致")
    return Panel("R12　新图开发集", "两候选", g, q, tuple(score["seeds"]), list(methods.values()),
                 [100*fresh[m]["original_top1"] for m in methods], [100*fresh[m]["quartet_all"] for m in methods])


def formal(root):
    directory, score = completed(root, "R09", "metrics.json")
    freeze = read(directory/"freeze.json")
    seeds = tuple(score["seeds"])
    original = score["question_views"]["original"]
    quartet = score["four_question"]["overall"]
    g, q = original["overall"]["graphs"], quartet["graphs"]
    primary = {"sharp_core":"封存模型\n硬选择","role_core":"封存模型\n软权重"}.get(freeze["module"], "封存\n模型")
    methods = {"first":"直接\n首位","pool":"池化\n模型","learned":primary}
    for m in methods:
        require(original["overall"]["accuracy"][m]["denominator"] == len(seeds)*g and
                quartet["all_four"][m]["denominator"] == len(seeds)*q, "R09 计分分母与固定种子不符")
    panel = Panel("R09　独立验收集", "两候选", g, q, seeds, list(methods.values()),
                  [original["macro_accuracy_percent"][m] for m in methods], [quartet["all_four"][m]["percent"] for m in methods])
    contrasts = score["paired_comparisons"]["original"]["contrasts_learned_minus"]
    comparisons = []
    for key, name in (("first","直接首位"), ("pool","池化模型")):
        x = contrasts[key]
        comparisons.append(f"对{name}{x['delta_pp']:+.2f}（下界 {x['lower_95_one_sided_pp']:+.2f}）")
    return panel, "正式验收原问增益（百分点，单侧 95% 下界）　" + "，".join(comparisons) + "。"


def four_candidates(root):
    directory, scores = completed(root, "R10", "scores.json")
    held = scores["challenge"]
    manifest = read(directory/"dataset-manifest.json")["challenge"]
    graphs = manifest["graphs"]
    require(held["questions_per_seed"] == manifest["questions"] == 4*graphs, "R10 同图四问数量不一致")
    require(held["gain_vs_onehop"]["twin_groups"]*2 == graphs, "R10 孪生组分母不一致")
    methods = {"first":"直接\n首位","onehop":"单跳\n模型","twohop":"二跳\n模型","symbolic":"符号路径\n枚举"}
    metric = held["methods"]
    return Panel("R10　未见关系组合", "四候选", graphs, graphs, tuple(int(s) for s in held["per_seed_top1"]),
                 list(methods.values()), [100*metric[m]["by_view"]["AB-normal"] for m in methods],
                 [100*metric[m]["four_question_accuracy"] for m in methods])


def draw(panels, contrast_note, output):
    regular = "#286989"
    joint = "#BE6436"
    ink = "#1C303A"
    muted = "#52656C"
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        try:
            findfont(family, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("需要可渲染中文的微软雅黑、思源黑体或黑体。")
    plt.rcParams.update({"font.family":family,"axes.unicode_minus":False,"axes.axisbelow":True})
    fig, axes = plt.subplots(1,3,figsize=(15,6.9),sharey=True,
                             gridspec_kw={"width_ratios":[4,3.35,4]})
    fig.subplots_adjust(left=.067,right=.982,bottom=.30,top=.71,wspace=.22)
    fig.text(.045,.945,"首跳选择与同图四问",fontsize=22,color=ink,weight="bold",va="top")
    fig.text(.047,.871,"经典两候选任务的开发与独立验收，以及四候选真实两跳挑战",fontsize=11.5,color=muted)
    handles=[Patch(color=regular,label="原问正确率"),Patch(color=joint,label="四问全对率")]
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.982,.956),ncol=2,
               frameon=False,handlelength=1,fontsize=11,columnspacing=1.8)
    seeds = panels[0].seeds
    require(len(seeds)==3 and all(set(p.seeds)==set(seeds) for p in panels), "各面板需有同样三个固定种子")
    for ax,panel in zip(axes,panels,strict=True):
        ax.set_title(panel.title,fontsize=14,loc="left",color=ink,weight="bold",pad=31)
        ax.text(0,1.037,f"{panel.kind}　原问 {panel.single_graphs:,} 图　四问 {panel.quartet_graphs:,} 图",
                transform=ax.transAxes,fontsize=10,color=muted)
        ids=list(range(len(panel.labels)))
        for data, offset, color in ((panel.single,-.2,regular),(panel.quartet,.2,joint)):
            require(all(0<=v<=100 for v in data),f"数值超界：{panel.title}")
            bars=ax.bar([x+offset for x in ids],data,width=.34,color=color,zorder=2)
            for bar,val in zip(bars,data,strict=True):
                ax.annotate(f"{val:.2f}",(bar.get_x()+bar.get_width()/2,val),xytext=(0,3),
                            textcoords="offset points",ha="center",fontsize=8.1,color=color)
        ax.set_xticks(ids,panel.labels,fontsize=10)
        ax.set_xlim(-.55,len(ids)-.45)
        ax.set_ylim(0,110)
        ax.set_yticks([0,20,40,60,80,100])
        ax.grid(axis="y",linewidth=.7,color="#DDE5E7")
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0,pad=6,colors=ink)
    axes[0].set_ylabel("准确率（%）",labelpad=4,color=muted,fontsize=10)
    fig.text(.047,.173,contrast_note,fontsize=10.5,color=ink)
    fig.text(.047,.115,"四问全对要求同一图、同一种子的两种关系顺序 × 两种分句顺序全部选对。单问和四问分别按各自图数计分。",fontsize=10,color=muted)
    fig.text(.047,.068,"四候选挑战取 AB 顺序、正常分句的原问；符号路径枚举是可解性对照。固定种子为 "+ "、".join(map(str,seeds))+"。",
             fontsize=10,color=muted)
    output.parent.mkdir(parents=True,exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("error",message="Glyph .* missing from font")
        fig.savefig(output,format="png",dpi=190,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,help="Default reports/科研结果.png")
    args=parser.parse_args()
    root=args.root.resolve()
    acceptance,note=formal(root)
    output=args.output or root/"reports/科研结果.png"
    draw([development(root),acceptance,four_candidates(root)],note,output)
    print(output)
