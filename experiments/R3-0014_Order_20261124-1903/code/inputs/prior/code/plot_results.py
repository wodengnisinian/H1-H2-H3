"""Plot paired original and paraphrased development accuracy from the score files."""

import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "本地开发-01"
METHODS = {
    "first": "直接首位", "pool": "池化模型", "direction": "方向底座",
    "old": "冻结旧首层", "uniform": "均匀词权重", "learned": "学习共享词权重",
}
COLORS = ("#205B95", "#B64B20")
MARKERS = ("o", "D")


def draw():
    font = "Microsoft YaHei"
    findfont(font, fallback_to_default=False)
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "axes.axisbelow": True})
    fig, panels = plt.subplots(1, 2, figsize=(11.8, 6.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.145, right=.975, top=.77, bottom=.17, wspace=.13)
    fig.suptitle("同一张图上，六种方法的原问与改写首跳准确率", x=.05, y=.96, ha="left",
                 fontsize=20, weight="bold", color="#192F3B")
    for panel, (filename, title) in zip(panels, (("archive_dev", "历史开发集"),
                                                 ("fresh_dev", "新生成开发集")), strict=True):
        scores = json.loads((RUN / f"{filename}-评分.json").read_text(encoding="utf-8"))
        question = scores["question_views"]
        original = question["original"]["macro_accuracy_percent"]
        paraphrase = question["paraphrase"]["macro_accuracy_percent"]
        count = question["original"]["overall"]["graphs"]
        assert set(original) == set(paraphrase) == set(METHODS)
        assert count == scores["paraphrase_same_ordered_candidates_by_graph"]["hits"]
        assert scores["seeds"] == [1701, 2701, 3701]
        panel.set_title(f"{title}   ·   {count:,} 张图", loc="left", pad=13,
                        fontsize=13, weight="bold", color="#213541")
        for pos, method in enumerate(METHODS):
            y = len(METHODS) - pos
            values = (original[method], paraphrase[method])
            # Separate the two y positions so coincident scores remain readable.
            positions = (y + .14, y - .14)
            panel.plot(values, positions, color="#B0BCC0", linewidth=1.3, zorder=1)
            for value, yy, color, marker in zip(values, positions, COLORS, MARKERS, strict=True):
                panel.plot(value, yy, marker=marker, color=color, markersize=6.5, zorder=3,
                           markeredgecolor="white", markeredgewidth=.6)
                panel.annotate(f"{value:.2f}", (value, yy), xytext=(7, 0),
                               textcoords="offset points", color=color, fontsize=9,
                               va="center", ha="left",
                               bbox={"facecolor": "white", "edgecolor": "none", "pad": .25})
        panel.set_xlim(55, 87.5)
        panel.set_ylim(.45, 6.55)
        panel.set_xticks(range(55, 86, 5))
        panel.set_yticks(list(range(6, 0, -1)), METHODS.values())
        panel.tick_params(axis="both", length=0, pad=7, labelsize=10, colors="#384D58")
        panel.grid(axis="x", color="#DCE4E6", linewidth=.8)
        for y in (1.5, 2.5, 3.5, 4.5, 5.5):
            panel.axhline(y, color="#EFF2F3", linewidth=.8, zorder=0)
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.set_xlabel("首跳 Top-1 准确率（%）", fontsize=11, labelpad=9, color="#344955")
    handles = [Line2D([], [], marker=m, color=c, markersize=7, linestyle="none", label=l)
               for m, c, l in zip(MARKERS, COLORS, ("原问", "等义改写"), strict=True)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.978, .957),
               frameon=False, ncol=2, handlelength=.7, columnspacing=1.2, fontsize=11)
    fig.text(.05, .042, "三个固定种子（1701、2701、3701），三种题型等权。原问与改写使用相同的图和有序候选。", 
             fontsize=10, color="#506270")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        fig.savefig(ROOT / "首跳准确率.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
