# NOTE: service address and credentials are read from environment
#       variables (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
#       This file was sanitised before public release.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成论文图 1~4 的【英文版】

背景：原 make_figures.py 生成的是中文图（标签、标题、图注都是中文），
      但投稿稿是英文 —— 审稿人不能看到中文标签的图。
      本脚本是它的英文对应版本：**只改文字，数据源与绘图逻辑完全一致**。

数据源：figdata/figdata.json（与原脚本共用，确保数值不变）
输出：figures_en/（不覆盖原中文图，便于对照）

与原脚本的差异：
  1 所有中文标签/标题/图注改为英文
  2 不再注册 Windows 中文字体（英文用 DejaVu Sans 即可）
  3 因此可以保留真减号 U+2212（SimHei 缺该字形，DejaVu 有）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FD = "/mnt/d/Users/Administrator/.dsh/ungrouped/thesis/figdata"
# 输出目录用【脚本所在目录】下的 figures/，不要硬编码本机绝对路径 ——
# 公开出去的脚本里写死某个人的目录，读者是跑不起来的。
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# 英文图用默认无衬线字体即可；DejaVu Sans 覆盖 U+2212（真减号）
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["font.size"] = 10
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 110

P = json.load(open(os.path.join(FD, "figdata.json"), encoding="utf-8"))
CFGS = ["A", "B", "C", "D"]
LBL = {"A": "A (PoG)", "B": "B (selective)",
       "C": "C (early stop)", "D": "D (SAGE-PoG)"}
COL = {"A": "#8c8c8c", "B": "#4c78a8", "C": "#e4a13a", "D": "#c0392b"}

# ============================================================
# Figure 2 (file fig1_accuracy.png)  Accuracy comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1),
                         gridspec_kw={"width_ratios": [2.05, 1]})
ax = axes[0]
batches = ["G1", "G2", "G3"]
w = 0.2
for i, c in enumerate(CFGS):
    vals = [P["hits"][f"{g}|{c}"] for g in batches]
    pos = [x + (i - 1.5) * w for x in range(len(batches))]
    ax.bar(pos, vals, w, label=LBL[c], color=COL[c],
           edgecolor="black", linewidth=0.5)
    for x, v in zip(pos, vals):
        ax.text(x, v + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6)
ax.set_xticks(range(len(batches)))
ax.set_xticklabels(["G1", "G2", "G3"])
ax.set_xlabel("Subgraph batch")
ax.set_ylabel("Hits@1 (%)")
ax.set_ylim(0, 58)
ax.set_title("(a) Hits@1 over the three batches", fontsize=11)
ax.legend(fontsize=8, ncol=4, loc="upper center", frameon=False)
ax.grid(axis="y", alpha=0.3, linestyle=":")

ax2 = axes[1]
means = [round((P["hits"][f"G1|{c}"] + P["hits"][f"G2|{c}"]
                + P["hits"][f"G3|{c}"]) / 3, 1) for c in CFGS]
tot = [sum(1 for g in batches if P["hits"][f"{g}|{c}"] >= P["hits"][f"{g}|A"])
       for c in CFGS]
ax2.bar(range(4), means, 0.6, color=[COL[c] for c in CFGS],
        edgecolor="black", linewidth=0.5)
for x, (c, v, t) in enumerate(zip(CFGS, means, tot)):
    ax2.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
    # ⚠️ 曾写成 f"{t}/3 batches\nnon-degrading" —— 文字超出柱宽被裁切，
    #    图上显示成 "2/3 batche: on-degradir"。
    #    改为紧凑写法；完整含义由 (b) 的标题 "Mean and non-degrading batches"
    #    以及图注承担。
    lab = "baseline\n(reference)" if c == "A" else f"{t}/3\nnon-degr."
    ax2.text(x, 2.0, lab, ha="center", va="bottom", fontsize=7.2,
             color="#f0f0f0" if c == "A" else "white")
ax2.set_xticks(range(4))
ax2.set_xticklabels([LBL[c] for c in CFGS], fontsize=8, rotation=16)
ax2.set_ylabel("Three-batch mean Hits@1 (%)")
ax2.set_ylim(0, 55)
ax2.set_title("(b) Mean and non-degrading batches", fontsize=11)
ax2.grid(axis="y", alpha=0.3, linestyle=":")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_accuracy.png"), bbox_inches="tight")
# 同一张图另存一份"论文里的编号名"，这样包里的文件名与论文一致
fig.savefig(os.path.join(OUT, "Fig2_accuracy.png"), bbox_inches="tight")
plt.close(fig)
print("  OK  fig1_accuracy.png")

# ============================================================
# Figure 3 (file fig2_phase.png)  Per-phase latency
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2),
                         gridspec_kw={"width_ratios": [1.15, 1]})
EN = {"subgraph": "Subgraph construction",
      "topic_path": "Topic path exploration",
      "supplement": "Supplementary exploration",
      "node_expand": "Node expansion"}
order = ["subgraph", "topic_path", "supplement", "node_expand"]
pcol = ["#9ecae1", "#6baed6", "#2171b5", "#08306b"]

ax = axes[0]
pa, pd_ = P["phase_G1_A"], P["phase_G1_D"]
oth_a, oth_d = P["total_G1_A"], P["total_G1_D"]
sa = sum(pa.get(n, 0) for n in order)
sd = sum(pd_.get(n, 0) for n in order)
bottom_a = bottom_d = 0.0
for n, c in zip(order, pcol):
    va, vd = pa.get(n, 0), pd_.get(n, 0)
    ax.bar(0, va, 0.5, bottom=bottom_a, color=c, edgecolor="white",
           linewidth=0.6, label=EN[n])
    ax.bar(1, vd, 0.5, bottom=bottom_d, color=c, edgecolor="white",
           linewidth=0.6)
    bottom_a += va
    bottom_d += vd
ax.bar(0, oth_a - sa, 0.5, bottom=bottom_a, color="#d9d9d9", edgecolor="white",
       linewidth=0.6, hatch="//",
       label="Other (answer generation / planning, etc.)")
ax.bar(1, oth_d - sd, 0.5, bottom=bottom_d, color="#d9d9d9", edgecolor="white",
       linewidth=0.6, hatch="//")
ax.text(0, oth_a + 12, f"{oth_a:.0f}s", ha="center", fontsize=9,
        fontweight="bold")
ax.text(1, oth_d + 12, f"{oth_d:.0f}s", ha="center", fontsize=9,
        fontweight="bold")
ax.annotate("", xy=(0.42, oth_d + 18), xytext=(0.42, oth_a + 18),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
ax.text(0.47, (oth_a + oth_d) / 2 + 18, "\u221224.0%", color="#c0392b",
        fontsize=9.5, fontweight="bold", va="center")
ax.set_xticks([0, 1])
ax.set_xticklabels(["A (PoG)", "D (SAGE-PoG)"])
ax.set_ylabel("G1 per-question average latency (s)")
ax.set_ylim(0, oth_a * 1.30)
ax.set_title("(a) G1 latency composition by phase", fontsize=11)
ax.legend(fontsize=7.0, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.25, linestyle=":")

ax2 = axes[1]
vals = {c: [P["eff"][c]["p50"], P["eff"][c]["p90"], P["eff"][c]["mx"]]
        for c in CFGS}
mk = {"A": "o", "B": "s", "C": "^", "D": "D"}
for c in CFGS:
    ax2.plot(["P50", "P90", "Max"], vals[c], marker=mk[c], color=COL[c],
             label=LBL[c], linewidth=1.6, markersize=6)
ax2.set_yscale("log")
ax2.set_ylabel("Per-question latency (s, log scale)")
ax2.set_title("(b) Long tail (300 questions pooled)", fontsize=11)
ax2.legend(fontsize=8, frameon=False)
ax2.grid(alpha=0.25, linestyle=":", which="both")
for c in CFGS:
    ax2.annotate(f"{P['eff'][c]['n1000']} >1000 s",
                 xy=("Max", vals[c][2]), xytext=(6, -3),
                 textcoords="offset points", fontsize=7.2, color=COL[c])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_phase.png"), bbox_inches="tight")
# 同一张图另存一份"论文里的编号名"，这样包里的文件名与论文一致
fig.savefig(os.path.join(OUT, "Fig3_phase.png"), bbox_inches="tight")
plt.close(fig)
print("  OK  fig2_phase.png")

# ============================================================
# Figure 4 (file fig3_sensitivity.png)  Hyper-parameter sensitivity
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
Ks = sorted(P["K"], key=lambda x: int(x))
ax = axes[0]
ax.plot([int(k) for k in Ks], [P["K"][k]["scan"] for k in Ks], "o--",
        color="#4c78a8", label="Sweep set (25 q)", markersize=6)
ax.plot([int(k) for k in Ks], [P["K"][k]["hold"] for k in Ks], "s--",
        color="#e4a13a", label="Holdout (25 q)", markersize=6)
ax.plot([int(k) for k in Ks], [P["K"][k]["merged"] for k in Ks], "D-",
        color="#c0392b", label="Pooled (50 q)", markersize=7, linewidth=2)
ax.axvline(10, color="gray", linestyle=":", linewidth=1)
ax.text(10.2, 41, "K = 10 chosen", fontsize=8, color="gray")
ax.set_xlabel("K (relations kept per hop)")
ax.set_ylabel("Hits@1 (%)")
ax.set_xticks([int(k) for k in Ks])
ax.set_ylim(20, 46)
ax.set_title("(a) Sensitivity to K", fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, linestyle=":")

ax2 = axes[1]
Ds = sorted(P["D"], key=lambda x: int(x))
ax2.plot([int(d) for d in Ds], [P["D"][d]["scan"] for d in Ds], "o--",
         color="#4c78a8", label="Sweep set (25 q)", markersize=6)
ax2.plot([int(d) for d in Ds], [P["D"][d]["hold"] for d in Ds], "s--",
         color="#e4a13a", label="Holdout (25 q)", markersize=6)
ax2.plot([int(d) for d in Ds], [P["D"][d]["merged"] for d in Ds], "D-",
         color="#c0392b", label="Pooled (50 q)", markersize=7, linewidth=2)
ax2.axvline(2, color="gray", linestyle=":", linewidth=1)
ax2.text(2.06, 38, "d_min = 2 chosen", fontsize=8, color="gray")
ax2.set_xlabel("d_min (minimum early-stopping depth)")
ax2.set_ylabel("Hits@1 (%)")
ax2.set_xticks([int(d) for d in Ds])
ax2.set_ylim(20, 42)
ax2.set_title("(b) Sensitivity to d_min", fontsize=11)
ax2.legend(fontsize=8, frameon=False)
ax2.grid(alpha=0.25, linestyle=":")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_sensitivity.png"), bbox_inches="tight")
# 同一张图另存一份"论文里的编号名"，这样包里的文件名与论文一致
fig.savefig(os.path.join(OUT, "Fig4_sensitivity.png"), bbox_inches="tight")
plt.close(fig)
print("  OK  fig3_sensitivity.png")

# ============================================================
# Figure 1 (file fig4_framework.png)  Framework
# ============================================================
fig, ax = plt.subplots(figsize=(10.6, 4.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 49)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fa", ec="#2c5f8d", fs=9, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
        facecolor=fc, edgecolor=ec, linewidth=1.2, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2, text="", color="#333", style="-|>", fs=7.6,
          off=(0, 1.6)):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, color=color,
        linewidth=1.3, linestyle="-" if style == "-|>" else "--"))
    if text:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text,
                ha="center", va="bottom", fontsize=fs, color=color)


box(1, 36, 15, 7.5, "Input question\n+ topic entities", fc="#f5f5f5", ec="#666")
box(19, 36, 17, 7.5, "Subgraph\nconstruction (\u2264 D_max hops)", fc="#e8f0f8",
    fs=8.4)
box(39, 36, 19, 7.5, "Path enumeration\nand pruning (W_max)", fc="#e8f0f8",
    fs=8.4)
box(61, 36, 17, 7.5, "Answer generation\n(PoG QA chain)", fc="#e8f0f8",
    bold=True, fs=8.4)
box(81, 36, 17, 7.5, "Output answer", fc="#f5f5f5", ec="#666")
for x1, x2 in ((16, 19), (36, 39), (58, 61), (78, 81)):
    arrow(x1, 39.7, x2, 39.7)

box(19, 22, 17, 8.5,
    "\u2460 Semantic\npre-filtering\nTop-K compression\n(per hop, shared cache)",
    fc="#fdeceb", ec="#c0392b", bold=True, fs=7.8)
box(61, 22, 17, 8.5,
    "\u2461 Sufficiency\njudgment\n(stop once the\nevidence suffices)\nd_min constraint",
    fc="#fdf6e3", ec="#e4a13a", fs=7.8)
arrow(27.5, 36, 27.5, 30.5, "before each hop", color="#c0392b", off=(0, 0.6),
      fs=7.0)
arrow(27.5, 22, 27.5, 17.5, "", color="#c0392b")
box(19, 10, 17, 7.5, "Compressed\ncandidate set\n(input length \u2193)",
    fc="#fafafa", ec="#999", fs=7.8)
arrow(36, 14, 45, 14, "", color="#c0392b")
arrow(45, 14, 45, 36, "", color="#c0392b")
ax.text(46.4, 25, "feeds\nlater\nexpansion", fontsize=7.2, color="#c0392b",
        va="center")

arrow(69.5, 36, 69.5, 30.5, "after each hop", color="#e4a13a", off=(0, 0.6),
      fs=7.0)
arrow(78, 26.5, 83.5, 35.5, "", color="#e4a13a")
ax.text(85.0, 31.0, "terminate early\nwhen sufficient", fontsize=7.2,
        color="#e4a13a", ha="left", va="center")
ax.text(50, 46.0, "SAGE-PoG overall framework", ha="center", fontsize=12.5,
        fontweight="bold")
ax.text(50, 3.2,
        "\u2460 main source of efficiency (G1 supplementary exploration "
        "\u221241.0%);  \u2461 fallback for recall risk (3 triggers in 300 questions)",
        ha="center", fontsize=7.8, color="#666")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_framework.png"), bbox_inches="tight")
# 同一张图另存一份"论文里的编号名"，这样包里的文件名与论文一致
fig.savefig(os.path.join(OUT, "Fig1_framework.png"), bbox_inches="tight")
plt.close(fig)
print("  OK  fig4_framework.png")

print()
print("  输出目录：", OUT)
print("  说明：每张图都存了两份 —— 内部名（figN_*.png）与论文编号名（FigN_*.png）")
print("        对应关系：fig1_accuracy=Fig2, fig2_phase=Fig3, "
      "fig3_sensitivity=Fig4, fig4_framework=Fig1")
for f in sorted(os.listdir(OUT)):
    print(f"     {f}  {os.path.getsize(os.path.join(OUT, f)):,} B")
