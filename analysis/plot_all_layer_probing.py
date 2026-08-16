"""
plot_all_layer_probing.py — 所有层 probing 曲线 (fig:all_layer_probing)

为每个语言对画一条 balanced_subsample 准确率 vs 层 ID 的曲线，
6 个语言对叠在一张图上，shaded band 表示 ±1 std。

用法:
  python analysis/plot_all_layer_probing.py
  python analysis/plot_all_layer_probing.py --model Qwen3-8B
  python analysis/plot_all_layer_probing.py --output-dir figures/Qwen3.5-9B
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ALL_TARGETS

LABELS = {
    "en-fa": "En→Fa", "fa-en": "Fa→En", "fr-en": "Fr→En",
    "fi-en": "Fi→En", "ko-en": "Ko→En", "ja-en": "Ja→En",
}
BASE = Path(__file__).resolve().parent.parent


def load_curve(lang_pair, model, group_b_suffix, exp_name=""):
    mech = BASE / "results" / lang_pair / model / "mechanistic"
    if exp_name:
        mech = mech / exp_name
    path = mech / f"probing_balanced{group_b_suffix}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    layers, bal_acc, bal_acc_std = [], [], []
    for item in sorted(data["results"], key=lambda x: x["layer"]):
        m = item["balanced_subsample"]
        layers.append(item["layer"])
        bal_acc.append(m["balanced_accuracy"])
        bal_acc_std.append(m.get("balanced_accuracy_std", 0.0))
    return layers, bal_acc, bal_acc_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--group-b", default="strict", choices=["strict", "loose"])
    parser.add_argument("--exp-name", default="",
                        help="读 mechanistic/{exp_name}/ 下的 probing（如 v4_gpt52）")
    parser.add_argument("--tag", default="",
                        help="输出文件名追加后缀（如 v4gpt52），避免覆盖默认图")
    parser.add_argument("--output-dir", default=None,
                        help="默认 figures/{model}")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"figures/{args.model}"

    suffix = "_strict" if args.group_b == "strict" else ""

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # figsize 宽 ≈ \columnwidth(~3.3in)，配 ~9pt 字体，在 width=\columnwidth 下近 1:1，
    # 字号与正文一致（不再被压到 ~4pt）。
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=240)
    colors = plt.get_cmap("tab10")

    for idx, lp in enumerate(ALL_TARGETS):
        layers, bal_acc, std = load_curve(lp, args.model, suffix, args.exp_name)
        c = colors(idx)
        ax.plot(layers, bal_acc, marker="o", markersize=2.6, linewidth=1.7,
                label=LABELS[lp], color=c)
        lower = [max(0.0, m - s) for m, s in zip(bal_acc, std)]
        upper = [min(1.0, m + s) for m, s in zip(bal_acc, std)]
        ax.fill_between(layers, lower, upper, color=c, alpha=0.10, linewidth=0)

    ax.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.45, 0.90)
    n_layers = max(layers) + 1
    ax.set_xlim(0, n_layers - 1)
    ax.set_xticks([0, 5, 10, 15, 20, 25, n_layers - 1])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.83])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"probing_all_layers_balanced_accuracy_{args.model}"
    if args.tag:
        stem += f"_{args.tag}"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"  [{ext}] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
