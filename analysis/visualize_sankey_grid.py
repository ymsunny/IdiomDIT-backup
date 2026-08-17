#!/usr/bin/env python3
"""
visualize_sankey_grid.py — 把 6 个语言对的 D→I→LTE Sankey 拼到一张图。

每个 model 会输出两张图：
  figures/sankey_grid_{model}_P1.{png,pdf}
  figures/sankey_grid_{model}_P2.{png,pdf}

Layout: 默认 3x2（3 行 × 2 列，大字版，论文定稿）；--layout 2x3 为紧凑版。
PDF 矢量化方便 LaTeX 引用，PNG 用于快速预览。

用法：
  python analysis/visualize_sankey_grid.py --model Qwen3.5-9B
  python analysis/visualize_sankey_grid.py --model Qwen3.5-9B Qwen3-8B Qwen3.5-4B Qwen3-4B
  python analysis/visualize_sankey_grid.py --model Qwen3.5-9B --output-dir figures
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config, ALL_TARGETS, MODEL_NAME
from exclusion_utils import load_exclusion_set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_sankey import (
    load_detection, load_interpretation, load_lte_eval,
    build_sankey_flows, draw_sankey_on_ax,
    JUDGE_FILES, DEFAULT_LTE_FILE,
)


def pretty_lp(lp):
    """en-fa -> En→Fa（与论文 caption 一致：首字母大写 + 箭头）。"""
    return "→".join(p.capitalize() for p in lp.split("-"))


# 布局预设。字号是脚本内 pt，乘以「论文宽 6.3in / figsize 宽」的缩放系数得到印出来的实际 pt。
#   2x3：紧凑（占版面矮），但 4 阶段挤在一行，标签印出来只有 ~5-6pt
#   3x2：面板更宽，标签印出来 ~8-9pt（接近 caption/正文），代价是图更高
LAYOUTS = {
    "2x3": {
        "grid": [["en-fa", "fr-en", "ko-en"],
                 ["fa-en", "fi-en", "ja-en"]],
        "figsize": (22, 10),
        "label_fs": 12, "stage_fs": 15, "title_fs": 17,
        "x_positions": None,
    },
    "3x2": {
        "grid": [["en-fa", "fa-en"],
                 ["fr-en", "fi-en"],
                 ["ko-en", "ja-en"]],
        "figsize": (12, 13),
        "label_fs": 16, "stage_fs": 19, "title_fs": 21,
        "x_positions": None,
    },
    # 1×3 主图（与 2x3 单行完全一致的字号/尺寸，只放 3 个语言对）
    "1x3_main": {
        "grid": [["en-fa", "fr-en", "fi-en"]],
        "figsize": (22, 5),
        "label_fs": 12, "stage_fs": 15, "title_fs": 17,
        "x_positions": None,
    },
    # 1×3 附录（另外 3 个语言对，同上）
    "1x3_remaining": {
        "grid": [["ko-en", "fa-en", "ja-en"]],
        "figsize": (22, 5),
        "label_fs": 12, "stage_fs": 15, "title_fs": 17,
        "x_positions": None,
    },
}


def render_grid(model, interp_group, output_dir, layout="2x3", lte_filename=DEFAULT_LTE_FILE,
                judge_tag=None):
    spec = LAYOUTS[layout]
    grid_order = spec["grid"]
    n_rows, n_cols = len(grid_order), len(grid_order[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=spec["figsize"])
    # Normalise to 2-D array regardless of shape
    import numpy as np
    axes = np.array(axes).reshape(n_rows, n_cols)

    drew = 0
    for r in range(n_rows):
        for c in range(n_cols):
            lp = grid_order[r][c]
            ax = axes[r, c]
            cfg = get_config(lp, model=model)

            excluded_set = load_exclusion_set(lp, model)
            detection = load_detection(cfg)
            interp = load_interpretation(cfg, interp_group)
            lte_items = load_lte_eval(cfg, prompt_filter="BasicPrompt", excluded_set=excluded_set,
                                      lte_filename=lte_filename, keep_unparseable=True)

            flows, total = build_sankey_flows(detection, interp, lte_items)
            n_full = len(lte_items)  # 数据量(含 LTE 未解析)，面板 n 用此而非有效流量 total

            if total == 0:
                ax.text(0.5, 0.5, f"{pretty_lp(lp)}\n(no data)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=14, color="#999999")
                ax.axis("off")
                continue

            stage_labels = {
                0: "All", 1: "D", 2: "I", 3: "T",
            }
            title = f"{pretty_lp(lp)}  (n={n_full})"
            draw_sankey_on_ax(
                ax, flows, total, stage_labels, title=title, n_total=total,
                label_fontsize=spec["label_fs"],
                stage_label_fontsize=spec["stage_fs"],
                title_fontsize=spec["title_fs"],
                x_positions=spec.get("x_positions"),
            )
            drew += 1

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(output_dir, exist_ok=True)
    # 文件名映射：2x3/1x3_main → 无后缀（论文引用标准名）；其余带后缀
    STEM_SUFFIX = {
        "2x3": "",
        "1x3_main": "",
        "1x3_remaining": "_remaining",
        "3x2": "_3x2",
    }
    suffix = STEM_SUFFIX.get(layout, f"_{layout}")
    stem = f"sankey_grid_{model}_{interp_group.upper()}{suffix}"
    if judge_tag:
        stem += f"_{judge_tag}"
    for ext in ("png", "pdf"):
        out = Path(output_dir) / f"{stem}.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  [{ext}] {out}  ({drew} panels)")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="6 语言对 cascade Sankey 拼图（每 model 一张）")
    parser.add_argument("--model", nargs="+", default=[MODEL_NAME], help="一个或多个模型名")
    parser.add_argument("--interp-group", nargs="+", default=["p1", "p2"],
                        choices=["p1", "p2"], help="interpretation 分组")
    parser.add_argument("--output-dir", default="figures",
                        help="输出目录（默认 figures/）")
    parser.add_argument("--layout", default="2x3", choices=list(LAYOUTS.keys()),
                        help="网格布局：2x3 全6对（默认）/ 1x3_main 前3对 / 1x3_remaining 后3对 / 3x2 大字版")
    parser.add_argument("--judge", default="v4-gpt52", choices=list(JUDGE_FILES),
                        help="LTE judge 结果文件：v4-gpt52=v4 GPT-5.2(论文采用,默认) / "
                             "v4=v4 GPT-4o-mini / legacy=旧GPT-4o-mini")
    parser.add_argument("--tag-judge", action="store_true",
                        help="文件名追加 judge 后缀（如 _v4gpt52），避免覆盖默认图")
    args = parser.parse_args()
    lte_filename = JUDGE_FILES[args.judge]
    judge_tag = args.judge.replace("-", "") if args.tag_judge else None
    print(f"[judge] {args.judge} -> {lte_filename}" + (f"  (tag=_{judge_tag})" if judge_tag else ""))

    for model in args.model:
        print(f"\n=== {model} ===")
        for grp in args.interp_group:
            render_grid(model, grp, args.output_dir, layout=args.layout,
                        lte_filename=lte_filename, judge_tag=judge_tag)


if __name__ == "__main__":
    main()
