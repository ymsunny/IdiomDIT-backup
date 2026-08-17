#!/usr/bin/env python3
"""
visualize_sankey.py — D→I→LTE Sankey 流向图

用法：
  python analysis/visualize_sankey.py --lang-pair en-fa --model Qwen3.5-4B
  python analysis/visualize_sankey.py --lang-pair en-fa fa-en --model Qwen3.5-4B
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config, ALL_TARGETS, MODEL_NAME
from exclusion_utils import load_exclusion_set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import numpy as np


# ==================== 数据加载 ====================

def load_detection(cfg):
    """从 detection_score.json 加载 → {id_str: 0 or 1}"""
    path = cfg["eval_output"] / "detection_score.json"
    if not path.exists():
        print(f"    [警告] 不存在: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for item in data.get("results", []):
        iid = str(item.get("id", "")).strip()
        d = item.get("detection_score")
        if d is not None and iid:
            results[iid] = int(d)
    return results


def load_interpretation(cfg, group="p1"):
    """从 interpretation_p1_score.json 加载 → {id_str: 0 or 1}"""
    fname = f"interpretation_{group}_score.json"
    path = cfg["eval_output"] / fname
    if not path.exists():
        print(f"    [警告] 不存在: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for item in data.get("results", []):
        iid = str(item.get("id", "")).strip()
        score = item.get("interpretation_score")
        if score is not None and iid:
            results[iid] = int(score)
    return results


def _parse_yes_no(s):
    """'Yes - ...' / 'No - ...' → True/False/None"""
    s = str(s).strip().lower()
    if s.startswith("yes"): return True
    if s.startswith("no"):  return False
    return None


JUDGE_FILES = {
    "legacy":   "translation_lte_score.json",          # 旧 GPT-4o-mini
    "v4":       "translation_lte_v4_score.json",        # v4 GPT-4o-mini
    "v4-gpt52": "translation_lte_v4_gpt52_score.json",  # v4 GPT-5.2（论文 LTE 采用，默认）
}
DEFAULT_LTE_FILE = JUDGE_FILES["v4-gpt52"]


def load_lte_eval(cfg, prompt_filter="BasicPrompt", excluded_set=None,
                  lte_filename=DEFAULT_LTE_FILE, keep_unparseable=False):
    """从 LTE 评估结果文件加载（默认 translation_lte_score.json），派生 lte 和 fidelity。

    新格式（eval_translation_lte_old.py）：
      - literal_check  : 'Yes/No + evidence' 字符串
      - meaning_check  : 'Yes/No + evidence' 字符串
    派生：
      - lit_preserved = literal_check.startswith('Yes')
      - mean_conveyed = meaning_check.startswith('Yes')
      - lte           = 1 if (lit_preserved AND NOT mean_conveyed) else 0
      - fidelity      = mean_conveyed   (供 build_sankey_flows 四分类)

    excluded_set: {(id, prompt_type), ...}，命中则跳过
    """
    path = cfg["eval_output"] / lte_filename
    if not path.exists():
        print(f"    [警告] 不存在: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = []
    skipped_excluded = 0
    for item in data.get("results", []):
        pt = str(item.get("prompt_type", "")).strip()
        if prompt_filter and pt != prompt_filter:
            continue
        iid = item.get("id")
        if excluded_set and (int(iid), pt) in excluded_set:
            skipped_excluded += 1
            continue
        lit_preserved = _parse_yes_no(item.get("literal_check", ""))
        mean_conveyed = _parse_yes_no(item.get("meaning_check", ""))
        if lit_preserved is None or mean_conveyed is None:
            if keep_unparseable:
                # 数据量计数用：保留未解析项（lte/fidelity=None，build_sankey_flows 不计入流量）
                results.append({"id": str(iid).strip(), "prompt_type": pt,
                                "lte": None, "fidelity": None})
            continue
        lte = 1 if (lit_preserved and not mean_conveyed) else 0
        results.append({"id": str(iid).strip(),
                        "prompt_type": pt,
                        "lte": lte,
                        "fidelity": mean_conveyed})
    if skipped_excluded:
        print(f"    [excluded] 跳过 {skipped_excluded} 条（exclusion_key 命中）")
    return results


# ==================== Sankey 构建 ====================

def build_sankey_flows(detection, interpretation, lte_items):
    flows = defaultdict(int)
    total = 0
    for item in lte_items:
        iid = item["id"]
        d = detection.get(iid)
        i = interpretation.get(iid)
        lte = item["lte"]
        fid = item["fidelity"]
        if d is None or i is None or lte is None or fid is None:
            continue
        total += 1
        d_label = f"D={d}"
        di_label = f"D={d}, I={i}"
        if lte == 1 and fid == False:
            end = "LTE"
        elif lte == 0 and fid == True:
            end = "Correct"
        elif lte == 0 and fid == False:
            end = "Non-LTE but wrong"
        elif lte == 1 and fid == True:
            end = "LTE but meaning ok"
        else:
            end = "Other"
        flows[("All", d_label)] += 1
        flows[(d_label, di_label)] += 1
        flows[(di_label, end)] += 1
    return flows, total


# ==================== Sankey 绘制 ====================

STAGE_ORDER = {
    "All": 0, "D=0": 1, "D=1": 1,
    "D=0, I=0": 2, "D=0, I=1": 2, "D=1, I=0": 2, "D=1, I=1": 2,
    "LTE": 3, "Correct": 3,
    "Non-LTE but wrong": 3, "LTE but meaning ok": 3, "Other": 3,
}
SORT_KEY = {
    "All": 0, "D=1": 0, "D=0": 1,
    "D=1, I=1": 0, "D=1, I=0": 1, "D=0, I=1": 2, "D=0, I=0": 3,
    "Correct": 0, "LTE but meaning ok": 1,
    "Non-LTE but wrong": 2, "LTE": 3, "Other": 4,
}
NODE_COLORS = {
    "All": "#888780", "D=1": "#1D9E75", "D=0": "#D85A30",
    "D=1, I=1": "#1D9E75", "D=1, I=0": "#BA7517",
    "D=0, I=1": "#5DCAA5", "D=0, I=0": "#D85A30",
    "Correct": "#1D9E75", "LTE": "#E24B4A",
    "Non-LTE but wrong": "#D85A30", "LTE but meaning ok": "#BA7517",
    "Other": "#888780",
}

# 精简后的节点显示名（仅用于标签渲染，不改 flows/颜色等内部 key）
DISPLAY_NAMES = {
    "Non-LTE but wrong": "Non-LTE",
    "LTE but meaning ok": "LTE(ok)",
}


def draw_sankey_on_ax(ax, flows, total, stage_labels, title=None, n_total=None,
                      label_fontsize=10, stage_label_fontsize=12, title_fontsize=14,
                      x_positions=None):
    """Draw a Sankey diagram on the given axes (ax-relative coords).

    Reusable for both single-figure mode (draw_sankey) and grid mode
    (visualize_sankey_grid.py).
    """
    all_nodes = set()
    for (src, tgt), v in flows.items():
        if v > 0:
            all_nodes.add(src)
            all_nodes.add(tgt)

    layers = defaultdict(list)
    for n in all_nodes:
        layers[STAGE_ORDER.get(n, 0)].append(n)
    for layer in layers:
        layers[layer].sort(key=lambda n: SORT_KEY.get(n, 99))

    node_out = defaultdict(int)
    node_in = defaultdict(int)
    for (src, tgt), v in flows.items():
        node_out[src] += v
        node_in[tgt] += v
    node_value = {}
    for n in all_nodes:
        node_value[n] = max(node_out.get(n, 0), node_in.get(n, 0))

    if x_positions is None:
        x_positions = {0: 0.04, 1: 0.22, 2: 0.43, 3: 0.78}
    node_width = 0.018
    height_scale = 0.70 / max(total, 1)
    gap = 0.018

    node_pos = {}
    for layer_id in sorted(layers.keys()):
        nodes_in_layer = layers[layer_id]
        x = x_positions.get(layer_id, 0.5)
        total_h = sum(max(node_value[n] * height_scale, 0.01) for n in nodes_in_layer)
        total_gap = gap * max(len(nodes_in_layer) - 1, 0)
        y_start = 0.5 + (total_h + total_gap) / 2
        y = y_start
        for n in nodes_in_layer:
            h = max(node_value[n] * height_scale, 0.01)
            node_pos[n] = (x, y - h, h)
            y -= h + gap

    node_out_cursor = {}
    node_in_cursor = {}
    for n, (x, yt, h) in node_pos.items():
        node_out_cursor[n] = yt + h
        node_in_cursor[n] = yt + h

    for (src, tgt), v in sorted(flows.items(), key=lambda kv: -kv[1]):
        if v == 0 or src not in node_pos or tgt not in node_pos:
            continue
        ribbon_h = max(v * height_scale, 0.003)
        y0_top = node_out_cursor[src] - ribbon_h
        node_out_cursor[src] = y0_top
        y1_top = node_in_cursor[tgt] - ribbon_h
        node_in_cursor[tgt] = y1_top
        y0_bot, y1_bot = y0_top + ribbon_h, y1_top + ribbon_h
        x0 = node_pos[src][0] + node_width
        x1 = node_pos[tgt][0]
        xm = (x0 + x1) / 2
        color = NODE_COLORS.get(src, "#888780")
        verts = [(x0, y0_top), (xm, y0_top), (xm, y1_top), (x1, y1_top),
                 (x1, y1_bot), (xm, y1_bot), (xm, y0_bot), (x0, y0_bot), (x0, y0_top)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        patch = mpatches.PathPatch(MPath(verts, codes), facecolor=color, alpha=0.28,
                                   edgecolor="none", transform=ax.transAxes)
        ax.add_patch(patch)

    for n, (x, yt, h) in node_pos.items():
        color = NODE_COLORS.get(n, "#888780")
        rect = mpatches.FancyBboxPatch((x, yt), node_width, h, boxstyle="round,pad=0.002",
                                        facecolor=color, edgecolor="white", linewidth=0.8,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        count = node_value[n]
        if count == 0:
            continue  # 跳过空节点，避免 "D=0,I=0 0%" 这类无用标签互相挤压
        denom = n_total if n_total else total
        pct = count / denom * 100
        disp = DISPLAY_NAMES.get(n, n).replace(", ", ",")
        # "All" 节点恒为 100%，省略百分比（总数已在子图标题 n= 中给出）
        label_str = "All" if n == "All" else f"{disp} {pct:.1f}%"
        fontweight = "bold" if STAGE_ORDER.get(n, 0) == 3 else "normal"
        fontcolor = NODE_COLORS.get(n, "#333333") if STAGE_ORDER.get(n, 0) == 3 else "#333333"
        ax.text(x + node_width + 0.008, yt + h / 2, label_str,
                transform=ax.transAxes, fontsize=label_fontsize, va="center", ha="left",
                color=fontcolor, fontweight=fontweight)

    for layer_id, label in stage_labels.items():
        x = x_positions.get(layer_id, 0.5)
        ax.text(x + node_width / 2, 0.95, label, transform=ax.transAxes,
                fontsize=stage_label_fontsize, fontweight="bold", va="top", ha="center", color="#444441")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=20)


def draw_sankey(flows, total, title, stage_labels, output_path, n_total=None):
    """Single-figure Sankey: create fig, draw, save."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))
    draw_sankey_on_ax(ax, flows, total, stage_labels, title=title, n_total=n_total)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    Sankey 已保存: {output_path}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="D→I→LTE Sankey 流向图")
    parser.add_argument("--lang-pair", nargs="+", required=True, choices=ALL_TARGETS)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--judge", default="v4-gpt52", choices=list(JUDGE_FILES),
                        help="LTE judge 结果文件：v4-gpt52=v4 GPT-5.2(论文采用,默认) / "
                             "v4=v4 GPT-4o-mini / legacy=旧GPT-4o-mini")
    args = parser.parse_args()
    lte_filename = JUDGE_FILES[args.judge]
    print(f"[judge] {args.judge} -> {lte_filename}")

    print("=" * 62)
    print(f"  Sankey 流向图")
    print(f"  语言对: {args.lang_pair}  模型: {args.model}")
    print("=" * 62)

    for lp in args.lang_pair:
        cfg = get_config(lp, model=args.model)
        direction = cfg["direction"].replace("_to_", " → ").replace("_", " ")
        output_dir = cfg["eval_output"]
        os.makedirs(str(output_dir), exist_ok=True)

        print(f"\n  {direction}:")
        excluded_set = load_exclusion_set(lp, args.model)
        if excluded_set:
            print(f"    [exclusion_key] {len(excluded_set)} 条 (id, prompt_type) 将被过滤")
        detection = load_detection(cfg)
        interp_p1 = load_interpretation(cfg, "p1")
        interp_p2 = load_interpretation(cfg, "p2")
        lte_items = load_lte_eval(cfg, prompt_filter="BasicPrompt", excluded_set=excluded_set,
                                  lte_filename=lte_filename)

        print(f"    D={len(detection)} I(P1)={len(interp_p1)} I(P2)={len(interp_p2)} LTE={len(lte_items)}")

        if not lte_items:
            print(f"    [跳过] 无数据")
            continue

        stage_labels = {0: "All samples", 1: "Detection (D)", 2: "Interpretation", 3: "Translation (T)"}

        # P1
        flows, total = build_sankey_flows(detection, interp_p1, lte_items)
        if total > 0:
            draw_sankey(flows, total,
                        f"{direction} (n={total})",
                        {**stage_labels, 2: "Interpretation (P1)"},
                        str(output_dir / f"sankey_{lp}_P1.png"),
                        n_total=total)

        # P2
        flows, total = build_sankey_flows(detection, interp_p2, lte_items)
        if total > 0:
            draw_sankey(flows, total,
                        f"{direction} (n={total})",
                        {**stage_labels, 2: "Interpretation (I)"},
                        str(output_dir / f"sankey_{lp}_P2.png"),
                        n_total=total)

    print(f"\n  完成!")


if __name__ == "__main__":
    main()
