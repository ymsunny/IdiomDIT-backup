"""
visualize_prompt_mitigation_stacked.py — Prompt mitigation 堆叠条形图

每个 prompt 一根水平堆叠条 =
    [绿色: 相对 BasicPrompt 的 LTE 降低量 pp] + [红色: 剩余 LTE rate]
绿+红之和 = BasicPrompt 的 baseline LTE（虚线）

布局:
  每张图 = 1 个模型
  6 个语言对作 2×3 subplot
  顺序与 Sankey grid 一致:
    [en-fa,  fr-en,  ko-en]
    [fa-en,  fi-en,  ja-en]
  每 subplot 内 4 根横条（4 prompt，按 LTE 降序）

LTE 派生（与 Sankey 保持一致）:
  literal_check.startswith("Yes") AND meaning_check.startswith("No") → LTE=1
  不读 LLM 输出的 literal_translation_error 字段

输出:
  figures/prompt_mitigation_stacked_{model}.{png,pdf}

用法:
  python analysis/visualize_prompt_mitigation_stacked.py
  python analysis/visualize_prompt_mitigation_stacked.py --model Qwen3.5-9B
  python analysis/visualize_prompt_mitigation_stacked.py --output-dir figures
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config
from exclusion_utils import load_exclusion_set

MODELS = ['Qwen3.5-9B', 'Qwen3-8B', 'Qwen3.5-4B', 'Qwen3-4B']


def pretty_lp(lp):
    """en-fa -> En→Fa（与论文 caption 一致：首字母大写 + 箭头）。"""
    return "→".join(p.capitalize() for p in lp.split("-"))

LAYOUTS = {
    "2x3": {
        "grid": [['en-fa', 'fr-en', 'ko-en'],
                 ['fa-en', 'fi-en', 'ja-en']],
        # 每格宽高与 1x3_main 一致(4x3)，在 width=\textwidth 下字号与主图相同
        "figsize": (12, 3.6),
    },
    "1x3_main": {
        "grid": [['en-fa', 'fr-en', 'fi-en']],
        "figsize": (12, 3.0),
    },
    "1x3_remaining": {
        "grid": [['ko-en', 'fa-en', 'ja-en']],
        "figsize": (12, 3.0),
    },
}
STEM_SUFFIX = {"2x3": "", "1x3_main": "", "1x3_remaining": "_remaining"}

PROMPT_DISPLAY = {
    'BasicPrompt': 'BasicPrompt',
    'Meaning-FirstTranslationPrompt': 'Meaning-First',
    'SuperLiteralBiasMitigationPrompt': 'SuperMitigation',
    'Anti-LiteralConstraintPrompt': 'Anti-Literal',
}
PROMPT_ORDER = [
    'BasicPrompt',
    'Meaning-FirstTranslationPrompt',
    'SuperLiteralBiasMitigationPrompt',
    'Anti-LiteralConstraintPrompt',
]

COLOR_GREEN = '#1D9E75'
COLOR_RED = '#E24B4A'
COLOR_GRAY = '#B4B2A9'


def _parse_yes_no(s):
    """'Yes - ...' / 'No - ...' → True/False/None"""
    s = str(s).strip().lower()
    if s.startswith("yes"): return True
    if s.startswith("no"):  return False
    return None


JUDGE_FILES = {
    "legacy":   "translation_lte_score.json",          # 旧 GPT-4o-mini（默认，保持现状）
    "v4":       "translation_lte_v4_score.json",        # v4 GPT-4o-mini
    "v4-gpt52": "translation_lte_v4_gpt52_score.json",  # v4 GPT-5.2（论文 LTE 采用）
}
DEFAULT_LTE_FILE = JUDGE_FILES["legacy"]


def load_lte_by_prompt(lp, model, excluded_set, lte_filename=DEFAULT_LTE_FILE):
    """Return dict: prompt -> {lte_rate(%), lte_count, total}.

    派生 LTE = literal_check.startswith("Yes") AND meaning_check.startswith("No")
    Skip rows in excluded_set.
    """
    cfg = get_config(lp, model=model)
    f = cfg["eval_output"] / lte_filename
    if not f.is_file():
        return {}
    try:
        data = json.load(open(f, encoding='utf-8'))
    except Exception:
        return {}

    counts = defaultdict(lambda: {'total': 0, 'lte': 0, 'n_full': 0})
    skipped = 0
    for item in data.get('results', []):
        pt = str(item.get('prompt_type', '')).strip()
        if pt not in PROMPT_DISPLAY:
            continue
        iid = item.get('id')
        if excluded_set and (int(iid), pt) in excluded_set:
            skipped += 1
            continue
        counts[pt]['n_full'] += 1   # 数据量: exclusion 后全部实例(含 LTE 未解析)
        lit = _parse_yes_no(item.get('literal_check', ''))
        mean_conv = _parse_yes_no(item.get('meaning_check', ''))
        if lit is None or mean_conv is None:
            continue
        lte = 1 if (lit and not mean_conv) else 0
        counts[pt]['total'] += 1
        counts[pt]['lte'] += lte
    if skipped:
        print(f"    [excluded] {lp} × {model}: 跳过 {skipped} 条")

    result = {}
    for pt in PROMPT_ORDER:
        if pt in counts and counts[pt]['total'] > 0:
            c = counts[pt]
            result[pt] = {
                'lte_rate': round(c['lte'] / c['total'] * 100, 1),
                'lte_count': c['lte'],
                'total': c['total'],
                'n_full': c['n_full'],
            }
    return result


def draw_subplot(ax, lp, prompt_data, show_xlabel=True):
    """单个语言对 subplot (堆叠条形)"""
    if 'BasicPrompt' not in prompt_data:
        ax.text(0.5, 0.5, f'{lp}\n(no data)', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.axis('off')
        return

    baseline = prompt_data['BasicPrompt']['lte_rate']
    baseline_n = prompt_data['BasicPrompt'].get('n_full', prompt_data['BasicPrompt']['total'])  # 数据量(含未解析)
    sorted_prompts = sorted(prompt_data.keys(),
                            key=lambda p: -prompt_data[p]['lte_rate'])
    labels = [PROMPT_DISPLAY[p] for p in sorted_prompts]
    lte_rates = [prompt_data[p]['lte_rate'] for p in sorted_prompts]
    reductions = [round(baseline - r, 1) for r in lte_rates]

    bar_h = 0.4
    spacing = bar_h * 1.3  # gap = bar_h * 0.3
    y_pos = np.arange(len(labels)) * spacing

    ax.barh(y_pos, reductions, height=bar_h,
            color=COLOR_GREEN, zorder=3, edgecolor='none')
    ax.barh(y_pos, lte_rates, height=bar_h,
            left=reductions, color=COLOR_RED, zorder=3, edgecolor='none')

    # 拉满 x 轴：所有条形长度都 = baseline，紧贴右侧（仅留 ~4% 边距）。
    # 每个数据单位占更多像素 → 绿色段按比例一起变宽，多数 -pp 能放进绿色里。
    # 不改条形数值，只收紧坐标轴范围，数据不失真。
    max_x = max(r + l for r, l in zip(reductions, lte_rates)) * 1.04
    ax.set_xlim(0, max_x)

    pad = max_x * 0.012  # -pp 左侧内边距（占面板固定比例，跨面板视觉一致）
    for i in range(len(labels)):
        red_center = reductions[i] + lte_rates[i] / 2
        ax.text(red_center, y_pos[i], f'{lte_rates[i]}%',
                ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=4)
        if reductions[i] > 0.5:
            # -pp 一律从条形最左端起、左对齐（白字）。红色 % 居中在红条里，
            # 二者不会相撞（% 中心始终远在 -pp 标签右侧）。
            ax.text(pad, y_pos[i], f'-{reductions[i]}pp',
                    ha='left', va='center', fontsize=10,
                    fontweight='bold', color='white', zorder=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight='500')
    ax.set_title(f'{pretty_lp(lp)}  (n={baseline_n})', fontsize=15, fontweight='bold', pad=8)
    ax.invert_yaxis()
    ax.set_ylim(y_pos[-1] + bar_h * 0.7, y_pos[0] - bar_h * 0.7)
    ax.grid(axis='x', alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)


def render_model(model, output_dir, layout="2x3", lte_filename=DEFAULT_LTE_FILE, judge_tag=None):
    print(f"\n=== {model} ({layout}) ===")
    spec = LAYOUTS[layout]
    grid = spec["grid"]
    n_rows, n_cols = len(grid), len(grid[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=spec["figsize"])
    import numpy as np
    axes = np.array(axes).reshape(n_rows, n_cols)

    for r in range(n_rows):
        for c in range(n_cols):
            lp = grid[r][c]
            excluded_set = load_exclusion_set(lp, model)
            data = load_lte_by_prompt(lp, model, excluded_set, lte_filename=lte_filename)
            show_xlabel = (r == n_rows - 1)  # 最下排显示 xlabel
            draw_subplot(axes[r, c], lp, data, show_xlabel=show_xlabel)

    # Legend removed: color meaning (red/green) is now explained inline in the LaTeX caption.
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = STEM_SUFFIX.get(layout, f"_{layout}")
    stem = f'prompt_mitigation_stacked_{model}{suffix}'
    if judge_tag:
        stem += f"_{judge_tag}"
    for ext in ('png', 'pdf'):
        out = Path(output_dir) / f'{stem}.{ext}'
        fig.savefig(out, dpi=160 if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f'  [{ext}] {out}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', default=MODELS,
                        help='一个或多个模型名（默认全部 4 个）')
    parser.add_argument('--output-dir', default='figures',
                        help='输出目录（默认 figures/）')
    parser.add_argument('--layout', default='2x3', choices=list(LAYOUTS.keys()),
                        help='网格布局：2x3 全6对（默认）/ 1x3_main / 1x3_remaining')
    parser.add_argument('--judge', default='legacy', choices=list(JUDGE_FILES),
                        help='LTE judge 结果文件：legacy=旧GPT-4o-mini(默认,现状) / '
                             'v4=v4 GPT-4o-mini / v4-gpt52=v4 GPT-5.2(论文采用)')
    parser.add_argument('--tag-judge', action='store_true',
                        help='文件名追加 judge 后缀（如 _v4gpt52），避免覆盖默认图')
    args = parser.parse_args()

    lte_filename = JUDGE_FILES[args.judge]
    judge_tag = args.judge.replace('-', '') if args.tag_judge else None
    print(f'[judge] {args.judge} -> {lte_filename}' + (f'  (tag=_{judge_tag})' if judge_tag else ''))
    for model in args.model:
        render_model(model, args.output_dir, layout=args.layout,
                     lte_filename=lte_filename, judge_tag=judge_tag)

    print(f'\nAll figures in: {args.output_dir}/')


if __name__ == '__main__':
    main()
