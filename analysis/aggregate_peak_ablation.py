"""aggregate_peak_ablation.py — Peak-layer LTD ablation 聚合脚本(reviewer #3 回复).

回应审稿意见:
> The all-layer ablation is quite coarse. Consider more localised causal analyses,
> for example ablating at the peak probing layer or nearby layers rather than all layers.

设计:
  - 保持 Table 5 (Finding 8) 中的 mixed-prompt 池(全 4 prompt)+ Group A "know-but-error strict"
  - 唯一改动:LTD 消融只在 peak grouped_cv layer 上做,不在全层
  - Random control:同一 peak layer × 3 seed({42,43,44})

Peak grouped_cv layers (from probing_balanced_strict.json):
  en-fa=15  fa-en=16  fi-en=17  fr-en=31  ja-en=11  ko-en=11

读取路径:
  results/{lp}/Qwen3.5-9B/evaluation/
    ablation_v4gpt52_peak_baseline_score.json               ← baseline(peak-layer)
    ablation_v4gpt52_peak_ablation_score.json               ← LTD peak-layer 消融
    ablation_L{peak}_dir_ablate_allprompts__rand_s{42,43,44}_ablation_score.json

指标同 basic_only:
  ΔLTE = LTE_rate(baseline) - LTE_rate(ablation)     [正数 = 消融降低了 LTE, pp]
  Fix = base LTE=1 且 abl LTE=0 的条数
  Harm = base LTE=0 且 abl LTE=1 的条数
  95% CI = 2000-iter paired bootstrap

输出:
  analysis/output/peak_layer_table5.csv
  analysis/output/peak_layer_table5.tex
  控制台:全表 + 与论文 all-layer pooled Table 5 对比 + 严谨解读
"""
import os, sys, io, json, statistics, random
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BOOT_N = 2000
BOOT_SEED = 20260710

MODEL = 'Qwen3.5-9B'
LANGS = [('ja-en', 'Ja→En'), ('en-fa', 'En→Fa'), ('fa-en', 'Fa→En'),
         ('fr-en', 'Fr→En'), ('ko-en', 'Ko→En'), ('fi-en', 'Fi→En')]
SEEDS = [42, 43, 44]
BASE = Path(__file__).resolve().parent.parent

PEAK_LAYER = {
    'en-fa': 15, 'fa-en': 16, 'fi-en': 17,
    'fr-en': 31, 'ja-en': 11, 'ko-en': 11,
}

# 论文 all-layer pooled 版 Table 5 数字(4123 PDF submission)
# 注意论文用符号 ΔLTE = ablation - baseline (负数 = LTE 下降),这里换算成正数 = 下降
PAPER_POOLED_ALL_LAYER = {
    'ja-en': {'n': 220, 'delta_ltd': -2.5,  'delta_rand': +5.6,  'fix_harm': '20:15'},
    'en-fa': {'n': 42,  'delta_ltd': +5.6,  'delta_rand': +16.8, 'fix_harm': '3:5'},
    'fa-en': {'n': 41,  'delta_ltd': +9.8,  'delta_rand': +10.1, 'fix_harm': '7:11'},
    'fr-en': {'n': 49,  'delta_ltd': +8.2,  'delta_rand': +4.1,  'fix_harm': '3:7'},
    'ko-en': {'n': 17,  'delta_ltd': +35.3, 'delta_rand': +19.6, 'fix_harm': '0:6'},
    'fi-en': {'n': 13,  'delta_ltd': +23.1, 'delta_rand': -9.3,  'fix_harm': '3:6'},
}


def load_score(lp, prefix):
    p = BASE / 'results' / lp / MODEL / 'evaluation' / f'{prefix}_score.json'
    if not p.exists():
        return None, 0
    d = json.load(open(p, encoding='utf-8'))
    labels = {}
    for r in d['results']:
        iid = str(r.get('id'))
        key = (iid, r.get('prompt_type', ''))
        labels[key] = r.get('literal_translation_error')
    return labels, len(d['results'])


def lte_rate(labels):
    valid = [v for v in labels.values() if v is not None]
    if not valid: return None
    return sum(1 for v in valid if v is True) / len(valid)


def paired_fix_harm(base_labels, abl_labels):
    fix = harm = matched = 0
    for k, b in base_labels.items():
        a = abl_labels.get(k)
        if b is None or a is None: continue
        matched += 1
        if b is True and a is False: fix += 1
        elif b is False and a is True: harm += 1
    return fix, harm, matched


def paired_delta_pp_bootstrap(base_labels, abl_labels, n_boot=BOOT_N, seed=BOOT_SEED):
    pairs = []
    for k, b in base_labels.items():
        a = abl_labels.get(k)
        if b is None or a is None: continue
        pairs.append((int(b is True), int(a is True)))
    if not pairs: return (None, None, None)
    n = len(pairs)
    delta_obs = (sum(p[0] for p in pairs) - sum(p[1] for p in pairs)) / n * 100
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = pairs
        d = (sum(s[i][0] for i in idx) - sum(s[i][1] for i in idx)) / n * 100
        boots.append(d)
    boots.sort()
    ci_low = boots[int(0.025 * n_boot)]
    ci_high = boots[int(0.975 * n_boot) - 1]
    return delta_obs, ci_low, ci_high


def aggregate_one(lp):
    L = PEAK_LAYER[lp]
    base_labels, n_base = load_score(lp, 'ablation_v4gpt52_peak_baseline')
    ltd_labels, n_ltd = load_score(lp, 'ablation_v4gpt52_peak_ablation')
    if base_labels is None or ltd_labels is None:
        return None

    r_base = lte_rate(base_labels)
    r_ltd = lte_rate(ltd_labels)

    fix_ltd, harm_ltd, matched_ltd = paired_fix_harm(base_labels, ltd_labels)
    delta_ltd, ci_ltd_lo, ci_ltd_hi = paired_delta_pp_bootstrap(base_labels, ltd_labels)

    rand_deltas, rand_cis, rand_fix_harm = [], [], []
    for s in SEEDS:
        rand_labels, _ = load_score(
            lp, f'ablation_L{L}_dir_ablate_allprompts__rand_s{s}_ablation')
        if rand_labels is None: continue
        d, lo, hi = paired_delta_pp_bootstrap(base_labels, rand_labels, seed=BOOT_SEED + s)
        if d is not None:
            rand_deltas.append(d); rand_cis.append((lo, hi))
        f, h, m = paired_fix_harm(base_labels, rand_labels)
        rand_fix_harm.append((f, h, m))

    delta_rand_mean = statistics.mean(rand_deltas) if rand_deltas else None
    delta_rand_std = statistics.stdev(rand_deltas) if len(rand_deltas) > 1 else None

    return {
        'lp': lp, 'peak_L': L,
        'n': n_base,
        'lte_rate_baseline': r_base, 'lte_rate_ltd': r_ltd,
        'delta_ltd_pp': delta_ltd, 'ci_ltd_pp': (ci_ltd_lo, ci_ltd_hi),
        'delta_rand_pp': delta_rand_mean, 'delta_rand_pp_std': delta_rand_std,
        'delta_rand_pp_perseed': rand_deltas, 'ci_rand_perseed_pp': rand_cis,
        'fix_harm_ltd': f'{fix_ltd}:{harm_ltd}',
        'fix_harm_rand_perseed': [f'{f}:{h}' for f, h, _ in rand_fix_harm],
    }


def fmt_pp(v):
    if v is None: return '  n/a '
    sign = '+' if v > 0 else ''
    return f'{sign}{v:.1f} pp'


def main():
    out_dir = BASE / 'analysis' / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print('=' * 118)
    print(f'{"方向":<8} {"L":>3} {"n":>4} {"LTE_base":>10} {"ΔLTE(LTD peak)":>16} '
          f'{"ΔLTE(rand peak,3s)":>20} {"Fix:Harm(LTD)":>14}')
    print('-' * 118)
    for lp, disp in LANGS:
        m = aggregate_one(lp)
        if m is None:
            print(f'{disp:<8} {"—":>3} {"—":>4} {"skipped":>10}')
            rows.append({'lp': lp, 'disp': disp, 'skipped': True})
            continue
        rows.append({**m, 'disp': disp, 'skipped': False})
        r_base_str = f'{m["lte_rate_baseline"]*100:.1f}%' if m['lte_rate_baseline'] is not None else 'n/a'
        d_rand_str = fmt_pp(m['delta_rand_pp'])
        if m.get('delta_rand_pp_std') is not None:
            d_rand_str = f'{d_rand_str} (±{m["delta_rand_pp_std"]:.1f})'
        print(f'{disp:<8} {m["peak_L"]:>3} {m["n"]:>4} {r_base_str:>10} {fmt_pp(m["delta_ltd_pp"]):>16} '
              f'{d_rand_str:>20} {m["fix_harm_ltd"]:>14}')
    print('=' * 118)

    print('\n[对比] 论文 Table 5 (pooled 4 prompt × all-layer ablation)')
    print(f'{"方向":<8} {"n":>4} {"ΔLTE(LTD all)":>15} {"ΔLTE(rand all)":>15} {"Fix:Harm":>10}')
    print('-' * 65)
    for lp, disp in LANGS:
        pp = PAPER_POOLED_ALL_LAYER.get(lp, {})
        n = pp.get('n', '—')
        d_ltd = pp.get('delta_ltd'); d_rand = pp.get('delta_rand')
        fh = pp.get('fix_harm', '—')
        print(f'{disp:<8} {n!s:>4} {fmt_pp(d_ltd):>15} {fmt_pp(d_rand):>15} {fh:>10}')
    print('=' * 65)

    # ---- CSV ----
    import csv
    csv_p = out_dir / 'peak_layer_table5.csv'
    with open(csv_p, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'peak_layer', 'n',
                    'LTE_rate_baseline', 'LTE_rate_LTD',
                    'delta_LTE_LTD_pp', 'delta_LTE_LTD_ci_low', 'delta_LTE_LTD_ci_high',
                    'delta_LTE_rand_pp_mean', 'delta_LTE_rand_pp_std',
                    'delta_LTE_rand_pp_perseed_42_43_44',
                    'fix_harm_LTD', 'fix_harm_rand_perseed_42_43_44',
                    'paper_all_layer_n', 'paper_all_layer_delta_LTD_pp',
                    'paper_all_layer_delta_rand_pp'])
        for r in rows:
            lp = r['lp']
            pp = PAPER_POOLED_ALL_LAYER.get(lp, {})
            if r.get('skipped'):
                w.writerow([lp, PEAK_LAYER.get(lp, ''), 'skipped'])
            else:
                perseed = ';'.join(f'{d:+.2f}' for d in r.get('delta_rand_pp_perseed', []))
                fh_rand = ';'.join(r.get('fix_harm_rand_perseed', []))
                ci_lo, ci_hi = r.get('ci_ltd_pp', (None, None))
                w.writerow([lp, r['peak_L'], r['n'],
                            f'{r["lte_rate_baseline"]:.4f}' if r['lte_rate_baseline'] is not None else '',
                            f'{r["lte_rate_ltd"]:.4f}' if r['lte_rate_ltd'] is not None else '',
                            f'{r["delta_ltd_pp"]:+.2f}' if r['delta_ltd_pp'] is not None else '',
                            f'{ci_lo:+.2f}' if ci_lo is not None else '',
                            f'{ci_hi:+.2f}' if ci_hi is not None else '',
                            f'{r["delta_rand_pp"]:+.2f}' if r['delta_rand_pp'] is not None else '',
                            f'{r["delta_rand_pp_std"]:.2f}' if r.get('delta_rand_pp_std') is not None else '',
                            perseed,
                            r.get('fix_harm_ltd', ''),
                            fh_rand,
                            pp.get('n', ''), pp.get('delta_ltd', ''), pp.get('delta_rand', '')])
    print(f'\nCSV: {csv_p}')

    # ---- LaTeX table ----
    tex_p = out_dir / 'peak_layer_table5.tex'
    with open(tex_p, 'w', encoding='utf-8') as f:
        f.write('% Auto-generated: analysis/aggregate_peak_ablation.py\n')
        f.write('% Peak-layer paired LTD ablation (Qwen3.5-9B, mixed prompts).\n')
        f.write('% Response to reviewer #3 "all-layer is too coarse; try peak probing layer".\n')
        f.write('\\begin{table}[!t]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n')
        f.write('\\begin{tabular}{lrrrrr}\n\\toprule\n')
        f.write(' & \\textbf{Peak L} & \\textbf{$n$} & \\textbf{$\\Delta$ LTE (LTD)} & '
                '\\textbf{$\\Delta$ LTE (rand., 3-seed)} & \\textbf{Fix:Harm} \\\\\n')
        f.write('\\midrule\n')
        for r in rows:
            if r.get('skipped'):
                f.write(f'{r["disp"]} & --- & --- & \\multicolumn{{3}}{{c}}{{skipped}} \\\\\n')
                continue
            def _pp(v):
                if v is None: return '---'
                return f'${v:+.1f}$ pp'
            rand_str = _pp(r["delta_rand_pp"])
            if r.get('delta_rand_pp_std') is not None:
                rand_str = f'${r["delta_rand_pp"]:+.1f} \\pm {r["delta_rand_pp_std"]:.1f}$ pp'
            f.write(f'{r["disp"]} & {r["peak_L"]} & {r["n"]} & {_pp(r["delta_ltd_pp"])} & '
                    f'{rand_str} & {r["fix_harm_ltd"]} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write('\\caption{Peak-layer paired LTD ablation on Group~A (Qwen3.5-9B, mixed prompts). '
                'Peak L is the leakage-safe grouped-CV peak layer per direction. '
                '$\\Delta$~LTE is positive when ablation reduces LTE (paired bootstrap over Group~A). '
                'Random-direction column is the mean $\\pm$ SD over three seeds \\{42,43,44\\} ablated at the same layer.}\n')
        f.write('\\label{tab:peak_layer_ablation}\n\\end{table}\n')
    print(f'LaTeX: {tex_p}')

    # ---- 带 CI 的全表 ----
    print('\n' + '=' * 100)
    print('Paired ΔLTE with 95% bootstrap CI (LTD peak-layer). Positive = ablation reduces LTE.')
    print('=' * 100)
    for r in rows:
        if r.get('skipped'): continue
        d_ltd = r['delta_ltd_pp']; lo, hi = r['ci_ltd_pp']
        d_rand = r['delta_rand_pp']
        print(f'  {r["disp"]:<8} L={r["peak_L"]:>2} n={r["n"]:>3}  '
              f'ΔLTE(LTD)={d_ltd:+.1f} pp [95% CI {lo:+.1f}, {hi:+.1f}]  '
              f'| ΔLTE(rand)={d_rand:+.1f} pp (3-seed mean)')

    # ---- 严谨解读 ----
    print('\n' + '=' * 100)
    print('Verdict (does the LTD 95% CI cover the rand mean? → cannot reject "LTD ≈ rand")')
    print('=' * 100)
    powered = [r for r in rows if not r.get('skipped') and r['n'] >= 30]
    marginal = [r for r in rows if not r.get('skipped') and 15 <= r['n'] < 30]
    underpowered = [r for r in rows if not r.get('skipped') and r['n'] < 15]

    def _verdict(r):
        d_ltd = r['delta_ltd_pp']; lo, hi = r['ci_ltd_pp']
        d_rand = r['delta_rand_pp']
        if None in (d_ltd, d_rand, lo, hi): return '(N/A)'
        if lo <= d_rand <= hi:
            return 'LTD ≈ rand (CI covers rand mean; cannot reject null)'
        elif d_ltd > d_rand:
            return 'LTD > rand (CI does NOT cover rand; LTD is stronger)'
        else:
            return 'rand > LTD (CI does NOT cover rand; rand is stronger)'

    if powered:
        print('  Well-powered (n ≥ 30):')
        for r in powered:
            print(f'    {r["disp"]:<8} L={r["peak_L"]:>2} n={r["n"]:>3}  →  {_verdict(r)}')
    if marginal:
        print('  Marginal (15 ≤ n < 30, CI 会很宽):')
        for r in marginal:
            print(f'    {r["disp"]:<8} L={r["peak_L"]:>2} n={r["n"]:>3}  →  {_verdict(r)}')
    if underpowered:
        print('  Underpowered (n < 15):')
        for r in underpowered:
            print(f'    {r["disp"]:<8} L={r["peak_L"]:>2} n={r["n"]:>3}  →  {_verdict(r)}')

    # ---- Peak vs all-layer 头对头 ----
    print('\n' + '=' * 100)
    print('Head-to-head: peak-layer (this run) vs all-layer (paper Table 5)')
    print('=' * 100)
    print(f'{"方向":<8} {"n":>4} {"peak L":>7} '
          f'{"peak ΔLTD":>10} {"all ΔLTD":>10} {"peak ΔRand":>11} {"all ΔRand":>10}')
    print('-' * 78)
    for r in rows:
        if r.get('skipped'): continue
        lp = r['lp']
        pp = PAPER_POOLED_ALL_LAYER.get(lp, {})
        print(f'  {r["disp"]:<6} {r["n"]:>4} {r["peak_L"]:>7} '
              f'{r["delta_ltd_pp"]:+9.1f} {pp.get("delta_ltd", "—"):>+9.1f} '
              f'{r["delta_rand_pp"]:+10.1f} {pp.get("delta_rand", "—"):>+9.1f}')

    print('\nOutput dir:', out_dir)


if __name__ == '__main__':
    main()
