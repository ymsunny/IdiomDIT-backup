"""aggregate_naivepeak_ablation.py — Naive-peak(Table 3 Peak L)LTD ablation 聚合脚本.

回应审稿意见 W4:
> The all-layer ablation is quite coarse. Consider more localised causal analyses,
> for example ablating at the peak probing layer or nearby layers rather than all layers.

论文自己的 "peak probing layer" 定义 = Table 3 印的 Peak L 列(naive
balanced_subsample 峰,Figure 5 同一定义),本实验即在这些层做单层消融:
  en-fa=27  fa-en=8  fr-en=20  fi-en=27  ko-en=16  ja-en=21

与早前 grouped-CV 峰实验(aggregate_peak_ablation.py, L15/16/17/31/11/11)完全独立。

设计:保持 Table 5 (Finding 8) 的 mixed-prompt 池 + Group A "know-but-error strict",
唯一改动 = 消融只在 naive peak layer 上做。Random control:同层 × 3 seed({42,43,44})。

读取路径:
  results/{lp}/Qwen3.5-9B/evaluation/
    ablation_v4gpt52_naivepeak_baseline_score.json
    ablation_v4gpt52_naivepeak_ablation_score.json
    ablation_L{naive_peak}_dir_ablate_allprompts__rand_s{42,43,44}_ablation_score.json

指标同 Table 5:
  ΔLTE = LTE_rate(ablation) - LTE_rate(baseline)   [论文符号:负数 = 消融降低 LTE]
  Fix = base LTE=1 且 abl LTE=0;Harm = base LTE=0 且 abl LTE=1
  95% CI = 2000-iter paired bootstrap

输出:
  analysis/output/naivepeak_table5.csv
  analysis/output/naivepeak_table5.tex
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

# Naive peak layers = paper Table 3 "Peak L" 列
PEAK_LAYER = {
    'en-fa': 27, 'fa-en': 8, 'fi-en': 27,
    'fr-en': 20, 'ja-en': 21, 'ko-en': 16,
}

# 论文 all-layer pooled Table 5 数字(4123 PDF submission;负数 = LTE 下降)
PAPER_POOLED_ALL_LAYER = {
    'ja-en': {'n': 204, 'delta_ltd': +2.5,  'delta_rand': -5.6,  'fix_harm': '15:20'},
    'en-fa': {'n': 36,  'delta_ltd': -5.6,  'delta_rand': -16.8, 'fix_harm': '5:3'},
    'fa-en': {'n': 41,  'delta_ltd': -9.8,  'delta_rand': -10.1, 'fix_harm': '11:7'},
    'fr-en': {'n': 49,  'delta_ltd': -8.2,  'delta_rand': -4.1,  'fix_harm': '7:3'},
    'ko-en': {'n': 17,  'delta_ltd': -35.3, 'delta_rand': -19.6, 'fix_harm': '6:0'},
    'fi-en': {'n': 13,  'delta_ltd': -23.1, 'delta_rand': +9.3,  'fix_harm': '6:3'},
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
    """配对 bootstrap;论文符号:delta = LTE(ablation) − LTE(baseline),负数 = 下降。"""
    pairs = []
    for k, b in base_labels.items():
        a = abl_labels.get(k)
        if b is None or a is None: continue
        pairs.append((int(b is True), int(a is True)))
    if not pairs: return (None, None, None)
    n = len(pairs)
    delta_obs = (sum(p[1] for p in pairs) - sum(p[0] for p in pairs)) / n * 100
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = pairs
        d = (sum(s[i][1] for i in idx) - sum(s[i][0] for i in idx)) / n * 100
        boots.append(d)
    boots.sort()
    ci_low = boots[int(0.025 * n_boot)]
    ci_high = boots[int(0.975 * n_boot) - 1]
    return delta_obs, ci_low, ci_high


def aggregate_one(lp):
    L = PEAK_LAYER[lp]
    base_labels, n_base = load_score(lp, 'ablation_v4gpt52_naivepeak_baseline')
    ltd_labels, n_ltd = load_score(lp, 'ablation_v4gpt52_naivepeak_ablation')
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
    print(f'{"方向":<8} {"L":>3} {"n":>4} {"LTE_base":>10} {"ΔLTE(LTD naive-pk)":>20} '
          f'{"ΔLTE(rand,3s)":>18} {"Fix:Harm(LTD)":>14}')
    print('-' * 118)
    for lp, disp in LANGS:
        m = aggregate_one(lp)
        if m is None:
            print(f'{disp:<8} {"—":>3} {"—":>4} {"skipped(未跑或未判审)":>10}')
            rows.append({'lp': lp, 'disp': disp, 'skipped': True})
            continue
        rows.append({**m, 'disp': disp, 'skipped': False})
        r_base_str = f'{m["lte_rate_baseline"]*100:.1f}%' if m['lte_rate_baseline'] is not None else 'n/a'
        d_rand_str = fmt_pp(m['delta_rand_pp'])
        if m.get('delta_rand_pp_std') is not None:
            d_rand_str = f'{d_rand_str} (±{m["delta_rand_pp_std"]:.1f})'
        print(f'{disp:<8} {m["peak_L"]:>3} {m["n"]:>4} {r_base_str:>10} {fmt_pp(m["delta_ltd_pp"]):>20} '
              f'{d_rand_str:>18} {m["fix_harm_ltd"]:>14}')
    print('=' * 118)

    print('\n[对比] 论文 Table 5 (pooled 4 prompt × all-layer ablation;负数 = 下降)')
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
    csv_p = out_dir / 'naivepeak_table5.csv'
    with open(csv_p, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'naive_peak_layer', 'n',
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
    tex_p = out_dir / 'naivepeak_table5.tex'
    with open(tex_p, 'w', encoding='utf-8') as f:
        f.write('% Auto-generated: analysis/aggregate_naivepeak_ablation.py\n')
        f.write('% Naive-peak (Table 3 Peak L) paired LTD ablation (Qwen3.5-9B, mixed prompts).\n')
        f.write('% Response to reviewer W4 "ablate at the peak probing layer".\n')
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
            f.write(f'{r["disp"]} & {r["peak_L"]} & {r["n"]} & {_pp(r["delta_ltd_pp"])} & '
                    f'{rand_str} & {r["fix_harm_ltd"]} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write('\\caption{Naive-peak paired LTD ablation on Group~A (Qwen3.5-9B, mixed prompts). '
                'Peak L is the peak probing layer reported in Table~3. '
                'Negative $\\Delta$~LTE indicates the ablation reduces LTE (paired over Group~A). '
                'Random-direction column is the mean over three seeds \\{42,43,44\\} ablated at the same layer.}\n')
        f.write('\\label{tab:naivepeak_ablation}\n\\end{table}\n')
    print(f'LaTeX: {tex_p}')

    # ---- 带 CI 的全表 ----
    print('\n' + '=' * 100)
    print('Paired ΔLTE with 95% bootstrap CI (LTD naive-peak). Negative = ablation reduces LTE.')
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
    def _verdict(r):
        d_ltd = r['delta_ltd_pp']; lo, hi = r['ci_ltd_pp']
        d_rand = r['delta_rand_pp']
        if None in (d_ltd, d_rand, lo, hi): return '(N/A)'
        if lo <= d_rand <= hi:
            return 'LTD ≈ rand (CI covers rand mean; cannot reject null)'
        elif abs(d_ltd) > abs(d_rand):
            return 'LTD stronger (CI does NOT cover rand)'
        else:
            return 'rand stronger (CI does NOT cover rand)'
    for r in rows:
        if r.get('skipped'): continue
        tag = 'well-powered' if r['n'] >= 30 else 'under-powered'
        print(f'  {r["disp"]:<8} L={r["peak_L"]:>2} n={r["n"]:>3} ({tag})  →  {_verdict(r)}')

    print('\nOutput dir:', out_dir)


if __name__ == '__main__':
    main()
