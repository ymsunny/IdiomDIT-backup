"""aggregate_basic_only_ablation.py — 聚合 Basic-only 版 LTD 消融结果,产 Table 5 附录版。

读取路径:
  results/{lp}/Qwen3.5-9B/evaluation/
    ablation_v4gpt52_basic_baseline_score.json               ← baseline(所有干预共享)
    ablation_v4gpt52_basic_ablation_score.json               ← LTD 消融
    ablation_Lall_dir_ablate_BasicPrompt__rand_s{42,43,44}_ablation_score.json  ← 3 seed 随机

指标:
  LTE_rate  = sum(literal_translation_error==True) / n
  ΔLTE      = LTE_rate(baseline) - LTE_rate(ablation)     [正数=下降,pp]
  Fix       = 配对里 baseline LTE=1 且 ablation LTE=0 的条数
  Harm      = 配对里 baseline LTE=0 且 ablation LTE=1 的条数
  ΔLTE(rand) = 3 seed 的 ΔLTE 均值(SD 一并报)

输出:
  analysis/output/basic_only_table5.csv
  analysis/output/basic_only_table5.md
  analysis/output/basic_only_table5.tex
  控制台:全表 + 与论文 pooled 结果对比 + 简短解读
"""
import os, sys, io, json, glob, statistics, random
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BOOT_N = 2000
BOOT_SEED = 20260710

MODEL = 'Qwen3.5-9B'
LANGS = [('ja-en', 'Ja→En'), ('en-fa', 'En→Fa'), ('fa-en', 'Fa→En'),
         ('fr-en', 'Fr→En'), ('ko-en', 'Ko→En'), ('fi-en', 'Fi→En')]
SEEDS = [42, 43, 44]
BASE = Path(__file__).resolve().parent.parent

# 论文 pooled 版 Table 5 数字(4123 PDF submission)
PAPER_POOLED = {
    'ja-en': {'n': 204, 'delta_ltd': +2.5, 'delta_rand': -5.6, 'fix_harm': '15:20'},
    'en-fa': {'n': 36,  'delta_ltd': -5.6, 'delta_rand': -16.8, 'fix_harm': '5:3'},
    'fa-en': {'n': 41,  'delta_ltd': -9.8, 'delta_rand': -10.1, 'fix_harm': '11:7'},
    'fr-en': {'n': 49,  'delta_ltd': -8.2, 'delta_rand': -4.1,  'fix_harm': '7:3'},
    'ko-en': {'n': 17,  'delta_ltd': -35.3,'delta_rand': -19.6, 'fix_harm': '6:0'},
    'fi-en': {'n': 13,  'delta_ltd': -23.1,'delta_rand': +9.3,  'fix_harm': '6:3'},
}


def load_score(lp, prefix):
    """读一个 score.json,返回 {id -> LTE(bool or None)} 和总条数"""
    p = BASE / 'results' / lp / MODEL / 'evaluation' / f'{prefix}_score.json'
    if not p.exists():
        return None, 0
    d = json.load(open(p, encoding='utf-8'))
    labels = {}
    for r in d['results']:
        iid = str(r.get('id'))
        # (id, prompt_type) 唯一键,防同 idiom 不同 prompt 冲突(Basic-only 下 prompt 只有 Basic)
        key = (iid, r.get('prompt_type', ''))
        labels[key] = r.get('literal_translation_error')
    return labels, len(d['results'])


def lte_rate(labels):
    valid = [v for v in labels.values() if v is not None]
    if not valid:
        return None
    return sum(1 for v in valid if v is True) / len(valid)


def paired_fix_harm(base_labels, abl_labels):
    fix = harm = matched = 0
    for k, b in base_labels.items():
        a = abl_labels.get(k)
        if b is None or a is None:
            continue
        matched += 1
        if b is True and a is False:
            fix += 1
        elif b is False and a is True:
            harm += 1
    return fix, harm, matched


def paired_delta_pp_bootstrap(base_labels, abl_labels, n_boot=BOOT_N, seed=BOOT_SEED):
    """配对 bootstrap: 返回 (delta_mean_pp, ci_low_pp, ci_high_pp)。
    delta = LTE_rate(baseline) − LTE_rate(ablation),正数=消融降低了 LTE。"""
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
    """对一个方向,返回 dict of metrics 或 None(如果没数据)"""
    base_labels, n_base = load_score(lp, 'ablation_v4gpt52_basic_baseline')
    ltd_labels, n_ltd = load_score(lp, 'ablation_v4gpt52_basic_ablation')
    if base_labels is None or ltd_labels is None:
        return None

    r_base = lte_rate(base_labels)
    r_ltd = lte_rate(ltd_labels)

    fix_ltd, harm_ltd, matched_ltd = paired_fix_harm(base_labels, ltd_labels)
    delta_ltd, ci_ltd_lo, ci_ltd_hi = paired_delta_pp_bootstrap(base_labels, ltd_labels)

    # 3 seed 随机(每个 seed 单独算 delta + CI,再算 3 seed 的平均及 spread)
    rand_deltas = []
    rand_cis = []
    rand_fix_harm = []
    for s in SEEDS:
        rand_labels, _ = load_score(lp, f'ablation_Lall_dir_ablate_BasicPrompt__rand_s{s}_ablation')
        if rand_labels is None:
            continue
        d, lo, hi = paired_delta_pp_bootstrap(base_labels, rand_labels, seed=BOOT_SEED + s)
        if d is not None:
            rand_deltas.append(d)
            rand_cis.append((lo, hi))
        f, h, m = paired_fix_harm(base_labels, rand_labels)
        rand_fix_harm.append((f, h, m))

    delta_rand_mean = statistics.mean(rand_deltas) if rand_deltas else None
    delta_rand_std = statistics.stdev(rand_deltas) if len(rand_deltas) > 1 else None

    return {
        'lp': lp,
        'n': n_base,
        'lte_rate_baseline': r_base,
        'lte_rate_ltd': r_ltd,
        'delta_ltd_pp': delta_ltd,
        'ci_ltd_pp': (ci_ltd_lo, ci_ltd_hi),
        'delta_rand_pp': delta_rand_mean,
        'delta_rand_pp_std': delta_rand_std,
        'delta_rand_pp_perseed': rand_deltas,
        'ci_rand_perseed_pp': rand_cis,
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
    print(f'{"方向":<8} {"n(A_Basic)":>10} {"LTE_rate baseline":>18} {"ΔLTE(LTD)":>12} '
          f'{"ΔLTE(rand mean)":>17} {"Fix:Harm(LTD)":>14}')
    print('-' * 118)
    for lp, disp in LANGS:
        m = aggregate_one(lp)
        if m is None:
            print(f'{disp:<8} {"—":>10} {"— (未跑或 skipped)":>65}')
            rows.append({'lp': lp, 'disp': disp, 'skipped': True})
            continue
        rows.append({**m, 'disp': disp, 'skipped': False})
        r_base_str = f'{m["lte_rate_baseline"]*100:.1f}%' if m['lte_rate_baseline'] is not None else 'n/a'
        d_rand_str = fmt_pp(m['delta_rand_pp'])
        if m.get('delta_rand_pp_std') is not None:
            d_rand_str = f'{d_rand_str} (±{m["delta_rand_pp_std"]:.1f})'
        print(f'{disp:<8} {m["n"]:>10} {r_base_str:>18} {fmt_pp(m["delta_ltd_pp"]):>12} '
              f'{d_rand_str:>17} {m["fix_harm_ltd"]:>14}')
    print('=' * 118)

    print('\n[对比] 论文 Table 5 (pooled 4 prompt)')
    print(f'{"方向":<8} {"n":>4} {"ΔLTE(LTD)":>12} {"ΔLTE(rand)":>12} {"Fix:Harm":>10}')
    print('-' * 60)
    for lp, disp in LANGS:
        pp = PAPER_POOLED.get(lp, {})
        n = pp.get('n', '—')
        d_ltd = pp.get('delta_ltd')
        d_rand = pp.get('delta_rand')
        fh = pp.get('fix_harm', '—')
        print(f'{disp:<8} {n!s:>4} '
              f'{fmt_pp(d_ltd):>12} {fmt_pp(d_rand):>12} {fh:>10}')
    print('=' * 60)

    # ---- CSV ----
    import csv
    csv_p = out_dir / 'basic_only_table5.csv'
    with open(csv_p, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'n', 'LTE_rate_baseline', 'LTE_rate_LTD',
                    'delta_LTE_LTD_pp', 'delta_LTE_LTD_ci_low', 'delta_LTE_LTD_ci_high',
                    'delta_LTE_rand_pp_mean', 'delta_LTE_rand_pp_std',
                    'delta_LTE_rand_pp_perseed_42_43_44',
                    'fix_harm_LTD', 'fix_harm_rand_perseed_42_43_44',
                    'paper_pooled_n', 'paper_delta_LTD_pp', 'paper_delta_rand_pp'])
        for r in rows:
            lp = r['lp']
            pp = PAPER_POOLED.get(lp, {})
            if r.get('skipped'):
                w.writerow([lp, 'skipped'])
            else:
                perseed = ';'.join(f'{d:+.2f}' for d in r.get('delta_rand_pp_perseed', []))
                fh_rand = ';'.join(r.get('fix_harm_rand_perseed', []))
                ci_lo, ci_hi = r.get('ci_ltd_pp', (None, None))
                w.writerow([lp, r['n'],
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
    print(f'\nCSV 写入: {csv_p}')

    # ---- LaTeX table (matches论文格式) ----
    tex_p = out_dir / 'basic_only_table5.tex'
    with open(tex_p, 'w', encoding='utf-8') as f:
        f.write('% Auto-generated: analysis/aggregate_basic_only_ablation.py\n')
        f.write('% Basic-only paired LTD ablation (Qwen3.5-9B), Appendix table for reviewer response.\n')
        f.write('\\begin{table}[!t]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n')
        f.write('\\begin{tabular}{lrrrr}\n\\toprule\n')
        f.write(' & \\textbf{$n$} & \\textbf{$\\Delta$ LTE} & \\textbf{$\\Delta$ LTE (rand.)} & \\textbf{Fix:Harm} \\\\\n')
        f.write('\\midrule\n')
        for r in rows:
            if r.get('skipped'):
                f.write(f'{r["disp"]} & --- & \\multicolumn{{3}}{{c}}{{Basic-only $|A|<5$; skipped}} \\\\\n')
                continue
            def _pp(v):
                if v is None: return '---'
                return f'${v:+.1f}$ pp'
            f.write(f'{r["disp"]} & {r["n"]} & {_pp(r["delta_ltd_pp"])} & '
                    f'{_pp(r["delta_rand_pp"])} & {r["fix_harm_ltd"]} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write('\\caption{Paired LTD ablation on Basic-only Group~A (Qwen3.5-9B). $\\Delta$~LTE and $\\Delta$~LTE~(rand.) are mean over three seeds. Compare with Table~5 (pooled).}\n')
        f.write('\\label{tab:steering_ablation_basic}\n\\end{table}\n')
    print(f'LaTeX 表: {tex_p}')

    # ---- 附解读 ----
    # ---- 带 CI 的全表 ----
    print('\n' + '=' * 90)
    print('带 95% bootstrap CI 的对比(正数 pp = 消融 后 LTE 下降;论文用相反符号)')
    print('=' * 90)
    for r in rows:
        if r.get('skipped'): continue
        d_ltd = r['delta_ltd_pp']; lo, hi = r['ci_ltd_pp']
        d_rand = r['delta_rand_pp']
        print(f'  {r["disp"]:<8} n={r["n"]:>3}  '
              f'ΔLTE(LTD) = {d_ltd:+.1f} pp  [95% CI {lo:+.1f}, {hi:+.1f}]  '
              f'| ΔLTE(rand) = {d_rand:+.1f} pp (3-seed mean)')

    # ---- 更严谨的解读 ----
    print('\n' + '=' * 90)
    print('解读(严谨版:用 LTD 的 95% CI 判断是否覆盖 rand mean)')
    print('=' * 90)
    powered = [r for r in rows if not r.get('skipped') and r['n'] >= 30]
    marginal = [r for r in rows if not r.get('skipped') and 15 <= r['n'] < 30]
    underpowered = [r for r in rows if not r.get('skipped') and r['n'] < 15]

    def _verdict(r):
        d_ltd = r['delta_ltd_pp']; lo, hi = r['ci_ltd_pp']
        d_rand = r['delta_rand_pp']
        if None in (d_ltd, d_rand, lo, hi): return '(N/A)'
        # LTD CI 是否包含 rand mean → 无法拒绝"LTD ≈ rand"
        if lo <= d_rand <= hi:
            return 'LTD ≈ rand(CI 覆盖 rand,无法拒绝打平)'
        elif d_ltd > d_rand:  # LTD reduce 更多(在我的正号约定下 LTD 更大)
            return 'LTD > rand(CI 不覆盖 rand,LTD 更强)'
        else:
            return 'rand > LTD(CI 不覆盖 rand,rand 更强)'

    if powered:
        print('  Well-powered (n ≥ 30):')
        for r in powered:
            print(f'    {r["disp"]:<8} n={r["n"]:>3}  →  {_verdict(r)}')
    if marginal:
        print('  Marginal (15 ≤ n < 30,CI 会很宽,谨慎):')
        for r in marginal:
            print(f'    {r["disp"]:<8} n={r["n"]:>3}  →  {_verdict(r)}')
    if underpowered:
        print('  Underpowered (n < 15,不列入结论):')
        for r in underpowered:
            print(f'    {r["disp"]:<8} n={r["n"]:>3}  →  {_verdict(r)} (仅参考)')

    print('\n' + '=' * 90)
    print('关键对比:Basic-only vs 论文 pooled(Ja→En 是唯一 well-powered 方向)')
    print('=' * 90)
    ja = next((r for r in rows if r['lp'] == 'ja-en' and not r.get('skipped')), None)
    if ja:
        pool = PAPER_POOLED['ja-en']
        print(f'  Basic-only Ja→En (n={ja["n"]}): LTD {ja["delta_ltd_pp"]:+.1f} pp, '
              f'rand {ja["delta_rand_pp"]:+.1f} pp → {_verdict(ja)}')
        print(f'  Pooled     Ja→En (n={pool["n"]}): LTD {-pool["delta_ltd"]:+.1f} pp, '
              f'rand {-pool["delta_rand"]:+.1f} pp (相同符号约定)')
        print('  → Basic-only 与 pooled 是否给出同一结论?见 CI 覆盖情况。')

    print('\n输出目录:', out_dir)


if __name__ == '__main__':
    main()
