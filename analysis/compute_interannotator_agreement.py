"""
compute_interannotator_agreement.py — 人工标注者之间的内部一致性（inter-annotator agreement）。

数据: results/human_eval/lte_human_eval_{lang_pair}_{annotator}.json
每条 (id, prompt_type) 有两个二元判断:
  literal_check_human  (0/1; -1=未评)
  meaning_check_human  (0/1; -1=未评)
派生 LTE = 1 iff literal==1 且 meaning==0 （与 Sankey / LTE judge 同口径）。

对每个语言对 × 每个维度(literal / meaning / LTE) 报告:
  - 两两 Cohen's κ + 一致率（各对取双方都评过的交集）
  - 三人 Fleiss' κ + 全体一致率（仅取三人都评过的条目，固定 3 个 rater）
注意: 若某标注者只评了 BasicPrompt 子集，三人 Fleiss 的 n 会小于其余两人两两的 n。

用法:
  python analysis/compute_interannotator_agreement.py            # fi-en + fr-en
  python analysis/compute_interannotator_agreement.py --lang-pairs fr-en
  python analysis/compute_interannotator_agreement.py --pooled   # 额外给两语言对合并
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ANNOTATORS = ["annotator1", "annotator2", "annotator3"]
DIMS = ["literal", "meaning", "lte"]


def derive(it, dim):
    """返回该维度的 0/1，未评返回 None。"""
    l = it.get("literal_check_human")
    m = it.get("meaning_check_human")
    if l in (None, -1) or m in (None, -1):
        # literal/meaning 任一缺失就视为该条未评（lte 需两者；单维度也按缺失处理保持口径一致）
        if dim == "literal":
            return l if l in (0, 1) else None
        if dim == "meaning":
            return m if m in (0, 1) else None
        return None  # lte 需要两者都在
    if dim == "literal":
        return int(l)
    if dim == "meaning":
        return int(m)
    return 1 if (l == 1 and m == 0) else 0  # lte


def load(lp, annotator, human_dir, prompt=None):
    p = Path(human_dir) / f"lte_human_eval_{lp}_{annotator}.json"
    if not p.exists():
        return None
    items = json.load(open(p, encoding="utf-8"))
    if prompt:
        items = [it for it in items if it.get("prompt_type") == prompt]
    return {(it["id"], it["prompt_type"]): it for it in items}


def cohen_kappa(a, b):
    """二元 Cohen's κ。a,b 同长 0/1 列表。"""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    k = 1.0 if pe >= 1 else (po - pe) / (1 - pe)
    return po, k


def fleiss_kappa(rows):
    """rows: 每行是一个 item 的 [n_cat0, n_cat1]，每行和 = 固定 rater 数 m。"""
    N = len(rows)
    if N == 0:
        return float("nan")
    m = sum(rows[0])
    if m < 2:
        return float("nan")
    # 每条的一致度 P_i
    P = [(sum(c * c for c in r) - m) / (m * (m - 1)) for r in rows]
    P_bar = sum(P) / N
    # 类别边际比例
    tot = m * N
    p_cat = [sum(r[j] for r in rows) / tot for j in range(len(rows[0]))]
    P_e = sum(p * p for p in p_cat)
    return 1.0 if P_e >= 1 else (P_bar - P_e) / (1 - P_e)


def analyze(lp, ann_data, label):
    present = [a for a in ANNOTATORS if ann_data.get(a)]
    print(f"\n{'='*68}\n[{label}]  标注者: {', '.join(present)}")
    # 覆盖度
    for a in present:
        for dim in DIMS:
            pass
    cov = {a: sum(1 for k in ann_data[a] if derive(ann_data[a][k], "lte") is not None)
           for a in present}
    print("  各标注者已评条目(可派生LTE): " +
          " | ".join(f"{a}={cov[a]}" for a in present))

    for dim in DIMS:
        print(f"\n  —— 维度: {dim} ——")
        # 两两 Cohen κ
        for x, y in combinations(present, 2):
            keys = sorted(set(ann_data[x]) & set(ann_data[y]))
            a, b = [], []
            for k in keys:
                va, vb = derive(ann_data[x][k], dim), derive(ann_data[y][k], dim)
                if va is not None and vb is not None:
                    a.append(va); b.append(vb)
            if a:
                po, k_ = cohen_kappa(a, b)
                print(f"    {x:>7} vs {y:<7}  n={len(a):>3}  一致={po*100:5.1f}%  κ={k_:+.3f}")
            else:
                print(f"    {x:>7} vs {y:<7}  无共同已评条目")
        # 三人 Fleiss κ（仅三人都评过的条目）
        if len(present) >= 3:
            keys = sorted(set.intersection(*[set(ann_data[a]) for a in present]))
            rows, agree3 = [], 0
            for k in keys:
                vals = [derive(ann_data[a][k], dim) for a in present]
                if any(v is None for v in vals):
                    continue
                rows.append([vals.count(0), vals.count(1)])
                if len(set(vals)) == 1:
                    agree3 += 1
            if rows:
                fk = fleiss_kappa(rows)
                print(f"    {'三人 Fleiss':>15}  n={len(rows):>3}  "
                      f"全体一致={agree3/len(rows)*100:5.1f}%  κ={fk:+.3f}")
            else:
                print(f"    {'三人 Fleiss':>15}  三人共同已评条目不足")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-pairs", nargs="+", default=["fi-en", "fr-en"])
    ap.add_argument("--human-dir", default="results/human_eval")
    ap.add_argument("--prompt", default=None, help="只统计某 prompt_type，如 BasicPrompt")
    ap.add_argument("--pooled", action="store_true", help="额外输出两语言对合并的结果")
    args = ap.parse_args()

    print("人工标注者内部一致性 (inter-annotator agreement)")
    print("LTE = 1 iff literal_check_human==1 且 meaning_check_human==0")
    if args.prompt:
        print(f"过滤: prompt_type = {args.prompt}")

    all_data = {}
    for lp in args.lang_pairs:
        ann_data = {a: load(lp, a, args.human_dir, args.prompt) for a in ANNOTATORS}
        ann_data = {a: d for a, d in ann_data.items() if d}
        if not ann_data:
            print(f"\n[{lp}] 无数据，跳过")
            continue
        all_data[lp] = ann_data
        analyze(lp, ann_data, lp)

    if args.pooled and len(all_data) > 1:
        merged = {a: {} for a in ANNOTATORS}
        for lp, ann_data in all_data.items():
            for a, d in ann_data.items():
                for k, it in d.items():
                    merged[a][(lp,) + k] = it
        merged = {a: d for a, d in merged.items() if d}
        analyze("POOLED", merged, "POOLED (fi-en + fr-en)")


if __name__ == "__main__":
    main()
