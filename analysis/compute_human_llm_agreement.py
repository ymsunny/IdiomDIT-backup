"""
compute_human_llm_agreement.py — 人工标注 vs LLM judge 一致性分析（多标注者 + 多数投票）。

对某语言对（默认 BasicPrompt）：
  1. 各标注者(sun/chaoyu/andrew)的有效评分数 + 人工 LTE 率
  2. 标注者两两 Cohen's κ（标注本身可靠性上限）
  3. 多数投票（每条取已评标注者的多数；1-1 平票跳过）作为 human ground truth
  4. 多数投票 human vs LLM judge（gpt-4o-mini / gpt-5.2）的一致率 + Cohen's κ + LTE 率

派生 LTE（人和 LLM 同一双条件）：
  - human: literal_check_human==1 且 meaning_check_human==0；未评(-1)跳过
  - LLM:   literal_check 以 Yes 开头 且 meaning_check 以 No 开头

现状（只有 sun 评了）也能跑：多数投票退化为 sun 自己，结果即 human(sun) vs LLM。

用法:
  python analysis/compute_human_llm_agreement.py --lang-pair fi-en --model Qwen3.5-9B
  python analysis/compute_human_llm_agreement.py --lang-pair fr-en \
      --annotators sun chaoyu andrew --human-dir results/human_eval
"""
import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config


def yes_no(s):
    s = str(s).strip().lower()
    return True if s.startswith("yes") else False if s.startswith("no") else None


def human_lte(it):
    l = it.get("literal_check_human")
    m = it.get("meaning_check_human")
    if l in (None, -1) or m in (None, -1):
        return None
    return 1 if (l == 1 and m == 0) else 0


def llm_lte(it):
    lit = yes_no(it.get("literal_check", ""))
    mc = yes_no(it.get("meaning_check", ""))
    if lit is None or mc is None:
        return None
    return 1 if (lit and not mc) else 0


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_human(path, prompt):
    out = {}
    for it in json.load(open(path, encoding="utf-8")):
        if prompt != "all" and str(it.get("prompt_type", "")) != prompt:
            continue
        v = human_lte(it)
        if v is not None:
            out[(it["id"], it.get("prompt_type"))] = v
    return out


def load_llm(path, prompt):
    out = {}
    for it in json.load(open(path, encoding="utf-8")).get("results", []):
        if prompt != "all" and str(it.get("prompt_type", "")) != prompt:
            continue
        v = llm_lte(it)
        if v is not None:
            out[(it["id"], it.get("prompt_type"))] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-pair", required=True)
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--annotators", nargs="+", default=["sun", "chaoyu", "andrew"])
    ap.add_argument("--human-dir", default="results/human_eval")
    ap.add_argument("--prompt", default="BasicPrompt",
                    help="某 prompt_type（如 BasicPrompt），或 all=跨 4 个 prompt 合并")
    ap.add_argument("--judge-version", default="v4", choices=["v4", "legacy"],
                    help="v4=translation_lte_v4_*（默认，最新）；legacy=旧 translation_lte_*")
    args = ap.parse_args()

    hdir = Path(args.human_dir)
    ev = Path(get_config(args.lang_pair, model=args.model)["eval_output"])

    # 1) 各标注者
    ann = {}
    for a in args.annotators:
        p = hdir / f"lte_human_eval_{args.lang_pair}_{a}.json"
        if p.exists():
            d = load_human(p, args.prompt)
            if d:
                ann[a] = d
    print(f"=== {args.lang_pair} / {args.model} / {args.prompt} ===")
    if not ann:
        print("没有任何标注者的有效评分"); return
    print("\n[各标注者]")
    for a, d in ann.items():
        print(f"  {a:<8} 有效 {len(d):>3} 条 | 人工 LTE {sum(d.values())/len(d)*100:5.1f}%")

    # 2) 标注者两两 Cohen κ
    names = list(ann.keys())
    if len(names) >= 2:
        print("\n[标注者两两一致性 Cohen κ]")
        for x, y in combinations(names, 2):
            ids = sorted(set(ann[x]) & set(ann[y]))
            a = [ann[x][i] for i in ids]
            b = [ann[y][i] for i in ids]
            ag = sum(1 for u, v in zip(a, b) if u == v) / len(ids) if ids else float("nan")
            print(f"  {x} vs {y}: n={len(ids)} | 一致 {ag*100:5.1f}% | κ={kappa(a,b):.3f}")
    else:
        print("\n[标注者两两一致性] 仅 1 名标注者，跳过（多数投票退化为该标注者）")

    # 3) 多数投票
    all_ids = set().union(*[set(d) for d in ann.values()])
    maj, ties = {}, 0
    for i in sorted(all_ids):
        votes = [ann[a][i] for a in names if i in ann[a]]
        ones = sum(votes); zeros = len(votes) - ones
        if ones > zeros:
            maj[i] = 1
        elif zeros > ones:
            maj[i] = 0
        else:
            ties += 1
    print(f"\n[多数投票 human] 有效 {len(maj)} 条（平票跳过 {ties}）| LTE {sum(maj.values())/len(maj)*100:.1f}%")

    # 4) 多数投票 vs LLM
    if args.judge_version == "v4":
        f_4o, f_52 = "translation_lte_v4_score.json", "translation_lte_v4_gpt52_score.json"
    else:
        f_4o, f_52 = "translation_lte_score.json", "translation_lte_gpt52_score.json"
    print(f"\n[多数投票 human vs LLM judge]  (judge={args.judge_version})")
    for name, p in [("gpt-4o-mini", ev / f_4o),
                    ("gpt-5.2", ev / f_52)]:
        if not p.exists():
            print(f"  vs {name}: 文件缺失 {p}"); continue
        L = load_llm(p, args.prompt)
        ids = sorted(set(maj) & set(L))
        a = [maj[i] for i in ids]
        b = [L[i] for i in ids]
        n = len(ids)
        if n == 0:
            print(f"  vs {name}: 无对齐条目"); continue
        ag = sum(1 for u, v in zip(a, b) if u == v) / n
        print(f"  vs {name:<11}: n={n} | human {sum(a)/n*100:5.1f}% / LLM {sum(b)/n*100:5.1f}% "
              f"| 一致 {ag*100:5.1f}% | κ={kappa(a,b):.3f}")


if __name__ == "__main__":
    main()
