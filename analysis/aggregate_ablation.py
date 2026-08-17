"""
aggregate_ablation.py — 聚合 LTD ablation 结果，输出 paper Table 5 的数据

▌输入
  results/{lp}/{model}/evaluation/
    ablation_Lall_dir_ablate_allprompts_baseline_score.json
    ablation_Lall_dir_ablate_allprompts_ablation_score.json

▌派生 LTE (与 probing/sankey 一致)
  LTE = literal_check.startswith("Yes") AND meaning_check.startswith("No")

▌按 id 配对，输出每个语言对一行:
  n       — Group A 样本总数
  Base    — baseline 的 LTE 计数
  Abl     — ablation 的 LTE 计数
  ΔLTE pp — (Abl - Base) / n × 100
  raw_fix — baseline LTE=1, ablation LTE=0, 译文文本变了 (未人工 audit)
  raw_harm — baseline LTE=0, ablation LTE=1, 译文文本变了
  p       — binomial test (min(fix,harm), fix+harm, 0.5)

用法:
  python analysis/aggregate_ablation.py --model Qwen3.5-9B
  python analysis/aggregate_ablation.py --model Qwen3-8B
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ALL_TARGETS

BASE = Path(__file__).resolve().parent.parent
ABL_NAME = "ablation_v4gpt52"

LABELS = {
    "en-fa": "En→Fa", "fa-en": "Fa→En", "fr-en": "Fr→En",
    "fi-en": "Fi→En", "ko-en": "Ko→En", "ja-en": "Ja→En",
}


def parse_yes_no(s):
    s = str(s).strip().lower()
    if s.startswith("yes"): return True
    if s.startswith("no"):  return False
    return None


def derive_lte(item):
    lit = parse_yes_no(item.get("literal_check", ""))
    mean = parse_yes_no(item.get("meaning_check", ""))
    if lit is None or mean is None:
        return None
    return 1 if (lit and not mean) else 0


def load_score_list(path):
    """按文件顺序返回 list（不去重）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for r in data["results"]:
        out.append({
            "id": r.get("id"),
            "source_prompt_type": r.get("source_prompt_type", ""),
            "sentence": r.get("sentence", ""),
            "lte": derive_lte(r),
            "model_translation": r.get("model_translation", ""),
            "literal_check": r.get("literal_check", ""),
            "meaning_check": r.get("meaning_check", ""),
        })
    return out


def pair_samples(base, abl):
    """
    优先按 (id, source_prompt_type) 配对（强保证：同样本同 prompt）。
    若任一文件没有 source_prompt_type 字段，fallback 到位置 + (id, sentence) 校验。
    返回: (pairs_list, mode_str, mismatch_count)
    """
    has_pt_base = any(b["source_prompt_type"] for b in base)
    has_pt_abl = any(a["source_prompt_type"] for a in abl)
    if has_pt_base and has_pt_abl:
        # 严格 key 配对
        abl_idx = {}
        for a in abl:
            k = (str(a["id"]), a["source_prompt_type"])
            abl_idx.setdefault(k, []).append(a)
        pairs = []
        for b in base:
            k = (str(b["id"]), b["source_prompt_type"])
            if k in abl_idx and abl_idx[k]:
                pairs.append((b, abl_idx[k].pop(0)))
        return pairs, "by_(id,prompt)", len(base) - len(pairs)
    # fallback: 位置配对 + (id, sentence) 校验
    pairs = []
    mismatch = 0
    for b, a in zip(base, abl):
        if str(b["id"]) != str(a["id"]) or b["sentence"][:80] != a["sentence"][:80]:
            mismatch += 1
            continue
        pairs.append((b, a))
    return pairs, "by_position (fallback, no source_prompt_type)", mismatch


def binomial_p(fix, harm):
    try:
        from scipy.stats import binomtest
        n = fix + harm
        if n == 0:
            return None
        return binomtest(min(fix, harm), n, 0.5).pvalue
    except Exception:
        return None


def aggregate_one(lp, model, abl_name=ABL_NAME, baseline_name=None):
    eval_dir = BASE / "results" / lp / model / "evaluation"
    base_path = eval_dir / f"{baseline_name or abl_name}_baseline_score.json"
    abl_path = eval_dir / f"{abl_name}_ablation_score.json"

    if not base_path.is_file() or not abl_path.is_file():
        return None

    base = load_score_list(base_path)
    abl = load_score_list(abl_path)

    pairs, pair_mode, mismatch = pair_samples(base, abl)

    base_lte_count = 0
    abl_lte_count = 0
    raw_fix = 0
    raw_harm = 0
    skipped_lte_none = 0

    for b, a in pairs:
        if b["lte"] is None or a["lte"] is None:
            skipped_lte_none += 1
            continue
        base_lte_count += b["lte"]
        abl_lte_count += a["lte"]
        text_changed = b["model_translation"] != a["model_translation"]
        if text_changed:
            if b["lte"] == 1 and a["lte"] == 0:
                raw_fix += 1
            elif b["lte"] == 0 and a["lte"] == 1:
                raw_harm += 1

    n = len(pairs) - skipped_lte_none
    delta_pp = (abl_lte_count - base_lte_count) / n * 100 if n else 0.0
    p = binomial_p(raw_fix, raw_harm)

    return {
        "lang_pair": lp,
        "n": n,
        "base_lte": base_lte_count,
        "abl_lte": abl_lte_count,
        "delta_pp": delta_pp,
        "raw_fix": raw_fix,
        "raw_harm": raw_harm,
        "p_value": p,
        "skipped_lte_none": skipped_lte_none,
        "pair_mode": pair_mode,
        "mismatch": mismatch,
        "n_base_file": len(base),
        "n_abl_file": len(abl),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="如 Qwen3.5-9B")
    parser.add_argument("--ablation-name", default=ABL_NAME,
                        help=f"ablation 输出前缀（默认 {ABL_NAME}）。"
                             f"super-strict 传 {ABL_NAME}__gb_superstrict；"
                             f"cross-lang 传 {ABL_NAME}__donor_fr-en 等")
    parser.add_argument("--baseline-name", default=None,
                        help="baseline 前缀（默认与 --ablation-name 相同）。"
                             "各变体共用同一份 baseline 时可指向主 run 的前缀，省 API")
    parser.add_argument("--output", default=None,
                        help="可选: 把结果存为 JSON，默认只打印")
    args = parser.parse_args()

    print(f"=== LTD ablation aggregation — {args.model} | {args.ablation_name} ===\n")
    print(f"{'Lang':>6} {'n':>5} {'Base':>5} {'Abl':>5} {'ΔLTE':>8} "
          f"{'Fix':>4} {'Harm':>5} {'p':>8} {'note':<20}")
    print("-" * 75)

    rows = []
    for lp in ALL_TARGETS:
        r = aggregate_one(lp, args.model, args.ablation_name, args.baseline_name)
        if r is None:
            print(f"{LABELS.get(lp, lp):>6}  -- score files missing --")
            continue
        note = [r["pair_mode"]]
        if r["skipped_lte_none"]:
            note.append(f"skip={r['skipped_lte_none']}")
        if r["mismatch"]:
            note.append(f"mismatch={r['mismatch']}")
        if r["n_base_file"] != r["n_abl_file"]:
            note.append(f"len_b={r['n_base_file']},len_a={r['n_abl_file']}")
        note_str = ", ".join(note)
        p_str = f"{r['p_value']:.3f}" if r["p_value"] is not None else "n/a"
        print(f"{LABELS.get(lp, lp):>6} {r['n']:>5} {r['base_lte']:>5} "
              f"{r['abl_lte']:>5} {r['delta_pp']:>+7.1f}pp "
              f"{r['raw_fix']:>4} {r['raw_harm']:>5} {p_str:>8} {note_str:<20}")
        rows.append(r)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "results": rows}, f, ensure_ascii=False, indent=2)
        print(f"\n保存到: {out_path}")

    # 论文 LaTeX 行（方便直接 copy 进 tab:steering_ablation）
    print("\n=== LaTeX rows (raw, before manual audit) ===")
    for r in rows:
        lp_tex = LABELS[r["lang_pair"]].replace("→", "$\\to$")
        print(f"{lp_tex} & {r['n']} & {r['base_lte']} & {r['abl_lte']} & "
              f"${r['delta_pp']:+.1f}$ pp & {r['raw_fix']}:{r['raw_harm']} \\\\")


if __name__ == "__main__":
    main()
