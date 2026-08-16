"""
audit_ablation_fix.py — 列出 ablation 的 raw_fix 候选，供人工 audit

raw_fix 定义: baseline LTE=1 AND ablation LTE=0 AND 译文文本变了
派生 LTE: literal_check.startswith("Yes") AND meaning_check.startswith("No")

用法:
  python analysis/audit_ablation_fix.py --model Qwen3.5-9B --lang-pair en-fa
  python analysis/audit_ablation_fix.py --model Qwen3.5-9B --lang-pair en-fa --kind harm
  python analysis/audit_ablation_fix.py --model Qwen3.5-9B --lang-pair en-fa --output audit_en-fa.txt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent
NAME = "ablation_v4gpt52"


def parse_yes_no(s):
    return str(s).strip().lower().startswith("yes")


def derive_lte(r):
    lit = parse_yes_no(r.get("literal_check", ""))
    mean = parse_yes_no(r.get("meaning_check", ""))
    return 1 if (lit and not mean) else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lang-pair", required=True)
    p.add_argument("--kind", default="fix", choices=["fix", "harm"],
                   help="fix: base LTE=1, abl LTE=0; harm: base LTE=0, abl LTE=1")
    p.add_argument("--output", default=None, help="保存到文件，默认只打印")
    args = p.parse_args()

    eval_dir = BASE / "results" / args.lang_pair / args.model / "evaluation"
    base = json.load(open(eval_dir / f"{NAME}_baseline_score.json", encoding="utf-8"))["results"]
    abl = json.load(open(eval_dir / f"{NAME}_ablation_score.json", encoding="utf-8"))["results"]

    lines = []
    n = 0
    for x, y in zip(base, abl):
        bx, by = derive_lte(x), derive_lte(y)
        text_changed = x["model_translation"] != y["model_translation"]
        if args.kind == "fix" and not (bx == 1 and by == 0 and text_changed):
            continue
        if args.kind == "harm" and not (bx == 0 and by == 1 and text_changed):
            continue
        n += 1
        lines.append(f"\n#{n} id={x['id']} | prompt={x.get('source_prompt_type','?')} | {x['idiom']}")
        lines.append(f"  EN  : {x['sentence'][:200]}")
        lines.append(f"  REF : {x.get('reference_meaning','')[:200]}")
        lines.append(f"  GOLD: {x.get('gold_translation','')[:200]}")
        lines.append(f"  BASE: {x['model_translation'][:200]}")
        lines.append(f"  ABL : {y['model_translation'][:200]}")
        lines.append(f"  BASE literal: {x.get('literal_check','')[:120]}")
        lines.append(f"  BASE meaning: {x.get('meaning_check','')[:120]}")
        lines.append(f"  ABL  literal: {y.get('literal_check','')[:120]}")
        lines.append(f"  ABL  meaning: {y.get('meaning_check','')[:120]}")

    header = f"=== {args.lang_pair} | {args.model} | raw_{args.kind}: {n} ==="
    out = header + "\n" + "\n".join(lines)
    print(out)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
