"""retry_failed_lte_v4.py — 只重评 v4 score 文件里 literal_translation_error=null 的失败条目，原地补回。

score.json 的每条已含 idiom/sentence/reference_meaning/model_translation，直接复用 v4 的 evaluate_single。
评委用 --judge-model（要和原 run 一致）。重评后重写 score.json + summary.json。

用法:
  python evaluation/retry_failed_lte_v4.py --lang-pair fi-en --model Qwen3.5-9B \
      --judge-model gpt-5.2 --output-prefix translation_lte_v4_gpt52
"""
import os, sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config, JUDGE_MODEL, API_REQUEST_INTERVAL
from evaluation.eval_translation_lte_v4 import init_client, evaluate_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-pair", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge-model", default=None, help="须与原 run 一致")
    ap.add_argument("--output-prefix", default="translation_lte_v4")
    args = ap.parse_args()

    judge = args.judge_model or JUDGE_MODEL
    ev = Path(get_config(args.lang_pair, model=args.model)["eval_output"])
    path = ev / f"{args.output_prefix}_score.json"
    if not path.exists():
        print(f"[缺] {path}"); return
    data = json.load(open(path, encoding="utf-8"))
    rows = data["results"]
    failed = [r for r in rows if r.get("literal_translation_error") is None]
    print(f"{args.lang_pair} | {args.output_prefix} | 失败 {len(failed)}/{len(rows)} 条 | judge={judge}")
    if not failed:
        print("无失败条目，无需重评。"); return

    client = init_client()
    fixed = 0
    for i, r in enumerate(failed):
        res = evaluate_single(client, r, judge)   # r 已含 idiom/sentence/reference_meaning/model_translation
        for k in ("literal_translation_error", "confidence", "literal_check",
                  "meaning_check", "final_judgment", "raw_response"):
            r[k] = res.get(k)
        if r.get("literal_translation_error") is not None:
            fixed += 1
        print(f"  [{i+1}/{len(failed)}] id={r.get('id')} -> LTE={r.get('literal_translation_error')}")
        time.sleep(API_REQUEST_INTERVAL)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # recompute summary
    valid = [r for r in rows if r.get("literal_translation_error") is not None]
    groups = {}
    for r in valid:
        groups.setdefault(r["prompt_type"], []).append(r)
    summary = [{
        "prompt_type": pt, "count": len(items),
        "error_count": sum(1 for x in items if x["literal_translation_error"] is True),
        "error_rate": sum(1 for x in items if x["literal_translation_error"] is True) / len(items) if items else 0,
        "avg_confidence": None,
    } for pt, items in sorted(groups.items())]
    with open(ev / f"{args.output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    still = sum(1 for r in rows if r.get("literal_translation_error") is None)
    print(f"补回 {fixed}/{len(failed)} 条；仍失败 {still} 条（可再跑一次）。")


if __name__ == "__main__":
    main()
