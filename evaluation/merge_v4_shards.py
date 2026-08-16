"""merge_v4_shards.py — 把按 prompt 分片的 v4 score 合并回单方向文件。

分片: {prefix}__{PromptType}_score.json  →  合并: {prefix}_score.json (+ _summary.json)
用法: python evaluation/merge_v4_shards.py --lang-pair ja-en --model Qwen3.5-9B --prefix translation_lte_v4_gpt52
"""
import os, sys, json, glob, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-pair", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix", default="translation_lte_v4_gpt52")
    args = ap.parse_args()

    cfg = get_config(args.lang_pair, model=args.model)
    ev = Path(cfg["eval_output"])
    shards = sorted(glob.glob(str(ev / f"{args.prefix}__*_score.json")))
    if not shards:
        print(f"[缺] 没有分片 {args.prefix}__*_score.json in {ev}"); return

    merged, direction = [], None
    for s in shards:
        d = json.load(open(s, encoding="utf-8"))
        direction = d.get("direction", direction)
        merged.extend(d["results"])
        print(f"  + {os.path.basename(s)}: {len(d['results'])} 条")

    out = ev / f"{args.prefix}_score.json"
    json.dump({"direction": direction, "total": len(merged), "results": merged},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    valid = [r for r in merged if r.get("literal_translation_error") is not None]
    groups = {}
    for r in valid:
        groups.setdefault(r["prompt_type"], []).append(r)
    summary = [{
        "prompt_type": pt, "count": len(items),
        "error_count": sum(1 for x in items if x["literal_translation_error"] is True),
        "error_rate": (sum(1 for x in items if x["literal_translation_error"] is True) / len(items)) if items else 0,
        "avg_confidence": None,
    } for pt, items in sorted(groups.items())]
    json.dump(summary, open(ev / f"{args.prefix}_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    failed = sum(1 for r in merged if r.get("literal_translation_error") is None)
    print(f"合并 {len(shards)} 片 → {out.name}: 共 {len(merged)} 条 (失败 {failed})")


if __name__ == "__main__":
    main()
