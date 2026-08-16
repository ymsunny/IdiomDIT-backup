"""
extract_known_lte_groups.py — 提取机制分析用的最小对比组

▌两组数据
  Group A (known-LTE):       D=1, I=1, LTE=1
    → 模型理解习语含义，但生成时走了逐词映射通路
  Group B loose (known-non-LTE):   D=1, I=1, LTE=0
    → 模型理解习语含义，且没有产生 LTE（含翻错但非直译的情况）
  Group B strict (known-idiomatic): D=1, I=1, LTE=0, MeaningConveyed=True
    → 模型理解习语含义，且翻译明确传达了习语语义（主分析对照）
  Group B super-strict (Lit=0,Mean=1): D=1, I=1, Literal=0, Meaning=1
    → 在 strict 基础上再剔除巧合保意 (Lit=1,Mean=1)，与 Group A (Lit=1,Mean=0) 严格反对角（concern #3 对照）

▌LTE 判定 (与 Sankey / 堆叠图 一致)
  literal_check.startswith("Yes") AND meaning_check.startswith("No") → LTE=1
  不读 LLM 输出的 literal_translation_error 字段
  exclusion_key.json 中的 (id, prompt_type) 跳过

▌默认设置 (基于人工评估 + LTE judge 文献)
  --interp-version p2  (with-context, 与翻译任务同构)
  D=1 必须 (extract_groups 内强制过滤)

▌输出 (新结构)
  results/{lang_pair}/{model}/mechanistic/[exp_name/]
    ├── group_known_lte.json              ← Group A
    ├── group_known_non_lte.json          ← Group B loose
    ├── group_known_non_lte_strict.json   ← Group B strict
    ├── group_known_non_lte_superstrict.json ← Group B super-strict (Lit=0, Mean=1)
    ├── summary.json
    └── report.txt

▌用法
  python mechanistic/extract_known_lte_groups.py --lang-pair fi-en --model Qwen3.5-9B
  python mechanistic/extract_known_lte_groups.py --all --model Qwen3.5-9B
"""

import os, json, argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config, ALL_TARGETS
from exclusion_utils import load_exclusion_set

PROMPT_TYPES = ["BasicPrompt", "Meaning-FirstTranslationPrompt",
                "Anti-LiteralConstraintPrompt", "SuperLiteralBiasMitigationPrompt"]


def parse_yes_no(s):
    """'Yes - ...' / 'No - ...' → True/False/None"""
    s = str(s).strip().lower()
    if s.startswith("yes"): return True
    if s.startswith("no"):  return False
    return None


# ============================================================
# 数据加载
# ============================================================

def _load_interp_file(json_path):
    """读一个 interpretation_{p1,p2}_score.json, 返回 {id: entry_dict}"""
    out = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("results", []):
        iid = str(item.get("id", "")).strip()
        score = item.get("interpretation_score")
        if score is None:
            score = 1 if item.get("interpretation", item.get("knowledge", False)) else 0
        out[iid] = {
            "id": iid,
            "idiom": item.get("idiom", ""),
            "sentence": item.get("sentence", ""),
            "reference_meaning": item.get("reference_meaning", ""),
            "interpretation_score": int(score),
        }
    return out


def load_interpretation(cfg, version="p2"):
    """加载 Interpretation 数据。p2(default) / p1 / union"""
    if version == "p1":
        return _load_interp_file(str(cfg["interpretation_json"]))
    p2_path = str(cfg.get("interpretation_p2_json"))
    if version == "p2":
        return _load_interp_file(p2_path)
    if version == "union":
        p1 = _load_interp_file(str(cfg["interpretation_json"]))
        p2 = _load_interp_file(p2_path)
        merged = {}
        for iid in set(p1) | set(p2):
            s1 = p1[iid]["interpretation_score"] if iid in p1 else 0
            s2 = p2[iid]["interpretation_score"] if iid in p2 else 0
            base = p1.get(iid) or p2[iid]
            entry = dict(base)
            entry["interpretation_score"] = int(s1 == 1 or s2 == 1)
            merged[iid] = entry
        return merged
    raise ValueError(f"unknown interp version: {version}")


def load_detection(cfg):
    """Load detection_score.json → {id: 0/1}"""
    json_path = str(cfg["eval_output"] / "detection_score.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for item in data.get("results", []):
        iid = str(item.get("id", "")).strip()
        score = item.get("detection_score")
        if iid and score is not None:
            results[iid] = int(score)
    return results


def load_eval_detail(cfg, excluded_set=None, lte_prefix="translation_lte"):
    """加载 LTE 评估详情，派生 LTE 和 meaning_conveyed。

    LTE = literal_check.startswith("Yes") AND meaning_check.startswith("No")
    excluded_set: (id, prompt_type) 集合，命中则跳过
    lte_prefix: LTE 评分文件前缀，读 {prefix}_score.json。换 judge 复现时传如
                translation_lte_gpt52（默认 translation_lte = gpt-4o-mini canonical）
    """
    eval_dir = cfg["eval_output"]
    json_path = str(eval_dir / f"{lte_prefix}_score.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = []
    skipped_excluded = 0
    for item in data.get("results", []):
        pt = str(item.get("prompt_type", "")).strip()
        iid = item.get("id")
        if excluded_set and (int(iid), pt) in excluded_set:
            skipped_excluded += 1
            continue

        lit = parse_yes_no(item.get("literal_check", ""))
        mean_conv = parse_yes_no(item.get("meaning_check", ""))
        if lit is None or mean_conv is None:
            lte = None
        else:
            lte = 1 if (lit and not mean_conv) else 0

        results.append({
            "id": str(iid).strip(),
            "prompt_type": pt,
            "idiom": item.get("idiom", ""),
            "sentence": item.get("sentence", ""),
            "reference_meaning": item.get("reference_meaning", ""),
            "gold_translation": item.get("gold_translation", ""),
            "model_translation": item.get("model_translation", ""),
            "lte": lte,
            "meaning_conveyed": mean_conv,
            "literal_mapping_preserved": lit,
            "literal_check": item.get("literal_check", ""),    # 完整 evidence 字符串
            "meaning_check": item.get("meaning_check", ""),
            "confidence": item.get("confidence"),
        })
    if skipped_excluded:
        print(f"  [excluded_key] 跳过 {skipped_excluded} 条")
    return results


# ============================================================
# 分组
# ============================================================

def extract_groups(detection, interp, eval_items, direction):
    """提取 Group A / B (Group B strict 在 run_one 里从 B loose 过滤)"""
    group_a, group_b = [], []
    skipped = {"no_detection": 0, "d_zero": 0, "no_interp": 0,
               "i_zero": 0, "lte_none": 0, "other": 0}

    for item in eval_items:
        iid = item["id"]
        d_score = detection.get(iid)
        if d_score is None:
            skipped["no_detection"] += 1; continue
        if d_score != 1:
            skipped["d_zero"] += 1; continue

        i_entry = interp.get(iid)
        if not i_entry:
            skipped["no_interp"] += 1; continue
        if i_entry["interpretation_score"] != 1:
            skipped["i_zero"] += 1; continue

        if item["lte"] is None:
            skipped["lte_none"] += 1; continue

        entry = {
            "id": iid,
            "direction": direction,
            "prompt_type": item["prompt_type"],
            "idiom": item["idiom"],
            "sentence": item["sentence"],
            "reference_meaning": item.get("reference_meaning") or i_entry.get("reference_meaning", ""),
            "gold_translation": item["gold_translation"],
            "model_translation": item["model_translation"],
            "detection_score": d_score,
            "interpretation_score": i_entry["interpretation_score"],
            "lte": item["lte"],
            "meaning_conveyed": item["meaning_conveyed"],
            "literal_mapping_preserved": item["literal_mapping_preserved"],
            "literal_check": item["literal_check"],
            "meaning_check": item["meaning_check"],
            "confidence": item["confidence"],
        }

        if item["lte"] == 1:
            entry["group"] = "A"; group_a.append(entry)
        elif item["lte"] == 0:
            entry["group"] = "B"; group_b.append(entry)
        else:
            skipped["other"] += 1

    return group_a, group_b, skipped


# ============================================================
# 报告
# ============================================================

def write_report(group_a, group_b, skipped, direction, output_dir):
    path = os.path.join(output_dir, "report.txt")
    S = "=" * 80
    with open(path, "w", encoding="utf-8") as f:
        def w(s=""): f.write(s + "\n")
        w(S); w(f"Mechanistic Analysis — Group A / B — {direction}"); w(S); w()
        w("【统计】")
        w(f"  Group A (D=1, I=1, LTE=1):  {len(group_a)} 条 ({len(set(x['id'] for x in group_a))} 个习语)")
        w(f"  Group B (D=1, I=1, LTE=0):  {len(group_b)} 条 ({len(set(x['id'] for x in group_b))} 个习语)")
        w()
        w("【跳过】")
        for k, v in skipped.items(): w(f"  {k}: {v}")
        w()
        w("【Group A 按 Prompt 分布】")
        for pt in PROMPT_TYPES:
            w(f"  {pt}: {sum(1 for x in group_a if x['prompt_type'] == pt)}")
        w()
        w("【Group B 按 Prompt 分布】")
        for pt in PROMPT_TYPES:
            w(f"  {pt}: {sum(1 for x in group_b if x['prompt_type'] == pt)}")
    print(f"  报告: {path}")


# ============================================================
# 主流程
# ============================================================

def run_one(target, model, interp_version="p2", exp_name="", lte_prefix="translation_lte",
            filter_prompt=None):
    cfg = get_config(target, model=model)
    direction = cfg["direction"]

    base_dir = cfg["mechanistic_dir"]
    if interp_version == "p2":
        output_dir = str(base_dir)
    else:
        output_dir = str(Path(str(base_dir)).parent / f"mechanistic_{interp_version}")
    if exp_name:
        output_dir = str(Path(output_dir) / exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # 写 README.md 到实验目录
    if exp_name:
        import datetime
        readme_path = os.path.join(output_dir, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"# Experiment: {exp_name}\n\n")
                f.write(f"- lang_pair: {target}\n- model: {model}\n")
                f.write(f"- interp_version: {interp_version}\n")
                f.write(f"- lte_source: {lte_prefix}_score.json\n")
                f.write(f"- filter_prompt: {filter_prompt or 'None (all 4 prompts pooled)'}\n")
                f.write(f"- created: {datetime.datetime.now().isoformat()}\n\n")
                f.write("## Files\n")
                f.write("- group_known_lte.json — Group A (D=1, I=1, LTE=1)\n")
                f.write("- group_known_non_lte.json — Group B loose (D=1, I=1, LTE=0)\n")
                f.write("- group_known_non_lte_strict.json — Group B strict (+ MeaningConveyed=True)\n")
                f.write("- group_known_non_lte_superstrict.json — Group B super-strict (Literal=0, Meaning=1)\n")

    print(f"\n{'=' * 60}\n  {direction}  [{model}]\n{'=' * 60}")

    # 加载 exclusion_key (id, prompt_type 二元组)
    excluded_set = load_exclusion_set(target, model)
    if excluded_set:
        print(f"  exclusion_key: {len(excluded_set)} 条 (id, prompt_type) 将被排除")

    detection = load_detection(cfg)
    interp = load_interpretation(cfg, version=interp_version)
    eval_items = load_eval_detail(cfg, excluded_set=excluded_set, lte_prefix=lte_prefix)

    if filter_prompt:
        if filter_prompt not in PROMPT_TYPES:
            raise ValueError(f"filter_prompt={filter_prompt!r} 不在 {PROMPT_TYPES}")
        before = len(eval_items)
        eval_items = [it for it in eval_items if it["prompt_type"] == filter_prompt]
        print(f"  [filter_prompt={filter_prompt}] eval 条数 {before} → {len(eval_items)}")

    print(f"  Detection: {len(detection)} 条 | Interpretation ({interp_version}): {len(interp)} 条 | LTE: {len(eval_items)} 条")

    group_a, group_b, skipped = extract_groups(detection, interp, eval_items, direction)
    group_b_strict = [x for x in group_b if x.get("meaning_conveyed") is True]
    # super-strict: Literal=0 AND Meaning=1，剔除巧合保意 (Lit=1,Mean=1)，与 Group A (Lit=1,Mean=0) 严格反对角
    group_b_superstrict = [x for x in group_b
                           if x.get("meaning_conveyed") is True
                           and x.get("literal_mapping_preserved") is False]
    # Non-LTE 内三个格子（Group A = (Lit=1,Mean=0) 不在 group_b 里，单列）
    cell_11 = [x for x in group_b
               if x.get("literal_mapping_preserved") is True and x.get("meaning_conveyed") is True]   # 巧合保意
    cell_00 = [x for x in group_b
               if x.get("literal_mapping_preserved") is False and x.get("meaning_conveyed") is False]  # 幻觉/乱译
    print(f"  Group A:                {len(group_a):4d} 条 ({len(set(x['id'] for x in group_a))} 习语)")
    print(f"  Group B loose:          {len(group_b):4d} 条 ({len(set(x['id'] for x in group_b))} 习语)")
    print(f"  Group B strict:         {len(group_b_strict):4d} 条 ({len(set(x['id'] for x in group_b_strict))} 习语)")
    print(f"  Group B super-strict:   {len(group_b_superstrict):4d} 条 ({len(set(x['id'] for x in group_b_superstrict))} 习语)  [Lit=0,Mean=1]")
    print(f"  ├ (1,1) 巧合保意:       {len(cell_11):4d} 条  (从 strict 剔除)")
    print(f"  └ (0,0) 幻觉/乱译:      {len(cell_00):4d} 条  (本就不在 strict)")

    # 保存 3 个 JSON
    def _save(name, payload):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    base_meta = {"direction": direction, "model": model,
                 "interpretation_version": interp_version,
                 "lte_source": f"{lte_prefix}_score.json",
                 "filter_prompt": filter_prompt}

    _save("group_known_lte.json", {**base_meta, "group": "A",
        "description": "D=1, I=1, LTE=1",
        "count": len(group_a), "unique_idioms": len(set(x["id"] for x in group_a)),
        "results": group_a})

    _save("group_known_non_lte.json", {**base_meta, "group": "B",
        "description": "D=1, I=1, LTE=0 (loose)",
        "count": len(group_b), "unique_idioms": len(set(x["id"] for x in group_b)),
        "results": group_b})

    _save("group_known_non_lte_strict.json", {**base_meta, "group": "B_strict",
        "description": "D=1, I=1, LTE=0, MeaningConveyed=True (strict)",
        "count": len(group_b_strict), "unique_idioms": len(set(x["id"] for x in group_b_strict)),
        "results": group_b_strict})

    _save("group_known_non_lte_superstrict.json", {**base_meta, "group": "B_superstrict",
        "description": "D=1, I=1, Literal=0, Meaning=1 (super-strict; 剔除巧合保意 (Lit=1,Mean=1))",
        "count": len(group_b_superstrict), "unique_idioms": len(set(x["id"] for x in group_b_superstrict)),
        "results": group_b_superstrict})

    summary = {
        **base_meta,
        "group_definition": {
            "A": "D=1, I=1, LTE=1",
            "B_loose": "D=1, I=1, LTE=0",
            "B_strict": "D=1, I=1, LTE=0, MeaningConveyed=True",
        },
        "total_eval_items": len(eval_items),
        "total_d1": sum(1 for v in detection.values() if v == 1),
        "total_i1": sum(1 for v in interp.values() if v["interpretation_score"] == 1),
        "excluded_by_key": len(excluded_set),
        "group_A": {"count": len(group_a), "unique_idioms": len(set(x["id"] for x in group_a)),
                    "by_prompt": {pt: sum(1 for x in group_a if x["prompt_type"] == pt) for pt in PROMPT_TYPES}},
        "group_B_loose": {"count": len(group_b), "unique_idioms": len(set(x["id"] for x in group_b)),
                          "by_prompt": {pt: sum(1 for x in group_b if x["prompt_type"] == pt) for pt in PROMPT_TYPES}},
        "group_B_strict": {"count": len(group_b_strict), "unique_idioms": len(set(x["id"] for x in group_b_strict)),
                           "by_prompt": {pt: sum(1 for x in group_b_strict if x["prompt_type"] == pt) for pt in PROMPT_TYPES}},
        "group_B_superstrict": {"count": len(group_b_superstrict), "unique_idioms": len(set(x["id"] for x in group_b_superstrict)),
                                "by_prompt": {pt: sum(1 for x in group_b_superstrict if x["prompt_type"] == pt) for pt in PROMPT_TYPES}},
        "cells": {
            "_note": "四象限 (Literal, Meaning); Group A=(1,0), Non-LTE=group_b={(1,1),(0,1),(0,0)}",
            "lit1_mean0_LTE_groupA": len(group_a),
            "lit1_mean1_coincidental": len(cell_11),
            "lit0_mean1_idiomatic_B_superstrict": len(group_b_superstrict),
            "lit0_mean0_hallucination": len(cell_00),
        },
        "skipped": skipped,
    }
    _save("summary.json", summary)

    write_report(group_a, group_b, skipped, direction, output_dir)
    print(f"  输出: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="提取 Group A / B 用于 linear probing 与机制分析")
    parser.add_argument("--lang-pair", choices=ALL_TARGETS, help="单个翻译方向")
    parser.add_argument("--model", required=True, help="模型名，如 Qwen3.5-9B")
    parser.add_argument("--all", action="store_true", help="跑全部 6 个方向")
    parser.add_argument("--interp-version", choices=["p1", "p2", "union"], default="p2",
                        help="P2 (with-context, 默认) / P1 (严格) / union")
    parser.add_argument("--exp-name", default="", help="实验名，输出到 mechanistic/{exp_name}/")
    parser.add_argument("--lte-prefix", default="translation_lte",
                        help="LTE 评分文件前缀，读 {prefix}_score.json（换 judge 复现用，如 translation_lte_gpt52）")
    parser.add_argument("--filter-prompt", default=None, choices=PROMPT_TYPES,
                        help="只保留单一 prompt 的实例（默认 None = 4 prompt 全 pool，与论文一致；"
                             "用 BasicPrompt 做 Basic-only 分析以回应 reviewer 关于 cross-prompt pooling 的顾虑）")
    args = parser.parse_args()

    if not args.lang_pair and not args.all:
        parser.error("请指定 --lang-pair 或 --all")

    targets = ALL_TARGETS if args.all else [args.lang_pair]
    for target in targets:
        try:
            run_one(target, model=args.model,
                    interp_version=args.interp_version,
                    exp_name=args.exp_name, lte_prefix=args.lte_prefix,
                    filter_prompt=args.filter_prompt)
        except Exception as e:
            print(f"  [错误] {target}: {e}")

    print("\n完成！")


if __name__ == "__main__":
    main()
