#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_ja_en.py — 清洗 Ja→En 习语数据集

原始数据：nightingal3/idiom-translation
  metaphor-translation/data/test_sets_final/jp/idiomatic_all.csv

输出 schema 与 En_Fa.csv / Fa-En.csv 一致：
  idiom, sentence, gold translation

用法：
  python scripts/clean_ja_en.py --input <path_or_url> --output Data/Ja-En-Idiom/ParallelData/Ja_En.csv
"""

import csv
import argparse
import urllib.request
import os
import sys


RAW_URL = (
    "https://raw.githubusercontent.com/nightingal3/idiom-translation/main"
    "/metaphor-translation/data/test_sets_final/jp/idiomatic_all.csv"
)

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data", "Ja-En-Idiom", "ParallelData", "Ja_En_clean.csv"
)


def load_csv(source: str) -> list[dict]:
    """从文件路径或 URL 读取 CSV，返回 dict 列表。"""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"  下载: {source}")
        with urllib.request.urlopen(source) as resp:
            content = resp.read().decode("utf-8")
        lines = content.splitlines()
        reader = csv.DictReader(lines)
    else:
        print(f"  读取: {source}")
        with open(source, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    return list(reader)


MIN_GOLD_WORDS = 3      # gold translation 最少单词数
MIN_SENTENCE_EXTRA = 3  # sentence 至少比 idiom 多出的字符数


def clean(rows: list[dict]) -> tuple[list[dict], dict]:
    """按清洗规则处理，返回 (cleaned_rows, stats)。"""
    stats = {
        "total": len(rows),
        "skip_multi_idiom": 0,
        "skip_short_sentence": 0,
        "skip_short_translation": 0,
        "skip_duplicate_idiom": 0,
        "final": 0,
    }

    seen_idioms: set[str] = set()
    result = []

    for row in rows:
        raw_idiom = row.get("contains_idioms", "").strip()
        sentence = row.get("original_text", "").strip()
        gold = row.get("text", "").strip()

        # Step 1: 跳过多习语行（含英文逗号）
        if "," in raw_idiom:
            stats["skip_multi_idiom"] += 1
            continue

        # Step 2: 重建 idiom（去掉空格分词）
        idiom = raw_idiom.replace(" ", "")

        # Step 3a: 跳过 sentence 过短的行（没有上下文，基本就是习语本身）
        if len(sentence) < len(idiom) + MIN_SENTENCE_EXTRA:
            stats["skip_short_sentence"] += 1
            continue

        # Step 3b: 跳过 gold translation 单词数不足的行（字幕省译噪声）
        if len(gold.split()) < MIN_GOLD_WORDS:
            stats["skip_short_translation"] += 1
            continue

        # Step 4: 按习语去重，每个习语只保留第一次出现
        if idiom in seen_idioms:
            stats["skip_duplicate_idiom"] += 1
            continue
        seen_idioms.add(idiom)

        result.append({
            "idiom": idiom,
            "sentence": sentence,
            "gold translation": gold,
        })

    stats["final"] = len(result)
    return result, stats


def write_csv(rows: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idiom", "sentence", "gold translation"])
        writer.writeheader()
        writer.writerows(rows)


def print_stats(stats: dict) -> None:
    print(f"\n  清洗统计：")
    print(f"    原始行数:         {stats['total']}")
    print(f"    跳过（多习语）:      {stats['skip_multi_idiom']}")
    print(f"    跳过（sentence过短）:{stats['skip_short_sentence']}")
    print(f"    跳过（译文词数<3）:  {stats['skip_short_translation']}")
    print(f"    跳过（习语重复）:    {stats['skip_duplicate_idiom']}")
    print(f"    ─────────────────────────")
    print(f"    最终行数:         {stats['final']}")


def verify(rows: list[dict]) -> None:
    """简单验证：idiom 列无空格、无逗号。"""
    issues = 0
    for i, row in enumerate(rows):
        idiom = row["idiom"]
        if " " in idiom:
            print(f"  [WARN] 行 {i+1}: idiom 含空格: {idiom!r}")
            issues += 1
        if "," in idiom:
            print(f"  [WARN] 行 {i+1}: idiom 含逗号: {idiom!r}")
            issues += 1
    if issues == 0:
        print(f"  验证通过：idiom 列格式正常")


def main():
    parser = argparse.ArgumentParser(description="清洗 Ja→En 习语数据集")
    parser.add_argument("--input", default=RAW_URL,
                        help=f"原始 CSV 路径或 URL（默认从 GitHub 下载）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"输出路径（默认 {DEFAULT_OUTPUT}）")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Ja→En 习语数据集清洗")
    print(f"{'='*55}")

    rows = load_csv(args.input)
    print(f"  读取完成：{len(rows)} 行")

    cleaned, stats = clean(rows)
    print_stats(stats)

    print(f"\n  验证输出...")
    verify(cleaned)

    write_csv(cleaned, args.output)
    print(f"\n  Done -> {args.output}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
