#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_ko_en.py — 清洗 Ko→En 习语数据集（KISS）

原始数据：Judy-Choi/KISS-Korean-english-Idioms-in-Sentences-dataSet
  KISS.csv（无 header，3列：韩语习语 / 韩语句子 / 英语译文）

输出 schema 与 En_Fa.csv / Fa-En.csv 一致：
  idiom, sentence, gold translation

用法：
  python scripts/clean_ko_en.py --input <path_or_url> --output Data/Ko-En-Idiom/ParallelData/Ko_En_clean.csv
"""

import csv
import argparse
import urllib.request
import os


RAW_URL = (
    "https://raw.githubusercontent.com/Judy-Choi/KISS-Korean-english-Idioms-in-Sentences-dataSet"
    "/master/KISS.csv"
)

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data", "Ko-En-Idiom", "ParallelData", "Ko_En_clean.csv"
)

FIELDNAMES = ["idiom", "sentence", "gold translation"]


def load_csv(source: str) -> list[dict]:
    """从文件路径或 URL 读取 CSV（无 header），返回 dict 列表。"""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"  下载: {source}")
        with urllib.request.urlopen(source) as resp:
            content = resp.read().decode("utf-8")
        lines = content.splitlines()
        reader = csv.DictReader(lines, fieldnames=FIELDNAMES)
    else:
        print(f"  读取: {source}")
        with open(source, encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=FIELDNAMES)
            return list(reader)
    return list(reader)


def clean(rows: list[dict]) -> tuple[list[dict], dict]:
    """按习语去重，返回 (cleaned_rows, stats)。"""
    stats = {
        "total": len(rows),
        "skip_duplicate_idiom": 0,
        "final": 0,
    }

    seen_idioms: set[str] = set()
    result = []

    for row in rows:
        idiom = (row.get("idiom") or "").strip()
        sentence = (row.get("sentence") or "").strip()
        gold = (row.get("gold translation") or "").strip()

        if not idiom or not sentence or not gold:
            continue

        # 按习语去重，每个习语只保留第一次出现
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
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(stats: dict) -> None:
    print(f"\n  清洗统计：")
    print(f"    原始行数:        {stats['total']}")
    print(f"    跳过（习语重复）:{stats['skip_duplicate_idiom']}")
    print(f"    ─────────────────────────")
    print(f"    最终行数:        {stats['final']}")


def main():
    parser = argparse.ArgumentParser(description="清洗 Ko→En 习语数据集（KISS）")
    parser.add_argument("--input", default=RAW_URL,
                        help="原始 CSV 路径或 URL（默认从 GitHub 下载）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"输出路径（默认 {DEFAULT_OUTPUT}）")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Ko->En 习语数据集清洗（KISS）")
    print(f"{'='*55}")

    rows = load_csv(args.input)
    print(f"  读取完成：{len(rows)} 行")

    cleaned, stats = clean(rows)
    print_stats(stats)

    write_csv(cleaned, args.output)
    print(f"\n  Done -> {args.output}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
