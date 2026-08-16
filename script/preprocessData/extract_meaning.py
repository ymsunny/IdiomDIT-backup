"""
extract_meaning.py — 为 CSV 中每条习语抽取引申义

输入: Data/<DATASET>/ParallelData/<CSV>
输出: Data/<DATASET>/Meaning/<MEANINGS>.json

用法:
  python extract_meaning.py --target all          # 跑全部数据集
  python extract_meaning.py --target en_fa        # 单个方向
  python extract_meaning.py --target ja_en ko_en  # 多个方向
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
from openai import OpenAI

# 自动加载项目根目录的 .env 文件
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    API_BASE_URL, API_KEY, API_TIMEOUT,
    API_MAX_RETRIES as MAX_RETRIES,
    API_RETRY_DELAY as RETRY_DELAY, API_REQUEST_INTERVAL as REQUEST_INTERVAL,
    _BASE,
)

# ============================================================
# 配置
# ============================================================

MODEL = "gemini-flash-latest-nothinking"
TEMPERATURE = 0
MAX_TOKENS = 256

# ============================================================
# 任务注册表
# ============================================================

from config import LANG_PAIRS, get_data_paths

# 从 LANG_PAIRS 自动生成任务列表
TASKS = {}
for _pair, _lp in LANG_PAIRS.items():
    for _d in _lp["directions"]:
        _code = f"{_d['src']}_{_d['tgt']}"
        _dp = get_data_paths(_pair, f"{_d['src']}-{_d['tgt']}")
        TASKS[_code] = {
            "csv":      _dp["csv_file"],
            "output":   _dp["meaning_file"],
            "src_lang": _dp["src_lang"],
            "tgt_lang": _dp["tgt_lang"],
        }

# ============================================================
# Prompt
# ============================================================

EXTRACT_PROMPT = """Provide the idiomatic meaning of the idiom "{idiom}" as used in the source sentence, based on the context of the sentence and its translation.
Source Sentence: {source}
Translation: {gold_translation}
Respond only with the idiom's meaning in English without any explanation, formatted as a JSON object:
{{ "output": "meaning content" }}"""

# ============================================================
# 核心函数
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(_BASE / "logs" / "extract_meaning.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def init_client() -> OpenAI:
    api_key = API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    return OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=API_TIMEOUT)


def call_llm(client: OpenAI, prompt: str) -> Optional[str]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API 失败 (第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def parse_meaning(response_text: str) -> Optional[str]:
    if not response_text:
        return None
    for text in [response_text, response_text.replace("```json", "").replace("```", "").strip()]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end]).get("output", "")
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def extract_meanings(csv_path: Path, output_path: Path, src_lang: str, tgt_lang: str):
    client = init_client()
    df = pd.read_csv(csv_path, encoding="utf-8")
    total = len(df)
    logger.info(f"加载 {csv_path.name}: {total} 条")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_path) + ".checkpoint"

    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as fp:
            completed = json.load(fp)
        logger.info(f"断点续传: 已完成 {len(completed)} 条")

    results = []
    for idx, row in df.iterrows():
        row_id = idx + 1
        key = str(row_id)

        if key in completed:
            results.append(completed[key])
            continue

        logger.info(f"[{row_id}/{total}] {row['idiom']}")

        prompt = EXTRACT_PROMPT.format(
            idiom=row["idiom"],
            source=row["sentence"],
            gold_translation=row["gold translation"]
        )

        raw = call_llm(client, prompt)
        meaning = parse_meaning(raw)

        entry = {
            "id": row_id,
            "idiom": row["idiom"],
            "sentence": row["sentence"],
            "gold_translation": row["gold translation"],
            "reference_meaning": meaning if meaning else ""
        }

        results.append(entry)
        completed[key] = entry

        if len(results) % 10 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as fp:
                json.dump(completed, fp, ensure_ascii=False)

        time.sleep(REQUEST_INTERVAL)

    output_data = {
        "config": {
            "model": MODEL,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, ensure_ascii=False, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    success = sum(1 for r in results if r["reference_meaning"])
    logger.info(f"完成: {success}/{total} 成功 → {output_path}")


def run_task(target: str):
    task = TASKS[target]
    csv_path = task["csv"]
    output_path = task["output"]

    if not csv_path.exists():
        logger.warning(f"找不到 {csv_path}，跳过")
        return

    # 已完整则跳过
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
            existing_count = len(existing.get("results", []))
            csv_count = len(pd.read_csv(csv_path))
            if existing_count == csv_count:
                logger.info(f"[{target}] 已完整 ({existing_count} 条)，跳过")
                return
        except Exception:
            pass

    logger.info(f"[{target}] 开始抽取: {csv_path.name} → {output_path.name}")
    extract_meanings(csv_path, output_path, task["src_lang"], task["tgt_lang"])


def main():
    parser = argparse.ArgumentParser(description="抽取习语引申义")
    parser.add_argument(
        "--lang-pair", nargs="+",
        choices=list(TASKS.keys()) + ["all"],
        default=["all"],
        help=f"要处理的数据集，可选: {', '.join(TASKS.keys())}, all（默认 all）"
    )
    args = parser.parse_args()

    targets = list(TASKS.keys()) if "all" in args.lang_pair else args.lang_pair

    logger.info(f"目标: {targets}")
    for t in targets:
        run_task(t)
    logger.info("全部完成!")


if __name__ == "__main__":
    main()
