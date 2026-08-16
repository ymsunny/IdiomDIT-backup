"""
eval_interpretation.py — Interpretation 评估（Judge 评分 + 报告）

读取 interpretation_inference.json，对每条结果进行 Judge 评估，
输出 interpretation_p1_score.json、interpretation_p2_score.json。

用法：
  python evaluation/eval_interpretation.py --lang-pair fi-en --model Qwen3.5-4B
"""

import os
import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from openai import OpenAI
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    get_config, ALL_TARGETS, MODEL_NAME,
    API_BASE_URL, API_KEY, JUDGE_MODEL,
    API_TIMEOUT, API_TEMPERATURE,
    API_MAX_RETRIES, API_RETRY_DELAY, setup_logger,
)

# ============================================================
# 配置
# ============================================================

NUM_RUNS = 1
JUDGE_TEMPERATURE = API_TEMPERATURE
JUDGE_MAX_TOKENS = 256
MAX_RETRIES = API_MAX_RETRIES
RETRY_DELAY = API_RETRY_DELAY

logger = logging.getLogger("eval_interpretation")

# ============================================================
# Probe key constants
# ============================================================

PROBE_PROMPTS = {
    "P1a": "What is the idiomatic meaning of \"{idiom}\"? Output JSON only: {\"meaning\": \"...\"}",
    "P1b": "Explain the idiomatic meaning of \"{idiom}\". Output JSON only: {\"meaning\": \"...\"}",
    "P1c": "What does \"{idiom}\" idiomatically mean? Output JSON only: {\"meaning\": \"...\"}",
    "P2a": "What is the idiomatic meaning of \"{idiom}\" in: \"{sentence}\"? Output JSON only: {\"meaning\": \"...\"}",
    "P2b": "Explain the idiomatic meaning of \"{idiom}\" in: \"{sentence}\". Output JSON only: {\"meaning\": \"...\"}",
    "P2c": "What does \"{idiom}\" idiomatically mean in: \"{sentence}\"? Output JSON only: {\"meaning\": \"...\"}",
}
P1_KEYS = ["P1a", "P1b", "P1c"]
P2_KEYS = ["P2a", "P2b", "P2c"]

# ============================================================
# Judge Prompt
# ============================================================

JUDGE_SYSTEM = """You are a judge evaluating whether a model's explanation of an idiom is correct.
Output ONLY a JSON object: {"correct": true/false, "reason": "brief reason"}
No markdown, no explanation outside JSON."""

JUDGE_USER = """Reference meaning: {reference_meaning}
Model's answer: {model_answer}

Does the model's answer correctly capture the essential meaning of the idiom?
- Be lenient: the answer doesn't need to be word-perfect, just semantically correct
- If the model says "I don't know" or gives a wrong meaning, mark as incorrect
- Output JSON only: {{"correct": true/false, "reason": "..."}}"""

# ============================================================
# Judge
# ============================================================


def init_judge():
    key = API_KEY or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("请设置 OPENAI_API_KEY")
    return OpenAI(base_url=API_BASE_URL, api_key=key, timeout=API_TIMEOUT)


def extract_meaning(answer: str) -> str:
    """Extract meaning field from JSON response, fall back to raw answer."""
    try:
        parsed = json.loads(re.sub(r"```json|```", "", answer).strip())
        return parsed.get("meaning", answer)
    except Exception:
        return answer


def judge_answer(client, reference_meaning: str, model_answer: str) -> Tuple[bool, str]:
    user_prompt = JUDGE_USER.format(
        reference_meaning=reference_meaning, model_answer=model_answer)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user_prompt}])
            raw = resp.choices[0].message.content.strip()
            for c in [raw, raw[raw.find("{"):raw.rfind("}")+1] if "{" in raw else ""]:
                if not c:
                    continue
                try:
                    parsed = json.loads(c)
                    correct = parsed.get("correct", False)
                    if isinstance(correct, str):
                        correct = correct.lower() in ("true", "yes", "1")
                    return bool(correct), parsed.get("reason", "")
                except Exception:
                    continue
            return False, f"JSON解析失败: {raw[:100]}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return False, f"API失败: {e}"


# ============================================================
# Judge 阶段
# ============================================================

def run_judge(results: List[Dict], meanings: Dict[str, str]) -> List[Dict]:
    client = init_judge()
    total_calls = len(results) * len(PROBE_PROMPTS) * NUM_RUNS
    logger.info(f"Judge 评估: {total_calls} 次 API 调用...")

    for i, item in enumerate(results):
        meaning = item.get('reference_meaning', '') or meanings.get(str(item['id']), '')
        for prompt_key in PROBE_PROMPTS:
            responses = item.get(f"{prompt_key}__responses", [])
            judgments = []
            any_correct = False
            for answer in responses:
                if answer.startswith("ERROR:"):
                    judgments.append({"correct": False, "reason": "模型报错"})
                    continue
                answer = extract_meaning(answer)
                correct, reason = judge_answer(client, meaning, answer)
                judgments.append({"correct": correct, "reason": reason})
                if correct:
                    any_correct = True
                time.sleep(0.3)
            item[f"{prompt_key}__judgments"] = judgments
            item[f"{prompt_key}__any_correct"] = any_correct
            item[f"{prompt_key}__hit_count"] = sum(1 for j in judgments if j["correct"])

        item["I_no_context"] = any(item.get(f"{pk}__any_correct", False) for pk in P1_KEYS)
        item["I_with_context"] = any(item.get(f"{pk}__any_correct", False) for pk in P2_KEYS)
        item["interpretation_score_P1"] = 1 if item["I_no_context"] else 0
        item["interpretation_score_P2"] = 1 if item["I_with_context"] else 0

        if (i + 1) % 10 == 0:
            logger.info(f"  Judge: {i+1}/{len(results)} 完成")

    return results


# ============================================================
# 报告输出
# ============================================================

def write_report(results, output_dir, source, model_name=None):
    _model = model_name or MODEL_NAME

    p1_results = []
    p2_results = []
    for item in results:
        base = {
            "id": item["id"], "idiom": item["idiom"],
            "sentence": item.get("sentence", ""),
            "reference_meaning": item.get("reference_meaning", ""),
        }
        p1_entry = {**base, "interpretation_score": item.get("interpretation_score_P1", 0)}
        for pk in P1_KEYS:
            p1_entry[f"{pk}_correct"] = int(item.get(f"{pk}__any_correct", False))
            p1_entry[f"{pk}_responses"] = item.get(f"{pk}__responses", [])
            p1_entry[f"{pk}_judgments"] = item.get(f"{pk}__judgments", [])
        p1_results.append(p1_entry)

        p2_entry = {**base, "interpretation_score": item.get("interpretation_score_P2", 0)}
        for pk in P2_KEYS:
            p2_entry[f"{pk}_correct"] = int(item.get(f"{pk}__any_correct", False))
            p2_entry[f"{pk}_responses"] = item.get(f"{pk}__responses", [])
            p2_entry[f"{pk}_judgments"] = item.get(f"{pk}__judgments", [])
        p2_results.append(p2_entry)

    p1_path = os.path.join(output_dir, "interpretation_p1_score.json")
    with open(p1_path, 'w', encoding='utf-8') as f:
        json.dump({"source": source, "model": _model, "num_runs": NUM_RUNS,
                   "group": "P1_no_context", "prompts": P1_KEYS,
                   "results": p1_results}, f, ensure_ascii=False, indent=2)

    p2_path = os.path.join(output_dir, "interpretation_p2_score.json")
    with open(p2_path, 'w', encoding='utf-8') as f:
        json.dump({"source": source, "model": _model, "num_runs": NUM_RUNS,
                   "group": "P2_with_context", "prompts": P2_KEYS,
                   "results": p2_results}, f, ensure_ascii=False, indent=2)

    logger.info(f"  P1: {p1_path}")
    logger.info(f"  P2: {p2_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Interpretation 评估（Judge 评分）")
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    model_name = args.model
    logger = setup_logger("eval_interpretation", lang_pair=args.lang_pair, model=model_name)
    cfg = get_config(args.lang_pair, model=model_name)

    eval_output_dir = str(cfg["eval_output"])
    os.makedirs(eval_output_dir, exist_ok=True)

    logger.info(f"eval_interpretation: target={args.lang_pair} model={model_name}")

    inference_path = cfg["interpretation_dir"] / "interpretation_inference.json"
    if not inference_path.is_file():
        logger.error(f"推理结果不存在: {inference_path}")
        return

    with open(inference_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if args.max_samples:
        results = results[:args.max_samples]
        logger.info(f"--max-samples={args.max_samples}, 实际处理 {len(results)} 条")

    logger.info(f"加载 {len(results)} 条推理结果")

    meaning_file = cfg["meaning_file"]
    meanings = {}
    if meaning_file and os.path.exists(meaning_file):
        with open(meaning_file, 'r', encoding='utf-8') as f:
            for item in json.load(f).get('results', []):
                meanings[str(item['id'])] = item.get('reference_meaning', '')
        logger.info(f"Meanings: {len(meanings)} 条")

    results = run_judge(results, meanings)

    failed = [r for r in results
              if any("API失败" in str(j.get("reason", ""))
                     for pk in P1_KEYS + P2_KEYS
                     for j in r.get(f"{pk}__judgments", []))]
    logger.info(f"总计: {len(results)} 条 | 失败: {len(failed)}")
    if failed:
        logger.error(f"⚠ {len(failed)} 条评估因 API 错误失败！")

    write_report(results, eval_output_dir, args.lang_pair, model_name=model_name)
    logger.info(f"完成！输出目录: {eval_output_dir}")


if __name__ == "__main__":
    main()
