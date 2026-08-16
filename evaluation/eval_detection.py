"""
eval_detection.py — Detection 评估（Judge 评分）

读取 detection_inference.json，对每条结果进行 Judge 评估，输出 detection_score.json。

评分逻辑：
  - has_idiom is None → detection_score = None
  - has_idiom is False → detection_score = 0
  - has_idiom is True and detected_idiom empty → detection_score = 0
  - has_idiom is True and detected_idiom not empty →
      quick_string_match → if match: 1
      else → judge_idiom_match API → 1 if match else 0

用法：
  python eval_detection.py --lang-pair fi-en --model Qwen3.5-4B
"""

import os
import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Tuple

from openai import OpenAI
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    get_config, ALL_TARGETS,
    API_BASE_URL, API_KEY, JUDGE_MODEL,
    API_TIMEOUT, API_MAX_RETRIES, API_RETRY_DELAY,
    API_REQUEST_INTERVAL, setup_logger,
)

# ============================================================
# 配置
# ============================================================

MAX_RETRIES = API_MAX_RETRIES
RETRY_DELAY = API_RETRY_DELAY
REQUEST_INTERVAL = API_REQUEST_INTERVAL

JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 128

logger = logging.getLogger("eval_detection")

# ============================================================
# Judge Prompt — 判断两个短语是否指同一个习语
# ============================================================

JUDGE_SYSTEM = """You are a linguistic judge. Your task is to determine whether two phrases refer to the same idiom, allowing for inflectional variations, pronoun changes, and partial forms.

Output ONLY a JSON object: {"match": true/false, "reason": "brief reason"}
No markdown, no explanation outside JSON."""

JUDGE_USER = """Gold idiom (dictionary form): {gold_idiom}
Model detected: {detected_idiom}
Sentence context: {sentence}

Do these refer to the same idiomatic expression? Consider:
- Inflectional forms: "drag one's feet" = "dragging my feet" = "dragged his feet"
- Pronoun substitution: "one's" = "my/his/her/their"
- Partial overlap: "break the ice" = "broke the ice"
- But reject completely different idioms or non-idiomatic phrases

Output JSON only: {{"match": true/false, "reason": "..."}}"""

# ============================================================
# Judge 客户端
# ============================================================


def init_judge():
    api_key = API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY")
    return OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=API_TIMEOUT)


def quick_string_match(gold_idiom: str, detected_idiom: str) -> bool:
    """快速字符串匹配 — 明显匹配的直接通过，不用调 Judge API"""
    if not gold_idiom or not detected_idiom:
        return False
    gold_lower = gold_idiom.lower().strip()
    detected_lower = detected_idiom.lower().strip()

    # 精确匹配
    if gold_lower == detected_lower:
        return True
    # 子串包含
    if gold_lower in detected_lower or detected_lower in gold_lower:
        return True
    # 去除标点后匹配
    gold_clean = re.sub(r'[^\w\s]', '', gold_lower)
    detected_clean = re.sub(r'[^\w\s]', '', detected_lower)
    if gold_clean == detected_clean:
        return True
    if gold_clean in detected_clean or detected_clean in gold_clean:
        return True
    return False


def judge_idiom_match(client, gold_idiom: str, detected_idiom: str,
                      sentence: str) -> Tuple[bool, str]:
    """用 GPT-4o-mini 判断 detected_idiom 是否与 gold_idiom 指同一个习语"""
    user_prompt = JUDGE_USER.format(
        gold_idiom=gold_idiom, detected_idiom=detected_idiom, sentence=sentence)

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
                    match = parsed.get("match", False)
                    if isinstance(match, str):
                        match = match.lower() in ("true", "yes", "1")
                    return bool(match), parsed.get("reason", "")
                except:
                    continue
            return False, f"JSON解析失败: {raw[:100]}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return False, f"API失败: {e}"


# ============================================================
# 主流程
# ============================================================

def run_eval(cfg, max_samples: int = None):
    """对 detection_inference.json 进行 Judge 评分，输出 detection_score.json"""
    judge_client = init_judge()

    inference_path = cfg["detection_dir"] / "detection_inference.json"
    if not inference_path.is_file():
        logger.error(f"推理结果不存在: {inference_path}")
        return

    with open(inference_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持 {"results": [...]} 或 [...] 两种格式
    if isinstance(data, dict):
        config_info = data.get("config", {})
        results = data.get("results", [])
    else:
        config_info = {}
        results = data

    if max_samples:
        results = results[:max_samples]
        logger.info(f"--max-samples={max_samples}, 实际处理 {len(results)} 条")

    logger.info(f"加载 {len(results)} 条推理结果: {inference_path}")
    judge_calls = 0

    for i, entry in enumerate(results):
        has_idiom = entry.get("has_idiom")
        detected_idiom = entry.get("detected_idiom", "")
        gold_idiom = entry.get("idiom", "")
        sentence = entry.get("sentence", "")

        if has_idiom is None:
            entry["detection_score"] = None
            entry["idiom_match"] = None
            entry["judge_reason"] = ""
        elif not has_idiom:
            entry["detection_score"] = 0
            entry["idiom_match"] = False
            entry["judge_reason"] = "Model said no idiom"
        elif not detected_idiom:
            entry["detection_score"] = 0
            entry["idiom_match"] = False
            entry["judge_reason"] = "Model said has_idiom=true but gave no idiom text"
        else:
            if quick_string_match(gold_idiom, detected_idiom):
                entry["detection_score"] = 1
                entry["idiom_match"] = True
                entry["judge_reason"] = "Quick string match"
            else:
                match, reason = judge_idiom_match(
                    judge_client, gold_idiom, detected_idiom, sentence)
                entry["detection_score"] = 1 if match else 0
                entry["idiom_match"] = match
                entry["judge_reason"] = reason
                judge_calls += 1
                time.sleep(0.3)

        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(results)} 完成")

    # 更新 config
    config_info["judge_model"] = JUDGE_MODEL

    # 输出
    eval_output = cfg["eval_output"]
    os.makedirs(str(eval_output), exist_ok=True)

    output_data = {
        "config": config_info,
        "results": results,
    }
    output_path = eval_output / "detection_score.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 统计
    valid = [r for r in results if r.get("detection_score") is not None]
    failed = [r for r in results if "API失败" in str(r.get("judge_reason", ""))]
    det_rate = sum(r["detection_score"] for r in valid) / len(valid) if valid else 0
    said_yes = sum(1 for r in valid if r.get("has_idiom") is True)
    logger.info(f"完成 → {output_path}")
    logger.info(f"  总计: {len(results)} 条 | 成功: {len(valid)} | 失败: {len(failed)}")
    if failed:
        logger.error(f"  ⚠ {len(failed)} 条评估因 API 错误失败！请检查 API key 和网络连接")
    logger.info(f"  Model said has_idiom=true: {said_yes}/{len(valid)}")
    logger.info(f"  Detection Score (judge-confirmed): {det_rate:.3f} "
                f"({sum(r['detection_score'] for r in valid)}/{len(valid)})")
    logger.info(f"  Judge API calls: {judge_calls}")


def main():
    parser = argparse.ArgumentParser(description="Detection 评估（Judge 评分）")
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS,
                        help="目标，如 en-fa, ja-en")
    parser.add_argument("--model", required=True,
                        help="模型名称，如 Qwen3.5-4B")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只处理前 N 条数据（测试用）")
    args = parser.parse_args()

    global logger
    logger = setup_logger("eval_detection", lang_pair=args.lang_pair, model=args.model)

    cfg = get_config(args.lang_pair, model=args.model)
    os.makedirs(str(cfg["eval_output"]), exist_ok=True)

    logger.info(f"eval_detection: target={args.lang_pair} model={args.model}")
    logger.info(f"  输入: {cfg['detection_dir'] / 'detection_inference.json'}")
    logger.info(f"  输出: {cfg['eval_output'] / 'detection_score.json'}")

    run_eval(cfg, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
