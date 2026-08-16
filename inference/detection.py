"""
Stage 1: 习语检测 — 推理 (Idiom Detection — Inference Only)
测试模型能否识别句子中的习语表达
- 被测模型: 本地 transformers (--local) 或 API (--api)
- 本文件只做推理 + 解析，不做 Judge 评分
- Judge 评分由 eval_detection.py 或 --judge-only 模式完成

输入: CSV 文件 (idiom, sentence, gold translation)
输出: detection_inference.json (has_idiom, detected_idiom, raw_response)

调用:
  python detection.py --target en-fa --local Qwen/Qwen3.5-4B
  python detection.py --target en-fa --api gpt-4o-mini
  python detection.py --target en-fa --judge-only   # 对已有推理结果跑 Judge 评分
"""

import os
import json
import time
import re
import logging
import argparse
import pandas as pd
from typing import Optional, Dict, Tuple
from openai import OpenAI

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    API_BASE_URL, API_KEY, API_TIMEOUT, ENABLE_THINKING,
    MAX_NEW_TOKENS_THINKING_DETECTION,
    JUDGE_MODEL, API_MAX_RETRIES as MAX_RETRIES,
    API_RETRY_DELAY as RETRY_DELAY, API_REQUEST_INTERVAL as REQUEST_INTERVAL,
    setup_logger,
)
from model_utils import strip_thinking_output

# ============================================================
# 配置
# ============================================================

TARGET_MAX_TOKENS = 256
TARGET_TEMPERATURE = 0.0
# thinking 模式下 CoT 在 </think> 前会消耗大量 token，256 会被截断；用专用预算。
DETECT_MAX_TOKENS = MAX_NEW_TOKENS_THINKING_DETECTION if ENABLE_THINKING else TARGET_MAX_TOKENS

JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 128

logger = logging.getLogger("detection")

# ============================================================
# Prompt
# ============================================================

DETECTION_PROMPT = """Read the following sentence carefully.

Sentence: {sentence}

Does this sentence contain an idiomatic expression (an expression whose meaning is different from the literal meaning of its individual words)?

If yes, identify the idiomatic expression and respond in JSON format:
{{ "has_idiom": true, "idiom": "the idiomatic expression" }}

If no idiomatic expression is found, respond:
{{ "has_idiom": false, "idiom": "" }}

Output only the JSON object, no explanation."""

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
# 被测模型
# ============================================================

class TargetModel:
    def generate(self, prompt: str) -> Optional[str]:
        raise NotImplementedError


class LocalModel(TargetModel):
    def __init__(self, model_path: str, gpu_id=None):
        from model_utils import load_local_model
        self.model, self.tokenizer, device = load_local_model(model_path, gpu_id)
        self.device = str(device)
        self.torch = __import__('torch')
        logger.info("本地模型加载完成")

    def generate(self, prompt: str) -> Optional[str]:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            _ct_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": ENABLE_THINKING}
            text = self.tokenizer.apply_chat_template(messages, **_ct_kwargs)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if ENABLE_THINKING:
            # Qwen3.5 思考模式易陷入重复循环（官方已知问题）；HF 无 presence_penalty，
            # 用 repetition_penalty(>1) 抑制逐字重复，并用更大的 thinking 预算。
            gen_kwargs = {"max_new_tokens": DETECT_MAX_TOKENS, "do_sample": True,
                          "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                          "repetition_penalty": 1.1,
                          "pad_token_id": self.tokenizer.eos_token_id}
        else:
            gen_kwargs = {"max_new_tokens": TARGET_MAX_TOKENS, "do_sample": False,
                          "pad_token_id": self.tokenizer.eos_token_id}
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response


class APIModel(TargetModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = API_KEY or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 API Key")
        self.client = OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=API_TIMEOUT)
        logger.info(f"API 被测模型: {model_name}")

    def generate(self, prompt: str) -> Optional[str]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name, temperature=TARGET_TEMPERATURE,
                    max_tokens=TARGET_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"API 失败 (第{attempt+1}次): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
        return None


# ============================================================
# Judge 客户端
# ============================================================

def init_judge():
    api_key = API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY")
    return OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=API_TIMEOUT)


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
            # 解析 JSON
            for c in [raw, raw[raw.find("{"):raw.rfind("}")+1] if "{" in raw else ""]:
                if not c: continue
                try:
                    parsed = json.loads(c)
                    match = parsed.get("match", False)
                    if isinstance(match, str):
                        match = match.lower() in ("true", "yes", "1")
                    return bool(match), parsed.get("reason", "")
                except: continue
            return False, f"JSON解析失败: {raw[:100]}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return False, f"API失败: {e}"


# ============================================================
# 解析与评分
# ============================================================

def parse_detection_response(response_text: str) -> Dict:
    """解析模型的检测回答"""
    if not response_text:
        return {"has_idiom": None, "detected_idiom": ""}

    # 尝试 JSON 解析
    for text in [response_text, response_text.replace("```json", "").replace("```", "").strip()]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                has = result.get("has_idiom")
                idiom = result.get("idiom", "")
                if isinstance(has, bool):
                    return {"has_idiom": has, "detected_idiom": idiom.strip() if idiom else ""}
                if isinstance(has, str):
                    return {"has_idiom": has.lower() == "true", "detected_idiom": idiom.strip() if idiom else ""}
        except (json.JSONDecodeError, ValueError):
            continue

    # 兜底: 从文本中推断
    lower = response_text.lower()
    if '"has_idiom": true' in lower or '"has_idiom":true' in lower:
        return {"has_idiom": True, "detected_idiom": ""}
    if '"has_idiom": false' in lower or '"has_idiom":false' in lower:
        return {"has_idiom": False, "detected_idiom": ""}

    return {"has_idiom": None, "detected_idiom": ""}


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


# ============================================================
# 检测流程
# ============================================================

def run_detection(csv_path: str, output_path: str, target: TargetModel,
                  target_model_name: str, src_lang: str, tgt_lang: str,
                  max_samples: int = None):

    df = pd.read_csv(csv_path, encoding="utf-8")
    if max_samples:
        df = df.head(max_samples)
        logger.info(f"--max-samples={max_samples}, 实际处理 {len(df)} 条")
    total = len(df)
    logger.info(f"加载 {csv_path}: {total} 条")

    # 断点
    checkpoint_path = output_path + ".checkpoint"
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

        prompt = DETECTION_PROMPT.format(sentence=row["sentence"])

        try:
            raw_response = target.generate(prompt)
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raw_response = None

        if raw_response is None:
            parsed_response = None
        elif ENABLE_THINKING:
            # 原生思考：需闭合 </think> 才取答案；截断/循环 → 置空，由 parse 判 None
            parsed_response = strip_thinking_output(raw_response) if "</think>" in raw_response else ""
        else:
            parsed_response = raw_response
        parsed = parse_detection_response(parsed_response)

        entry = {
            "id": row_id,
            "idiom": row["idiom"],
            "sentence": row["sentence"],
            "gold_translation": row["gold translation"],
            "has_idiom": parsed["has_idiom"],
            "detected_idiom": parsed["detected_idiom"],
            "raw_response": raw_response,
            "parsed_response": parsed_response
        }

        results.append(entry)
        completed[key] = entry

        if len(results) % 10 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as fp:
                json.dump(completed, fp, ensure_ascii=False)

        time.sleep(REQUEST_INTERVAL)

    # 保存
    output_data = {
        "config": {
            "task": "idiom_detection",
            "target_model": target_model_name,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, ensure_ascii=False, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # 统计
    said_yes = sum(1 for r in results if r["has_idiom"] == True)
    logger.info(f"完成 → {output_path}")
    logger.info(f"  总数: {len(results)}, Model said has_idiom=true: {said_yes}/{len(results)}")


# ============================================================
# Judge-only 模式：对已有结果重新跑 Judge
# ============================================================

def run_judge_only(result_dir: str):
    """对已有 detection JSON 结果重新跑 Judge 评分"""
    judge_client = init_judge()

    # 扫描 result_dir 及其子目录下的 detection JSON
    json_files = []
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            if f.endswith(".json") and "detection" in f.lower() and not f.endswith(".checkpoint"):
                json_files.append(os.path.join(root, f))

    if not json_files:
        logger.error(f"在 {result_dir} 下未找到 detection JSON 文件")
        return

    for fpath in json_files:
        logger.info(f"重新评分: {fpath}")

        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        results = data.get("results", [])
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
        if "config" not in data:
            data["config"] = {}
        data["config"]["judge_model"] = JUDGE_MODEL

        # 覆盖保存
        with open(fpath, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

        # 统计
        valid = [r for r in results if r["detection_score"] is not None]
        det_rate = sum(r["detection_score"] for r in valid) / len(valid) if valid else 0
        said_yes = sum(1 for r in valid if r.get("has_idiom") == True)
        logger.info(f"  完成 → {fpath}")
        logger.info(f"  Model said has_idiom=true: {said_yes}/{len(valid)}")
        logger.info(f"  Detection Score (judge-confirmed): {det_rate:.3f} ({sum(r['detection_score'] for r in valid)}/{len(valid)})")
        logger.info(f"  Judge API calls: {judge_calls}")


# ============================================================
# 多 GPU 支持
# ============================================================

def detection_worker(gpu_id, data_shard, output_path, model_name, src_lang, tgt_lang):
    """Worker for multi-GPU detection — each process loads model on one GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    target = LocalModel(model_name, gpu_id=0)  # Always 0 after CUDA_VISIBLE_DEVICES

    results = []
    total = len(data_shard)

    for i, row in enumerate(data_shard):
        row_id = row["id"]
        logger.info(f"[GPU {gpu_id}] [{i+1}/{total}] {row['idiom']}")

        prompt = DETECTION_PROMPT.format(sentence=row["sentence"])

        try:
            raw_response = target.generate(prompt)
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] 推理失败: {e}")
            raw_response = None

        if raw_response is None:
            parsed_response = None
        elif ENABLE_THINKING:
            # 原生思考：需闭合 </think> 才取答案；截断/循环 → 置空，由 parse 判 None
            parsed_response = strip_thinking_output(raw_response) if "</think>" in raw_response else ""
        else:
            parsed_response = raw_response
        parsed = parse_detection_response(parsed_response)

        entry = {
            "id": row_id,
            "idiom": row["idiom"],
            "sentence": row["sentence"],
            "gold_translation": row["gold_translation"],
            "has_idiom": parsed["has_idiom"],
            "detected_idiom": parsed["detected_idiom"],
            "raw_response": raw_response,
            "parsed_response": parsed_response
        }
        results.append(entry)
        time.sleep(REQUEST_INTERVAL)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"[GPU {gpu_id}] 完成 {len(results)} 条")


def run_multi_gpu_detection(data_rows, output_path, model_name, src_lang, tgt_lang, num_gpus):
    """Split data across GPUs, spawn workers, merge results"""
    import torch.multiprocessing as mp
    from model_utils import split_list

    shards = split_list(data_rows, num_gpus)
    out_dir = os.path.dirname(output_path)
    tasks = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        shard_path = os.path.join(out_dir, f"_tmp_detection_shard{i}.json")
        tasks.append((i, shard, shard_path, model_name, src_lang, tgt_lang))

    logger.info(f"多 GPU 检测: {len(data_rows)} 条, {len(tasks)} GPU 并行, 每 GPU ~{len(data_rows)//len(tasks)} 条")

    mp.set_start_method("spawn", force=True)
    procs = []
    for t in tasks:
        p = mp.Process(target=detection_worker, args=t)
        p.start()
        procs.append((p, t))
    for p, (gpu_id, _, _, _, _, _) in procs:
        p.join()
        if p.exitcode != 0:
            logger.error(f"[错误] GPU {gpu_id} 退出码 {p.exitcode}")

    # Merge shard results
    all_results = []
    for _, _, shard_path, _, _, _ in tasks:
        if os.path.exists(shard_path):
            with open(shard_path, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
            os.remove(shard_path)
    all_results.sort(key=lambda x: x["id"])

    # Build final output
    output_data = {
        "config": {
            "task": "idiom_detection",
            "target_model": model_name,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        },
        "results": all_results
    }
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, ensure_ascii=False, indent=2)

    # Stats
    said_yes = sum(1 for r in all_results if r["has_idiom"] == True)
    logger.info(f"完成 → {output_path}")
    logger.info(f"  总数: {len(all_results)}, Model said has_idiom=true: {said_yes}/{len(all_results)}")


# ============================================================
# vLLM 批量推理
# ============================================================

def run_detection_vllm(csv_path: str, output_path: str, model_name: str,
                       src_lang: str, tgt_lang: str, max_samples: int = None):
    """使用 vLLM 进行批量习语检测推理"""
    from model_utils import load_vllm_model
    from vllm import SamplingParams

    df = pd.read_csv(csv_path, encoding="utf-8")
    total = len(df)
    logger.info(f"加载 {csv_path}: {total} 条")

    # 构建所有数据
    data = []
    for idx, row in df.iterrows():
        data.append({
            "id": idx + 1,
            "idiom": row["idiom"],
            "sentence": row["sentence"],
            "gold_translation": row["gold translation"],
        })

    if max_samples:
        data = data[:max_samples]
        logger.info(f"--max-samples={max_samples}, 实际处理 {len(data)} 条")

    # 构建所有对话
    conversations = []
    for item in data:
        prompt = DETECTION_PROMPT.format(sentence=item["sentence"])
        conversations.append([{"role": "user", "content": prompt}])

    # 加载模型并批量推理
    llm = load_vllm_model(model_name)
    from model_utils import make_sampling_params
    params = make_sampling_params(DETECT_MAX_TOKENS, ENABLE_THINKING)

    logger.info(f"开始 vLLM 批量推理: {total} 条")
    outputs = llm.chat(conversations, params,
                       chat_template_kwargs={"enable_thinking": ENABLE_THINKING})
    logger.info(f"vLLM 批量推理完成")

    # 解析结果
    results = []
    for item, output in zip(data, outputs):
        raw_response = output.outputs[0].text.strip()
        if ENABLE_THINKING and "</think>" not in raw_response:
            parsed_response = ""   # 思考被截断/循环，未产出 JSON → 判失败
        else:
            parsed_response = strip_thinking_output(raw_response)
        parsed = parse_detection_response(parsed_response)

        entry = {
            "id": item["id"],
            "idiom": item["idiom"],
            "sentence": item["sentence"],
            "gold_translation": item["gold_translation"],
            "has_idiom": parsed["has_idiom"],
            "detected_idiom": parsed["detected_idiom"],
            "raw_response": raw_response,
            "parsed_response": parsed_response,
            "finish_reason": getattr(output.outputs[0], "finish_reason", None),
        }
        results.append(entry)

    # 保存
    output_data = {
        "config": {
            "task": "idiom_detection",
            "target_model": model_name,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, ensure_ascii=False, indent=2)

    # 统计
    said_yes = sum(1 for r in results if r["has_idiom"] is True)
    logger.info(f"完成 → {output_path}")
    logger.info(f"  总数: {len(results)}, Model said has_idiom=true: {said_yes}/{len(results)}")


# ============================================================
# main
# ============================================================

def main():
    from config import ALL_TARGETS, get_config
    from model_utils import get_num_gpus

    parser = argparse.ArgumentParser(description="Stage 1: 习语检测")
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS,
                        help="目标，如 en-fa, ja-en")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--local", metavar="MODEL_PATH", help="本地模型路径")
    group.add_argument("--api", metavar="MODEL_NAME", help="API 模型名称")
    group.add_argument("--judge-only", action="store_true",
                       help="跳过模型推理，只对已有结果重跑 Judge 评分")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="GPU 数量（默认自动检测）")
    parser.add_argument("--single", action="store_true",
                        help="强制单 GPU / 单进程运行")
    parser.add_argument("--vllm", action="store_true",
                        help="使用 vLLM 批量推理")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只处理前 N 条数据（测试用）")
    args = parser.parse_args()

    # Determine model name first so get_config can route to model-specific dirs
    if args.judge_only:
        target_model_name = None
    elif args.local:
        target_model_name = args.local
    else:
        target_model_name = args.api

    logger = setup_logger("detection", lang_pair=args.lang_pair, model=target_model_name)

    # vLLM 输出到独立目录
    model_for_config = (target_model_name + "_vllm") if (args.vllm and target_model_name) else target_model_name
    cfg = get_config(args.lang_pair, model=model_for_config)
    out_dir = str(cfg["detection_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # Judge-only 模式
    if args.judge_only:
        run_judge_only(out_dir)
        return

    csv_path = str(cfg["csv_file"])
    output_path = os.path.join(out_dir, "detection_inference.json")

    if args.vllm:
        run_detection_vllm(csv_path, output_path, target_model_name,
                           cfg["src_lang"], cfg["tgt_lang"],
                           max_samples=args.max_samples)
        logger.info("Stage 1 完成!")
        return

    # Decide single vs multi-GPU
    num_gpus = args.num_gpus or get_num_gpus()
    use_multi_gpu = (args.local and not args.single and num_gpus > 1)

    if use_multi_gpu:
        # Load CSV and convert to list of dicts for worker shards
        df = pd.read_csv(csv_path, encoding="utf-8")
        data_rows = []
        for idx, row in df.iterrows():
            data_rows.append({
                "id": idx + 1,
                "idiom": row["idiom"],
                "sentence": row["sentence"],
                "gold_translation": row["gold translation"],
            })
        if args.max_samples:
            data_rows = data_rows[:args.max_samples]
            logger.info(f"--max-samples={args.max_samples}, 实际处理 {len(data_rows)} 条")
        logger.info(f"多 GPU 模式: {num_gpus} GPUs, {len(data_rows)} 条数据")
        run_multi_gpu_detection(data_rows, output_path, target_model_name,
                                cfg["src_lang"], cfg["tgt_lang"], num_gpus)
    else:
        # Single process
        if args.local:
            target = LocalModel(target_model_name)
        else:
            target = APIModel(target_model_name)
        run_detection(csv_path, output_path, target, target_model_name,
                      cfg["src_lang"], cfg["tgt_lang"],
                      max_samples=args.max_samples)

    logger.info("Stage 1 完成!")


if __name__ == "__main__":
    main()
