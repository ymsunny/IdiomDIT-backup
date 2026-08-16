#!/usr/bin/env python3
"""
run_all_inference.py — 统一推理入口（模型只加载 1 次）

对每个模型：加载 → 跑完所有语言对的 detection/interpretation/translation → 卸载

用法：
  python run_all_inference.py --lang-pairs fi-en ko-en --model Qwen3.5-4B
  python run_all_inference.py --lang-pairs fi-en --model Qwen3.5-4B --max-samples 10
  python run_all_inference.py --lang-pairs fi-en --model Qwen3.5-4B --all-prompts
  python run_all_inference.py --lang-pairs fi-en --model Qwen3.5-4B --prompt BasicPrompt
"""

import sys
import os
import csv
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

from config import (get_config, ALL_TARGETS, ALL_PROMPT_TYPES, MODEL_NAME,
                    ENABLE_THINKING, MAX_NEW_TOKENS, PROMPTS, setup_logger,
                    is_thinking_model,
                    MAX_NEW_TOKENS_THINKING_DETECTION,
                    MAX_NEW_TOKENS_THINKING_INTERP,
                    MAX_NEW_TOKENS_THINKING_TRANSLATION)
from model_utils import load_local_model, load_vllm_model, make_sampling_params

# 从推理脚本导入核心函数（不触发模型加载）
sys.path.insert(0, str(_BASE / "inference"))
from detection import (DETECTION_PROMPT, parse_detection_response,
                       TARGET_MAX_TOKENS, TARGET_TEMPERATURE)
from Interpretation import (PROBE_PROMPTS, probe_single, P1_KEYS, P2_KEYS, NUM_RUNS)
from translate import (build_messages, translate_sentence, make_filename, load_data)
from model_utils import strip_thinking_output


def _vllm_chat(llm, conversations, params, enable_thinking):
    """Call vLLM chat with Qwen chat-template kwargs when the installed vLLM supports it."""
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    try:
        return llm.chat(conversations, params, **chat_kwargs)
    except TypeError:
        print("  [vLLM] chat_template_kwargs unsupported by this vLLM version; using model/template defaults.")
        return llm.chat(conversations, params)


def _vllm_text(output):
    return output.outputs[0].text.strip()


# ==================== Detection ====================

def run_detection_with_model(model, tokenizer, device, cfg, model_name,
                             max_samples=None, logger=None,
                             max_new_tokens=TARGET_MAX_TOKENS):
    """用已加载的模型跑 detection"""
    import torch
    import pandas as pd

    det_dir = str(cfg["detection_dir"])
    os.makedirs(det_dir, exist_ok=True)
    output_path = os.path.join(det_dir, "detection_inference.json")

    if os.path.exists(output_path):
        if logger: logger.info(f"  跳过 detection（已存在）")
        return

    csv_path = str(cfg["csv_file"])
    df = pd.read_csv(csv_path, encoding="utf-8")
    if max_samples:
        df = df.head(max_samples)

    if logger: logger.info(f"  Detection: {len(df)} 条")

    results = []
    for idx, row in df.iterrows():
        row_id = idx + 1
        prompt = DETECTION_PROMPT.format(sentence=row["sentence"])

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        if ENABLE_THINKING:
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True,
                          "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                          "pad_token_id": tokenizer.eos_token_id}
        else:
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False,
                          "pad_token_id": tokenizer.eos_token_id}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed_response = strip_thinking_output(raw_response) if ENABLE_THINKING else raw_response

        parsed = parse_detection_response(parsed_response)
        results.append({
            "id": row_id,
            "idiom": row["idiom"],
            "sentence": row["sentence"],
            "gold_translation": row["gold translation"],
            "has_idiom": parsed["has_idiom"],
            "detected_idiom": parsed["detected_idiom"],
            "raw_response": raw_response,
            "parsed_response": parsed_response,
        })

        if (row_id) % 5 == 0 and logger:
            logger.info(f"    [{row_id}/{len(df)}]")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"task": "idiom_detection", "target_model": model_name,
                       "src_lang": cfg["src_lang"], "tgt_lang": cfg["tgt_lang"]},
            "results": results
        }, f, ensure_ascii=False, indent=2)

    has_idiom_count = sum(1 for r in results if r.get("has_idiom") == True)
    if logger:
        logger.info(f"  Detection 完成: {len(results)} 条, has_idiom={has_idiom_count} → {output_path}")


def run_detection_with_vllm(llm, cfg, model_name, max_samples=None, logger=None,
                            max_new_tokens=TARGET_MAX_TOKENS, enable_thinking=False):
    """Run detection through vLLM batch chat."""
    import pandas as pd

    det_dir = str(cfg["detection_dir"])
    os.makedirs(det_dir, exist_ok=True)
    output_path = os.path.join(det_dir, "detection_inference.json")

    if os.path.exists(output_path):
        if logger: logger.info("  Skip detection[vLLM] (exists)")
        return

    df = pd.read_csv(str(cfg["csv_file"]), encoding="utf-8")
    if max_samples:
        df = df.head(max_samples)

    if logger: logger.info(f"  Detection[vLLM]: {len(df)} items")

    conversations = []
    rows = []
    for idx, row in df.iterrows():
        rows.append((idx + 1, row))
        conversations.append([{"role": "user",
                               "content": DETECTION_PROMPT.format(sentence=row["sentence"])}])

    params = make_sampling_params(max_new_tokens, enable_thinking)
    outputs = _vllm_chat(llm, conversations, params, enable_thinking)

    results = []
    for (row_id, row), output in zip(rows, outputs):
        raw_response = _vllm_text(output)
        parsed_response = strip_thinking_output(raw_response) if enable_thinking else raw_response
        parsed = parse_detection_response(parsed_response)
        results.append({
            "id": row_id,
            "idiom": row["idiom"],
            "sentence": row["sentence"],
            "gold_translation": row["gold translation"],
            "has_idiom": parsed["has_idiom"],
            "detected_idiom": parsed["detected_idiom"],
            "raw_response": raw_response,
            "parsed_response": parsed_response,
            "finish_reason": getattr(output.outputs[0], "finish_reason", None),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"task": "idiom_detection", "target_model": model_name,
                       "src_lang": cfg["src_lang"], "tgt_lang": cfg["tgt_lang"],
                       "backend": "vllm", "enable_thinking": enable_thinking,
                       "max_new_tokens": max_new_tokens},
            "results": results
        }, f, ensure_ascii=False, indent=2)

    has_idiom_count = sum(1 for r in results if r.get("has_idiom") == True)
    if logger:
        logger.info(f"  Detection[vLLM] done: {len(results)} items, has_idiom={has_idiom_count} -> {output_path}")


# ==================== Interpretation ====================

def run_interpretation_with_model(model, tokenizer, device, cfg, model_name,
                                  max_samples=None, logger=None, max_new_tokens=256):
    """用已加载的模型跑 interpretation"""
    interp_dir = str(cfg["interpretation_dir"])
    os.makedirs(interp_dir, exist_ok=True)
    output_path = os.path.join(interp_dir, "interpretation_inference.json")

    if os.path.exists(output_path):
        if logger: logger.info(f"  跳过 interpretation（已存在）")
        return

    csv_path = str(cfg["csv_file"])
    meaning_file = cfg["meaning_file"]

    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            data.append({'id': i + 1, 'idiom': row['idiom'], 'sentence': row['sentence']})

    meanings = {}
    if meaning_file and os.path.exists(meaning_file):
        with open(meaning_file, 'r', encoding='utf-8') as f:
            mdata = json.load(f)
        for item in mdata.get('results', []):
            meanings[str(item['id'])] = item.get('reference_meaning', '')

    if max_samples:
        data = data[:max_samples]

    if logger: logger.info(f"  Interpretation: {len(data)} 条 × {len(PROBE_PROMPTS)} prompts")

    results = []
    for i, item in enumerate(data):
        item_result = {
            'id': item['id'], 'idiom': item['idiom'],
            'sentence': item['sentence'],
            'reference_meaning': meanings.get(str(item['id']), ''),
        }
        for pk in PROBE_PROMPTS:
            messages = [{"role": "user",
                        "content": PROBE_PROMPTS[pk].format(
                            idiom=item['idiom'], sentence=item['sentence'])}]
            responses = []
            raw_responses = []
            for _ in range(NUM_RUNS):
                try:
                    parsed_response, raw_response = probe_single(
                        model, tokenizer, messages,
                        enable_thinking=ENABLE_THINKING,
                        max_new_tokens=max_new_tokens,
                        return_raw=True)
                    responses.append(parsed_response)
                    raw_responses.append(raw_response)
                except Exception as e:
                    responses.append(f"ERROR: {e}")
                    raw_responses.append(f"ERROR: {e}")
            item_result[f"{pk}__responses"] = responses
            item_result[f"{pk}__raw_responses"] = raw_responses
        results.append(item_result)

        if (i + 1) % 5 == 0 and logger:
            logger.info(f"    [{i+1}/{len(data)}]")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if logger: logger.info(f"  Interpretation 完成: {len(results)} 条 × {len(PROBE_PROMPTS)} prompts → {output_path}")


def run_interpretation_with_vllm(llm, cfg, model_name, max_samples=None,
                                 logger=None, max_new_tokens=256,
                                 enable_thinking=False):
    """Run interpretation probes through vLLM batch chat."""
    interp_dir = str(cfg["interpretation_dir"])
    os.makedirs(interp_dir, exist_ok=True)
    output_path = os.path.join(interp_dir, "interpretation_inference.json")

    if os.path.exists(output_path):
        if logger: logger.info("  Skip interpretation[vLLM] (exists)")
        return

    data = []
    with open(str(cfg["csv_file"]), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            data.append({'id': i + 1, 'idiom': row['idiom'], 'sentence': row['sentence']})

    meanings = {}
    meaning_file = cfg["meaning_file"]
    if meaning_file and os.path.exists(meaning_file):
        with open(meaning_file, 'r', encoding='utf-8') as f:
            mdata = json.load(f)
        for item in mdata.get('results', []):
            meanings[str(item['id'])] = item.get('reference_meaning', '')

    if max_samples:
        data = data[:max_samples]

    if logger: logger.info(f"  Interpretation[vLLM]: {len(data)} items x {len(PROBE_PROMPTS)} prompts")

    conversations = []
    metas = []
    for item_idx, item in enumerate(data):
        for pk in PROBE_PROMPTS:
            messages = [{"role": "user",
                         "content": PROBE_PROMPTS[pk].format(
                             idiom=item['idiom'], sentence=item['sentence'])}]
            for _ in range(NUM_RUNS):
                conversations.append(messages)
                metas.append((item_idx, pk))

    params = make_sampling_params(max_new_tokens, enable_thinking)
    outputs = _vllm_chat(llm, conversations, params, enable_thinking)

    results = []
    for item in data:
        item_result = {
            'id': item['id'], 'idiom': item['idiom'],
            'sentence': item['sentence'],
            'reference_meaning': meanings.get(str(item['id']), ''),
        }
        for pk in PROBE_PROMPTS:
            item_result[f"{pk}__responses"] = []
            item_result[f"{pk}__raw_responses"] = []
            item_result[f"{pk}__finish_reasons"] = []
        results.append(item_result)

    for (item_idx, pk), output in zip(metas, outputs):
        raw_text = _vllm_text(output)
        parsed_text = strip_thinking_output(raw_text) if enable_thinking else raw_text
        results[item_idx][f"{pk}__responses"].append(parsed_text)
        results[item_idx][f"{pk}__raw_responses"].append(raw_text)
        results[item_idx][f"{pk}__finish_reasons"].append(
            getattr(output.outputs[0], "finish_reason", None))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if logger:
        logger.info(f"  Interpretation[vLLM] done: {len(results)} items x {len(PROBE_PROMPTS)} prompts -> {output_path}")


# ==================== Translation ====================

def run_translation_with_model(model, tokenizer, device, cfg, model_name,
                               prompt_types, max_samples=None, logger=None,
                               max_new_tokens=MAX_NEW_TOKENS):
    """用已加载的模型跑 translation"""
    output_dir = str(cfg["translation_input"])
    os.makedirs(output_dir, exist_ok=True)

    data, meanings = load_data(cfg)
    if max_samples:
        data = data[:max_samples]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    for pt in prompt_types:
        filename = make_filename(model_name, pt, src_lang, tgt_lang, timestamp)
        filepath = os.path.join(output_dir, filename)

        # 检查是否已有该 prompt 的结果
        existing = [f for f in os.listdir(output_dir)
                    if pt in f and not any(x in f for x in ["_progress", "_tmp_"])]
        if existing:
            if logger: logger.info(f"  跳过 {pt}（已存在）")
            continue

        if logger: logger.info(f"  Translation [{pt}]: {len(data)} 条")

        results = []
        for i, item in enumerate(data):
            messages = build_messages(pt, src_lang, tgt_lang, item['sentence'])
            try:
                translation, raw_translation = translate_sentence(
                    model, tokenizer, messages,
                    enable_thinking=ENABLE_THINKING,
                    max_new_tokens=max_new_tokens,
                    return_raw=True)
            except Exception as e:
                translation = f"ERROR: {str(e)}"
                raw_translation = f"ERROR: {str(e)}"

            results.append({
                'id': item['id'],
                'idiom': item['idiom'],
                'sentence': item['sentence'],
                'gold_translation': item['gold_translation'],
                'reference_meaning': meanings.get(str(item['id']), ''),
                'model_translation': translation,
                'raw_model_translation': raw_translation,
            })

            if (i + 1) % 5 == 0 and logger:
                logger.info(f"    [{i+1}/{len(data)}]")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'model': model_name, 'prompt_type': pt,
                    'src_lang': src_lang, 'tgt_lang': tgt_lang,
                    'enable_thinking': ENABLE_THINKING,
                    'max_new_tokens': max_new_tokens,
                    'timestamp': timestamp,
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)

        if logger: logger.info(f"  Translation [{pt}] 完成: {len(results)} 条 → {filepath}")


def run_translation_with_vllm(llm, cfg, model_name, prompt_types,
                              max_samples=None, logger=None,
                              max_new_tokens=MAX_NEW_TOKENS,
                              enable_thinking=False):
    """Run translation through vLLM batch chat."""
    output_dir = str(cfg["translation_input"])
    os.makedirs(output_dir, exist_ok=True)

    data, meanings = load_data(cfg)
    if max_samples:
        data = data[:max_samples]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    params = make_sampling_params(max_new_tokens, enable_thinking)

    for pt in prompt_types:
        filename = make_filename(model_name, pt, src_lang, tgt_lang, timestamp)
        filepath = os.path.join(output_dir, filename)

        existing = [f for f in os.listdir(output_dir)
                    if pt in f and not any(x in f for x in ["_progress", "_tmp_"])]
        if existing:
            if logger: logger.info(f"  Skip translation[vLLM] {pt} (exists)")
            continue

        if logger: logger.info(f"  Translation[vLLM] [{pt}]: {len(data)} items")

        conversations = [
            build_messages(pt, src_lang, tgt_lang, item['sentence'])
            for item in data
        ]
        outputs = _vllm_chat(llm, conversations, params, enable_thinking)

        results = []
        for item, output in zip(data, outputs):
            raw_translation = _vllm_text(output)
            translation = strip_thinking_output(raw_translation) if enable_thinking else raw_translation
            results.append({
                'id': item['id'],
                'idiom': item['idiom'],
                'sentence': item['sentence'],
                'gold_translation': item['gold_translation'],
                'reference_meaning': meanings.get(str(item['id']), ''),
                'model_translation': translation,
                'raw_model_translation': raw_translation,
                'finish_reason': getattr(output.outputs[0], "finish_reason", None),
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'model': model_name, 'prompt_type': pt,
                    'src_lang': src_lang, 'tgt_lang': tgt_lang,
                    'enable_thinking': enable_thinking,
                    'max_new_tokens': max_new_tokens,
                    'backend': 'vllm',
                    'timestamp': timestamp,
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)

        if logger:
            logger.info(f"  Translation[vLLM] [{pt}] done: {len(results)} items -> {filepath}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="统一推理（模型只加载 1 次）")
    parser.add_argument("--lang-pairs", nargs="+", required=True, choices=ALL_TARGETS,
                        help="语言对列表")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"模型名称（默认 {MODEL_NAME}）")
    parser.add_argument("--prompt", choices=ALL_PROMPT_TYPES, default=None)
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只处理前 N 条（测试用）")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3],
                        help="要跑的步骤：1=detection 2=interpretation 3=translation")
    parser.add_argument("--thinking", action="store_true",
                        help="启用 thinking 模式；输出目录自动加 -thinking 后缀以隔离结果")
    parser.add_argument("--vllm", action="store_true",
                        help="使用 vLLM 批量推理后端；输出目录自动加 _vllm 后缀")
    parser.add_argument("--vllm-max-model-len", type=int, default=16384,
                        help="vLLM max_model_len（默认 16384）")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None,
                        help="vLLM tensor_parallel_size，例如多卡单模型时设为 2/4")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None,
                        help="多 GPU 并行：自动将语言对分配到指定 GPU（如 --gpu-ids 0 1 2 3）")
    args = parser.parse_args()

    if 3 in args.steps and not args.prompt and not args.all_prompts:
        parser.error("翻译阶段需要指定 --prompt 或 --all-prompts")

    # ---- 多 GPU 并行模式：拆分语言对，各 GPU 独立子进程 ----
    if args.gpu_ids and len(args.gpu_ids) > 1:
        import subprocess
        gpu_ids = args.gpu_ids
        lps = args.lang_pairs
        # round-robin 分配语言对
        chunks = [[] for _ in gpu_ids]
        for i, lp in enumerate(lps):
            chunks[i % len(gpu_ids)].append(lp)

        base_cmd = [sys.executable, __file__]
        # 透传除 --gpu-ids 以外的所有参数
        passthrough = ["--model", args.model,
                       "--steps"] + [str(s) for s in args.steps]
        if args.thinking:    passthrough.append("--thinking")
        if args.vllm:
            passthrough += ["--vllm",
                            "--vllm-max-model-len", str(args.vllm_max_model_len)]
            if args.vllm_tensor_parallel_size:
                passthrough += ["--vllm-tensor-parallel-size", str(args.vllm_tensor_parallel_size)]
        if args.all_prompts: passthrough.append("--all-prompts")
        elif args.prompt:    passthrough += ["--prompt", args.prompt]
        if args.max_samples: passthrough += ["--max-samples", str(args.max_samples)]

        procs = []
        for gpu_id, chunk in zip(gpu_ids, chunks):
            if not chunk:
                continue
            cmd = base_cmd + ["--lang-pairs"] + chunk + passthrough
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"  GPU {gpu_id} → {chunk}")
            procs.append(subprocess.Popen(cmd, env=env))

        for p in procs:
            p.wait()
        print(f"\n  全部 GPU 子进程完成. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return

    prompt_types = ALL_PROMPT_TYPES if args.all_prompts else ([args.prompt] if args.prompt else [])

    # --thinking / --vllm 只影响输出目录名；实际加载原始权重。
    effective_model = args.model
    if args.thinking:
        effective_model += "-thinking"
    if args.vllm:
        effective_model += "_vllm"

    # 加载模型（只加载 1 次）
    print("=" * 62)
    print(f"  统一推理 — 模型加载 1 次，跑完所有任务")
    print(f"  模型: {args.model}  (输出目录: {effective_model})")
    print(f"  Backend: {'vLLM' if args.vllm else 'Transformers'}")
    print(f"  Thinking: {'开启' if args.thinking else '关闭'}"
          + (f"  (det={MAX_NEW_TOKENS_THINKING_DETECTION} / interp={MAX_NEW_TOKENS_THINKING_INTERP} / trans={MAX_NEW_TOKENS_THINKING_TRANSLATION} tokens)" if args.thinking else ""))
    print(f"  语言对: {args.lang_pairs}")
    print(f"  步骤: {args.steps}")
    if args.max_samples:
        print(f"  测试模式: 前 {args.max_samples} 条")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # 动态设置各子模块的 ENABLE_THINKING
    import inference.detection as _det_mod
    import inference.Interpretation as _interp_mod
    import inference.translate as _trans_mod
    thinking_on = args.thinking and is_thinking_model(args.model)
    # 更新本模块的全局变量（run_detection/interpretation/translation_with_model 直接引用它）
    global ENABLE_THINKING
    ENABLE_THINKING = thinking_on
    _det_mod.ENABLE_THINKING = thinking_on
    _interp_mod.ENABLE_THINKING = thinking_on
    _trans_mod.ENABLE_THINKING = thinking_on
    # thinking 模式下 CoT 大幅消耗 token，需扩大上限
    if thinking_on:
        _det_mod.TARGET_MAX_TOKENS = MAX_NEW_TOKENS_THINKING_DETECTION
    det_max    = MAX_NEW_TOKENS_THINKING_DETECTION   if thinking_on else _det_mod.TARGET_MAX_TOKENS
    interp_max = MAX_NEW_TOKENS_THINKING_INTERP      if thinking_on else 256
    trans_max  = MAX_NEW_TOKENS_THINKING_TRANSLATION if thinking_on else MAX_NEW_TOKENS

    if args.vllm:
        llm = load_vllm_model(
            args.model,
            max_model_len=args.vllm_max_model_len,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
        )
        model = tokenizer = device = None
    else:
        llm = None
        model, tokenizer, device = load_local_model(args.model)

    for lp in args.lang_pairs:
        cfg = get_config(lp, model=effective_model)

        print(f"\n>>> {lp}")

        if 1 in args.steps:
            det_logger = setup_logger(f"det_{lp}_{effective_model}", lang_pair=lp,
                                      model=effective_model, log_filename="detection.log")
            if args.vllm:
                run_detection_with_vllm(llm, cfg, effective_model,
                                        max_samples=args.max_samples, logger=det_logger,
                                        max_new_tokens=det_max,
                                        enable_thinking=thinking_on)
            else:
                run_detection_with_model(model, tokenizer, device, cfg, effective_model,
                                         max_samples=args.max_samples, logger=det_logger,
                                         max_new_tokens=det_max)

        if 2 in args.steps:
            interp_logger = setup_logger(f"interp_{lp}_{effective_model}", lang_pair=lp,
                                          model=effective_model, log_filename="interpretation.log")
            if args.vllm:
                run_interpretation_with_vllm(llm, cfg, effective_model,
                                             max_samples=args.max_samples,
                                             logger=interp_logger,
                                             max_new_tokens=interp_max,
                                             enable_thinking=thinking_on)
            else:
                run_interpretation_with_model(model, tokenizer, device, cfg, effective_model,
                                              max_samples=args.max_samples, logger=interp_logger,
                                              max_new_tokens=interp_max)

        if 3 in args.steps:
            trans_logger = setup_logger(f"trans_{lp}_{effective_model}", lang_pair=lp,
                                         model=effective_model, log_filename="translation.log")
            if args.vllm:
                run_translation_with_vllm(llm, cfg, effective_model,
                                          prompt_types, max_samples=args.max_samples,
                                          logger=trans_logger,
                                          max_new_tokens=trans_max,
                                          enable_thinking=thinking_on)
            else:
                run_translation_with_model(model, tokenizer, device, cfg, effective_model,
                                           prompt_types, max_samples=args.max_samples,
                                           logger=trans_logger, max_new_tokens=trans_max)

    print(f"\n{'=' * 62}")
    print(f"  全部完成! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
