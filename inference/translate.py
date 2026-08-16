#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate.py — 跨语言习语翻译脚本（统一版）
项目：Cross-Lingual Literal Translation Bias

▌用法
  # 单方向单 prompt
  python translate.py --target en-fa --prompt BasicPrompt

  # 单方向全部 4 个 prompt
  python translate.py --target ja-en --all-prompts

  # 指定 GPU
  CUDA_VISIBLE_DEVICES=0 python translate.py --target fa-en --all-prompts

  # 指定模型
  python translate.py --target en-fa --all-prompts --model Qwen/Qwen3.5-4B
"""

import argparse
import csv
import json
import os
import re
from typing import List, Dict, Tuple
from datetime import datetime
from multiprocessing import Process

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (MODEL_NAME, ENABLE_THINKING, MAX_NEW_TOKENS,
                        MAX_NEW_TOKENS_THINKING_TRANSLATION,
                        PROMPTS, ALL_PROMPT_TYPES, ALL_TARGETS,
                        get_config)
from model_utils import (load_local_model, load_vllm_model, get_num_gpus,
                         split_list, strip_thinking_output)

# thinking 模式下 CoT 会额外消耗大量 token；用更大的预算，否则 </think>
# 之前就被截断，模型还没产出译文就到上限。非 thinking 维持原值。
EFFECTIVE_MAX_TOKENS = MAX_NEW_TOKENS_THINKING_TRANSLATION if ENABLE_THINKING else MAX_NEW_TOKENS


# ==================== 数据加载 ====================

def load_data(cfg: dict) -> Tuple[List[Dict], Dict[str, str]]:
    """从 config 加载数据和 meaning"""
    data_file = str(cfg["csv_file"])
    meaning_file = cfg["meaning_file"]

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            gold = row.get('gold translation', row.get('gold_translation', ''))
            data.append({
                'id': i + 1,
                'idiom': row['idiom'],
                'sentence': row['sentence'],
                'gold_translation': gold,
            })

    meanings = {}
    if meaning_file and os.path.exists(meaning_file):
        with open(meaning_file, 'r', encoding='utf-8') as f:
            mdata = json.load(f)
        for item in mdata.get('results', []):
            meanings[str(item['id'])] = item.get('reference_meaning', '')

    print(f"  数据文件: {data_file}")
    print(f"  数据量: {len(data)} 条")
    print(f"  Meanings: {len(meanings)} 条")
    return data, meanings


# ==================== Prompt 构建（chat template）====================

def build_messages(prompt_type: str, src_lang: str, tgt_lang: str,
                   sentence: str) -> list:
    system_prompt = PROMPTS[prompt_type].replace("[SRC]", src_lang).replace("[TGT]", tgt_lang)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": sentence},
    ]


# ==================== 模型加载 ====================

def load_model(model_name: str, gpu_id=None):
    """统一模型加载，委托给 model_utils.load_local_model"""
    model, tokenizer, device = load_local_model(model_name, gpu_id)
    return model, tokenizer


# ==================== 翻译核心 ====================

def translate_sentence(model, tokenizer, messages,
                       enable_thinking=False, max_new_tokens=512,
                       return_raw=False):
    _ct_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(messages, **_ct_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Qwen3 官方推荐：thinking 模式用采样，非 thinking 保持贪心
    if enable_thinking:
        # Qwen3.5 思考模式易陷入重复循环（官方已知问题）。HF 没有 presence_penalty，
        # 用其类比 repetition_penalty(>1) 抑制逐字重复；取保守 1.1 以减小对译文质量/语言混杂的影响。
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True,
                          temperature=0.6, top_p=0.95, top_k=20,
                          repetition_penalty=1.1,
                          pad_token_id=tokenizer.eos_token_id)
    else:
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][input_len:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if enable_thinking:
        # 原生思考：必须出现闭合的 </think> 才算产出译文。
        # 没有闭合 → 思考被截断/陷入循环，判为空（失败），
        # 不要把推理文本误当成译文返回。
        parsed_output = strip_thinking_output(raw_output) if "</think>" in raw_output else ""
    else:
        parsed_output = raw_output

    if return_raw:
        return parsed_output.strip(), raw_output.strip()
    return parsed_output.strip()


# ==================== 文件命名 ====================

def make_filename(model_name: str, prompt_type: str,
                  src_lang: str, tgt_lang: str, timestamp: str) -> str:
    model_short = model_name.split("/")[-1]
    return f"{model_short}_{prompt_type}_{src_lang}_to_{tgt_lang}_{timestamp}.json"


# ==================== 多 GPU Worker ====================

def translation_worker(gpu_id: int, data_shard: List[Dict],
                       meanings: Dict[str, str], cfg: dict,
                       prompt_types: List[str], model_name: str,
                       output_dir: str):
    """Worker process for multi-GPU translation.

    Each worker handles a subset of the data on a single GPU.
    Results are saved to temporary JSON files per (gpu_id, prompt_type).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model, tokenizer = load_model(model_name, gpu_id=0)

    for pt in prompt_types:
        print(f"  [GPU {gpu_id}] Prompt: {pt}  ({len(data_shard)} items)")
        results = []
        for i, item in enumerate(data_shard):
            messages = build_messages(pt, cfg["src_lang"], cfg["tgt_lang"],
                                      item['sentence'])
            try:
                translation, raw_translation = translate_sentence(
                    model, tokenizer, messages,
                    enable_thinking=ENABLE_THINKING,
                    max_new_tokens=EFFECTIVE_MAX_TOKENS,
                    return_raw=True,
                )
                status = "✓"
            except Exception as e:
                translation = f"ERROR: {str(e)}"
                raw_translation = f"ERROR: {str(e)}"
                status = f"✗ {e}"

            results.append({
                'id': item['id'],
                'idiom': item['idiom'],
                'sentence': item['sentence'],
                'gold_translation': item['gold_translation'],
                'reference_meaning': meanings.get(str(item['id']), ''),
                'model_translation': translation,
                'raw_model_translation': raw_translation,
            })

            print(f"  [GPU {gpu_id}][{i+1}/{len(data_shard)}] {status}  {item['idiom']}")

            if (i + 1) % 50 == 0:
                progress_path = os.path.join(
                    output_dir, f"_progress_{pt}_gpu{gpu_id}.json")
                with open(progress_path, 'w', encoding='utf-8') as f:
                    json.dump({'results': results}, f, ensure_ascii=False, indent=2)
                print(f"    [GPU {gpu_id}] 进度已保存 ({i+1}/{len(data_shard)})")

        tmp_path = os.path.join(
            output_dir, f"_tmp_shard{gpu_id}_{pt}.json")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  [GPU {gpu_id}] {pt}: {len(results)} 条 → {tmp_path}")

        # 清除进度文件
        progress_path = os.path.join(
            output_dir, f"_progress_{pt}_gpu{gpu_id}.json")
        if os.path.exists(progress_path):
            os.remove(progress_path)


# ==================== 主流程 ====================

def run_translation(cfg: dict, prompt_types: List[str], model_name: str,
                    num_gpus: int = 1, max_samples: int = None):
    """对指定目标和 prompt 类型执行翻译

    Args:
        cfg: 配置字典
        prompt_types: 要运行的 prompt 类型列表
        model_name: 模型名称
        num_gpus: GPU 数量。<=1 为单进程，>1 使用 multiprocessing 数据并行
        max_samples: 只处理前 N 条数据（测试用）
    """
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    output_dir = str(cfg["translation_input"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"  跨语言习语翻译实验")
    print(f"{'=' * 62}")
    print(f"  模型: {model_name}")
    print(f"  方向: {src_lang} → {tgt_lang}")
    print(f"  Thinking: {'开启' if ENABLE_THINKING else '关闭'}")
    print(f"  max_new_tokens: {EFFECTIVE_MAX_TOKENS}")
    print(f"  Prompts: {prompt_types}")
    print(f"  GPUs: {num_gpus}")
    print(f"  输出: {output_dir}")
    print(f"{'=' * 62}")

    data, meanings = load_data(cfg)
    if max_samples:
        data = data[:max_samples]
        print(f"  --max-samples={max_samples}, 实际处理 {len(data)} 条")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 单 GPU / CPU 模式 ----
    if num_gpus <= 1:
        model, tokenizer = load_model(model_name)

        for pt in prompt_types:
            print(f"\n{'─' * 62}")
            print(f"  Prompt: {pt}")
            print(f"{'─' * 62}")

            results = []
            for i, item in enumerate(data):
                messages = build_messages(pt, src_lang, tgt_lang,
                                          item['sentence'])

                try:
                    translation, raw_translation = translate_sentence(
                        model, tokenizer, messages,
                        enable_thinking=ENABLE_THINKING,
                        max_new_tokens=EFFECTIVE_MAX_TOKENS,
                        return_raw=True,
                    )
                    status = "✓"
                except Exception as e:
                    translation = f"ERROR: {str(e)}"
                    raw_translation = f"ERROR: {str(e)}"
                    status = f"✗ {e}"

                results.append({
                    'id': item['id'],
                    'idiom': item['idiom'],
                    'sentence': item['sentence'],
                    'gold_translation': item['gold_translation'],
                    'reference_meaning': meanings.get(str(item['id']), ''),
                    'model_translation': translation,
                    'raw_model_translation': raw_translation,
                })

                print(f"  [{i+1}/{len(data)}] {status}  {item['idiom']}")
                print(f"    译文: {translation}")

                if (i + 1) % 50 == 0:
                    progress_path = os.path.join(output_dir,
                                                 f"_progress_{pt}.json")
                    with open(progress_path, 'w', encoding='utf-8') as f:
                        json.dump({'results': results}, f,
                                  ensure_ascii=False, indent=2)
                    print(f"    进度已保存 ({i+1}/{len(data)})")

            filename = make_filename(model_name, pt, src_lang, tgt_lang,
                                     timestamp)
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': {
                        'model': model_name,
                        'prompt_type': pt,
                        'src_lang': src_lang,
                        'tgt_lang': tgt_lang,
                        'enable_thinking': ENABLE_THINKING,
                        'max_new_tokens': EFFECTIVE_MAX_TOKENS,
                        'timestamp': timestamp,
                    },
                    'results': results
                }, f, ensure_ascii=False, indent=2)

            print(f"\n  ✓ {pt}: {len(results)} 条 → {filename}")

            progress_path = os.path.join(output_dir, f"_progress_{pt}.json")
            if os.path.exists(progress_path):
                os.remove(progress_path)

    # ---- 多 GPU 数据并行模式 ----
    else:
        shards = split_list(data, num_gpus)
        print(f"  数据分片: {[len(s) for s in shards]}")

        for pt in prompt_types:
            print(f"\n{'─' * 62}")
            print(f"  Prompt: {pt}  (multi-GPU, {num_gpus} workers)")
            print(f"{'─' * 62}")

            workers = []
            for gid in range(num_gpus):
                if not shards[gid]:
                    continue
                p = Process(
                    target=translation_worker,
                    args=(gid, shards[gid], meanings, cfg,
                          [pt], model_name, output_dir),
                )
                p.start()
                workers.append(p)

            for p in workers:
                p.join()

            # 合并各分片结果
            merged: List[Dict] = []
            for gid in range(num_gpus):
                tmp_path = os.path.join(
                    output_dir, f"_tmp_shard{gid}_{pt}.json")
                if os.path.exists(tmp_path):
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        merged.extend(json.load(f))
                    os.remove(tmp_path)

            # 按 id 排序，保持与单 GPU 输出一致
            merged.sort(key=lambda x: x['id'])

            filename = make_filename(model_name, pt, src_lang, tgt_lang,
                                     timestamp)
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': {
                        'model': model_name,
                        'prompt_type': pt,
                        'src_lang': src_lang,
                        'tgt_lang': tgt_lang,
                        'enable_thinking': ENABLE_THINKING,
                        'max_new_tokens': EFFECTIVE_MAX_TOKENS,
                        'timestamp': timestamp,
                    },
                    'results': merged
                }, f, ensure_ascii=False, indent=2)

            print(f"\n  ✓ {pt}: {len(merged)} 条 → {filename}")

    print(f"\n{'=' * 62}")
    print(f"  全部完成！")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 62}")


# ==================== vLLM 批量推理 ====================

def run_translation_vllm(cfg: dict, prompt_types: List[str], model_name: str,
                         max_samples: int = None):
    """使用 vLLM 进行批量翻译推理

    Args:
        cfg: 配置字典
        prompt_types: 要运行的 prompt 类型列表
        model_name: 模型名称
        max_samples: 只处理前 N 条数据（测试用）
    """
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    output_dir = str(cfg["translation_input"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 62}")
    print(f"  跨语言习语翻译实验 (vLLM 批量推理)")
    print(f"{'=' * 62}")
    print(f"  模型: {model_name}")
    print(f"  方向: {src_lang} → {tgt_lang}")
    print(f"  max_new_tokens: {EFFECTIVE_MAX_TOKENS}")
    print(f"  Prompts: {prompt_types}")
    print(f"  输出: {output_dir}")
    print(f"{'=' * 62}")

    data, meanings = load_data(cfg)
    if max_samples:
        data = data[:max_samples]
        print(f"  --max-samples={max_samples}, 实际处理 {len(data)} 条")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 加载 vLLM 模型（一次加载，多个 prompt 复用）
    llm = load_vllm_model(model_name)

    from model_utils import make_sampling_params
    params = make_sampling_params(EFFECTIVE_MAX_TOKENS, ENABLE_THINKING)

    for pt in prompt_types:
        print(f"\n{'─' * 62}")
        print(f"  Prompt: {pt}  (vLLM batch)")
        print(f"{'─' * 62}")

        # 构建所有对话
        conversations = []
        for item in data:
            messages = build_messages(pt, src_lang, tgt_lang, item['sentence'])
            conversations.append(messages)

        print(f"  批量推理 {len(conversations)} 条...")
        outputs = llm.chat(conversations, params,
                           chat_template_kwargs={"enable_thinking": ENABLE_THINKING})

        # 构建结果
        results = []
        for i, (item, output) in enumerate(zip(data, outputs)):
            raw_translation = output.outputs[0].text.strip()
            if ENABLE_THINKING and "</think>" not in raw_translation:
                # 思考被截断/循环，未产出译文 → 判为空（失败）
                translation = ""
            else:
                translation = strip_thinking_output(raw_translation)

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

        filename = make_filename(model_name, pt, src_lang, tgt_lang, timestamp)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'model': model_name,
                    'prompt_type': pt,
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang,
                    'enable_thinking': ENABLE_THINKING,
                    'max_new_tokens': EFFECTIVE_MAX_TOKENS,
                    'timestamp': timestamp,
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n  ✓ {pt}: {len(results)} 条 → {filename}")

    print(f"\n{'=' * 62}")
    print(f"  全部完成！")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 62}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="跨语言习语翻译（统一版）")
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS,
                        help="翻译目标，如 en-fa, ja-en")
    parser.add_argument("--prompt", choices=ALL_PROMPT_TYPES, default=None,
                        help="单个 prompt 类型")
    parser.add_argument("--all-prompts", action="store_true",
                        help="跑全部 4 个 prompt")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"模型名称（默认 {MODEL_NAME}）")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="使用的 GPU 数量（默认 1，>1 启用数据并行）")
    parser.add_argument("--single", action="store_true",
                        help="强制单 GPU 运行（忽略 --num-gpus）")
    parser.add_argument("--vllm", action="store_true",
                        help="使用 vLLM 批量推理")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只处理前 N 条数据（测试用）")
    args = parser.parse_args()

    if not args.prompt and not args.all_prompts:
        parser.error("请指定 --prompt 或 --all-prompts")

    model_for_config = args.model + ("_vllm" if args.vllm else "")
    cfg = get_config(args.lang_pair, model=model_for_config)
    prompt_types = ALL_PROMPT_TYPES if args.all_prompts else [args.prompt]

    if args.vllm:
        run_translation_vllm(cfg, prompt_types, args.model,
                             max_samples=args.max_samples)
    else:
        num_gpus = 1 if args.single else args.num_gpus
        run_translation(cfg, prompt_types, args.model, num_gpus=num_gpus,
                        max_samples=args.max_samples)


if __name__ == "__main__":
    main()
