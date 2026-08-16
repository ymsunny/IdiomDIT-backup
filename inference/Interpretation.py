#!/usr/bin/env python3
"""
Interpretation.py — 习语释义探测（多次探测 + 对照组）

▌实验设计
  P1（主实验，无上下文）× 3 种问法 + P2（有上下文，对照组）× 3 种问法 = 6 prompts
  每个 temperature=0 跑 1 次，任一命中 → I=1

▌输出（推理阶段）
  results/{target}/{model}/interpretation/
    └── interpretation_inference.json  ← 模型原始探测响应（评估由 eval_interpretation.py 完成）

▌用法
  python Interpretation.py --target en-fa       # 测英文习语（En→Fa 方向）
  python Interpretation.py --target ja-en       # 测日文习语
  python Interpretation.py --target en-fa --single  # 单 GPU 调试
"""

import csv, json, os, re, sys, argparse, math
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import time, logging, warnings

warnings.filterwarnings("ignore")

# ==================== 配置 ====================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (MODEL_NAME, ENABLE_THINKING,
                        MAX_NEW_TOKENS_THINKING_INTERP,
                        API_BASE_URL, API_KEY, JUDGE_MODEL,
                        API_TIMEOUT, API_TEMPERATURE,
                        API_MAX_RETRIES, API_RETRY_DELAY,
                        ALL_TARGETS, get_config, setup_logger)
from model_utils import strip_thinking_output

NUM_RUNS = 1  # 每个 prompt 跑 1 次（temperature=0 结果固定）
# thinking 模式下 CoT 在 </think> 前会消耗大量 token，256 会被截断；
# 非 thinking 维持 256（释义 JSON 很短，够用）。
INTERP_MAX_TOKENS = MAX_NEW_TOKENS_THINKING_INTERP if ENABLE_THINKING else 256
JUDGE_TEMPERATURE = API_TEMPERATURE
JUDGE_MAX_TOKENS  = 256
MAX_RETRIES  = API_MAX_RETRIES
RETRY_DELAY  = API_RETRY_DELAY

logger = logging.getLogger("interpretation")

# ==================== Probe Prompts ====================

FINAL_JSON_INSTRUCTION = (
    " If you reason internally, keep it brief and put the final response after exactly this prefix: "
    "Final Answer: After that prefix, output only one JSON object: {{\"meaning\": \"...\"}}"
)

PROBE_PROMPTS = {
    "P1a": "What is the idiomatic meaning of \"{idiom}\"?" + FINAL_JSON_INSTRUCTION,
    "P1b": "Explain the idiomatic meaning of \"{idiom}\"." + FINAL_JSON_INSTRUCTION,
    "P1c": "What does \"{idiom}\" idiomatically mean?" + FINAL_JSON_INSTRUCTION,
    "P2a": "What is the idiomatic meaning of \"{idiom}\" in: \"{sentence}\"?" + FINAL_JSON_INSTRUCTION,
    "P2b": "Explain the idiomatic meaning of \"{idiom}\" in: \"{sentence}\"." + FINAL_JSON_INSTRUCTION,
    "P2c": "What does \"{idiom}\" idiomatically mean in: \"{sentence}\"?" + FINAL_JSON_INSTRUCTION,
}
P1_KEYS = ["P1a", "P1b", "P1c"]
P2_KEYS = ["P2a", "P2b", "P2c"]

# ==================== Judge Prompt ====================

JUDGE_SYSTEM = """You are a judge evaluating whether a model's explanation of an idiom is correct.
Output ONLY a JSON object: {"correct": true/false, "reason": "brief reason"}
No markdown, no explanation outside JSON."""

JUDGE_USER = """Reference meaning: {reference_meaning}
Model's answer: {model_answer}

Does the model's answer correctly capture the essential meaning of the idiom?
- Be lenient: the answer doesn't need to be word-perfect, just semantically correct
- If the model says "I don't know" or gives a wrong meaning, mark as incorrect
- Output JSON only: {{"correct": true/false, "reason": "..."}}"""

# ==================== 数据加载 ====================

def load_data(cfg: dict) -> Tuple[List[Dict], Dict[str, str]]:
    data_file = str(cfg["csv_file"])
    meaning_file = cfg["meaning_file"]

    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            data.append({'id': i + 1, 'idiom': row['idiom'], 'sentence': row['sentence']})
    print(f"  数据: {len(data)} 条 ({data_file})")

    meanings = {}
    if meaning_file and os.path.exists(meaning_file):
        with open(meaning_file, 'r', encoding='utf-8') as f:
            mdata = json.load(f)
        for item in mdata.get('results', []):
            meanings[str(item['id'])] = item.get('reference_meaning', '')
        print(f"  Meanings: {len(meanings)} 条")
    else:
        print(f"  Meanings 不存在: {meaning_file}")

    return data, meanings

# ==================== 模型推理 ====================

def build_probe_messages(prompt_key: str, idiom: str, sentence: str) -> list:
    template = PROBE_PROMPTS[prompt_key]
    content = template.format(idiom=idiom, sentence=sentence)
    return [{"role": "user", "content": content}]

def probe_single(model, tokenizer, messages, enable_thinking=False, max_new_tokens=256,
                 return_raw=False):
    import torch
    _ct_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(messages, **_ct_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    if enable_thinking:
        # Qwen3.5 思考模式易陷入重复循环（官方已知问题）。HF 无 presence_penalty，
        # 用 repetition_penalty(>1) 抑制逐字重复，取保守 1.1。
        _gen = dict(max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.6, top_p=0.95, top_k=20,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id)
    else:
        _gen = dict(max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        outputs = model.generate(**inputs, **_gen)
    generated_ids = outputs[0][input_len:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if enable_thinking:
        # 原生思考：必须出现闭合的 </think> 才算产出答案。
        # 没有闭合 → 思考被截断/陷入循环，判为空（失败），
        # 不要把整段推理误当成释义返回。
        parsed = strip_thinking_output(raw) if "</think>" in raw else ""
    else:
        parsed = raw
    if return_raw:
        return parsed.strip(), raw.strip()
    return parsed.strip()

# ==================== Judge ====================

def init_judge():
    from openai import OpenAI
    key = API_KEY or os.environ.get("OPENAI_API_KEY")
    if not key: raise ValueError("请设置 OPENAI_API_KEY")
    return OpenAI(base_url=API_BASE_URL, api_key=key, timeout=API_TIMEOUT)

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
                if not c: continue
                try:
                    parsed = json.loads(c)
                    correct = parsed.get("correct", False)
                    if isinstance(correct, str):
                        correct = correct.lower() in ("true", "yes", "1")
                    return bool(correct), parsed.get("reason", "")
                except: continue
            return False, f"JSON解析失败: {raw[:100]}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return False, f"API失败: {e}"

# ==================== Worker ====================

try:
    import torch
except ImportError:
    pass

def worker_fn(gpu_id: int, data_shard: List[Dict], meanings: Dict[str, str],
              output_path: str, model_name: str = None):
    from model_utils import load_local_model

    _model = model_name or MODEL_NAME
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] 加载模型...")
    model, tokenizer, device = load_local_model(_model, gpu_id=0)

    print(f"[GPU {gpu_id}] 开始探测 ({len(data_shard)} 条 × {len(PROBE_PROMPTS)} prompts × {NUM_RUNS} runs)...")
    results = []
    for i, item in enumerate(data_shard):
        item_result = {
            'id': item['id'], 'idiom': item['idiom'],
            'sentence': item['sentence'],
            'reference_meaning': meanings.get(str(item['id']), ''),
        }
        for prompt_key in PROBE_PROMPTS:
            messages = build_probe_messages(prompt_key, item['idiom'], item['sentence'])
            responses = []
            for _ in range(NUM_RUNS):
                try:
                    responses.append(probe_single(model, tokenizer, messages,
                                                   enable_thinking=ENABLE_THINKING,
                                                   max_new_tokens=INTERP_MAX_TOKENS))
                except Exception as e:
                    responses.append(f"ERROR: {e}")
            item_result[f"{prompt_key}__responses"] = responses
        results.append(item_result)
        if (i + 1) % 10 == 0 or (i + 1) == len(data_shard):
            print(f"[GPU {gpu_id}] {i+1}/{len(data_shard)} 完成")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] ✓ 保存 {len(results)} 条")

# ==================== Judge 阶段 ====================

def run_judge(results: List[Dict], meanings: Dict[str, str]) -> List[Dict]:
    client = init_judge()
    total_calls = len(results) * len(PROBE_PROMPTS) * NUM_RUNS
    print(f"\nJudge 评估: {total_calls} 次 API 调用...")

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
                try:
                    parsed = json.loads(re.sub(r"```json|```", "", answer).strip())
                    answer = parsed.get("meaning", answer)
                except Exception:
                    pass
                correct, reason = judge_answer(client, meaning, answer)
                judgments.append({"correct": correct, "reason": reason})
                if correct: any_correct = True
                time.sleep(0.3)
            item[f"{prompt_key}__judgments"] = judgments
            item[f"{prompt_key}__any_correct"] = any_correct
            item[f"{prompt_key}__hit_count"] = sum(1 for j in judgments if j["correct"])

        # P1/P2 各自独立判定
        item["I_no_context"] = any(item.get(f"{pk}__any_correct", False) for pk in P1_KEYS)
        item["I_with_context"] = any(item.get(f"{pk}__any_correct", False) for pk in P2_KEYS)
        # P1 的 interpretation_score（主实验，供下游级联分析使用）
        item["interpretation_score_P1"] = 1 if item["I_no_context"] else 0
        # P2 的 interpretation_score（对照组）
        item["interpretation_score_P2"] = 1 if item["I_with_context"] else 0

        if (i + 1) % 10 == 0:
            print(f"  Judge: {i+1}/{len(results)} 完成")

    return results

# ==================== 并行调度 ====================

from model_utils import get_num_gpus, split_list

def run_model_inference(data, meanings, num_gpus, output_dir, model_name=None):
    import torch.multiprocessing as mp
    _model = model_name or MODEL_NAME
    shards = split_list(data, num_gpus)
    tasks = []
    for i, shard in enumerate(shards):
        if not shard: continue
        tasks.append((i, shard, meanings, os.path.join(output_dir, f"_tmp_shard{i}.json"), _model))

    print(f"\n模型推理: {len(data)} 条 × {len(PROBE_PROMPTS)} prompts × {NUM_RUNS} runs")
    print(f"  {len(tasks)} GPU 并行, 每 GPU ~{len(data)//len(tasks)} 条")

    mp.set_start_method("spawn", force=True)
    procs = []
    for t in tasks:
        p = mp.Process(target=worker_fn, args=t); p.start(); procs.append((p, t))
    for p, (gpu_id, _, _, _, _) in procs:
        p.join()
        if p.exitcode != 0: print(f"[错误] GPU {gpu_id} 退出码 {p.exitcode}")

    all_results = []
    for _, _, _, path, _ in tasks:
        if os.path.exists(path):
            with open(path) as f: all_results.extend(json.load(f))
            os.remove(path)
    all_results.sort(key=lambda x: x['id'])
    return all_results

def run_serial_inference(data, meanings, output_dir, model_name=None):
    from model_utils import load_local_model
    model, tokenizer, device = load_local_model(model_name or MODEL_NAME)

    results = []
    for i, item in enumerate(data):
        item_result = {'id': item['id'], 'idiom': item['idiom'],
                       'sentence': item['sentence'],
                       'reference_meaning': meanings.get(str(item['id']), '')}
        for pk in PROBE_PROMPTS:
            messages = build_probe_messages(pk, item['idiom'], item['sentence'])
            responses = []
            for _ in range(NUM_RUNS):
                try: responses.append(probe_single(model, tokenizer, messages, ENABLE_THINKING,
                                                    max_new_tokens=INTERP_MAX_TOKENS))
                except Exception as e: responses.append(f"ERROR: {e}")
            item_result[f"{pk}__responses"] = responses
        results.append(item_result)
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(data)}")
    return results

# ==================== 报告（全部 JSON 输出）====================

def write_report(results, output_dir, source):
    # === 按 P1/P2 分别构建结果列表 ===
    p1_results = []
    p2_results = []
    for item in results:
        base = {
            "id": item["id"], "idiom": item["idiom"],
            "sentence": item.get("sentence", ""),
            "reference_meaning": item.get("reference_meaning", ""),
        }
        # P1 条目
        p1_entry = {**base, "interpretation_score": item.get("interpretation_score_P1", 0)}
        for pk in P1_KEYS:
            p1_entry[f"{pk}_correct"] = int(item.get(f"{pk}__any_correct", False))
            p1_entry[f"{pk}_responses"] = item.get(f"{pk}__responses", [])
            p1_entry[f"{pk}_judgments"] = item.get(f"{pk}__judgments", [])
        p1_results.append(p1_entry)

        # P2 条目
        p2_entry = {**base, "interpretation_score": item.get("interpretation_score_P2", 0)}
        for pk in P2_KEYS:
            p2_entry[f"{pk}_correct"] = int(item.get(f"{pk}__any_correct", False))
            p2_entry[f"{pk}_responses"] = item.get(f"{pk}__responses", [])
            p2_entry[f"{pk}_judgments"] = item.get(f"{pk}__judgments", [])
        p2_results.append(p2_entry)

    # === P1 输出：interpretation_P1_detail.json（主实验，下游读取此文件）===
    p1_path = os.path.join(output_dir, "interpretation_P1_detail.json")
    with open(p1_path, 'w', encoding='utf-8') as f:
        json.dump({"source": source, "model": MODEL_NAME, "num_runs": NUM_RUNS,
                   "group": "P1_no_context", "prompts": P1_KEYS,
                   "results": p1_results}, f, ensure_ascii=False, indent=2)

    # === P2 输出：interpretation_P2_detail.json（对照组）===
    p2_path = os.path.join(output_dir, "interpretation_P2_detail.json")
    with open(p2_path, 'w', encoding='utf-8') as f:
        json.dump({"source": source, "model": MODEL_NAME, "num_runs": NUM_RUNS,
                   "group": "P2_with_context", "prompts": P2_KEYS,
                   "results": p2_results}, f, ensure_ascii=False, indent=2)

    # === 可读报告 ===
    n = len(results)
    i_p1 = sum(1 for x in p1_results if x["interpretation_score"] == 1)
    i_p2 = sum(1 for x in p2_results if x["interpretation_score"] == 1)

    rp = os.path.join(output_dir, "interpretation_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        def w(s=""): f.write(s + "\n")
        S = "=" * 74
        w(S)
        w(f"Interpretation Probe 报告 — source={source}")
        w(f"模型: {MODEL_NAME}  |  P1×3（主实验）+ P2×3（对照组）")
        w(f"习语数: {n}")
        w(S); w()

        w("【总体结果】")
        w(f"  P1 (无上下文, 主实验):   {i_p1/n*100:.1f}%  ({i_p1}/{n})")
        w(f"  P2 (有上下文, 对照组):   {i_p2/n*100:.1f}%  ({i_p2}/{n})")
        w()

        w("【各 Prompt 命中率】")
        w(f"  {'Prompt':<45} {'命中率':>8} {'命中数':>8}")
        w("-" * 65)
        for pk in P1_KEYS:
            hit = sum(1 for x in p1_results if x[f"{pk}_correct"] == 1)
            w(f"  {pk} (无上下文){'':<30} {hit/n*100:>7.1f}% ({hit}/{n})")
        for pk in P2_KEYS:
            hit = sum(1 for x in p2_results if x[f"{pk}_correct"] == 1)
            w(f"  {pk} (有上下文){'':<30} {hit/n*100:>7.1f}% ({hit}/{n})")
        w()

        w("【P1 vs P2 对比分析】")
        p1_set = {x["id"] for x in p1_results if x["interpretation_score"]==1}
        p2_set = {x["id"] for x in p2_results if x["interpretation_score"]==1}
        all_ids = {x["id"] for x in p1_results}
        both    = len(p1_set & p2_set)
        only_p1 = len(p1_set - p2_set)
        only_p2 = len(p2_set - p1_set)
        neither = len(all_ids - p1_set - p2_set)
        w(f"  P1+P2 都命中:   {both:>4} ({both/n*100:.1f}%)")
        w(f"  仅 P1 命中:     {only_p1:>4} ({only_p1/n*100:.1f}%)")
        w(f"  仅 P2 命中:     {only_p2:>4} ({only_p2/n*100:.1f}%)")
        w(f"  都未命中:       {neither:>4} ({neither/n*100:.1f}%)")
        w()

        no_interp = [x for x in p1_results if x["interpretation_score"] == 0]
        if no_interp:
            w(f"【P1 未命中的习语（前 20 个）】")
            for x in no_interp[:20]:
                w(f"  id={x['id']}  {x['idiom']}")
                w(f"    含义: {str(x['reference_meaning'])[:60]}")

    print(f"输出:")
    print(f"  P1（主实验）: {p1_path}")
    print(f"  P2（对照组）: {p2_path}")
    print(f"  报告: {rp}")

    # === 可视化 ===
    plot_interpretation(p1_results, p2_results, output_dir, source)


def plot_interpretation(p1_results, p2_results, output_dir, source):
    """生成 Interpretation 可视化图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(p1_results)

    # ---------- 图1: P1 vs P2 各 prompt 命中率对比 ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    all_keys = P1_KEYS + P2_KEYS
    labels = [f"{pk}\n(no ctx)" if pk in P1_KEYS else f"{pk}\n(with ctx)" for pk in all_keys]
    hits = []
    for pk in P1_KEYS:
        hits.append(sum(1 for x in p1_results if x[f"{pk}_correct"] == 1) / n * 100)
    for pk in P2_KEYS:
        hits.append(sum(1 for x in p2_results if x[f"{pk}_correct"] == 1) / n * 100)

    colors = ["#4C72B0"] * len(P1_KEYS) + ["#DD8452"] * len(P2_KEYS)
    bars = ax.bar(range(len(all_keys)), hits, color=colors, edgecolor="white", linewidth=0.8)
    for bar, h in zip(bars, hits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title(f"Interpretation Probe — Per-Prompt Hit Rate (source={source})")
    ax.legend(handles=[plt.Rectangle((0,0),1,1, fc="#4C72B0"), plt.Rectangle((0,0),1,1, fc="#DD8452")],
              labels=["P1 (no context)", "P2 (with context)"], loc="upper right")
    ax.set_ylim(0, max(hits) * 1.2 if hits else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "interpretation_prompt_hit_rate.png"), dpi=150)
    plt.close(fig)

    # ---------- 图2: P1 vs P2 总体对比 + 四格交叉 ----------
    i_p1 = sum(1 for x in p1_results if x["interpretation_score"] == 1)
    i_p2 = sum(1 for x in p2_results if x["interpretation_score"] == 1)
    p1_set = {x["id"] for x in p1_results if x["interpretation_score"] == 1}
    p2_set = {x["id"] for x in p2_results if x["interpretation_score"] == 1}
    all_ids = {x["id"] for x in p1_results}
    both    = len(p1_set & p2_set)
    only_p1 = len(p1_set - p2_set)
    only_p2 = len(p2_set - p1_set)
    neither = len(all_ids - p1_set - p2_set)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：P1 vs P2 总体
    ax = axes[0]
    bars = ax.bar(["P1\n(no context)", "P2\n(with context)"],
                  [i_p1/n*100, i_p2/n*100],
                  color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Interpretation Rate (%)")
    ax.set_title("P1 vs P2 Overall")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 右图：四格交叉分布
    ax = axes[1]
    categories = ["Both P1+P2", "Only P1", "Only P2", "Neither"]
    values = [both, only_p1, only_p2, neither]
    pcts = [v/n*100 for v in values]
    colors_pie = ["#55A868", "#4C72B0", "#DD8452", "#C44E52"]
    wedges, texts, autotexts = ax.pie(
        values, labels=categories, colors=colors_pie, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9})
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    ax.set_title("P1 vs P2 Cross Distribution")

    plt.suptitle(f"Interpretation Probe — source={source}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "interpretation_p1_vs_p2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  图表: interpretation_prompt_hit_rate.png, interpretation_p1_vs_p2.png")

# ==================== vLLM 批量推理 ====================

def run_vllm_inference(data, meanings, output_dir, model_name=None):
    from model_utils import load_vllm_model
    from vllm import SamplingParams

    _model = model_name or MODEL_NAME
    llm = load_vllm_model(_model)

    # Build ALL prompts at once: N items x 6 probe prompts = 6N conversations
    conversations = []
    prompt_keys = []
    item_indices = []

    for idx, item in enumerate(data):
        for pk in PROBE_PROMPTS:
            template = PROBE_PROMPTS[pk]
            content = template.format(idiom=item['idiom'], sentence=item['sentence'])
            conversations.append([{"role": "user", "content": content}])
            prompt_keys.append(pk)
            item_indices.append(idx)

    print(f"\nvLLM 批量推理: {len(data)} 条 × {len(PROBE_PROMPTS)} prompts = {len(conversations)} 请求")

    from model_utils import make_sampling_params
    params = make_sampling_params(INTERP_MAX_TOKENS, ENABLE_THINKING)
    outputs = llm.chat(conversations, params,
                       chat_template_kwargs={"enable_thinking": ENABLE_THINKING})

    # Reassemble results: map each output back to its item and prompt_key
    results = [None] * len(data)
    for i, item in enumerate(data):
        results[i] = {
            'id': item['id'],
            'idiom': item['idiom'],
            'sentence': item['sentence'],
            'reference_meaning': meanings.get(str(item['id']), ''),
        }
        for pk in PROBE_PROMPTS:
            results[i][f"{pk}__responses"] = []

    for out_idx, output in enumerate(outputs):
        idx = item_indices[out_idx]
        pk = prompt_keys[out_idx]
        text = output.outputs[0].text.strip()
        if ENABLE_THINKING and "</think>" not in text:
            # 思考被截断/循环，未产出答案 → 判为空（失败）
            text = ""
        else:
            text = strip_thinking_output(text)
        results[idx][f"{pk}__responses"].append(text)

    print(f"vLLM 推理完成: {len(results)} 条结果")
    return results


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="Interpretation Probe")
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS,
                        help="目标，如 en-fa, ja-en")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"模型名称（默认 {MODEL_NAME}）")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--vllm", action="store_true", help="使用 vLLM 批量推理")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只处理前 N 条数据（测试用）")
    args = parser.parse_args()

    model_name = args.model
    logger = setup_logger("interpretation", lang_pair=args.lang_pair, model=model_name)

    model_for_config = model_name + ("_vllm" if args.vllm else "")
    cfg = get_config(args.lang_pair, model=model_for_config)
    output_dir = str(cfg["interpretation_dir"])
    os.makedirs(output_dir, exist_ok=True)

    source_label = args.lang_pair

    print("=" * 62)
    print(f"  Interpretation Probe — target={args.lang_pair}")
    print(f"  模型: {model_name}  |  {NUM_RUNS} runs/prompt")
    print("=" * 62)

    data, meanings = load_data(cfg)
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"  --max-samples={args.max_samples}, 实际处理 {len(data)} 条")

    if args.vllm:
        results = run_vllm_inference(data, meanings, output_dir, model_name=model_name)
    else:
        num_gpus = args.num_gpus or get_num_gpus()
        print(f"\n  GPU: {num_gpus}")
        if args.single or num_gpus <= 1:
            results = run_serial_inference(data, meanings, output_dir, model_name=model_name)
        else:
            results = run_model_inference(data, meanings, num_gpus, output_dir, model_name=model_name)

    output_path = os.path.join(output_dir, "interpretation_inference.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"推理结果已保存: {output_path}")

    print(f"\n完成！输出: {output_dir}")


if __name__ == "__main__":
    main()
