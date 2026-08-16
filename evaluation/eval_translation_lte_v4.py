"""
eval_translation_lte_v4.py — LTE 评估 (v4 judge prompt)

与 eval_translation_lte_old.py 的区别：v4 prompt 显式要求先逐词抽取 idiom 的字面翻译，
再给两个独立子判断 (literal_check / meaning_check)，LTE 由子判断程序化派生（模型不直接给
LTE 字段）。人工对照显示 v4 比 old prompt 校准更好（fi-en Cohen κ 0.44→0.62，LTE 率贴合人工）。

新增（相对 LLM-LTB 原版）：
  --judge-model   覆盖评委模型（默认 config.JUDGE_MODEL；与论文一致用 gpt-4o-mini）
  --output-prefix 输出前缀（默认 translation_lte_v4，避免覆盖 old 的 translation_lte_score.json）

用法：
  python evaluation/eval_translation_lte_v4.py --lang-pair fi-en --model Qwen3.5-9B --no-confirm \
      --judge-model gpt-4o-mini --output-prefix translation_lte_v4
  python evaluation/eval_translation_lte_v4.py --lang-pair fi-en --model Qwen3.5-9B --no-confirm \
      --judge-model gpt-5.2 --output-prefix translation_lte_v4_gpt52
"""

import os, sys, json, time, logging, warnings
from pathlib import Path
from openai import OpenAI
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (get_target_arg, get_config,
                        API_BASE_URL, API_KEY, JUDGE_MODEL,
                        API_TIMEOUT, API_TEMPERATURE, API_MAX_TOKENS,
                        API_MAX_RETRIES, API_RETRY_DELAY, API_REQUEST_INTERVAL,
                        setup_logger)

warnings.filterwarnings("ignore")

PRIMARY_JUDGE_MODEL = JUDGE_MODEL
TEMPERATURE  = API_TEMPERATURE
MAX_TOKENS   = API_MAX_TOKENS
MAX_RETRIES  = API_MAX_RETRIES
RETRY_DELAY  = API_RETRY_DELAY
REQUEST_INTERVAL = API_REQUEST_INTERVAL

logger = logging.getLogger("eval_translation")

# ============================================================
# Prompt (v4)
# ============================================================

SYSTEM_PROMPT = """You are an expert in cross-lingual idiom translation.

# Global Constraints
⚠️ Output Format: Valid JSON object ONLY.
- Do NOT wrap in ```json or ```
- Do NOT add explanations before/after JSON
"""

USER_PROMPT_TEMPLATE = """# Task
Evaluate two properties of the translation regarding the idiom.

# Important
- The two judgments must be made independently.
- Both properties can be true or false independently.

# Subjudgments (answer each independently and exact evidence from the translation)

Important distinction:
Literal mapping: the literal translation of words in the idiom (e.g., in the idiom "drop the ball", "ball" is translated literally as "球")
Intended meaning: the actual figurative meaning of the idiom (e.g., in the idiom "drop the ball", it is translated figuratively as "犯错")

Step 1: Extract the literal translation of each content word in the idiom.
Step 2: Literal Check - Does the translation preserve this literal mapping? (Yes / No)
Step 3: Meaning Check - Does the translation correctly express the intended meaning? (Yes / No)

# Input
- Idiom: "{idiom}"
- Intended meaning of the idiom: "{meaning}"
- Source Sentence: "{sentence}"
- Translation: "{model_translation}"

# Output Format (JSON only)
{{
  "idiom": "{idiom}",
  "intended_meaning": "{meaning}",
  "source_sentence": "{sentence}",
  "translation": "{model_translation}",
  "reasoning": {{
    "explicit_literal_mapping": "Extract the literal translation of each content word in the idiom",
    "literal_check": "Yes/No + 10-20 word evidence",
    "meaning_check": "Yes/No + 10-20 word evidence",
    "confidence": 0.0-1.0
  }}
}}"""

# ============================================================
# API
# ============================================================

def init_client():
    key = API_KEY
    if not key: raise ValueError("请设置 OPENAI_API_KEY")
    return OpenAI(base_url=API_BASE_URL, api_key=key, timeout=API_TIMEOUT)

def call_llm(client, system_prompt, user_prompt, model):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}])
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            return content.strip()
        except Exception as e:
            logger.warning(f"API 失败 ({attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY * (attempt+1))
    return None

def parse_json(text):
    if not text: return None
    for c in [text, text[text.find("{"):text.rfind("}")+1] if "{" in text else "",
              text.replace("```json","").replace("```","").strip()]:
        if not c: continue
        try: return json.loads(c)
        except: continue

    # 兜底：正则提取关键字段（应对源文本中引号导致 JSON 解析失败的情况）
    import re
    lte_match = re.search(r'"literal_translation_error"\s*:\s*(true|false|null)', text, re.IGNORECASE)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    lit_match = re.search(r'"literal_check"\s*:\s*"([^"]*)"', text)
    mean_match = re.search(r'"meaning_check"\s*:\s*"([^"]*)"', text)
    final_match = re.search(r'"final_judgment"\s*:\s*"([^"]*)"', text)

    if lit_match or mean_match or lte_match:
        lte = None
        if lte_match:
            v = lte_match.group(1).lower()
            lte = True if v == 'true' else False if v == 'false' else None
        return {
            "literal_translation_error": lte,
            "confidence": float(conf_match.group(1)) if conf_match else None,
            "reasoning": {
                "literal_check": lit_match.group(1) if lit_match else "",
                "meaning_check": mean_match.group(1) if mean_match else "",
                "final_judgment": final_match.group(1) if final_match else "",
            }
        }

    logger.warning(f"JSON 解析失败: {text[:300]}")
    return None

# ============================================================
# 数据加载
# ============================================================

def load_meanings(cfg):
    path = cfg["meaning_file"]
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        meanings = {str(it['id']): it.get('reference_meaning','') for it in data.get('results',[])}
        logger.info(f"Meanings: {len(meanings)} 条 ({path})")
        return meanings
    else:
        logger.warning(f"Meanings 不存在: {path}")
        return {}

def load_json_files(input_dir, direction, meanings, prompt_filter=None):
    """加载指定方向的翻译 JSON。prompt_filter 不为空时只保留该 prompt_type（用于分片并行）。"""
    all_rows = []
    for f in sorted(Path(input_dir).glob("*.json")):
        if any(x in f.name for x in ["_progress","_tmp_","_checkpoint"]): continue
        try:
            with open(f, "r", encoding="utf-8") as fp: data = json.load(fp)
        except: continue
        c = data.get("config", {})
        if not c.get("prompt_type"): continue
        file_direction = f"{c.get('src_lang','?')}_to_{c.get('tgt_lang','?')}"
        if file_direction != direction: continue
        results = data.get("results", [])
        if not results: continue
        logger.info(f"加载: {f.name} ({len(results)} 条)")
        for item in results:
            iid = str(item.get("id",""))
            all_rows.append({
                "file_name": f.name, "model": c.get("model","unknown"),
                "prompt_type": c.get("prompt_type","unknown"),
                "direction": file_direction,
                "system_id": f"{c.get('prompt_type','?')}__{file_direction}",
                "id": item.get("id"), "idiom": item.get("idiom",""),
                "sentence": item.get("sentence",""),
                "gold_translation": item.get("gold_translation",""),
                "reference_meaning": item.get("reference_meaning","") or meanings.get(iid,""),
                "model_translation": item.get("model_translation",""),
            })
    if prompt_filter:
        all_rows = [r for r in all_rows if r["prompt_type"] == prompt_filter]
        logger.info(f"  prompt 过滤 = {prompt_filter}: 保留 {len(all_rows)} 条")
    if not all_rows: raise ValueError(f"没有 {direction} 方向的数据 (目录: {input_dir}, prompt={prompt_filter})")
    logger.info(f"加载 {len(all_rows)} 条 ({direction})")
    return all_rows

def _load_single_json(path, direction, meanings):
    """评估单个文件（如 ablation 的 _for_eval_*.json）。保留 source_prompt_type 供配对。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    c = data.get("config", {})
    rows = []
    for item in data.get("results", []):
        iid = str(item.get("id", ""))
        rows.append({
            "file_name": os.path.basename(path), "model": c.get("model", "unknown"),
            "prompt_type": item.get("source_prompt_type") or c.get("prompt_type", "unknown"),
            "direction": direction,
            "source_prompt_type": item.get("source_prompt_type", ""),
            "system_id": item.get("source_prompt_type", ""),
            "id": item.get("id"), "idiom": item.get("idiom", ""),
            "sentence": item.get("sentence", ""),
            "gold_translation": item.get("gold_translation", ""),
            "reference_meaning": item.get("reference_meaning", "") or meanings.get(iid, ""),
            "model_translation": item.get("model_translation", ""),
        })
    logger.info(f"加载单文件: {os.path.basename(path)} ({len(rows)} 条)")
    return rows

# ============================================================
# 评估
# ============================================================

def evaluate_single(client, row, model):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        idiom=row["idiom"], meaning=row.get("reference_meaning") or "N/A",
        sentence=row["sentence"], model_translation=row["model_translation"])
    raw = call_llm(client, SYSTEM_PROMPT, user_prompt, model)
    parsed = parse_json(raw)
    if not isinstance(parsed, dict):
        return {"literal_translation_error": None, "confidence": None,
                "literal_check": "", "meaning_check": "",
                "final_judgment": "评估失败", "raw_response": raw}
    reasoning = parsed.get("reasoning", {})
    if not isinstance(reasoning, dict):
        reasoning = {}

    def _as_text(v):
        # 判官偶尔把 literal_check/meaning_check 返回成 dict/非字符串，统一转字符串，避免 .strip() 崩溃
        if isinstance(v, str):
            return v
        if v is None:
            return ""
        return json.dumps(v, ensure_ascii=False)

    lit_check = _as_text(reasoning.get("literal_check", ""))
    mean_check = _as_text(reasoning.get("meaning_check", ""))
    # derive literal_translation_error from sub-judgments (v4 prompt has no top-level field)
    error = parsed.get("literal_translation_error")
    if isinstance(error, str): error = error.lower() in ("true","yes","1")
    elif not isinstance(error, bool):
        lc = lit_check.strip().lower()
        mc = mean_check.strip().lower()
        if lc.startswith("yes") and mc.startswith("no"):
            error = True
        elif lc.startswith(("yes", "no")) and mc.startswith(("yes", "no")):
            error = False
        else:
            error = None
    conf = reasoning.get("confidence") or parsed.get("confidence")
    try: conf = float(conf) if conf is not None else None
    except: conf = None
    return {"literal_translation_error": error, "confidence": conf,
            "literal_check": lit_check,
            "meaning_check": mean_check,
            "final_judgment": reasoning.get("final_judgment",""),
            "raw_response": raw}

FAIL_RESULT = {"literal_translation_error": None, "confidence": None,
               "literal_check": "", "meaning_check": "",
               "final_judgment": "评估失败", "raw_response": None}

# ============================================================
# 主流程
# ============================================================

def run_evaluation(cfg, max_samples: int = None, output_dir: str = None,
                   judge_model: str = None, output_prefix: str = None,
                   translation_input: str = None, prompt_type: str = None):
    client = init_client()
    input_dir  = str(cfg["translation_input"])
    output_dir = output_dir if output_dir else str(cfg["eval_output"])
    direction  = cfg["direction"]
    os.makedirs(output_dir, exist_ok=True)
    judge = judge_model or PRIMARY_JUDGE_MODEL
    prefix = output_prefix or "translation_lte_v4"
    logger.info(f"评委模型: {judge} | prompt: v4 | 前缀: {prefix}")

    meanings = load_meanings(cfg)
    if translation_input:
        data_rows = _load_single_json(translation_input, direction, meanings)
    else:
        data_rows = load_json_files(input_dir, direction, meanings, prompt_filter=prompt_type)
    if max_samples:
        data_rows = data_rows[:max_samples]
        logger.info(f"--max-samples={max_samples}, 实际处理 {len(data_rows)} 条")
    total = len(data_rows)

    ckpt_path = os.path.join(output_dir, f"_{prefix}_checkpoint.json")
    completed = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f: completed = json.load(f)
        logger.info(f"断点恢复: {len(completed)} 条")

    processed = 0
    for row in data_rows:
        key = f"{row['file_name']}__{row['id']}"
        if key in completed: continue
        logger.info(f"[{processed+1}/{total}] id={row['id']} | {str(row['idiom'])[:40]}...")
        result = evaluate_single(client, row, judge)
        completed[key] = result if result else FAIL_RESULT.copy()
        processed += 1
        time.sleep(REQUEST_INTERVAL)
        if processed % 10 == 0:
            with open(ckpt_path, "w", encoding="utf-8") as f: json.dump(completed, f, ensure_ascii=False)

    with open(ckpt_path, "w", encoding="utf-8") as f: json.dump(completed, f, ensure_ascii=False)

    # === score.json ===
    detail_results = []
    for row in data_rows:
        key = f"{row['file_name']}__{row['id']}"
        r = completed.get(key, {})
        detail_results.append({
            "file_name": row["file_name"], "model": row["model"],
            "prompt_type": row["prompt_type"], "direction": row["direction"],
            "source_prompt_type": row.get("source_prompt_type", ""),
            "system_id": row["system_id"],
            "id": row["id"], "idiom": row["idiom"],
            "sentence": row["sentence"],
            "gold_translation": row["gold_translation"],
            "reference_meaning": row["reference_meaning"],
            "model_translation": row["model_translation"],
            "literal_translation_error": r.get("literal_translation_error"),
            "confidence": r.get("confidence"),
            "literal_check": r.get("literal_check",""),
            "meaning_check": r.get("meaning_check",""),
            "final_judgment": r.get("final_judgment",""),
            "raw_response": r.get("raw_response",""),
        })

    detail_path = os.path.join(output_dir, f"{prefix}_score.json")
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump({"direction": direction, "total": total,
                   "results": detail_results}, f, ensure_ascii=False, indent=2)

    # === summary.json ===
    valid = [r for r in detail_results if r["literal_translation_error"] is not None]
    prompt_groups = {}
    for r in valid:
        pt = r["prompt_type"]
        if pt not in prompt_groups: prompt_groups[pt] = []
        prompt_groups[pt].append(r)

    summary = []
    for pt, items in sorted(prompt_groups.items()):
        errs = sum(1 for x in items if x["literal_translation_error"] == True)
        confs = [x["confidence"] for x in items if x["confidence"] is not None]
        summary.append({
            "prompt_type": pt, "count": len(items),
            "error_count": errs,
            "error_rate": errs / len(items) if items else 0,
            "avg_confidence": sum(confs) / len(confs) if confs else None,
        })

    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    failed = [r for r in detail_results if r["literal_translation_error"] is None]
    logger.info(f"  总计: {total} 条 | 有效: {len(valid)} | 失败: {len(failed)}")
    if failed:
        logger.error(f"  ⚠ {len(failed)} 条评估失败（API 错误或解析失败）")

    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    return detail_results


if __name__ == "__main__":
    args = get_target_arg(extra_args=[
        (["--model"], {"default": None, "help": "模型名称，如 Qwen3.5-9B"}),
        (["--no-confirm"], {"action": "store_true", "help": "跳过确认"}),
        (["--max-samples"], {"type": int, "default": None, "help": "只处理前 N 条（测试用）"}),
        (["--output-dir"], {"default": None, "help": "覆盖输出目录"}),
        (["--judge-model"], {"default": None,
            "help": "覆盖评委模型（默认 config.JUDGE_MODEL）；与论文一致用 gpt-4o-mini"}),
        (["--output-prefix"], {"default": None,
            "help": "输出前缀，默认 translation_lte_v4（避免覆盖 old 的 translation_lte_score.json）"}),
        (["--translation-input"], {"default": None,
            "help": "评估单个 JSON 文件（如 ablation 的 _for_eval_*.json）；不提供则扫 translation 目录"}),
        (["--prompt-type"], {"default": None,
            "help": "只评某个 prompt_type（如 BasicPrompt）；用于按 prompt 分片并行。配合不同 --output-prefix"}),
    ])
    logger = setup_logger("eval_translation", lang_pair=args.lang_pair, model=args.model)
    cfg = get_config(args.lang_pair, model=args.model)
    eff_output = args.output_dir or cfg["eval_output"]
    print(f"方向: {cfg['direction']}  judge: {args.judge_model or PRIMARY_JUDGE_MODEL}  "
          f"prefix: {args.output_prefix or 'translation_lte_v4'}  输出: {eff_output}")
    if not args.no_confirm:
        print("按 Enter 开始...")
        try: input()
        except: exit(0)
    run_evaluation(cfg, max_samples=args.max_samples, output_dir=args.output_dir,
                   judge_model=args.judge_model, output_prefix=args.output_prefix,
                   translation_input=args.translation_input, prompt_type=args.prompt_type)
