"""
score_compositionality.py — 1-5 anchored compositionality scoring for idioms.

审稿人 concern #3 的回应实验之二。文献链条:
  Nunberg, Sag & Wasow 1994(可分解性三分类)
  → Titone & Connine 1994(连续 Likert + 多 annotator norming)
  → Nordmann et al. 2014(报告成语评分 inter-rater reliability 低 + 母语/非母语差异;
    未建议锚定——锚例是我们为应对该 reliability 问题自行加的)
  → Kim et al. 2025 MIDAS(LLM prompt-based rating: idiom+meaning+定义,1-5 分,无锚例;
    anchored 模式是我们在此基础上按 Nordmann 2014 的锚定建议做的扩展)
  → Our study(GPT-5.2, 6 languages)

组合性(compositionality)= 透明度(transparency)= 可分解性(decomposability),
在 idiom-norming 传统里是同一 construct 的不同称呼。

Prompt 模式:
  - zero-shot: 只给定义 + scale,不给示例
  - anchored:  给定义 + 每个源语言 3 个锚定示例(score 1, 3, 5)
  两种模式都跑,报告 Spearman(anchored, zero-shot)作为 prompt 鲁棒性对照。

数据规模(unique idioms):
  en-fa=200, fa-en=200, fi-en=85, fr-en=138, ja-en=1188, ko-en=422
  合计 2233 × 2 modes = 4466 次 API 调用

用法:
  python analysis/score_compositionality.py --pilot                   # 20 条 ja-en × 2 mode
  python analysis/score_compositionality.py --lang-pair ja-en         # 全 ja-en × 2 mode
  python analysis/score_compositionality.py --lang-pair all --mode anchored
  python analysis/score_compositionality.py --judge-model gpt-5.2     # 默认 gpt-5.2

输出:
  analysis/output/compositionality/{direction}_{judge}_{mode}.json
"""
import os
import sys
import json
import time
import re
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    API_BASE_URL, API_KEY, API_TIMEOUT,
    API_MAX_RETRIES as MAX_RETRIES,
    API_RETRY_DELAY as RETRY_DELAY,
    API_REQUEST_INTERVAL as REQUEST_INTERVAL,
    LANG_PAIRS, get_data_paths,
)

_BASE = Path(__file__).resolve().parent.parent
OUT_DIR = _BASE / "analysis" / "output" / "compositionality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE = 0
MAX_TOKENS = 256

os.makedirs(_BASE / "logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(_BASE / "logs" / "score_compositionality.log"),
                            encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 锚定示例(每源语言 3 个:score 1, 3, 5)
# 均不在评估语料中(和 LLM-Literal-Translation-Bias 版本一致,已验证)
# ------------------------------------------------------------------

ANCHOR_EXAMPLES = {
    "en": [
        (1, "fly off the handle", "to suddenly become very angry",
         '"fly", "off", "handle" have no connection to "anger"'),
        (3, "break the ice", "to relieve tension in a social situation",
         '"break" suggests disrupting, "ice" suggests coldness/stiffness — requires metaphorical reasoning'),
        (5, "break one's heart", "to cause deep emotional pain",
         '"break" directly maps to damage, "heart" directly maps to emotions'),
    ],
    "fa": [
        (1, "جا زدن", "to back out / to chicken out",
         '"جا" (place) and "زدن" (to hit) have no connection to backing out'),
        (3, "آب از آب تکان نمی‌خورد", "nothing is happening / everything is calm",
         '"water does not move from water" — water stillness relates to calmness, but requires reasoning'),
        (5, "دست و پا شکسته", "clumsy / broken (language)",
         '"hands and feet" + "broken" directly suggests clumsiness/dysfunction'),
    ],
    "fi": [
        (1, "mennä mönkään", "to go wrong / to fail",
         '"mennä" (to go) + "mönkään" (no clear literal meaning) — opaque'),
        (3, "olla jäällä", "to be in a difficult/uncertain situation",
         '"to be on ice" — ice suggests instability, but mapping to "difficult situation" requires inference'),
        (5, "katsoa sormien läpi", "to turn a blind eye / to overlook",
         '"to look through fingers" directly suggests deliberately not seeing'),
    ],
    "fr": [
        (1, "tomber dans les pommes", "to faint",
         '"to fall in the apples" — apples have no connection to fainting'),
        (3, "casser la glace", "to break the ice / to relieve tension",
         '"break" + "ice" — similar to English, requires metaphorical reasoning'),
        (5, "fermer les yeux", "to ignore / to turn a blind eye",
         '"to close the eyes" directly maps to choosing not to see'),
    ],
    "ja": [
        (1, "猿も木から落ちる", "even experts make mistakes",
         '"even monkeys fall from trees" — no direct link to expertise or mistakes'),
        (3, "首を長くする", "to wait eagerly",
         '"to make one\'s neck long" — stretching neck suggests looking/waiting, but requires metaphorical step'),
        (5, "耳が痛い", "criticism is hard to hear / it hurts to hear the truth",
         '"ears hurt" directly maps to hearing something painful'),
    ],
    "ko": [
        (1, "콩밥을 먹다", "to be in prison",
         '"to eat bean rice" — beans and rice have no connection to imprisonment'),
        (3, "입이 가볍다", "to be a blabbermouth / cannot keep secrets",
         '"mouth is light" — lightness suggests lack of control, but mapping requires inference'),
        (5, "눈이 높다", "to have high standards / to be picky",
         '"eyes are high" directly suggests looking up / aiming high'),
    ],
}


# ------------------------------------------------------------------
# Prompt 模板
# ------------------------------------------------------------------

DEFINITION = """Compositionality is the degree to which an idiom's figurative meaning can be inferred from the literal meanings of its component words.

Rating scale:
- 1 (Fully opaque): The meaning cannot be inferred at all from the individual words.
- 2 (Mostly opaque): Very little of the meaning can be guessed from the components.
- 3 (Semi-compositional): Some aspects of the meaning can be inferred through metaphorical reasoning.
- 4 (Mostly compositional): The meaning is largely inferable from the components.
- 5 (Fully compositional): The meaning is directly derivable from the individual words."""

ZERO_SHOT_PROMPT = """You are a linguist evaluating the compositionality of idiomatic expressions.

{definition}

Idiom: "{idiom}"
Meaning: "{meaning}"

Rate the compositionality of this idiom on a 1-5 scale.
Output ONLY a JSON object: {{"compositionality_score": <1-5>, "reason": "brief reason in 10-20 words"}}"""

ANCHORED_PROMPT = """You are a linguist evaluating the compositionality of idiomatic expressions.

{definition}

Here are calibration examples:

{examples}

Now rate this idiom:

Idiom: "{idiom}"
Meaning: "{meaning}"

Rate the compositionality of this idiom on a 1-5 scale.
Output ONLY a JSON object: {{"compositionality_score": <1-5>, "reason": "brief reason in 10-20 words"}}"""


def format_examples(src_code: str) -> str:
    examples = ANCHOR_EXAMPLES.get(src_code, ANCHOR_EXAMPLES["en"])
    lines = []
    for score, idiom, meaning, reason in examples:
        lines.append(f'- Score {score}: "{idiom}" = "{meaning}"\n  Reason: {reason}')
    return "\n".join(lines)


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------

def init_client() -> OpenAI:
    api_key = API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY")
    return OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=API_TIMEOUT)


def call_llm(client: OpenAI, prompt: str, model: str) -> Optional[str]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def parse_score(response_text: str) -> Optional[Dict]:
    if not response_text:
        return None
    for text in [response_text,
                 response_text.replace("```json", "").replace("```", "").strip()]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                score = parsed.get("compositionality_score")
                if score is not None:
                    score = int(score)
                    if 1 <= score <= 5:
                        return {"score": score, "reason": parsed.get("reason", "")}
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    m = re.search(r'"compositionality_score"\s*:\s*(\d)', response_text)
    if m:
        s = int(m.group(1))
        if 1 <= s <= 5:
            return {"score": s, "reason": ""}
    return None


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def score_direction(direction: Dict, mode: str, judge_model: str,
                    n_limit: Optional[int] = None):
    """
    direction: {pair, direction_code, src_code, meaning_file}
    """
    meaning_file = Path(direction["meaning_file"])
    if not meaning_file.exists():
        logger.warning(f"Missing: {meaning_file}"); return

    with open(meaning_file, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if n_limit:
        results = results[:n_limit]
    total = len(results)

    out_file = OUT_DIR / f"{direction['direction_code']}_{judge_model}_{mode}.json"

    # 断点续传:优先看 checkpoint;若无 checkpoint,fallback 到已存在的完整 output JSON
    completed: Dict[str, Dict] = {}
    ckpt = out_file.with_suffix(".checkpoint")
    if ckpt.exists():
        try:
            with open(ckpt, encoding="utf-8") as f:
                completed = json.load(f)
            logger.info(f"Resume from checkpoint: {len(completed)} done")
        except Exception:
            pass
    elif out_file.exists():
        try:
            with open(out_file, encoding="utf-8") as f:
                existing = json.load(f)
            for r in existing.get("results", []):
                if r.get("compositionality_score") is not None:
                    completed[str(r["id"])] = r
            valid = len(completed)
            if valid >= total:
                logger.info(f"[{direction['direction_code']} | {mode}] fully scored ({valid}/{total}), SKIP")
                return
            logger.info(f"Resume from existing output: {valid} done, {total-valid} to go")
        except Exception as e:
            logger.warning(f"Failed to read existing {out_file}: {e}")

    logger.info(f"==== {direction['direction_code']} | {judge_model} | mode={mode} | n={total} ====")

    client = init_client()
    scored: List[Dict] = []
    failed = 0

    for i, item in enumerate(results, 1):
        key = str(item["id"])
        idiom = item.get("idiom", "").strip()
        meaning = item.get("reference_meaning", "").strip()

        if key in completed:
            scored.append(completed[key])
            continue
        if not meaning or not idiom:
            entry = {**item, "compositionality_score": None,
                     "compositionality_reason": "no meaning/idiom",
                     "scoring_mode": mode, "judge_model": judge_model}
            scored.append(entry); completed[key] = entry; continue

        logger.info(f"  [{i}/{total}] id={key} | {idiom[:40]}")
        if mode == "anchored":
            prompt = ANCHORED_PROMPT.format(
                definition=DEFINITION,
                examples=format_examples(direction["src_code"]),
                idiom=idiom, meaning=meaning)
        else:
            prompt = ZERO_SHOT_PROMPT.format(
                definition=DEFINITION, idiom=idiom, meaning=meaning)

        raw = call_llm(client, prompt, judge_model)
        parsed = parse_score(raw)
        if parsed:
            entry = {**item, "compositionality_score": parsed["score"],
                     "compositionality_reason": parsed["reason"],
                     "scoring_mode": mode, "judge_model": judge_model}
        else:
            failed += 1
            entry = {**item, "compositionality_score": None,
                     "compositionality_reason": f"parse fail: {str(raw)[:80]}",
                     "scoring_mode": mode, "judge_model": judge_model}
        scored.append(entry); completed[key] = entry

        if i % 10 == 0:
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(completed, f, ensure_ascii=False)
        if REQUEST_INTERVAL: time.sleep(REQUEST_INTERVAL)

    # 保存
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {**data.get("config", {}), "judge_model": judge_model,
                       "scoring_mode": mode, "direction": direction["direction_code"]},
            "results": scored,
        }, f, ensure_ascii=False, indent=2)
    if ckpt.exists(): ckpt.unlink()

    valid = [s for s in scored if s.get("compositionality_score") is not None]
    scores = [s["compositionality_score"] for s in valid]
    logger.info(f"→ {out_file.name}")
    logger.info(f"  Success: {len(valid)}/{total} | Failed: {failed}")
    if valid:
        avg = sum(scores) / len(scores)
        dist = {i: scores.count(i) for i in range(1, 6)}
        logger.info(f"  Avg={avg:.2f} | Dist={dist}")


def enumerate_directions() -> List[Dict]:
    dirs = []
    for pair, spec in LANG_PAIRS.items():
        for d in spec["directions"]:
            dp = get_data_paths(pair, f"{d['src']}-{d['tgt']}")
            dirs.append({
                "pair": pair,
                "direction_code": f"{d['src']}-{d['tgt']}",
                "src_code": d["src"],
                "meaning_file": str(dp["meaning_file"]),
            })
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-pair", nargs="+", default=["all"],
                    help="'all' or pair codes: fa-en ja-en fr-en fi-en ko-en")
    ap.add_argument("--mode", choices=["zero-shot", "anchored", "both"],
                    default="both")
    ap.add_argument("--judge-model", default="gpt-5.2",
                    help="LLM annotator (default: gpt-5.2)")
    ap.add_argument("--pilot", action="store_true",
                    help="Only score 20 ja-en idioms for validation")
    args = ap.parse_args()

    dirs = enumerate_directions()

    if args.pilot:
        dirs = [d for d in dirs if d["direction_code"] == "ja-en"]
        n_limit = 20
        logger.info("PILOT: 20 ja-en idioms only")
    else:
        n_limit = None
        if "all" not in args.lang_pair:
            dirs = [d for d in dirs if d["pair"] in args.lang_pair
                    or d["direction_code"] in args.lang_pair]

    modes = ["anchored", "zero-shot"] if args.mode == "both" else [args.mode]

    logger.info(f"Directions: {[d['direction_code'] for d in dirs]} | Modes: {modes} | Judge: {args.judge_model}")

    for mode in modes:
        for direction in dirs:
            score_direction(direction, mode, args.judge_model, n_limit)

    logger.info("All done.")


if __name__ == "__main__":
    main()
