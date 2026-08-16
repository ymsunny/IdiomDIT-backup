"""build_lte_human_eval.py — 生成 LTE 人工评估文件(补齐 fi-en/fr-en 之外的语言对)。

原 fi-en/fr-en 的 lte_human_eval_{lp}_{annotator}.json 是一次性手工构建的,没脚本追溯。
本脚本按同一 schema 从 translation_lte_v4_gpt52_score.json 生成新语言对的标注种子文件:
  1. 读 score.json,收集 (id, prompt_type) 全部条目;
  2. 按 idiom_id 均匀随机采样 K 个(seed 固定,可复现);
  3. 采样 idiom 的全 4 prompt (K × 4 = 4K 条)写进标注文件;
  4. literal_translation 逐字直译由 GPT-5.2 批量生成(word-by-word gloss,匹配 fi/fr 风格);
  5. literal_check_human / meaning_check_human 初始 -1,human_note = ""。

输出与现有格式完全一致:
  results/human_eval/lte_human_eval_{lp}_{annotator}.json  (list of dicts)
可直接送标注者;完成后走 analysis/compute_human_llm_agreement.py 算 κ。

用法:
  # Ja→En 138 idioms(对齐 fr-en scale)、通用 annotator 名(TBD)、GPT-5.2 生成 gloss
  python analysis/build_lte_human_eval.py --lang-pair ja-en --n-idioms 138 --annotator TBD

  # 不调 API 先看采样对不对(dry run,gloss 留空)
  python analysis/build_lte_human_eval.py --lang-pair ja-en --n-idioms 138 --annotator TBD --skip-gloss

  # 之后指定实际标注者(会覆盖同名文件)
  python analysis/build_lte_human_eval.py --lang-pair ja-en --n-idioms 138 --annotator sun
"""
import os, sys, io, json, time, random, argparse
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_config, API_BASE_URL, API_KEY

LANG_NAME = {
    "ja-en": ("Japanese", "English"),
    "fi-en": ("Finnish", "English"),
    "fr-en": ("French", "English"),
    "ko-en": ("Korean", "English"),
    "fa-en": ("Persian", "English"),
    "en-fa": ("English", "Persian"),
}

GLOSS_PROMPT_TEMPLATE = """Task: Produce a word-by-word literal English translation of the following {src} sentence, keeping the source word order.

STRICT rules:
- Use only natural English words. Hyphenate multi-word glosses of one source token when needed (e.g., "donkey-bridge", "to-set-oneself", "those-there").
- NEVER emit grammatical labels. FORBIDDEN tokens: GEN, ACC, NOM, DAT, TOP, COP, QUOT, PRT, LOC, INS, OBJ, SBJ, "(subject)", "(object)", "(object-marker)", or any similar interlinear markup. Never write them under any circumstances.
- Handle particles and case markers naturally, without labels:
    * genitive markers (Japanese の, etc.) → render as "'s" or "of"
    * object markers (Japanese を, etc.) → OMIT the marker, keep only the noun and verb
    * subject markers (Japanese が, etc.) → OMIT the marker
    * dative markers (Japanese に, etc.) → "to" or "at"
    * topic markers (Japanese は, etc.) → OMIT or "as-for"
    * quotative markers (Japanese と, etc.) → "that"
- Do NOT interpret idioms or paraphrase — the output must be a naive literal translation.
- Preserve punctuation.

Reference examples (matching the desired style):
Finnish source: Yritäpä vielä asettua poikkiteloin.
English: Try-even still to-set-oneself crosswise.

French source: Ah la vache, ils sont costauds, ceux-là.
English: Ah the cow, they are strong, those-there.

Japanese source: 彼が本を読む。
English: he book reads.

Japanese source: 私の犬。
English: my dog.

Source ({src}): {sentence}

Output only the English word-by-word translation, one line, no quotes, no explanation, no grammatical labels."""


def gpt52_gloss_batch(pairs, src_name, model="gpt-5.2", timeout=60, sleep=0.0):
    """pairs: list of (key, sentence). Returns dict {key: gloss_string}."""
    from openai import OpenAI
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未设置 —— 请在 .env 里配好或跳过 --skip-gloss")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=timeout)
    out = {}
    total = len(pairs)
    for i, (key, sent) in enumerate(pairs, 1):
        prompt = GLOSS_PROMPT_TEMPLATE.format(src=src_name, sentence=sent)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            gloss = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [{i}/{total}] key={key} FAIL: {type(e).__name__}: {e}")
            gloss = ""
        out[key] = gloss
        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] gloss OK (len {len(gloss)}) preview: {gloss[:80]!r}")
        if sleep: time.sleep(sleep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-pair", required=True, choices=list(LANG_NAME.keys()))
    ap.add_argument("--model", default="Qwen3.5-9B",
                    help="生成 model_translation 的 MT 模型(默认 Qwen3.5-9B,与论文一致)")
    ap.add_argument("--lte-prefix", default="translation_lte_v4_gpt52",
                    help="从哪个 LTE 评分文件读源数据(默认 v4_gpt52,与论文正文一致)")
    ap.add_argument("--n-idioms", type=int, default=138,
                    help="采样多少 idioms(默认 138 对齐 fr-en)")
    ap.add_argument("--annotator", required=True,
                    help="标注者标签(如 sun / chaoyu / andrew / TBD);写进输出文件名")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-gloss", action="store_true",
                    help="不调 GPT-5.2 API 生成 literal_translation,全部留空(dry run)")
    ap.add_argument("--gloss-model", default="gpt-5.2")
    args = ap.parse_args()

    lp = args.lang_pair
    src, tgt = LANG_NAME[lp]
    root = Path(__file__).resolve().parent.parent
    cfg = get_config(lp, model=args.model)
    score_p = Path(cfg["eval_output"]) / f"{args.lte_prefix}_score.json"
    if not score_p.exists():
        raise FileNotFoundError(f"没找到 {score_p}")

    print(f"=== 构建 LTE 人工评估文件: {lp} ({src}→{tgt}) ===")
    print(f"  MT model: {args.model}")
    print(f"  LTE source: {score_p.name}")
    print(f"  采样: {args.n_idioms} idioms | seed={args.seed} | annotator={args.annotator}")

    d = json.load(open(score_p, encoding='utf-8'))
    rows = d.get("results", [])
    print(f"  score.json 共 {len(rows)} 条")

    # 每个 idiom_id 的 4 个 prompt 条聚起来
    by_id = defaultdict(list)
    for r in rows:
        by_id[str(r["id"])].append(r)
    all_ids = sorted(by_id.keys(), key=lambda x: int(x) if x.isdigit() else x)
    print(f"  独立 idiom 数: {len(all_ids)}")

    # 均匀采样 K
    rng = random.Random(args.seed)
    if args.n_idioms >= len(all_ids):
        sampled = all_ids
        print(f"  K({args.n_idioms}) >= 全量({len(all_ids)}),全取")
    else:
        sampled = rng.sample(all_ids, args.n_idioms)
        sampled.sort(key=lambda x: int(x) if x.isdigit() else x)
    print(f"  实际采样 {len(sampled)} idioms")

    # 收集采样 idiom 全 4 prompt 条(如果原文件某 prompt 缺就跳过)
    entries = []
    for iid in sampled:
        for r in by_id[iid]:
            entries.append(r)
    # 与 fi-en/fr-en 保持一致的排序:先按 prompt 类型分块(BasicPrompt →
    # Anti-Literal → Meaning-First → SuperLiteral),块内按 id 升序。
    PROMPT_ORDER = [
        "BasicPrompt",
        "Anti-LiteralConstraintPrompt",
        "Meaning-FirstTranslationPrompt",
        "SuperLiteralBiasMitigationPrompt",
    ]
    prompt_rank = {p: i for i, p in enumerate(PROMPT_ORDER)}
    entries.sort(key=lambda r: (prompt_rank.get(str(r.get("prompt_type", "")), 99),
                                int(r["id"]) if str(r["id"]).isdigit() else 0))
    print(f"  展开为 {len(entries)} 条(idiom × prompt)")

    # 逐字直译:去重(每 idiom_id 一个 sentence,4 个 prompt 共用同一 gloss)
    uniq_sentence = {}
    for r in entries:
        key = str(r["id"])
        if key not in uniq_sentence:
            uniq_sentence[key] = r.get("sentence", "")
    print(f"  需要生成 gloss 的独立 sentence 数: {len(uniq_sentence)}")

    glosses = {}
    if args.skip_gloss:
        print("  --skip-gloss 生效: literal_translation 全部留空")
        glosses = {k: "" for k in uniq_sentence}
    else:
        print(f"  调用 {args.gloss_model} 批量生成 word-by-word gloss ...")
        pairs = list(uniq_sentence.items())
        glosses = gpt52_gloss_batch(pairs, src, model=args.gloss_model)

    # 组装输出条目
    out_rows = []
    for r in entries:
        key = str(r["id"])
        out_rows.append({
            "id": int(r["id"]) if str(r["id"]).isdigit() else r["id"],
            "prompt_type": r.get("prompt_type", ""),
            "idiom": r.get("idiom", ""),
            "source_sentence": r.get("sentence", ""),
            "literal_translation": glosses.get(key, ""),
            "gold_translation": r.get("gold_translation", ""),
            "reference_meaning": r.get("reference_meaning", ""),
            "model_translation": r.get("model_translation", ""),
            "literal_check_human": -1,
            "meaning_check_human": -1,
            "human_note": "",
        })

    out_dir = root / "results" / "human_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"lte_human_eval_{lp}_{args.annotator}.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    from collections import Counter
    dist = Counter(r["prompt_type"] for r in out_rows)
    print(f"\n输出: {out_p}")
    print(f"  条数 {len(out_rows)}(prompt 分布:{dict(dist)})")
    print(f"  literal_check_human / meaning_check_human 全部初始化为 -1(待人工填)")
    if any(not g for g in glosses.values()):
        empty = sum(1 for g in glosses.values() if not g)
        print(f"  ⚠ 有 {empty} 条 gloss 生成失败,literal_translation 留空,建议重跑或标注者忽略此列")


if __name__ == "__main__":
    main()
