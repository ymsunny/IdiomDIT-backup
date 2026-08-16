"""build_di_human_eval.py — 生成 Detection / Interpretation 人工评估数据 (fi-en, fr-en, ja-en; Qwen3.5-9B)

D (Detection): 每 idiom 一行；给 gold idiom、句子、模型的 has_idiom / detected_idiom，
  人工判断模型是否正确识别了该 idiom → 填 detection_human (1/0)。
I (Interpretation, any-hit): 每 idiom 一行；展示三个 probe (P2a/P2b/P2c) 的模型答案，
  人工判断三者中是否有任一正确传达 reference_meaning → 填 interpretation_human (1/0)。

盲标：不含自动 judge 结果（detection_score / interpretation_score），只给模型输出 + 参考，避免带偏。
Detection / Interpretation 是 per-idiom 的（与翻译 prompt 无关），所以每方向 = idiom 数
（fi-en 85, fr-en 138 全量；ja-en 1188 个 idiom 太多，取样：与 lte_human_eval_ja-en_chaoyu.json
一致的 138 个 idiom id，保证同一批 idiom 在 LTE / Detection / Interpretation 三个人工评估任务里对齐）。

每个方向额外按 3 位标注者(andrew/chaoyu/sun)各出一份独立副本,条目、顺序完全一致,
只有 detection_human/interpretation_human/human_note 各自独立填写 —— 与
lte_human_eval_{lp}_{annotator}.json 的多标注者模式一致。

输出: results/human_eval/{detection,interpretation}_human_eval_{lang}.json (未分标注者的源文件)
     results/human_eval/{detection,interpretation}_human_eval_{lang}_{annotator}.json (标注者副本)
用法: python analysis/build_di_human_eval.py
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "human_eval"
MODEL = "Qwen3.5-9B"
LANGS = ["fi-en", "fr-en", "ja-en"]
ANNOTATORS = ["andrew", "chaoyu", "sun"]

# ja-en 的 idiom 全集太大（1188），取样对齐到 LTE 人工评估已用的 138 个 idiom id
ID_FILTER_SOURCE = {
    "ja-en": OUT / "lte_human_eval_ja-en_chaoyu.json",
}


def load_id_filter(lp):
    src = ID_FILTER_SOURCE.get(lp)
    if src is None:
        return None
    rows = json.load(open(src, encoding="utf-8"))
    return sorted(set(r["id"] for r in rows))


def parse_meaning(resp_list):
    """probe 答案形如 ['{"meaning": "..."}'] → 取 meaning 文本；解析失败则原样返回。"""
    if not resp_list:
        return ""
    s = resp_list[0] if isinstance(resp_list, list) else resp_list
    try:
        return json.loads(s).get("meaning", str(s))
    except Exception:
        return str(s)


def load_results(p):
    d = json.load(open(p, encoding="utf-8"))
    return d.get("results", d) if isinstance(d, dict) else d


def write_annotator_copies(rows, out_stem):
    """rows 写出 {out_stem}.json,再各复制一份给每个 annotator: {out_stem}_{annotator}.json。"""
    base_path = OUT / f"{out_stem}.json"
    json.dump(rows, open(base_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for ann in ANNOTATORS:
        ann_rows = copy.deepcopy(rows)
        ann_path = OUT / f"{out_stem}_{ann}.json"
        json.dump(ann_rows, open(ann_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return base_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for lp in LANGS:
        base = ROOT / "results" / lp / MODEL
        det = load_results(base / "evaluation" / "detection_score.json")
        itp = load_results(base / "evaluation" / "interpretation_p2_score.json")

        id_filter = load_id_filter(lp)
        if id_filter is not None:
            id_set = set(id_filter)
            det = [r for r in det if r["id"] in id_set]
            itp = [r for r in itp if r["id"] in id_set]
            det.sort(key=lambda r: r["id"])
            itp.sort(key=lambda r: r["id"])
            print(f"{lp}: 按 {ID_FILTER_SOURCE[lp].name} 过滤到 {len(id_set)} 个 idiom id "
                  f"(detection 命中 {len(det)}, interpretation 命中 {len(itp)})")

        det_rows = [{
            "id": r["id"],
            "lang_pair": lp,
            "idiom": r.get("idiom", ""),
            "sentence": r.get("sentence", ""),
            "gold_translation": r.get("gold_translation", ""),
            "model_has_idiom": r.get("has_idiom"),
            "model_detected_idiom": r.get("detected_idiom", ""),
            "detection_human": -1,   # 人工：模型是否正确识别了该 idiom (1=对, 0=错, -1=未评)
            "human_note": "",
        } for r in det]
        det_path = write_annotator_copies(det_rows, f"detection_human_eval_{lp}")

        itp_rows = [{
            "id": r["id"],
            "lang_pair": lp,
            "idiom": r.get("idiom", ""),
            "sentence": r.get("sentence", ""),
            "reference_meaning": r.get("reference_meaning", ""),
            "model_answer_P2a": parse_meaning(r.get("P2a_responses")),
            "model_answer_P2b": parse_meaning(r.get("P2b_responses")),
            "model_answer_P2c": parse_meaning(r.get("P2c_responses")),
            # 人工(any-hit)：三个答案中任一正确传达 reference_meaning 则 1，否则 0；-1=未评
            "interpretation_human": -1,
            "human_note": "",
        } for r in itp]
        itp_path = write_annotator_copies(itp_rows, f"interpretation_human_eval_{lp}")

        print(f"{lp}: detection {len(det_rows)} 条 -> {det_path.name} (+{len(ANNOTATORS)} 标注者副本) | "
              f"interpretation {len(itp_rows)} 条 -> {itp_path.name} (+{len(ANNOTATORS)} 标注者副本)")


if __name__ == "__main__":
    main()
