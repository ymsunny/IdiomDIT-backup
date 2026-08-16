"""ltb_conditional_regression.py — LTB-conditional regression(reviewer #3 精确回应).

问题:
  Reviewer 说的是 "remaining LTB cases",而 paper 定义 LTB = (D=1, I=1, LTE=1)
  的 know-but-error 条件性概念。原始的 8932 全数据回归混入了 detection failure +
  knowledge deficit 造成的 LTE,不能直接回答"哪些语言学属性预测残余 LTB"。

方案:
  过滤到 D=1 AND I=1 的 idiom × prompt instances,以此为 LTB-conditional 子集,
  在这个子集上做 GEE logistic regression:
       LTE ~ compositionality + length + (1 | lang_pair) + (1 | idiom_id)
  outcome:
       LTE=1 → know-but-error(即 LTB)
       LTE=0 → model 检测理解了但未产生 LTE(正确翻译)

用法:
  python analysis/ltb_conditional_regression.py
  python analysis/ltb_conditional_regression.py --model Qwen3.5-9B --judge v4_gpt52

输入:
  results/{pair}/Qwen3.5-9B/evaluation/detection_score.json         (D, per-idiom)
  results/{pair}/Qwen3.5-9B/evaluation/interpretation_p2_score.json (I, per-idiom, P2 = in-context)
  results/{pair}/Qwen3.5-9B/evaluation/translation_lte_v4_gpt52_score.json (LTE, per idiom×prompt)
  analysis/output/compositionality/{direction}_gpt-5.2_{anchored,zero-shot}.json
  analysis/output/linguistic_predictors/entity_counts.csv (for length)

输出:
  analysis/output/linguistic_predictors/ltb_conditional_dataset.csv
  analysis/output/linguistic_predictors/ltb_conditional_report.md
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LANG_PAIRS, get_data_paths

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
LING_DIR = _BASE / "analysis" / "output" / "linguistic_predictors"
COMP_DIR = _BASE / "analysis" / "output" / "compositionality"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def enumerate_directions() -> List[Dict]:
    dirs = []
    for pair, spec in LANG_PAIRS.items():
        for d in spec["directions"]:
            dirs.append({
                "pair": pair,
                "direction_code": f"{d['src']}-{d['tgt']}",
                "src_code": d["src"],
            })
    return dirs


def load_detection(pair: str, model: str) -> pd.DataFrame:
    fp = _BASE / "results" / pair / model / "evaluation" / "detection_score.json"
    if not fp.exists():
        logger.warning(f"missing detection: {fp}")
        return pd.DataFrame(columns=["direction", "id", "D"])
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    rows = [{"direction": pair, "id": r["id"],
             "D": 1 if r.get("detection_score") == 1 else 0}
            for r in data.get("results", [])]
    return pd.DataFrame(rows)


def load_interpretation(pair: str, model: str) -> pd.DataFrame:
    """P2 = in-context Interpretation,和翻译任务一样的输入"""
    fp = _BASE / "results" / pair / model / "evaluation" / "interpretation_p2_score.json"
    if not fp.exists():
        logger.warning(f"missing interpretation p2: {fp}")
        return pd.DataFrame(columns=["direction", "id", "I"])
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    rows = [{"direction": pair, "id": r["id"],
             "I": 1 if r.get("interpretation_score") == 1 else 0}
            for r in data.get("results", [])]
    return pd.DataFrame(rows)


def load_lte(pair: str, model: str, judge: str) -> pd.DataFrame:
    fp = _BASE / "results" / pair / model / "evaluation" / f"translation_lte_{judge}_score.json"
    if not fp.exists():
        logger.warning(f"missing LTE: {fp}")
        return pd.DataFrame()
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    rows = [{
        "direction": pair,
        "id": r["id"],
        "idiom": r["idiom"],
        "prompt_type": r["prompt_type"],
        "LTE": 1 if r.get("literal_translation_error") else 0,
    } for r in data.get("results", [])]
    return pd.DataFrame(rows)


def load_compositionality(direction_code: str, mode: str) -> pd.DataFrame:
    fp = COMP_DIR / f"{direction_code}_gpt-5.2_{mode}.json"
    if not fp.exists():
        return pd.DataFrame(columns=["direction", "id", f"composit_{mode.replace('-','')}"])
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    col = f"composit_{mode.replace('-','')}"
    rows = [{"direction": direction_code, "id": r["id"],
             col: r.get("compositionality_score")}
            for r in data.get("results", [])]
    return pd.DataFrame(rows)


def load_length(direction_code: str) -> pd.DataFrame:
    fp = LING_DIR / "entity_counts.csv"
    df = pd.read_csv(fp)
    sub = df[df["direction"] == direction_code][["id", "length"]].copy()
    sub["direction"] = direction_code
    return sub


def build_ltb_conditional_dataset(model: str, judge: str) -> pd.DataFrame:
    """构建 D=1 & I=1 过滤后的 idiom × prompt 长表 with all features."""
    logger.info(f"Building LTB-conditional dataset (model={model}, judge={judge})")
    frames = []
    for d in enumerate_directions():
        pair = d["pair"]
        dc = d["direction_code"]

        det = load_detection(dc, model)  # note: results/ organized by direction code
        interp = load_interpretation(dc, model)
        lte = load_lte(dc, model, judge)
        if lte.empty:
            continue

        # D 和 I 是 per-idiom(不管 prompt),join by (direction, id)
        merged = lte.merge(det, on=["direction", "id"], how="left")\
                    .merge(interp, on=["direction", "id"], how="left")

        # 补 D/I 缺失为 NaN → drop
        n_before = len(merged)
        merged = merged.dropna(subset=["D", "I"])
        # 只保留 D=1 & I=1 的 idiom × prompt outcomes
        cond = merged[(merged["D"] == 1) & (merged["I"] == 1)].copy()
        logger.info(f"  {dc}: LTE {n_before} → D=1&I=1 {len(cond)} "
                    f"({cond['id'].nunique()} unique idioms; "
                    f"LTB rate = {cond['LTE'].mean():.3f})")

        # 特征合并
        for mode in ["anchored", "zero-shot"]:
            comp = load_compositionality(dc, mode)
            cond = cond.merge(comp, on=["direction", "id"], how="left")
        length = load_length(dc)
        cond = cond.merge(length, on=["direction", "id"], how="left")

        frames.append(cond)

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total LTB-conditional rows: {len(df)}  |  idioms: {df.groupby('direction')['id'].nunique().to_dict()}")
    return df


def descriptive(df: pd.DataFrame) -> str:
    lines = ["## Descriptive: LTB-conditional subset (D=1 AND I=1)", ""]
    lines.append(f"- Total (idiom × prompt) outcomes: **{len(df)}**")
    lines.append(f"- Unique idioms (satisfying D=1 & I=1 in at least one prompt): "
                 f"**{df.groupby('direction')['id'].nunique().sum()}**")
    lines.append(f"- Overall LTB rate (= P(LTE=1 | D=1, I=1)): **{df['LTE'].mean():.3f}**")
    lines.append("")

    lines.append("### Per-direction breakdown")
    lines.append("| Direction | idioms (D=1 & I=1) | outcomes | LTB rate |")
    lines.append("|---|---:|---:|---:|")
    for pair, sub in df.groupby("direction"):
        lines.append(f"| {pair} | {sub['id'].nunique()} | {len(sub)} | {sub['LTE'].mean():.3f} |")
    lines.append("")
    return "\n".join(lines)


def fit(df: pd.DataFrame, composit_col: str) -> Dict:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    clean = df.dropna(subset=[composit_col, "length", "LTE"]).copy()
    clean["idiom_key"] = clean["direction"] + "_" + clean["id"].astype(str)
    clean[composit_col] = clean[composit_col].astype(float)
    clean["length"] = clean["length"].astype(float)

    formula = f"LTE ~ {composit_col} + length + C(direction)"
    model = smf.gee(
        formula, groups="idiom_key", data=clean,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit(maxiter=100)
    return {"model": model, "n_obs": len(clean),
            "n_idioms": clean["idiom_key"].nunique()}


def coeff_table(result: Dict, composit_col: str, header: str) -> str:
    m = result["model"]
    lines = [header, "",
             f"- N (idiom × prompt outcomes): {result['n_obs']}",
             f"- N idioms: {result['n_idioms']}",
             f"- Model: GEE logistic (Exchangeable), idiom-clustered SE, direction fixed effect",
             "",
             "| Predictor | β | 95% CI | OR | p | |",
             "|---|---:|---|---:|---:|---|"]
    for term, name in [
        (composit_col, f"Compositionality ({'anchored' if 'anch' in composit_col else 'zero-shot'})"),
        ("length", "Length (# tokens, control)"),
    ]:
        if term not in m.params.index: continue
        b = m.params[term]
        lo, hi = m.conf_int().loc[term]
        p = m.pvalues[term]
        or_ = np.exp(b)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        lines.append(f"| {name} | {b:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {or_:.3f} | {p:.2e} | {sig} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--judge", default="v4_gpt52")
    args = ap.parse_args()

    df = build_ltb_conditional_dataset(args.model, args.judge)
    LING_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LING_DIR / "ltb_conditional_dataset.csv", index=False, encoding="utf-8")

    md = ["# LTB-Conditional Regression Report",
          f"Model: **{args.model}** | Judge: **{args.judge}** | "
          "Filter: (D=1 AND I=1) on paper's Interpretation P2 (in-context)", ""]
    md.append(descriptive(df))

    for col in ["composit_anchored", "composit_zeroshot"]:
        try:
            result = fit(df, col)
            md.append(coeff_table(result, col,
                                  f"## Regression ({col})"))
            md.append("")
        except Exception as e:
            logger.error(f"Regression {col}: {e}")
            md.append(f"### Regression ({col}): FAILED — {e}")

    # Compare with full-data regression numbers (reference)
    md.append("## Comparison with full-data regression (all 8932 outcomes)")
    md.append("")
    md.append("| Regression | Composit (anchored) β | p | Composit (zero-shot) β | p |")
    md.append("|---|---:|---:|---:|---:|")
    md.append("| Full data (LTE on all outcomes) | −0.320 | 2.1e-6 | −0.517 | 2.8e-10 |")
    for col in ["composit_anchored", "composit_zeroshot"]:
        pass  # already in tables above
    md.append("| **LTB-conditional (this analysis)** | *see above* | | | |")

    report_fp = LING_DIR / "ltb_conditional_report.md"
    report_fp.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"Wrote {report_fp}")

    print("\n".join(md))


if __name__ == "__main__":
    main()
