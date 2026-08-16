"""
linguistic_predictors_regression.py — 混合效应 logistic regression:
LTE 是否由 idiom 的语言学特征预测?

审稿人 concern #3 的最终整合脚本。
将 entity count + compositionality 特征与 LTE 结果合并,做:
  1. Descriptive: 各 pair 特征分布 + 与 LTE rate 的边际关系
  2. Pooled mixed-effects logistic:
       LTE ~ compositionality + n_nouns + length
             + (1 | lang_pair) + (1 | idiom_id)
  3. Robustness: 分别用 anchored vs zero-shot compositionality
                 报告 Spearman(anchored, zero-shot)
  4. 输出 Markdown 表 + CSV

输入(要求这些文件都存在):
  - analysis/output/linguistic_predictors/entity_counts.csv
       from analysis/count_idiom_entities.py
  - analysis/output/compositionality/{direction}_gpt-5.2_{anchored,zero-shot}.json
       from analysis/score_compositionality.py
  - results/{pair}/Qwen3.5-9B/evaluation/translation_lte_v4_gpt52_score.json
       (paper canonical judge)

用法:
  python analysis/linguistic_predictors_regression.py
  python analysis/linguistic_predictors_regression.py --model Qwen3.5-9B --judge v4_gpt52

输出:
  analysis/output/linguistic_predictors/merged_dataset.csv
  analysis/output/linguistic_predictors/regression_report.md
  analysis/output/linguistic_predictors/regression_coefficients.csv
"""
import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LANG_PAIRS, get_data_paths

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
LING_DIR = _BASE / "analysis" / "output" / "linguistic_predictors"
COMP_DIR = _BASE / "analysis" / "output" / "compositionality"
LING_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 6 direction 枚举
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


# ------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------

def load_lte_scores(model: str, judge: str) -> pd.DataFrame:
    """加载 6 direction × Qwen3.5-9B × judge 的 LTE score。
    results/ 按 DIRECTION 组织(en-fa 和 fa-en 是两个独立目录),不按 LANG_PAIRS key。"""
    rows = []
    for direction in enumerate_directions():
        dc = direction["direction_code"]
        # 结果目录用 direction code(en-fa 和 fa-en 分开)
        score_file = _BASE / "results" / dc / model / "evaluation" / f"translation_lte_{judge}_score.json"
        if not score_file.exists():
            logger.warning(f"missing: {score_file}")
            continue
        with open(score_file, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            rows.append({
                "lang_pair": dc,                # 用 direction 作为分组
                "direction": dc,
                "id": r["id"],
                "idiom": r["idiom"],
                "prompt_type": r["prompt_type"],
                "LTE": 1 if r.get("literal_translation_error") else 0,
            })
    df = pd.DataFrame(rows)
    logger.info(f"LTE rows: {len(df)} | directions={df['lang_pair'].nunique()} | "
                f"idioms/dir: {df.groupby('lang_pair')['id'].nunique().to_dict()}")
    return df


def load_entity_counts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    logger.info(f"Entity counts: {len(df)} idioms")
    # 用 direction 作为主键(不用 pair,因为 en-fa 和 fa-en 是两个独立方向)
    return df[["direction", "id", "length",
               "n_nouns", "n_proper", "n_verbs", "n_adj", "n_content"]]


def load_compositionality(judge: str, mode: str) -> pd.DataFrame:
    """加载 6 direction 的 compositionality 分数"""
    rows = []
    for direction in enumerate_directions():
        dc = direction["direction_code"]
        score_file = COMP_DIR / f"{dc}_{judge}_{mode}.json"
        if not score_file.exists():
            logger.warning(f"missing: {score_file}")
            continue
        with open(score_file, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            rows.append({
                "direction": dc,
                "id": r["id"],
                f"composit_{mode.replace('-', '')}": r.get("compositionality_score"),
            })
    df = pd.DataFrame(rows)
    logger.info(f"Composit ({mode}): {len(df)} idioms | non-null: {df.iloc[:,-1].notna().sum()}")
    return df


# ------------------------------------------------------------------
# 合并
# ------------------------------------------------------------------

def merge(model: str, judge: str) -> pd.DataFrame:
    lte = load_lte_scores(model, judge)
    ent = load_entity_counts(LING_DIR / "entity_counts.csv")

    df = lte.merge(ent, on=["direction", "id"], how="left")

    for mode in ["anchored", "zeroshot"]:
        mode_file = "anchored" if mode == "anchored" else "zero-shot"
        comp = load_compositionality("gpt-5.2", mode_file)
        df = df.merge(comp, on=["direction", "id"], how="left")

    logger.info(f"Merged: {len(df)} rows | complete cases: "
                f"{df.dropna(subset=['n_nouns','composit_anchored']).shape[0]}")
    return df


# ------------------------------------------------------------------
# 描述统计 + 回归
# ------------------------------------------------------------------

def descriptive(df: pd.DataFrame) -> str:
    lines = ["## Descriptive statistics", ""]
    lines.append(f"- Total (idiom × prompt) rows: **{len(df)}**")
    lines.append(f"- Language pairs: {sorted(df['lang_pair'].unique().tolist())}")
    lines.append(f"- Overall LTE rate: **{df['LTE'].mean():.3f}**")
    lines.append("")

    lines.append("### LTE rate by lang_pair")
    lines.append("| pair | n_rows | idioms | LTE_rate |")
    lines.append("|---|---:|---:|---:|")
    for pair, sub in df.groupby("lang_pair"):
        lines.append(f"| {pair} | {len(sub)} | {sub['id'].nunique()} | {sub['LTE'].mean():.3f} |")
    lines.append("")

    lines.append("### Entity count distribution (n_nouns)")
    lines.append("| pair | mean | median | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    per_idiom = df.drop_duplicates(subset=["lang_pair", "id"])
    for pair, sub in per_idiom.groupby("lang_pair"):
        v = sub["n_nouns"].dropna()
        if len(v):
            lines.append(f"| {pair} | {v.mean():.2f} | {v.median():.0f} | {int(v.min())} | {int(v.max())} |")
    lines.append("")

    lines.append("### Compositionality (anchored) distribution")
    lines.append("| pair | mean | 1 | 2 | 3 | 4 | 5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pair, sub in per_idiom.groupby("lang_pair"):
        v = sub["composit_anchored"].dropna()
        if len(v):
            dist = v.value_counts().reindex([1,2,3,4,5], fill_value=0)
            lines.append(f"| {pair} | {v.mean():.2f} | "
                         f"{int(dist[1])} | {int(dist[2])} | {int(dist[3])} | {int(dist[4])} | {int(dist[5])} |")
    lines.append("")
    return "\n".join(lines)


def prompt_robustness(df: pd.DataFrame) -> str:
    """Spearman(anchored, zero-shot) per pair"""
    from scipy.stats import spearmanr
    lines = ["## Prompt-mode robustness (compositionality)", ""]
    lines.append("Spearman ρ between anchored and zero-shot GPT-5.2 scores:")
    lines.append("")
    lines.append("| pair | n | ρ | p |")
    lines.append("|---|---:|---:|---:|")
    per_idiom = df.drop_duplicates(subset=["lang_pair", "id"])
    for pair, sub in per_idiom.groupby("lang_pair"):
        a = sub["composit_anchored"].dropna()
        z = sub["composit_zeroshot"].dropna()
        common = sub.dropna(subset=["composit_anchored", "composit_zeroshot"])
        if len(common) < 5:
            lines.append(f"| {pair} | {len(common)} | — | insufficient |")
            continue
        rho, p = spearmanr(common["composit_anchored"], common["composit_zeroshot"])
        lines.append(f"| {pair} | {len(common)} | {rho:.3f} | {p:.2e} |")
    lines.append("")
    return "\n".join(lines)


def fit_regression(df: pd.DataFrame, composit_col: str) -> Dict:
    """
    Mixed-effects logistic:
        LTE ~ composit + n_nouns + length + (1 | lang_pair) + (1 | idiom_id)

    Statsmodels' MixedLM 不原生支持 logit 链接 + 多层 random effect,
    我们退而求其次:
      主表: GEE (Generalized Estimating Equations) logistic w/ pair-clustered SE
            + fixed dummy for pair(足够处理 6 pair 的边际效应)
      idiom-level random effect 用 idiom_id 作 GEE 的 subject (每 idiom 4 prompt 重复测量)
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    clean = df.dropna(subset=[composit_col, "n_nouns", "length", "LTE"]).copy()
    clean["idiom_key"] = clean["lang_pair"] + "_" + clean["id"].astype(str)
    clean[composit_col] = clean[composit_col].astype(float)
    clean["n_nouns"] = clean["n_nouns"].astype(float)
    clean["length"] = clean["length"].astype(float)

    logger.info(f"Regression n={len(clean)} | idioms={clean['idiom_key'].nunique()}")

    # GEE logistic, cluster by idiom_key, pair fixed effect
    formula = f"LTE ~ {composit_col} + n_nouns + length + C(lang_pair)"
    model = smf.gee(
        formula, groups="idiom_key", data=clean,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit(maxiter=100)

    result = {"model": model, "n_obs": len(clean),
              "n_idioms": clean["idiom_key"].nunique()}
    return result


def coefficients_table(result: Dict, composit_col: str) -> str:
    m = result["model"]
    params = m.params
    conf = m.conf_int()
    pvals = m.pvalues

    rows = []
    for term, name_display in [
        (composit_col, "Compositionality (1-5)"),
        ("n_nouns", "Entity count (# nouns)"),
        ("length", "Length (# tokens)"),
    ]:
        if term in params.index:
            b = params[term]
            lo, hi = conf.loc[term]
            p = pvals[term]
            or_ = np.exp(b)
            or_lo = np.exp(lo)
            or_hi = np.exp(hi)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            rows.append((name_display, b, lo, hi, or_, or_lo, or_hi, p, sig))

    lines = ["## Mixed-effects logistic regression",
             "",
             f"- N (idiom × prompt): {result['n_obs']}",
             f"- N idioms: {result['n_idioms']}",
             f"- Compositionality variant: **{composit_col}**",
             "- Model: GEE logistic (Exchangeable), idiom-clustered SE, pair fixed effect",
             "",
             "| Predictor | β | 95% CI | OR | OR 95% CI | p | |",
             "|---|---:|---|---:|---|---:|---|"]
    for name, b, lo, hi, or_, or_lo, or_hi, p, sig in rows:
        lines.append(f"| {name} | {b:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {or_:.3f} | [{or_lo:.3f}, {or_hi:.3f}] | {p:.2e} | {sig} |")
    lines.append("")
    lines.append("Significance: `***` p<0.001, `**` p<0.01, `*` p<0.05")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--judge", default="v4_gpt52",
                    help="LTE judge suffix (file: translation_lte_{judge}_score.json)")
    args = ap.parse_args()

    df = merge(args.model, args.judge)
    df.to_csv(LING_DIR / "merged_dataset.csv", index=False, encoding="utf-8")
    logger.info(f"Wrote merged dataset → {LING_DIR / 'merged_dataset.csv'}")

    # Descriptive + robustness
    md = ["# Linguistic Predictors of LTE — Regression Report",
          f"Model: **{args.model}** | Judge: **{args.judge}**", ""]
    md.append(descriptive(df))
    md.append(prompt_robustness(df))

    # Regression: primary = anchored, secondary = zero-shot
    coef_rows = []
    for composit_col in ["composit_anchored", "composit_zeroshot"]:
        try:
            result = fit_regression(df, composit_col)
            md.append(coefficients_table(result, composit_col))

            for term in ["composit_anchored", "composit_zeroshot", "n_nouns", "length"]:
                if term in result["model"].params.index:
                    b = result["model"].params[term]
                    lo, hi = result["model"].conf_int().loc[term]
                    p = result["model"].pvalues[term]
                    coef_rows.append({
                        "composit_variant": composit_col,
                        "predictor": term,
                        "beta": b,
                        "ci_low": lo,
                        "ci_high": hi,
                        "odds_ratio": np.exp(b),
                        "p_value": p,
                    })
        except Exception as e:
            logger.error(f"Regression with {composit_col} failed: {e}")
            md.append(f"### Regression ({composit_col}): FAILED — {e}")

    # 写文件
    (LING_DIR / "regression_report.md").write_text("\n".join(md), encoding="utf-8")
    logger.info(f"Wrote {LING_DIR / 'regression_report.md'}")

    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(
            LING_DIR / "regression_coefficients.csv", index=False, encoding="utf-8")
        logger.info(f"Wrote {LING_DIR / 'regression_coefficients.csv'}")

    print("\n".join(md))


if __name__ == "__main__":
    main()
