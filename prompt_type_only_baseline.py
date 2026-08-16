"""prompt_type_only_baseline.py — control #3 的硬证据(纯 CPU，无需 hidden states）

只用 prompt 类型(4 类 one-hot）作为特征预测 Group A vs B，看能到多高 balanced acc。
若远低于 hidden-state probe(~0.83），说明 probe 不是在读 prompt 条件。
两种设定都报：
  - class_weight='balanced'（与主 probe 一致）
  - 随机 5-fold StratifiedKFold
另报 prompt-type 与 label 的卡方/Cramér's V（关联强度）。
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALL_TARGETS, get_config

PROMPTS = ["BasicPrompt", "Meaning-FirstTranslationPrompt",
           "Anti-LiteralConstraintPrompt", "SuperLiteralBiasMitigationPrompt"]
PIDX = {p: i for i, p in enumerate(PROMPTS)}
MODEL = "Qwen3.5-9B"
LABELS = {"en-fa": "En->Fa", "fa-en": "Fa->En", "fr-en": "Fr->En",
          "fi-en": "Fi->En", "ko-en": "Ko->En", "ja-en": "Ja->En"}


def load_group(mech_dir, fname):
    with open(Path(mech_dir) / fname, encoding="utf-8") as f:
        return json.load(f)["results"]


def cramers_v(prompts, y):
    # 2 x 4 contingency
    tab = np.zeros((2, 4))
    for p, yy in zip(prompts, y):
        tab[yy, PIDX[p]] += 1
    n = tab.sum()
    row = tab.sum(1, keepdims=True)
    col = tab.sum(0, keepdims=True)
    exp = row @ col / n
    chi2 = np.nansum((tab - exp) ** 2 / np.where(exp == 0, np.nan, exp))
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return chi2, v


print(f"{'dir':>7} {'nA':>5} {'nB':>5} {'maj':>6} {'cw_bal':>7} {'cw_auc':>7} {'rnd_bal':>8} {'CramerV':>8}")
print("-" * 64)
for lp in ALL_TARGETS:
    cfg = get_config(lp, model=MODEL)
    mech = str(cfg["mechanistic_dir"])
    A = load_group(mech, "group_known_lte.json")
    B = load_group(mech, "group_known_non_lte_strict.json")
    prompts = np.array([it["prompt_type"] for it in A] + [it["prompt_type"] for it in B])
    y = np.array([0] * len(A) + [1] * len(B))
    X = np.eye(4)[[PIDX[p] for p in prompts]]  # one-hot

    maj = np.bincount(y).max() / len(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    # class_weight balanced (与主 probe 一致)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    yp = cross_val_predict(clf, X, y, cv=skf)
    ys = cross_val_predict(clf, X, y, cv=skf, method="decision_function")
    cw_bal = balanced_accuracy_score(y, yp)
    cw_auc = roc_auc_score(y, ys)

    # 随机(无 class_weight)
    clf2 = LogisticRegression(max_iter=2000, random_state=42)
    yp2 = cross_val_predict(clf2, X, y, cv=skf)
    rnd_bal = balanced_accuracy_score(y, yp2)

    _, v = cramers_v(prompts, y)
    print(f"{LABELS[lp]:>7} {len(A):>5} {len(B):>5} {maj:>6.3f} "
          f"{cw_bal:>7.3f} {cw_auc:>7.3f} {rnd_bal:>8.3f} {v:>8.3f}")
