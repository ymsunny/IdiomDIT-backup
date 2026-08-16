"""
probe_ltb_by_hidden_state_balanced.py — Balanced probing: 处理 Group A/B 类别不平衡

Group A (known-LTE):      D=1, I=1, LTE=1
Group B (known-non-LTE):  D=1, I=1, LTE=0 [default: strict = MeaningConveyed=True]

两种平衡策略对比：
  1. balanced_subsample: 每 fold 从 B 组随机采样 |A| 个
  2. class_weight:       LR 的 class_weight='balanced'，不丢数据

hidden states 写入 npz 缓存，之后可 --skip-model-load 零 GPU 复算指标。

输出: results/{lang_pair}/{model}/mechanistic/
  probing_hidden_states[_strict].npz
  probing_balanced[_strict].json
  probing_balanced[_strict]_report.txt

用法:
  CUDA_VISIBLE_DEVICES=0 python mechanistic/probe_ltb_by_hidden_state_balanced.py \\
    --lang-pair en-fa --model Qwen3.5-9B
  CUDA_VISIBLE_DEVICES=0 python mechanistic/probe_ltb_by_hidden_state_balanced.py \\
    --lang-pair en-fa --model Qwen3.5-9B --group-b-file group_known_non_lte.json
"""

import os
import sys
import json
import argparse
import warnings
import builtins
import datetime
from pathlib import Path

_real_print = builtins.print
def print(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _real_print(f"[{ts}]", *args, **kwargs)
builtins.print = print

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODEL_NAME, PROMPTS, get_config, ALL_TARGETS, is_thinking_model
from model_utils import load_local_model

N_FOLDS = 5
RESAMPLE_SEEDS = [42, 123, 456, 789, 2026]


def load_model(model_name):
    model, tokenizer, _ = load_local_model(model_name)
    n_layers = model.config.num_hidden_layers
    return model, tokenizer, n_layers


def build_chat_prompt(sentence, prompt_type, src_lang, tgt_lang, model_name, tokenizer):
    system_prompt = PROMPTS[prompt_type].replace("[SRC]", src_lang).replace("[TGT]", tgt_lang)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if is_thinking_model(model_name):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def extract_last_token_hidden(model, tokenizer, text, n_layers):
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    activations = [None] * n_layers

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            activations[layer_idx] = h[0, -1, :].detach().cpu().float().numpy()
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return activations


def probe_balanced_subsample(X_all, y_all, n_seeds=5):
    n_a = np.sum(y_all == 0)
    n_b = np.sum(y_all == 1)
    minority_n = min(n_a, n_b)
    idx_a = np.where(y_all == 0)[0]
    idx_b = np.where(y_all == 1)[0]

    all_balanced_acc, all_cv_acc, all_f1, all_auc = [], [], [], []

    for seed in RESAMPLE_SEEDS[:n_seeds]:
        rng = np.random.RandomState(seed)
        if n_a < n_b:
            sampled_b = rng.choice(idx_b, minority_n, replace=False)
            indices = np.concatenate([idx_a, sampled_b])
        else:
            sampled_a = rng.choice(idx_a, minority_n, replace=False)
            indices = np.concatenate([sampled_a, idx_b])
        X = X_all[indices]
        y = y_all[indices]

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        bal_scores, acc_scores, f1_scores, auc_scores = [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
            clf.fit(X_train, y[train_idx])
            y_pred = clf.predict(X_test)
            y_score = clf.decision_function(X_test)
            bal_scores.append(balanced_accuracy_score(y[test_idx], y_pred))
            acc_scores.append(accuracy_score(y[test_idx], y_pred))
            f1_scores.append(f1_score(y[test_idx], y_pred))
            auc_scores.append(roc_auc_score(y[test_idx], y_score))

        all_balanced_acc.append(np.mean(bal_scores))
        all_cv_acc.append(np.mean(acc_scores))
        all_f1.append(np.mean(f1_scores))
        all_auc.append(np.mean(auc_scores))

    return {
        "balanced_accuracy": float(np.mean(all_balanced_acc)),
        "balanced_accuracy_std": float(np.std(all_balanced_acc)),
        "accuracy": float(np.mean(all_cv_acc)),
        "f1": float(np.mean(all_f1)),
        "roc_auc": float(np.mean(all_auc)),
        "roc_auc_std": float(np.std(all_auc)),
        "n_seeds": n_seeds,
    }


def probe_class_weight(X_all, y_all):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    bal_scores, acc_scores, f1_scores, auc_scores = [], [], [], []
    for train_idx, test_idx in skf.split(X_all, y_all):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[train_idx])
        X_test = scaler.transform(X_all[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 random_state=42, class_weight="balanced")
        clf.fit(X_train, y_all[train_idx])
        y_pred = clf.predict(X_test)
        y_score = clf.decision_function(X_test)
        bal_scores.append(balanced_accuracy_score(y_all[test_idx], y_pred))
        acc_scores.append(accuracy_score(y_all[test_idx], y_pred))
        f1_scores.append(f1_score(y_all[test_idx], y_pred))
        auc_scores.append(roc_auc_score(y_all[test_idx], y_score))
    return {
        "balanced_accuracy": float(np.mean(bal_scores)),
        "balanced_accuracy_std": float(np.std(bal_scores)),
        "accuracy": float(np.mean(acc_scores)),
        "f1": float(np.mean(f1_scores)),
        "roc_auc": float(np.mean(auc_scores)),
        "roc_auc_std": float(np.std(auc_scores)),
    }


def probe_grouped(X_all, y_all, groups, n_splits=N_FOLDS):
    """GroupKFold by idiom id: 同一 idiom 的不同 prompt 版本不跨 train/test（防泄漏，回应 reviewer concern #1）。
    用 class_weight='balanced' 处理不平衡，不丢样本。任一类的 group 数 < n_splits 时返回 None。"""
    n_groups_per_class = [len(np.unique(groups[y_all == c])) for c in np.unique(y_all)]
    if min(n_groups_per_class) < n_splits:
        return None
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        splits = list(sgkf.split(X_all, y_all, groups=groups))
    except Exception:
        return None
    bal_scores, acc_scores, f1_scores, auc_scores = [], [], [], []
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[train_idx])
        X_test = scaler.transform(X_all[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 random_state=42, class_weight="balanced")
        clf.fit(X_train, y_all[train_idx])
        y_pred = clf.predict(X_test)
        y_score = clf.decision_function(X_test)
        bal_scores.append(balanced_accuracy_score(y_all[test_idx], y_pred))
        acc_scores.append(accuracy_score(y_all[test_idx], y_pred))
        f1_scores.append(f1_score(y_all[test_idx], y_pred))
        auc_scores.append(roc_auc_score(y_all[test_idx], y_score))
    return {
        "balanced_accuracy": float(np.mean(bal_scores)),
        "balanced_accuracy_std": float(np.std(bal_scores)),
        "accuracy": float(np.mean(acc_scores)),
        "f1": float(np.mean(f1_scores)),
        "roc_auc": float(np.mean(auc_scores)),
        "roc_auc_std": float(np.std(auc_scores)),
        "n_splits": n_splits,
        "n_groups": int(len(np.unique(groups))),
    }


def probe_leave_one_prompt_out(X_all, y_all, prompts):
    """Leave-one-prompt-out: 用其余 prompt 训练、在 held-out prompt 上测。
    考验 prompt 泛化（direction 跨 prompt 是否一致 / 是不是只在预测 prompt 类型）。
    注意：idiom 在 train/test 间共享，本测不控制 idiom 泄漏——高分不能反驳 GroupKFold 的结论。"""
    uniq = sorted(set(p for p in prompts if p))
    if len(uniq) < 2:
        return None
    bals, aucs, per_prompt = [], [], {}
    for held in uniq:
        tr = prompts != held
        te = prompts == held
        if len(np.unique(y_all[tr])) < 2 or len(np.unique(y_all[te])) < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[tr])
        X_test = scaler.transform(X_all[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 random_state=42, class_weight="balanced")
        clf.fit(X_train, y_all[tr])
        ba = balanced_accuracy_score(y_all[te], clf.predict(X_test))
        au = roc_auc_score(y_all[te], clf.decision_function(X_test))
        per_prompt[held] = {"balanced_accuracy": float(ba), "roc_auc": float(au),
                            "n_test": int(te.sum())}
        bals.append(ba)
        aucs.append(au)
    if not bals:
        return None
    return {
        "balanced_accuracy": float(np.mean(bals)),
        "balanced_accuracy_std": float(np.std(bals)),
        "roc_auc": float(np.mean(aucs)),
        "n_folds": len(bals),
        "per_prompt": per_prompt,
    }


def probe_within_idiom(X_all, y_all, groups, n_splits=N_FOLDS, n_perms=20):
    """去 idiom 身份后 A/B 是否仍可分（测 within-idiom、prompt 调制的窄 claim）。
    只用 matched idiom（同时含 A 和 B），每个 idiom 减自己的均值（unsupervised，无标签泄漏），
    再在残差上 GroupKFold-by-idiom 训 LR(balanced)。matched idiom < n_splits 时返回 None。
    ⚠ 2-vs-2 小样本下 centering 本身会上偏，"chance" 不是 0.5；用 idiom 内标签置换得到 null，
    real >> null (p_perm 小) 才说明有真正的 within-idiom generation-time 信号。"""
    import collections
    labels_by_g = collections.defaultdict(set)
    for g, yy in zip(groups, y_all):
        labels_by_g[g].add(int(yy))
    matched = {g for g, s in labels_by_g.items() if 0 in s and 1 in s}
    if len(matched) < n_splits:
        return None
    mask = np.array([g in matched for g in groups])
    Xm = X_all[mask].astype(np.float64).copy()
    ym = y_all[mask]
    gm = groups[mask]
    for g in matched:                       # 逐 idiom 去均值（label-agnostic）
        idx = (gm == g)
        Xm[idx] -= Xm[idx].mean(axis=0, keepdims=True)
    n_groups_per_class = [len(np.unique(gm[ym == c])) for c in np.unique(ym)]
    if len(np.unique(ym)) < 2 or min(n_groups_per_class) < n_splits:
        return None
    def _cv_balaccs(labels):
        # 每个标签集各自分层，避免"按真实标签分层的折"给真实评估主场优势
        try:
            sp = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                           random_state=42).split(Xm, labels, groups=gm))
        except Exception:
            return None
        b = []
        for tr, te in sp:
            if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[te])) < 2:
                return None
            scaler = StandardScaler()
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                     random_state=42, class_weight="balanced")
            clf.fit(scaler.fit_transform(Xm[tr]), labels[tr])
            b.append(balanced_accuracy_score(labels[te], clf.predict(scaler.transform(Xm[te]))))
        return b

    real_b = _cv_balaccs(ym)
    if real_b is None:
        return None
    aucs = []
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=42).split(Xm, ym, groups=gm):
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 random_state=42, class_weight="balanced")
        clf.fit(scaler.fit_transform(Xm[tr]), ym[tr])
        aucs.append(roc_auc_score(ym[te], clf.decision_function(scaler.transform(Xm[te]))))

    # 置换 null：每个 idiom 内打乱标签（保留 centering/grouping 结构，破坏 残差→标签 关联）
    perm_rng = np.random.RandomState(0)
    null_means = []
    for _ in range(n_perms):
        yp = ym.copy()
        for g in matched:
            idx = np.where(gm == g)[0]
            yp[idx] = perm_rng.permutation(yp[idx])
        nb = _cv_balaccs(yp)
        if nb is not None:
            null_means.append(float(np.mean(nb)))

    real_mean = float(np.mean(real_b))
    null_mean = float(np.mean(null_means)) if null_means else None
    null_std = float(np.std(null_means)) if null_means else None
    p_perm = (float((np.sum(np.array(null_means) >= real_mean) + 1) / (len(null_means) + 1))
              if null_means else None)
    return {
        "balanced_accuracy": real_mean,
        "balanced_accuracy_std": float(np.std(real_b)),
        "roc_auc": float(np.mean(aucs)),
        "null_mean": null_mean,
        "null_std": null_std,
        "p_perm": p_perm,
        "n_perms": len(null_means),
        "n_matched_idioms": len(matched),
        "n_samples": int(mask.sum()),
    }


def majority_baseline(y_all):
    counts = np.bincount(y_all.astype(int))
    return float(counts.max() / counts.sum())


def main():
    parser = argparse.ArgumentParser(description="Balanced probing: LTB direction in hidden states")
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS)
    parser.add_argument("--model", required=True, help="模型名，如 Qwen3.5-9B")
    parser.add_argument("--group-b-file", default="group_known_non_lte_strict.json",
                        help="Group B 文件名 (默认: strict; loose: group_known_non_lte.json)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-reextract", action="store_true",
                        help="忽略缓存，强制重新前向模型")
    parser.add_argument("--skip-model-load", action="store_true",
                        help="假定缓存存在，跳过模型加载（纯 CPU 复算指标）")
    parser.add_argument("--exp-name", default="",
                        help="实验名称（输出到 mechanistic/{exp_name}/）")
    args = parser.parse_args()

    cfg = get_config(args.lang_pair, model=args.model)
    direction = cfg["direction"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    mech_dir = str(cfg["mechanistic_dir"])
    if args.exp_name:
        mech_dir = str(Path(mech_dir) / args.exp_name)

    # cache / output 文件名后缀
    group_b_filename = args.group_b_file
    if group_b_filename == "group_known_non_lte.json":
        cache_suffix = ""
        output_suffix = ""
    else:
        stem = Path(group_b_filename).stem
        tag = stem.replace("group_known_non_lte", "").lstrip("_") or "custom"
        cache_suffix = f"_{tag}"
        output_suffix = f"_{tag}"

    # 加载分组
    with open(os.path.join(mech_dir, "group_known_lte.json")) as f:
        group_a = json.load(f)["results"]
    with open(os.path.join(mech_dir, group_b_filename)) as f:
        group_b = json.load(f)["results"]

    if args.max_samples:
        group_a = group_a[:args.max_samples]
        group_b = group_b[:args.max_samples]

    print(f"语言对: {args.lang_pair} | 模型: {args.model}")
    print(f"Group A (known-LTE): {len(group_a)} | Group B: {len(group_b)} ({group_b_filename})")
    print(f"类别比例: 1:{len(group_b) / max(len(group_a), 1):.1f}")

    cache_path = os.path.join(mech_dir, f"probing_hidden_states{cache_suffix}.npz")
    can_skip_model = (
        args.skip_model_load
        and os.path.exists(cache_path)
        and not args.force_reextract
    )

    if can_skip_model:
        print(f"跳过模型加载，复用缓存: {cache_path}")
        model = tokenizer = None
        n_layers = None
    else:
        model, tokenizer, n_layers = load_model(args.model)
        print(f"模型层数: {n_layers}")

    if os.path.exists(cache_path) and not args.force_reextract:
        print(f"\n加载 hidden states 缓存: {cache_path}")
        blob = np.load(cache_path)
        hidden_tensor = blob["hidden"]
        y_all = blob["y"]
        cached_n_a = int(blob["n_a"]) if "n_a" in blob.files else None
        cached_n_b = int(blob["n_b"]) if "n_b" in blob.files else None
        if cached_n_a is not None and (cached_n_a != len(group_a) or cached_n_b != len(group_b)):
            print(f"  警告: 缓存样本数 (A={cached_n_a}, B={cached_n_b}) 与当前 group "
                  f"(A={len(group_a)}, B={len(group_b)}) 不一致。使用 --force-reextract 重跑。")
    else:
        print("\n提取 hidden states...")
        hidden_per_sample = []
        labels = []
        for i, item in enumerate(group_a):
            text = build_chat_prompt(item["sentence"], item["prompt_type"],
                                     src_lang, tgt_lang, args.model, tokenizer)
            h = extract_last_token_hidden(model, tokenizer, text, n_layers)
            hidden_per_sample.append(np.stack(h, axis=0))
            labels.append(0)
            if (i + 1) % 50 == 0:
                print(f"  A: {i+1}/{len(group_a)}")
        for i, item in enumerate(group_b):
            text = build_chat_prompt(item["sentence"], item["prompt_type"],
                                     src_lang, tgt_lang, args.model, tokenizer)
            h = extract_last_token_hidden(model, tokenizer, text, n_layers)
            hidden_per_sample.append(np.stack(h, axis=0))
            labels.append(1)
            if (i + 1) % 50 == 0:
                print(f"  B: {i+1}/{len(group_b)}")

        hidden_tensor = np.stack(hidden_per_sample, axis=0).transpose(1, 0, 2).astype(np.float32)
        y_all = np.array(labels)
        np.savez_compressed(
            cache_path,
            hidden=hidden_tensor,
            y=y_all,
            n_a=np.array(len(group_a)),
            n_b=np.array(len(group_b)),
        )
        print(f"  缓存写入: {cache_path}  shape={hidden_tensor.shape}")

    # groups: 按 idiom id 对齐样本顺序（group_a 在前、group_b 在后），用于 GroupKFold 防泄漏
    groups_all = np.array([str(it.get("id", "")) for it in group_a]
                          + [str(it.get("id", "")) for it in group_b])
    if len(groups_all) != len(y_all):
        print(f"  警告: groups 数 ({len(groups_all)}) != 样本数 ({len(y_all)})，"
              f"GroupKFold 跳过（缓存与当前 group 不一致？用 --force-reextract）。")
        groups_all = None

    # prompts: 按样本顺序，用于 leave-one-prompt-out（考验 prompt 泛化）
    prompts_all = np.array([str(it.get("prompt_type", "")) for it in group_a]
                           + [str(it.get("prompt_type", "")) for it in group_b])
    if len(prompts_all) != len(y_all):
        prompts_all = None

    baseline = majority_baseline(y_all)

    print(f"\n逐层 balanced probing ({hidden_tensor.shape[0]} 层)...")
    n_layers = hidden_tensor.shape[0]
    per_layer_results = []
    for layer in range(n_layers):
        X = hidden_tensor[layer]

        result_sub = probe_balanced_subsample(X, y_all, n_seeds=5)
        result_cw = probe_class_weight(X, y_all)
        result_grouped = probe_grouped(X, y_all, groups_all) if groups_all is not None else None
        result_lopo = probe_leave_one_prompt_out(X, y_all, prompts_all) if prompts_all is not None else None
        result_within = probe_within_idiom(X, y_all, groups_all) if groups_all is not None else None

        # direction_imbalanced（旧 knockout 方法，保留作对比）
        scaler_imb = StandardScaler()
        clf_imb = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
        clf_imb.fit(scaler_imb.fit_transform(X), y_all)
        w_imb = clf_imb.coef_[0] / scaler_imb.scale_
        w_imb = w_imb / (np.linalg.norm(w_imb) + 1e-10)

        # direction_balanced（class_weight='balanced'，主分析用）
        scaler_bal = StandardScaler()
        clf_bal = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                     random_state=42, class_weight="balanced")
        clf_bal.fit(scaler_bal.fit_transform(X), y_all)
        w_bal = clf_bal.coef_[0] / scaler_bal.scale_
        w_bal = w_bal / (np.linalg.norm(w_bal) + 1e-10)

        cos_sim = float(np.dot(w_imb, w_bal))

        per_layer_results.append({
            "layer": layer,
            "balanced_subsample": result_sub,
            "class_weight": result_cw,
            "grouped_cv": result_grouped,
            "lopo_cv": result_lopo,
            "within_idiom_cv": result_within,
            "direction_imbalanced": w_imb.tolist(),
            "direction_balanced": w_bal.tolist(),
            "direction_cosine_similarity": cos_sim,
        })

        g_str = (f" | grouped bal_acc={result_grouped['balanced_accuracy']:.3f}"
                 f"±{result_grouped['balanced_accuracy_std']:.3f} auc={result_grouped['roc_auc']:.3f}"
                 if result_grouped else " | grouped=NA")
        l_str = (f" | LOPO bal_acc={result_lopo['balanced_accuracy']:.3f}"
                 if result_lopo else "")
        w_str = (f" | within-idiom bal_acc={result_within['balanced_accuracy']:.3f}"
                 if result_within else "")
        print(f"  Layer {layer:>2}: "
              f"subsample bal_acc={result_sub['balanced_accuracy']:.3f}±{result_sub['balanced_accuracy_std']:.3f} "
              f"auc={result_sub['roc_auc']:.3f} "
              f"| cw bal_acc={result_cw['balanced_accuracy']:.3f} "
              f"auc={result_cw['roc_auc']:.3f}" + g_str + l_str + w_str)

    output = {
        "config": {
            "lang_pair": args.lang_pair,
            "model": args.model,
            "direction": direction,
            "group_b_file": group_b_filename,
            "n_group_a": len(group_a),
            "n_group_b": len(group_b),
            "imbalance_ratio": len(group_b) / max(len(group_a), 1),
            "majority_baseline": baseline,
        },
        "results": per_layer_results,
    }
    out_path = os.path.join(mech_dir, f"probing_balanced{output_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(mech_dir, f"probing_balanced{output_suffix}_report.txt")
    peak_sub = max(per_layer_results, key=lambda r: r["balanced_subsample"]["balanced_accuracy"])
    peak_cw = max(per_layer_results, key=lambda r: r["class_weight"]["balanced_accuracy"])
    grouped_layers = [r for r in per_layer_results if r.get("grouped_cv")]
    peak_grouped = max(grouped_layers, key=lambda r: r["grouped_cv"]["balanced_accuracy"]) if grouped_layers else None
    lopo_layers = [r for r in per_layer_results if r.get("lopo_cv")]
    peak_lopo = max(lopo_layers, key=lambda r: r["lopo_cv"]["balanced_accuracy"]) if lopo_layers else None
    within_layers = [r for r in per_layer_results if r.get("within_idiom_cv")]
    def _within_margin(r):
        wc = r["within_idiom_cv"]
        nm = wc.get("null_mean")
        return wc["balanced_accuracy"] - (nm if nm is not None else 0.5)
    peak_within = max(within_layers, key=_within_margin) if within_layers else None
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"Balanced Probing — {direction}\n")
        f.write(f"Model: {args.model} | Group B: {group_b_filename}\n")
        f.write(f"Group A: {len(group_a)} | Group B: {len(group_b)} "
                f"(ratio 1:{output['config']['imbalance_ratio']:.1f})\n")
        f.write(f"Majority baseline: {baseline:.4f}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Peak (balanced subsample): Layer {peak_sub['layer']} "
                f"= {peak_sub['balanced_subsample']['balanced_accuracy']:.4f} "
                f"± {peak_sub['balanced_subsample']['balanced_accuracy_std']:.4f} "
                f"(AUC {peak_sub['balanced_subsample']['roc_auc']:.4f})\n")
        f.write(f"Peak (class_weight):       Layer {peak_cw['layer']} "
                f"= {peak_cw['class_weight']['balanced_accuracy']:.4f} "
                f"(AUC {peak_cw['class_weight']['roc_auc']:.4f})\n")
        if peak_grouped:
            f.write(f"Peak (GroupKFold/idiom):   Layer {peak_grouped['layer']} "
                    f"= {peak_grouped['grouped_cv']['balanced_accuracy']:.4f} "
                    f"± {peak_grouped['grouped_cv']['balanced_accuracy_std']:.4f} "
                    f"(AUC {peak_grouped['grouped_cv']['roc_auc']:.4f}, "
                    f"{peak_grouped['grouped_cv']['n_groups']} groups)\n")
        else:
            f.write("Peak (GroupKFold/idiom):   NA (某类 group 数 < fold 数)\n")
        if peak_lopo:
            f.write(f"Peak (LeaveOnePromptOut):  Layer {peak_lopo['layer']} "
                    f"= {peak_lopo['lopo_cv']['balanced_accuracy']:.4f} "
                    f"± {peak_lopo['lopo_cv']['balanced_accuracy_std']:.4f} "
                    f"(AUC {peak_lopo['lopo_cv']['roc_auc']:.4f}, "
                    f"{peak_lopo['lopo_cv']['n_folds']} prompts held out)\n")
        else:
            f.write("Peak (LeaveOnePromptOut):  NA\n")
        if peak_within:
            wc = peak_within['within_idiom_cv']
            null_s = (f", null {wc['null_mean']:.4f}±{wc['null_std']:.4f}, p_perm={wc['p_perm']:.3f}"
                      if wc.get('null_mean') is not None else "")
            f.write(f"Peak (WithinIdiom/center): Layer {peak_within['layer']} "
                    f"= {wc['balanced_accuracy']:.4f} "
                    f"± {wc['balanced_accuracy_std']:.4f} "
                    f"(AUC {wc['roc_auc']:.4f}, {wc['n_matched_idioms']} matched idioms{null_s})\n\n")
        else:
            f.write("Peak (WithinIdiom/center): NA (matched idiom < fold 数)\n\n")
        best_bal = peak_sub['balanced_subsample']['balanced_accuracy']
        diff = best_bal - baseline
        f.write(f"vs 多数类基线: {diff:+.4f} "
                f"({'显著优于基线' if diff > 0.1 else '可能偏差较小'})\n\n")
        f.write("逐层 balanced accuracy / ROC-AUC:\n")
        f.write(f"{'Layer':>5} {'sub_bal_acc':>12} {'sub_AUC':>10} "
                f"{'cw_bal_acc':>12} {'cw_AUC':>10}\n")
        for r in per_layer_results:
            f.write(f"{r['layer']:>5} "
                    f"{r['balanced_subsample']['balanced_accuracy']:>12.4f} "
                    f"{r['balanced_subsample']['roc_auc']:>10.4f} "
                    f"{r['class_weight']['balanced_accuracy']:>12.4f} "
                    f"{r['class_weight']['roc_auc']:>10.4f}\n")

    print(f"\n结果: {out_path}")
    print(f"报告: {report_path}")
    print(f"\n多数类基线: {baseline:.4f}")
    print(f"Peak bal_acc (subsample): L{peak_sub['layer']} = "
          f"{peak_sub['balanced_subsample']['balanced_accuracy']:.4f}")
    print(f"vs 基线: {peak_sub['balanced_subsample']['balanced_accuracy'] - baseline:+.4f}")


if __name__ == "__main__":
    main()
