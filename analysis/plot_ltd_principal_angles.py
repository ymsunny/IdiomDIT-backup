"""
plot_ltd_principal_angles.py — 子空间级 LTD 对齐（principal angles）

单向量 cosine 只测 1-D 相似度。这里用 principal angles 测子空间对齐:
  - top-5 layer 子空间 (5-D per language)
  - all-32 layer 子空间 (32-D per language)

输出: figures/principal_angles_spectrum_{model}.{png,pdf}

用法:
  python analysis/plot_ltd_principal_angles.py
  python analysis/plot_ltd_principal_angles.py --model Qwen3-8B --output-dir figures
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ALL_TARGETS

LABELS = {
    "en-fa": "En→Fa", "fa-en": "Fa→En", "fr-en": "Fr→En",
    "fi-en": "Fi→En", "ko-en": "Ko→En", "ja-en": "Ja→En",
}
BASE = Path(__file__).resolve().parent.parent


def load_directions(lp, model, suffix, exp_name=""):
    mech = BASE / "results" / lp / model / "mechanistic"
    if exp_name:
        mech = mech / exp_name
    f = mech / f"probing_balanced{suffix}.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    layers_sorted = sorted(data["results"], key=lambda r: r["layer"])
    W = np.stack(
        [np.asarray(r["direction_balanced"], dtype=np.float64) for r in layers_sorted],
        axis=0,
    )
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    accs = np.asarray([r["balanced_subsample"]["balanced_accuracy"] for r in layers_sorted])
    return W, accs


def orthonormal_basis(M):
    Q, _ = np.linalg.qr(M.T)
    return Q.T


def principal_cosines(M1, M2):
    Q1 = orthonormal_basis(M1)
    Q2 = orthonormal_basis(M2)
    C = Q1 @ Q2.T
    svals = np.linalg.svd(C, compute_uv=False)
    return np.sort(np.clip(svals, 0.0, 1.0))[::-1]


def random_baseline(k, D, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    top, mean_across = [], []
    for _ in range(trials):
        A = rng.standard_normal((k, D))
        B = rng.standard_normal((k, D))
        coss = principal_cosines(A, B)
        top.append(coss[0])
        mean_across.append(coss.mean())
    return {
        "k": k, "D": D, "trials": trials,
        "top1_mean": float(np.mean(top)), "top1_std": float(np.std(top)),
        "mean_mean": float(np.mean(mean_across)),
        "mean_std": float(np.std(mean_across)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--group-b", default="strict", choices=["strict", "loose"])
    parser.add_argument("--exp-name", default="",
                        help="读 mechanistic/{exp_name}/ 下的 probing（如 v4_gpt52）")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    suffix = "_strict" if args.group_b == "strict" else ""

    per_lang_W, per_lang_accs = {}, {}
    for lp in ALL_TARGETS:
        W, a = load_directions(lp, args.model, suffix, args.exp_name)
        per_lang_W[lp] = W
        per_lang_accs[lp] = a
    D = next(iter(per_lang_W.values())).shape[1]
    avail = list(per_lang_W.keys())
    n_layers = next(iter(per_lang_W.values())).shape[0]
    print(f"Loaded {len(avail)} lang pairs, dim D = {D}, layers = {n_layers}")

    bases_all = {lp: per_lang_W[lp] for lp in avail}
    bases_top5 = {}
    for lp in avail:
        idx = np.argsort(per_lang_accs[lp])[::-1][:5]
        bases_top5[lp] = per_lang_W[lp][idx]

    print("Computing random-subspace nulls (trials=200)...")
    null_top5 = random_baseline(k=5, D=D, trials=200, seed=0)
    null_all = random_baseline(k=n_layers, D=D, trials=100, seed=1)
    print(f"  top5: top1 {null_top5['top1_mean']:.3f}±{null_top5['top1_std']:.3f}")
    print(f"  all{n_layers}: top1 {null_all['top1_mean']:.3f}±{null_all['top1_std']:.3f}")

    all_top5 = {(a, b): principal_cosines(bases_top5[a], bases_top5[b])
                for a, b in combinations(avail, 2)}
    all_full = {(a, b): principal_cosines(bases_all[a], bases_all[b])
                for a, b in combinations(avail, 2)}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    for (a, b), coss in all_top5.items():
        ax.plot(range(1, len(coss) + 1), coss, "-o", ms=4, lw=1.2, alpha=0.75,
                label=f"{LABELS[a]}↔{LABELS[b]}")
    ax.axhline(null_top5["mean_mean"], color="gray", ls="--", lw=0.8)
    ax.axhspan(null_top5["mean_mean"] - 3 * null_top5["mean_std"],
               null_top5["mean_mean"] + 3 * null_top5["mean_std"],
               color="gray", alpha=0.15, label="Null ±3σ")
    ax.set_title(f"Top-5-layer subspace (k=5 in $\\mathbb{{R}}^{{{D}}}$)")
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("cos(principal angle)")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

    ax = axes[1]
    for (a, b), coss in all_full.items():
        ax.plot(range(1, min(15, len(coss)) + 1), coss[:15], "-o", ms=3, lw=1.0,
                alpha=0.75, label=f"{LABELS[a]}↔{LABELS[b]}")
    ax.axhline(null_all["mean_mean"], color="gray", ls="--", lw=0.8)
    ax.axhspan(null_all["mean_mean"] - 3 * null_all["mean_std"],
               null_all["mean_mean"] + 3 * null_all["mean_std"],
               color="gray", alpha=0.15)
    ax.set_title(f"All-{n_layers}-layer subspace (first 15 components)")
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("cos(principal angle)")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(axis="y", ls="--", alpha=0.4)

    fig.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"principal_angles_spectrum_{args.model}"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"  [{ext}] {out}")
    plt.close(fig)

    t1_top5 = np.array([all_top5[k][0] for k in all_top5])
    t1_full = np.array([all_full[k][0] for k in all_full])
    print(f"\nTop-5 subspace top1 cosine: mean={t1_top5.mean():.3f} "
          f"(null top1={null_top5['top1_mean']:.3f})")
    print(f"All-{n_layers} subspace top1 cosine: mean={t1_full.mean():.3f} "
          f"(null top1={null_all['top1_mean']:.3f})")


if __name__ == "__main__":
    main()
