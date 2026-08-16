"""
plot_ltd_geometry.py — LTD 几何诊断图 (Finding 7 配套)

生成三张图:
  fig:ltd_cosine_heatmap        — 6 个 lp 各自 peak layer 的 6x6 cosine 矩阵
  fig:ltd_crosslang_by_layer    — 每层跨语言平均 off-diagonal cosine (折线 + random null band)
  fig:ltd_within_lang_drift     — 每个语言对自己邻层 cosine (每层 vs 下一层)

读: results/{lp}/{model}/mechanistic/probing_balanced_strict.json 里的 direction_balanced

用法:
  python analysis/plot_ltd_geometry.py
  python analysis/plot_ltd_geometry.py --model Qwen3-8B --output-dir figures/Qwen3-8B
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ALL_TARGETS

LABELS = {
    "en-fa": "En→Fa", "fa-en": "Fa→En", "fr-en": "Fr→En",
    "fi-en": "Fi→En", "ko-en": "Ko→En", "ja-en": "Ja→En",
}
BASE = Path(__file__).resolve().parent.parent


def load_directions(lp, model, suffix, exp_name=""):
    """Return (layers_dict, accs_dict). layers[L] is unit-norm direction."""
    mech = BASE / "results" / lp / model / "mechanistic"
    if exp_name:
        mech = mech / exp_name
    f = mech / f"probing_balanced{suffix}.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    layers, accs = {}, {}
    for r in data["results"]:
        v = np.asarray(r["direction_balanced"], dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-12)
        layers[r["layer"]] = v
        accs[r["layer"]] = r["balanced_subsample"]["balanced_accuracy"]
    return layers, accs


def random_null(D, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, D)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = rng.standard_normal((n, D)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    cos = np.einsum("nd,nd->n", A, B)
    return float(cos.mean()), float(cos.std()), float(np.quantile(np.abs(cos), 0.99))


def plot_heatmap(names, M, peak_info, out_dir, model, tag=""):
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # custom soft diverging palette: light steel-blue → white → light salmon
    # off-diag (≈0) → nearly white = visual evidence of near-orthogonality;
    # diagonal (1.0) → soft salmon, not saturated red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "soft_bwr", ["#5b8cb8", "#ffffff", "#d68877"], N=256
    )
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    # 宽 ≈ \columnwidth，width=\columnwidth 下近 1:1，字号≈正文
    fig, ax = plt.subplots(figsize=(2.85, 2.4))
    im = ax.imshow(M, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels([LABELS[n] for n in names], rotation=30, ha="right", fontsize=6.5)
    ax.set_yticklabels([LABELS[n] for n in names], fontsize=6.5)
    for i in range(len(names)):
        for j in range(len(names)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
    cbar.set_label("cosine similarity", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stem = f"peak_layer_cosine_heatmap_{model}" + (f"_{tag}" if tag else "")
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  [{ext}] {out}")
    plt.close(fig)


def plot_crosslang_by_layer(per_layer_mean, null_std, out_dir, model, tag=""):
    layers = sorted(per_layer_mean.keys())
    y = [per_layer_mean[L] for L in layers]
    # 宽 ≈ \columnwidth，width=\columnwidth 下近 1:1，字号≈正文
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.set_facecolor("white")
    ax.plot(layers, y, "-o", color="#1f77b4", lw=1.8, ms=4,
            label="Mean off-diag. cosine")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    # null band：用 ±3σ 虚线边界表示，不填灰色 → 保持白底
    ax.axhline(3 * null_std, color="gray", lw=0.9, ls=":", alpha=0.85)
    ax.axhline(-3 * null_std, color="gray", lw=0.9, ls=":", alpha=0.85,
               label=f"Random null ±3σ (σ≈{null_std:.3f})")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean cross-language cosine")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    stem = f"cross_lang_mean_cosine_by_layer_{model}" + (f"_{tag}" if tag else "")
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"  [{ext}] {out}")
    plt.close(fig)


def plot_within_lang_drift(all_dirs, out_dir, model, tag=""):
    # 宽 ≈ \columnwidth，width=\columnwidth 下近 1:1，字号≈正文
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for lp in ALL_TARGETS:
        if lp not in all_dirs:
            continue
        layers = all_dirs[lp]
        ks = sorted(layers.keys())
        coss = [float(layers[ks[i]] @ layers[ks[i + 1]]) for i in range(len(ks) - 1)]
        ax.plot(ks[:-1], coss, "-o", lw=1.4, ms=3, label=LABELS[lp])
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"Layer $\ell$")
    ax.set_ylabel(r"cos($\hat{w}_\ell$, $\hat{w}_{\ell+1}$)")
    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    stem = f"per_lang_within_layer_drift_{model}" + (f"_{tag}" if tag else "")
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"  [{ext}] {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--group-b", default="strict", choices=["strict", "loose"])
    parser.add_argument("--exp-name", default="",
                        help="读 mechanistic/{exp_name}/ 下的 probing（如 v4_gpt52）")
    parser.add_argument("--tag", default="",
                        help="输出文件名追加后缀（如 v4gpt52），避免覆盖默认图")
    parser.add_argument("--output-dir", default=None,
                        help="默认 figures/{model}")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"figures/{args.model}"

    suffix = "_strict" if args.group_b == "strict" else ""

    all_dirs, all_accs = {}, {}
    for lp in ALL_TARGETS:
        layers, accs = load_directions(lp, args.model, suffix, args.exp_name)
        all_dirs[lp] = layers
        all_accs[lp] = accs
        peak_L = max(accs, key=lambda L: accs[L])
        print(f"  {lp}: peak L={peak_L:>2}, bal_acc={accs[peak_L]:.4f}")

    D = len(next(iter(all_dirs.values()))[0])
    null_mean, null_std, null_q99 = random_null(D)
    print(f"\nRandom null (d={D}): mean={null_mean:+.4f}, std={null_std:.4f}, |99% q|={null_q99:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. peak-layer 6x6 cosine matrix
    names = list(ALL_TARGETS)
    peak_vecs = []
    for lp in names:
        peak_L = max(all_accs[lp], key=lambda L: all_accs[lp][L])
        peak_vecs.append(all_dirs[lp][peak_L])
    V = np.stack(peak_vecs, axis=0)
    M_peak = V @ V.T
    peak_info = ", ".join(f"{lp}:L{max(all_accs[lp], key=lambda L: all_accs[lp][L])}" for lp in names)
    plot_heatmap(names, M_peak, peak_info, out_dir, args.model, args.tag)

    # 2. per-layer mean off-diagonal cosine
    n_layers = len(next(iter(all_dirs.values())))
    per_layer_mean = {}
    for L in range(n_layers):
        vecs = np.stack([all_dirs[lp][L] for lp in names], axis=0)
        M_L = vecs @ vecs.T
        off = M_L[np.triu_indices(len(names), k=1)]
        per_layer_mean[L] = float(off.mean())
    plot_crosslang_by_layer(per_layer_mean, null_std, out_dir, args.model, args.tag)

    # 3. within-lang drift
    plot_within_lang_drift(all_dirs, out_dir, args.model, args.tag)

    # 4. console summary for paper text
    off_peak = M_peak[np.triu_indices(len(names), k=1)]
    print(f"\nPeak-layer off-diagonal cosine:")
    print(f"  range = [{off_peak.min():+.4f}, {off_peak.max():+.4f}]")
    print(f"  mean  = {off_peak.mean():+.4f}")
    print(f"  std   = {off_peak.std():.4f}")


if __name__ == "__main__":
    main()
