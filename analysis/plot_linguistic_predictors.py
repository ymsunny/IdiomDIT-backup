"""
plot_linguistic_predictors.py — Visualization for the Linguistic Predictors
appendix section (审稿人 concern #3 rebuttal).

两幅图:
  (a) LTE rate stratified by compositionality bin (1-2 / 3 / 4-5), pooled
      across directions — headline chart.
  (b) LTE rate by compositionality score (1-5), one line per direction,
      Cleveland dot-plot with 95% Wilson CI.

输入:  analysis/output/linguistic_predictors/merged_dataset.csv
输出:  figures/linguistic_predictors_stratified_lte.{pdf,png}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_BASE = Path(__file__).resolve().parent.parent

# 色板:6 direction — 用 categorical-safe 手工调色板
#   (Okabe & Ito 2008 CVD-safe 8-color palette, dropped brown+black)
# Validated categorical palette (6 slots, PASS lightness+chroma+CVD)
# Assignment by ascending pooled LTE rate so hue-order tracks the story.
PALETTE = {
    "ko-en": "#2a78d6",  # blue    (lowest LTE)
    "en-fa": "#1baf7a",  # aqua
    "ja-en": "#eda100",  # yellow
    "fa-en": "#4a3aa7",  # violet
    "fr-en": "#e34948",  # red
    "fi-en": "#e87ba4",  # magenta (highest LTE)
}
LABELS = {"en-fa": "En→Fa", "fa-en": "Fa→En", "fr-en": "Fr→En",
          "fi-en": "Fi→En", "ja-en": "Ja→En", "ko-en": "Ko→En"}
DIR_ORDER = ["ko-en", "en-fa", "ja-en", "fa-en", "fr-en", "fi-en"]  # by ascending LTE


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["LTE", "composit_anchored"]).copy()
    df["composit_bin"] = pd.cut(
        df["composit_anchored"], bins=[0, 2.5, 3.5, 5.5],
        labels=["Low (1-2)", "Mid (3)", "High (4-5)"], right=False,
    )
    return df


def panel_a_pooled(ax, df: pd.DataFrame):
    """Panel A: LTE rate by compositionality bin, pooled across directions."""
    bins = ["Low (1-2)", "Mid (3)", "High (4-5)"]
    x = np.arange(len(bins))

    rates, cis_lo, cis_hi, counts = [], [], [], []
    for b in bins:
        sub = df[df["composit_bin"] == b]
        k, n = int(sub["LTE"].sum()), len(sub)
        r = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        rates.append(r); cis_lo.append(lo); cis_hi.append(hi); counts.append(n)

    yerr = np.array([[r - lo for r, lo in zip(rates, cis_lo)],
                     [hi - r for r, hi in zip(rates, cis_hi)]])

    # 单色 sequential:LTE rate 是 magnitude → 用一个 hue
    bar_color = "#2C7FB8"
    bars = ax.bar(x, rates, yerr=yerr, width=0.55,
                  color=bar_color, edgecolor="white", linewidth=2,
                  error_kw={"capsize": 4, "elinewidth": 1.2, "ecolor": "#555"})
    # 数据顶端 label
    for xi, r, hi, c in zip(x, rates, cis_hi, counts):
        ax.text(xi, hi + 0.008, f"{r:.1%}\nn={c:,}",
                ha="center", va="bottom", fontsize=8,
                color="#333")

    ax.set_xticks(x); ax.set_xticklabels(bins)
    ax.set_xlabel("Compositionality bin (anchored)")
    ax.set_ylabel("LTE rate")
    ax.set_title("(a) LTE rate by compositionality (pooled)", loc="left", fontsize=10)
    ax.set_ylim(0, max(cis_hi) * 1.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, color="#DDD", zorder=0)
    ax.set_axisbelow(True)


def panel_b_per_dir(ax, df: pd.DataFrame):
    """Panel B: LTE rate by composit score (1-5), one line per direction."""
    scores = [1, 2, 3, 4, 5]
    x = np.array(scores, dtype=float)

    for dc in DIR_ORDER:
        sub = df[df["direction"] == dc]
        if len(sub) == 0: continue
        rates, cis_lo, cis_hi = [], [], []
        valid_x = []
        for s in scores:
            ss = sub[sub["composit_anchored"] == s]
            k, n = int(ss["LTE"].sum()), len(ss)
            if n < 3:  # 少于 3 条 skip 该点
                continue
            r = k / n
            lo, hi = wilson_ci(k, n)
            valid_x.append(s)
            rates.append(r); cis_lo.append(lo); cis_hi.append(hi)

        if not rates: continue
        vx = np.array(valid_x, dtype=float)
        rates = np.array(rates); cis_lo = np.array(cis_lo); cis_hi = np.array(cis_hi)

        ax.plot(vx, rates, marker="o", linewidth=1.6, markersize=5.5,
                color=PALETTE[dc], label=LABELS[dc])
        ax.fill_between(vx, cis_lo, cis_hi, color=PALETTE[dc], alpha=0.12,
                        linewidth=0)

    ax.set_xticks(scores)
    ax.set_xlabel("Compositionality score (anchored, 1-5)")
    ax.set_ylabel("LTE rate")
    ax.set_title("(b) LTE rate by compositionality, per direction", loc="left", fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, color="#DDD", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=8,
              columnspacing=1.0, handletextpad=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",
                    default=str(_BASE / "analysis" / "output" / "linguistic_predictors" / "merged_dataset.csv"),
                    dest="input")
    ap.add_argument("--out",
                    default=str(_BASE / "figures" / "linguistic_predictors_stratified_lte"))
    args = ap.parse_args()

    df = load(Path(args.input))
    print(f"Loaded {len(df)} rows, {df['direction'].nunique()} directions")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    panel_a_pooled(axes[0], df)
    panel_b_per_dir(axes[1], df)
    plt.tight_layout()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.out).name
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=200)
    print(f"Wrote → {out_dir / stem}.pdf")
    print(f"Wrote → {out_dir / stem}.png")


if __name__ == "__main__":
    main()
