"""
check_ltd_cosine.py — 验证 Finding 7: 6 个语言对的 LTD 是否近似正交

对每个语言对取 peak balanced_subsample 层的 direction_balanced 向量，
两两算 cosine，汇总 off-diagonal 范围和均值。
"""
import json
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LPS = ["en-fa", "fa-en", "ja-en", "fr-en", "fi-en", "ko-en"]
BASE = Path(__file__).resolve().parent.parent / "results"


def load_peak_direction(lp, model):
    path = BASE / lp / model / "mechanistic" / "probing_balanced_strict.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    layers = data["results"]
    peak = max(layers, key=lambda r: r["balanced_subsample"]["balanced_accuracy"])
    return peak["layer"], np.array(peak["direction_balanced"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3.5-9B")
    args = parser.parse_args()

    print(f"=== {args.model} ===")
    peaks, dirs = {}, {}
    for lp in LPS:
        layer, w = load_peak_direction(lp, args.model)
        peaks[lp] = layer
        dirs[lp] = w
        print(f"  {lp}: peak L={layer:>2}, |w|={np.linalg.norm(w):.4f}")

    print("\nCosine matrix:")
    print("      " + "  ".join(f"{lp:>7}" for lp in LPS))
    cos_mat = np.zeros((6, 6))
    for i, lp1 in enumerate(LPS):
        row = []
        for j, lp2 in enumerate(LPS):
            c = float(np.dot(dirs[lp1], dirs[lp2]))
            cos_mat[i, j] = c
            row.append(f"{c:+.4f}")
        print(f"  {lp1:>5} " + "  ".join(f"{r:>7}" for r in row))

    off = []
    for i in range(6):
        for j in range(6):
            if i != j:
                off.append(cos_mat[i, j])
    off = np.array(off)
    print(f"\nOff-diagonal: min={off.min():+.4f}  max={off.max():+.4f}  "
          f"mean={off.mean():+.4f}  std={off.std():.4f}")

    # null band (random unit vectors in R^d, d = len(direction))
    d = len(dirs[LPS[0]])
    rng = np.random.RandomState(42)
    rand_vecs = rng.randn(10000, d)
    rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)
    rand_cos = (rand_vecs[:5000] * rand_vecs[5000:]).sum(axis=1)
    print(f"\nRandom null (d={d}): mean={rand_cos.mean():+.4f}  "
          f"std={rand_cos.std():.4f}  "
          f"|99% q|={np.percentile(np.abs(rand_cos), 99):.4f}")


if __name__ == "__main__":
    main()
