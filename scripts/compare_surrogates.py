"""
compare_surrogates.py
---------------------
Side-by-side comparison of the three LIME surrogates produced in this
project:

    * CPU-only        - scripts/cpu_surrogate.py    (NumPy)
    * Framework (M2)  - scripts/surrogate_train.py  (PyTorch, gradient method)
    * GPU-native (M3) - lime_m3                     (CUDA, Cholesky / GD)

For each pair we report:
    * cosine similarity of coefficient vectors
    * top-K feature overlap and Spearman correlation
    * weighted R^2  (against the same X / preds / weights)
    * surrogate prediction at x0 (and delta vs black-box)

The script is read-only: it consumes pre-computed .npz files and never
trains anything itself.

Usage:
    python scripts/compare_surrogates.py \
        --m2  attributions.npz \
        --m3  attributions_gpu.npz \
        --cpu attributions_cpu.npz \
        --top-k 10
"""

import argparse
import numpy as np


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na * nb > 1e-12 else 0.0


def spearman(a, b):
    """Rank correlation, matching scipy.stats.spearmanr without the import."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    n  = len(a)
    if n < 2:
        return 0.0
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 1e-12 else 0.0


def topk_overlap(a, b, k):
    ia = set(np.argsort(np.abs(a))[::-1][:k].tolist())
    ib = set(np.argsort(np.abs(b))[::-1][:k].tolist())
    return len(ia & ib) / max(k, 1)


def load(path):
    if path is None: return None
    d = np.load(path, allow_pickle=True)
    out = {"coeff": d["coeff"].astype(np.float32),
           "intercept": float(d["intercept"]),
           "r2": float(d["r2_weighted"]) if "r2_weighted" in d.files else None,
           "surr_x0": float(d["surrogate_pred_x0"]) if "surrogate_pred_x0" in d.files else None,
           "bb_x0":   float(d["black_box_pred_x0"]) if "black_box_pred_x0" in d.files else None,
           "feature_names": d["feature_names"].tolist() if "feature_names" in d.files else None}
    return out


def report(name_a, a, name_b, b, top_k):
    print(f"  {name_a} vs {name_b}")
    print(f"    cos(coeff)            = {cosine(a['coeff'], b['coeff']):.4f}")
    print(f"    Spearman(|coeff|)     = {spearman(np.abs(a['coeff']), np.abs(b['coeff'])):.4f}")
    print(f"    Top-{top_k} overlap        = {topk_overlap(a['coeff'], b['coeff'], top_k):.0%}")
    print(f"    max|delta coeff|      = {np.max(np.abs(a['coeff']-b['coeff'])):.4e}")
    print(f"    intercept delta       = {abs(a['intercept']-b['intercept']):.4e}")
    if a["r2"] is not None and b["r2"] is not None:
        print(f"    R^2: {name_a}={a['r2']:.4f}  {name_b}={b['r2']:.4f}  "
              f"delta={abs(a['r2']-b['r2']):.4e}")
    if a["surr_x0"] is not None and b["surr_x0"] is not None:
        print(f"    f(x0): {name_a}={a['surr_x0']:.4f}  {name_b}={b['surr_x0']:.4f}  "
              f"delta={abs(a['surr_x0']-b['surr_x0']):.4e}")


def print_top_table(sources, top_k):
    """Pretty side-by-side table of top-K features by |coefficient|."""
    primary = next((s for s in sources.values() if s is not None), None)
    if primary is None: return

    fn = primary["feature_names"]
    if fn is None:
        print("  (No feature_names available — skipping side-by-side table.)")
        return

    order = np.argsort(np.abs(primary["coeff"]))[::-1][:top_k]
    headers = list(sources.keys())
    print(f"\n  Top-{top_k} features (rank by primary):")
    print(f"    {'#':>2}  {'feature':<40s}", end="")
    for h in headers:
        print(f"  {h:>14s}", end="")
    print()
    for r, idx in enumerate(order, 1):
        print(f"    {r:>2}  {fn[idx][:40]:<40s}", end="")
        for h in headers:
            s = sources[h]
            if s is None:
                print(f"  {'-':>14s}", end="")
            else:
                print(f"  {s['coeff'][idx]:+.6f}".rjust(16), end="")
        print()


def main():
    p = argparse.ArgumentParser(description="Compare M2 vs M3 vs CPU surrogates")
    p.add_argument("--m2",  type=str, default=None, help="attributions.npz from surrogate_train.py")
    p.add_argument("--m3",  type=str, default=None, help="attributions_gpu.npz from gpu_surrogate_to_npz.py")
    p.add_argument("--cpu", type=str, default=None, help="attributions_cpu.npz from cpu_surrogate.py")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    sources = {
        "M2 (PyTorch)": load(args.m2),
        "M3 (GPU)":     load(args.m3),
        "CPU (NumPy)":  load(args.cpu),
    }
    sources = {k: v for k, v in sources.items() if v is not None}
    if len(sources) < 2:
        print("Need at least two sources to compare. Provide --m2, --m3 and/or --cpu.")
        return

    print("\n=========================================================")
    print(" Pairwise comparison")
    print("=========================================================")
    keys = list(sources.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            report(keys[i], sources[keys[i]], keys[j], sources[keys[j]], args.top_k)
            print()

    print("=========================================================")
    print(" Per-source summary")
    print("=========================================================")
    for k, s in sources.items():
        msg = f"  {k:<14s}  intercept={s['intercept']:+.4f}"
        if s["r2"]      is not None: msg += f"  R^2={s['r2']:.4f}"
        if s["surr_x0"] is not None: msg += f"  f(x0)={s['surr_x0']:.4f}"
        if s["bb_x0"]   is not None: msg += f"  bb(x0)={s['bb_x0']:.4f}"
        print(msg)

    print_top_table(sources, args.top_k)


if __name__ == "__main__":
    main()
