"""
explain_instance.py (M2)
-------------------
Generates explanation plots and validation output from the
trained surrogate model produced by surrogate_train.py.

Outputs:
    feature_importance.png   horizontal bar chart of top-K attributions
    loss_curve.png           surrogate training convergence
    ablation_validation.png  progressive feature ablation curve (optional)

The ablation plot requires --X, --preds, and --weights.  Without them the
first two plots are still produced and a consistency check is printed.

Usage:
    python scripts/explain_instance.py
    python scripts/explain_instance.py --attributions attributions.npz --top-k 20
    python scripts/explain_instance.py --attributions attributions.npz \
        --X X.bin --preds preds.bin --weights weights.bin --top-k 15
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(coeff, feature_names, top_k, out_path):
    """
    Horizontal bar chart of the top_k features ranked by |coefficient|.
    Positive attributions are green, negative are red.
    """
    order = np.argsort(np.abs(coeff))[-top_k:]
    vals  = coeff[order]
    names = (
        [feature_names[i] for i in order]
        if feature_names is not None
        else [f"feature_{i}" for i in order]
    )
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.38)))
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Attribution coefficient")
    ax.set_title(f"LIME Feature Attributions  (top {top_k} by |coefficient|)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_loss_curve(loss_curve, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_curve, color="#1f77b4", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE loss")
    ax.set_title("Surrogate Training Convergence")
    ax.set_yscale("log")
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_ablation(coeff, feature_names, X, preds, weights, out_path, max_features=20):
    """
    Progressive feature ablation: replaces the highest-|coefficient| features
    one at a time with their column mean, then measures the weighted MAE between
    the ablated surrogate predictions and the original black-box predictions.
    A rising curve confirms that ablated features are informative.
    """
    order     = np.argsort(np.abs(coeff))[::-1]
    col_means = X.mean(axis=0)
    n         = min(max_features, len(coeff))
    w_norm    = weights / weights.sum()

    mae_values = []
    X_ab       = X.copy()

    for k in range(n + 1):
        surr_pred = X_ab @ coeff
        wmae      = float(np.dot(w_norm, np.abs(surr_pred - preds)))
        mae_values.append(wmae)
        if k < n:
            X_ab[:, order[k]] = col_means[order[k]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(n + 1), mae_values, marker="o", color="#ff7f0e", linewidth=1.5,
            markersize=4)
    ax.set_xlabel("Number of top features ablated")
    ax.set_ylabel("Weighted MAE  (surrogate vs black-box)")
    ax.set_title("Feature Ablation Validation")
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="M2: explanation generation and validation")
    parser.add_argument("--attributions", type=str, default="attributions.npz",
                        help="Path to attributions.npz from surrogate_train.py")
    parser.add_argument("--X",       type=str, default=None, help="X.bin (enables ablation plot)")
    parser.add_argument("--preds",   type=str, default=None, help="preds.bin (enables ablation plot)")
    parser.add_argument("--weights", type=str, default=None, help="weights.bin (enables ablation plot)")
    parser.add_argument("--top-k",   type=int, default=15,   help="Features shown in importance chart")
    parser.add_argument("--out",     type=str, default=".",  help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data         = np.load(args.attributions, allow_pickle=True)
    coeff        = data["coeff"]
    intercept    = float(data["intercept"])
    loss_curve   = data["loss_curve"]
    r2           = float(data["r2_weighted"])
    feature_names = data["feature_names"].tolist() if "feature_names" in data else None
    x0           = data["x0"]           if "x0"           in data else None
    surr_x0      = float(data["surrogate_pred_x0"]) if "surrogate_pred_x0" in data else None
    bb_x0        = float(data["black_box_pred_x0"])  if "black_box_pred_x0"  in data else None

    D = len(coeff)

    print(f"\nLoaded attributions  D={D}  weighted R2={r2:.4f}")
    if surr_x0 is not None and bb_x0 is not None:
        delta  = abs(surr_x0 - bb_x0)
        status = "PASS" if delta < 0.1 else "WARN"
        print(f"Consistency at x0 : surrogate={surr_x0:.4f}  black_box={bb_x0:.4f}  "
              f"|delta|={delta:.4f}  [{status}]")

    order = np.argsort(np.abs(coeff))[::-1]
    print(f"\nTop-{min(args.top_k, D)} features:")
    for rank, idx in enumerate(order[:args.top_k], 1):
        name = feature_names[idx] if feature_names else f"feature_{idx}"
        print(f"  {rank:2d}. {name:<40s}  {coeff[idx]:+.6f}")

    print("\nGenerating plots...")

    plot_feature_importance(
        coeff, feature_names, min(args.top_k, D),
        os.path.join(args.out, "feature_importance.png"),
    )
    plot_loss_curve(
        loss_curve,
        os.path.join(args.out, "loss_curve.png"),
    )

    ablation_available = args.X and args.preds and args.weights
    if ablation_available:
        X       = np.fromfile(args.X,       dtype=np.float32).reshape(-1, D)
        preds   = np.fromfile(args.preds,   dtype=np.float32)
        weights = np.fromfile(args.weights, dtype=np.float32)
        plot_ablation(
            coeff, feature_names, X, preds, weights,
            os.path.join(args.out, "ablation_validation.png"),
        )
    else:
        print("  [INFO] Ablation plot skipped (pass --X --preds --weights to enable)")

    print(f"\nWeighted R2: {r2:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()