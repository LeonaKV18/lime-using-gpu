"""
surrogate_train.py (M2)
------------------
Trains a weighted linear regression surrogate model using PyTorch
on outputs produced by the M1 GPU pipeline.

The surrogate approximates the black-box model locally around x0 by fitting
a linear model to the perturbed samples, weighted by their proximity to x0.

Inputs (from M1 GPU run):
    X.bin          (B, D) float32  perturbed samples
    preds.bin      (B,)   float32  black-box predictions on perturbed samples
    weights.bin    (B,)   float32  similarity-based sample weights

Optional metadata:
    <dataset>.npz          x0, W, bias, feature_names from train_model.py

Outputs:
    attributions.npz  coeff, intercept, feature_names, x0,
                      surrogate_pred_x0, black_box_pred_x0,
                      r2_weighted, loss_curve

Usage:
    python scripts/surrogate_train.py
    python scripts/surrogate_train.py --X X.bin --preds preds.bin --weights weights.bin
    python scripts/surrogate_train.py --model models/breast_cancer.npz --epochs 500
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LinearSurrogate(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.linear = nn.Linear(D, 1, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    w      = weights / weights.sum()
    y_mean = float(np.dot(w, y_true))
    ss_res = float(np.dot(w, (y_true - y_pred) ** 2))
    ss_tot = float(np.dot(w, (y_true - y_mean) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def train_surrogate(
    X_np: np.ndarray,
    labels_np: np.ndarray,
    weights_np: np.ndarray,
    epochs: int = 300,
    lr: float = 0.01,
) -> tuple:
    """
    Trains a weighted linear regression surrogate via Adam.

    Features are z-scored before training so all dimensions contribute
    comparably to gradient updates. Coefficients are mapped back to the
    original feature scale before returning so attributions are interpretable.

    Returns:
        coeff      (D,) float32   feature attributions in original feature space
        intercept  float32        bias term in original feature space
        loss_curve (epochs,) float32
    """
    D = X_np.shape[1]

    feat_mean = X_np.mean(axis=0)
    feat_std  = X_np.std(axis=0) + 1e-8
    X_norm    = ((X_np - feat_mean) / feat_std).astype(np.float32)

    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(labels_np.astype(np.float32))
    w_t = torch.from_numpy((weights_np / weights_np.sum()).astype(np.float32))

    model     = LinearSurrogate(D)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_curve = np.empty(epochs, dtype=np.float32)
    for epoch in range(epochs):
        pred = model(X_t)
        loss = (w_t * (pred - y_t) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve[epoch] = loss.item()

    # Map coefficients from normalised space back to original feature space.
    coeff_norm    = model.linear.weight.data[0].numpy().copy().astype(np.float32)
    intercept_norm = float(model.linear.bias.data[0].item())

    coeff     = (coeff_norm / feat_std).astype(np.float32)
    intercept = np.float32(intercept_norm - float(np.dot(coeff_norm, feat_mean / feat_std)))

    return coeff, intercept, loss_curve


def main():
    parser = argparse.ArgumentParser(description="M2: train weighted linear surrogate")
    parser.add_argument("--X",       type=str,   default="X.bin",           help="Perturbed samples from M1")
    parser.add_argument("--preds",   type=str,   default="preds.bin",       help="Black-box predictions from M1")
    parser.add_argument("--weights", type=str,   default="weights.bin",     help="Similarity weights from M1")
    parser.add_argument("--model",   type=str,   default=None,              help="Model .npz from train_model.py")
    parser.add_argument("--D",       type=int,   default=None,              help="Feature dimension (inferred from model if given)")
    parser.add_argument("--B",       type=int,   default=16384,             help="Number of perturbed samples")
    parser.add_argument("--epochs",  type=int,   default=300,               help="Training epochs")
    parser.add_argument("--lr",      type=float, default=0.01,              help="Adam learning rate")
    parser.add_argument("--out",     type=str,   default="attributions.npz",help="Output .npz path")
    args = parser.parse_args()

    feature_names = None
    x0            = None
    W_bb          = None
    bias_bb       = None

    if args.model:
        md            = np.load(args.model, allow_pickle=True)
        x0            = md["x0"].astype(np.float32)
        W_bb          = md["W"].astype(np.float32)
        bias_bb       = float(md["bias"])
        feature_names = md["feature_names"].tolist() if "feature_names" in md else None
        D             = len(x0)
    elif args.D:
        D = args.D
    else:
        raise ValueError("Provide --model or --D to determine feature dimension.")

    B = args.B
    X       = np.fromfile(args.X,       dtype=np.float32).reshape(B, D)
    preds   = np.fromfile(args.preds,   dtype=np.float32)
    weights = np.fromfile(args.weights, dtype=np.float32)

    print(f"Loaded  X={X.shape}  preds={preds.shape}  weights={weights.shape}")
    print(f"preds  : min={preds.min():.4f}  max={preds.max():.4f}  mean={preds.mean():.4f}")
    print(f"weights: min={weights.min():.4f}  max={weights.max():.4f}  mean={weights.mean():.4f}")

    print(f"\nTraining surrogate  D={D}  B={B}  epochs={args.epochs}  lr={args.lr}")
    coeff, intercept, loss_curve = train_surrogate(
        X, preds, weights, epochs=args.epochs, lr=args.lr
    )
    print(f"Final loss: {loss_curve[-1]:.6f}")

    surr_preds = X @ coeff + intercept
    r2 = weighted_r2(preds, surr_preds, weights)
    print(f"Weighted R2: {r2:.4f}")

    surr_x0 = None
    bb_x0   = None
    if x0 is not None:
        surr_x0 = float(np.dot(x0, coeff) + intercept)
    if W_bb is not None and bias_bb is not None and x0 is not None:
        logit = float(np.dot(x0, W_bb) + bias_bb)
        bb_x0 = float(1.0 / (1.0 + np.exp(-logit)))

    if surr_x0 is not None and bb_x0 is not None:
        print(f"\nAt x0:  surrogate={surr_x0:.4f}  black_box={bb_x0:.4f}  "
              f"|delta|={abs(surr_x0 - bb_x0):.4f}")

    if feature_names:
        order = np.argsort(np.abs(coeff))[::-1]
        print("\nTop-10 features by |attribution|:")
        for rank, idx in enumerate(order[:10], 1):
            print(f"  {rank:2d}. {feature_names[idx]:<40s}  {coeff[idx]:+.6f}")

    payload = {
        "coeff":       coeff,
        "intercept":   np.float32(intercept),
        "loss_curve":  loss_curve,
        "r2_weighted": np.float32(r2),
    }
    if x0            is not None: payload["x0"]                = x0
    if feature_names is not None: payload["feature_names"]     = np.array(feature_names)
    if surr_x0       is not None: payload["surrogate_pred_x0"] = np.float32(surr_x0)
    if bb_x0         is not None: payload["black_box_pred_x0"] = np.float32(bb_x0)

    np.savez(args.out, **payload)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()