"""
cpu_surrogate.py  (Milestone 3 reference)
-----------------------------------------
Pure-NumPy reference implementation of the weighted linear-regression
surrogate.  Mirrors the GPU pipeline in src/m3_kernels.cu so the two
implementations can be cross-validated.

The numeric path is intentionally identical to the GPU one:

    1. z-score features            (per-feature mean / std over B)
    2. build A = X~^T diag(w) X~ + lambda I
       build b = X~^T diag(w) y
    3. Cholesky factorize A = L L^T, solve L L^T beta = b
    4. de-normalize beta to original feature space
    5. weighted R^2  +  surrogate prediction at x0

No PyTorch (or any deep-learning framework) is used — only NumPy.

Usage:
    python scripts/cpu_surrogate.py \
        --X build/X_m3.bin --preds build/preds_m3.bin --weights build/weights_m3.bin \
        --model models/breast_cancer.npz --solver cholesky --ridge 1e-3 \
        --out attributions_cpu.npz
"""

import argparse
import time
import numpy as np


# ---------------------------------------------------------------------------
#                       Numerical building blocks
# ---------------------------------------------------------------------------

def build_normal_equations(X_norm: np.ndarray,
                           y: np.ndarray,
                           w: np.ndarray,
                           ridge: float) -> tuple:
    """
    Build A = X~^T diag(w) X~ + ridge I  and  b = X~^T diag(w) y .

    X~ is the (B, D+1) augmented matrix whose first column is all ones.
    To save memory we don't materialize X~ — we accumulate the bias terms
    separately (sum of w; sum of w*y; etc.).
    """
    B, D = X_norm.shape
    N    = D + 1

    # Apply weights to features: X_w[k,j] = w[k] * X_norm[k,j]
    Xw = X_norm * w[:, None]

    A           = np.zeros((N, N), dtype=np.float64)
    A[1:, 1:]   = Xw.T @ X_norm                     # bottom-right block
    A[0, 1:]    = Xw.sum(axis=0)                    # top row (bias x feature)
    A[1:, 0]    = A[0, 1:]                          # symmetric
    A[0, 0]     = w.sum()                           # bias x bias = sum w

    b           = np.zeros(N, dtype=np.float64)
    b[0]        = (w * y).sum()
    b[1:]       = Xw.T @ y

    A          += ridge * np.eye(N)
    return A, b


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cholesky factorize A = L L^T and solve A x = b via two triangular solves.
    Implemented with NumPy primitives so the algorithm matches the GPU code
    exactly rather than delegating to a black-box LAPACK routine.
    """
    N = A.shape[0]
    L = np.zeros_like(A)
    for k in range(N):
        s = A[k, k] - (L[k, :k] ** 2).sum()
        L[k, k] = np.sqrt(max(s, 1e-12))
        for i in range(k + 1, N):
            L[i, k] = (A[i, k] - (L[i, :k] * L[k, :k]).sum()) / L[k, k]

    z = np.zeros(N)
    for i in range(N):
        z[i] = (b[i] - (L[i, :i] * z[:i]).sum()) / L[i, i]

    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (z[i] - (L[i + 1:, i] * x[i + 1:]).sum()) / L[i, i]
    return x


def gradient_descent_solve(X_norm: np.ndarray, y: np.ndarray, w: np.ndarray,
                           ridge: float, iters: int, lr: float) -> np.ndarray:
    """
    Iterative weighted ridge regression in the augmented space.

    The step size is rescaled by 1/sum(w) so the user-facing learning rate is
    insensitive to weight magnitude.  The fixed-point equation
    (X~^T W r + lambda*beta = 0) is unchanged, so this converges to the same
    minimizer as the closed-form solution.
    """
    B, D = X_norm.shape
    N    = D + 1
    beta = np.zeros(N, dtype=np.float64)
    sum_w  = float(w.sum())
    lr_eff = lr / max(sum_w, 1e-12)
    for _ in range(iters):
        # Augmented matvec: pred[k] = beta[0] + sum_j beta[j+1] * X_norm[k,j]
        pred = beta[0] + X_norm @ beta[1:]
        res  = pred - y
        wres = w * res
        grad = np.empty(N)
        grad[0]  = wres.sum() + ridge * beta[0]
        grad[1:] = X_norm.T @ wres + ridge * beta[1:]
        beta -= lr_eff * grad
    return beta


# ---------------------------------------------------------------------------
#                      End-to-end CPU surrogate trainer
# ---------------------------------------------------------------------------

def train_cpu_surrogate(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                        ridge: float = 1e-3,
                        solver: str = "cholesky",
                        gd_iters: int = 500,
                        gd_lr: float = 1e-2):
    """
    Returns (coeff, intercept, weighted_r2, elapsed_ms).
    """
    t0 = time.perf_counter()

    # 1. z-score normalize
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X_n  = ((X - mean) / std).astype(np.float64)

    # 2./3. solve
    if solver == "cholesky":
        A, b      = build_normal_equations(X_n, y.astype(np.float64),
                                           w.astype(np.float64), ridge)
        beta_norm = cholesky_solve(A, b)
    elif solver == "gd":
        beta_norm = gradient_descent_solve(X_n, y.astype(np.float64),
                                           w.astype(np.float64), ridge,
                                           gd_iters, gd_lr)
    else:
        raise ValueError(f"Unknown solver '{solver}'")

    # 4. de-normalize
    coeff     = (beta_norm[1:] / std).astype(np.float32)
    intercept = float(beta_norm[0] - np.dot(coeff, mean))

    # 5. weighted R^2
    pred   = X.astype(np.float64) @ coeff + intercept
    sw     = float(w.sum())
    sw_y   = float((w * y).sum())
    ss_tot = float((w * y * y).sum() - (sw_y * sw_y) / max(sw, 1e-12))
    ss_res = float((w * (y - pred) ** 2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return coeff, np.float32(intercept), float(r2), float(elapsed_ms)


# ---------------------------------------------------------------------------
#                                 Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="M3 CPU surrogate reference (NumPy, no PyTorch)")
    p.add_argument("--X",       type=str, default="X.bin")
    p.add_argument("--preds",   type=str, default="preds.bin")
    p.add_argument("--weights", type=str, default="weights.bin")
    p.add_argument("--model",   type=str, default=None,
                   help="model .npz from train_model.py (gives D, x0, feature_names)")
    p.add_argument("--D",       type=int, default=None)
    p.add_argument("--B",       type=int, default=16384)
    p.add_argument("--solver",  type=str, default="cholesky",
                   choices=["cholesky", "gd"])
    p.add_argument("--ridge",   type=float, default=1e-3)
    p.add_argument("--gd-iters", type=int,   default=500)
    p.add_argument("--gd-lr",    type=float, default=1e-2)
    p.add_argument("--out",     type=str, default="attributions_cpu.npz")
    args = p.parse_args()

    # Resolve D / x0 / feature names
    feature_names = None
    x0 = None
    W_bb, bias_bb = None, None

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
        raise ValueError("Provide --model or --D.")

    B = args.B

    X     = np.fromfile(args.X,       dtype=np.float32).reshape(B, D)
    preds = np.fromfile(args.preds,   dtype=np.float32)
    w     = np.fromfile(args.weights, dtype=np.float32)
    print(f"Loaded X={X.shape}  preds={preds.shape}  weights={w.shape}")

    coeff, intercept, r2, ms = train_cpu_surrogate(
        X, preds, w,
        ridge=args.ridge, solver=args.solver,
        gd_iters=args.gd_iters, gd_lr=args.gd_lr,
    )

    print(f"\nCPU solver={args.solver}  elapsed={ms:.2f} ms")
    print(f"Weighted R^2 = {r2:.4f}")

    surr_x0 = None
    bb_x0   = None
    if x0 is not None:
        surr_x0 = float(np.dot(x0, coeff) + intercept)
    if W_bb is not None and bias_bb is not None and x0 is not None:
        logit = float(np.dot(x0, W_bb) + bias_bb)
        bb_x0 = float(1.0 / (1.0 + np.exp(-logit)))
    if surr_x0 is not None and bb_x0 is not None:
        print(f"At x0:  surrogate={surr_x0:.4f}  black_box={bb_x0:.4f}  "
              f"|delta|={abs(surr_x0 - bb_x0):.4f}")

    if feature_names:
        order = np.argsort(np.abs(coeff))[::-1]
        print("\nTop-10 features by |coeff|:")
        for r, i in enumerate(order[:10], 1):
            print(f"  {r:2d}. {feature_names[i]:<40s}  {coeff[i]:+.6f}")

    payload = {
        "coeff":       coeff,
        "intercept":   np.float32(intercept),
        "r2_weighted": np.float32(r2),
        "solver":      np.array(args.solver),
        "ridge":       np.float32(args.ridge),
        "elapsed_ms":  np.float32(ms),
    }
    if x0            is not None: payload["x0"]                = x0
    if feature_names is not None: payload["feature_names"]     = np.array(feature_names)
    if surr_x0       is not None: payload["surrogate_pred_x0"] = np.float32(surr_x0)
    if bb_x0         is not None: payload["black_box_pred_x0"] = np.float32(bb_x0)

    np.savez(args.out, **payload)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
