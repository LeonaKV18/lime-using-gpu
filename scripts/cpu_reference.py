"""
cpu_reference.py
----------------
CPU reference implementation of the full LIME pipeline.
Mirrors the three CUDA kernels in kernels.cu exactly:
  1. generate_perturbations  →  cpu_generate_perturbations()
  2. cuBLAS Sgemv + sigmoid  →  cpu_infer()
  3. distances_and_weights   →  cpu_distances_and_weights()

Usage:
    python cpu_reference.py                     # defaults: D=128, B=16384
    python cpu_reference.py --D=256 --B=4096
    python cpu_reference.py --read-X X.bin --D=128 --B=16384
"""

import argparse
import time
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_model_npz(path):
    """
    Loads model parameters exported by train_model.py.
    Returns (x0, means, W, bias) all as float32.
    """
    data = np.load(path, allow_pickle=True)
    x0    = data["x0"].astype(np.float32)
    means = data["means"].astype(np.float32)
    W     = data["W"].astype(np.float32)
    bias  = np.float32(data["bias"])
    return x0, means, W, bias

def make_params(D, model_path=None):
    """
    Returns (x0, means, W, bias) for the given configuration.
    When model_path is provided the values come from a trained model;
    otherwise a synthetic fallback is used for benchmarking.
    """
    if model_path is not None:
        return load_model_npz(model_path)
    x0    = np.array([1.0 if i % 5 == 0 else 0.5 for i in range(D)], dtype=np.float32)
    means = np.full(D, 0.5, dtype=np.float32)
    W     = np.array([0.02 * (i + 1) for i in range(D)], dtype=np.float32)
    bias  = np.float32(-1.0)
    return x0, means, W, bias


# ── Stage 1: Perturbation generation ─────────────────────────────────────────

def cpu_generate_perturbations(x0, means, B, mask_prob=0.2, noise_std=0.1, seed=1234):
    """
    Mirrors generate_perturbations kernel (kernels.cu).

    Grid: B blocks x D threads — one block per sample, one thread per feature.
    For each (sample, feature):
        u ~ Uniform(0,1)
        if u < mask_prob  →  X = means[feat]            (feature masked out)
        else              →  X = x0[feat] + ns * N(0,1) (Gaussian perturbation)
    """
    rng   = np.random.default_rng(seed)
    D     = len(x0)
    u     = rng.random((B, D)).astype(np.float32)
    noise = rng.standard_normal((B, D)).astype(np.float32)
    X     = np.where(u < mask_prob,
                     means[np.newaxis, :],
                     x0[np.newaxis, :] + noise_std * noise)
    return X.astype(np.float32)


# ── Stage 2: Logistic regression inference ───────────────────────────────────

def cpu_infer(X, W, bias):
    """
    Mirrors cuBLAS Sgemv + add_bias kernel + apply_sigmoid kernel (main.cu).

    cuBLAS computes:  logit = X @ W          (Sgemv with OP_T trick)
    add_bias:         logit = logit + bias
    apply_sigmoid:    pred  = 1 / (1 + exp(-logit))
    """
    logit = (X @ W) + bias                             # shape (B,)
    pred  = (1.0 / (1.0 + np.exp(-logit.astype(np.float64)))).astype(np.float32)
    return pred


# ── Stage 3: Distances and Gaussian weights ───────────────────────────────────

def cpu_distances_and_weights(X, x0, kw=1.0):
    """
    Mirrors distances_and_weights kernel (kernels.cu).

    Each thread (one per sample) computes:
        dist[i] = sum_j ( X[i,j] - x0[j] )^2    (squared L2 distance)
        w[i]    = exp( -dist[i] / kw^2 )          (Gaussian kernel weight)

    Samples close to x0 → weight ≈ 1  (high influence on LIME fit)
    Samples far from x0 → weight ≈ 0  (low influence on LIME fit)
    """
    diff = (X - x0[np.newaxis, :]).astype(np.float64)
    dist = np.sum(diff ** 2, axis=1).astype(np.float32)
    w    = np.exp(-dist / (kw ** 2)).astype(np.float32)
    return dist, w


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(D, B, mask_prob=0.2, noise_std=0.1, kw=1.0, X_in=None, model_path=None):
    """
    Run all three stages and print timings + summary statistics.
    If X_in is provided, skip generation (useful for validation against GPU).
    """
    x0, means, W, bias = make_params(D, model_path=model_path)
    D = len(W)

    # Stage 1
    t0 = time.perf_counter()
    X  = cpu_generate_perturbations(x0, means, B, mask_prob, noise_std) \
         if X_in is None else X_in
    t_gen = (time.perf_counter() - t0) * 1e3

    # Stage 2
    t0    = time.perf_counter()
    preds = cpu_infer(X, W, bias)
    t_inf = (time.perf_counter() - t0) * 1e3

    # Stage 3
    t0           = time.perf_counter()
    dist, weights = cpu_distances_and_weights(X, x0, kw)
    t_wei        = (time.perf_counter() - t0) * 1e3

    t_total = t_gen + t_inf + t_wei
    print(f"Timing (ms):  gen {t_gen:.3f}  infer {t_inf:.3f}  "
          f"weights {t_wei:.3f}  total {t_total:.3f}")
    print(f"Means:  preds {preds.mean():.6f}  weights {weights.mean():.6f}")

    return preds, weights, X


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPU reference for lime_gpu")
    parser.add_argument("--D",       type=int,   default=128,   help="Number of features")
    parser.add_argument("--B",       type=int,   default=16384, help="Number of samples")
    parser.add_argument("--mask",    type=float, default=0.2,   help="Mask probability")
    parser.add_argument("--ns",      type=float, default=0.1,   help="Noise std dev")
    parser.add_argument("--kw",      type=float, default=1.0,   help="Kernel width")
    parser.add_argument("--read-X",  type=str,   default=None,  help="Load X from this .bin file")
    parser.add_argument("--out-preds",   type=str, default=None, help="Save preds to .bin")
    parser.add_argument("--out-weights", type=str, default=None, help="Save weights to .bin")
    parser.add_argument("--model",       type=str,   default=None,  help="Model .npz from train_model.py")
    args = parser.parse_args()

    D = args.D
    if args.model:
        _md = np.load(args.model, allow_pickle=True)
        D   = int(len(_md["W"]))

    # Optionally load a pre-generated X (e.g. the one the GPU used)
    X_in = None
    if args.read_X:
        X_in = np.fromfile(args.read_X, dtype=np.float32).reshape(args.B, args.D)
        print(f"Loaded X from {args.read_X}  shape={X_in.shape}")

    print(f"Running CPU reference  D={args.D}  B={args.B}")
    preds, weights, _ = run_pipeline(D, args.B, args.mask, args.ns, args.kw, X_in, model_path=args.model)

    if args.out_preds:
        preds.tofile(args.out_preds)
        print(f"Saved preds   → {args.out_preds}")
    if args.out_weights:
        weights.tofile(args.out_weights)
        print(f"Saved weights → {args.out_weights}")


if __name__ == "__main__":
    main()
