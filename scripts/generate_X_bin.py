"""
generate_X_bin.py
-----------------
Generates a deterministic X.bin file (B x D float32 matrix)
that can be fed to lime_gpu via:  ./lime_gpu --D=128 --B=16384 --read-X X.bin

Usage:
    python generate_X_bin.py                        # defaults: D=128, B=16384
    python generate_X_bin.py --D=256 --B=4096
    python generate_X_bin.py --D=128 --B=16384 --out=my_X.bin
"""

import argparse
import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model_npz(path):
    """Loads x0 and means from a model .npz file produced by train_model.py."""
    data = np.load(path, allow_pickle=True)
    return data["x0"].astype(np.float32), data["means"].astype(np.float32)

def make_params(D, model_path=None):
    """
    Returns (x0, means) for perturbation generation.
    When model_path is provided the values come from a trained model;
    otherwise a synthetic fallback is used.
    """
    if model_path is not None:
        return load_model_npz(model_path)
    x0    = np.array([1.0 if i % 5 == 0 else 0.5 for i in range(D)], dtype=np.float32)
    means = np.full(D, 0.5, dtype=np.float32)
    return x0, means


def generate_X(D, B, mask_prob=0.2, noise_std=0.1, seed=42, model_path=None):
    """
    CPU mirror of the generate_perturbations CUDA kernel.

    For every (sample, feature) pair:
        u ~ Uniform(0, 1)
        if u < mask_prob  →  X[sample, feature] = means[feature]
        else              →  X[sample, feature] = x0[feature] + noise_std * N(0,1)

    Parameters
    ----------
    D          : number of features
    B          : number of perturbed samples
    mask_prob  : probability of masking a feature (replacing with its mean)
    noise_std  : standard deviation of the Gaussian perturbation
    seed       : random seed for reproducibility

    Returns
    -------
    X : np.ndarray, shape (B, D), dtype float32
    """
    x0, means = make_params(D, model_path=model_path)
    D = len(x0)
    rng   = np.random.default_rng(seed)
    u     = rng.random((B, D)).astype(np.float32)
    noise = rng.standard_normal((B, D)).astype(np.float32)
    X     = np.where(u < mask_prob,
                     means[np.newaxis, :],
                     x0[np.newaxis, :] + noise_std * noise)
    return X.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate deterministic X.bin for lime_gpu")
    parser.add_argument("--D",    type=int,   default=128,     help="Number of features")
    parser.add_argument("--B",    type=int,   default=16384,   help="Number of samples")
    parser.add_argument("--mask", type=float, default=0.2,     help="Mask probability")
    parser.add_argument("--ns",   type=float, default=0.1,     help="Noise std dev")
    parser.add_argument("--seed", type=int,   default=42,      help="Random seed")
    parser.add_argument("--out",  type=str,   default="X.bin", help="Output file path")
    parser.add_argument("--model",type=str,   default=None,    help="Model .npz from train_model.py; sets x0 and means")
    args = parser.parse_args()

    print(f"Generating X  D={args.D}  B={args.B}  mask_prob={args.mask}  "
          f"noise_std={args.ns}  seed={args.seed}")
   
    D = args.D
    if args.model:
        _md = np.load(args.model, allow_pickle=True)
        D   = int(len(_md["x0"]))

    X = generate_X(D, args.B, args.mask, args.ns, args.seed, model_path=args.model)

    X.tofile(args.out)
    print(f"Saved {args.B} x {args.D} matrix  →  {args.out}")
    print(f"File size: {X.nbytes / 1024:.1f} KB")
    print(f"X stats:  min={X.min():.4f}  max={X.max():.4f}  mean={X.mean():.4f}")


if __name__ == "__main__":
    main()
