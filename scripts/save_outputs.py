"""
save_outputs.py
---------------
Packages raw binary outputs into a single .npz file for easy inspection.

Loads X.bin, preds.bin, weights.bin (and optional zprime.bin) and saves
them as a single NumPy .npz
so results can be easily loaded and analysed in Python without re-running
the GPU binary.

Usage:
    python save_outputs.py                          # defaults D=128 B=16384
    python save_outputs.py --D=256 --B=4096
    python save_outputs.py --D=128 --B=16384 --out=results.npz
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Package binary outputs into .npz")
    parser.add_argument("--D",       type=int, default=128,          help="Number of features")
    parser.add_argument("--B",       type=int, default=16384,        help="Number of samples")
    parser.add_argument("--X",       type=str, default="X.bin",      help="Path to X.bin")
    parser.add_argument("--preds",   type=str, default="preds.bin",  help="Path to preds.bin")
    parser.add_argument("--weights", type=str, default="weights.bin",help="Path to weights.bin")
    parser.add_argument("--zprime",  type=str, default=None,         help="Path to zprime.bin (optional)")
    parser.add_argument("--out",     type=str, default="outputs.npz",help="Output .npz path")
    args = parser.parse_args()

    print(f"Loading binary files  D={args.D}  B={args.B}")
    X       = np.fromfile(args.X,       dtype=np.float32).reshape(args.B, args.D)
    preds   = np.fromfile(args.preds,   dtype=np.float32)
    weights = np.fromfile(args.weights, dtype=np.float32)

    payload = {"X": X, "preds": preds, "weights": weights}
    if args.zprime:
        zprime = np.fromfile(args.zprime, dtype=np.uint8).reshape(args.B, args.D)
        payload["zprime"] = zprime

    np.savez(args.out, **payload)

    print(f"Saved -> {args.out}")
    print(f"  X        shape={X.shape}       min={X.min():.4f}  max={X.max():.4f}  mean={X.mean():.4f}")
    print(f"  preds    shape={preds.shape}   min={preds.min():.4f}  max={preds.max():.4f}  mean={preds.mean():.4f}")
    print(f"  weights  shape={weights.shape} min={weights.min():.4f}  max={weights.max():.4f}  mean={weights.mean():.4f}")
    if args.zprime:
        print(f"  zprime   shape={zprime.shape}  zeros={(zprime == 0).sum()}  ones={(zprime == 1).sum()}")
    print(f"\nTo load later:  data = np.load('{args.out}')")


if __name__ == "__main__":
    main()
