"""
compare_and_validate.py
-----------------------
Validates GPU output (preds.bin, weights.bin, optional zprime.bin)
against the CPU reference.

Workflow:
    1. Run lime_gpu with --write-X X.bin --write-zprime zprime.bin
  2. Run this script — it loads X.bin, runs CPU math on it, and compares

Usage:
    # Basic (defaults D=128, B=16384)
    python compare_and_validate.py

    # Custom sizes
    python compare_and_validate.py --D=256 --B=4096

    # Point to specific files
    python compare_and_validate.py --X X.bin --preds preds.bin --weights weights.bin
    python compare_and_validate.py --X X.bin --zprime zprime.bin --tol_abs 1e-4 --tol_rel 1e-3

Expected GPU run before this:
    ./lime_gpu --D=128 --B=16384 --write-X X.bin --write-zprime zprime.bin
"""

import argparse
import sys
import numpy as np


# ── Import CPU reference functions ───────────────────────────────────────────
# Keep cpu_reference.py in the same folder as this script.
import importlib.util, pathlib

def _load_cpu_ref():
    here  = pathlib.Path(__file__).parent
    spec  = importlib.util.spec_from_file_location("cpu_reference", here / "cpu_reference.py")
    mod   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ref = _load_cpu_ref()


# ── Validation logic ─────────────────────────────────────────────────────────

def validate(D, B, X_path, preds_path, weights_path,
             zprime_path=None, tol_abs=1e-4, tol_rel=1e-3):

    print(f"\n{'='*60}")
    print(f"  Validating  D={D}  B={B}")
    print(f"{'='*60}")

    # 1. Load GPU's X matrix
    try:
        X_gpu = np.fromfile(X_path, dtype=np.float32).reshape(B, D)
    except Exception as e:
        print(f"[ERROR] Could not load {X_path}: {e}")
        print("  → Make sure you ran:  ./lime_gpu --write-X X.bin")
        sys.exit(1)

    zprime_gpu = None
    if zprime_path:
        try:
            zprime_gpu = np.fromfile(zprime_path, dtype=np.uint8).reshape(B, D)
        except Exception as e:
            print(f"[ERROR] Could not load {zprime_path}: {e}")
            sys.exit(1)

    # 2. Load GPU outputs
    try:
        preds_gpu   = np.fromfile(preds_path,   dtype=np.float32)
        weights_gpu = np.fromfile(weights_path, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Could not load output files: {e}")
        sys.exit(1)

    if preds_gpu.shape[0] != B or weights_gpu.shape[0] != B:
        print(f"[ERROR] Expected B={B} values, got {preds_gpu.shape[0]} preds "
              f"and {weights_gpu.shape[0]} weights. Did --B match?")
        sys.exit(1)

    # 3. Run CPU on the exact same X
    x0, _, W, bias = ref.make_params(D)
    preds_cpu         = ref.cpu_infer(X_gpu, W, bias)
    _, weights_cpu    = ref.cpu_distances_and_weights(X_gpu, x0)

    # 4. Compute differences
    diff_p = np.abs(preds_gpu   - preds_cpu)
    diff_w = np.abs(weights_gpu - weights_cpu)

    # 5. Print report
    def report(name, ref, diff):
        rel = diff / (np.abs(ref) + 1e-12)
        pass_mask = diff <= (tol_abs + tol_rel * np.abs(ref))
        status = "✓ PASS" if np.all(pass_mask) else "✗ FAIL"
        print(f"  {name:<10}  max|Δ|={diff.max():.3e}   "
              f"mean|Δ|={diff.mean():.3e}   "
              f"max|rel|={rel.max():.3e}   {status}  "
              f"(abs={tol_abs:.0e}, rel={tol_rel:.0e})")
        return np.all(pass_mask)

    print(f"\n  GPU preds   mean={preds_gpu.mean():.6f}   "
          f"CPU preds   mean={preds_cpu.mean():.6f}")
    print(f"  GPU weights mean={weights_gpu.mean():.6f}   "
          f"CPU weights mean={weights_cpu.mean():.6f}")
    print()

    ok_p = report("preds", preds_cpu, diff_p)
    ok_w = report("weights", weights_cpu, diff_w)

    ok_z = True
    if zprime_gpu is not None:
        _, means, _, _ = ref.make_params(D)
        z_from_x = np.where(np.isclose(X_gpu, means[np.newaxis, :], atol=1e-7), 0, 1).astype(np.uint8)
        bad_values = np.count_nonzero((zprime_gpu != 0) & (zprime_gpu != 1))
        mismatches = np.count_nonzero(zprime_gpu != z_from_x)
        total = B * D
        ok_z = (bad_values == 0) and (mismatches == 0)
        status = "✓ PASS" if ok_z else "✗ FAIL"
        print(f"  zprime      mismatches={mismatches}/{total}   bad_values={bad_values}   {status}")

    # 6. Histogram of differences (text-based, no matplotlib needed)
    print(f"\n  Difference histogram (preds):")
    _text_histogram(diff_p)
    print(f"\n  Difference histogram (weights):")
    _text_histogram(diff_w)

    print()
    if ok_p and ok_w and ok_z:
        print("  ✓ ALL CHECKS PASSED — GPU and CPU outputs match.\n")
    else:
        print("  ✗ VALIDATION FAILED — see differences above.\n")
        sys.exit(1)

    return ok_p and ok_w and ok_z


def _text_histogram(arr, bins=8, width=40):
    """Print a simple text-based histogram — no matplotlib required."""
    counts, edges = np.histogram(arr, bins=bins)
    max_count     = counts.max()
    for i, (lo, hi, cnt) in enumerate(zip(edges, edges[1:], counts)):
        bar = '█' * int(width * cnt / max_count) if max_count > 0 else ''
        print(f"    [{lo:.1e}, {hi:.1e})  {bar}  {cnt}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare lime_gpu outputs against CPU reference")
    parser.add_argument("--D",       type=int,   default=128,          help="Number of features")
    parser.add_argument("--B",       type=int,   default=16384,        help="Number of samples")
    parser.add_argument("--X",       type=str,   default="X.bin",      help="Path to X.bin (GPU-generated)")
    parser.add_argument("--preds",   type=str,   default="preds.bin",  help="Path to preds.bin")
    parser.add_argument("--weights", type=str,   default="weights.bin",help="Path to weights.bin")
    parser.add_argument("--zprime",  type=str,   default=None,         help="Path to zprime.bin (optional)")
    parser.add_argument("--tol",     type=float, default=None,         help="Legacy absolute tolerance (sets rel=0)")
    parser.add_argument("--tol_abs", type=float, default=1e-4,         help="Absolute tolerance")
    parser.add_argument("--tol_rel", type=float, default=1e-3,         help="Relative tolerance")
    args = parser.parse_args()

    tol_abs = args.tol_abs
    tol_rel = args.tol_rel
    if args.tol is not None:
        tol_abs = args.tol
        tol_rel = 0.0

    validate(
        D=args.D,
        B=args.B,
        X_path=args.X,
        preds_path=args.preds,
        weights_path=args.weights,
        zprime_path=args.zprime,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
    )


if __name__ == "__main__":
    main()
