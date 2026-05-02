"""
gpu_surrogate_to_npz.py
-----------------------
Converts the binary attribution file produced by lime_m3 into the same
NPZ format used by surrogate_train.py (M2).  This makes the existing
explain_instance.py plotting script work with M3 outputs without any
modification.

main_m3.cu uses double-precision for numerical stability and falls 
back to GD if Cholesky fails. The solver_id in the output reflects 
the final solver used (0=cholesky success, 1=gd fallback or direct GD).

Binary layout (see save_attributions in src/main_m3.cu):
    int32    D
    int32    solver_id        0 = cholesky, 1 = gd
    float32  ridge
    float32  coeff[D]
    float32  intercept
    float32  surrogate_pred_x0
    float32  black_box_pred_x0
    float32  weighted_r2
    float32  total_ms
    int32    num_iters
    int32    B

Usage:
    python scripts/gpu_surrogate_to_npz.py \
        --in build/attributions_gpu.bin \
        --model models/breast_cancer.npz \
        --out attributions_gpu.npz
"""

import argparse
import numpy as np
import struct


def read_attributions_bin(path):
    with open(path, "rb") as f:
        D          = struct.unpack("<i", f.read(4))[0]
        solver_id  = struct.unpack("<i", f.read(4))[0]
        ridge      = struct.unpack("<f", f.read(4))[0]
        coeff      = np.frombuffer(f.read(D * 4), dtype=np.float32).copy()
        intercept  = struct.unpack("<f", f.read(4))[0]
        surr_x0    = struct.unpack("<f", f.read(4))[0]
        bb_x0      = struct.unpack("<f", f.read(4))[0]
        r2         = struct.unpack("<f", f.read(4))[0]
        total_ms   = struct.unpack("<f", f.read(4))[0]
        num_iters  = struct.unpack("<i", f.read(4))[0]
        B          = struct.unpack("<i", f.read(4))[0]
    return {
        "D": D,
        "solver_id": solver_id,
        "solver": "cholesky" if solver_id == 0 else "gd",
        "ridge": ridge,
        "coeff": coeff,
        "intercept": intercept,
        "surrogate_pred_x0": surr_x0,
        "black_box_pred_x0": bb_x0,
        "r2_weighted": r2,
        "total_ms": total_ms,
        "num_iters": num_iters,
        "B": B,
    }


def main():
    p = argparse.ArgumentParser(description="Convert lime_m3 binary -> .npz")
    p.add_argument("--in", dest="inp", type=str, default="attributions_gpu.bin")
    p.add_argument("--model", type=str, default=None,
                   help="model .npz from train_model.py — adds x0 / feature_names")
    p.add_argument("--out", type=str, default="attributions_gpu.npz")
    args = p.parse_args()

    a = read_attributions_bin(args.inp)
    print(f"Loaded {args.inp}  D={a['D']}  solver={a['solver']}  "
          f"ridge={a['ridge']:.2e}  R2={a['r2_weighted']:.4f}  "
          f"train={a['total_ms']:.3f} ms")
    print(f"At x0:  surrogate={a['surrogate_pred_x0']:.4f}  "
          f"black_box={a['black_box_pred_x0']:.4f}  "
          f"|delta|={abs(a['surrogate_pred_x0']-a['black_box_pred_x0']):.4f}")

    payload = {
        "coeff":              a["coeff"],
        "intercept":          np.float32(a["intercept"]),
        "r2_weighted":        np.float32(a["r2_weighted"]),
        "surrogate_pred_x0":  np.float32(a["surrogate_pred_x0"]),
        "black_box_pred_x0":  np.float32(a["black_box_pred_x0"]),
        "solver":             np.array(a["solver"]),
        "ridge":              np.float32(a["ridge"]),
        "total_ms":           np.float32(a["total_ms"]),
        "num_iters":          np.int32(a["num_iters"]),
        # explain_instance.py looks for loss_curve to produce a plot.
        # Cholesky has no curve; emit a flat single-point curve so the plot
        # script does not crash, and label the y-axis appropriately.
        "loss_curve":         np.array([a["r2_weighted"]], dtype=np.float32),
    }
    if args.model:
        md = np.load(args.model, allow_pickle=True)
        if "x0"            in md.files: payload["x0"]            = md["x0"].astype(np.float32)
        if "feature_names" in md.files: payload["feature_names"] = md["feature_names"]

    np.savez(args.out, **payload)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
