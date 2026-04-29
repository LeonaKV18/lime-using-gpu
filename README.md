# LIME-GPU: Local Interpretable Model-agnostic Explanations with GPU Acceleration

This project implements the LIME (Local Interpretable Model-agnostic Explanations) algorithm from scratch, focusing on GPU acceleration using CUDA. The goal is to progressively build a fully GPU-native LIME pipeline, emphasizing correctness, parallelism, and interpretability.

---

## Table of Contents

- [Introduction and Overview](#introduction--overview)
- [Key Focus Areas](#key-focus-areas)
- [Current Status](#current-status)
- [Repository Layout](#repository-layout)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
  - [Milestone 1 / 2](#milestone-1--2)
  - [Milestone 3 (fully GPU-native LIME)](#milestone-3-fully-gpu-native-lime)

---

## Introduction / Overview

**LIME** is an Explainable AI (XAI) technique that explains individual predictions made by any black-box machine learning model. For a given input, LIME:

1. Generates perturbed samples around the input
2. Queries the black-box model on those samples
3. Weights each sample by similarity to the original input
4. Trains a simple surrogate model (e.g., linear regression) on the weighted samples
5. Uses the surrogate's coefficients as feature-importance explanations

Steps 1–3 involve processing thousands of samples independently, making them highly amenable to GPU parallelism. **Milestone 3 also accelerates step 4** with a custom GPU implementation of weighted linear regression (Cholesky-based closed-form *and* gradient descent), removing the last framework dependency.

### Key Focus Areas

1. **Perturbation on the GPU:** Generating large batches of perturbed samples in parallel using CUDA kernels.
2. **Parallel Inference:** Running forward-pass inference (logistic regression / shallow MLP) across all perturbed samples simultaneously on the GPU.
3. **Weight Computation:** Computing similarity-based kernel weights using GPU-parallelized distance metrics.
4. **Surrogate Model Training:** First validated using PyTorch (M2), then replaced with a custom GPU implementation using Cholesky-based closed-form and gradient descent (M3).
5. **Performance Analysis:** Comparing CPU-only, framework-assisted, and fully GPU-native implementations.

### Current Status

| Task                                                        | Milestone | Status         |
| :---------------------------------------------------------- | :-------- | :------------- |
| GPU-based perturbation, inference, and weight computation   | M1        | ✅ Complete    |
| Surrogate training via PyTorch                              | M2        | ✅ Complete    |
| Fully custom GPU-based surrogate model training             | M3        | ✅ Complete    |
| Performance comparison: CPU vs framework vs GPU             | M3        | ✅ Complete    |
| Technical report on numerical stability and design          | M3        | ✅ See `M3_DESIGN.md` |

---

## Repository Layout

```
src/
  kernels.cu / kernels.h      M1 kernels: perturbation, inference, weights
  m3_kernels.cu / m3_kernels.h  M3 kernels: feature stats, normal-equation
                               build, Cholesky, triangular solves,
                               de-normalization, R^2, gradient descent
  main.cu                     Driver for M1 / M2 (writes preds.bin etc.)
  main_m3.cu                  End-to-end GPU-native LIME driver (M3)
  utils.h                     CUDA error-checking macro

scripts/
  train_model.py              Train logistic-regression black box
  cpu_reference.py            CPU mirror of the M1 kernels (validation)
  compare_and_validate.py     M1 GPU vs CPU correctness check
  surrogate_train.py          M2: PyTorch weighted linear surrogate
  explain_instance.py         M2/M3 plots: importance, loss, ablation
  cpu_surrogate.py            M3: NumPy weighted linear-regression reference
  gpu_surrogate_to_npz.py     M3: lime_m3 .bin -> .npz (compatible with M2)
  compare_surrogates.py       M3: GPU vs M2-PyTorch vs CPU side-by-side
  generate_X_bin.py           Optional deterministic X for replay
  save_outputs.py             Pack raw .bin outputs into .npz
  convert_timings_sweep.py    Parse timing logs -> CSV
  plot_results.py             Speed-up / breakdown plots

run_m1_eval.bat               M1 sweep + correctness + plots
run_m2_eval.bat               M2 end-to-end pipeline
run_m3_eval.bat               M3 end-to-end pipeline + 4-way comparison

M3_DESIGN.md                  Technical report for Milestone 3
```

---

## Build Instructions

Requires CUDA Toolkit (11+ recommended) and CMake 3.18+. On Windows the
default generator is Visual Studio 2022 (see `CMakePresets.json`).

```cmd
cmake --preset vs2022-cuda
cmake --build --preset vs2022-cuda-release
```

This produces two executables in `build/vs2022/Release/`:

* `lime_m1.exe` – stages 1–3 only (M1 / M2 driver)
* `lime_m3.exe` – stages 1–4, fully GPU-native (M3)

---

## Usage

### Milestone 1 / 2

```cmd
REM Train model + run M1 GPU pipeline + PyTorch surrogate (M2)
run_m2_eval.bat
```

Manually:

```cmd
python scripts\train_model.py --dataset=breast_cancer
build\vs2022\Release\lime_m1.exe --B=16384 --model=models\breast_cancer.bin ^
    --write-X X.bin --write-preds preds.bin --write-weights weights.bin
python scripts\surrogate_train.py --X X.bin --preds preds.bin --weights weights.bin ^
    --model models\breast_cancer.npz --out attributions.npz
python scripts\explain_instance.py --attributions attributions.npz ^
    --X X.bin --preds preds.bin --weights weights.bin --out plots\m2
```

### Milestone 3 (fully GPU-native LIME)

```cmd
REM Train model + GPU LIME (Cholesky + GD) + CPU/M2 cross-validation + plots
run_m3_eval.bat
```

Manually, the *single* CUDA executable runs the entire LIME pipeline including
surrogate training:

```cmd
build\vs2022\Release\lime_m3.exe ^
    --B=16384 --model=models\breast_cancer.bin ^
    --solver=cholesky --ridge=1e-3 ^
    --write-attributions build\attributions_gpu.bin

python scripts\gpu_surrogate_to_npz.py ^
    --in build\attributions_gpu.bin --model models\breast_cancer.npz ^
    --out build\attributions_gpu.npz

python scripts\explain_instance.py --attributions build\attributions_gpu.npz ^
    --out plots\m3
```

Available `lime_m3` flags:

| Flag                               | Default        | Description |
| :--------------------------------- | :------------- | :---------- |
| `--B=<int>`                        | 16384          | Number of perturbed samples |
| `--D=<int>`                        | 30             | Feature count (overridden by `--model`) |
| `--model=<path.bin>`               | (synthetic)    | Black-box parameters from `train_model.py` |
| `--solver=cholesky\|gd`            | `cholesky`     | Closed-form Cholesky **or** gradient descent |
| `--ridge=<float>`                  | 1e-3           | L2 / Tikhonov regularization (also used as ridge by GD) |
| `--gd-iters=<int>`                 | 500            | GD iteration count |
| `--gd-lr=<float>`                  | 1e-2           | GD learning rate |
| `--perturb=per-sample\|per-feature`| `per-sample`   | Perturbation kernel layout |
| `--infer=custom\|cublas`           | `custom`       | Inference path (custom kernel or cuBLAS GEMV) |
| `--write-X <path>`                 | -              | Dump perturbations to disk |
| `--write-preds <path>`             | -              | Dump black-box predictions |
| `--write-weights <path>`           | -              | Dump similarity weights |
| `--write-attributions <path>`      | `attributions_gpu.bin` | Final coefficients/intercept/R²/timings |

Cross-validation against the framework-based and CPU-only references:

```cmd
python scripts\compare_surrogates.py ^
    --m2  build\attributions_m2.npz ^
    --m3  build\attributions_gpu.npz ^
    --cpu build\attributions_cpu.npz ^
    --top-k 10
```

See [`M3_DESIGN.md`](M3_DESIGN.md) for the technical write-up: kernel design,
numerical-stability choices, performance trade-offs, and end-to-end timing
comparisons.

---

## License

This project is for educational and research purposes.
