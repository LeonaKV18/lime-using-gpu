# LIME-GPU: Local Interpretable Model-agnostic Explanations with GPU Acceleration

This project implements the LIME (Local Interpretable Model-agnostic Explanations) algorithm from scratch, focusing on GPU acceleration using CUDA/OpenCL. The goal is to progressively build a fully GPU-native LIME pipeline, emphasizing correctness, parallelism, and interpretability.

---

## Table of Contents

- [Introduction and Overview](#introduction--overview)
- [Key Focus Areas](#key-focus-areas)
- [Current Status](#current-status)

---

## Introduction / Overview

**LIME** is an Explainable AI (XAI) technique that explains individual predictions made by any black-box machine learning model. For a given input, LIME:

1. Generates perturbed samples around the input
2. Queries the black-box model on those samples
3. Weights each sample by similarity to the original input
4. Trains a simple surrogate model (e.g., linear regression) on the weighted samples
5. Uses the surrogate's coefficients as feature-importance explanations

Steps 1–3 involve processing thousands of samples independently, making them highly amenable to GPU parallelism. This project exploits that parallelism through explicit low-level GPU programming.

### Key Focus Areas

1. **Perturbation on the GPU:** Generating large batches of perturbed samples in parallel using CUDA / OpenCL kernels.
2. **Parallel Inference:** Running forward-pass inference (logistic regression / shallow MLP) across all perturbed samples simultaneously on the GPU.
3. **Weight Computation:** Computing similarity-based kernel weights using GPU-parallelized distance metrics.
4. **Surrogate Model Training:** First validated using PyTorch, then replaced with a custom GPU implementation using a closed-form or gradient-descent solution.
5. **Performance Analysis:** Comparing CPU-only, framework-assisted, and fully GPU-native implementations.

### Current Status

Task Description | Status |
:---|:---|
GPU-based Perturbation, Inference and Weight Computation | 🔲 In Progress |
Surrogate Training via PyTorch | 🔲 Not Started |
Fully custom GPU-based surrogate model training  | 🔲 Not Started |

---

## License

This project is for educational and research purposes.
