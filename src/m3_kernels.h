#pragma once
//
// m3_kernels.h
// ------------
// Custom CUDA kernels that implement the surrogate-training stage of LIME
// for Milestone 3.  Together with the perturbation / inference / weight
// kernels from kernels.h these form a fully GPU-native LIME pipeline:
//
//   X (B,D)  +  y (B,)  +  w (B,)
//        |         |         |
//        +---------+---------+
//                  |
//        z-score features    (compute_feature_stats, normalize_features)
//                  |
//   Build A = X~^T diag(w) X~ + lambda*I    (build_normal_matrix)
//   Build b = X~^T diag(w) y                 (build_normal_rhs)
//        where X~ = [1 | X_normalized]   (bias column handled implicitly)
//                  |
//   Cholesky:  A = L L^T                    (cholesky_decomp)
//   Solve   :  L L^T beta = b               (cholesky_solve)
//                  |
//   beta in normalized space  ->  beta in original space
//                                          (denormalize_coeff)
//                  |
//   Weighted R^2 of surrogate              (compute_residuals,
//   Surrogate prediction at x0              weighted_metrics)
//
// All buffers live on the device.  Only the final (D+1) coefficients and a
// handful of scalar metrics are copied back to the host.
//
#include <cuda_runtime.h>


// ---------------------------------------------------------------------------
// Stage A : feature standardization (z-score)
// ---------------------------------------------------------------------------
// Each column j of X is rescaled so it has zero mean and unit variance over
// the B perturbed samples.  Standardization improves the conditioning of the
// normal matrix dramatically when raw feature scales differ by many orders
// of magnitude (typical for tabular data).

// Computes per-feature sums and squared sums via parallel reduction across
// the B samples.  One block per feature column, threads cooperate on B.
__global__ void compute_feature_sums(
    const float *X, float *sum, float *sum_sq, int B, int D);

// Converts (sum, sum_sq) into (mean, std).  std has a small epsilon floor
// so a constant feature does not divide by zero in the next stage.
__global__ void finalize_mean_std(
    const float *sum, const float *sum_sq,
    float *mean, float *std,
    int B, int D, float eps);

// Element-wise z-score: X_norm[k,j] = (X[k,j] - mean[j]) / std[j].
__global__ void normalize_features(
    const float *X, const float *mean, const float *std,
    float *X_norm, int B, int D);


// ---------------------------------------------------------------------------
// Stage B : build the weighted normal equations
// ---------------------------------------------------------------------------
// We never materialize the augmented matrix X~ = [1 | X_norm].  Instead the
// bias column (all ones) is treated as a virtual column 0 inside each
// kernel.  This saves B*sizeof(float) bytes and one extra kernel launch.

// Builds A = X~^T diag(w) X~  with shape (D+1) x (D+1).
// Grid is (D+1) x (D+1); each block reduces over the B samples for one
// matrix entry.  Block size should be a power of two (e.g. 256).
__global__ void build_normal_matrix(
    const float *X_norm, const float *w,
    float *A, int B, int D);

// Builds b = X~^T diag(w) y  with shape (D+1).
// Grid is (D+1) blocks of 256 threads, one block per output entry.
__global__ void build_normal_rhs(
    const float *X_norm, const float *w, const float *y,
    float *b, int B, int D);

// Adds lambda * I to A (Tikhonov regularization).  Single block, N threads.
__global__ void add_ridge(float *A, int N, float lambda);


// ---------------------------------------------------------------------------
// Stage C : Cholesky factorization and triangular solves
// ---------------------------------------------------------------------------
// A is symmetric positive definite (after ridge).  Cholesky gives A = L L^T
// with L lower triangular.  We do the entire factorization inside a single
// block because the matrix is tiny (N = D+1, typically <= a few hundred):
// launching kernels per pivot would be dominated by launch overhead.

// In-place Cholesky factorization.  Lower triangle of A is overwritten with
// L; upper triangle is left untouched and ignored by the solver.
// Launch with <<<1, blockDim, blockDim*sizeof(float)>>> where blockDim is
// a power of two (e.g. 256).
__global__ void cholesky_decomp(float *A, int N);

// Solve  L L^T beta = b  via forward then back substitution.
// Launch with <<<1, blockDim, N*sizeof(float)>>>; z is held in shared memory.
__global__ void cholesky_solve(
    const float *L, const float *b, float *beta, int N);


// ---------------------------------------------------------------------------
// Stage D : map coefficients back to the original feature space
// ---------------------------------------------------------------------------
// If the surrogate was fit on z-scored features then for an original-space
// vector x the prediction is
//
//   beta_norm[0] + sum_j beta_norm[j+1] * (x[j] - mean[j]) / std[j]
//   = (beta_norm[0] - sum_j beta_norm[j+1]*mean[j]/std[j])    <- new intercept
//     + sum_j (beta_norm[j+1] / std[j]) * x[j]                <- new coeff j
//
// This kernel performs that algebraic rewrite directly on the device so the
// host receives ready-to-use feature attributions.

__global__ void denormalize_coeff(
    const float *beta_norm,
    const float *mean, const float *std,
    float *coeff, float *intercept, int D);


// ---------------------------------------------------------------------------
// Stage E : surrogate evaluation metrics (all on the GPU)
// ---------------------------------------------------------------------------
// surrogate predictions on the B perturbed samples, in original space:
//   pred[k] = intercept + sum_j coeff[j] * X[k, j]
__global__ void surrogate_predict(
    const float *X, const float *coeff, float intercept,
    float *pred, int B, int D);

// Computes weighted R^2 against the black-box predictions y.  Three partial
// reductions are produced (sum_w*y, sum_w, sum_w*(y-mean)^2, sum_w*(y-pred)^2)
// then combined on the host.  This avoids a separate kernel launch chain
// just for an aggregate metric.
__global__ void weighted_r2_partials(
    const float *pred, const float *y, const float *w,
    float *partials,    // [sum_w, sum_wy, sum_wy2, sum_wr2] (4 floats)
    int B);

// Surrogate prediction at the original x0 (single thread block, single
// thread is fine because it is just a dot product of size D).
__global__ void surrogate_predict_x0(
    const float *coeff, float intercept, const float *x0,
    float *out, int D);


// ---------------------------------------------------------------------------
// Stage F : iterative gradient descent solver (alternative to Cholesky)
// ---------------------------------------------------------------------------
// Used when D is very large or the user explicitly requests --solver=gd.
// Uses normalized features and the augmented bias column.

// Computes residual r[k] = X~[k,:] beta - y[k].  Grid (B+255)/256.
__global__ void gd_compute_residuals(
    const float *X_norm, const float *y, const float *beta,
    float *res, int B, int D);

// Computes gradient g = X~^T (w * r) + lambda * beta.  Grid (D+1) blocks.
__global__ void gd_compute_gradient(
    const float *X_norm, const float *w, const float *res,
    const float *beta, float *grad, int B, int D, float lambda);

// In-place update beta -= lr * grad. Grid (D+1) blocks.
__global__ void gd_update_beta(float *beta, const float *grad, int N, float lr);

// Single-block reduction: out[0] = sum_k v[k].  Used to make the GD step
// size invariant to the magnitude of the weight vector (sum_w can vary by
// orders of magnitude depending on kernel-width / B).
__global__ void reduce_sum(const float *v, float *out, int N);
