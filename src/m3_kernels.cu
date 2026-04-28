//
// m3_kernels.cu
// -------------
// Implementation of the GPU surrogate-training kernels declared in
// m3_kernels.h.  See that header for the high-level pipeline.
//
// Conventions used throughout:
//
//   B   -- number of perturbed samples (rows)
//   D   -- number of original features (cols of X)
//   N   -- D + 1                          (size of the augmented system,
//                                          one extra dimension for the bias)
//   X is (B, D) row-major.
//   X~ is (B, N) where the (virtual) column 0 is all-ones; we never
//   materialize it.
//
// Most kernels follow the same parallel-reduction template:
//   1. Each thread accumulates a partial sum across a strided slice of B.
//   2. The block reduces those partials in shared memory.
//   3. Thread 0 writes the final value to global memory.
//
#include "m3_kernels.h"
#include <math.h>


// ===========================================================================
//                           STAGE A : Z-SCORE
// ===========================================================================
//
// To compute the (mean, std) of every feature in a single pass we accumulate
// both sum and sum-of-squares.  Variance is then  E[X^2] - E[X]^2.  This is
// the classical "naive" two-moment formula; for the magnitudes typical in
// LIME perturbations (small noise around x0) it is numerically perfectly
// adequate.  A Welford-style streaming update would be more robust for
// extreme dynamic ranges, but is unnecessary here.
//

// One block per feature; threads cooperate to reduce across the B samples.
__global__ void compute_feature_sums(
    const float *X, float *sum, float *sum_sq, int B, int D)
{
    int j = blockIdx.x;
    if (j >= D) return;

    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sq  = sdata + blockDim.x;

    int tid = threadIdx.x;
    float a = 0.0f, b = 0.0f;
    for (int k = tid; k < B; k += blockDim.x)
    {
        float v = X[k * D + j];
        a += v;
        b += v * v;
    }
    s_sum[tid] = a;
    s_sq[tid]  = b;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid]  += s_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        sum[j]    = s_sum[0];
        sum_sq[j] = s_sq[0];
    }
}

// 1D launch: each thread finalises one feature.
__global__ void finalize_mean_std(
    const float *sum, const float *sum_sq,
    float *mean, float *std,
    int B, int D, float eps)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float m   = sum[j] / (float)B;
    float var = sum_sq[j] / (float)B - m * m;
    if (var < 0.0f) var = 0.0f;          // guard against fp cancellation
    mean[j] = m;
    std[j]  = sqrtf(var) + eps;          // eps avoids division by zero later
}

// 2D launch: each thread normalises one (sample, feature) entry.
__global__ void normalize_features(
    const float *X, const float *mean, const float *std,
    float *X_norm, int B, int D)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= B || j >= D) return;
    int idx     = k * D + j;
    X_norm[idx] = (X[idx] - mean[j]) / std[j];
}


// ===========================================================================
//                  STAGE B : BUILD WEIGHTED NORMAL EQUATIONS
// ===========================================================================
//
// The naive matrix multiplication X~^T W X~ is replaced here by a 2D grid of
// reductions: one CUDA block per output entry A[i,j].  Each block does a
// strided sum over the B-axis.  Although this duplicates some reads (lower
// triangle is computed even though A is symmetric) the kernel stays simple
// and the pattern is bandwidth-bound -- a half-triangle optimization buys
// only a 2x reduction at the cost of significant kernel complexity.
//
// The bias column (i=0 or j=0) is materialized lazily inside each thread so
// we never allocate the (B, N) augmented buffer.
//

__global__ void build_normal_matrix(
    const float *X_norm, const float *w,
    float *A, int B, int D)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int N = D + 1;
    if (i >= N || j >= N) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int k = tid; k < B; k += blockDim.x)
    {
        float xi = (i == 0) ? 1.0f : X_norm[k * D + (i - 1)];
        float xj = (j == 0) ? 1.0f : X_norm[k * D + (j - 1)];
        acc += w[k] * xi * xj;
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) A[i * N + j] = sdata[0];
}

__global__ void build_normal_rhs(
    const float *X_norm, const float *w, const float *y,
    float *b, int B, int D)
{
    int i = blockIdx.x;
    int N = D + 1;
    if (i >= N) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int k = tid; k < B; k += blockDim.x)
    {
        float xi = (i == 0) ? 1.0f : X_norm[k * D + (i - 1)];
        acc += w[k] * xi * y[k];
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) b[i] = sdata[0];
}

// 1D launch: 1 block of N threads.  Adds lambda to every diagonal entry.
__global__ void add_ridge(float *A, int N, float lambda)
{
    int i = threadIdx.x;
    if (i >= N) return;
    A[i * N + i] += lambda;
}


// ===========================================================================
//                       STAGE C : CHOLESKY  A = L L^T
// ===========================================================================
//
// We launch a single block.  Each pivot k of the factorization is done
// sequentially -- the dependency chain is inherently serial -- but every
// scalar inner product inside a pivot is parallelized across the threads
// of the block using a shared-memory reduction.
//
// Numerical guard: if the diagonal residual goes non-positive due to
// cancellation we clamp it to a tiny positive number.  This is preferable
// to NaN-ing out the entire surrogate; downstream code monitors the
// residual norm to detect a genuinely ill-conditioned system.
//
// Layout: A is row-major (N, N).  After this kernel
//   A[i, j] for i >= j  == L[i, j]
//   the upper triangle is left untouched and ignored by the solver.

__global__ void cholesky_decomp(float *A, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    for (int k = 0; k < N; ++k)
    {
        // ---- Compute L[k,k] = sqrt(A[k,k] - sum_{j<k} L[k,j]^2) ----
        float partial = 0.0f;
        for (int j = tid; j < k; j += bs)
        {
            float v = A[k * N + j];
            partial += v * v;
        }
        sdata[tid] = partial;
        __syncthreads();

        for (int s = bs / 2; s > 0; s >>= 1)
        {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0)
        {
            float val = A[k * N + k] - sdata[0];
            A[k * N + k] = sqrtf(val > 1e-12f ? val : 1e-12f);
        }
        __syncthreads();
        float diag = A[k * N + k];

        // ---- Compute L[i,k] for i > k.  Each thread owns a stride of i. ----
        // Inner sum sum_{j<k} L[i,j]*L[k,j] is sequential (k iterations) but
        // independent across i, so we get N-k-1 parallel rows per pivot.
        for (int i = k + 1 + tid; i < N; i += bs)
        {
            float s = A[i * N + k];
            for (int j = 0; j < k; ++j)
                s -= A[i * N + j] * A[k * N + j];
            A[i * N + k] = s / diag;
        }
        __syncthreads();
    }
}


// Forward then back substitution to solve L L^T beta = b.
// Inner sums are again parallelized via shared-memory reduction.
__global__ void cholesky_solve(
    const float *L, const float *b, float *beta, int N)
{
    extern __shared__ float sdata[];
    float *z = sdata;                     // intermediate vector L z = b
    float *p = sdata + N;                 // reduction buffer (size = bs)

    int tid = threadIdx.x;
    int bs  = blockDim.x;

    // ---- Forward solve : L z = b ----
    for (int i = 0; i < N; ++i)
    {
        float partial = 0.0f;
        for (int j = tid; j < i; j += bs)
            partial += L[i * N + j] * z[j];
        p[tid] = partial;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1)
        {
            if (tid < s) p[tid] += p[tid + s];
            __syncthreads();
        }
        if (tid == 0) z[i] = (b[i] - p[0]) / L[i * N + i];
        __syncthreads();
    }

    // ---- Back solve : L^T beta = z ----
    // Note: L^T is upper-triangular; row i depends on entries beta[j] for j > i.
    // We sweep from the bottom row upwards so each beta[i] becomes available
    // before any earlier row needs it.
    for (int i = N - 1; i >= 0; --i)
    {
        float partial = 0.0f;
        for (int j = i + 1 + tid; j < N; j += bs)
            partial += L[j * N + i] * beta[j];
        p[tid] = partial;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1)
        {
            if (tid < s) p[tid] += p[tid + s];
            __syncthreads();
        }
        if (tid == 0) beta[i] = (z[i] - p[0]) / L[i * N + i];
        __syncthreads();
    }
}


// ===========================================================================
//                  STAGE D : DE-NORMALIZE THE COEFFICIENTS
// ===========================================================================

__global__ void denormalize_coeff(
    const float *beta_norm,
    const float *mean, const float *std,
    float *coeff, float *intercept, int D)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    // First: each thread maps a slice of features and accumulates the
    //        intercept-correction term  sum_j beta_norm[j+1] * mean[j]/std[j]
    float corr = 0.0f;
    for (int j = tid; j < D; j += bs)
    {
        float bj = beta_norm[j + 1] / std[j];
        coeff[j] = bj;                                   // original-space slope
        corr    += bj * mean[j];
    }
    sdata[tid] = corr;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *intercept = beta_norm[0] - sdata[0];
}


// ===========================================================================
//                    STAGE E : SURROGATE EVALUATION METRICS
// ===========================================================================

__global__ void surrogate_predict(
    const float *X, const float *coeff, float intercept,
    float *pred, int B, int D)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B) return;

    float s   = intercept;
    int row   = k * D;
    for (int j = 0; j < D; ++j)
        s += X[row + j] * coeff[j];
    pred[k] = s;
}


// Single block.  Computes the four scalar reductions needed for weighted R^2
// (denominator and numerator) in a single pass.  Output layout:
//   partials[0] = sum_k w[k]
//   partials[1] = sum_k w[k] * y[k]
//   partials[2] = sum_k w[k] * y[k]^2
//   partials[3] = sum_k w[k] * (y[k] - pred[k])^2
__global__ void weighted_r2_partials(
    const float *pred, const float *y, const float *w,
    float *partials, int B)
{
    extern __shared__ float sdata[];
    float *s_w   = sdata;
    float *s_wy  = sdata +     blockDim.x;
    float *s_wy2 = sdata + 2 * blockDim.x;
    float *s_wr2 = sdata + 3 * blockDim.x;

    int tid = threadIdx.x;
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    for (int k = tid; k < B; k += blockDim.x)
    {
        float wk = w[k];
        float yk = y[k];
        float pk = pred[k];
        float rk = yk - pk;
        a += wk;
        b += wk * yk;
        c += wk * yk * yk;
        d += wk * rk * rk;
    }
    s_w[tid]   = a;
    s_wy[tid]  = b;
    s_wy2[tid] = c;
    s_wr2[tid] = d;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_w[tid]   += s_w[tid + s];
            s_wy[tid]  += s_wy[tid + s];
            s_wy2[tid] += s_wy2[tid + s];
            s_wr2[tid] += s_wr2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partials[0] = s_w[0];
        partials[1] = s_wy[0];
        partials[2] = s_wy2[0];
        partials[3] = s_wr2[0];
    }
}


// Surrogate prediction at the original x0 instance.  One block, parallel
// reduction across D.  out is a single float on device.
__global__ void surrogate_predict_x0(
    const float *coeff, float intercept, const float *x0,
    float *out, int D)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    float partial = 0.0f;
    for (int j = tid; j < D; j += bs)
        partial += coeff[j] * x0[j];
    sdata[tid] = partial;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0] + intercept;
}


// ===========================================================================
//                     STAGE F : GRADIENT DESCENT SOLVER
// ===========================================================================
//
// An alternative to closed-form Cholesky: cheap per-iteration, handles
// large D where forming the (D+1)^2 normal matrix becomes wasteful, and
// converges to the same minimizer as the normal equations when run to
// convergence.  Provided as a comparison path; closed-form is the default.

// Residual r[k] = X~[k,:] beta - y[k]
__global__ void gd_compute_residuals(
    const float *X_norm, const float *y, const float *beta,
    float *res, int B, int D)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B) return;

    float s   = beta[0];                 // bias column contribution
    int row   = k * D;
    for (int j = 0; j < D; ++j)
        s += beta[j + 1] * X_norm[row + j];
    res[k] = s - y[k];
}

// Gradient g[i] = sum_k X~[k,i] * w[k] * r[k]  + lambda * beta[i]
// Grid: (D+1) blocks; each block reduces over B for one output entry.
__global__ void gd_compute_gradient(
    const float *X_norm, const float *w, const float *res,
    const float *beta, float *grad, int B, int D, float lambda)
{
    int i = blockIdx.x;
    int N = D + 1;
    if (i >= N) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int k = tid; k < B; k += blockDim.x)
    {
        float xi = (i == 0) ? 1.0f : X_norm[k * D + (i - 1)];
        acc += xi * w[k] * res[k];
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) grad[i] = sdata[0] + lambda * beta[i];
}

// In-place update beta -= lr * grad
__global__ void gd_update_beta(float *beta, const float *grad, int N, float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    beta[i] -= lr * grad[i];
}

// Single-block sum reduction. out[0] = sum_k v[k]
__global__ void reduce_sum(const float *v, float *out, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    float acc = 0.0f;
    for (int k = tid; k < N; k += bs) acc += v[k];
    sdata[tid] = acc;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}
