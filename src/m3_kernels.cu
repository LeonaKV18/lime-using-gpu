#include "m3_kernels.h"
#include <math.h>

__global__ void compute_feature_sums(const float *X, float *sum, float *sum_sq, int B, int D)
{
    int j = blockIdx.x;
    if (j >= D) return;
    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sq = sdata + blockDim.x;
    int tid = threadIdx.x;
    float a = 0.0f, b = 0.0f;
    for (int k = tid; k < B; k += blockDim.x)
    {
        float v = X[k * D + j];
        a += v;
        b += v * v;
    }
    s_sum[tid] = a;
    s_sq[tid] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid] += s_sq[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        sum[j] = s_sum[0];
        sum_sq[j] = s_sq[0];
    }
}

__global__ void finalize_mean_std(const float *sum, const float *sum_sq, float *mean, float *std, int B, int D, float eps)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float m = sum[j] / (float)B;
    float var = sum_sq[j] / (float)B - m * m;
    if (var < 0.0f) var = 0.0f;
    mean[j] = m;
    std[j] = sqrtf(var) + eps;
}

__global__ void normalize_features(const float *X, const float *mean, const float *std, float *X_norm, int B, int D)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= B || j >= D) return;
    int idx = k * D + j;
    X_norm[idx] = (X[idx] - mean[j]) / std[j];
}

__global__ void build_normal_matrix(const float *X_norm, const float *w, float *A, int B, int D)
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

__global__ void build_normal_rhs(const float *X_norm, const float *w, const float *y, float *b, int B, int D)
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

__global__ void add_ridge(float *A, int N, float lambda)
{
    int i = threadIdx.x;
    if (i >= N) return;
    A[i * N + i] += lambda;
}

__global__ void build_normal_matrix_d(const float *X_norm, const float *w, double *A, int B, int D)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int N = D + 1;
    if (i >= N || j >= N) return;
    extern __shared__ double sd_d[];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int k = tid; k < B; k += blockDim.x)
    {
        double xi = (i == 0) ? 1.0 : (double)X_norm[k * D + (i - 1)];
        double xj = (j == 0) ? 1.0 : (double)X_norm[k * D + (j - 1)];
        acc += (double)w[k] * xi * xj;
    }
    sd_d[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
            if (tid < s) sd_d[tid] += sd_d[tid + s];
        __syncthreads();
    }
    if (tid == 0) A[i * N + j] = sd_d[0];
}

__global__ void build_normal_rhs_d(const float *X_norm, const float *w, const float *y, double *b, int B, int D)
{
    int i = blockIdx.x;
    int N = D + 1;
    if (i >= N) return;
    extern __shared__ double sd_d[];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int k = tid; k < B; k += blockDim.x)
    {
        double xi = (i == 0) ? 1.0 : (double)X_norm[k * D + (i - 1)];
        acc += (double)w[k] * xi * (double)y[k];
    }
    sd_d[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
            if (tid < s) sd_d[tid] += sd_d[tid + s];
        __syncthreads();
    }
    if (tid == 0) b[i] = sd_d[0];
}

__global__ void add_ridge_d(double *A, int N, double lambda)
{
    int i = threadIdx.x;
    if (i >= N) return;
    A[i * N + i] += lambda;
}

__global__ void cholesky_decomp(float *A, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int k = 0; k < N; ++k)
    {
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
        for (int i = k + 1 + tid; i < N; i += bs)
        {
            float s = A[i * N + k];
            for (int j = 0; j < k; ++j) s -= A[i * N + j] * A[k * N + j];
            A[i * N + k] = s / diag;
        }
        __syncthreads();
    }
}

__global__ void cholesky_solve(const float *L, const float *b, float *beta, int N)
{
    extern __shared__ float sdata[];
    float *z = sdata;
    float *p = sdata + N;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int i = 0; i < N; ++i)
    {
        float partial = 0.0f;
        for (int j = tid; j < i; j += bs) partial += L[i * N + j] * z[j];
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
    for (int i = N - 1; i >= 0; --i)
    {
        float partial = 0.0f;
        for (int j = i + 1 + tid; j < N; j += bs) partial += L[j * N + i] * beta[j];
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

__global__ void cholesky_decomp_d(double *A, int N)
{
    extern __shared__ double sd_d[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int k = 0; k < N; ++k)
    {
        double partial = 0.0;
        for (int j = tid; j < k; j += bs)
        {
            double v = A[k * N + j];
            partial += v * v;
        }
        sd_d[tid] = partial;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1)
        {
            if (tid < s) sd_d[tid] += sd_d[tid + s];
            __syncthreads();
        }
        if (tid == 0)
        {
            double val = A[k * N + k] - sd_d[0];
            A[k * N + k] = sqrt(val > 1e-24 ? val : 1e-24);
        }
        __syncthreads();
        double diag = A[k * N + k];
        for (int i = k + 1 + tid; i < N; i += bs)
        {
            double s = A[i * N + k];
            for (int j = 0; j < k; ++j) s -= A[i * N + j] * A[k * N + j];
            A[i * N + k] = s / diag;
        }
        __syncthreads();
    }
}

__global__ void cholesky_solve_d(const double *L, const double *b, double *beta, int N)
{
    extern __shared__ double sd_d[];
    double *z = sd_d;
    double *p = sd_d + N;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int i = 0; i < N; ++i)
    {
        double partial = 0.0;
        for (int j = tid; j < i; j += bs) partial += L[i * N + j] * z[j];
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
    for (int i = N - 1; i >= 0; --i)
    {
        double partial = 0.0;
        for (int j = i + 1 + tid; j < N; j += bs) partial += L[j * N + i] * beta[j];
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

__global__ void denormalize_coeff(const float *beta_norm, const float *mean, const float *std, float *coeff, float *intercept, int D)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    float corr = 0.0f;
    for (int j = tid; j < D; j += bs)
    {
        float bj = beta_norm[j + 1] / std[j];
        coeff[j] = bj;
        corr += bj * mean[j];
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

__global__ void denormalize_coeff_d(const double *beta_norm, const float *mean, const float *std, float *coeff, float *intercept, int D)
{
    extern __shared__ double sd_d[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    double corr = 0.0;
    for (int j = tid; j < D; j += bs)
    {
        float bj = (float)(beta_norm[j + 1] / (double)std[j]);
        coeff[j] = bj;
        corr += (double)bj * (double)mean[j];
    }
    sd_d[tid] = corr;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1)
    {
        if (tid < s) sd_d[tid] += sd_d[tid + s];
        __syncthreads();
    }
    if (tid == 0) *intercept = (float)(beta_norm[0] - sd_d[0]);
}

__global__ void surrogate_predict(const float *X, const float *coeff, float intercept, float *pred, int B, int D)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B) return;
    float s = intercept;
    int row = k * D;
    for (int j = 0; j < D; ++j) s += X[row + j] * coeff[j];
    pred[k] = s;
}

__global__ void weighted_r2_partials(const float *pred, const float *y, const float *w, float *partials, int B)
{
    extern __shared__ float sdata[];
    float *s_w = sdata;
    float *s_wy = sdata + blockDim.x;
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
    s_w[tid] = a; s_wy[tid] = b; s_wy2[tid] = c; s_wr2[tid] = d;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_w[tid] += s_w[tid + s];
            s_wy[tid] += s_wy[tid + s];
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

__global__ void surrogate_predict_x0(const float *coeff, float intercept, const float *x0, float *out, int D)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    float partial = 0.0f;
    for (int j = tid; j < D; j += bs) partial += coeff[j] * x0[j];
    sdata[tid] = partial;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0] + intercept;
}

__global__ void gd_compute_residuals(const float *X_norm, const float *y, const float *beta, float *res, int B, int D)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B) return;
    float s = beta[0];
    int row = k * D;
    for (int j = 0; j < D; ++j) s += beta[j + 1] * X_norm[row + j];
    res[k] = s - y[k];
}

__global__ void gd_compute_gradient(const float *X_norm, const float *w, const float *res, const float *beta, float *grad, int B, int D, float lambda)
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

__global__ void gd_update_beta(float *beta, const float *grad, int N, float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    beta[i] -= lr * grad[i];
}

__global__ void reduce_sum(const float *v, float *out, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
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
