#pragma once
#include <cuda_runtime.h>

__global__ void compute_feature_sums(const float *X, float *sum, float *sum_sq, int B, int D);
__global__ void finalize_mean_std(const float *sum, const float *sum_sq, float *mean, float *std, int B, int D, float eps);
__global__ void normalize_features(const float *X, const float *mean, const float *std, float *X_norm, int B, int D);

__global__ void build_normal_matrix(const float *X_norm, const float *w, float *A, int B, int D);
__global__ void build_normal_rhs(const float *X_norm, const float *w, const float *y, float *b, int B, int D);
__global__ void add_ridge(float *A, int N, float lambda);

__global__ void build_normal_matrix_d(const float *X_norm, const float *w, double *A, int B, int D);
__global__ void build_normal_rhs_d(const float *X_norm, const float *w, const float *y, double *b, int B, int D);
__global__ void add_ridge_d(double *A, int N, double lambda);

__global__ void cholesky_decomp(float *A, int N);
__global__ void cholesky_solve(const float *L, const float *b, float *beta, int N);
__global__ void cholesky_decomp_d(double *A, int N);
__global__ void cholesky_solve_d(const double *L, const double *b, double *beta, int N);

__global__ void denormalize_coeff(const float *beta_norm, const float *mean, const float *std, float *coeff, float *intercept, int D);
__global__ void denormalize_coeff_d(const double *beta_norm, const float *mean, const float *std, float *coeff, float *intercept, int D);

__global__ void surrogate_predict(const float *X, const float *coeff, float intercept, float *pred, int B, int D);
__global__ void weighted_r2_partials(const float *pred, const float *y, const float *w, float *partials, int B);
__global__ void surrogate_predict_x0(const float *coeff, float intercept, const float *x0, float *out, int D);

__global__ void gd_compute_residuals(const float *X_norm, const float *y, const float *beta, float *res, int B, int D);
__global__ void gd_compute_gradient(const float *X_norm, const float *w, const float *res, const float *beta, float *grad, int B, int D, float lambda);
__global__ void gd_update_beta(float *beta, const float *grad, int N, float lr);
__global__ void reduce_sum(const float *v, float *out, int N);
