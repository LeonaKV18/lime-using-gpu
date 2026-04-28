//
// main_m3.cu
// ----------
// End-to-end GPU-native LIME pipeline (Milestone 3).
//
// Stages:
//   1.  Perturbation generation                -- kernels.cu (M1)
//   2.  Black-box inference                    -- kernels.cu (M1)
//   3.  Weight (similarity-kernel) computation -- kernels.cu (M1)
//   4.  Surrogate training (NEW for M3)        -- m3_kernels.cu
//         * z-score features
//         * build weighted normal equations  A beta = b
//         * Cholesky factorization
//         * triangular solves
//         * de-normalize coefficients
//         * weighted R^2 / surrogate prediction at x0
//
// The entire pipeline runs inside a single executable.  After stage 1-3 the
// results stay on the device and are consumed directly by the surrogate
// kernels.  Only the final  (D+1) coefficients + 4 scalar metrics are
// transferred back to host memory at the very end.
//
// Build:
//     cmake --build build --config Release --target lime_m3
//
// Typical run:
//     lime_m3 --B=16384 --model=models\breast_cancer.bin
//             --solver=cholesky --ridge=1e-3
//             --write-attributions attributions_gpu.bin
//
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include "kernels.h"
#include "m3_kernels.h"
#include "utils.h"


// ---------------------------------------------------------------------------
//                          Model file I/O helpers
// ---------------------------------------------------------------------------
//
// model.bin layout (written by scripts/train_model.py):
//   int32    D
//   float32  W[D]
//   float32  bias
//   float32  x0[D]
//   float32  means[D]
//
static bool load_model_bin(
    const char *path, int &D,
    std::vector<float> &W, float &bias,
    std::vector<float> &x0, std::vector<float> &means)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open model file: %s\n", path); return false; }

    int file_D = 0;
    if (fread(&file_D, sizeof(int),   1, f) != 1)             { fclose(f); return false; }
    D = file_D;
    W.resize(D); x0.resize(D); means.resize(D);
    if (fread(W.data(),     sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    if (fread(&bias,        sizeof(float), 1, f) != 1)         { fclose(f); return false; }
    if (fread(x0.data(),    sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    if (fread(means.data(), sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    fclose(f);
    return true;
}


// attributions.bin layout (consumed by gpu_surrogate_to_npz.py):
//   int32    D
//   int32    solver_id        (0 = cholesky, 1 = gd)
//   float32  ridge
//   float32  coeff[D]                       (original-space slope per feature)
//   float32  intercept                      (original-space bias)
//   float32  surrogate_pred_x0
//   float32  black_box_pred_x0
//   float32  weighted_r2
//   float32  total_ms                       (training time only)
//   int32    num_iters         (>=1 for GD, 1 for Cholesky)
//   int32    B                              (echo for sanity)
//
static void save_attributions(
    const char *path, int D, int solver_id, float ridge,
    const float *coeff, float intercept,
    float surr_x0, float bb_x0, float r2,
    float total_ms, int num_iters, int B)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }
    fwrite(&D, sizeof(int), 1, f);
    fwrite(&solver_id, sizeof(int), 1, f);
    fwrite(&ridge, sizeof(float), 1, f);
    fwrite(coeff, sizeof(float), D, f);
    fwrite(&intercept, sizeof(float), 1, f);
    fwrite(&surr_x0, sizeof(float), 1, f);
    fwrite(&bb_x0, sizeof(float), 1, f);
    fwrite(&r2, sizeof(float), 1, f);
    fwrite(&total_ms, sizeof(float), 1, f);
    fwrite(&num_iters, sizeof(int), 1, f);
    fwrite(&B, sizeof(int), 1, f);
    fclose(f);
    printf("Saved attributions -> %s\n", path);
}


LimeModel create_lime_model(int D, float *dW, float bias)
{
    LimeModel m; m.W = dW; m.bias = bias; m.D = D;
    return m;
}


int main(int argc, char **argv)
{
    // -----------------------------------------------------------------------
    //                          Configuration
    // -----------------------------------------------------------------------
    int   D    = 30;
    int   B    = 16384;
    float mp   = 0.2f;
    float ns   = 0.1f;
    float kw   = 1.0f;
    unsigned long seed = 1234;

    bool  use_per_feature = false;
    bool  use_cublas      = false;

    // M3-specific
    enum class Solver { Cholesky, GD };
    Solver solver       = Solver::Cholesky;
    float  ridge        = 1e-3f;
    int    gd_iters     = 500;
    float  gd_lr        = 1e-2f;

    const char *model_path  = nullptr;
    const char *out_attribs = "attributions_gpu.bin";
    const char *out_preds   = nullptr;
    const char *out_weights = nullptr;
    const char *out_X       = nullptr;

    for (int i = 1; i < argc; ++i)
    {
        if      (!strncmp(argv[i], "--D=", 4))               D  = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--B=", 4))               B  = atoi(argv[i] + 4);
        else if (!strcmp(argv[i], "--perturb=per-feature"))  use_per_feature = true;
        else if (!strcmp(argv[i], "--perturb=per-sample"))   use_per_feature = false;
        else if (!strcmp(argv[i], "--infer=cublas"))         use_cublas      = true;
        else if (!strcmp(argv[i], "--infer=custom"))         use_cublas      = false;
        else if (!strcmp(argv[i], "--solver=cholesky"))      solver          = Solver::Cholesky;
        else if (!strcmp(argv[i], "--solver=gd"))            solver          = Solver::GD;
        else if (!strncmp(argv[i], "--ridge=", 8))           ridge           = (float)atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--gd-iters=", 11))       gd_iters        = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--gd-lr=", 8))           gd_lr           = (float)atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--model=", 8))           model_path      = argv[i] + 8;
        else if (!strcmp(argv[i], "--write-attributions"))   out_attribs     = argv[++i];
        else if (!strcmp(argv[i], "--write-preds"))          out_preds       = argv[++i];
        else if (!strcmp(argv[i], "--write-weights"))        out_weights     = argv[++i];
        else if (!strcmp(argv[i], "--write-X"))              out_X           = argv[++i];
    }

    // -----------------------------------------------------------------------
    //                  Load (or synthesize) model parameters
    // -----------------------------------------------------------------------
    std::vector<float> hx0, hm, hW;
    float hb = 0.0f;

    if (model_path)
    {
        if (!load_model_bin(model_path, D, hW, hb, hx0, hm))
        {
            fprintf(stderr, "[ERROR] Failed to load model from %s\n", model_path);
            return 1;
        }
        printf("Loaded model from %s  D=%d\n", model_path, D);
    }
    else
    {
        hx0.resize(D); hm.resize(D); hW.resize(D);
        for (int i = 0; i < D; ++i)
        {
            hx0[i] = (i % 5 == 0) ? 1.0f : 0.5f;
            hm[i]  = 0.5f;
            hW[i]  = 0.02f * (i + 1);
        }
        hb = -1.0f;
        printf("Using synthetic parameters  D=%d  B=%d\n", D, B);
    }

    int N = D + 1;     // size of augmented system (bias + features)

    printf("Config: D=%d  B=%d  perturb=%s  infer=%s  solver=%s  ridge=%.2e\n",
           D, B,
           use_per_feature ? "per-feature" : "per-sample",
           use_cublas      ? "cublas"      : "custom",
           solver == Solver::Cholesky ? "cholesky" : "gd",
           ridge);

    // -----------------------------------------------------------------------
    //                         Device-side allocation
    // -----------------------------------------------------------------------
    float *dx0, *dm, *dW, *dX, *dXn, *dlog, *dp, *dd, *dw;
    unsigned char *dz;
    curandStatePhilox4_32_10_t *ds;
    LimeModel *dmodel;

    // M3-specific buffers
    float *d_sum, *d_sumsq, *d_mean, *d_std;
    float *d_A, *d_b, *d_beta_norm;
    float *d_coeff, *d_intercept, *d_pred, *d_partials, *d_x0_pred;
    float *d_res, *d_grad;

    CUDA_CALL(cudaMalloc(&dx0,   D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dm,    D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dW,    D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dX,    (size_t)B * D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dXn,   (size_t)B * D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dz,    (size_t)B * D * sizeof(unsigned char)));
    CUDA_CALL(cudaMalloc(&dlog,  B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dp,    B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dd,    B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dw,    B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&ds,    B * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CALL(cudaMalloc(&dmodel, sizeof(LimeModel)));

    CUDA_CALL(cudaMalloc(&d_sum,       D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_sumsq,     D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_mean,      D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_std,       D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_A,         (size_t)N * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_b,         N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_beta_norm, N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_coeff,     D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_intercept, sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_pred,      B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_partials,  4 * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_x0_pred,   sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_res,       B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_grad,      N * sizeof(float)));

    CUDA_CALL(cudaMemcpy(dx0, hx0.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dm,  hm.data(),  D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dW,  hW.data(),  D * sizeof(float), cudaMemcpyHostToDevice));

    LimeModel h_model = create_lime_model(D, dW, hb);
    CUDA_CALL(cudaMemcpy(dmodel, &h_model, sizeof(LimeModel), cudaMemcpyHostToDevice));

    cublasHandle_t cbh;
    cublasCreate(&cbh);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    float gen_ms = 0, infer_ms = 0, weight_ms = 0;
    float norm_ms = 0, build_ms = 0, solve_ms = 0, post_ms = 0;

    // =======================================================================
    // STAGE 1 : perturbation generation
    // =======================================================================
    CUDA_CALL(cudaEventRecord(t0));
    init_curand<<<(B + 127) / 128, 128>>>(ds, seed, B);
    if (!use_per_feature)
    {
        int blk = D < 1024 ? D : 1024;
        generate_perturbations_per_sample<<<B, blk>>>(
            dx0, dm, ds, dX, dz, B, D, mp, ns);
    }
    else
    {
        int blk = B < 1024 ? B : 1024;
        generate_perturbations_per_feature<<<D, blk>>>(
            dx0, dm, ds, dX, dz, B, D, mp, ns);
    }
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&gen_ms, t0, t1);

    // =======================================================================
    // STAGE 2 : black-box inference
    // =======================================================================
    CUDA_CALL(cudaEventRecord(t0));
    if (!use_cublas)
    {
        int grid = (B + 255) / 256;
        infer_custom<<<grid, 256>>>(dX, dmodel, dp, B);
    }
    else
    {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(cbh, CUBLAS_OP_T, D, B, &alpha, dX, D, dW, 1, &beta, dlog, 1);
        int grid = (B + 255) / 256;
        add_bias_sigmoid<<<grid, 256>>>(dlog, hb, B);
        CUDA_CALL(cudaMemcpy(dp, dlog, B * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&infer_ms, t0, t1);

    // =======================================================================
    // STAGE 3 : weights
    // =======================================================================
    CUDA_CALL(cudaEventRecord(t0));
    {
        int grid = (B + 255) / 256;
        distances_and_weights<<<grid, 256>>>(dX, dx0, dd, dw, B, D, kw);
    }
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&weight_ms, t0, t1);

    // =======================================================================
    // STAGE 4a : feature normalization (z-score, in place into dXn)
    // =======================================================================
    CUDA_CALL(cudaEventRecord(t0));
    {
        // Sums + sums-of-squares per feature.  Block reduction over B.
        int bs = 256;
        size_t shm = 2 * bs * sizeof(float);
        compute_feature_sums<<<D, bs, shm>>>(dX, d_sum, d_sumsq, B, D);

        finalize_mean_std<<<(D + 127) / 128, 128>>>(
            d_sum, d_sumsq, d_mean, d_std, B, D, 1e-8f);

        // 2D launch to standardize every entry.
        dim3 blk(16, 16);
        dim3 grd((B + blk.x - 1) / blk.x, (D + blk.y - 1) / blk.y);
        normalize_features<<<grd, blk>>>(dX, d_mean, d_std, dXn, B, D);
    }
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&norm_ms, t0, t1);

    int num_iters = 1;
    int solver_id = (solver == Solver::Cholesky) ? 0 : 1;

    if (solver == Solver::Cholesky)
    {
        // ===================================================================
        // STAGE 4b : build A and b, then Cholesky-solve A beta = b
        // ===================================================================
        CUDA_CALL(cudaEventRecord(t0));
        {
            int bs = 256;
            size_t shm = bs * sizeof(float);

            // A = X~^T diag(w) X~  (size N x N)
            dim3 grdA(N, N);
            build_normal_matrix<<<grdA, bs, shm>>>(dXn, dw, d_A, B, D);

            // b = X~^T diag(w) y
            build_normal_rhs<<<N, bs, shm>>>(dXn, dw, dp, d_b, B, D);

            // A += ridge * I  (single block)
            add_ridge<<<1, N>>>(d_A, N, ridge);
        }
        CUDA_CALL(cudaEventRecord(t1));
        CUDA_CALL(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&build_ms, t0, t1);

        CUDA_CALL(cudaEventRecord(t0));
        {
            int bs = 256;
            size_t shm_decomp = bs * sizeof(float);
            cholesky_decomp<<<1, bs, shm_decomp>>>(d_A, N);

            size_t shm_solve = (N + bs) * sizeof(float);
            cholesky_solve<<<1, bs, shm_solve>>>(d_A, d_b, d_beta_norm, N);
        }
        CUDA_CALL(cudaEventRecord(t1));
        CUDA_CALL(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&solve_ms, t0, t1);
    }
    else
    {
        // ===================================================================
        // STAGE 4b' : iterative gradient descent (alternative to Cholesky)
        // ===================================================================
        // Step size is rescaled by 1/sum_w so the user-facing learning rate
        // stays insensitive to the magnitude of the LIME similarity weights.
        // The fixed-point equation is unchanged, so the minimizer matches
        // the Cholesky path exactly.
        CUDA_CALL(cudaEventRecord(t0));
        {
            CUDA_CALL(cudaMemset(d_beta_norm, 0, N * sizeof(float)));
            int bs = 256;
            size_t shm = bs * sizeof(float);

            // sum_w := reduce_sum(dw)  -> 1 float on device, copied once to host
            float *d_sum_w;
            CUDA_CALL(cudaMalloc(&d_sum_w, sizeof(float)));
            reduce_sum<<<1, bs, shm>>>(dw, d_sum_w, B);
            float h_sum_w = 0.0f;
            CUDA_CALL(cudaMemcpy(&h_sum_w, d_sum_w, sizeof(float),
                                 cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaFree(d_sum_w));
            float lr_eff = gd_lr / (h_sum_w > 1e-12f ? h_sum_w : 1e-12f);

            for (int it = 0; it < gd_iters; ++it)
            {
                int grid = (B + 255) / 256;
                gd_compute_residuals<<<grid, 256>>>(
                    dXn, dp, d_beta_norm, d_res, B, D);

                gd_compute_gradient<<<N, bs, shm>>>(
                    dXn, dw, d_res, d_beta_norm, d_grad, B, D, ridge);

                int gu = (N + 127) / 128;
                gd_update_beta<<<gu, 128>>>(d_beta_norm, d_grad, N, lr_eff);
            }
            num_iters = gd_iters;
        }
        CUDA_CALL(cudaEventRecord(t1));
        CUDA_CALL(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&solve_ms, t0, t1);
        build_ms = 0.0f;
    }

    // =======================================================================
    // STAGE 4c : de-normalize, predict, R^2, surrogate-at-x0  (all on GPU)
    // =======================================================================
    CUDA_CALL(cudaEventRecord(t0));
    {
        int bs = 256;
        size_t shm = bs * sizeof(float);
        denormalize_coeff<<<1, bs, shm>>>(
            d_beta_norm, d_mean, d_std, d_coeff, d_intercept, D);

        // Read intercept once into a register so we can pass it by value
        // to the prediction kernels.  This is a single-float D2H copy.
        float h_intercept = 0.0f;
        CUDA_CALL(cudaMemcpy(&h_intercept, d_intercept, sizeof(float),
                             cudaMemcpyDeviceToHost));

        int grid = (B + 255) / 256;
        surrogate_predict<<<grid, 256>>>(dX, d_coeff, h_intercept,
                                          d_pred, B, D);

        size_t shm_r2 = 4 * bs * sizeof(float);
        weighted_r2_partials<<<1, bs, shm_r2>>>(d_pred, dp, dw, d_partials, B);

        surrogate_predict_x0<<<1, bs, shm>>>(d_coeff, h_intercept, dx0,
                                              d_x0_pred, D);
    }
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&post_ms, t0, t1);

    // -----------------------------------------------------------------------
    //              Pull the small final results back to the host
    // -----------------------------------------------------------------------
    std::vector<float> h_coeff(D);
    float h_intercept = 0.0f;
    float h_partials[4] = {0,0,0,0};
    float h_surr_x0 = 0.0f;
    CUDA_CALL(cudaMemcpy(h_coeff.data(), d_coeff,    D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&h_intercept,   d_intercept,    sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_partials,     d_partials, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&h_surr_x0,     d_x0_pred,      sizeof(float), cudaMemcpyDeviceToHost));

    // Weighted R^2: 1 - SS_res / SS_tot
    //   SS_res = sum w*(y - pred)^2
    //   SS_tot = sum w*y^2 - (sum w*y)^2 / sum w
    double sw   = h_partials[0];
    double swy  = h_partials[1];
    double swy2 = h_partials[2];
    double swr2 = h_partials[3];
    double ss_tot = swy2 - (swy * swy) / (sw > 1e-12 ? sw : 1e-12);
    double r2     = (ss_tot > 1e-12) ? 1.0 - swr2 / ss_tot : 0.0;

    // Black-box prediction at x0 = sigmoid( <W, x0> + bias ).  Computed on
    // host because it is a single dot product, and the model parameters are
    // already in host memory.
    double bb_logit = hb;
    for (int j = 0; j < D; ++j) bb_logit += hW[j] * hx0[j];
    float h_bb_x0 = (float)(1.0 / (1.0 + exp(-bb_logit)));

    float surr_total = norm_ms + build_ms + solve_ms + post_ms;
    float pipe_total = gen_ms + infer_ms + weight_ms + surr_total;

    printf("\n");
    printf("Timing (ms): gen %.3f  infer %.3f  weights %.3f  "
           "norm %.3f  build %.3f  solve %.3f  post %.3f\n",
           gen_ms, infer_ms, weight_ms,
           norm_ms, build_ms, solve_ms, post_ms);
    printf("Surrogate total %.3f ms   Full pipeline %.3f ms\n",
           surr_total, pipe_total);
    printf("Surrogate weighted R^2 = %.4f\n", r2);
    printf("At x0:  surrogate = %.4f   black_box = %.4f   |delta| = %.4f\n",
           h_surr_x0, h_bb_x0, fabs(h_surr_x0 - h_bb_x0));

    // Top-5 features by |coeff|
    {
        std::vector<int> idx(D);
        for (int i = 0; i < D; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fabs(h_coeff[a]) > fabs(h_coeff[b]);
        });
        printf("Top-5 features by |coeff|:\n");
        int show = D < 5 ? D : 5;
        for (int r = 0; r < show; ++r)
            printf("  %2d. feature_%-3d  %+.6f\n", r + 1, idx[r], h_coeff[idx[r]]);
    }

    // -----------------------------------------------------------------------
    //               Optional intermediate outputs (debug / repro)
    // -----------------------------------------------------------------------
    if (out_X)
    {
        std::vector<float> Xh((size_t)B * D);
        CUDA_CALL(cudaMemcpy(Xh.data(), dX, (size_t)B * D * sizeof(float),
                             cudaMemcpyDeviceToHost));
        FILE *f = fopen(out_X, "wb");
        fwrite(Xh.data(), sizeof(float), Xh.size(), f);
        fclose(f);
        printf("Saved X       -> %s\n", out_X);
    }
    if (out_preds)
    {
        std::vector<float> ph(B);
        CUDA_CALL(cudaMemcpy(ph.data(), dp, B * sizeof(float),
                             cudaMemcpyDeviceToHost));
        FILE *f = fopen(out_preds, "wb");
        fwrite(ph.data(), sizeof(float), B, f);
        fclose(f);
        printf("Saved preds   -> %s\n", out_preds);
    }
    if (out_weights)
    {
        std::vector<float> wh(B);
        CUDA_CALL(cudaMemcpy(wh.data(), dw, B * sizeof(float),
                             cudaMemcpyDeviceToHost));
        FILE *f = fopen(out_weights, "wb");
        fwrite(wh.data(), sizeof(float), B, f);
        fclose(f);
        printf("Saved weights -> %s\n", out_weights);
    }

    save_attributions(out_attribs, D, solver_id, ridge,
                      h_coeff.data(), h_intercept,
                      h_surr_x0, h_bb_x0, (float)r2,
                      surr_total, num_iters, B);

    // -----------------------------------------------------------------------
    //                                Cleanup
    // -----------------------------------------------------------------------
    cublasDestroy(cbh);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dx0);  cudaFree(dm);   cudaFree(dW);  cudaFree(dX);   cudaFree(dXn);
    cudaFree(dz);   cudaFree(dlog); cudaFree(dp);  cudaFree(dd);   cudaFree(dw);
    cudaFree(ds);   cudaFree(dmodel);
    cudaFree(d_sum); cudaFree(d_sumsq); cudaFree(d_mean); cudaFree(d_std);
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_beta_norm);
    cudaFree(d_coeff); cudaFree(d_intercept);
    cudaFree(d_pred); cudaFree(d_partials); cudaFree(d_x0_pred);
    cudaFree(d_res); cudaFree(d_grad);

    return 0;
}
