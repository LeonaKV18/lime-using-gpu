// main_m3_custom.cu
// End-to-end custom GPU-native LIME pipeline (Milestone 3): uses Cholesky + GD fallback.

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
//                          Model file I/O helpers (copied minimal)
// ---------------------------------------------------------------------------
static bool load_model_bin(const char *path, int &D, std::vector<float> &W, float &bias, std::vector<float> &x0, std::vector<float> &means)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open model file: %s\n", path); return false; }
    int file_D = 0;
    if (fread(&file_D, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    D = file_D;
    W.resize(D); x0.resize(D); means.resize(D);
    if (fread(W.data(), sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    if (fread(&bias, sizeof(float), 1, f) != 1) { fclose(f); return false; }
    if (fread(x0.data(), sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    if (fread(means.data(), sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    fclose(f);
    return true;
}


static void save_attributions(const char *path, int D, int solver_id, float ridge, const float *coeff, float intercept, float surr_x0, float bb_x0, float r2, float total_ms, int num_iters, int B)
{
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "[ERROR] Cannot write %s\n", path); return; }
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
    int D = 30; int B = 16384; float mp = 0.2f; float ns = 0.1f; float kw = 1.0f; unsigned long seed = 1234;
    enum class Solver { Cholesky, GD };
    Solver solver = Solver::Cholesky;
    float ridge = 1e-3f; int gd_iters = 500; float gd_lr = 1e-2f;
    const char *model_path = nullptr; const char *out_attribs = "attributions_gpu_custom.bin";
    const char *out_preds = nullptr; const char *out_weights = nullptr; const char *out_X = nullptr;

    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--D=", 4)) D = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--B=", 4)) B = atoi(argv[i] + 4);
        else if (!strcmp(argv[i], "--solver=cholesky")) solver = Solver::Cholesky;
        else if (!strcmp(argv[i], "--solver=gd")) solver = Solver::GD;
        else if (!strncmp(argv[i], "--ridge=", 8)) ridge = (float)atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--gd-iters=", 11)) gd_iters = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--gd-lr=", 8)) gd_lr = (float)atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--model=", 8)) model_path = argv[i] + 8;
        else if (!strcmp(argv[i], "--write-attributions")) out_attribs = argv[++i];
        else if (!strcmp(argv[i], "--write-preds")) out_preds = argv[++i];
        else if (!strcmp(argv[i], "--write-weights")) out_weights = argv[++i];
        else if (!strcmp(argv[i], "--write-X")) out_X = argv[++i];
    }

    std::vector<float> hx0, hm, hW; float hb = 0.0f;
    if (model_path)
    {
        if (!load_model_bin(model_path, D, hW, hb, hx0, hm)) { fprintf(stderr, "[ERROR] Failed to load model\n"); return 1; }
        printf("Loaded model from %s  D=%d\n", model_path, D);
    } else {
        hx0.resize(D); hm.resize(D); hW.resize(D); for (int i=0;i<D;++i){ hx0[i]=0.5f; hm[i]=0.5f; hW[i]=0.02f*(i+1);} hb=-1.0f;
    }

    int N = D + 1;
    printf("Config: D=%d  B=%d  solver=%s  ridge=%.2e\n", D, B, solver==Solver::Cholesky?"cholesky":"gd", ridge);

    // Device allocations (same as original, trimmed)
    float *dx0, *dm, *dW, *dX, *dXn, *dlog, *dp, *dd, *dw;
    unsigned char *dz; curandStatePhilox4_32_10_t *ds; LimeModel *dmodel;
    float *d_sum, *d_sumsq, *d_mean, *d_std; float *d_A, *d_b, *d_beta_norm; float *d_coeff, *d_intercept, *d_pred, *d_partials, *d_x0_pred; float *d_res, *d_grad;
    double *d_A_d, *d_b_d, *d_beta_d;

    CUDA_CALL(cudaMalloc(&dx0, D * sizeof(float))); CUDA_CALL(cudaMalloc(&dm, D * sizeof(float))); CUDA_CALL(cudaMalloc(&dW, D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dX, (size_t)B * D * sizeof(float))); CUDA_CALL(cudaMalloc(&dXn, (size_t)B * D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dz, (size_t)B * D * sizeof(unsigned char))); CUDA_CALL(cudaMalloc(&dlog, B * sizeof(float))); CUDA_CALL(cudaMalloc(&dp, B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dd, B * sizeof(float))); CUDA_CALL(cudaMalloc(&dw, B * sizeof(float))); CUDA_CALL(cudaMalloc(&ds, B * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CALL(cudaMalloc(&dmodel, sizeof(LimeModel)));

    CUDA_CALL(cudaMalloc(&d_sum, D * sizeof(float))); CUDA_CALL(cudaMalloc(&d_sumsq, D * sizeof(float))); CUDA_CALL(cudaMalloc(&d_mean, D * sizeof(float))); CUDA_CALL(cudaMalloc(&d_std, D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_A, (size_t)N * N * sizeof(float))); float *d_A_orig=nullptr; CUDA_CALL(cudaMalloc(&d_A_orig, (size_t)N * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_b, N * sizeof(float))); float *d_b_orig=nullptr; CUDA_CALL(cudaMalloc(&d_b_orig, N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_beta_norm, N * sizeof(float))); CUDA_CALL(cudaMalloc(&d_coeff, D * sizeof(float))); CUDA_CALL(cudaMalloc(&d_intercept, sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_pred, B * sizeof(float))); CUDA_CALL(cudaMalloc(&d_partials, 4 * sizeof(float))); CUDA_CALL(cudaMalloc(&d_x0_pred, sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_res, B * sizeof(float))); CUDA_CALL(cudaMalloc(&d_grad, N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_A_d, (size_t)N * N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_b_d, N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_beta_d, N * sizeof(double)));

    CUDA_CALL(cudaMemcpy(dx0, hx0.data(), D * sizeof(float), cudaMemcpyHostToDevice)); CUDA_CALL(cudaMemcpy(dm, hm.data(), D * sizeof(float), cudaMemcpyHostToDevice)); CUDA_CALL(cudaMemcpy(dW, hW.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    LimeModel h_model = create_lime_model(D, dW, hb); CUDA_CALL(cudaMemcpy(dmodel, &h_model, sizeof(LimeModel), cudaMemcpyHostToDevice));

    cublasHandle_t cbh; cublasCreate(&cbh);

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float gen_ms=0, infer_ms=0, weight_ms=0, norm_ms=0, build_ms=0, solve_ms=0, post_ms=0;

    // STAGE 1: perturbations
    init_curand<<<(B + 127) / 128, 128>>>(ds, seed, B);
    int blk = D < 1024 ? D : 1024;
    generate_perturbations_per_sample<<<B, blk>>>(dx0, dm, ds, dX, dz, B, D, mp, ns);

    // STAGE 2: inference (custom)
    { int grid = (B + 255) / 256; infer_custom<<<grid, 256>>>(dX, dmodel, dp, B); }

    // STAGE 3: weights
    { int grid = (B + 255) / 256; distances_and_weights<<<grid, 256>>>(dX, dx0, dd, dw, B, D, kw); }

    // STAGE 4a: z-score
    {
        int bs = 256; size_t shm = 2 * bs * sizeof(float);
        compute_feature_sums<<<D, bs, shm>>>(dX, d_sum, d_sumsq, B, D);
        finalize_mean_std<<<(D + 127) / 128, 128>>>(d_sum, d_sumsq, d_mean, d_std, B, D, 1e-8f);
        dim3 blk2(16,16); dim3 grd((B + blk2.x - 1)/blk2.x, (D + blk2.y - 1)/blk2.y);
        normalize_features<<<grd, blk2>>>(dX, d_mean, d_std, dXn, B, D);
    }

    int num_iters = 1; int solver_id = (solver==Solver::Cholesky)?0:1; bool used_double = false;

    if (solver == Solver::Cholesky)
    {
        // Build A and b in double precision
        int bs = 256; size_t shm = bs * sizeof(float);
        dim3 grdA(N, N);
        build_normal_matrix_d<<<grdA, bs, bs * sizeof(double)>>>(dXn, dw, d_A_d, B, D);
        build_normal_rhs_d<<<N, bs, bs * sizeof(double)>>>(dXn, dw, dp, d_b_d, B, D);
        add_ridge_d<<<1, N>>>(d_A_d, N, (double)ridge);

        size_t shm_decomp_d = bs * sizeof(double);
        cholesky_decomp_d<<<1, bs, shm_decomp_d>>>(d_A_d, N);
        size_t shm_solve_d = (N + bs) * sizeof(double);
        cholesky_solve_d<<<1, bs, shm_solve_d>>>(d_A_d, d_b_d, d_beta_d, N);

        std::vector<double> h_beta_d(N);
        CUDA_CALL(cudaMemcpy(h_beta_d.data(), d_beta_d, N * sizeof(double), cudaMemcpyDeviceToHost));
        bool bad = false;
        for (int i = 0; i < N; ++i)
            if (!isfinite(h_beta_d[i])) { bad = true; break; }

        if (bad)
        {
            printf("[WARN] Double-precision Cholesky failed; falling back to GD.\n");
            CUDA_CALL(cudaMemset(d_beta_norm, 0, N * sizeof(float)));
            int bs2 = 256; size_t shm2 = bs2 * sizeof(float);
            float *d_sum_w; CUDA_CALL(cudaMalloc(&d_sum_w, sizeof(float)));
            reduce_sum<<<1, bs2, shm2>>>(dw, d_sum_w, B);
            float h_sum_w = 0.0f; CUDA_CALL(cudaMemcpy(&h_sum_w, d_sum_w, sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CALL(cudaFree(d_sum_w));
            float lr_eff = gd_lr / (h_sum_w > 1e-12f ? h_sum_w : 1e-12f);
            for (int it=0; it<gd_iters; ++it)
            {
                int grid = (B + 255) / 256;
                gd_compute_residuals<<<grid, 256>>>(dXn, dp, d_beta_norm, d_res, B, D);
                gd_compute_gradient<<<N, bs2, shm2>>>(dXn, dw, d_res, d_beta_norm, d_grad, B, D, ridge);
                int gu = (N + 127) / 128;
                gd_update_beta<<<gu, 128>>>(d_beta_norm, d_grad, N, lr_eff);
            }
            num_iters = gd_iters; solver_id = 1;
        }
        else
        {
            std::vector<float> h_beta_f(N);
            for (int i = 0; i < N; ++i) h_beta_f[i] = (float)h_beta_d[i];
            CUDA_CALL(cudaMemcpy(d_beta_norm, h_beta_f.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            solver_id = 0;
            num_iters = 1;
            used_double = true;
        }
    }
    else
    {
        // GD direct path
        CUDA_CALL(cudaMemset(d_beta_norm, 0, N * sizeof(float)));
        int bs = 256; size_t shm = bs * sizeof(float);
        float *d_sum_w; CUDA_CALL(cudaMalloc(&d_sum_w, sizeof(float)));
        reduce_sum<<<1, bs, shm>>>(dw, d_sum_w, B);
        float h_sum_w = 0.0f; CUDA_CALL(cudaMemcpy(&h_sum_w, d_sum_w, sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CALL(cudaFree(d_sum_w));
        float lr_eff = gd_lr / (h_sum_w > 1e-12f ? h_sum_w : 1e-12f);
        for (int it=0; it<gd_iters; ++it)
        {
            int grid = (B + 255) / 256;
            gd_compute_residuals<<<grid, 256>>>(dXn, dp, d_beta_norm, d_res, B, D);
            gd_compute_gradient<<<N, bs, shm>>>(dXn, dw, d_res, d_beta_norm, d_grad, B, D, ridge);
            int gu = (N + 127) / 128;
            gd_update_beta<<<gu, 128>>>(d_beta_norm, d_grad, N, lr_eff);
        }
        num_iters = gd_iters; solver_id = 1;
    }

    // Post: denormalize, predict, r2
    {
        int bs = 256; size_t shm = bs * sizeof(float);
        if (used_double)
        {
            denormalize_coeff_d<<<1, bs, bs * sizeof(double)>>>(d_beta_d, d_mean, d_std, d_coeff, d_intercept, D);
        }
        else
        {
            denormalize_coeff<<<1, bs, shm>>>(d_beta_norm, d_mean, d_std, d_coeff, d_intercept, D);
        }
        float h_intercept = 0.0f; CUDA_CALL(cudaMemcpy(&h_intercept, d_intercept, sizeof(float), cudaMemcpyDeviceToHost));
        int grid = (B + 255) / 256; surrogate_predict<<<grid, 256>>>(dX, d_coeff, h_intercept, d_pred, B, D);
        size_t shm_r2 = 4 * bs * sizeof(float); weighted_r2_partials<<<1, bs, shm_r2>>>(d_pred, dp, dw, d_partials, B);
        surrogate_predict_x0<<<1, bs, shm>>>(d_coeff, h_intercept, dx0, d_x0_pred, D);
    }

    std::vector<float> h_coeff(D); float h_intercept=0.0f; float h_partials[4]={0,0,0,0}; float h_surr_x0=0.0f;
    CUDA_CALL(cudaMemcpy(h_coeff.data(), d_coeff, D * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CALL(cudaMemcpy(&h_intercept, d_intercept, sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CALL(cudaMemcpy(h_partials, d_partials, 4 * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CALL(cudaMemcpy(&h_surr_x0, d_x0_pred, sizeof(float), cudaMemcpyDeviceToHost));

    double sw = h_partials[0], swy = h_partials[1], swy2 = h_partials[2], swr2 = h_partials[3];
    double ss_tot = swy2 - (swy * swy) / (sw > 1e-12 ? sw : 1e-12); double r2 = (ss_tot > 1e-12) ? 1.0 - swr2 / ss_tot : 0.0;

    double bb_logit = hb; for (int j=0;j<D;++j) bb_logit += hW[j] * hx0[j]; float h_bb_x0 = (float)(1.0 / (1.0 + exp(-bb_logit)));

    printf("Surrogate weighted R^2 = %.4f\n", r2);
    printf("At x0:  surrogate = %.4f   black_box = %.4f   |delta| = %.4f\n", h_surr_x0, h_bb_x0, fabs(h_surr_x0 - h_bb_x0));

    // --------------------------------------------------
    // Optional dumps for downstream scripts (M2, CPU)
    // --------------------------------------------------
    if (out_X)
    {
        std::vector<float> hX(B * D);
        CUDA_CALL(cudaMemcpy(hX.data(), dX, B * D * sizeof(float), cudaMemcpyDeviceToHost));
        FILE *f = fopen(out_X, "wb");
        fwrite(hX.data(), sizeof(float), B * D, f);
        fclose(f);
        printf("Saved X -> %s\n", out_X);
    }

    if (out_preds)
    {
        std::vector<float> hpred(B);
        CUDA_CALL(cudaMemcpy(hpred.data(), dp, B * sizeof(float), cudaMemcpyDeviceToHost));
        FILE *f = fopen(out_preds, "wb");
        fwrite(hpred.data(), sizeof(float), B, f);
        fclose(f);
        printf("Saved preds -> %s\n", out_preds);
    }

    if (out_weights)
    {
        std::vector<float> hw(B);
        CUDA_CALL(cudaMemcpy(hw.data(), dw, B * sizeof(float), cudaMemcpyDeviceToHost));
        FILE *f = fopen(out_weights, "wb");
        fwrite(hw.data(), sizeof(float), B, f);
        fclose(f);
        printf("Saved weights -> %s\n", out_weights);
    }
    save_attributions(out_attribs, D, solver_id, ridge, h_coeff.data(), h_intercept, h_surr_x0, h_bb_x0, (float)r2, 0.0f, num_iters, B);

    cublasDestroy(cbh);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dx0); cudaFree(dm); cudaFree(dW); cudaFree(dX); cudaFree(dXn); cudaFree(dz); cudaFree(dlog); cudaFree(dp); cudaFree(dd); cudaFree(dw); cudaFree(ds); cudaFree(dmodel);
    cudaFree(d_sum); cudaFree(d_sumsq); cudaFree(d_mean); cudaFree(d_std);
    cudaFree(d_A); cudaFree(d_A_orig); cudaFree(d_b); cudaFree(d_b_orig); cudaFree(d_beta_norm);
    cudaFree(d_coeff); cudaFree(d_intercept); cudaFree(d_pred); cudaFree(d_partials); cudaFree(d_x0_pred); cudaFree(d_res); cudaFree(d_grad);
    cudaFree(d_A_d); cudaFree(d_b_d); cudaFree(d_beta_d);

    return 0;
}
