#include <cstdio>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include "kernels.h"
#include "utils.h"

// Reads model parameters written by train_model.py.
// Binary layout: [int32 D][float32*D W][float32 bias][float32*D x0][float32*D means]
// On success D is updated to the value stored in the file.
static bool load_model_bin(
    const char *path, int &D,
    std::vector<float> &W, float &bias,
    std::vector<float> &x0, std::vector<float> &means)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open model file: %s\n", path); return false; }

    int file_D = 0;
    if (fread(&file_D, sizeof(int), 1, f) != 1)           { fclose(f); return false; }
    D = file_D;
    W.resize(D); x0.resize(D); means.resize(D);
    if (fread(W.data(),     sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    if (fread(&bias,        sizeof(float), 1, f) != 1)         { fclose(f); return false; }
    if (fread(x0.data(),    sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    if (fread(means.data(), sizeof(float), D, f) != (size_t)D) { fclose(f); return false; }
    fclose(f);
    return true;
}

LimeModel create_lime_model(int D, float *dW, float bias)
{
    LimeModel m; m.W = dW; m.bias = bias; m.D = D;
    return m;
}

int main(int c, char **v)
{
    int   D    = 128;
    int   B    = 16384;
    float mp   = 0.2f;
    float ns   = 0.1f;
    float kw   = 1.0f;
    unsigned long seed = 1234;

    bool use_per_feature = false;
    bool use_cublas      = false;

    const char *rx          = nullptr;
    const char *wx          = nullptr;
    const char *wz          = nullptr;
    const char *out_preds   = nullptr;
    const char *out_weights = nullptr;
    const char *model_path  = nullptr;

    for (int i = 1; i < c; ++i)
    {
        if      (!strncmp(v[i], "--D=", 4))               D  = atoi(v[i] + 4);
        else if (!strncmp(v[i], "--B=", 4))               B  = atoi(v[i] + 4);
        else if (!strcmp(v[i], "--perturb=per-feature"))  use_per_feature = true;
        else if (!strcmp(v[i], "--perturb=per-sample"))   use_per_feature = false;
        else if (!strcmp(v[i], "--infer=cublas"))          use_cublas      = true;
        else if (!strcmp(v[i], "--infer=custom"))          use_cublas      = false;
        else if (!strcmp(v[i], "--read-X"))                rx          = v[++i];
        else if (!strcmp(v[i], "--write-X"))               wx          = v[++i];
        else if (!strcmp(v[i], "--write-zprime"))          wz          = v[++i];
        else if (!strcmp(v[i], "--write-preds"))           out_preds   = v[++i];
        else if (!strcmp(v[i], "--write-weights"))         out_weights = v[++i];
        else if (!strncmp(v[i], "--model=", 8))            model_path  = v[i] + 8;
    }

    std::vector<float> hx0, hm, hW;
    float hb = 0.0f;

    if (model_path)
    {
        // D is overridden by the value embedded in the model file.
        if (!load_model_bin(model_path, D, hW, hb, hx0, hm))
        {
            fprintf(stderr, "[ERROR] Failed to load model from %s\n", model_path);
            return 1;
        }
        printf("Loaded model from %s  D=%d\n", model_path, D);
    }
    else
    {
        // Synthetic fallback used for performance benchmarking where D is swept freely.
        hx0.resize(D); hm.resize(D); hW.resize(D);
        for (int i = 0; i < D; ++i)
        {
            hx0[i] = (i % 5 == 0) ? 1.0f : 0.5f;
            hm[i]  = 0.5f;
            hW[i]  = 0.02f * (i + 1);
        }
        hb = -1.0f;
    }

    printf("Config: D=%d  B=%d  perturb=%s  infer=%s\n",
           D, B,
           use_per_feature ? "per-feature" : "per-sample",
           use_cublas      ? "cublas"       : "custom");

    float *dx0, *dm, *dW, *dX, *dlog, *dp, *dd, *dw;
    unsigned char *dz;
    curandStatePhilox4_32_10_t *ds;
    LimeModel *dmodel;

    CUDA_CALL(cudaMalloc(&dx0,   D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dm,    D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dW,    D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dX,    (size_t)B * D * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dz,    (size_t)B * D * sizeof(unsigned char)));
    CUDA_CALL(cudaMalloc(&dlog,  B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dp,    B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dd,    B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dw,    B * sizeof(float)));
    CUDA_CALL(cudaMalloc(&ds,    B * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CALL(cudaMalloc(&dmodel, sizeof(LimeModel)));

    CUDA_CALL(cudaMemcpy(dx0, hx0.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dm,  hm.data(),  D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dW,  hW.data(),  D * sizeof(float), cudaMemcpyHostToDevice));

    LimeModel h_model = create_lime_model(D, dW, hb);
    CUDA_CALL(cudaMemcpy(dmodel, &h_model, sizeof(LimeModel), cudaMemcpyHostToDevice));

    cublasHandle_t cbh;
    cublasCreate(&cbh);

    // cuBLAS warmup to exclude initialization overhead from timed runs
    {
        const float a = 1.0f, b = 0.0f;
        float *dtemp;
        CUDA_CALL(cudaMalloc(&dtemp, B * sizeof(float)));
        cublasSgemv(cbh, CUBLAS_OP_T, D, B, &a, dX, D, dW, 1, &b, dtemp, 1);
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaFree(dtemp));
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    float g = 0.0f, gi = 0.0f, gp = 0.0f, gio = 0.0f, infer_ms = 0.0f, w_ms = 0.0f;

    // ===== Stage 1: Perturbation =====
    if (rx)
    {
        CUDA_CALL(cudaEventRecord(t0));
        FILE *f = fopen(rx, "rb");
        if (!f) { fprintf(stderr, "[ERROR] Cannot open %s\n", rx); return 1; }
        std::vector<float> X((size_t)B * D);
        fread(X.data(), sizeof(float), (size_t)B * D, f);
        fclose(f);
        CUDA_CALL(cudaMemcpy(dX, X.data(), (size_t)B * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaEventRecord(t1));
        CUDA_CALL(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&gio, t0, t1);
        g = gio;
        if (wz) printf("[WARN] --write-zprime ignored with --read-X\n");
    }
    else
    {
        CUDA_CALL(cudaEventRecord(t0));
        init_curand<<<(B + 127) / 128, 128>>>(ds, seed, B);
        CUDA_CALL(cudaEventRecord(t1));
        CUDA_CALL(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&gi, t0, t1);

        CUDA_CALL(cudaEventRecord(t0));
        if (!use_per_feature)
        {
            int blk = min(D, 1024);
            generate_perturbations_per_sample<<<B, blk>>>(
                dx0, dm, ds, dX, dz, B, D, mp, ns);
        }
        else
        {
            int blk = min(B, 1024);
            generate_perturbations_per_feature<<<D, blk>>>(
                dx0, dm, ds, dX, dz, B, D, mp, ns);
        }
        CUDA_CALL(cudaEventRecord(t1));
        CUDA_CALL(cudaEventSynchronize(t1));
        cudaEventElapsedTime(&gp, t0, t1);
        g = gi + gp;

        if (wx)
        {
            std::vector<float> X((size_t)B * D);
            CUDA_CALL(cudaMemcpy(X.data(), dX, (size_t)B * D * sizeof(float), cudaMemcpyDeviceToHost));
            FILE *f = fopen(wx, "wb");
            fwrite(X.data(), sizeof(float), (size_t)B * D, f);
            fclose(f);
        }
        if (wz)
        {
            std::vector<unsigned char> Z((size_t)B * D);
            CUDA_CALL(cudaMemcpy(Z.data(), dz, (size_t)B * D, cudaMemcpyDeviceToHost));
            FILE *f = fopen(wz, "wb");
            fwrite(Z.data(), 1, (size_t)B * D, f);
            fclose(f);
        }
    }

    // ===== Stage 2: Inference =====
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

    // ===== Stage 3: Distances + Weights =====
    CUDA_CALL(cudaEventRecord(t0));
    {
        int grid = (B + 255) / 256;
        distances_and_weights<<<grid, 256>>>(dX, dx0, dd, dw, B, D, kw);
    }
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&w_ms, t0, t1);

    std::vector<float> hp(B), hw(B);
    CUDA_CALL(cudaMemcpy(hp.data(), dp, B * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hw.data(), dw, B * sizeof(float), cudaMemcpyDeviceToHost));

    double mpred = 0.0, mwei = 0.0;
    for (int k = 0; k < B; ++k) { mpred += hp[k]; mwei += hw[k]; }

    printf("Timing detail (ms): gen_init %.3f  perturb %.3f  read_x_io %.3f\n", gi, gp, gio);
    printf("Timing (ms): gen %.3f  infer %.3f  weights %.3f  total %.3f\n",
           g, infer_ms, w_ms, g + infer_ms + w_ms);
    printf("Means: preds %.5f  weights %.5f\n", mpred / B, mwei / B);

    const char *preds_file   = out_preds   ? out_preds   : "preds.bin";
    const char *weights_file = out_weights ? out_weights : "weights.bin";
    FILE *f1 = fopen(preds_file,   "wb"); fwrite(hp.data(), sizeof(float), B, f1); fclose(f1);
    FILE *f2 = fopen(weights_file, "wb"); fwrite(hw.data(), sizeof(float), B, f2); fclose(f2);

    cublasDestroy(cbh);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dx0);  cudaFree(dm);   cudaFree(dW);  cudaFree(dX);
    cudaFree(dz);   cudaFree(dlog); cudaFree(dp);  cudaFree(dd);
    cudaFree(dw);   cudaFree(ds);   cudaFree(dmodel);

    return 0;
}