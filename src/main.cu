#include <cstdio>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include "kernels.h"
#include "utils.h"

__global__ void add_bias(float *z, float bias, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    z[i] += bias;
}

__global__ void apply_sigmoid(float *z, float *p, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = 1.0f / (1.0f + expf(-z[i]));
}

int main(int c, char **v)
{
  int D = 128, B = 16384;
  float mp = 0.2f, ns = 0.1f, kw = 1.0f;
  unsigned long seed = 1234;
  const char *rx = nullptr;
  const char *wx = nullptr;
  const char *wz = nullptr;
  for (int i = 1; i < c; ++i)
  {
    if (!strncmp(v[i], "--D=", 4))
      D = atoi(v[i] + 4);
    else if (!strncmp(v[i], "--B=", 4))
      B = atoi(v[i] + 4);
    else if (!strcmp(v[i], "--read-X"))
      rx = v[++i];
    else if (!strcmp(v[i], "--write-X"))
      wx = v[++i];
    else if (!strcmp(v[i], "--write-zprime"))
      wz = v[++i];
  }

  std::vector<float> hx0(D), hm(D), hW(D);
  for (int i = 0; i < D; ++i)
  {
    hx0[i] = (i % 5) ? 0.5f : 1.0f;
    hm[i] = 0.5f;
    hW[i] = 0.02f * (i + 1);
  }
  float hb = -1.0f;
  float *dx0, *dm, *dW, *dX, *dlog, *dp, *dd, *dw;
  unsigned char *dz;
  curandStatePhilox4_32_10_t *ds;
  CUDA_CALL(cudaMalloc(&dx0, D * 4));
  CUDA_CALL(cudaMalloc(&dm, D * 4));
  CUDA_CALL(cudaMalloc(&dW, D * 4));
  CUDA_CALL(cudaMalloc(&dX, (size_t)B * D * 4));
  CUDA_CALL(cudaMalloc(&dz, (size_t)B * D));
  CUDA_CALL(cudaMalloc(&dlog, B * 4));
  CUDA_CALL(cudaMalloc(&dp, B * 4));
  CUDA_CALL(cudaMalloc(&dd, B * 4));
  CUDA_CALL(cudaMalloc(&dw, B * 4));
  CUDA_CALL(cudaMalloc(&ds, B * sizeof(curandStatePhilox4_32_10_t)));
  CUDA_CALL(cudaMemcpy(dx0, hx0.data(), D * 4, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dm, hm.data(), D * 4, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dW, hW.data(), D * 4, cudaMemcpyHostToDevice));

  cublasHandle_t h;
  cublasCreate(&h);
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  float g = 0, gi = 0, gp = 0, gio = 0, i = 0, w = 0;

  // Generate
  CUDA_CALL(cudaEventRecord(t0));
  if (rx)
  {
    CUDA_CALL(cudaEventRecord(t0));
    FILE *f = fopen(rx, "rb");
    std::vector<float> X(B * D);
    fread(X.data(), 4, B * D, f);
    fclose(f);
    CUDA_CALL(cudaMemcpy(dX, X.data(), B * D * 4, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&gio, t0, t1);
    g = gio;
    if (wz)
      printf("Warning: --write-zprime requested with --read-X. zprime is only available when perturbations are generated in this run.\n");
  }
  else
  {
    CUDA_CALL(cudaEventRecord(t0));
    init_curand<<<(B + 127) / 128, 128>>>(ds, seed, B);
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&gi, t0, t1);

    CUDA_CALL(cudaEventRecord(t0));
    generate_perturbations<<<B, min(D, 1024)>>>(dx0, dm, ds, dX, dz, B, D, mp, ns);
    CUDA_CALL(cudaEventRecord(t1));
    CUDA_CALL(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&gp, t0, t1);
    g = gi + gp;

    if (wx)
    {
      std::vector<float> X(B * D);
      CUDA_CALL(cudaMemcpy(X.data(), dX, B * D * 4, cudaMemcpyDeviceToHost));
      FILE *f = fopen(wx, "wb");
      fwrite(X.data(), 4, B * D, f);
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

  // Inference via cuBLAS
  // X is row-major B×D: in memory X[samp*D+feat].
  // cuBLAS is column-major, so treat the flat array as a D×B column-major matrix
  // (lda=D), then use CUBLAS_OP_T to compute (A^T)*W = X*W with result length B.
  CUDA_CALL(cudaEventRecord(t0));
  const float a = 1.0f, b = 0.0f;
  cublasSgemv(h, CUBLAS_OP_T, D, B, &a, dX, D, dW, 1, &b, dlog, 1);
  add_bias<<<(B + 255) / 256, 256>>>(dlog, hb, B);
  apply_sigmoid<<<(B + 255) / 256, 256>>>(dlog, dp, B);
  CUDA_CALL(cudaEventRecord(t1));
  CUDA_CALL(cudaEventSynchronize(t1));
  cudaEventElapsedTime(&i, t0, t1);

  // Weights
  CUDA_CALL(cudaEventRecord(t0));
  distances_and_weights<<<(B + 255) / 256, 256>>>(dX, dx0, dd, dw, B, D, kw);
  CUDA_CALL(cudaEventRecord(t1));
  CUDA_CALL(cudaEventSynchronize(t1));
  cudaEventElapsedTime(&w, t0, t1);

  std::vector<float> hp(B), hw(B);
  CUDA_CALL(cudaMemcpy(hp.data(), dp, B * 4, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hw.data(), dw, B * 4, cudaMemcpyDeviceToHost));
  double mpred = 0, mwei = 0;
  for (int k = 0; k < B; ++k)
  {
    mpred += hp[k];
    mwei += hw[k];
  }
  printf("Timing detail (ms): gen_init %.3f perturb %.3f read_x_io %.3f\n", gi, gp, gio);
  printf("Timing (ms): gen %.3f infer %.3f weights %.3f total %.3f\n", g, i, w, g + i + w);
  printf("Means: preds %.5f weights %.5f\n", mpred / B, mwei / B);
  FILE *f1 = fopen("preds.bin", "wb");
  fwrite(hp.data(), 4, B, f1);
  fclose(f1);
  FILE *f2 = fopen("weights.bin", "wb");
  fwrite(hw.data(), 4, B, f2);
  fclose(f2);
  cublasDestroy(h);
  cudaFree(dx0);
  cudaFree(dm);
  cudaFree(dW);
  cudaFree(dX);
  cudaFree(dz);
  cudaFree(dlog);
  cudaFree(dp);
  cudaFree(dd);
  cudaFree(dw);
  cudaFree(ds);
}
