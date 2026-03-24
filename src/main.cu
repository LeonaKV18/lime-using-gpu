#include <cstdio>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include "kernels.h"
#include "utils.h"

__global__ void add_bias(float *z, float bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] += bias;
}

__global__ void apply_sigmoid(float *z, float *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.0f / (1.0f + expf(-z[i]));
}

LimeModel create_lime_model(int D, float *dW, float bias) {
  LimeModel model;
  model.W = dW;
  model.bias = bias;
  model.D = D;
  return model;
}

int main(int c, char **v) {
  int D = 128, B = 16384;
  float mp = 0.2f, ns = 0.1f, kw = 1.0f;
  unsigned long seed = 1234;
  
  std::vector<float> hx0(D), hm(D), hW(D);
  for (int i = 0; i < D; ++i) {
    hx0[i] = (i % 5) ? 0.5f : 1.0f;
    hm[i] = 0.5f;
    hW[i] = 0.02f * (i + 1);
  }
  float hb = -1.0f; 
  
  float *dx0, *dm, *dW, *dX, *dlog, *dp, *dd, *dw;
  unsigned char *dz;
  curandStatePhilox4_32_10_t *ds;
  LimeModel *dmodel;  
  
  CUDA_CALL(cudaMalloc(&dx0, D * 4));
  CUDA_CALL(cudaMalloc(&dm, D * 4));
  CUDA_CALL(cudaMalloc(&dW, D * 4));
  CUDA_CALL(cudaMalloc(&dX, (size_t)B * D * 4));
  CUDA_CALL(cudaMalloc(&dz, (size_t)B * D));
  CUDA_CALL(cudaMalloc(&dlog, B * 4));
  CUDA_CALL(cudaMalloc(&dp, B * 4));
  CUDA_CALL(cudaMalloc(&dd, B * 4));
  CUDA_CALL(cudaMalloc(&dw, B * 4));
  
  // [THE FIX: MEMORY ALLOCATION]
  // We allocate B * D states instead of just B so every feature gets its own random stream.
  CUDA_CALL(cudaMalloc(&ds, B * D * sizeof(curandStatePhilox4_32_10_t)));
  
  CUDA_CALL(cudaMalloc(&dmodel, sizeof(LimeModel)));
  CUDA_CALL(cudaMemcpy(dx0, hx0.data(), D * 4, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dm, hm.data(), D * 4, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dW, hW.data(), D * 4, cudaMemcpyHostToDevice));

  LimeModel h_model = create_lime_model(D, dW, hb);
  CUDA_CALL(cudaMemcpy(dmodel, &h_model, sizeof(LimeModel), cudaMemcpyHostToDevice));

  cublasHandle_t h;
  cublasCreate(&h);
  
  // [THE FIX: THE PROFILING ILLUSION]
  // We force the GPU to run the ENTIRE cuBLAS pipeline once *before* we start the timer.
  // This forces the GPU to load the cuBLAS instructions into active memory (fixing the cold start),
  // so the stopwatch measures the raw math speed, not the library load time.
  {
    const float alpha = 1.0f, beta = 0.0f;
    int grid_size_infer = (B + 255) / 256;
    cublasSgemv(h, CUBLAS_OP_T, D, B, &alpha, dX, D, dW, 1, &beta, dlog, 1);
    add_bias<<<grid_size_infer, 256>>>(dlog, hb, B);
    apply_sigmoid<<<grid_size_infer, 256>>>(dlog, dp, B);
    CUDA_CALL(cudaDeviceSynchronize()); 
  }
  
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  float g = 0, gi = 0, gp = 0, i = 0, w = 0;

  // ========== Stage 1: Generate perturbations ==========
  CUDA_CALL(cudaEventRecord(t0));
  
  // [THE FIX: RNG INITIALIZATION]
  // We launch enough threads to initialize all B * D random states.
  int total_elements = B * D;
  init_curand<<<(total_elements + 255) / 256, 256>>>(ds, seed, total_elements);
  
  CUDA_CALL(cudaEventRecord(t1));
  CUDA_CALL(cudaEventSynchronize(t1));
  cudaEventElapsedTime(&gi, t0, t1);
  
  CUDA_CALL(cudaEventRecord(t0));
  
  // --- BASELINE: PER-SAMPLE (Now mathematically correct) ---
  //int shared_bytes = 2 * D * sizeof(float);
  //int block_size = min(D, 1024);
  //generate_perturbations<<<B, block_size, shared_bytes>>>(dx0, dm, ds, dX, dz, B, D, mp, ns);

  // --- EXPERIMENT: PER-FEATURE (Patched for race conditions) ---
   dim3 grid(D, (B + 255) / 256);
   dim3 block(256);
   generate_perturbations_per_feature<<<grid, block>>>(dx0, dm, ds, dX, dz, B, D, mp, ns);
  
  CUDA_CALL(cudaEventRecord(t1));
  CUDA_CALL(cudaEventSynchronize(t1));
  cudaEventElapsedTime(&gp, t0, t1);
  g = gi + gp;

  // ========== Stage 2: Inference ==========
  CUDA_CALL(cudaEventRecord(t0));
  
  // --- BASELINE: CUSTOM KERNEL ---
  //int shared_bytes_infer = D * sizeof(float);
  //int grid_size_infer = (B + 255) / 256;
  //infer_with_model<<<grid_size_infer, 256, shared_bytes_infer>>>(dX, dmodel, dp, B);

  // --- EXPERIMENT: cuBLAS INFERENCE ---
   const float alpha = 1.0f;
   const float beta = 0.0f;
   int grid_size_infer = (B + 255) / 256;
   cublasSgemv(h, CUBLAS_OP_T, D, B, &alpha, dX, D, dW, 1, &beta, dlog, 1);
   add_bias<<<grid_size_infer, 256>>>(dlog, hb, B);
   apply_sigmoid<<<grid_size_infer, 256>>>(dlog, dp, B);
  
  CUDA_CALL(cudaEventRecord(t1));
  CUDA_CALL(cudaEventSynchronize(t1));
  cudaEventElapsedTime(&i, t0, t1);

  // ========== Stage 3: Compute distances and weights ==========
  CUDA_CALL(cudaEventRecord(t0));
  
  int shared_bytes_weights = D * sizeof(float);
  int grid_size_weights = (B + 255) / 256;
  distances_and_weights<<<grid_size_weights, 256, shared_bytes_weights>>>(dX, dx0, dd, dw, B, D, kw);
  
  CUDA_CALL(cudaEventRecord(t1));
  CUDA_CALL(cudaEventSynchronize(t1));
  cudaEventElapsedTime(&w, t0, t1);

  std::vector<float> hp(B), hw(B);
  CUDA_CALL(cudaMemcpy(hp.data(), dp, B * 4, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hw.data(), dw, B * 4, cudaMemcpyDeviceToHost));
  
  double mpred = 0, mwei = 0;
  for (int k = 0; k < B; ++k) {
    mpred += hp[k];
    mwei += hw[k];
  }
  
  printf("\n--- PATCHED BENCHMARK RESULTS ---\n");
  printf("Timing detail (ms): gen_init %.3f perturb %.3f\n", gi, gp);
  printf("Timing (ms): gen %.3f infer %.3f weights %.3f total %.3f\n", g, i, w, g + i + w);
  printf("Means: preds %.5f weights %.5f\n", mpred / B, mwei / B);
  
  cublasDestroy(h);
  cudaFree(dx0); cudaFree(dm); cudaFree(dW); cudaFree(dX);
  cudaFree(dz); cudaFree(dlog); cudaFree(dp); cudaFree(dd);
  cudaFree(dw); cudaFree(ds); cudaFree(dmodel);
  
  return 0;
}