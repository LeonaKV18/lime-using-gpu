#pragma once
#include <curand_kernel.h>

// Generalized model structure for LIME inference
struct LimeModel
{
    float *W;        // Model weights (size D)
    float bias;      // Model bias (scalar)
    int D;           // Number of dimensions
};

__global__ void init_curand(curandStatePhilox4_32_10_t *, unsigned long, int);
__global__ void generate_perturbations(const float *, const float *, curandStatePhilox4_32_10_t *, float *, unsigned char *, int, int, float, float);
__global__ void distances_and_weights(const float *, const float *, float *, float *, int, int, float);
__global__ void infer_with_model(const float *, const LimeModel *, float *, int);
