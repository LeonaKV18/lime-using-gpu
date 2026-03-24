#pragma once
#include <curand_kernel.h>

struct LimeModel
{
    float *W;
    float bias;
    int D;
};

__global__ void init_curand(curandStatePhilox4_32_10_t *, unsigned long, int);

__global__ void generate_perturbations_per_sample(
    const float *, const float *,
    curandStatePhilox4_32_10_t *,
    float *, unsigned char *,
    int, int, float, float);

__global__ void generate_perturbations_per_feature(
    const float *, const float *,
    curandStatePhilox4_32_10_t *,
    float *, unsigned char *,
    int, int, float, float);

__global__ void distances_and_weights(
    const float *, const float *,
    float *, float *,
    int, int, float);

__global__ void infer_custom(
    const float *, const LimeModel *, float *, int);

__global__ void add_bias_sigmoid(float *, float, int);