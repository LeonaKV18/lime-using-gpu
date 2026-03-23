#pragma once
#include <curand_kernel.h>
__global__ void init_curand(curandStatePhilox4_32_10_t *, unsigned long, int);
__global__ void generate_perturbations(const float *, const float *, curandStatePhilox4_32_10_t *, float *, unsigned char *, int, int, float, float);
__global__ void distances_and_weights(const float *, const float *, float *, float *, int, int, float);
