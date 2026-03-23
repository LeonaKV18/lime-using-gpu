#include "kernels.h"
#include <math.h>

__global__ void init_curand(curandStatePhilox4_32_10_t *s, unsigned long seed, int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B)
        curand_init(seed, i, 0, &s[i]);
}

__global__ void generate_perturbations(const float *x0, const float *means, curandStatePhilox4_32_10_t *s, float *X, unsigned char *zprime, int B, int D, float mp, float ns)
{
    int samp = blockIdx.x, feat = threadIdx.x;
    if (samp >= B || feat >= D)
        return;
    int idx = samp * D + feat;
    curandStatePhilox4_32_10_t st = s[samp];
    float u = curand_uniform(&st), v;
    unsigned char z;
    if (u < mp)
    {
        v = means[feat];
        z = 0;
    }
    else
    {
        v = x0[feat] + ns * curand_normal(&st);
        z = 1;
    }
    X[idx] = v;
    zprime[idx] = z;
    s[samp] = st;
}

__global__ void distances_and_weights(const float *X, const float *x0, float *dist, float *w, int B, int D, float kw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B)
        return;
    float s2 = 0;
    int b = i * D;
    for (int j = 0; j < D; ++j)
    {
        float d = X[b + j] - x0[j];
        s2 += d * d;
    }
    dist[i] = s2;
    w[i] = expf(-s2 / (kw * kw));
}
