#include "kernels.h"
#include <math.h>

__global__ void init_curand(curandStatePhilox4_32_10_t *s, unsigned long seed, int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B)
        curand_init(seed, i, 0, &s[i]);
}

// Per-Sample: B blocks x min(D,1024) threads
// One block owns one sample, threads cover all features
__global__ void generate_perturbations_per_sample(
    const float *x0, const float *means,
    curandStatePhilox4_32_10_t *states,
    float *X, unsigned char *zprime,
    int B, int D, float mp, float ns)
{
    int samp = blockIdx.x;
    int feat = threadIdx.x;
    if (samp >= B || feat >= D) return;

    curandStatePhilox4_32_10_t st = states[samp];
    float u = curand_uniform(&st);
    float v;
    unsigned char z;

    if (u < mp) { v = means[feat]; z = 0; }
    else        { v = x0[feat] + ns * curand_normal(&st); z = 1; }

    X[samp * D + feat]      = v;
    zprime[samp * D + feat] = z;

    if (feat == 0)
        states[samp] = st;
}

// Per-Feature: D blocks x min(B,1024) threads
// One block owns one feature, threads cover all samples
__global__ void generate_perturbations_per_feature(
    const float *x0, const float *means,
    curandStatePhilox4_32_10_t *states,
    float *X, unsigned char *zprime,
    int B, int D, float mp, float ns)
{
    int feat = blockIdx.x;
    if (feat >= D) return;

    float x0_f   = x0[feat];
    float mean_f = means[feat];

    for (int samp = threadIdx.x; samp < B; samp += blockDim.x)
    {
        curandStatePhilox4_32_10_t st = states[samp];
        skipahead((unsigned long long)feat, &st);

        float u = curand_uniform(&st);
        float v;
        unsigned char z;

        if (u < mp) { v = mean_f; z = 0; }
        else        { v = x0_f + ns * curand_normal(&st); z = 1; }

        X[samp * D + feat]      = v;
        zprime[samp * D + feat] = z;
    }
}

__global__ void distances_and_weights(
    const float *X, const float *x0,
    float *dist, float *w,
    int B, int D, float kw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    float s2  = 0.0f;
    int   row = i * D;
    for (int j = 0; j < D; ++j)
    {
        float d = X[row + j] - x0[j];
        s2 += d * d;
    }
    dist[i] = s2;
    w[i]    = expf(-s2 / (kw * kw));
}

// Custom inference: each thread does full dot-product for one sample
__global__ void infer_custom(
    const float *X, const LimeModel *model,
    float *pred, int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    float logit    = model->bias;
    int   row_start = i * model->D;
    for (int j = 0; j < model->D; ++j)
        logit += X[row_start + j] * model->W[j];

    pred[i] = 1.0f / (1.0f + expf(-logit));
}

// cuBLAS post-processing: fuse bias + sigmoid after Sgemv
__global__ void add_bias_sigmoid(float *logits, float bias, int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    logits[i] = 1.0f / (1.0f + expf(-(logits[i] + bias)));
}