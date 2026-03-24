#include <curand_kernel.h>
#include "kernels.h"

__global__ void init_curand(curandStatePhilox4_32_10_t *s, unsigned long seed, int total_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements)
        // [THE FIX: RNG STATE] 
        // We now initialize one random state for EVERY single feature of every sample (B * D),
        // instead of just one state per sample. 
        curand_init(seed, i, 0, &s[i]);
}

__global__ void generate_perturbations(const float *x0, const float *means, curandStatePhilox4_32_10_t *s, float *X, unsigned char *zprime, int B, int D, float mp, float ns)
{
    extern __shared__ char shared_mem[];
    float *shared_x0 = (float *)shared_mem;
    float *shared_means = shared_x0 + D;

    for (int idx = threadIdx.x; idx < D; idx += blockDim.x) {
        shared_x0[idx] = x0[idx];
        shared_means[idx] = means[idx];
    }
    __syncthreads();

    int samp = blockIdx.x, feat = threadIdx.x;
    if (samp >= B || feat >= D) return;
    
    // [THE FIX: FATAL MATH BUG]
    // In the original code, all 128 threads in a block shared the exact same random state s[samp].
    // This meant LIME was perturbing ALL features identically! 
    // Now, every thread grabs its own UNIQUE random state based on its exact 1D index.
    int idx = samp * D + feat;
    curandStatePhilox4_32_10_t st = s[idx]; 
    
    float u = curand_uniform(&st), v;
    unsigned char z;
    if (u < mp) {
        v = shared_means[feat];
        z = 0;
    } else {
        v = shared_x0[feat] + ns * curand_normal(&st);
        z = 1;
    }
    X[idx] = v;
    zprime[idx] = z;
    
    // [THE FIX: RACE CONDITION]
    // Because each thread owns 's[idx]', there are no more memory collisions when writing the state back.
    s[idx] = st;
}

__global__ void generate_perturbations_per_feature(const float *x0, const float *means, curandStatePhilox4_32_10_t *s, float *X, unsigned char *zprime, int B, int D, float mp, float ns)
{
    int feat = blockIdx.x; 
    int samp = threadIdx.x + blockIdx.y * blockDim.x; 
    if (samp >= B || feat >= D) return;
    
    int idx = samp * D + feat;
    
    // [THE FIX: RACE CONDITION] 
    // This per-feature kernel originally crashed because multiple threads fought over s[samp].
    // Now that the array has B*D independent states, this works flawlessly.
    curandStatePhilox4_32_10_t st = s[idx]; 
    float u = curand_uniform(&st), v;
    unsigned char z;
    
    if (u < mp) {
        v = means[feat];
        z = 0;
    } else {
        v = x0[feat] + ns * curand_normal(&st);
        z = 1;
    }
    X[idx] = v;
    zprime[idx] = z;
    s[idx] = st; 
}

__global__ void distances_and_weights(const float *X, const float *x0, float *dist, float *w, int B, int D, float kw)
{
    extern __shared__ float shared_x0[];
    for (int idx = threadIdx.x; idx < D; idx += blockDim.x) {
        shared_x0[idx] = x0[idx];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float s2 = 0;
    int b = i * D;
    for (int j = 0; j < D; ++j) {
        float d = X[b + j] - shared_x0[j];
        s2 += d * d;
    }
    dist[i] = s2;
    w[i] = expf(-s2 / (kw * kw));
}

__global__ void infer_with_model(const float *X, const LimeModel *model, float *pred, int B)
{
    extern __shared__ float shared_w[];
    for (int idx = threadIdx.x; idx < model->D; idx += blockDim.x) {
        shared_w[idx] = model->W[idx];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    float logit = model->bias;
    int row_start = i * model->D;
    for (int j = 0; j < model->D; ++j) {
        logit += X[row_start + j] * shared_w[j];
    }
    pred[i] = 1.0f / (1.0f + expf(-logit));
}