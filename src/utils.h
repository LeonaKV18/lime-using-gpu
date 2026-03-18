#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CALL(x)                                                                             \
    do                                                                                           \
    {                                                                                            \
        cudaError_t e = (x);                                                                     \
        if (e != cudaSuccess)                                                                    \
        {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)
