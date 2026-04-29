cmake_minimum_required(VERSION 3.18)
project(lime_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Locate CUDA toolkit (includes cuBLAS + cuRAND)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Milestone 1 / Milestone 2 executable
#   Stages : perturbation, inference, weights
# ---------------------------------------------------------------------------
add_executable(lime_m1
  src/main.cu
  src/kernels.cu
)
target_include_directories(lime_m1 PRIVATE src)
target_link_libraries(lime_m1 PUBLIC CUDA::cublas CUDA::curand)

# ---------------------------------------------------------------------------
# Milestone 3 executable
#   Adds GPU-native weighted linear regression (Cholesky and gradient descent)
#   on top of the M1 kernels for a fully end-to-end GPU LIME pipeline.
# ---------------------------------------------------------------------------
add_executable(lime_m3
  src/main_m3.cu
  src/kernels.cu
  src/m3_kernels.cu
)
target_include_directories(lime_m3 PRIVATE src)
target_link_libraries(lime_m3 PUBLIC CUDA::cublas CUDA::curand)
