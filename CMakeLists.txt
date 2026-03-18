cmake_minimum_required(VERSION 3.18)
project(lime_gpu_m1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Locate CUDA toolkit (includes cuBLAS + cuRAND)
find_package(CUDAToolkit REQUIRED)

# Define executable and source files
add_executable(lime_m1
  src/main.cu
  src/kernels.cu
)

# Include headers
target_include_directories(lime_m1 PRIVATE src)

# Link CUDA libraries
target_link_libraries(lime_m1 PUBLIC CUDA::cublas CUDA::curand)
