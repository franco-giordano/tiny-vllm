cmake_minimum_required(VERSION 3.24)

project(my_tiny_vllm LANGUAGES CXX CUDA)

# Export compiler flags and include paths for clangd (used by Zed).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(my_tiny_vllm
  src/main.cpp
  src/kernels.cu
)

target_include_directories(my_tiny_vllm PRIVATE src)
target_include_directories(my_tiny_vllm PRIVATE include)

target_link_libraries(my_tiny_vllm PRIVATE CUDA::cudart CUDA::cublas)
