cmake_minimum_required(VERSION 3.18)
project(cuda_minimal LANGUAGES CXX CUDA)

# C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G -O0")
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -O0")
endif()

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(sgemm_v0 sgemm_v0.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v0  PRIVATE CUDA::cudart)

add_executable(sgemm_v1 sgemm_v1.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v1  PRIVATE CUDA::cudart)

add_executable(sgemm_v2 sgemm_v2.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v2  PRIVATE CUDA::cudart)

add_executable(sgemm_v3  sgemm_v3.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v3  PRIVATE CUDA::cudart)