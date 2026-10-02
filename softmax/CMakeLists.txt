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

# 添加可执行文件: 将测试框架与核函数实现联合编译链接
add_executable(softmax_benchmark softmax_benchmark.cu softmax_naive.cu)
