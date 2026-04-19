cmake_minimum_required(VERSION 3.18)
project(cuda_minimal LANGUAGES CXX CUDA)

# C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G -O0")
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -O0")
endif()

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(reduce_v1_v2 src/reduce_v1_v2.cu)

# 链接 CUDA 运行时
target_link_libraries(reduce_v1_v2  PRIVATE CUDA::cudart)

# 添加可执行文件
add_executable(reduce_v3 src/reduce_v3.cu)

# 链接 CUDA 运行时
target_link_libraries(reduce_v3  PRIVATE CUDA::cudart)

# 添加可执行文件
add_executable(reduce_v4 src/reduce_v4.cu)

# 链接 CUDA 运行时
target_link_libraries(reduce_v4  PRIVATE CUDA::cudart)

# 添加可执行文件
add_executable(reduce_v5 src/reduce_v5.cu)

# 链接 CUDA 运行时
target_link_libraries(reduce_v5  PRIVATE CUDA::cudart)

# 添加可执行文件
add_executable(reduce_v6 src/reduce_v6.cu)

# 链接 CUDA 运行时
target_link_libraries(reduce_v6  PRIVATE CUDA::cudart)

# 添加可执行文件
add_executable(reduce_v7 src/reduce_v7.cu)

# 链接 CUDA 运行时
target_link_libraries(reduce_v7  PRIVATE CUDA::cudart)
