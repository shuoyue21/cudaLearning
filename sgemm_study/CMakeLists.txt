cmake_minimum_required(VERSION 3.18)
project(cuda_minimal LANGUAGES CXX CUDA)

# C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 已有构建目录可能缓存了空的 Release flags，补齐优化选项。
if(CMAKE_CUDA_FLAGS_RELEASE STREQUAL "")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING
      "Flags used by the CUDA compiler during RELEASE builds." FORCE)
endif()

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

add_executable(sgemm_v4  sgemm_v4.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v4  PRIVATE CUDA::cudart)

add_executable(sgemm_v5  sgemm_v5.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v5  PRIVATE CUDA::cudart)

add_executable(sgemm_v6  sgemm_v6.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v6  PRIVATE CUDA::cudart)

add_executable(sgemm_v7  sgemm_v7.cu)

# 链接 CUDA 运行时
target_link_libraries(sgemm_v7  PRIVATE CUDA::cudart)

# 将原版本编译为独立 namespace 下的 launcher，保留各自的学习用 main。
add_executable(sgemm_benchmark
  sgemm_benchmark.cu
  sgemm_v1.cu
  sgemm_v2.cu
  sgemm_v3.cu
  sgemm_v4.cu
  sgemm_v5.cu
  sgemm_v6.cu
  sgemm_v7.cu
  sgemm_warp_tiling_benchmark.cu
)
target_compile_definitions(sgemm_benchmark PRIVATE SGEMM_BENCHMARK)
target_link_libraries(sgemm_benchmark PRIVATE CUDA::cudart CUDA::cublas)

option(SGEMM_ENABLE_CUTLASS "Include CUTLASS FP32 SIMT comparison kernels" OFF)
set(CUTLASS_ROOT "" CACHE PATH "Path to a CUTLASS source checkout")
if(SGEMM_ENABLE_CUTLASS)
  find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h
    HINTS "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}")
  if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR "CUTLASS headers not found; set CUTLASS_ROOT")
  endif()
  target_sources(sgemm_benchmark PRIVATE sgemm_cutlass_benchmark.cu)
  target_include_directories(sgemm_benchmark SYSTEM PRIVATE "${CUTLASS_INCLUDE_DIR}")
  target_compile_definitions(sgemm_benchmark PRIVATE SGEMM_WITH_CUTLASS)
  message(STATUS "CUTLASS headers: ${CUTLASS_INCLUDE_DIR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(WARNING "sgemm_benchmark should be built in Release mode for timing")
endif()
