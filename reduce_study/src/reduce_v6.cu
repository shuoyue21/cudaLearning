#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
constexpr int THREAD_PER_BLOCK{256};

__device__ void wrapReduce(volatile float* cache, unsigned int tid) {
  cache[tid] += cache[tid + 32];
  cache[tid] += cache[tid + 16];
  cache[tid] += cache[tid + 8];
  cache[tid] += cache[tid + 4];
  cache[tid] += cache[tid + 2];
  cache[tid] += cache[tid + 1];
}

// v6 multi add
template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float* d_input, float* d_output, int size) {
  int idx = threadIdx.x;
  float* input_begin = d_input + blockIdx.x * NUM_PER_BLOCK;
  __shared__ float shared[THREAD_PER_BLOCK];
  shared[idx] = 0;
  for (int i = 0; i < NUM_PER_THREAD; ++i)
    shared[idx] += input_begin[idx + i * blockDim.x];
  __syncthreads();
#pragma unroll
  for (int i = blockDim.x >> 1; i > 32; i >>= 1) {
    if (threadIdx.x < i) shared[idx] += shared[idx + i];
    __syncthreads();
  }
  // if (idx < 128) shared[idx] += shared[idx + 128];
  // __syncthreads();

  // if (idx < 64) shared[idx] += shared[idx + 64];
  // __syncthreads();

  if (idx < 32) {
    wrapReduce(shared, idx);
  }
  if (idx == 0) d_output[blockIdx.x] = shared[0];
}

bool check(float* out, float* res, int size) {
  for (int i{0}; i < size; ++i) {
    if (std::fabs(out[i] - res[i]) > 1e-3f * std::fabs(res[i]) + 1.0f) {
      return false;
    }
  }
  return true;
}

int main() {
  constexpr int N{32 * 1024 * 1024};
  float* input = new float[N];
  float* d_input;
  cudaMalloc((void**)&d_input, N * sizeof(float));

  constexpr int block_num = 1024;
  constexpr int num_per_block = N / block_num;
  constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;
  float* output = new float[block_num];
  float* d_output;
  cudaMalloc((void**)&d_output, block_num * sizeof(float));

  for (int i{0}; i < N; ++i) {
    input[i] = i * .5f;
  }

  // cpu reduction
  float* res = new float[block_num];
  for (int i{0}; i < block_num; ++i) {
    float cur = .0f;
    for (int j{0}; j < num_per_block; ++j) {
      cur += input[i * num_per_block + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  reduce<num_per_block, num_per_thread>
      <<<block_num, THREAD_PER_BLOCK>>>(d_input, d_output, N);
  cudaMemcpy(output, d_output, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, res, block_num)) {
    std::cout << "Result correct!" << std::endl;
  } else {
    std::cout << "Result incorrect!" << std::endl;
  }
  return 0;
}