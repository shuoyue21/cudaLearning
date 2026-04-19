#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
constexpr int THREAD_PER_BLOCK{256};

// v0
__global__ void reduce_naives(float* d_input, float* d_output, int size) {
  float* input_begin =
      d_input +
      blockDim.x * blockIdx.x;  // 这样input_begin[0]永远是当前block的起始元素
  for (int i{1}; i < blockDim.x; i *= 2) {
    if (threadIdx.x % (i * 2) == 0)
      input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
    __syncthreads();
  }
  if (threadIdx.x == 0) d_output[blockIdx.x] = input_begin[0];
}

// v1
__global__ void reduce_sharedmem(float* d_input, float* d_output, int size) {
  float* input_begin = d_input + blockDim.x * blockIdx.x;
  __shared__ float shared[THREAD_PER_BLOCK];
  shared[threadIdx.x] = input_begin[threadIdx.x];  // 一个线程搬运一个元素
  __syncthreads();
  for (int i{1}; i < blockDim.x; i *= 2) {
    if (threadIdx.x % (i * 2) == 0)
      shared[threadIdx.x] += shared[threadIdx.x + i];
    __syncthreads();
  }
  if (threadIdx.x == 0) d_output[blockIdx.x] = shared[0];
}

// v2 no_warp_divergence_and_bank_conflict
__global__ void reduce_v2(float* d_input, float* d_output, int size) {
  float* input_begin = d_input + blockDim.x * blockIdx.x;
  __shared__ float shared[THREAD_PER_BLOCK];
  shared[threadIdx.x] = input_begin[threadIdx.x];  // 一个线程搬运一个元素
  __syncthreads();
  for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (threadIdx.x < i) shared[threadIdx.x] += shared[threadIdx.x + i];
    __syncthreads();
  }
  /*
  64 thread :
  if idx = 0,1,....31:
  shared[idx] += shared[idx+32]
  if idx = 0,1,...15:
  shared[idx] += shared[idx+16]
  if idx = 0,1,...7:
  shared[idx] += shared[idx+8]
  if idx =0,1,..3:

  if idx = 0,1:

  if idx = 0:
  */
  if (threadIdx.x == 0) d_output[blockIdx.x] = shared[0];
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

  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;  // ceil div
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
    for (int j{0}; j < THREAD_PER_BLOCK; ++j) {
      cur += input[i * THREAD_PER_BLOCK + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  reduce_v2<<<block_num, THREAD_PER_BLOCK>>>(d_input, d_output, N);
  cudaMemcpy(output, d_output, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, res, block_num)) {
    std::cout << "Result correct!" << std::endl;
  } else {
    std::cout << "Result incorrect!" << std::endl;
  }
  return 0;
}