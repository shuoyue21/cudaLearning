#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
constexpr int THREAD_PER_BLOCK{256};

// v3_A solve idle thread ,每个线程处理更多的数据
__global__ void reduce_v3_planA(float* d_input, float* d_output, int size) {
  float* input_begin = d_input + blockDim.x * 2 * blockIdx.x;
  __shared__ float shared[THREAD_PER_BLOCK];
  shared[threadIdx.x] =
      input_begin[threadIdx.x] +
      input_begin[threadIdx.x + blockDim.x];  // 一个线程搬运2个元素并且相加
  __syncthreads();
  for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (threadIdx.x < i) shared[threadIdx.x] += shared[threadIdx.x + i];
    __syncthreads();
  }
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

  int block_num =
      ((N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK) / 2;  // ceil div
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
    for (int j{0}; j < 2 * THREAD_PER_BLOCK; ++j) {
      cur += input[i * 2 * THREAD_PER_BLOCK + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  reduce_v3_planA<<<block_num, THREAD_PER_BLOCK>>>(d_input, d_output, N);
  cudaMemcpy(output, d_output, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, res, block_num)) {
    std::cout << "Result correct!" << std::endl;
  } else {
    std::cout << "Result incorrect!" << std::endl;
  }
  return 0;
}