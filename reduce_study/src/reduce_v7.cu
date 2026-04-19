#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int THREAD_PER_BLOCK{256};

// v7 shuffle
template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float *d_input, float *d_output, int size) {
  int idx = threadIdx.x;
  float *input_begin = d_input + blockIdx.x * NUM_PER_BLOCK;
  float sum = .0f;
#pragma unroll
  for (int i = 0; i < NUM_PER_THREAD; ++i)
    sum += input_begin[idx + i * blockDim.x];

  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);

  __shared__ float shared[32];
  const int warpId = idx / 32;
  const int laneId = idx % 32;

  if (laneId == 0)
    shared[warpId] = sum;

  __syncthreads();

  if (warpId == 0) {
    sum = (laneId < blockDim.x / 32) ? shared[laneId] : 0.f;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }
  if (idx == 0)
    d_output[blockIdx.x] = sum;
}

bool check(float *out, float *res, int size) {
  for (int i{0}; i < size; ++i) {
    if (std::fabs(out[i] - res[i]) > 1e-3f * std::fabs(res[i]) + 1.0f) {
      return false;
    }
  }
  return true;
}

int main() {
  constexpr int N{32 * 1024 * 1024};
  float *input = new float[N];
  float *d_input;
  CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(float)));

  constexpr int block_num = 1024;
  constexpr int num_per_block = N / block_num;
  constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;
  float *output = new float[block_num];
  float *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_output, block_num * sizeof(float)));

  for (int i{0}; i < N; ++i) {
    input[i] = i * .5f;
  }

  // cpu reduction
  float *res = new float[block_num];
  for (int i{0}; i < block_num; ++i) {
    float cur = .0f;
    for (int j{0}; j < num_per_block; ++j) {
      cur += input[i * num_per_block + j];
    }
    res[i] = cur;
  }

  CUDA_CHECK(
      cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice));

  reduce<num_per_block, num_per_thread>
      <<<block_num, THREAD_PER_BLOCK>>>(d_input, d_output, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output, d_output, block_num * sizeof(float),
                        cudaMemcpyDeviceToHost));
  if (check(output, res, block_num)) {
    std::cout << "Result correct!" << std::endl;
  } else {
    std::cout << "Result incorrect!" << std::endl;
  }
  return 0;
}