#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float* inp_row = inp + i * C;
    float* out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) maxval = inp_row[j];
    }

    float sum = 0.f;
    for (int j = 0; j < C; ++j) {
      out_row[j] = std::exp(inp_row[j] - maxval);
      sum += out_row[j];
    }

    float norm = 1.f / sum;
    for (int j = 0; j < C; ++j) {
      out_row[j] *= norm;
    }
  }
}

template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float* d_input, float* d_output, int size) {
  int idx = threadIdx.x;
  float* input_begin = d_input + blockIdx.x * NUM_PER_BLOCK;

  // 每个线程得到自己的寄存器和,一个Block里面每个线程都是部分和
  float sum = .0f;
#pragma unroll
  for (int i = 0; i < NUM_PER_THREAD; ++i)
    sum += input_begin[idx + i * blockDim.x];

  // 每个wrap,将自己wrap内的线程的部分和规约到lane0
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);

  __shared__ float shared[32];
  const int warpId = idx / 32;
  const int laneId = idx % 32;

  // 把每个wrap里第一个线程的sum放入shared_mem
  if (laneId == 0) shared[warpId] = sum;

  __syncthreads();

  // 第一个wrap内在进行规约
  if (warpId == 0) {
    sum = (laneId < blockDim.x / 32) ? shared[laneId] : 0.f;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }
  if (idx == 0) d_output[blockIdx.x] = sum;
}
// CUDA kernel
__global__ void softmax_forward_kernel2(float* out, const float* inp, int N,
                                        int C) {
  // 1 Block ,1 line
  extern __shared__ float shared[];
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  int blocksize = blockDim.x;
  const float* start = inp + idx * C;

  float maxval = -INFINITY;
  for (int i = tid; i < C; i += blocksize) {
    maxval = fmaxf(maxval, start[i]);
  }
  shared[tid] = maxval;
  __syncthreads();

  // max
  for (int stride = blocksize / 2; stride >= 1; stride /= 2) {
    if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    __syncthreads();
  }

  float offset = shared[0];

  // compute exp and write the result
  for (int i = tid; i < C; i += blocksize) {
    out[idx * C + i] = std::exp(start[i] - offset);
  }
  __syncthreads();
  // thread coarsening again,for the sum
  float sumval = 0.f;
  for (int i = tid; i < C; i += blocksize) {
    sumval += out[idx * C + i];
  }
  shared[tid] = sumval;
  __syncthreads();

  // reduction
  for (int stride = blocksize / 2; stride >= 1; stride /= 2) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }

  // broadcast the sum to all threads in the block
  float sum = shared[0];

  for (int i = tid; i < C; i += blocksize) {
    out[idx * C + i] = out[idx * C + i] / sum;
  }
}