#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
void random_matrix(int m, int n, float *matrix) {
  // 1. 实例化硬件随机设备，用于生成非确定性种子
  std::random_device rd;

  // 2. 使用 Mersenne Twister 算法初始化伪随机数引擎
  std::mt19937 gen(rd());

  // 3. 针对神经网络的矩阵乘法，使用 Xavier 均匀分布初始化以维持方差
  float limit = std::sqrt(6.0f / (m + n));
  std::uniform_real_distribution<float> dist(-limit, limit);

  // 4. 将矩阵视为扁平的一维数组进行连续访问，最大化利用 CPU 缓存
  int total_elements = m * n;
  for (int i = 0; i < total_elements; ++i) {
    matrix[i] = dist(gen);
  }
}

void cpu_sgemm(int m, int n, int k, const float *A, const float *B, float *C) {
  // 1. 使用三重循环实现矩阵乘法，外层循环遍历结果矩阵 C 的行和列
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      // 内层循环遍历A和行和B的列，计算点积
      for (int p = 0; p < k; ++p) {
        sum += A[i * k + p] * B[p * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

float calc_max_diff(int m, int n, const float *cpu_result,
                    const float *gpu_result) {
  float max_diff = 0.0f;
  const int total = m * n;
  for (int i = 0; i < total; ++i) {
    const float diff = std::fabs(cpu_result[i] - gpu_result[i]);
    max_diff = std::fmax(max_diff, diff);
  }
  return max_diff;
}

// 增加每个线程的工作量
template <unsigned int BLOCK_SIZE>
__global__ void cuda_sgemm(float *A, float *B, float *C, const int M,
                           const int N, const int K) {

  float *A_start = A + blockIdx.y * blockDim.y * K;
  float *B_start = B + blockDim.x * blockIdx.x;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float temp = 0.f;
  for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {

    if (row < M and (t * BLOCK_SIZE + tx) < K) {
      As[ty][tx] = A_start[ty * K + t * BLOCK_SIZE + tx];
    } else {
      As[ty][tx] = 0.f;
    }
    if (col < N and (t * BLOCK_SIZE + ty) < K) {
      Bs[ty][tx] = B_start[(t * BLOCK_SIZE + ty) * N + tx];
    } else {
      Bs[ty][tx] = 0.f;
    }

    __syncthreads();

    for (int e = 0; e < BLOCK_SIZE; ++e) {
      temp += As[ty][e] * Bs[e][tx];
    }

    __syncthreads();
  }

  if (row < M and col < N) {
    C[row * N + col] = temp;
  }
}

int main() {

  constexpr int32_t m = 512; // 矩阵A的行
  constexpr int32_t k = 256; // 矩阵A的列，也是矩阵B的行
  constexpr int32_t n = 256; // 矩阵B的列

  const size_t mem_size_A = m * k * sizeof(float);
  const size_t mem_size_B = k * n * sizeof(float);
  const size_t mem_size_C = m * n * sizeof(float);

  float *matrix_A_host = (float *)std::malloc(mem_size_A);
  float *matrix_B_host = (float *)std::malloc(mem_size_B);

  float *matrix_C_host_gpu_calc = (float *)std::malloc(mem_size_C);
  float *matrix_C_host_cpu_calc = (float *)std::malloc(mem_size_C);

  // 初始化矩阵
  random_matrix(m, k, matrix_A_host);
  random_matrix(k, n, matrix_B_host);
  // 将结果矩阵初始化为0
  memset(matrix_C_host_gpu_calc, 0, mem_size_C);
  memset(matrix_C_host_cpu_calc, 0, mem_size_C);

  float *matrix_A_device;
  float *matrix_B_device;
  float *matrix_C_device;
  // 分配设备内存
  cudaMalloc((void **)&matrix_A_device, mem_size_A);
  cudaMalloc((void **)&matrix_B_device, mem_size_B);
  cudaMalloc((void **)&matrix_C_device, mem_size_C);
  // 将数据从CPU复制到GPU
  cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A,
             cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B,
             cudaMemcpyHostToDevice);
  constexpr int BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);

  cpu_sgemm(m, n, k, matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc);
  cuda_sgemm<BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device,
                                     matrix_C_device, m, n, k);
  cudaDeviceSynchronize();

  cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C,
             cudaMemcpyDeviceToHost);

  constexpr float kMaxDiffTolerance = 1e-3f;
  const float max_diff =
      calc_max_diff(m, n, matrix_C_host_cpu_calc, matrix_C_host_gpu_calc);
  const bool is_correct = max_diff <= kMaxDiffTolerance;

  std::printf("max diff: %.8f\n", max_diff);
  std::printf("result: %s\n", is_correct ? "correct" : "incorrect");

  return 0;
}
