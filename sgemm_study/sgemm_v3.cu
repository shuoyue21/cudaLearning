#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
void random_matrix(int m, int n, float* matrix) {
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

void cpu_sgemm(int m, int n, int k, const float* A, const float* B, float* C) {
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

float calc_max_diff(int m, int n, const float* cpu_result,
                    const float* gpu_result) {
  float max_diff = 0.0f;
  const int total = m * n;
  for (int i = 0; i < total; ++i) {
    const float diff = std::fabs(cpu_result[i] - gpu_result[i]);
    max_diff = std::fmax(max_diff, diff);
  }
  return max_diff;
}

// 增加每个线程的工作量
// 每个线程负责处理在一个方向上的STRIDE个元素
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm(float* A, float* B, float* C, const int M,
                           const int N, const int K) {
  constexpr int SIZE =
      BLOCK_SIZE * STRIDE;  // 一个BLOCK在一个方向上处理的元素大小
  float* A_start = A + blockIdx.y * K * SIZE;
  float* B_start = B + blockIdx.x * SIZE;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float As[SIZE][SIZE];
  __shared__ float Bs[SIZE][SIZE];

  float temp[STRIDE][STRIDE] = {0.f};  // 每个线程在一个方向上处理STRIDE个元素
  // 对于矩阵A,offset是x方向的偏移量,对于矩阵B,offset是y方向的偏移量
  for (int offset = 0; offset < K; offset += SIZE) {
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        As[ty * STRIDE + i][tx * STRIDE + j] = A_start
            [(ty * STRIDE + i) * K + offset + tx * STRIDE +
             j];  // A_start只是在y方向"start",在x方向仍然是从头开始算,所以要加偏移量,B_start同理
        Bs[ty * STRIDE + i][tx * STRIDE + j] =
            B_start[(ty * STRIDE + i + offset) * N + tx * STRIDE + j];
      }
    }

    __syncthreads();
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int e = 0; e < SIZE; ++e) {
          temp[i][j] += As[ty * STRIDE + i][e] * Bs[e][tx * STRIDE + j];
        }
      }
    }
    __syncthreads();
  }

  float* C_start = C + N * blockIdx.y * SIZE + blockIdx.x * SIZE;
  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      C_start[N * (ty * STRIDE + i) + tx * STRIDE + j] = temp[i][j];
    }
  }
}

int main() {
  constexpr int32_t m = 512;  // 矩阵A的行
  constexpr int32_t k = 256;  // 矩阵A的列，也是矩阵B的行
  constexpr int32_t n = 256;  // 矩阵B的列

  const size_t mem_size_A = m * k * sizeof(float);
  const size_t mem_size_B = k * n * sizeof(float);
  const size_t mem_size_C = m * n * sizeof(float);

  float* matrix_A_host = (float*)std::malloc(mem_size_A);
  float* matrix_B_host = (float*)std::malloc(mem_size_B);

  float* matrix_C_host_gpu_calc = (float*)std::malloc(mem_size_C);
  float* matrix_C_host_cpu_calc = (float*)std::malloc(mem_size_C);

  // 初始化矩阵
  random_matrix(m, k, matrix_A_host);
  random_matrix(k, n, matrix_B_host);
  // 将结果矩阵初始化为0
  memset(matrix_C_host_gpu_calc, 0, mem_size_C);
  memset(matrix_C_host_cpu_calc, 0, mem_size_C);

  float* matrix_A_device;
  float* matrix_B_device;
  float* matrix_C_device;
  // 分配设备内存
  cudaMalloc((void**)&matrix_A_device, mem_size_A);
  cudaMalloc((void**)&matrix_B_device, mem_size_B);
  cudaMalloc((void**)&matrix_C_device, mem_size_C);
  // 将数据从CPU复制到GPU
  cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A,
             cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B,
             cudaMemcpyHostToDevice);
  constexpr int STRIDE = 2;  // 一个线程处理单个方向2个元素
  constexpr int BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((n + BLOCK - 1) / BLOCK / STRIDE, (m + BLOCK - 1) / BLOCK / STRIDE);

  cpu_sgemm(m, n, k, matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc);
  cuda_sgemm<BLOCK, STRIDE><<<grid, block>>>(matrix_A_device, matrix_B_device,
                                             matrix_C_device, m, n, k);
  cudaDeviceSynchronize();
  // 强制 CPU 等待 GPU 完成，并捕获 Kernel 运行时的异步错误
  // (例如非法的内存访问、寄存器溢出等)
  cudaError_t err_sync = cudaDeviceSynchronize();
  if (err_sync != cudaSuccess) {
    std::printf("Kernel execution failed: %s\n", cudaGetErrorString(err_sync));
    return -1;
  }

  // 捕获 Kernel 启动时的同步错误 (例如 grid/block 参数非法)
  cudaError_t err_async = cudaGetLastError();
  if (err_async != cudaSuccess) {
    std::printf("Kernel launch failed: %s\n", cudaGetErrorString(err_async));
    return -1;
  }

  cudaError_t err_cpy = cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device,
                                   mem_size_C, cudaMemcpyDeviceToHost);
  if (err_cpy != cudaSuccess) {
    std::printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err_cpy));
    return -1;
  }
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
