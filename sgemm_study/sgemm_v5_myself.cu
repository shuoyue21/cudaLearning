
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

__device__ __forceinline__ float4 load_float4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}

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

template <unsigned int Bm, unsigned int Bn, unsigned int Bk,
          unsigned int Tm,  // 每线程计算的行数
          unsigned int Tn   // 每线程计算的列数
          >
__global__ void cuda_sgemm(float* A, float* B, float* C, const int M,
                           const int N, const int K) {
  // 同一批物理线程有两种逻辑映射：
  // 1. 使用 threadIdx.x/y 决定每个线程计算哪个 TM×TN 输出块
  // 2. 使用一维 tid 给线程分配 float4 搬运任务
  // 物理线程的 布局的关系
  constexpr int VEC_WIDTH = 4;
  constexpr int THREAD_M = Bm / Tm;  // C tile 需要多少行线程
  constexpr int THREAD_N = Bn / Tn;
  constexpr int NUM_THREADS = THREAD_M * THREAD_N;  // 总线程数

  // 计算线程id 这里为什么一致呢?
  const int thread_m = threadIdx.y;
  const int thread_n = threadIdx.x;

  // 搬运线程tid
  const int tid = threadIdx.y * THREAD_N + threadIdx.x;

  // 当前 block 对应的 C tile 左上角全局坐标。
  const int block_row = blockIdx.y * Bm;
  const int block_col = blockIdx.x * Bn;

  // A tile 在 shared memory 中转置存储。
  // 计算时固定 kk，不同线程读取不同 m，
  // 将 m 放在连续的第二维可以改善 shared-memory bank 访问。
  __shared__ float As[Bk][Bm];
  __shared__ float Bs[Bk][Bn];

  // reg
  float a_reg[Tm] = {0.f};
  float b_reg[Tn] = {0.f};
  float accumulation[Tm][Tn] = {0.f};

  for (int offset = 0; offset < K; offset += Bk) {
    // 先开始搬数据

    // 从A向As搬运
    // 采用搬运线程tid 以一维的视角来看待数据搬运
    constexpr int A_VEC_PER_ROW = Bk / VEC_WIDTH;
    constexpr int A_VEC_COUNT = A_VEC_PER_ROW * Bm;  // 总共有多少个搬运任务
    for (int vec_id = tid; vec_id < A_VEC_COUNT; vec_id += NUM_THREADS) {
      // 这里的vec_id 是搬运任务id

      // 将搬运任务id转换为A tile的坐标关系
      const int local_row = vec_id / A_VEC_PER_ROW;
      const int local_col =
          (vec_id % A_VEC_PER_ROW) *
          VEC_WIDTH;  // 因为一个线程搬运四个数据,所以要乘width
      const float4 value =
          load_float4(A + (local_row + block_row) * K + offset + local_col);
      // 写入转置As
      // As本来应该是[Bm][Bk] 所以是 As[local_row][local_col]
      // 但是经过了转置
      // 因为原本是按照行读取的float4,写入不转置的As也要按行,但是转置了
      As[local_col + 0][local_row] = value.x;
      As[local_col + 1][local_row] = value.y;
      As[local_col + 2][local_row] = value.z;
      As[local_col + 3][local_row] = value.w;
    }

    constexpr int B_VEC_PER_ROW = Bn / VEC_WIDTH;
    constexpr int B_VEC_COUNT = B_VEC_PER_ROW * Bk;
    for (int vec_id = tid; vec_id < B_VEC_COUNT; vec_id += NUM_THREADS) {
      const int local_row = vec_id / B_VEC_PER_ROW;
      const int local_col = (vec_id % B_VEC_PER_ROW) * VEC_WIDTH;

      const float4 value =
          load_float4(B + (local_row + offset) * N + local_col + block_col);

      Bs[local_row][local_col + 0] = value.x;
      Bs[local_row][local_col + 1] = value.y;
      Bs[local_row][local_col + 2] = value.z;
      Bs[local_row][local_col + 3] = value.w;
    }

    __syncthreads();

// 开始计算每一个tile
#pragma unroll
    for (int kk = 0; kk < Bk; ++kk) {
// 先把数据从As搬运到reg
// 这里的搬运要采用计算线程
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
        // 如果 As 没有转置，应访问 As[m][kk]
        // 当前 As 按 [kk][m] 转置存储，因此访问 As[kk][m]
        a_reg[i] = As[kk][thread_m * Tm + i];
      }

#pragma unroll
      for (int j = 0; j < Tn; ++j) {
        b_reg[j] = Bs[kk][thread_n * Tn + j];
      }

      // 开始计算 外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) {
          accumulation[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }

    __syncthreads();
  }
  const int output_row = block_row + Tm * thread_m;
  const int output_col = block_col + Tn * thread_n;

  for (int i = 0; i < Tm; ++i) {
    for (int j = 0; j < Tn; ++j) {
      C[(output_row + i) * N + output_col + j] = accumulation[i][j];
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
  constexpr unsigned int BM = 32;
  constexpr unsigned int BN = 32;
  constexpr unsigned int BK = 32;
  constexpr unsigned int TM = 4;
  constexpr unsigned int TN = 4;

  static_assert(BM % TM == 0);
  static_assert(BN % TN == 0);

  if (m % BM != 0 || n % BN != 0 || k % BK != 0 || k % 4 != 0 || n % 4 != 0) {
    std::fprintf(stderr,
                 "Current kernel requires M%%BM=N%%BN=K%%BK=0 "
                 "and K,N divisible by 4\n");
    return -1;
  }

  dim3 block(BN / TN, BM / TM);
  dim3 grid(n / BN, m / BM);

  cuda_sgemm<BM, BN, BK, TM, TN><<<grid, block>>>(
      matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

  cpu_sgemm(m, n, k, matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc);
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
