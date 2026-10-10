
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
  // 整体思路是 在数据搬运阶段 采用float4 把搬运的条目和线程都一维化
  // 在计算阶段 每个线程在各自计算自己的数据
  // 采用搬运和计算分离的方式
  constexpr int VEC_WIDTH = 4;  // 一个向量读取的宽度

  // C tile 在两个方向分别需要多少个线程
  constexpr int THREAD_M = Bm / Tm;
  constexpr int THREAD_N = Bn / Tn;
  constexpr int NUM_THREADS = THREAD_M * THREAD_N;

  static_assert(Bm % Tm == 0, "BM must be divisible by TM");
  static_assert(Bn % Tn == 0, "BN must be divisible by TN");
  static_assert(Bk % VEC_WIDTH == 0, "BK must be divisible by 4");
  static_assert(Bn % VEC_WIDTH == 0, "BN must be divisible by 4");
  static_assert(NUM_THREADS <= 1024, "too many threads in one block");
  /*
   * 线程坐标直接对应计算布局：
   *
   * thread_m 决定该线程负责哪 TM 行
   * thread_n 决定该线程负责哪 TN 列
   */
  // 计算布局线程坐标:
  // 这里直接相等是因为物理线程布局和逻辑计算任务布局一致
  // 一个Block负责32*32的tile 一个线程负责Tm*Tn 总共需要NUM_THREADS个线程
  // 而启动时 dim3 block(THREAD_N, THREAD_M)
  // 所以物理线程布局和逻辑计算任务布局一致
  const int thread_m = threadIdx.y;
  const int thread_n = threadIdx.x;
  // 线性线程号仅用于合作搬运数据

  const int tid = threadIdx.y * THREAD_N + threadIdx.x;

  const int block_row = blockIdx.y * Bm;
  const int block_col = blockIdx.x * Bn;
  /*
   * A 在共享内存中转置存储。
   *
   * 原始逻辑：A_shared[BM][BK]
   * 这里存为：A_shared[BK][BM]
   *
   * 计算时固定 k，线程访问不同的 M 位置，
   * 转置后通常能减少 shared-memory bank conflict。
   */
  __shared__ float As[Bk][Bm];
  __shared__ float Bs[Bk][Bn];

  float a_reg[Tm] = {0.f};
  float b_reg[Tn] = {0.f};
  float temp[Tm][Tn] = {0.f};
  for (int offset = 0; offset < K; offset += Bk) {
    /*
     * 合作加载 A tile。
     *
     * A tile 一共有：
     * BM * BK / 4 个 float4
     *
     * 不再要求“一个线程必须刚好加载一个 float4”。
     * 如果加载任务比线程多，每个线程就多迭代几次。
     */

    // A原本的tile逻辑布局是 A_tile[m][k]
    // 写入共享内存转置后变成了 As[k][m]
    // 因此 As[kk][m_local] 对应 A[m_local+block_row][kk+offset]
    // 所以 As[kk][m_local] = A[(m_local+block_row)*K+kk+offset]
    constexpr int A_VEC_PER_ROW = Bk / VEC_WIDTH;    // 一行有几个float4
    constexpr int A_VEC_COUNT = Bm * A_VEC_PER_ROW;  // 总共有多少个float4
    for (int vec_id = tid; vec_id < A_VEC_COUNT; vec_id += NUM_THREADS) {
      // 当前tile的局部坐标
      // tid 是线程编号 vec_id是当前线程领取的搬运任务编号
      // 这里把任务编号转换为对应的二维坐标
      // local_row local_col 就是要搬的对应的数据在原本的A tile对应的起始下标
      const int local_row = vec_id / A_VEC_PER_ROW;
      const int local_col = (vec_id % A_VEC_PER_ROW) * VEC_WIDTH;
      // 先float4从global取数据
      const float4 value =
          load_float4(A + (block_row + local_row) * K + offset + local_col);

      // 加载时顺便转置写入 shared memory
      // local_row 和 local_col 是根据 Global Memory 中 A
      // 的原始布局计算出来的，但 As 采用的是另一种布局。
      // 这里的转置是为了方便后面计算阶段,沿着As的行方向读 是一种访存优化
      As[local_col + 0][local_row] = value.x;
      As[local_col + 1][local_row] = value.y;
      As[local_col + 2][local_row] = value.z;
      As[local_col + 3][local_row] = value.w;
    }

    /*
     * 合作加载 B tile。
     *
     * B tile 一共有：
     * BK * BN / 4 个 float4
     */
    constexpr int B_VEC_PER_ROW = Bn / VEC_WIDTH;    // 一行有几个float4
    constexpr int B_VEC_COUNT = Bk * B_VEC_PER_ROW;  // 总共有多少个float4
    for (int vec_id = tid; vec_id < B_VEC_COUNT; vec_id += NUM_THREADS) {
      const int local_row = vec_id / B_VEC_PER_ROW;
      const int local_col = (vec_id % B_VEC_PER_ROW) * VEC_WIDTH;
      // 先float4从global取数据
      const float4 value =
          load_float4(B + (offset + local_row) * N + block_col + local_col);

      Bs[local_row][local_col + 0] = value.x;
      Bs[local_row][local_col + 1] = value.y;
      Bs[local_row][local_col + 2] = value.z;
      Bs[local_row][local_col + 3] = value.w;
    }

    __syncthreads();

    /*
     * 当前线程负责：
     *
     * 行：
     * thread_m * TM ... thread_m * TM + TM - 1
     *
     * 列：
     * thread_n * TN ... thread_n * TN + TN - 1
     */
    // 先搬数据
#pragma unroll
    for (int kk = 0; kk < Bk; ++kk) {
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
        a_reg[i] = As[kk][thread_m * Tm + i];
      }
#pragma unroll
      for (int i = 0; i < Tn; ++i) {
        b_reg[i] = Bs[kk][thread_n * Tn + i];
      }

      // TM × TN 外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) {
          temp[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }

    // 所有线程读完当前 shared tile 后才能覆盖它
    __syncthreads();
  }
  const int output_row = block_row + thread_m * Tm;
  const int output_col = block_col + thread_n * Tn;

  for (int i = 0; i < Tm; ++i) {
    for (int j = 0; j < Tn; ++j) {
      C[(output_row + i) * N + output_col + j] = temp[i][j];
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
