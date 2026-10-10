#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda/pipeline>
#include <random>

void random_matrix(int m, int n, float* matrix) {
  std::random_device rd;
  std::mt19937 gen(rd());
  float limit = std::sqrt(6.0f / (m + n));
  std::uniform_real_distribution<float> dist(-limit, limit);
  int total_elements = m * n;
  for (int i = 0; i < total_elements; ++i) {
    matrix[i] = dist(gen);
  }
}

void cpu_sgemm(int m, int n, int k, const float* A, const float* B, float* C) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
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

// ============================================================
// 异步搬运一个 A tile 和一个 B tile 到指定 shared stage
// ============================================================

template <int BM, int BN, int BK, int PAD, int NUM_THREADS, typename Pipeline>
__device__ __forceinline__ void copy_tile_async(
    const float* __restrict__ A, const float* __restrict__ B,

    // 分别指向 As[stage][0][0] 和 Bs[stage][0][0]
    float* As_stage, float* Bs_stage,

    int block_row, int block_col, int k_base,

    int N, int K, int tid,

    Pipeline& pipeline) {
  constexpr int VEC_WIDTH = 4;

  // 一次 memcpy_async 搬运 4 个 float，即 16 字节。
  constexpr int COPY_BYTES = VEC_WIDTH * sizeof(float);

  // shared memory 每一行的真实跨度，包含 padding。
  constexpr int AS_STRIDE = BK + PAD;
  constexpr int BS_STRIDE = BN + PAD;

  // --------------------------------------------------------
  // 搬运 A tile：[BM, BK]
  // --------------------------------------------------------

  constexpr int A_VECS_PER_ROW = BK / VEC_WIDTH;

  constexpr int A_VEC_COUNT = BM * A_VECS_PER_ROW;

  for (int vec_id = tid; vec_id < A_VEC_COUNT; vec_id += NUM_THREADS) {
    // 当前 float4 任务位于 A tile 的哪一行。
    const int local_row = vec_id / A_VECS_PER_ROW;

    // 当前 float4 从这一行的哪一列开始。
    const int local_col = (vec_id % A_VECS_PER_ROW) * VEC_WIDTH;

    // global memory 源地址：
    // &A[block_row + local_row][k_base + local_col]
    const float* global_src =
        A + (block_row + local_row) * K + k_base + local_col;

    // shared memory 目标地址：
    // &As[stage][local_row][local_col]
    float* shared_dst = As_stage + local_row * AS_STRIDE + local_col;

    // 异步复制连续 16 字节：
    // A 的 4 个 float -> shared memory 的 4 个 float
    cuda::memcpy_async(shared_dst, global_src,
                       cuda::aligned_size_t<16>(COPY_BYTES), pipeline);
  }

  // --------------------------------------------------------
  // 搬运 B tile：[BK, BN]
  // --------------------------------------------------------

  constexpr int B_VECS_PER_ROW = BN / VEC_WIDTH;

  constexpr int B_VEC_COUNT = BK * B_VECS_PER_ROW;

  for (int vec_id = tid; vec_id < B_VEC_COUNT; vec_id += NUM_THREADS) {
    const int local_row = vec_id / B_VECS_PER_ROW;

    const int local_col = (vec_id % B_VECS_PER_ROW) * VEC_WIDTH;

    // global memory 源地址：
    // &B[k_base + local_row][block_col + local_col]
    const float* global_src =
        B + (k_base + local_row) * N + block_col + local_col;

    // shared memory 目标地址：
    // &Bs[stage][local_row][local_col]
    float* shared_dst = Bs_stage + local_row * BS_STRIDE + local_col;

    cuda::memcpy_async(shared_dst, global_src,
                       cuda::aligned_size_t<16>(COPY_BYTES), pipeline);
  }
}

// ============================================================
// 双缓冲 SGEMM kernel
// ============================================================

template <int BM,      // 一个 block 计算 C 的行数
          int BN,      // 一个 block 计算 C 的列数
          int BK,      // 一个 K tile 的长度
          int TM,      // 一个线程计算的行数
          int TN,      // 一个线程计算的列数
          int PAD = 4  // shared memory 行 padding
          >
__global__ void cuda_sgemm_double_buffer(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C, int M, int N,
                                         int K) {
  constexpr int VEC_WIDTH = 4;
  constexpr int NUM_STAGES = 2;

  constexpr int THREAD_M = BM / TM;

  constexpr int THREAD_N = BN / TN;

  constexpr int NUM_THREADS = THREAD_M * THREAD_N;

  static_assert(BM % TM == 0);
  static_assert(BN % TN == 0);

  static_assert(BK % VEC_WIDTH == 0);
  static_assert(BN % VEC_WIDTH == 0);

  // 每一行的 shared 地址仍然必须保持 16 字节对齐。
  static_assert((BK + PAD) % VEC_WIDTH == 0);
  static_assert((BN + PAD) % VEC_WIDTH == 0);

  static_assert(NUM_THREADS <= 1024);

  const int thread_m = threadIdx.y;
  const int thread_n = threadIdx.x;

  const int tid = threadIdx.y * THREAD_N + threadIdx.x;

  const int block_row = blockIdx.y * BM;

  const int block_col = blockIdx.x * BN;

  // 当前版本没有边界判断。
  (void)M;

  /*
   * 两套 A/B shared buffer：
   *
   * As[0]、Bs[0]：stage 0
   * As[1]、Bs[1]：stage 1
   *
   * A 不再转置存储，而是保持 As[m][k]。
   * 这样 global 中连续的 4 个 float 可以直接异步复制到
   * shared 中连续的 4 个 float。
   */
  __shared__ __align__(16) float As[NUM_STAGES][BM][BK + PAD];

  __shared__ __align__(16) float Bs[NUM_STAGES][BK][BN + PAD];

  float a_reg[TM];
  float b_reg[TN];

  float accumulation[TM][TN] = {};

  // 每个线程建立自己的 pipeline。
  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

  const int num_k_tiles = K / BK;

  // ========================================================
  // Prologue：先异步提交 tile 0 到 stage 0
  // ========================================================

  pipeline.producer_acquire();  // 线程开始组织一个新的异步拷贝批次

  copy_tile_async<BM, BN, BK, PAD, NUM_THREADS>(A, B,

                                                // stage 0 的首地址
                                                &As[0][0][0], &Bs[0][0][0],

                                                block_row, block_col,

                                                // 第一个 tile 的 K 起点
                                                0,

                                                N, K, tid, pipeline);

  pipeline.producer_commit();  // 所有memcpy_async已经添加完,作为一个批次提交

  // ========================================================
  // 主循环
  // ========================================================

  for (int tile_id = 0; tile_id < num_k_tiles; ++tile_id) {
    /*
     * tile 0 使用 stage 0
     * tile 1 使用 stage 1
     * tile 2 使用 stage 0
     * tile 3 使用 stage 1
     */
    const int read_stage = tile_id & 1;

    /*
     * 等待 pipeline 中最早提交的 tile 完成。
     *
     * 对 tile_id=0，等待 tile 0；
     * 对 tile_id=1，等待 tile 1；
     * 以此类推。
     */
    pipeline.consumer_wait();

    /*
     * consumer_wait 只负责当前线程提交的异步复制。
     *
     * 一个完整 tile 是整个 block 的线程合作搬入的，
     * 所以还要等待 block 内其他线程。
     */
    __syncthreads();

    // ----------------------------------------------------
    // 提前提交下一个 K tile 的异步拷贝
    // ----------------------------------------------------

    const bool has_next_tile = tile_id + 1 < num_k_tiles;

    if (has_next_tile) {
      /*
       * 当前读 stage 0，则下一块写 stage 1；
       * 当前读 stage 1，则下一块写 stage 0。
       */
      const int write_stage = read_stage ^ 1;

      const int next_k_base = (tile_id + 1) * BK;

      // 获取一个新的 pipeline 生产槽位。
      pipeline.producer_acquire();

      copy_tile_async<BM, BN, BK, PAD, NUM_THREADS>(
          A, B,

          &As[write_stage][0][0], &Bs[write_stage][0][0],

          block_row, block_col, next_k_base,

          N, K, tid, pipeline);

      /*
       * 把上面的所有 memcpy_async 作为一个批次提交。
       *
       * 提交之后不等待，直接计算当前 tile。
       */
      pipeline.producer_commit();
    }

    // ----------------------------------------------------
    // 计算当前 read_stage
    //
    // 与此同时，下一块正在搬到 write_stage。
    // ----------------------------------------------------

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        /*
         * 注意 A 现在没有转置：
         *
         * 之前：
         *     As[kk][m]
         *
         * 现在：
         *     As[m][kk]
         */
        a_reg[i] = As[read_stage][thread_m * TM + i][kk];
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_reg[j] = Bs[read_stage][kk][thread_n * TN + j];
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          accumulation[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }

    /*
     * 当前 pipeline stage 已消费完毕。
     *
     * release 只是更新当前线程 pipeline 的状态；
     * 后面的 __syncthreads 保证整个 block 都已经读取完
     * 当前 shared buffer，之后才能覆盖它。
     */
    pipeline.consumer_release();

    __syncthreads();
  }

  // ========================================================
  // 写回 C
  // ========================================================

  const int output_row = block_row + thread_m * TM;

  const int output_col = block_col + thread_n * TN;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      C[(output_row + i) * N + output_col + j] = accumulation[i][j];
    }
  }
}

int main() {
  constexpr int32_t m = 512;  // 矩阵 A 的行数
  constexpr int32_t k = 256;  // 矩阵 A 的列数，也是矩阵 B 的行数
  constexpr int32_t n = 256;  // 矩阵 B 的列数

  const size_t mem_size_A = m * k * sizeof(float);
  const size_t mem_size_B = k * n * sizeof(float);
  const size_t mem_size_C = m * n * sizeof(float);

  float* matrix_A_host = static_cast<float*>(std::malloc(mem_size_A));
  float* matrix_B_host = static_cast<float*>(std::malloc(mem_size_B));
  float* matrix_C_host_gpu_calc = static_cast<float*>(std::malloc(mem_size_C));
  float* matrix_C_host_cpu_calc = static_cast<float*>(std::malloc(mem_size_C));

  // 初始化矩阵数据
  random_matrix(m, k, matrix_A_host);
  random_matrix(k, n, matrix_B_host);
  std::memset(matrix_C_host_gpu_calc, 0, mem_size_C);
  std::memset(matrix_C_host_cpu_calc, 0, mem_size_C);

  float* matrix_A_device = nullptr;
  float* matrix_B_device = nullptr;
  float* matrix_C_device = nullptr;

  // 分配设备内存
  cudaMalloc(reinterpret_cast<void**>(&matrix_A_device), mem_size_A);
  cudaMalloc(reinterpret_cast<void**>(&matrix_B_device), mem_size_B);
  cudaMalloc(reinterpret_cast<void**>(&matrix_C_device), mem_size_C);

  // 拷贝输入数据到设备
  cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A,
             cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B,
             cudaMemcpyHostToDevice);

  // Tile 与分块配置参数
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;
  constexpr int PAD = 4;

  static_assert(BM % TM == 0);
  static_assert(BN % TN == 0);

  if (m % BM != 0 || n % BN != 0 || k % BK != 0 || k % 4 != 0 || n % 4 != 0) {
    std::fprintf(stderr,
                 "Current kernel requires M%%BM=N%%BN=K%%BK=0 "
                 "and K,N divisible by 4\n");
    return -1;
  }

  // 线程块与网格维度配置：
  // 1. block.x 对应 C 列方向的线程数 (BN / TN)
  // 2. block.y 对应 C 行方向的线程数 (BM / TM)
  // 3. grid.x 对应 C 列方向的 block 数 (N / BN)
  // 4. grid.y 对应 C 行方向的 block 数 (M / BM)
  dim3 block(BN / TN, BM / TM);
  dim3 grid(n / BN, m / BM);

  // 启动 Kernel
  cuda_sgemm_double_buffer<BM, BN, BK, TM, TN, PAD><<<grid, block>>>(
      matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

  // 同步并检查执行错误
  cudaError_t err_sync = cudaDeviceSynchronize();
  if (err_sync != cudaSuccess) {
    std::printf("Kernel execution failed: %s\n", cudaGetErrorString(err_sync));
    return -1;
  }

  // 检查启动错误
  cudaError_t err_async = cudaGetLastError();
  if (err_async != cudaSuccess) {
    std::printf("Kernel launch failed: %s\n", cudaGetErrorString(err_async));
    return -1;
  }

  // CPU 验证计算
  cpu_sgemm(m, n, k, matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc);

  // 将 GPU 结果拷回 Host
  cudaError_t err_cpy = cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device,
                                   mem_size_C, cudaMemcpyDeviceToHost);
  if (err_cpy != cudaSuccess) {
    std::printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err_cpy));
    return -1;
  }

  // 验证精度
  constexpr float kMaxDiffTolerance = 1e-3f;
  const float max_diff =
      calc_max_diff(m, n, matrix_C_host_cpu_calc, matrix_C_host_gpu_calc);
  const bool is_correct = max_diff <= kMaxDiffTolerance;

  std::printf("max diff: %.8f\n", max_diff);
  std::printf("result: %s\n", is_correct ? "correct" : "incorrect");

  // 耗时与性能测速 (Warmup + 重复测速)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    cuda_sgemm_double_buffer<BM, BN, BK, TM, TN, PAD><<<grid, block>>>(
        matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
  }
  cudaDeviceSynchronize();

  constexpr int kRepeat = 100;
  cudaEventRecord(start);
  for (int i = 0; i < kRepeat; ++i) {
    cuda_sgemm_double_buffer<BM, BN, BK, TM, TN, PAD><<<grid, block>>>(
        matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  const float avg_ms = elapsed_ms / kRepeat;
  const double gflops =
      (2.0 * static_cast<double>(m) * n * k * 1e-9) / (avg_ms / 1000.0);
  std::printf("avg kernel time: %.4f ms, perf: %.2f GFLOPS\n", avg_ms, gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // 释放资源
  cudaFree(matrix_A_device);
  cudaFree(matrix_B_device);
  cudaFree(matrix_C_device);
  std::free(matrix_A_host);
  std::free(matrix_B_host);
  std::free(matrix_C_host_gpu_calc);
  std::free(matrix_C_host_cpu_calc);

  return 0;
}