
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda/pipeline>
#include <random>
#ifdef SGEMM_BENCHMARK
#include "sgemm_benchmark_kernels.cuh"
namespace sgemm_v7 {
#endif

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

// 检查地址是否16字节对齐
__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0x0F) == 0;
}

// 一次搬运4个(如果数量够且能够对齐的情况下) 不够就fallback回普通的搬运
// 如果超过范围就写入零
template <typename Pipeline>
__device__ __forceinline__ void copy_vec4_or_zero_async(float* shared_dst,
                                                        const float* global_src,
                                                        int valid_count,
                                                        Pipeline& pipeline) {
  constexpr int VEC_WIDTH = 4;
  constexpr int COPY_BYTES = VEC_WIDTH * sizeof(float);

  if (valid_count == VEC_WIDTH && is_aligned_16(shared_dst) &&
      is_aligned_16(global_src)) {
    cuda::memcpy_async(shared_dst, global_src,
                       cuda::aligned_size_t<16>(COPY_BYTES), pipeline);
  } else {
    // 这里为什么是VEC_WIDTH ? 因为存在不对齐/以及
    // valid_count不足的情况
    // 这两种情况都要按照VEC_WIDTH情况搬,不对齐肯定了,然后因为shared_mem是按照16字节对齐来填充的,多余的数据要补0
#pragma unroll
    for (int i = 0; i < VEC_WIDTH; ++i) {
      // 在搬运的有效范围内
      if (i < valid_count) {
        shared_dst[i] = global_src[i];
      } else {
        shared_dst[i] = 0.f;
      }
    }
  }
}

template <int Bm, int Bn, int Bk, int A_PAD, int B_PAD, int NUM_THREADS,
          typename Pipeline>
__device__ __forceinline__ void copy_tile_predicated_async(
    const float* __restrict__ A, const float* __restrict__ B,

    // 分别指向 As[stage][0][0] 和 Bs[stage][0][0]
    float* As_stage, float* Bs_stage,

    int block_row, int block_col, int k_base,

    int M, int N, int K, int tid,

    Pipeline& pipeline) {
  constexpr int VEC_WIDTH = 4;

  constexpr int AS_STRIDE = Bk + A_PAD;
  constexpr int BS_STRIDE = Bn + B_PAD;

  constexpr int A_VEC_PER_ROW = Bk / VEC_WIDTH;    // 一行有几个float4
  constexpr int A_VEC_COUNT = Bm * A_VEC_PER_ROW;  // 总共有多少个float4
#pragma unroll
  for (int vec_id = tid; vec_id < A_VEC_COUNT; vec_id += NUM_THREADS) {
    // 当前tile的局部坐标
    // tid 是线程编号 vec_id是当前线程领取的搬运任务编号
    // 这里把任务编号转换为对应的二维坐标
    // local_row local_col 就是要搬的对应的数据在原本的A tile对应的起始下标
    const int local_row = vec_id / A_VEC_PER_ROW;
    const int local_col = (vec_id % A_VEC_PER_ROW) * VEC_WIDTH;

    const int global_row =
        block_row + local_row;  // 用来计算global_src 但是有可能非法
    const int global_col = k_base + local_col;
    /*
     * shared_dst 始终合法，因为它属于固定大小的
     * shared-memory tile。
     */
    float* shared_dst = As_stage + local_row * AS_STRIDE + local_col;

    const float* global_src = A;
    int valid_count = 0;
    if (global_row < M &&
        global_col <
            K) {  // 因为每个当前策略是每个block搬运固定大小tile(也就是一个shared_mem大小)的块,所以存在超过global的情况
      const int remaining = K - global_col;
      valid_count = remaining >= VEC_WIDTH ? VEC_WIDTH : remaining;
      global_src = A + static_cast<std::size_t>(global_row) * K + global_col;
    }
    copy_vec4_or_zero_async(shared_dst, global_src, valid_count, pipeline);
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

    const int global_row = k_base + local_row;
    const int global_col = block_col + local_col;
    float* shared_dst = Bs_stage + local_row * BS_STRIDE + local_col;

    const float* global_src = B;
    int valid_count = 0;
    if (global_row < K && global_col < N) {
      const int remaining = N - global_col;
      valid_count = remaining >= VEC_WIDTH ? VEC_WIDTH : remaining;
      global_src = B + static_cast<std::size_t>(global_row) * N + global_col;
    }
    copy_vec4_or_zero_async(shared_dst, global_src, valid_count, pipeline);
  }
}

// warp tile for any size
template <unsigned int Bm,  // block tile M
          unsigned int Bn,  // block tile N
          unsigned int Bk,  // block tile K
          unsigned int Wm,  // warp tile M
          unsigned int Wn,  // warp tile N
          unsigned int Tm,  // 每线程计算的行数
          unsigned int Tn,  // 每线程计算的列数
          unsigned int A_PAD = 4, unsigned int B_PAD = 4>
__global__ void cuda_sgemm(const float* A, const float* B, float* C,
                           const int M, const int N, const int K) {
  // 第一步:确定Block tile
  // 第二步:确定Warp tile
  // 第三步:确定Thread tile
  // 第四步:设计Lane到输出元素的映射
  // 第五步:组合全局坐标
  constexpr int NUM_STAGES = 2;
  constexpr int VEC_WIDTH = 4;
  constexpr int WARP_SIZE = 32;

  constexpr int WARPS_M = Bm / Wm;  // 在行方向(M) 有多少个warp tile
  constexpr int WARPS_N = Bn / Wn;

  constexpr int NUM_WARPS = WARPS_M * WARPS_N;
  constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

  // lane 布局
  constexpr int LANES_M = Wm / Tm;
  constexpr int LANES_N = Wn / Tn;

  static_assert(Bm % Wm == 0);
  static_assert(Bn % Wn == 0);

  static_assert(Wm % Tm == 0);
  static_assert(Wn % Tn == 0);

  static_assert(LANES_M * LANES_N == WARP_SIZE);

  /*
   * tile 本身仍然按 float4 划分搬运任务。
   * 这是 tile 参数约束，不是 M/N/K 的约束。
   */
  static_assert(Bk % VEC_WIDTH == 0);
  static_assert(Bn % VEC_WIDTH == 0);
  static_assert(Tn % VEC_WIDTH == 0);

  /*
   * shared-memory 每一行必须保持 16-byte 对齐。
   */
  static_assert((Bk + A_PAD) % VEC_WIDTH == 0);
  static_assert((Bn + B_PAD) % VEC_WIDTH == 0);

  static_assert(NUM_THREADS <= 1024);

  // 各种warp和lane的坐标
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id % WARPS_N;
  const int lane_id = tid % WARP_SIZE;
  const int lane_m = lane_id / LANES_N;
  const int lane_n = lane_id % LANES_N;

  // block,warp,thread的输出位置
  const int block_row = blockIdx.y * Bm;
  const int block_col = blockIdx.x * Bn;

  const int warp_row = warp_m * Wm;
  const int warp_col = warp_n * Wn;

  /*
   * M 方向采用交错行映射：
   *
   * thread_row_base
   * thread_row_base + LANES_M
   * thread_row_base + 2*LANES_M
   * ...
   */
  const int thread_row_base =
      warp_row +
      lane_m;  // 当前线程在 Warp Tile 中负责计算的输出元素的起始行、起始列。

  const int thread_col_base = warp_col + lane_n * Tn;

  __shared__ __align__(16) float As[NUM_STAGES][Bm][Bk + A_PAD];
  __shared__ __align__(16) float Bs[NUM_STAGES][Bk][Bn + B_PAD];

  // reg
  float a_fragment[Tm] = {0.f};
  float b_fragment[Tn] = {0.f};
  float accumulation[Tm][Tn] = {0.f};

  const int num_k_tile =
      (K + Bk - 1) /
      Bk;  // 向上取整 tile多出来的部分由copy_tile_predicated_async补0
  if (num_k_tile > 0) {
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
    pipeline.producer_acquire();

    copy_tile_predicated_async<Bm, Bn, Bk, A_PAD, B_PAD, NUM_THREADS>(
        A, B, &As[0][0][0], &Bs[0][0][0], block_row, block_col, 0, M, N, K, tid,
        pipeline);
    /*
     * 边界判断可能让 warp 内部分 lane 进入不同分支。
     * commit 前重新收敛，避免 pipeline batch 过度分裂。
     */
    __syncwarp();
    pipeline.producer_commit();
#pragma unroll
    for (int tile_id = 0; tile_id < num_k_tile; ++tile_id) {
      const int read_stage = tile_id & 1;

      bool has_next_tile = tile_id + 1 < num_k_tile;

      if (has_next_tile) {
        const int write_stage = read_stage ^ 1;
        int next_k_base = (tile_id + 1) * Bk;
        pipeline.producer_acquire();
        copy_tile_predicated_async<Bm, Bn, Bk, A_PAD, B_PAD, NUM_THREADS>(
            A, B, &As[write_stage][0][0], &Bs[write_stage][0][0], block_row,
            block_col, next_k_base, M, N, K, tid, pipeline);
        __syncwarp();
        pipeline.producer_commit();
      }
      pipeline.consumer_wait();
      __syncthreads();

      // compute

#pragma unroll
      for (int kk = 0; kk < Bk; ++kk) {
#pragma unroll
        for (int i = 0; i < Tm; ++i) {
          const int local_row = thread_row_base + i * LANES_M;
          a_fragment[i] = As[read_stage][local_row][kk];
        }
#pragma unroll
        for (int i = 0; i < Tn; ++i) {
          const int local_col = thread_col_base + i;
          b_fragment[i] = Bs[read_stage][kk][local_col];
        }

        // TM × TN 外积
#pragma unroll
        for (int i = 0; i < Tm; ++i) {
#pragma unroll
          for (int j = 0; j < Tn; ++j) {
            accumulation[i][j] += a_fragment[i] * b_fragment[j];
          }
        }
      }
      pipeline.consumer_release();
      __syncthreads();
    }

    // 写回 以每个线程的寄存器 accumulate作为写回单位
    constexpr int C_VEC_PER_ROW = Tn / VEC_WIDTH;
    // 遍历每一行
#pragma unroll
    for (int i = 0; i < Tm; ++i) {
      const int local_row = thread_row_base + i * LANES_M;
      const int global_row = block_row + local_row;

      // 因为最后的tile可能大于C的大小,而且只能跳过不能return
      if (global_row >= M) {
        continue;
      }
#pragma unroll
      for (int vec_id = 0; vec_id < C_VEC_PER_ROW; ++vec_id) {
        const int reg_col = vec_id * VEC_WIDTH;
        const int local_col = thread_col_base + reg_col;
        const int global_col = block_col + local_col;
        if (global_col >= N) continue;

        const int remaining = N - global_col;
        const int valid_count = remaining >= VEC_WIDTH ? VEC_WIDTH : remaining;
        // 转换线性坐标
        float* global_dst =
            C + static_cast<std::size_t>(global_row) * N + global_col;

        if (valid_count == VEC_WIDTH and is_aligned_16(global_dst)) {
          const float4 value = make_float4(
              accumulation[i][reg_col], accumulation[i][reg_col + 1],
              accumulation[i][reg_col + 2], accumulation[i][reg_col + 3]);
          *reinterpret_cast<float4*>(global_dst) = value;
        } else {
#pragma unroll
          for (int j = 0; j < VEC_WIDTH; ++j) {
            if (j < valid_count) global_dst[j] = accumulation[i][reg_col + j];
          }
        }
      }
    }
  }
}

#ifdef SGEMM_BENCHMARK
  void launch(float* A, float* B, float* C, int M, int N, int K,
              cudaStream_t stream) {
    constexpr unsigned int BM = 32;
    constexpr unsigned int BN = 32;
    constexpr unsigned int BK = 32;
    constexpr unsigned int WM = 16;
    constexpr unsigned int WN = 32;
    constexpr unsigned int TM = 4;
    constexpr unsigned int TN = 4;
    constexpr unsigned int NUM_WARPS = (BM / WM) * (BN / WN);
    const dim3 block(NUM_WARPS * 32);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cuda_sgemm<BM, BN, BK, WM, WN, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  }
}  // namespace sgemm_v7
#else
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
    constexpr unsigned int WM = 16;
    constexpr unsigned int WN = 32;
    constexpr unsigned int TM = 4;
    constexpr unsigned int TN = 4;

    static_assert(BM % WM == 0);
    static_assert(BN % WN == 0);
    static_assert(WM % TM == 0);
    static_assert(WN % TN == 0);
    static_assert((WM / TM) * (WN / TN) == 32);

    constexpr unsigned int NUM_WARPS = (BM / WM) * (BN / WN);
    constexpr unsigned int NUM_THREADS = NUM_WARPS * 32;

    dim3 block(NUM_THREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    cuda_sgemm<BM, BN, BK, WM, WN, TM, TN><<<grid, block>>>(
        matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    cpu_sgemm(m, n, k, matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc);
    cudaDeviceSynchronize();
    // 强制 CPU 等待 GPU 完成，并捕获 Kernel 运行时的异步错误
    // (例如非法的内存访问、寄存器溢出等)
    cudaError_t err_sync = cudaDeviceSynchronize();
    if (err_sync != cudaSuccess) {
      std::printf("Kernel execution failed: %s\n",
                  cudaGetErrorString(err_sync));
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
#endif  // SGEMM_BENCHMARK
