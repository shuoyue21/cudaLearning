#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cuda/pipeline>

// ============================================================
// 基础辅助函数
// ============================================================

__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0x0F) == 0;
}

/*
 * 将最多 4 个 float 搬入 shared memory。
 *
 * valid_count == 4 且地址 16-byte 对齐：
 *     使用 16-byte cuda::memcpy_async
 *
 * 否则：
 *     使用普通标量复制；
 *     无效位置写 0。
 *
 * valid_count 范围为 [0, 4]。
 */
template <typename Pipeline>
__device__ __forceinline__ void copy_vec4_or_zero_async(float* shared_dst,
                                                        const float* global_src,
                                                        int valid_count,
                                                        Pipeline& pipeline) {
  constexpr int VEC_WIDTH = 4;
  constexpr std::size_t COPY_BYTES = VEC_WIDTH * sizeof(float);

  if (valid_count == VEC_WIDTH && is_aligned_16(shared_dst) &&
      is_aligned_16(global_src)) {
    cuda::memcpy_async(shared_dst, global_src,
                       cuda::aligned_size_t<16>(COPY_BYTES), pipeline);

  } else {
#pragma unroll
    for (int i = 0; i < VEC_WIDTH; ++i) {
      if (i < valid_count) {
        shared_dst[i] = global_src[i];
      } else {
        shared_dst[i] = 0.0f;
      }
    }
  }
}

// ============================================================
// 搬运一个任意边界的 A/B tile
// ============================================================

template <int BM, int BN, int BK, int A_PAD, int B_PAD, int NUM_THREADS,
          typename Pipeline>
__device__ __forceinline__ void copy_tile_predicated_async(
    const float* __restrict__ A, const float* __restrict__ B,

    float* As_stage, float* Bs_stage,

    int block_row, int block_col, int k_base,

    int M, int N, int K,

    int tid, Pipeline& pipeline) {
  constexpr int VEC_WIDTH = 4;

  constexpr int AS_STRIDE = BK + A_PAD;
  constexpr int BS_STRIDE = BN + B_PAD;

  // ========================================================
  // A tile：[BM, BK]
  // ========================================================

  constexpr int A_VECS_PER_ROW = BK / VEC_WIDTH;

  constexpr int A_VEC_COUNT = BM * A_VECS_PER_ROW;

  for (int vec_id = tid; vec_id < A_VEC_COUNT; vec_id += NUM_THREADS) {
    const int local_row = vec_id / A_VECS_PER_ROW;

    const int local_col = (vec_id % A_VECS_PER_ROW) * VEC_WIDTH;

    const int global_row = block_row + local_row;

    const int global_col = k_base + local_col;

    /*
     * shared_dst 始终合法，因为它属于固定大小的
     * shared-memory tile。
     */
    float* shared_dst = As_stage + local_row * AS_STRIDE + local_col;

    /*
     * global_src 只有在至少存在一个合法元素时才计算。
     * 默认值不会被读取，因为 valid_count=0。
     */
    const float* global_src = A;
    int valid_count = 0;

    if (global_row < M && global_col < K) {
      const int remaining = K - global_col;

      valid_count = remaining >= VEC_WIDTH ? VEC_WIDTH : remaining;

      global_src = A + static_cast<std::size_t>(global_row) * K + global_col;
    }

    copy_vec4_or_zero_async(shared_dst, global_src, valid_count, pipeline);
  }

  // ========================================================
  // B tile：[BK, BN]
  // ========================================================

  constexpr int B_VECS_PER_ROW = BN / VEC_WIDTH;

  constexpr int B_VEC_COUNT = BK * B_VECS_PER_ROW;

  for (int vec_id = tid; vec_id < B_VEC_COUNT; vec_id += NUM_THREADS) {
    const int local_row = vec_id / B_VECS_PER_ROW;

    const int local_col = (vec_id % B_VECS_PER_ROW) * VEC_WIDTH;

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

// ============================================================
// 任意 M/N/K 的显式 Warp-Tiled SGEMM
// ============================================================

template <int BM,  // block tile M
          int BN,  // block tile N
          int BK,  // block tile K

          int WM,  // warp tile M
          int WN,  // warp tile N

          int TM,  // thread tile M
          int TN,  // thread tile N

          int A_PAD = 4, int B_PAD = 0>
__global__ void cuda_sgemm_warp_tiled_any_size(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C, int M,
                                               int N, int K) {
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_STAGES = 2;
  constexpr int VEC_WIDTH = 4;

  // --------------------------------------------------------
  // Warp 布局
  // --------------------------------------------------------

  constexpr int WARPS_M = BM / WM;

  constexpr int WARPS_N = BN / WN;

  constexpr int NUM_WARPS = WARPS_M * WARPS_N;

  constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

  // --------------------------------------------------------
  // Lane 布局
  // --------------------------------------------------------

  constexpr int LANES_M = WM / TM;

  constexpr int LANES_N = WN / TN;

  // --------------------------------------------------------
  // 编译期约束
  // --------------------------------------------------------

  static_assert(BM % WM == 0);
  static_assert(BN % WN == 0);

  static_assert(WM % TM == 0);
  static_assert(WN % TN == 0);

  static_assert(LANES_M * LANES_N == WARP_SIZE);

  /*
   * tile 本身仍然按 float4 划分搬运任务。
   * 这是 tile 参数约束，不是 M/N/K 的约束。
   */
  static_assert(BK % VEC_WIDTH == 0);
  static_assert(BN % VEC_WIDTH == 0);
  static_assert(TN % VEC_WIDTH == 0);

  /*
   * shared-memory 每一行必须保持 16-byte 对齐。
   */
  static_assert((BK + A_PAD) % VEC_WIDTH == 0);
  static_assert((BN + B_PAD) % VEC_WIDTH == 0);

  static_assert(NUM_THREADS <= 1024);

  // --------------------------------------------------------
  // Warp / lane 坐标
  // --------------------------------------------------------

  const int tid = static_cast<int>(threadIdx.x);

  const int warp_id = tid / WARP_SIZE;

  const int lane_id = tid % WARP_SIZE;

  const int warp_m = warp_id / WARPS_N;

  const int warp_n = warp_id % WARPS_N;

  const int lane_m = lane_id / LANES_N;

  const int lane_n = lane_id % LANES_N;

  // --------------------------------------------------------
  // Block、warp、thread 输出位置
  // --------------------------------------------------------

  const int block_row = static_cast<int>(blockIdx.y) * BM;

  const int block_col = static_cast<int>(blockIdx.x) * BN;

  const int warp_row = warp_m * WM;

  const int warp_col = warp_n * WN;

  /*
   * M 方向采用交错行映射：
   *
   * thread_row_base
   * thread_row_base + LANES_M
   * thread_row_base + 2*LANES_M
   * ...
   */
  const int thread_row_base = warp_row + lane_m;

  /*
   * N 方向每线程负责连续 TN 列。
   */
  const int thread_col_base = warp_col + lane_n * TN;

  // --------------------------------------------------------
  // 双缓冲 shared memory
  // --------------------------------------------------------

  __shared__ __align__(16) float As[NUM_STAGES][BM][BK + A_PAD];

  __shared__ __align__(16) float Bs[NUM_STAGES][BK][BN + B_PAD];

  // --------------------------------------------------------
  // 寄存器 tile
  // --------------------------------------------------------

  float a_fragment[TM];
  float b_fragment[TN];

  float accumulator[TM][TN] = {};

  // --------------------------------------------------------
  // Pipeline
  // --------------------------------------------------------

  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

  /*
   * 向上取整。
   *
   * 例如 K=35、BK=16：
   * num_k_tiles = 3
   *
   * 最后一轮只包含 k=32~34，
   * 其余位置由 copy_tile_predicated_async 补 0。
   */
  const int num_k_tiles = (K + BK - 1) / BK;

  if (num_k_tiles > 0) {
    // ====================================================
    // Prologue：tile 0 -> stage 0
    // ====================================================

    pipeline.producer_acquire();

    copy_tile_predicated_async<BM, BN, BK, A_PAD, B_PAD, NUM_THREADS>(
        A, B,

        &As[0][0][0], &Bs[0][0][0],

        block_row, block_col, 0,

        M, N, K,

        tid, pipeline);

    /*
     * 边界判断可能让 warp 内部分 lane 进入不同分支。
     * commit 前重新收敛，避免 pipeline batch 过度分裂。
     */
    __syncwarp();

    pipeline.producer_commit();

    // ====================================================
    // Mainloop
    // ====================================================

    for (int tile_id = 0; tile_id < num_k_tiles; ++tile_id) {
      const int read_stage = tile_id & 1;

      const bool has_next_tile = tile_id + 1 < num_k_tiles;

      // ------------------------------------------------
      // 预取下一个 tile
      // ------------------------------------------------

      if (has_next_tile) {
        const int write_stage = read_stage ^ 1;

        const int next_k_base = (tile_id + 1) * BK;

        pipeline.producer_acquire();

        copy_tile_predicated_async<BM, BN, BK, A_PAD, B_PAD, NUM_THREADS>(
            A, B,

            &As[write_stage][0][0], &Bs[write_stage][0][0],

            block_row, block_col, next_k_base,

            M, N, K,

            tid, pipeline);

        __syncwarp();

        pipeline.producer_commit();
      }

      /*
       * 等当前最老的 K tile 搬运完成。
       */
      pipeline.consumer_wait();

      /*
       * 每个线程只等待自己的复制。
       * 完整 tile 由整个 block 合作搬运，因此仍需同步。
       */
      __syncthreads();

      // ------------------------------------------------
      // 计算当前 K tile
      // ------------------------------------------------

#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          const int local_row = thread_row_base + i * LANES_M;

          a_fragment[i] = As[read_stage][local_row][kk];
        }

#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int local_col = thread_col_base + j;

          b_fragment[j] = Bs[read_stage][kk][local_col];
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            accumulator[i][j] += a_fragment[i] * b_fragment[j];
          }
        }
      }

      pipeline.consumer_release();

      /*
       * 保证所有线程都不再读取 read_stage，
       * 下一轮才能覆盖对应 shared buffer。
       */
      __syncthreads();
    }
  }

  // ========================================================
  // Predicated epilogue
  // ========================================================

  constexpr int C_VECS_PER_THREAD = TN / VEC_WIDTH;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int local_row = thread_row_base + i * LANES_M;

    const int global_row = block_row + local_row;

    /*
     * 最后一块的部分线程可能完全位于 M 之外。
     * 不能提前 return，只在写回阶段跳过。
     */
    if (global_row >= M) {
      continue;
    }

#pragma unroll
    for (int vec = 0; vec < C_VECS_PER_THREAD; ++vec) {
      const int reg_col = vec * VEC_WIDTH;

      const int local_col = thread_col_base + reg_col;

      const int global_col = block_col + local_col;

      if (global_col >= N) {
        continue;
      }

      const int remaining = N - global_col;

      const int valid_count = remaining >= VEC_WIDTH ? VEC_WIDTH : remaining;

      float* global_dst =
          C + static_cast<std::size_t>(global_row) * N + global_col;

      /*
       * 完整且对齐：
       *     使用 float4 store。
       *
       * 边缘或未对齐：
       *     使用标量 store。
       */
      if (valid_count == VEC_WIDTH && is_aligned_16(global_dst)) {
        const float4 value = make_float4(
            accumulator[i][reg_col + 0], accumulator[i][reg_col + 1],
            accumulator[i][reg_col + 2], accumulator[i][reg_col + 3]);

        *reinterpret_cast<float4*>(global_dst) = value;

      } else {
#pragma unroll
        for (int j = 0; j < VEC_WIDTH; ++j) {
          if (j < valid_count) {
            global_dst[j] = accumulator[i][reg_col + j];
          }
        }
      }
    }
  }
}