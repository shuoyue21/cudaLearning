
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
namespace sgemm_v6 {
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

template <int Bm, int Bn, int Bk, int PAD, int NUM_THREADS, typename Pipeline>
__device__ __forceinline__ void copy_tile_async(
    const float* __restrict__ A, const float* __restrict__ B,

    // 分别指向 As[stage][0][0] 和 Bs[stage][0][0]
    float* As_stage, float* Bs_stage,

    int block_row, int block_col, int k_base,

    int N, int K, int tid,

    Pipeline& pipeline) {
  constexpr int VEC_WIDTH = 4;

  constexpr int AS_STRIDE = Bk + PAD;
  constexpr int BS_STRIDE = Bn + PAD;

  constexpr int COPY_BITES = sizeof(float) * VEC_WIDTH;
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
    const float* global_src =
        A + (block_row + local_row) * K + k_base + local_col;

    float* shared_src = As_stage + local_row * AS_STRIDE + local_col;

    cuda::memcpy_async(shared_src, global_src,
                       cuda::aligned_size_t<16>(COPY_BITES), pipeline);
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
    const float* global_src =
        B + (local_row + k_base) * N + block_col + local_col;

    float* shared_src = Bs_stage + local_row * BS_STRIDE + local_col;

    cuda::memcpy_async(shared_src, global_src,
                       cuda::aligned_size_t<16>(COPY_BITES), pipeline);
  }
}

template <unsigned int Bm, unsigned int Bn, unsigned int Bk,
          unsigned int Tm,  // 每线程计算的行数
          unsigned int Tn,  // 每线程计算的列数
          unsigned int PAD = 4>
__global__ void cuda_sgemm(const float* A, const float* B, float* C,
                           const int M, const int N, const int K) {
  // 同一批物理线程有两种逻辑映射：
  // 1. 使用 threadIdx.x/y 决定每个线程计算哪个 TM×TN 输出块
  // 2. 使用一维 tid 给线程分配 float4 搬运任务
  // 物理线程的 布局的关系
  constexpr int NUM_STAGES = 2;
  constexpr int VEC_WIDTH = 4;
  constexpr int THREAD_M = Bm / Tm;  // C tile 需要多少行线程
  constexpr int THREAD_N = Bn / Tn;
  constexpr int NUM_THREADS = THREAD_M * THREAD_N;  // 总线程数

  static_assert(Bm % Tm == 0);
  static_assert(Bn % Tn == 0);

  static_assert(Bk % VEC_WIDTH == 0);
  static_assert(Bn % VEC_WIDTH == 0);

  // 每一行的 shared 地址仍然必须保持 16 字节对齐。
  static_assert((Bk + PAD) % VEC_WIDTH == 0);
  static_assert((Bn + PAD) % VEC_WIDTH == 0);

  static_assert(NUM_THREADS <= 1024);

  // 计算线程id 这里为什么一致呢?
  const int thread_m = threadIdx.y;
  const int thread_n = threadIdx.x;

  // 搬运线程tid
  const int tid = threadIdx.y * THREAD_N + threadIdx.x;

  // 当前 block 对应的 C tile 左上角全局坐标。
  const int block_row = blockIdx.y * Bm;
  const int block_col = blockIdx.x * Bn;

  __shared__ __align__(16) float As[NUM_STAGES][Bm][Bk + PAD];
  __shared__ __align__(16) float Bs[NUM_STAGES][Bk][Bn + PAD];

  // reg
  float a_reg[Tm] = {0.f};
  float b_reg[Tn] = {0.f};
  float accumulation[Tm][Tn] = {0.f};

  const int num_k_tile = K / Bk;
  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
  pipeline.producer_acquire();

  copy_tile_async<Bm, Bn, Bk, PAD, NUM_THREADS>(
      A, B, &As[0][0][0], &Bs[0][0][0], block_row, block_col, 0, N, K, tid,
      pipeline);

  pipeline.producer_commit();

  for (int tile_id = 0; tile_id < num_k_tile; ++tile_id) {
    const int read_stage = tile_id & 1;

    bool has_next_tile = tile_id + 1 < num_k_tile;

    if (has_next_tile) {
      const int write_stage = read_stage ^ 1;
      int next_k_base = (tile_id + 1) * Bk;
      pipeline.producer_acquire();
      copy_tile_async<Bm, Bn, Bk, PAD, NUM_THREADS>(
          A, B, &As[write_stage][0][0], &Bs[write_stage][0][0], block_row,
          block_col, next_k_base, N, K, tid, pipeline);
      pipeline.producer_commit();
    }
    pipeline.consumer_wait();
    __syncthreads();
    // compute
#pragma unroll
    for (int kk = 0; kk < Bk; ++kk) {
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
        a_reg[i] = As[read_stage][thread_m * Tm + i][kk];
      }
#pragma unroll
      for (int i = 0; i < Tn; ++i) {
        b_reg[i] = Bs[read_stage][kk][thread_n * Tn + i];
      }

      // TM × TN 外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) {
          accumulation[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }
    pipeline.consumer_release();
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

#ifdef SGEMM_BENCHMARK
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream) {
  const dim3 block(8, 8);
  const dim3 grid(N / 32, M / 32);
  cuda_sgemm<32, 32, 32, 4, 4><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
}  // namespace sgemm_v6
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
#endif  // SGEMM_BENCHMARK
