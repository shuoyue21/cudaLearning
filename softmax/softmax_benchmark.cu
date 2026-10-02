#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                                 \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// -----------------------------------------------------------------------------
// 前向声明: 来自外部算子文件 (如 softmax_naive.cu) 的 CPU 基准实现与核函数
// -----------------------------------------------------------------------------
extern void softmax_forward_cpu(float* out, const float* inp, int N, int C);
__global__ void softmax_forward_kernel2(float* out, const float* inp, int N,
                                        int C);

// -----------------------------------------------------------------------------
// 误差检验辅助函数
// -----------------------------------------------------------------------------
bool check_results(const float* cpu_out, const float* gpu_out, int size,
                   float tol = 1e-4f, float* max_diff_out = nullptr) {
  float max_diff = 0.0f;
  bool match = true;
  for (int i = 0; i < size; ++i) {
    float diff = std::fabs(cpu_out[i] - gpu_out[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tol) {
      match = false;
    }
  }
  if (max_diff_out) *max_diff_out = max_diff;
  return match;
}

// -----------------------------------------------------------------------------
// 单个核函数的测试与评测函数
// -----------------------------------------------------------------------------
float benchmark_kernel(int kernel_id, float* d_out, const float* d_inp,
                       float* h_out_gpu, const float* h_out_cpu, int N, int C,
                       float cpu_time_ms, int repeat, int warmup) {
  size_t bytes = static_cast<size_t>(N) * C * sizeof(float);

  // 根据核函数配置网格
  int blockSize = 128;
  int numBlocks = 0;
  size_t shared_mem_size = 0;

  if (kernel_id == 1) {
    blockSize = 128;
    numBlocks = (N + blockSize - 1) / blockSize;
    shared_mem_size = 0;
  } else if (kernel_id == 2) {
    blockSize = 128;
    numBlocks = N;
    shared_mem_size = blockSize * sizeof(float);
  }

  // 1. 预热 (Warm-up)
  for (int w = 0; w < warmup; ++w) {
    softmax_forward_kernel2<<<numBlocks, blockSize, shared_mem_size>>>(
        d_out, d_inp, N, C);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2. CUDA Event 测量 GPU 时间
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < repeat; ++r) {
    softmax_forward_kernel2<<<numBlocks, blockSize, shared_mem_size>>>(
        d_out, d_inp, N, C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float gpu_time_ms = total_ms / static_cast<float>(repeat);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // 3. 拷回数据并校验准确性
  CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
  float max_diff = 0.0f;
  bool match = check_results(h_out_cpu, h_out_gpu, N * C, 1e-4f, &max_diff);

  // 4. 打印格式 (与预期一致)
  std::printf("Results match: %s\n", match ? "YES" : "NO");
  std::printf("CPU time: %.4f ms\n", cpu_time_ms);
  std::printf("GPU time: %.6f ms\n", gpu_time_ms);
  std::printf("Speedup: %.3fx\n", cpu_time_ms / gpu_time_ms);

  return gpu_time_ms;
}

// -----------------------------------------------------------------------------
// 主函数: 参数解析与评测控制
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  // 默认配置
  std::string choice = "all";  // 可选: "1", "2", "all"
  int N = 320;                 // 默认行数
  int C = 4096;                // 默认每行元素数
  int repeat = 100;            // 默认评测轮数
  int warmup = 10;             // 默认预热轮数

  // 命令行参数解析:
  // 用法: ./softmax_benchmark [kernel_choice: 1|2|all] [N] [C] [repeat]
  if (argc > 1) {
    choice = argv[1];
  }
  if (argc > 2) {
    N = std::atoi(argv[2]);
  }
  if (argc > 3) {
    C = std::atoi(argv[3]);
  }
  if (argc > 4) {
    repeat = std::atoi(argv[4]);
  }

  std::printf("====================================================\n");
  std::printf(" Softmax Benchmark Framework\n");
  std::printf(" Dimensions : N = %d, C = %d (Total Elements: %zu)\n", N, C,
              static_cast<size_t>(N) * C);
  std::printf(" Iterations : %d (Warmup: %d)\n", repeat, warmup);
  std::printf(" Target     : Kernel %s\n", choice.c_str());
  std::printf("====================================================\n\n");

  size_t total_elements = static_cast<size_t>(N) * C;
  size_t bytes = total_elements * sizeof(float);

  // 1. 分配 Host 内存
  float* h_inp = static_cast<float*>(std::malloc(bytes));
  float* h_out_cpu = static_cast<float*>(std::malloc(bytes));
  float* h_out_gpu = static_cast<float*>(std::malloc(bytes));

  if (!h_inp || !h_out_cpu || !h_out_gpu) {
    std::fprintf(stderr, "Host memory allocation failed!\n");
    return EXIT_FAILURE;
  }

  // 2. 随机初始化数据
  std::srand(2026);
  for (size_t i = 0; i < total_elements; ++i) {
    h_inp[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }

  // 3. 运行 CPU 基准测试
  std::printf("[1/3] Running CPU reference...\n");
  auto cpu_t0 = std::chrono::high_resolution_clock::now();
  softmax_forward_cpu(h_out_cpu, h_inp, N, C);
  auto cpu_t1 = std::chrono::high_resolution_clock::now();
  float cpu_time_ms =
      std::chrono::duration<float, std::milli>(cpu_t1 - cpu_t0).count();
  std::printf("CPU done in %.4f ms\n\n", cpu_time_ms);

  // 4. 分配 Device 显存并拷贝输入
  float *d_inp = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_inp, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_inp, h_inp, bytes, cudaMemcpyHostToDevice));

  // 5. 执行选定的核函数评测
  float t_gpu1 = 0.0f;
  float t_gpu2 = 0.0f;

  if (choice == "1" || choice == "all") {
    std::printf("--- Kernel 1 (Naive: 1 Thread / Row) ---\n");
    t_gpu1 = benchmark_kernel(1, d_out, d_inp, h_out_gpu, h_out_cpu, N, C,
                              cpu_time_ms, repeat, warmup);
    std::printf("\n");
  }

  if (choice == "2" || choice == "all") {
    std::printf("--- Kernel 2 (Block Coarsening & Shared Memory) ---\n");
    t_gpu2 = benchmark_kernel(2, d_out, d_inp, h_out_gpu, h_out_cpu, N, C,
                              cpu_time_ms, repeat, warmup);
    std::printf("\n");
  }

  // 6. 如果同时跑了两个核函数，输出对比总结
  if (choice == "all") {
    std::printf("================ Performance Summary ================\n");
    std::printf("Kernel 1 GPU Time : %.6f ms\n", t_gpu1);
    std::printf("Kernel 2 GPU Time : %.6f ms\n", t_gpu2);
    std::printf("Kernel 2 vs Kernel 1 Speedup : %.2fx\n", t_gpu1 / t_gpu2);
    std::printf("====================================================\n");
  }

  // 7. 释放资源
  CUDA_CHECK(cudaFree(d_inp));
  CUDA_CHECK(cudaFree(d_out));
  std::free(h_inp);
  std::free(h_out_cpu);
  std::free(h_out_gpu);

  return 0;
}
