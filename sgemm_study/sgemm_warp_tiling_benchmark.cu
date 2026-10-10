#include "sgemm_benchmark_kernels.cuh"

// 模板定义直接来自学习版本，修改 kernel 后重编译即可参与对比。
#include "sgemm_warp_tiling.cu"

namespace sgemm_warp_tiling {
namespace {

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN,
          int A_PAD, int B_PAD>
void launch_config(float* A, float* B, float* C, int M, int N, int K,
                   cudaStream_t stream) {
  constexpr int NUM_WARPS = (BM / WM) * (BN / WN);
  const dim3 block(NUM_WARPS * 32);
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  cuda_sgemm_warp_tiled_any_size<BM, BN, BK, WM, WN, TM, TN, A_PAD, B_PAD>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

}  // namespace

void launch32(float* A, float* B, float* C, int M, int N, int K,
              cudaStream_t stream) {
  // 与 v6 保持相同的 block/thread tile、BK、双缓冲和 padding。
  launch_config<32, 32, 32, 16, 32, 4, 4, 4, 4>(A, B, C, M, N, K, stream);
}

void launch64(float* A, float* B, float* C, int M, int N, int K,
              cudaStream_t stream) {
  // 更大的输出 tile 增加 A/B 复用；BK=16 控制 shared 占用。
  launch_config<64, 64, 16, 16, 32, 4, 4, 4, 0>(A, B, C, M, N, K, stream);
}

}  // namespace sgemm_warp_tiling
