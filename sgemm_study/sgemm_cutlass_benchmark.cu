#include "sgemm_benchmark_kernels.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/version.h>

#include <stdexcept>
#include <string>

namespace sgemm_cutlass {
namespace {

// SIMT FP32 FMA：输入、输出和累加均为 float，不使用 Tensor Core/TF32。
// alignment=1 和标量 epilogue 保留任意 M/N/K 的边界支持。
template <int BM, int BN, int WM, int WN>
using SimtGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<BM, BN, 8>,
    cutlass::gemm::GemmShape<WM, WN, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, 1, 1, false, cutlass::arch::OpMultiplyAdd>;

using Gemm32 = SimtGemm<32, 32, 16, 32>;
using Gemm64 = SimtGemm<64, 64, 16, 32>;
using Gemm128 = SimtGemm<128, 128, 32, 64>;

// benchmark 单线程逐项测试，每个配置在计时前重新初始化一次。
Gemm32 gemm32;
Gemm64 gemm64;
Gemm128 gemm128;

void check_cutlass(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cutlassGetStatusString(status));
  }
}

template <typename Gemm>
void prepare(Gemm& gemm, float* A, float* B, float* C, int M, int N, int K,
             cudaStream_t stream) {
  typename Gemm::Arguments arguments(
      {M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1.0f, 0.0f});
  check_cutlass(Gemm::can_implement(arguments), "CUTLASS can_implement");
  if (Gemm::get_workspace_size(arguments) != 0) {
    throw std::runtime_error("CUTLASS SIMT configuration unexpectedly needs workspace");
  }
  check_cutlass(gemm.initialize(arguments, nullptr, stream), "CUTLASS initialize");
}

}  // namespace

const char* version() {
  // 源码 checkout 的 getVersionString 可能保留 CMake 占位符，使用数值版本。
  static const std::string value =
      std::to_string(cutlass::getVersionMajor()) + "." +
      std::to_string(cutlass::getVersionMinor()) + "." +
      std::to_string(cutlass::getVersionPatch());
  return value.c_str();
}

void prepare32(float* A, float* B, float* C, int M, int N, int K,
               cudaStream_t stream) {
  prepare(gemm32, A, B, C, M, N, K, stream);
}
void prepare64(float* A, float* B, float* C, int M, int N, int K,
               cudaStream_t stream) {
  prepare(gemm64, A, B, C, M, N, K, stream);
}
void prepare128(float* A, float* B, float* C, int M, int N, int K,
                cudaStream_t stream) {
  prepare(gemm128, A, B, C, M, N, K, stream);
}

void launch32(float*, float*, float*, int, int, int, cudaStream_t stream) {
  check_cutlass(gemm32.run(stream), "CUTLASS run32");
}
void launch64(float*, float*, float*, int, int, int, cudaStream_t stream) {
  check_cutlass(gemm64.run(stream), "CUTLASS run64");
}
void launch128(float*, float*, float*, int, int, int, cudaStream_t stream) {
  check_cutlass(gemm128.run(stream), "CUTLASS run128");
}

}  // namespace sgemm_cutlass
