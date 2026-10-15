#pragma once

#include <cuda_runtime.h>

// benchmark 的 host 启动入口；tile 和线程配置见各实现。
namespace sgemm_v1 {
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream);
}
namespace sgemm_v2 {
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream);
}
namespace sgemm_v3 {
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream);
}
namespace sgemm_v4 {
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream);
}
namespace sgemm_v5 {
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream);
}
namespace sgemm_v6 {
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream);
}
namespace sgemm_v7 {
void launch(float* A, float* B, float* C, int M, int N, int K,
            cudaStream_t stream);
}

namespace sgemm_warp_tiling {
void launch32(float* A, float* B, float* C, int M, int N, int K,
              cudaStream_t stream);
void launch64(float* A, float* B, float* C, int M, int N, int K,
              cudaStream_t stream);
}

#ifdef SGEMM_WITH_CUTLASS
namespace sgemm_cutlass {
const char* version();
void prepare32(float* A, float* B, float* C, int M, int N, int K,
               cudaStream_t stream);
void prepare64(float* A, float* B, float* C, int M, int N, int K,
               cudaStream_t stream);
void prepare128(float* A, float* B, float* C, int M, int N, int K,
                cudaStream_t stream);
void launch32(float* A, float* B, float* C, int M, int N, int K,
              cudaStream_t stream);
void launch64(float* A, float* B, float* C, int M, int N, int K,
              cudaStream_t stream);
void launch128(float* A, float* B, float* C, int M, int N, int K,
               cudaStream_t stream);
}
#endif
