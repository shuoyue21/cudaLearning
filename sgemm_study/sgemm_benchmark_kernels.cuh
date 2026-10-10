#pragma once

#include <cuda_runtime.h>

// 与各版本 main 中相同的配置；benchmark 直接链接原文件里的 kernel。
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
