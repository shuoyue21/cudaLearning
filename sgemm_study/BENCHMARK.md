# SGEMM v1～v6 benchmark

benchmark 直接编译 `sgemm_v1.cu`～`sgemm_v6.cu` 中的 kernel，使用每个版本原有的 tile 和线程配置。后续修改这些文件，重新编译 benchmark 就能测到变化；单独构建各版本的程序仍然使用原有 `main`。

## 编译和运行

在仓库根目录执行，使用单独的 Release 构建目录：

```bash
cmake -S sgemm_study -B build/sgemm-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/sgemm-bench --target sgemm_benchmark -j6
./build/sgemm-bench/sgemm_benchmark --csv sgemm_results.csv
```

`native` 需要 CMake 3.24 或更高版本，并在配置时能访问 GPU。如果不能自动识别，改为 GPU 的实际架构编号，例如 `-DCMAKE_CUDA_ARCHITECTURES=80`。可以用 `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` 指定 CUDA 编译器。v6 的硬件异步 global → shared 拷贝需要 sm_80 或更新架构，并编译出相应的设备代码。

如果构建目录缓存的 `CMAKE_CUDA_FLAGS_RELEASE` 为空，重新配置时会自动补上 `-O3 -DNDEBUG`。`nvcc -O3` 主要优化 CPU 端的计时循环和 kernel launcher；没有 `-G` 时，GPU 设备代码默认仍会优化。CPU 提交过慢可能让 GPU 在两个 kernel 之间空等，这段间隔也会进入 Event 测得的整组时间，所以小矩阵测试也需要关注 CPU 端编译选项。[nvcc 编译选项说明](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#optimize)

不要用 `Debug` 的 `-G -O0` 结果评判优化效果；不同版本使用相同 Release 编译选项。测试时尽量避免其他任务同时占用 GPU。

## 测试维度

维度顺序统一为 **M、N、K**，对应 `A[M,K] × B[K,N] = C[M,N]`。

| 测试组 | M × N × K | 用途 |
| --- | --- | --- |
| 方阵 | 32³、64³、96³、128³、256³、512³、1024³、2048³ | 从小矩阵到大矩阵；32/64/96 还覆盖 v6 的单块、两块、奇数 K tile |
| 原程序维度 | 512 × 256 × 256 | 对照当前学习代码 |
| 高矩阵 | 1024 × 256 × 1024 | 改变输出 tile 在 M 方向上的数量 |
| 宽矩阵 | 256 × 1024 × 1024 | 改变输出 tile 在 N 方向上的数量 |
| 短 K | 1024 × 1024 × 256 | 较少的归约步骤 |
| 长 K | 512 × 512 × 2048 | 更多的归约步骤与数据搬运 |
| 非整齐边界 | 127 × 193 × 65 | 检查 v1/v2 的边界处理；v3～v6 显式 SKIP |

v3～v6 当前没有边界保护，只测试 M、N、K 均为 32 倍数的输入。benchmark 会跳过不支持的版本，不会为了测试尺寸改变 kernel 的实现。

自定义多个维度或计时参数：

```bash
./build/sgemm-bench/sgemm_benchmark \
  --shape 512 256 256 --shape 1024 1024 1024 --shape 512 512 4096 \
  --warmup 20 --repeat 200 --samples 7 --device 0 \
  --csv sgemm_results.csv

# 无需 GPU，查看测试用例和支持的版本
./build/sgemm-bench/sgemm_benchmark --list
./build/sgemm-bench/sgemm_benchmark --help
```

## 校验和计时

每组维度使用固定种子 2026 生成同一份均匀分布于 [-0.5, 0.5] 的 A、B。cuBLAS 在 `CUBLAS_PEDANTIC_MATH` 模式下生成 FP32 参考矩阵，并用 CPU double 点积抽查九个位置，检查行列主序转换。每个版本计时前逐元素校验，阈值为 `abs(got-ref) <= 1e-3 + 1e-4 * abs(ref)`，NaN/Inf 直接判错。C 在校验前填充 NaN，以捕获遗漏写回。

校验通过才会预热和计时。默认预热 20 次，然后测量 7 组，每组连续运行 200 次。使用同一 CUDA stream 上的 Event 测量每组耗时，除以运行次数，再取各组单次耗时的中位数。数据分配、校验、数据传输和 cuBLAS 参考计算均在计时之外。连续运行会复用相同数据，因此这是预热后的重复调用表现。增加重复次数可以减少短时间测量的波动，但不会消除 CPU 提交间隔；小矩阵的结果仍可能受 kernel 启动间隔影响。

| 输出 | 含义 |
| --- | --- |
| `Status` | PASS：正确；FAIL：数值错误，取消计时；SKIP：尺寸不支持 |
| `Time(ms)` | 单次调用平均耗时的中位数，单位毫秒 |
| `GFLOPS` | `2*M*N*K / (Time(ms)*1e6)`，把一次乘加计为两次浮点运算 |
| `Speedup/v1` | 同一维度下 `v1耗时 / 当前版本耗时`，越大越快；v1 错误时无有效基线 |
| `Max abs error` | 相对 cuBLAS 参考结果的最大绝对误差 |

CSV 每个维度、每个版本输出一行，错误或跳过的耗时/吞吐留空。任意数值校验失败时退出码为 2；CUDA、参数或文件错误时退出码为 1；全部支持的测试通过时为 0。

目前 v1、v2 的计算逻辑相同，预期性能接近。后续版本是否更快取决于矩阵大小和 GPU：比较不同维度下的曲线，再判断增加每线程工作量或双缓冲是否带来了收益。

参考：[cuBLAS 存储布局与数学模式](https://docs.nvidia.com/cuda/cublas/index.html)、[CUDA Event 计时](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)。
