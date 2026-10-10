# SGEMM v1～v6、Warp tiling 与 CUTLASS benchmark

benchmark 直接编译 `sgemm_v1.cu`～`sgemm_v6.cu` 中的 kernel，使用每个版本原有的 tile 和线程配置。`sgemm_warp_tiling_benchmark.cu` 直接包含 `sgemm_warp_tiling.cu` 的模板定义，加入 `warp32`、`warp64` 两组配置。后续修改 kernel，重新编译 benchmark 就能测到变化；单独构建 v1～v6 仍然使用原有 `main`。

## 编译和运行

在仓库根目录执行，使用单独的 Release 构建目录：

```bash
cmake -S sgemm_study -B build/sgemm-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/sgemm-bench --target sgemm_benchmark -j6
./build/sgemm-bench/sgemm_benchmark --csv sgemm_results.csv
```

`native` 需要 CMake 3.24 或更高版本，并在配置时能访问 GPU。如果不能自动识别，改为 GPU 的实际架构编号，例如 `-DCMAKE_CUDA_ARCHITECTURES=80`。可以用 `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` 指定 CUDA 编译器。v6、warp32、warp64 的硬件异步 global → shared 拷贝需要 sm_80 或更新架构，并编译出相应的设备代码。

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
| 非整齐边界 | 127 × 193 × 65 | 检查 v1/v2、warp32/warp64 的边界处理；v3～v6 显式 SKIP |

v3～v6 当前没有边界保护，只测试 M、N、K 均为 32 倍数的输入。benchmark 会跳过不支持的版本，不会为了测试尺寸改变 kernel 的实现。warp32、warp64 支持任意正整数 M、N、K，包括未对齐行跨度和尾部不足四个元素的情况。

两组 warp tiling 配置共用同一个 kernel：

| 名称 | BM × BN × BK | WM × WN | TM × TN | 线程/block | A_PAD / B_PAD | shared/block |
| --- | --- | --- | --- | --- | --- | --- |
| warp32 | 32 × 32 × 32 | 16 × 32 | 4 × 4 | 64 | 4 / 4 | 18 KiB |
| warp64 | 64 × 64 × 16 | 16 × 32 | 4 × 4 | 256 | 4 / 0 | 18 KiB |

warp32 与 v6 使用相同的 block tile、thread tile、BK、stage 数和 padding；warp32 还包含边界和对齐判断，因此对比结果同时包含这些检查的成本。warp64 使用更大的输出 tile 和更短的 K tile，是另一组候选配置，需要通过测量判断适用的矩阵形状。

默认 14 组尺寸、不启用 CUTLASS 时有 8 个版本/配置，共输出 112 行结果，其中 4 行是 v3～v6 对非整齐尺寸的 SKIP。启用 CUTLASS 后增加 3 个配置；`--kernels` 可以筛选参与对比的版本。所有版本使用相同输入、参考结果、校验阈值和计时流程。

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

CSV 的 `version` 列会包含 `warp32`、`warp64`，可以与 v1～v6 按同一 M/N/K 比较。建议使用 `--csv sgemm_results_warp.csv` 保存新结果，便于对照原有 CSV。

目前 v1、v2 的计算逻辑相同，预期性能接近。后续版本是否更快取决于矩阵大小和 GPU：比较不同维度下的曲线，再判断增加每线程工作量或双缓冲是否带来了收益。

参考：[cuBLAS 存储布局与数学模式](https://docs.nvidia.com/cuda/cublas/index.html)、[CUDA Event 计时](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)。


## CUTLASS FP32 对比

使用 [NVIDIA CUTLASS v3.5.1](https://github.com/NVIDIA/cutlass/tree/v3.5.1)，对应提交 `f7b19de32c5d1f3cedfc735c2849f12b537522ee`，兼容当前 CUDA 12.1。依赖放在忽略提交的 `.cache` 中；通过 CMake 显式启用，普通 benchmark 构建不需要 CUTLASS。

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git .cache/cutlass-v3.5.1
cmake -S sgemm_study -B build/sgemm-bench \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native \
  -DSGEMM_ENABLE_CUTLASS=ON -DCUTLASS_ROOT="$PWD/.cache/cutlass-v3.5.1"
cmake --build build/sgemm-bench --target sgemm_benchmark -j6
./build/sgemm-bench/sgemm_benchmark --device 1 --suite extended \
  --kernels warp32,warp64,cutlass32,cutlass64,cutlass128 \
  --warmup 20 --repeat 200 --samples 7 --csv sgemm_results_cutlass.csv
```

依赖已经存在时跳过 clone；用 `--device` 指定当前空闲的 GPU。用 `-DSGEMM_ENABLE_CUTLASS=OFF` 可恢复无 CUTLASS 依赖的构建。

CUTLASS 主对照使用 `OpClassSimt`、`OpMultiplyAdd`，输入、输出、累加均为 FP32，和手写 Warp tiling 使用同一误差阈值。这里比较的是三个固定 CUDA-core 配置，并非 CUTLASS 所有配置的最优值；Tensor Core 的 TF32/FP16/BF16 实现属于其他计算路径。

| 配置 | BM × BN × BK | WM × WN × WK | stage 数 | A/B 对齐元素数 |
| --- | --- | --- | --- | --- |
| cutlass32 | 32 × 32 × 8 | 16 × 32 × 8 | 2 | 1 |
| cutlass64 | 64 × 64 × 8 | 16 × 32 × 8 | 2 | 1 |
| cutlass128 | 128 × 128 × 8 | 32 × 64 × 8 | 2 | 1 |

A、B、C 都使用行主序，`alpha=1`、`beta=0`，不转换或转置输入。使用 alignment=1、标量 epilogue 支持任意正整数 M/N/K。CUTLASS 的 `can_implement` 和 `initialize` 在每组尺寸计时之前执行一次，计时循环只调用已初始化的 `run`；这三组配置不需要额外 workspace。

`--suite extended` 有 24 组尺寸：原来的 14 组，再加上 1³、1000³、4096³、256×256×4096、2048×2048×32、2048×128×1024、128×2048×1024、4096×256×256、256×4096×256、513×257×129。显式提供 `--shape M N K` 时使用自定义尺寸。

扩展测试覆盖小到大的方阵、较少/较多的 K tile、细长输出、对齐但 tile 不整齐的尾部，以及不满足 float4 对齐的行跨度。只选择上述 5 个版本时，共输出 120 行结果。未选择 v1 时，`speedup_v1` 留空；对比脚本按同一尺寸计算 CUTLASS 与 Warp tiling 的耗时比。


生成逐尺寸报告（只需 Python 标准库）：

```bash
python3 sgemm_study/compare_cutlass.py sgemm_results_cutlass.csv \
  --output-prefix sgemm_cutlass_comparison --no-plot
```

报告在同一尺寸中分别选取两个 Warp 候选、三个 CUTLASS 候选的最低耗时，计算 `Warp最快耗时 / CUTLASS最快耗时`；大于 1 表示选测的 CUTLASS 配置更快。接近 1 的差异需要重复测试确认。

可选：在独立环境中生成 PNG/SVG 图，避免现有 Python 包的 ABI 冲突（Python 3.10+）：

```bash
python3 -m venv .cache/sgemm-plot-venv
.cache/sgemm-plot-venv/bin/python -m pip install --only-binary=:all: matplotlib==3.10.0 numpy==2.2.6
.cache/sgemm-plot-venv/bin/python sgemm_study/compare_cutlass.py \
  sgemm_results_cutlass.csv --output-prefix sgemm_cutlass_comparison
```

本次运行日志保存在 `sgemm_results_cutlass.log`，含 GPU、CUDA、CUTLASS 版本和计时设置。所有 24×5=120 项数值校验通过；6 组边界尺寸的 Compute Sanitizer memcheck 也报告 0 个错误。
