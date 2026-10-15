#include "sgemm_benchmark_kernels.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

struct DeviceBuffer {
  float* data = nullptr;
  explicit DeviceBuffer(size_t count) {
    check_cuda(cudaMalloc(&data, count * sizeof(float)), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
  cudaStream_t value = nullptr;
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }
  ~Stream() { cudaStreamDestroy(value); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

struct Event {
  cudaEvent_t value = nullptr;
  Event() { check_cuda(cudaEventCreate(&value), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

struct BlasHandle {
  cublasHandle_t value = nullptr;
  BlasHandle() { check_blas(cublasCreate(&value), "cublasCreate"); }
  ~BlasHandle() { cublasDestroy(value); }
  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;
};

struct Shape {
  int m, n, k;
};

const std::vector<Shape> kDefaultShapes = {
    {32, 32, 32},       {64, 64, 64},       {96, 96, 96},
    {128, 128, 128},    {256, 256, 256},    {512, 512, 512},
    {1024, 1024, 1024}, {2048, 2048, 2048}, {512, 256, 256},
    {1024, 256, 1024},  {256, 1024, 1024},  {1024, 1024, 256},
    {512, 512, 2048},   {127, 193, 65},
};

// 扩展比较：保留原来的 14 组，再覆盖大方阵、短/长 K、细长和尾部尺寸。
const std::vector<Shape> kExtendedShapes = {
    {1, 1, 1},          {32, 32, 32},       {64, 64, 64},
    {96, 96, 96},       {128, 128, 128},    {256, 256, 256},
    {512, 512, 512},    {1000, 1000, 1000}, {1024, 1024, 1024},
    {2048, 2048, 2048}, {4096, 4096, 4096}, {512, 256, 256},
    {1024, 256, 1024},  {256, 1024, 1024},  {1024, 1024, 256},
    {512, 512, 2048},   {256, 256, 4096},   {2048, 2048, 32},
    {2048, 128, 1024},  {128, 2048, 1024},  {4096, 256, 256},
    {256, 4096, 256},   {127, 193, 65},     {513, 257, 129},
};

struct Options {
  int device = 0;
  int warmup = 20;
  int repeat = 200;
  int samples = 7;
  bool list = false;
  std::string csv_path;
  std::string suite = "default";
  std::vector<std::string> kernels;
  std::vector<Shape> shapes;
};

using Launch = void (*)(float*, float*, float*, int, int, int, cudaStream_t);

struct Kernel {
  const char* name;
  const char* description;
  int alignment;
  Launch launch;
  Launch prepare = nullptr;
};

const std::vector<Kernel> kKernels = {
    {"v1", "16x16 tile, 1 output/thread", 1, sgemm_v1::launch},
    {"v2", "16x16 tile, currently same computation as v1", 1,
     sgemm_v2::launch},
    {"v3", "32x32 tile, 2x2 outputs/thread", 32, sgemm_v3::launch},
    {"v4", "32x32 tile, float4 copies, 1x4 outputs/thread", 32,
     sgemm_v4::launch},
    {"v5", "32x32 tile, transposed As, 4x4 outputs/thread", 32,
     sgemm_v5::launch},
    {"v6", "32x32 tile, async copies, two stages, 4x4 outputs/thread", 32,
     sgemm_v6::launch},
    {"v7", "32x32 block, 16x32 warp, 4x4 thread, BK=32, boundary checks", 1,
     sgemm_v7::launch},
    {"warp32", "32x32 block, 16x32 warp, 4x4 thread, BK=32, boundary checks",
     1, sgemm_warp_tiling::launch32},
    {"warp64", "64x64 block, 16x32 warp, 4x4 thread, BK=16, boundary checks",
     1, sgemm_warp_tiling::launch64},
#ifdef SGEMM_WITH_CUTLASS
    {"cutlass32", "CUTLASS FP32 SIMT, 32x32x8 block, 16x32x8 warp, stages=2",
     1, sgemm_cutlass::launch32, sgemm_cutlass::prepare32},
    {"cutlass64", "CUTLASS FP32 SIMT, 64x64x8 block, 16x32x8 warp, stages=2",
     1, sgemm_cutlass::launch64, sgemm_cutlass::prepare64},
    {"cutlass128", "CUTLASS FP32 SIMT, 128x128x8 block, 32x64x8 warp, stages=2",
     1, sgemm_cutlass::launch128, sgemm_cutlass::prepare128},
#endif
};

bool selected(const Options& options, const Kernel& kernel) {
  return options.kernels.empty() ||
         std::find(options.kernels.begin(), options.kernels.end(), kernel.name) !=
             options.kernels.end();
}

bool supports(const Kernel& kernel, const Shape& shape) {
  return shape.m % kernel.alignment == 0 &&
         shape.n % kernel.alignment == 0 &&
         shape.k % kernel.alignment == 0;
}

int parse_int(const std::string& text, bool allow_zero = false) {
  size_t consumed = 0;
  const long long value = std::stoll(text, &consumed);
  if (consumed != text.size() || value < (allow_zero ? 0 : 1) ||
      value > std::numeric_limits<int>::max() - 32) {
    throw std::runtime_error("Invalid integer: " + text);
  }
  return static_cast<int>(value);
}

void validate_shape(const Shape& shape) {
  // 原 kernel 使用 int 索引，先拒绝可能溢出的维度组合。
  const size_t limit = std::numeric_limits<int>::max();
  if (static_cast<size_t>(shape.m) * shape.k > limit ||
      static_cast<size_t>(shape.k) * shape.n > limit ||
      static_cast<size_t>(shape.m) * shape.n > limit) {
    throw std::runtime_error("Matrix element count exceeds kernel int indexing");
  }
}

void print_usage(const char* program) {
  std::printf(
      "Usage: %s [options]\n"
      "  --suite NAME   default (14 shapes) or extended (24 shapes)\n"
      "  --kernels LIST Comma-separated names; default: all compiled kernels\n"
      "  --shape M N K  Test this shape; repeat to test multiple shapes\n"
      "                 A[M,K] * B[K,N] = C[M,N]\n"
      "  --warmup N     Warmup launches per version (default: 20)\n"
      "  --repeat N     Launches per timing sample (default: 200)\n"
      "  --samples N    Timing samples; report median (default: 7)\n"
      "  --device N     CUDA device index (default: 0)\n"
      "  --csv PATH     Save results as CSV\n"
      "  --list         List shapes and supported versions without using GPU\n"
      "  --help         Show this help\n"
      "Without --shape, run the built-in square/rectangular/boundary suite.\n",
      program);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto next = [&]() -> std::string {
      if (++i == argc) {
        throw std::runtime_error("Missing value for " + argument);
      }
      return argv[i];
    };
    if (argument == "--suite") {
      options.suite = next();
      if (options.suite != "default" && options.suite != "extended") {
        throw std::runtime_error("Unknown suite: " + options.suite);
      }
    } else if (argument == "--kernels") {
      const std::string names = next();
      size_t begin = 0;
      while (true) {
        const size_t end = names.find(',', begin);
        const std::string name = names.substr(begin, end - begin);
        if (name.empty()) throw std::runtime_error("Empty kernel name");
        if (std::find(options.kernels.begin(), options.kernels.end(), name) !=
            options.kernels.end()) {
          throw std::runtime_error("Duplicate kernel: " + name);
        }
        options.kernels.push_back(name);
        if (end == std::string::npos) break;
        begin = end + 1;
      }
    } else if (argument == "--shape") {
      const int m = parse_int(next());
      const int n = parse_int(next());
      const int k = parse_int(next());
      options.shapes.push_back({m, n, k});
    } else if (argument == "--warmup") {
      options.warmup = parse_int(next(), true);
    } else if (argument == "--repeat") {
      options.repeat = parse_int(next());
    } else if (argument == "--samples") {
      options.samples = parse_int(next());
    } else if (argument == "--device") {
      options.device = parse_int(next(), true);
    } else if (argument == "--csv") {
      options.csv_path = next();
      if (options.csv_path.empty()) {
        throw std::runtime_error("CSV path must not be empty");
      }
    } else if (argument == "--list") {
      options.list = true;
    } else {
      throw std::runtime_error("Unknown option: " + argument);
    }
  }
  if (options.shapes.empty()) {
    options.shapes = options.suite == "extended" ? kExtendedShapes : kDefaultShapes;
  }
  for (const std::string& name : options.kernels) {
    const auto found = std::find_if(kKernels.begin(), kKernels.end(),
                                   [&](const Kernel& kernel) {
                                     return name == kernel.name;
                                   });
    if (found == kKernels.end()) {
      throw std::runtime_error("Unknown/uncompiled kernel: " + name);
    }
  }
  for (const Shape& shape : options.shapes) validate_shape(shape);
  return options;
}

void print_shapes(const Options& options) {
  for (const Kernel& kernel : kKernels) {
    if (!selected(options, kernel)) continue;
    std::printf("%s: %s\n", kernel.name, kernel.description);
  }
  std::printf("\n%7s %7s %7s  Supported versions\n", "M", "N", "K");
  for (const Shape& shape : options.shapes) {
    std::printf("%7d %7d %7d ", shape.m, shape.n, shape.k);
    for (const Kernel& kernel : kKernels) {
      if (selected(options, kernel) && supports(kernel, shape)) {
        std::printf(" %s", kernel.name);
      }
    }
    std::printf("\n");
  }
}

constexpr float kAtol = 1e-3f;
constexpr float kRtol = 1e-4f;

struct Validation {
  size_t mismatches = 0;
  size_t first_mismatch = 0;
  float max_abs_error = 0.0f;
};

Validation compare(const std::vector<float>& reference,
                   const std::vector<float>& output) {
  Validation validation;
  for (size_t i = 0; i < reference.size(); ++i) {
    const bool finite = std::isfinite(output[i]) && std::isfinite(reference[i]);
    const float error = finite ? std::fabs(reference[i] - output[i])
                               : std::numeric_limits<float>::infinity();
    validation.max_abs_error = std::max(validation.max_abs_error, error);
    if (!finite || error > kAtol + kRtol * std::fabs(reference[i])) {
      if (validation.mismatches == 0) validation.first_mismatch = i;
      ++validation.mismatches;
    }
  }
  return validation;
}

void check_reference_on_cpu(const Shape& shape, const std::vector<float>& a,
                            const std::vector<float>& b,
                            const std::vector<float>& reference) {
  // 用 double 点积抽查角点和内部点，检查 cuBLAS 的行列主序转换。
  const std::array<std::pair<int, int>, 9> positions = {{
      {0, 0}, {0, shape.n - 1}, {shape.m - 1, 0},
      {shape.m - 1, shape.n - 1}, {shape.m / 2, shape.n / 2},
      {shape.m / 3, shape.n / 5}, {shape.m / 5, shape.n / 3},
      {shape.m / 7, shape.n / 2}, {shape.m / 2, shape.n / 7},
  }};
  for (const auto& position : positions) {
    const int row = position.first, col = position.second;
    double expected = 0.0;
    for (int p = 0; p < shape.k; ++p) {
      expected += static_cast<double>(a[static_cast<size_t>(row) * shape.k + p]) *
                  b[static_cast<size_t>(p) * shape.n + col];
    }
    const float actual = reference[static_cast<size_t>(row) * shape.n + col];
    if (!std::isfinite(actual) ||
        std::fabs(expected - actual) > kAtol + kRtol * std::fabs(expected)) {
      throw std::runtime_error("cuBLAS reference failed CPU double spot check");
    }
  }
}

float measure(const Kernel& kernel, const Shape& shape, const Options& options,
              float* a, float* b, float* c, cudaStream_t stream) {
  auto launch = [&]() {
    kernel.launch(a, b, c, shape.m, shape.n, shape.k, stream);
  };
  for (int i = 0; i < options.warmup; ++i) launch();
  check_cuda(cudaGetLastError(), "warmup launch");
  check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

  Event start, stop;
  std::vector<float> measurements;
  measurements.reserve(options.samples);
  for (int sample = 0; sample < options.samples; ++sample) {
    // 同一 stream 内的 Event 只围住 kernel；拷贝、校验、分配都在计时外。
    check_cuda(cudaEventRecord(start.value, stream), "start event");
    for (int i = 0; i < options.repeat; ++i) launch();
    check_cuda(cudaEventRecord(stop.value, stream), "stop event");
    check_cuda(cudaGetLastError(), "timed launch");
    check_cuda(cudaEventSynchronize(stop.value), "timed synchronize");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start.value, stop.value),
               "cudaEventElapsedTime");
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
      throw std::runtime_error("Invalid timing; increase --repeat");
    }
    measurements.push_back(elapsed_ms / options.repeat);
  }
  std::sort(measurements.begin(), measurements.end());
  const size_t middle = measurements.size() / 2;
  return measurements.size() % 2 != 0
             ? measurements[middle]
             : (measurements[middle - 1] + measurements[middle]) / 2.0f;
}

struct Result {
  const char* status = "SKIP";
  float ms = 0.0f;
  Validation validation;
};

bool benchmark_shape(const Shape& shape, const Options& options,
                     cudaStream_t stream, cublasHandle_t blas,
                     std::ofstream& csv) {
  std::printf("\nM=%d N=%d K=%d  A[%d,%d] * B[%d,%d]\n", shape.m,
              shape.n, shape.k, shape.m, shape.k, shape.k, shape.n);
  std::fflush(stdout);
  const size_t a_count = static_cast<size_t>(shape.m) * shape.k;
  const size_t b_count = static_cast<size_t>(shape.k) * shape.n;
  const size_t c_count = static_cast<size_t>(shape.m) * shape.n;
  const size_t c_bytes = c_count * sizeof(float);
  std::vector<float> a(a_count), b(b_count), reference(c_count), output(c_count);
  std::mt19937 generator(2026);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (float& value : a) value = distribution(generator);
  for (float& value : b) value = distribution(generator);

  DeviceBuffer device_a(a_count), device_b(b_count), device_c(c_count),
      device_reference(c_count);
  check_cuda(cudaMemcpyAsync(device_a.data, a.data(), a_count * sizeof(float),
                             cudaMemcpyHostToDevice, stream), "copy A");
  check_cuda(cudaMemcpyAsync(device_b.data, b.data(), b_count * sizeof(float),
                             cudaMemcpyHostToDevice, stream), "copy B");

  // cuBLAS 按列主序解释内存：C^T[N,M] = B^T[N,K] * A^T[K,M]。
  const float alpha = 1.0f, beta = 0.0f;
  check_blas(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, shape.n, shape.m,
                         shape.k, &alpha, device_b.data, shape.n, device_a.data,
                         shape.k, &beta, device_reference.data, shape.n),
             "cublasSgemm reference");
  check_cuda(cudaMemcpyAsync(reference.data(), device_reference.data, c_bytes,
                             cudaMemcpyDeviceToHost, stream), "copy reference");
  check_cuda(cudaStreamSynchronize(stream), "reference synchronize");
  check_reference_on_cpu(shape, a, b, reference);

  std::printf("%-12s %-7s %12s %12s %12s %14s\n", "Version", "Status",
              "Time(ms)", "GFLOPS", "Speedup/v1", "Max abs error");
  const double flops = 2.0 * shape.m * shape.n * shape.k;
  bool correct = true;
  float baseline_ms = 0.0f;
  for (size_t index = 0; index < kKernels.size(); ++index) {
    const Kernel& kernel = kKernels[index];
    if (!selected(options, kernel)) continue;
    Result result;
    if (supports(kernel, shape)) {
      try {
        // 库的参数初始化在校验和计时之前完成，计时中只执行已准备的 GEMM。
        if (kernel.prepare) {
          kernel.prepare(device_a.data, device_b.data, device_c.data, shape.m,
                         shape.n, shape.k, stream);
        }
        // 0xff 对 float 是 NaN：未写回的 C 元素不能靠旧结果蒙混过关。
        check_cuda(cudaMemsetAsync(device_c.data, 0xff, c_bytes, stream),
                   "poison output");
        kernel.launch(device_a.data, device_b.data, device_c.data, shape.m,
                      shape.n, shape.k, stream);
        check_cuda(cudaGetLastError(), "validation launch");
        check_cuda(cudaMemcpyAsync(output.data(), device_c.data, c_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "copy validation result");
        check_cuda(cudaStreamSynchronize(stream), "validation synchronize");
        result.validation = compare(reference, output);
        if (result.validation.mismatches == 0) {
          result.status = "PASS";
          result.ms = measure(kernel, shape, options, device_a.data,
                              device_b.data, device_c.data, stream);
          if (index == 0) baseline_ms = result.ms;
        } else {
          result.status = "FAIL";
          correct = false;
        }
      } catch (const std::exception& error) {
        throw std::runtime_error(std::string(kernel.name) + " on " +
                                 std::to_string(shape.m) + "x" +
                                 std::to_string(shape.n) + "x" +
                                 std::to_string(shape.k) + ": " + error.what());
      }
    }
    const double gflops = result.ms > 0 ? flops / (result.ms * 1e6) : 0.0;
    const double speedup = result.ms > 0 ? baseline_ms / result.ms : 0.0;
    if (result.ms > 0) {
      char speedup_text[32] = "-";
      if (baseline_ms > 0) {
        std::snprintf(speedup_text, sizeof(speedup_text), "%.2fx", speedup);
      }
      std::printf("%-12s %-7s %12.6f %12.2f %12s %14.3e\n", kernel.name,
                  result.status, result.ms, gflops, speedup_text,
                  result.validation.max_abs_error);
    } else if (result.validation.mismatches > 0) {
      const size_t first = result.validation.first_mismatch;
      std::printf("%-12s %-7s %12s %12s %12s %14.3e\n", kernel.name,
                  result.status, "-", "-", "-", result.validation.max_abs_error);
      std::printf("  %zu mismatches; first C[%zu,%zu]: ref=%.8g got=%.8g\n",
                  result.validation.mismatches, first / shape.n, first % shape.n,
                  reference[first], output[first]);
    } else {
      std::printf("%-12s %-7s M, N, K must all be multiples of %d\n",
                  kernel.name, result.status, kernel.alignment);
    }
    std::fflush(stdout);

    if (csv.is_open()) {
      csv << shape.m << ',' << shape.n << ',' << shape.k << ',' << kernel.name
          << ',' << result.status << ',';
      if (result.ms > 0) csv << result.ms;
      csv << ',';
      if (result.ms > 0) csv << gflops;
      csv << ',';
      if (result.ms > 0 && baseline_ms > 0) csv << speedup;
      csv << ',';
      if (supports(kernel, shape)) csv << result.validation.max_abs_error;
      csv << ',' << result.validation.mismatches << ',' << options.repeat
          << ',' << options.samples << '\n';
      if (!csv) throw std::runtime_error("Failed to write CSV results");
    }
  }
  return correct;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--help") {
        print_usage(argv[0]);
        return 0;
      }
    }
    const Options options = parse_options(argc, argv);
    if (options.list) {
      print_shapes(options);
      return 0;
    }
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (options.device >= device_count) {
      throw std::runtime_error("CUDA device index unavailable");
    }
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, options.device),
               "cudaGetDeviceProperties");
    int driver_version = 0, runtime_version = 0;
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    std::printf("GPU: %s (sm_%d%d), memory %.1f GiB\n", properties.name,
                properties.major, properties.minor,
                properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("CUDA driver/runtime: %d/%d\n", driver_version, runtime_version);
    std::printf("Warmup=%d repeat=%d samples=%d; median time per launch\n",
                options.warmup, options.repeat, options.samples);
    std::printf("Reference: cuBLAS pedantic FP32 + CPU double spot checks; "
                "atol=%.0e rtol=%.0e; seed=2026\n", kAtol, kRtol);
    std::printf("Timing excludes initialization, allocation, transfers and validation.\n");
#ifdef SGEMM_WITH_CUTLASS
    std::printf("CUTLASS %s: FP32 SIMT CUDA-core FMA, alpha=1 beta=0; "
                "Tensor Cores/TF32 disabled by operator selection.\n",
                sgemm_cutlass::version());
#endif
#ifdef __CUDACC_DEBUG__
    std::fprintf(stderr, "WARNING: CUDA device debugging is enabled; rebuild "
                         "with CMAKE_BUILD_TYPE=Release for performance.\n");
#endif
    if (properties.major < 8) {
      std::printf("Note: v6/warp32/warp64 async copies use a fallback on "
                  "GPUs older than sm_80.\n");
    }

    std::ofstream csv;
    if (!options.csv_path.empty()) {
      csv.open(options.csv_path);
      if (!csv) throw std::runtime_error("Cannot open CSV: " + options.csv_path);
      csv << "m,n,k,version,status,time_ms,gflops,speedup_v1,max_abs_error,"
             "mismatches,repeat,samples\n";
      csv << std::setprecision(9);
    }
    Stream stream;
    BlasHandle blas;
    check_blas(cublasSetStream(blas.value, stream.value), "cublasSetStream");
    // 禁止把 FP32 输入降为 TF32，避免参考值自身引入额外精度差异。
    check_blas(cublasSetMathMode(blas.value, CUBLAS_PEDANTIC_MATH),
               "cublasSetMathMode");
    bool correct = true;
    for (const Shape& shape : options.shapes) {
      if (!benchmark_shape(shape, options, stream.value, blas.value, csv)) {
        correct = false;
      }
    }
    if (csv.is_open()) {
      csv.flush();
      if (!csv) throw std::runtime_error("Failed to flush CSV results");
      std::printf("\nCSV: %s\n", options.csv_path.c_str());
    }
    return correct ? 0 : 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Benchmark failed: %s\n", error.what());
    return 1;
  }
}
