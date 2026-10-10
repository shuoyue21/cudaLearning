#!/usr/bin/env python3
"""Compare the tested FP32 Warp tiling and CUTLASS configurations."""

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import geometric_mean

WARP = ("warp32", "warp64")
CUTLASS = ("cutlass32", "cutlass64", "cutlass128")
VERSIONS = WARP + CUTLASS


def read_results(path):
    grouped = {}
    timing_settings = set()
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            version = row["version"]
            if version not in VERSIONS:
                continue
            shape = tuple(int(row[name]) for name in ("m", "n", "k"))
            if row["status"] != "PASS" or int(row["mismatches"]) != 0:
                raise ValueError(f"Result did not pass: {shape} {version}")
            latency = float(row["time_ms"]) * 1000.0
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"Invalid latency: {shape} {version}")
            versions = grouped.setdefault(shape, {})
            if version in versions:
                raise ValueError(f"Duplicate result: {shape} {version}")
            versions[version] = latency
            timing_settings.add((int(row["repeat"]), int(row["samples"])))
    if not grouped or len(timing_settings) != 1:
        raise ValueError("No results or mixed repeat/sample settings")
    comparisons = []
    for shape, versions in grouped.items():
        missing = set(VERSIONS) - versions.keys()
        if missing:
            raise ValueError(f"Incomplete comparison for {shape}: missing {missing}")
        warp = min(WARP, key=versions.get)
        cutlass = min(CUTLASS, key=versions.get)
        comparisons.append({
            "shape": shape, "times": versions, "warp": warp,
            "cutlass": cutlass, "speedup": versions[warp] / versions[cutlass],
        })
    return comparisons, next(iter(timing_settings))


def plot_results(comparisons, prefix):
    # Keep matplotlib's cache inside the project instead of changing HOME.
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    indices = list(range(len(comparisons)))
    labels = ["x".join(map(str, row["shape"])) for row in comparisons]
    warp = [row["times"][row["warp"]] for row in comparisons]
    cutlass = [row["times"][row["cutlass"]] for row in comparisons]
    ratios = [row["speedup"] for row in comparisons]
    figure, (latency_ax, speedup_ax) = plt.subplots(
        2, 1, figsize=(17, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.2]}, layout="constrained")
    latency_ax.bar([x - 0.18 for x in indices], warp, 0.36,
                   label="Fastest tested Warp config", color="#386cb0")
    latency_ax.bar([x + 0.18 for x in indices], cutlass, 0.36,
                   label="Fastest tested CUTLASS config", color="#f28e2b")
    latency_ax.set_yscale("log")
    latency_ax.set_ylabel("Latency (microseconds, log scale)")
    latency_ax.set_title("Warp tiling vs CUTLASS: FP32 SIMT; "
                         "Warp tiles: 32/64; CUTLASS tiles: 32/64/128")
    latency_ax.legend()
    latency_ax.grid(axis="y", alpha=0.2)
    latency_ax.set_axisbelow(True)
    colors = ["#228b68" if value >= 1 else "#c75a5a" for value in ratios]
    speedup_ax.bar(indices, ratios, color=colors)
    speedup_ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
    speedup_ax.set_ylabel("Speedup = Warp / CUTLASS")
    speedup_ax.set_ylim(0, max(1.2, max(ratios) * 1.12))
    speedup_ax.set_xticks(indices, labels, rotation=60, ha="right", fontsize=8)
    speedup_ax.set_xlabel("Matrix shape: M x N x K; >1 means CUTLASS is faster")
    speedup_ax.grid(axis="y", alpha=0.2)
    speedup_ax.set_axisbelow(True)
    for suffix in (".png", ".svg"):
        figure.savefig(prefix.with_suffix(suffix), dpi=180)
    plt.close(figure)


def write_report(comparisons, settings, source, prefix, plots):
    wins = sum(row["speedup"] > 1 for row in comparisons)
    gm = geometric_mean(row["speedup"] for row in comparisons)
    lines = ["# Warp tiling 与 CUTLASS FP32 SIMT 对比", "",
             f"数据文件：`{source.name}`。共 {len(comparisons)} 组尺寸，每组比较 5 个配置，"
             f"每个计时样本运行 {settings[0]} 次，取 {settings[1]} 组单次平均耗时的中位数。", "",
             "两类实现都使用 FP32 输入、FP32 输出和 FP32 累加；CUTLASS 使用 CUDA Core SIMT FMA。"
             "初始化、分配、传输和校验在计时之外。", "",
             "这里的最快值仅指选测配置：Warp 的 warp32/warp64 二选一，"
             "CUTLASS 的 cutlass32/cutlass64/cutlass128 三选一；没有穷举所有配置或使用 Tensor Core。", "",
             f"本轮 CUTLASS 的选测最快配置在 {wins}/{len(comparisons)} 组中耗时更低。"
             f"逐尺寸加速比的几何平均为 {gm:.3f}×，各尺寸等权，不代表某个业务负载的总加速比。", "",
             "运行间隔和 GPU 状态会影响测量；接近 1× 的差异应通过重复测试判断。", ""]
    log = source.with_suffix(".log")
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith(("GPU:", "CUDA driver/runtime:", "Warmup=", "CUTLASS ")):
                lines.append(f"- `{line}`")
        lines.append("")
    lines += ["| M × N × K | warp32 μs | warp64 μs | cutlass32 μs | cutlass64 μs | cutlass128 μs | Warp 最快 | CUTLASS 最快 | CUTLASS 加速比 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |"]
    for row in comparisons:
        shape = " × ".join(map(str, row["shape"]))
        values = " | ".join(f"{row['times'][name]:.3f}" for name in VERSIONS)
        lines.append(f"| {shape} | {values} | {row['warp']} | {row['cutlass']} | {row['speedup']:.3f}× |")
    lines += ["", "加速比 = 同尺寸 Warp 选测最快耗时 / CUTLASS 选测最快耗时；大于 1 表示 CUTLASS 更快。", ""]
    if plots:
        lines += [f"![FP32 SIMT comparison]({prefix.with_suffix('.png').name})", ""]
    prefix.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("sgemm_cutlass_comparison"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    comparisons, settings = read_results(args.csv)
    # 先输出标准库报告，即使可选绘图库不可用也保留完整的数值对比。
    write_report(comparisons, settings, args.csv, args.output_prefix, False)
    wins = sum(row["speedup"] > 1 for row in comparisons)
    print(f"Verified {len(comparisons)} shapes; CUTLASS candidate wins {wins}/{len(comparisons)}.")
    print(f"Report: {args.output_prefix.with_suffix('.md')}")
    if not args.no_plot:
        plot_results(comparisons, args.output_prefix)
        write_report(comparisons, settings, args.csv, args.output_prefix, True)
        print(f"Plot: {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
