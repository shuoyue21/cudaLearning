| 变量 | 所属坐标系 | 含义 |
|---|---|---|
| `block_row/block_col` | 全局矩阵 | 当前 block 的 C tile 左上角 |
| `k_base` | 全局 K 维 | 当前 K tile 的起点 |
| `local_row/local_col` | 当前 A/B tile | 某个搬运任务在 tile 内的位置 |
| `warp_row/warp_col` | 当前 block tile | 当前 warp tile 的左上角 |
| `thread_row_base/thread_col_base` | 当前 block tile | 当前线程输出区域的位置 |
| `global_row/global_col` | 全局矩阵 | 最终真实读写坐标 |


执行结构:

确定当前 block 对应的 BM×BN 逻辑输出块

把第 0 个 K tile 预取到 shared stage 0

for 每一个 K tile:

    如果还有下一个 K tile:
        异步搬到另一套 shared buffer

    等当前 K tile 搬完

    所有线程同步

    每个 warp 计算自己的 WM×WN

    当前 shared stage 使用完毕

    所有线程同步

只把合法的 C 元素写回