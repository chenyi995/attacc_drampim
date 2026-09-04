# 归档:2026-09-03 之前的 sweep workload(C=32、sys=16)

**归档时间**:2026-09-03
**为什么**:chenyi9 当天两条裁决改了 workload 的形状。

1. **sys:16 token → 256 token(整块)**。一个 1024 B 的 DRAM row 恰好装 256 个
   token,16-token 的 sys 让每条上下文都差一点点、凑不成整数个 row,放置各档的
   行数算术于是被这个碎片左右,而不是被要测的机制左右(旧 baseline 上下文是
   `8464 = 33×256 + 16`)。改完是 `18×256 = 4608`,不带零头。
2. **复用块数 C:32 → 16**(轴从 16/32/64 改成 **8/16/40**)。每个复用块重算
   k=8 个 token,恰好是**一个 col**(32 B / 8 token),所以一个 head 的修正就是
   C 个 col。C=16 时四个一起生成的 head 合起来 `4×16×8 = 512 token = 恰好 2 行`
   打包在 diff 通道上,而 A3b 让每次修正各占一行 —— **这个对比就是 master/diff
   分离的收益**,C=16 比 C=32 更大。C=32 另有个副作用:32 个块在 16 条 channel 上
   正好铺满两遍,任何放置策略都完美均衡,档与档反而分不开。

**被什么取代**:`workload/sweep/` 里的新一批 ——
`wl_baseline_alltoall_N16_C16_D2.json`、`wl_C8.json`、`wl_C40.json`,
其余文件名不变但按 C=16 / sys=256 重生成。生成器 `workload/gen_sweep.py` 的默认值
也已改成 `SYS = BLOCK`、`--C 16`。

**这批还有什么用**:2026-09-03 之前的所有结果
(`output/sweep_baseline_20260902_postfix/`、`output/sweep_postfix_20260903_rungs5/`、
`output/analysis/` 下的 CSV、`docs/archived/README_*results*.md`)都是在这批
workload 上跑的。要复现或对照那些数,用这里的文件。
