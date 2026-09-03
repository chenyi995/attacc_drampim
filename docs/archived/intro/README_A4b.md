# A4b：+ 全局 co-read placement table

> **已归档 2026-09-03。** 两处已经不对：
> (1)「代码」一节写 A4b 走 `_layout_channel_loads` 的 `master-diff-table` 分支 ——
> 现在的 policy 是 `master-diff-table-append`，走 `_striped_append_channel_extents`。
> (2)「结果预期」里说「真实 `num_hbm=16` + LLAMA3-8B(8 KV head) 时 heads_per_hbm=1、
> A4 ≈ A4b」—— sweep 实际用的是 `--num-hbm 1/1/10/10/40/40`，且 heads-per-HBM 的
> 口径已于 `d3a3c4c` 修正（堆栈也按 GPU 切），LLAMA3-8B 是 8 而不是 1。
> 「全局 co-read table 取代固定切片」这个机制描述本身仍然成立。
> **当前口径见 `../README_data_layout_walkthrough.md` §6 与 `../sessions/2026-09-03.md` §8、§11。**


（阶梯定位见 `../README.md` §3。2026-08-29 加。A5/A6 的布局建在 A4b 上。）

## 比上一档（A4）多做的一件事

**A4** = A3b（head 切片）+ **master/diff 分离**：16 条 channel 里 master 池 ch0–14、
diff 池 ch15；master 仍按**固定 head 切片**（每个 head 占 `15/heads_per_hbm` 条），
重算修正行进 diff channel。固定切片的问题：若 head 数不整除、或各 head 负载不均，切片
会**浪费/不均**（某条 channel 排队，另一条闲）。

**A4b** 用**全局 placement table** 取代固定切片：master 池的 15 条 channel 上，把
**所有 head 的 master chunk 一起排**，让**会被一起读到的 chunk（跨 head 或同 head）落到
不同 channel** → 全局平衡。修正行仍进 diff channel（master/diff 分离保留）。

## 配置（`src/ablation.py` PRESETS）

`kv_mapping = master-diff`，`channel_placement = table`。批命令 replicate，prefill GPU。

## 代码

`_layout_channel_loads` 的 `master-diff-table` 分支：master chunk 全局轮转 15 条
master channel（co-read 分散），diff 落 ch15；`_append_placement_pim_scan` 逐 channel
真实 Ramulator 计价、取 max。

## 结果预期

相对 A4，当 `heads_per_hbm > 1`（head 挤一个 HBM、固定切片会不均）时,全局 table 把
master 负载摊得更平 → 最忙 channel 更低 → 扫描更快。**`heads_per_hbm = 1`（每个 head
独占一个 HBM 的全部 15 条 master channel）时,固定切片与全局 table 重合,A4 ≈ A4b**——
这是真实 `num_hbm=16` + LLAMA3-8B(8 KV head)的情形;要看出区别需 head 挤 HBM
(小 num_hbm 或多 head 模型,见 `README_run_sweep_guide.md` §2 的 tier 表)。
