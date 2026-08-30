# A3b：+ head 切片（head slicing）

（阶梯定位见 `../README.md` §3；物理模型见那里的"一个 head 的一个 chunk = 一条
channel 的一个 row"。2026-08-29 加。）

## 比上一档（A3）多做的一件事

**A3** 不切片：一个 head 的**全部 chunk 压在它自己那一条 channel 上**（head h →
channel `h % 16`），其余 channel 闲着——一条 channel 扛这个 head 的整段 context，最慢。

**A3b** 加 **head 切片**：每个 head 分到 `stripe = 16 / heads_per_hbm` 条 channel
（它在这个 HBM 里的切片），该 head 的 chunk 在**自己的切片内轮转**，于是**同一个
head 的不同 chunk 落到不同 channel**、并行。

- `heads_per_hbm = ceil(局部 KV head / num_hbm)`；1 head/HBM → stripe=16；
  8 head/HBM → stripe=2；32 head/HBM → stripe≈1（挤，2 head/channel）。
- 还没有 master/diff 分离（修正行和普通行混在一起，这是 A4 才加的）。

## 配置（`src/ablation.py` PRESETS）

`kv_mapping = naive`，`channel_placement = slice`。批命令 replicate，prefill 在 GPU。

## 代码

`workload_runner._layout_channel_loads` 的 `slice` 分支产出各 channel 负载；
`_append_placement_pim_scan` 给每条活跃 channel 造真实 Ramulator run、跨 channel 取 max。

## 结果预期

相对 A3，一个 head 的 context 从"压一条 channel"摊到 `stripe` 条 → decode 扫描的最忙
channel 负载降到约 `1/stripe` → 扫描时间大降。`heads_per_hbm` 越小（HBM 越多）stripe
越大、越明显；`heads_per_hbm=1` 时 stripe=16，最明显。
