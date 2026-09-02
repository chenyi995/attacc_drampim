# 归档：2026-08-30 sweep，**修复前调度器**的产出

这三个 CSV 是 `sweep_models_20260830-163226` 的完整提取结果，**2026-09-02 13:46
生成，705 个档、75/78 个任务九档齐全**。

| 文件 | 行数 | sha256（前 16） |
|---|---:|---|
| `sweep_rungs.csv` | 705 | `409e1797444c500f` |
| `sweep_tiers.csv` | 1674 | `da23c5f24ec7967c` |
| `sweep_completeness.csv` | 84 | `188d9f8ce77824c7` |

## ⚠️ 目录名里的 `prefix_scheduler` 是什么意思

**这批数据是在 `75da860` 之前的引擎上跑的，那个提交修掉了一个调度缺陷：**

> 没有 `--pipeopt` 时，调度器把每个事件挂在同一个串行资源上，于是一次 KV 扫描的
> 十六条通道 lane 被**求和**而不是**重叠**。硬件上这些 lane 是并发的，扫描耗时
> 应当取**最忙的那条通道**。

修复后的实测对比（LLAMA-7B，同一格）：

| | 修复前 | 修复后 |
|---|---:|---:|
| baseline A3 makespan | 141.6 s | **19.5 s** |
| baseline A6 makespan | 100.1 s | **15.3 s** |

**偏差是 3.2–7.6 倍，而且不均匀。**

### 哪些列作废，哪些列仍然有效

| 列 | 状态 |
|---|---|
| `makespan_s`、`duration_s`、`prefill_s`、`decode_s`、`prefill_time_s`、`decode_time_s`、三个 `power_*_w` | **作废** —— 全部依赖 makespan |
| `energy_nj` 及各 `energy_*_nj` | **有效** |
| `link_bytes` | **有效** |
| `event_count` | **有效** |

能量、链路字节、事件数这三项在修复前后**逐格不变**（`75da860` 与 `ce0cde6` 均已
逐格验证），所以基于它们的结论不受影响。

## 另外两个已知问题（与调度无关）

1. **A3b 在 `--num-hbm 1` 上不是 A3b。** 切片把一个 head 摊到
   `max(1, 16 // heads_per_hbm)` 个通道；LLAMA-7B 是 32 个 KV head 挤一个堆栈，
   stripe 钳到 1，载荷向量与 A3 逐位相同。**本目录里 LLAMA-7B 的全部
   A3b 行等同于 A3**，不是独立测量。见 `ce0cde6`。
2. **`decode_time_s` 有 84 行为空，正好是全部 A2 行。** A2 的 decode 走 GPU 路径，
   tier 结构与其余各档不同。`sweep_tiers.csv` 里 A2 的 `decode_s` 仍有值。

## 保留它的理由

这是**修复前引擎实际产出了什么**的完整记录，一个数都没有改动。重跑新引擎之后，
它是唯一能用来对账"修复改变了什么、没改变什么"的基线。

重新生成（在新引擎上会得到不同的 makespan）：

```bash
bash output/analysis/extract_sweep.sh
```
