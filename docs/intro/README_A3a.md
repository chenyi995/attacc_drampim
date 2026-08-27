# A3a:naive 乱序布局 + 可掩(2026-08-26 增档)

**一句话**:与 A3 完全相同的页化软件布局(256-token 页按 append 序轮换
16 channel、无 master/diff 之分),唯一差别是**消费者能掩 (mask)**——
陈旧的重算行随 master 流读出后被掩出 score,run **不断开**;A3 则无掩,
必须跳过该行,run 在缺口处劈开("act 一段、act 一个 token、再 act
一段")。两档相减 = **断流代价**;A3a 与 A4 相减 = **乱序布局本身的
代价**——问题②由此分解成两根可读的柱子。

## 代码定位

| 步骤 | 位置 |
|---|---|
| 预设 | `src/ablation.py::PRESETS["A3a"]`(prefill gpu / decode pim / kv `naive-mask` / replicate) |
| 布局 | `src/workload_runner.py::NaiveMaskKVLayout`(继承页化 `NaiveKVLayout`,仅 `shadow_reads = True`) |
| 掩/断流开关 | `CacheBlendTLB.shadow_reads` + `_pool_reads`(decode)与 prefill pim 分支的 old_reads 装配 |
| 运行 | `run_dag_ladder.sh` 七档并行之一;重算行由 `--reuse recompute` 随机抽取(位置分布只影响 A3,不影响本档) |

## 语义出处

裁决(chenyi9 2026-08-26):"如果 GPU 能 mask 那就当作 A3a,目前的这个
还是叫做 A3"。读掩本体是 Fugue 的 die 侧硬件特性(D_i 位图 + score
掩);A3a 把"可掩"能力单独授予 naive 布局,用来隔离断流与乱序两层
效应。台账条目:R15。
