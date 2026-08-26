# 计划:多 Q 单命令(MQ-MAC)批处理改造 — 大纲 + 详细计划

chenyi9 2026-08-21 定的命令语义(以下称 **MQ 模式**,multi-query MAC):

1. `ACT_AB` 不变:仍是把一个 row 拉进 row buffer(感放)。
2. **一条 `MAC_AB` = GEMV buffer 里的全部 n 条 Q 轮流与这个 col 里的 K 片段做乘加**
   (读一次 col,PE 内部做 n 次 op)。
3. 乘法器与加法树全流水 → 每拍一次 op,PE 频率上推;吞吐由频率定。
4. **n 的上限 = min(最高 PE 频率允许的条数, GEMV buffer 面积允许的条数)**。
5. 在确定的功耗约束(AttAcc 的 IDD7 读环预算)下,核算实际能插几条。

适用两处:prefill attention 的多 Q,与 decode 阶段多 agent 复用同一 chunk。
目标:PE 100% 利用率 + 最少 ACT 次数,然后与 dense(每 agent 一份私有 KV 各扫一遍)
比 latency。

---

## 一、大纲

| # | 步骤 | 产出 |
|---|---|---|
| O1 | trace 生成器加 MQ 模式:MAC_AB 不复制,WRGB/MVSB/SFM/MVGB 仍 ×n | `gen_trace_attacc_bank.py --mq` |
| O2 | 包装器:按功耗与 PE 频率计算有效 MAC 间隔,经 YAML `nCCDAB` 覆盖注入 Ramulator;签名缓存区分模式 | `ramulator_wrapper.py` |
| O3 | 上限逻辑:n_cap = min(buffer 面积档, 用户 batch);超界拆 sweep | `ablation.py` / `workload_runner.py` |
| O4 | CLI:`--pim-batch-command {mq,replicate}`(main.py 默认 mq)、`--pe-freq-ghz`、`--gemv-buffer-bytes` | `main.py` |
| O5 | 微基准验证:间隔公式 vs 实测 cycles/row;ACT/row = 1 | `experiments/mq_command/` |
| O6 | 对比实验:decode 多 agent 共享 chunk、prefill 多 Q,MQ vs 复制命令 vs dense;PE util、ACT 数、latency、能耗 | 结果表 + README |
| O7 | 回归:27 个既有单测不破(内部默认 replicate 不动,mq 走旗标) | 测试通过记录 |

## 二、详细计划(每个选择附理由;理由以 Fugue 正文与 AttAcc 原文为准)

### 2.1 命令语义与时序模型

**MQ 模式的每条 `MAC_AB`**:读一次 col(32 B/bank),PE 对 GEMV buffer 里 n 条 Q
各做一次 16 路 FP16 乘加。相邻 MAC_AB 的最小间隔取三个约束的最大值(整数周期):

```
interval(n) = max( ceil(nCCDAB_pc × (E_col + n·E_q)/(E_col + E_q)),   # 功耗
                   ceil(n / (f_PE × tCK)),                            # PE 吞吐
                   nCCDAB_npc )                                       # DRAM 数据通路
其中(全部沿用仓库/AttAcc 既有口径):
  nCCDAB_pc = 6, nCCDAB_npc = 4(HBM3_5.2Gbps preset)
  tCK = 0.769 ns(1.3 GHz 命令时钟,包装器换算口径)
  E_col = (0.11+0.44) pJ/bit × 256 bit = 140.8 pJ   # 读 32 B 到 PE(FGDRAM 表)
  E_q   = 16×0.32 + 32×0.0034 ≈ 5.23 pJ             # 一次 16 路 MAC + buffer 读
无功耗约束(NPC)运行时去掉第一项。
```

理由:
- 功耗项:AttAcc §4.2 "We calculate the HBM power budget using the loop pattern of
  all-bank interleave read current (IDD7)" — 预算固定,MQ 每加一条 Q 只加 PE/SRAM
  能量(≈3.6%/命令),按能量比例拉伸命令间隔即维持同一预算。能量数字用仓库
  `config.py` 的 ENERGY_TABLE(来源 FGDRAM/MICRO'17),与现有能耗模型同源。
- PE 项:AttAcc §7.1 "GEMV unit and accumulator operate at 666 MHz considering
  tCCDS (1.5 ns)" — PE 每拍一 op 的流水假设是 AttAcc 自己的;频率作参数
  (默认 0.666 GHz = 原综合点,可扫)。
- 实现为 YAML 逐项时序覆盖(Ramulator2 原生支持,`HBM3-PIM.cpp` "Overwrite timing
  parameters with any user-provided value"),不改 C++、不加新命令操作码。
  理由:Fugue 正文 §5.1 "the DRAM command set is unchanged" — n 由配置携带,
  AttAcc §5.1 的 `PIM_SET_CONFIG` 本就写入 "batch size, and L of each request",
  MQ 恰好落在这个既有接口里。

**trace 形状(O1)**:现有 `--shared-queries` 的 ×B 展开改为:`PIM_MAC_AB` 保持一份;
`PIM_WR_GB`(装 n 条 Q)、`PIM_MV_SB`(n 份部分分数上行)、`PIM_SFM`(n 次
query-private softmax)、`PIM_MV_GB`(n 份概率下行)仍 ×n。
理由:Fugue 正文 §4.3/§4.5.3 — softmax 与概率回流是按 agent(query)私有的
("the die assembles one score per position of the agent's context";mask gate 按
agent 作用),这些数据量真实随 n 增长;而 K/V 的列读在 MQ 下物理上只发生一次。

### 2.2 n 的上限与 sweep 拆分(O3)

```
n_cap = min( floor(gemv_buffer_bytes / 64),   # score 相:每条 Q 切片 64 B/bank
             用户给的 batch 值 )
超过 n_cap 的批,拆成 ≤ n_cap 的多个 sweep(多趟扫描,趟间有依赖)。
```

理由:
- 64 B/条:d_head=128、BF16,BG 内 4 bank 分摊 → 每 bank 每条 Q 64 B(=2 项
  32 B 缓冲项);AttAcc §5.1 的 GEMV buffer = "double-buffered 16 256-bit buffers"
  = 512 B → 默认 `--gemv-buffer-bytes 512` → n_cap=8,零面积改动;面积扫描就是
  改这个参数。
- 拆 sweep:Fugue 正文 §4.5.3 "…bounded by the per-agent state …, beyond which
  **the sweep splits**" — 正文已有"超界拆扫"的语义,这里把 bank 侧
  (GEMV buffer/PE)也纳入同一 bound 结构。正文目前只列了 die 侧
  (softmax buffer、decoder metadata buffer)——bank 侧这半句是正文将来要补的,
  此处先在模型里实现(见 2.5 注)。
- PE 频率不单独设 cap:频率不够时 interval(n) 自动变长(时序惩罚),不禁止 —
  与 DRAM 时序的表达方式一致。

### 2.3 代码落点(O2–O4)

| 文件 | 改动 |
|---|---|
| `ramulator2/trace_gen/gen_trace_attacc_bank.py` | `--mq` 旗标:展开循环里 `PIM_MAC_AB` 不复制,其余非 barrier 命令 ×n |
| `src/ramulator_wrapper.py` | Layer 新属性 `pim_batch_command`/`pim_pe_freq_ghz`/`pim_gemv_buffer_bytes`;interval(n) 计算;`make_yaml_file` 写 `nCCDAB:` 覆盖;`--mq` 透传;运行签名加(模式, interval) |
| `src/ablation.py` | `_pim_scan`:query_batch 按 n_cap 截断;op 上带 MQ 属性(AblationConfig 加字段) |
| `src/workload_runner.py` | decode 批扫描与 prefill 分组扫描两处:组按 n_cap 拆 sweep;op 上带 MQ 属性 |
| `src/main.py` | `--pim-batch-command {mq,replicate}`(默认 **mq**)、`--pe-freq-ghz`(默认 0.666)、`--gemv-buffer-bytes`(默认 512) |

兼容性决定:**main.py 的用户入口默认 mq(这是定稿设计);代码内部 Layer 的默认值
保持 replicate**,既有 27 个单测与旧脚本不受影响(它们不走新旗标)。
理由:实验可复现性(旧结果仍可复跑对照),同时新设计成为对外默认。

### 2.4 实验(O5–O6;64 核,`--ramulator-workers 48`)

1. **微基准(校时序)**:每行 k 条 MAC 的合成 trace,PC/NPC × f_PE∈{0.666,1.3} ×
   n∈{1,2,3,4,6,8},实测 cycles/row 对照 interval(n) 公式;确认 ACT/row=1。
2. **decode 多 agent 共享 chunk**:一段 L∈{1024,4096} 的 chunk,N_ag∈{2,4,8} 条 Q
   一个 sweep;三方案对照:**MQ** / **replicate(现行 ×B 命令)** / **dense**
   (每 agent 一份私有 KV、各自单独扫,AttAcc 参照的形状)。
   指标:每步扫描 latency、ACT 次数(按行数从 run 几何推出,并注明口径)、
   能耗、PE 利用率(= n×MAC 命令数 / (扫描时长×f_PE))。
   理由:Fugue 正文 §4.5.3 Eq.(actcost) 正是 "n_act ≈ n_row(1+ρ_b N_ag) vs
   dense N_ag·n_row" 的对比;latency-vs-dense 是chenyi9 定的目标口径。
3. **prefill 多 Q**:复用 L=4096,计算 token 数 n_r∈{4,8,16,32},按 n_cap=8 分组
   拆 sweep;同样三方案对照。
   理由:Fugue 正文 §4.5.3 "a prefill supplies the n_r queries of its computed
   tokens" — 与 decode 用同一行内批处理机制。
4. 端到端冒烟:relay workload 走 main.py 默认(mq)跑通;27 个单测全绿。

### 2.5 与 Fugue 正文的对照(实现完成后要向用户报告的两处正文张力,不代改稿)

- 正文 §4.5.3 "…while **the column accesses** its MACs consume **grow with n_r**":
  MQ 下列访问不再随 n 增长(改为 PE op 随 n 增长)。
- 正文 §4.5.3 的 batch 上界只列 die 侧两个 buffer;MQ 的上界还含 bank 侧
  GEMV buffer 面积与 PE 频率。

## 三、验收标准

- 微基准实测 cycles/row 与 interval(n) 公式逐点吻合(±refresh 摊销)。
- MQ 相对 replicate:同 n 下扫描 latency 下降、DRAM 读能耗 ≈ ÷n、ACT 数相同;
  相对 dense:ACT 数 ≈ ÷N_ag,latency 显著低于 N_ag 次独立扫描。
- 全部既有单测通过;replicate 路径数值与改动前逐位一致。
