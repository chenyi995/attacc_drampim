# A5:所有 prefill 注意力进 PIM + MQ attention batching

一句话:在 A4 之上,把 **prefill 注意力也整块搬进 bank**(物理形态叫
bank-whole:本批新 K/V **先落库**,每条查询扫**整个已着陆范围**,DIE 按
因果丢弃非法位置),并且从本档起启用 **MQ 批命令**(一次 DRAM 列读服务
GEMV buffer 内全部驻留查询)——attention batching 与 prefill 上 PIM
是同步引入的(2026-08-24 约定)。

## 1. 配置(`src/ablation.py::PRESETS["A5"]`)

| 旋钮 | 值 |
|---|---|
| prefill_attn | **pim** |
| decode_attn | pim |
| kv_mapping | master-diff |
| pim_batch_command | **mq**(preset 自带;`--pim-batch-command` 可显式覆盖) |
| pim_pe_freq_ghz / gemv_buffer_bytes | 自由参数(`--pe-freq-ghz`/`--gemv-buffer-bytes`,不预设 RTL 结论) |

## 2. 模型的每一步在哪里做(解析路径)

| 步 | 做什么 | 代码位置 |
|---|---|---|
| 1–4 | 解析/复用计划/组批/历史加宽 | 同 A4 |
| 5 | prefill 投影/FF 按 GPU 定价;注意力**跳过 GPU** | `_prefill_batch` 顶部循环(attention 且 prefill_attn != "gpu" 时 continue) |
| 6 | 层分类:全重算层(cacheblend 族)走 GPU 重建 + PIM 扫历史;携带复用的层走 PIM 整段 | `_prefill_batch` 的 `classes` 组装与 `carries_reuse` 分支(scan_rows = lin + history) |
| 7 | PIM 扫描定价:每波 ≤ 容量条查询共享一次扫描,`容量 = gemv_buffer_bytes/64`,超出拆波 (passes) | `src/ablation.py::_pim_scan`(`mq_query_capacity` 封顶、`_apply_pim_batch` 语义) |
| 8 | MQ 时序(**R19,2026-08-27**):列命令间隔 = max(地板 6 PC/4 NPC, ⌈n/(f·tCK)⌉, **PC 能量钳位** ⌈6·(E_col+n·E_op·ê)/E₆⌉)——PC 的 6-tCK 地板本质是每窗口能量预算(E₆=列读+一次 MAC),n=8 → **8 tCK** 与频率无关;NPC 无钳位;PE 功率另有 116 W 诊断账 | `src/ramulator_wrapper.py::mq_interval_cycles` / `mq_pe_power_w`(YAML nCCDAB 覆盖注入) |
| 9 | Q/ctx 链路费与 PIM softmax | `_prefill_batch` 的 link_time 与 sfm 定价 |
| 10 | decode 同 A4(扫描也吃 mq 命令) | `_decode_block_time` → `_pim_scan` |

## 3. 物理事件路径的对应物(`--pim-prefill-mode pim`)

| 步 | 事件/机制 | 代码位置 |
|---|---|---|
| 着陆序 | 本层 fresh/corrected K/V 先落库(`dram_store_diff_and_live` 是扫描的前置依赖) | `_run_cacheblend_prefill` 的 pim(bank-whole)分支 |
| 全范围扫描 | scan_locations = 复用行(被修正者以母本占位、读掩码)+ 全部新行;每波 ≤ `mq_query_capacity` 条查询,Q 旋转分发 + TLB 规划 + 一次共享扫描 | 同上(sweep 循环;`_apply_pim_batch` 打 MQ 戳) |
| 因果丢弃 | DIE 逐查询组装分数、丢 key>query 的上三角(被扫仍计费;无 GPU 三角、无 LSE) | `die_score_assembly` 事件;契约在 `validate_cacheblend_events` |
| D_i 位图 | master 写过滤位图装载(EPIC 每 agent 一次、CacheBlend 每 partial 层一次) | bitmap 块(5.L1) |
| decode | 共享 KV 的批 decode 按容量拆波(每波重扫全范围;波内 MQ) | `_append_cacheblend_decode_batched`(sweep_cap 循环、逐波准入审计) |
| trace 层 | `--mq` 折叠 MAC(一列一条)、`--phase` 切相;查询私有搬运(WRGB/MVSB/SFM/MVGB)仍每查询一份;**流式 P**:概率向量不驻留,MV_GB 经 TSV 计价(同 BG 连发受 nCCDL,方向转向 nRTW/nWTRL) | `ramulator2/trace_gen/gen_trace_attacc_bank.py`;C++ 转向约束在 `ramulator2/src/dram/impl/HBM3-PIM.cpp` |

## 4. 怎么跑

```bash
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json \
  --reuse epic --epic-prefix-recompute-tokens 8 \
  --ablation A5 --history-len 3 --pipeopt --workload-report /tmp/a5.json
# 物理孪生:
python3 main.py ... --history-len 3 --pim-prefill-mode pim \
  --workload-report /tmp/a5_dag.json
```

自检:报告 `ablation.pim_batch_command == "mq"`(A5 preset 自带)。

## 5. 与相邻档的关系

A5 对 A4:隔离"prefill 上 PIM + batching"的贡献(省掉回读与 GPU 大块,
代价是扫描含被丢弃的上三角);A5 对 A6:A5 是**强制全 PIM**,A6 把
"上不上 PIM"交给逐请求的动态规则——A5 恰是 A6 在"bank 路永远更快"的
极限下的行为。

---

## 状态更新(2026-08-26/27,详见总台账 R14–R17 与 sessions/)

- MQ 设计点按裁决改 **n_cap=8**:GEMV buffer 512 B、PE **1.3004 GHz**
  (=1/tCK,每命令拍一次 MAC;PC 能量钳位使间隔为 8 tCK,R19)
  (R16,替代 12 驻留/768 B/2.6 GHz);
- decode 服务批宽 8(与各档一致);MQ 批命令仅本档与 A6 启用。
