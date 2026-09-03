# A4:软件复用 + AttAcc,分裂 channel(master/diff 分池)

一句话:在 A3 之上,把 KV 布局换成 **PIM 感知的双池**:16 条 channel 分
成 master 池(不可变共享行,密排连续)与 diff 池(被重算覆盖的修正行,
紧凑追加),被修正的 master 行**读掩码 (read-mask)**——照常随主流读出
但不计分,保持 master run 连续不碎裂。prefill 注意力仍在 GPU。

## 1. 配置(`src/ablation.py::PRESETS["A4"]`)

| 旋钮 | 值 |
|---|---|
| prefill_attn | gpu |
| decode_attn | pim |
| kv_mapping | **master-diff** |
| pim_batch_command | replicate |
| master_pool_channels | **默认 15**(R5,2026-08-24 裁决:master 占 ch0–14、diff 占 ch15;`--kv-pool-split` 可调;本行旧稿"默认 8/8"为 R5 前口径,已改) |
| master_shadow | 默认 read-mask(`--master-shadow`,另一选项 skip 会碎裂 run) |

## 2. 模型的每一步在哪里做(解析路径)

| 步 | 做什么 | 代码位置 |
|---|---|---|
| 1–6 | 同 A3(解析/计划/组批/历史/GPU prefill/KV 回读) | `build_reuse_plan`、`run_ablation_report`、`_prefill_batch` |
| 7 | **双池布局建模**:master 行连续密排进 master 池,diff 行紧凑进 diff 池;两池 channel 集不相交、扫描并发 | `src/ablation.py::_batch_scan_profile`(kv_mapping=="master-diff" 分支;`master_pool_channels` 切池;历史行是每请求私有 extent,前置在 master 池) |
| 8 | 读掩码语义:被修正行随 master run 读出、掩码丢分;skip 模式则从地址流剔除并碎裂 run | `_batch_scan_profile`(`master_shadow` 分支)+ 物理侧 `_masked_rows_per_run` |
| 9 | decode 扫描:池间并发取 max、池内 run 串行 | `_decode_block_time` 的 `profile.pools` 循环 |
| 10 | 报告 | `run_ablation_report` |

## 3. 物理事件路径的对应物(A4 是物理 TLB 的原生布局)

`--pim-prefill-mode gpu` 的事件路径 = A4 的物理孪生:

| 步 | 事件/机制 | 代码位置 |
|---|---|---|
| 布局 | master ch0–14 / diff ch15(`_MASTER_CHANNELS_DEFAULT`,与本页 §1 一致;旧稿此处误写 8/8,2026-09-03 改)、V 在 K 上方 8 MiB、行放置表 | `src/workload_runner.py::CacheBlendTLB`(`_prepare_cacheblend_tlb`) |
| prefill | `kv_pim_to_gpu` 回读 LINK 事件(字节数=行×2×hidden,校验器核对)→ `gpu_prefill_score/softmax/context` 全上下文 GPU 块 → `dram_store_diff_and_live` 新 KV 落库 | `_run_cacheblend_prefill` 的 gpu 分支 |
| D_i 位图 | agent 的重算集位图下发 die(master 写过滤) | `_run_cacheblend_prefill` 的 bitmap 块(`di_bitmap_gpu_to_die`/`die_load_di_bitmap`) |
| decode | TLB 规划 (`tlb_lookup_and_bank_plan`) → 池并发扫描 (`decode_pim_kv_scan_score_softmax_pv`,掩码行统计) | `_append_cacheblend_decode_batched` / `_append_physical_pim_scan` |
| 校验 | 事件契约(链路字节、扫描依赖地址规划、store 顺序) | `validate_cacheblend_events` |

## 4. 怎么跑

```bash
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json \
  --reuse epic --epic-prefix-recompute-tokens 8 \
  --ablation A4 --history-len 3 --pipeopt --workload-report /tmp/a4.json
# 物理孪生:
python3 main.py ... --reuse epic --epic-prefix-recompute-tokens 8 \
  --history-len 3 --pim-prefill-mode gpu --workload-report /tmp/a4_dag.json
```

## 5. 与相邻档的关系

A4 对 A3:隔离 PIM 感知布局的价值;A4 对 A5:prefill 注意力搬进 PIM
(附带 MQ batching)——A4 是"prefill 在 GPU"一侧的最强形态,也是 A6
dynamic 规则里 xPU 路的原型。


---

## 状态更新(2026-08-27)

- **R18 条带映射**:master 池 15 channel 承载**单 KV 头自己的 token
  条带**(废除"每 channel 一个 head"的上游语义);`--num-hbm` 大于
  KV 头数时头内序列切分(每头独占多堆叠并发扫);
- 真实负载默认 `--engine dag` 出数;PC 默认开(R19),replicate 列
  节拍为 PC preset 的 nCCDAB=6。
