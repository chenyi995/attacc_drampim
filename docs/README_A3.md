# A3:软件复用 + AttAcc,乱序布局(不分 channel)

一句话:软件复用(同 A2)+ **decode 进 PIM、KV 驻留 PIM**,但 KV 布局是
naive 的:共享/私有/重算行按软件到达顺序混放在一个 channel 池里,
**没有任何 PIM 感知的重映射**,也不把 master(不可变共享行)与
diff(重算修正行)分开。A3 暴露"把软件复用直接怼到 PIM 上"的代价:
扫描碎成很多不连续的物理段 (run)。

## 1. 配置(`src/ablation.py::PRESETS["A3"]`)

| 旋钮 | 值 |
|---|---|
| prefill_attn | gpu |
| decode_attn | pim |
| kv_mapping | **naive** |
| pim_batch_command | replicate |

## 2. 模型的每一步在哪里做(解析路径)

| 步 | 做什么 | 代码位置 |
|---|---|---|
| 1–4 | 解析/复用计划/组批/历史加宽/scale 折算 | 同 A2(`build_reuse_plan`、`run_ablation_report`) |
| 5 | prefill 投影/FF/注意力按 GPU 定价 | `_prefill_batch` 顶部循环 |
| 6 | **KV 回读** (readback):prefill 要 attend 的复用行与历史行驻在 PIM,须经链路拉回 GPU(`kv_pim_to_gpu`,每行 2×hidden 字节) | `_prefill_batch` 的 `config.prefill_attn == "gpu"` 分支(`prefill_kv_readback`,行数 = reused_rows + 每请求 history) |
| 7 | **naive 布局建模**:decode 扫描的物理 run 列表按"到达顺序混放、无重映射"生成——共享段与私有段交错,run 数多、每 run 短 | `src/ablation.py::_batch_scan_profile`(kv_mapping=="naive" 分支;历史行前置一段驻留 extent) |
| 8 | decode score 扫描:每个 channel 池独立、池内 run 串行,Ramulator 实测 | `_decode_block_time` 的 `profile.pools` 循环 → `_pim_scan` |
| 9 | 重算行的处置:naive 下重算行覆盖原位置(无 diff 池概念) | `_batch_scan_profile` naive 分支 |
| 10 | 报告(含占用:`_memory_report` 计 KV 行、历史行) | `run_ablation_report` |

## 3. 物理事件路径的对应物

物理 TLB(`src/workload_runner.py::CacheBlendTLB`)天然是 master/diff
分池的(A4 语义),**没有 naive 布局的物理孪生**——A3 的"乱序、不分
channel"目前只在解析路径有模型。这是已知差距,对比实验以解析路径为准。

## 4. 怎么跑

```bash
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json \
  --reuse cacheblend --cacheblend-recompute-ratio 0.15 \
  --cacheblend-full-layers 0 --cacheblend-partial-layers 1,2 \
  --ablation A3 --history-len 3 --pipeopt --workload-report /tmp/a3.json
```

## 5. 与相邻档的关系

A3 对 A2:隔离"decode 进 PIM + KV 驻留"的收益(代价是 prefill 回读);
A3 对 A4:同一切,只把布局换成 master/diff 分池——隔离"PIM 感知布局"
本身的价值(run 数、行激活次数、掩码读的差异)。
