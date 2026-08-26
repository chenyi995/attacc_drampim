# A2:纯 GPU 跑软件复用(软件上游基线)

一句话:**没有 PIM 的世界**——prefill 与 decode 注意力都在 GPU,KV 缓存
留在 GPU 显存(kv_mapping = none),但打开软件复用策略
(CacheBlend/EPIC/promptcache/cachecraft/cachetune 任一)。A2 回答:
"只靠软件复用、不加任何硬件,能拿到多少?"

## 1. 配置(`src/ablation.py::PRESETS["A2"]`)

| 旋钮 | 值 | 含义 |
|---|---|---|
| prefill_attn | gpu | prefill 注意力在 GPU |
| decode_attn | gpu | decode 注意力也在 GPU(与其他档的本质区别) |
| kv_mapping | none | KV 不进 PIM,留在 GPU 显存 |
| pim_batch_command | replicate | 无 PIM,批命令无效果 |

约束(`resolve_config` 校验):`decode_attn gpu` 必须配
`kv_mapping none`;反之 none 只能配 gpu decode。

## 2. 模型的每一步在哪里做(解析路径)

| 步 | 做什么 | 代码位置 |
|---|---|---|
| 1 | 解析 workload | `src/workload.py::load_workload` |
| 2 | 复用计划:哪些段可复用、每段要重算哪些行(策略族语义见 `README_software_upstream.md`) | `src/workload.py::build_reuse_plan`(cacheblend 族采样 `cacheblend_partial_rows`;epic 族给每个决策 `epic_prefix_rows` 前缀) |
| 3 | tier 组批 + 历史加宽 | `run_ablation_report`(同 A1) |
| 4 | prefill:复用折算成有效行数 scale(cacheblend 族按 recompute_fraction,epic 族扣除前缀行) | `run_ablation_report` 中 `saved_rows`/`effective_rows` 计算(按 `CACHEBLEND_FAMILY`/`EPIC_FAMILY` 分支) |
| 5 | prefill 投影/FF/注意力按 GPU 定价(行数用 scale 折算) | `src/ablation.py::_prefill_batch` 顶部循环 |
| 6 | **无 KV 回读**:KV 本来就在 GPU(与 A3/A4 的关键区别) | `_prefill_batch` 中 readback 分支条件 `config.decode_attn == "pim"` 为假,不收链路费 |
| 7 | decode 注意力照常 GPU 定价(无扫描、无 PIM 事件) | `src/ablation.py::_decode_block_time`(decode_attn=="pim" 分支不进) |
| 8 | 报告 | `run_ablation_report` |

## 3. 有效性边界

A2 在 `--system dgx-attacc` 下照常可跑(校验只对 pim/dynamic 模式要求
PIM 系统);它不产生任何 PIM 事件,`kv_mapping none` 下也没有占用
(memory report 的 PIM 侧为基线口径)。GPU 显存是否装得下 KV **不在**
本模型的约束里——这是解析口径的已知简化,报告对比时需注明。

## 4. 怎么跑

```bash
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json \
  --reuse epic --epic-prefix-recompute-tokens 8 \
  --ablation A2 --history-len 3 --pipeopt --workload-report /tmp/a2.json
```

## 5. 与相邻档的关系

A2 对 A1:隔离"软件复用"的贡献(但 decode 回到 GPU,带宽瓶颈重现);
A2 对 A3:同样的软件复用,decode 搬进 PIM、KV 进 PIM(naive 布局)——
隔离"PIM decode + KV 驻留"的贡献。
