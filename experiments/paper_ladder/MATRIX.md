# paper_ladder:论文主矩阵(同一 workload × A1–A6 × 多模型)

目的:给论文的**问题 1(放置)**提供成体系的数据——同一批 workload、
同一软件上游、只动放置,六档阶梯 × 三个真实模型;并在 A6 点上单独扫
"选择规则"维,证明放置结论对软件上游不敏感。**问题 2(MQ 批处理微架构)**
由 experiment 分支的 C 系列实测支撑,本矩阵消费其机制(A5/A6 跑 mq 命令)
而不重测。

## 1. 轴定义

- **workload(六档完全相同)**:`workloads/` 里四个 case——
  Mooncake toolagent(生产 agent trace)、ShareGPT(多轮)、
  MultiHop-RAG(共享文档)、relay(合成对照);统一 `--history-len 3`
  (multi-round agentic 默认口径)。
- **模型**:LLAMA-7B(32 层)/ LLAMA-65B(80 层)/ GPT-175B(96 层),
  维度见 `src/config.py::model_table`;物理 DAG 的 dynamic 选边比例另用
  CACHEBLEND-TINY + LLAMA-7B(事件路径在 80+ 层大模型上超冒烟预算)。
- **阶梯**:A1..A6(A1 定义性配 no-reuse)。
- **软件上游(2026-08-25 裁决)**:只用**保证重算**的选择族,成员只差
  "选哪些 token 重算":阶梯行固定用 EPIC 特例(选每位移段前 k=8);
  A6 点扫 cacheblend(r=0.15)/ cachecraft(α=0.1)/ cachetune
  (r=0.15 离线)。零重算(promptcache)不进矩阵。

作业数:4×3×6 阶梯 = 72 + 4×3×3 选择扫 = 36 + 8 个 DAG dynamic
= 116;`run_matrix.py` 以 16 并发 × 每作业 4 个 Ramulator worker = 64 核。

## 2. 指标(`collect_results.py`)

| 指标 | 论文含义 | 报告字段 |
|---|---|---|
| **TTFT** | prefill 侧收益(问题 1 的主维) | `prefill_s` |
| **TBT** | decode 每 token 延迟(放置不应伤 decode) | tier 的 `decode_per_token_s` 按步数加权 |
| **压缩率** | KV 占用 vs 无复用基线(共享省下的容量) | `memory.kv_bytes_vs_no_reuse` |
| **dynamic 走 PIM 比例** | Fugue 规则的实际行为 | 解析:breakdown 里 PIM/GPU 侧 prefill 时间份额;物理:`pim_prefill_sides` 请求份额 + 事件计数份额 |

## 3. 复现

```bash
cd experiments/paper_ladder
python3 run_matrix.py                 # 全矩阵(可 --only 过滤,断点续跑)
python3 collect_results.py > results/summary.json
```

结果与逐作业日志在 `results/`;对照文章 claim 的核对在
`CLAIMS_CHECK.md`。
