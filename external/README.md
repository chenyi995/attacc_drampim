# external:本项目用到/参照的外部软件库与数据源(只留链接,不 vendor 代码)

裁决(chenyi9 2026-08-25):外部库单独放这个文件夹,**保留链接即可**;
本地克隆统一在仓库外的 `/data2/chenyi9/KV-PIM/workload/`(见其
`SOURCES.md`),不进本仓库。

## 1. 复用策略(软件上游)的参考实现与出处

| 名称 | 用途 | 链接 | 许可 |
|---|---|---|---|
| CacheBlend (EuroSys'25) | `--reuse cacheblend` 的机制出处(偏差采样 r) | https://github.com/YaoJiayi/CacheBlend · arXiv:2405.16444 | Apache-2.0 |
| LMCache | CacheBlend 的产品化实现(机制对照) | https://github.com/LMCache/LMCache | Apache-2.0 |
| EPIC / LegoLink (ICML'25) | `--reuse epic`(每位移段前 k)出处;无官方代码 | arXiv:2410.15332 | — |
| PromptCache (MLSys'24) | `--reuse promptcache` 零重算端点出处 | https://github.com/yale-sys/prompt-cache · arXiv:2311.04934 | MIT |
| Cache-Craft (SIGMOD'25) | `--reuse cachecraft`(重叠度变长前缀)机制出处;代码未公开 | arXiv:2502.15734 | — |
| CacheTune 风格 | `--reuse cachetune`(离线选行)机制出处;代码未公开 | arXiv:2605.24022 | — |
| APE (ICLR'25) | 零重算近族对照(注意力重标定) | https://github.com/Infini-AI-Lab/APE | MIT |
| KVLink | prefix 族近亲(可训练 link token) | https://github.com/UCSB-NLP-Chang/KVLink | — |
| TurboRAG | 零重算+微调对照 | https://github.com/MooreThreads/TurboRAG | — |
| Comb | 编码器式 PIC 对照 | https://github.com/shijuzhao/Comb | — |
| ContextPilot (MLSys'26) | 上下文索引/重排对照 | https://github.com/EfficientContext/ContextPilot | — |

(完整 29 篇文献地图见 `docs/README_software_upstream.md` §2。)

## 2. 真实 workload 数据源

| 名称 | 用途 | 链接 | 许可 |
|---|---|---|---|
| Mooncake traces (FAST'25) | `mooncake`(toolagent)与 `mooncakemt`(conversation 真多轮)两个 case | https://github.com/kvcache-ai/Mooncake | Apache-2.0 |
| MultiHop-RAG (COLM'24) | `multihop` case(共享文档 RAG) | https://github.com/yixuantt/MultiHop-RAG · https://huggingface.co/datasets/yixuantt/MultiHopRAG | ODC-By |
| ShareGPT V3 | `sharegpt` case(真实多轮会话;只作长度/结构统计,不再分发文本) | https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered | apache-2.0(社区抓取,注意口径) |
| SWE-agent trajectories | 备用 agent 轨迹源(已下载未转换) | https://huggingface.co/datasets/nebius/SWE-agent-trajectories | CC-BY-4.0 |
| WildChat-1M | 备用多轮源 | https://huggingface.co/datasets/allenai/WildChat-1M | ODC-By |
| Azure LLM traces 2023 | 到达率/长度边缘分布 | https://github.com/Azure/AzurePublicDataset | CC-BY |
| BurstGPT | 到达率 trace(无会话结构) | https://github.com/HPMLL/BurstGPT | CC-BY-4.0 |

## 3. 仿真/工具链上游

| 名称 | 用途 | 链接 |
|---|---|---|
| AttAcc simulator (ASPLOS'24) | 本仓库的上游基座(`origin` 即其 fork) | https://github.com/scale-snu/attacc_simulator |
| Ramulator 2.0 | DRAM 周期仿真内核(HBM3-PIM patch 在本仓库) | https://github.com/CMU-SAFARI/ramulator2 |
| Samsung PIMSimulator | FIMDRAM 先例(MQ 时序模型的佐证) | https://github.com/SAITPublic/PIMSimulator |
| tiktoken | 文本源的 token 计数(转换器) | https://github.com/openai/tiktoken |
| kvpim-rtl(本项目) | bank PE / BG 归约 / die 侧 RTL 与综合 | https://github.com/chenyi995/kvpim-rtl |
