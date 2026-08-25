# 真实 workload:来源、转换与首批结果

本仓库自带的 `workload/*.json` 是合成编排;为了让 A 系列在**真实的**
多智能体/多轮负载上出数,我们在 `/data2/chenyi9/KV-PIM/workload/`
(仓库外的共享文件夹)收集了公开数据集并写了转换器。完整来源清单、
许可与字段映射见该文件夹的 `SOURCES.md`(2026-08-24 收集)。

## 1. 已转换、已跑通的三个

| 来源 | 是什么 | 转换后 | 映射要点 |
|---|---|---|---|
| **Mooncake**(FAST'25,Kimi 生产 trace,Apache-2.0) | 每请求给出 512-token 前缀块的哈希序列,块哈希相同即可共享 KV;`toolagent_trace` 是真实工具/agent 负载 | `converted/workload_mooncake_toolagent_n40_o0.json`(40 请求,479k 输入 token) | 每块 = 一个段,指纹 = trace 自带块哈希;首块 sys、中间 doc、尾部 query;无需分词器(trace 以 token 计) |
| **ShareGPT V3**(94k 真实多轮会话) | 多轮对话文本 | `converted/workload_sharegpt_c10_r3-8_o0.json`(10 会话 ≤8 轮 → 52 个 agent,22k 历史 token) | supervisor DAG:轮 = tier,上一轮回复 = `parent_out` 段,此前全部会话 token 进 `history_len`;tiktoken 计数 |
| **MultiHop-RAG**(COLM'24,ODC-By) | 2,556 个查询,证据横跨 609 篇新闻中的 2–4 篇;共被引文章 = 真实共享文档 chunk | `converted/workload_multihoprag_n32_o0.json`(32 请求,213k token,66 篇去重文档) | 固定指令模板 = 共享 sys 指纹;每篇证据 = doc 段(sha1 指纹,跨请求共享);gold 答案长度作 lout |

另已下载待转换:SWE-agent 轨迹(6,670 条,agent 多轮)、WildChat-1M
分片、Azure/BurstGPT 到达率 trace(无会话结构,只可做长度/到达边缘
分布)。未获取:LMSYS-Chat-1M、GAIA(HF 门禁)。

## 2. 结果

首批冒烟数字已被 **145 作业全矩阵**取代:同一批 workload × A1–A6 ×
LLAMA-7B/65B/GPT-175B,三维指标(TTFT/TBT/压缩率)与 A6 选边比例见
`../experiments/paper_ladder/results/summary.json` 与
`../experiments/paper_ladder/CLAIMS_CHECK.md`(逐 claim 判定)。

## 3. 复现

```bash
# 重新生成转换文件(参数可调,见各脚本 --help)
python3 /data2/chenyi9/KV-PIM/workload/convert_mooncake.py ...
python3 /data2/chenyi9/KV-PIM/workload/convert_sharegpt.py ...
python3 /data2/chenyi9/KV-PIM/workload/convert_multihop_rag.py ...
# 跑一个
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload /data2/chenyi9/KV-PIM/workload/converted/workload_sharegpt_c10_r3-8_o0.json \
  --reuse epic --epic-prefix-recompute-tokens 8 --ablation A6 \
  --pipeopt --workload-report /tmp/sg_a6.json
```

许可注意:ShareGPT 为社区抓取语料,只用于长度/结构统计,不再分发文本;
Mooncake/MultiHop-RAG/SWE-agent/WildChat 均为宽松许可(Apache/ODC-By/
CC-BY),引用见 `SOURCES.md`。
