# k=2 阶梯结果（LLAMA3-8B，MQ PIM @ 1.30 GHz）

由 `make_results_tables.py` 自动生成。k 是 reuse policy 对每个 shifted chunk 重算的 token 数（此处 2）。每个 workload 取最新的完整 run；某 workload 在此 k 下不足 7 档则跳过。**所有数值均为仿真实测**，来源文件与字段在每处标注。

## 各 workload 的编排（分 tier 讲）

### star-repair

star topology（星型），仿 AutoGen / MetaGPT / AgentCoder。一个 **main** agent（300-token system prompt + 200-token task）指挥 **三个 worker**（各 300-token system prompt），共 **5 轮**。每轮 main 读三个 worker 的 256-token 回复、发一条 128-token instruction 给全体 worker；共享一个 47-chunk（12,032-token）codebase，分 5 个 stage 释放。20 个 request，最深 context 40,692 token。**layer = 编排轮次**（main 层与 3-worker 层交替，layer 0 是冷启动的 main）。

**分 tier**（tier map 取自 `20260828-125506_workload_star_repair_r5w3k47_LLAMA3-8B_k2/dag_A2.json` 的 `workload.tiers`；agent ID 为 workload 原样字段）:

| tier | agents | 干什么 |
|---|---|---|
| 0 | main.r0 | round 0：main 读 200-token task，发 128-token instruction 给 worker |
| 1 | w0.r0、w1.r0、w2.r0 | round 0：三个 worker 各读 main 的 instruction，各产出一条 256-token 回复（在本轮释放的 codebase stage 上） |
| 2 | main.r1 | round 1：main 读三个 worker 的 256-token 回复，发下一条 128-token instruction |
| 3 | w0.r1、w1.r1、w2.r1 | round 1：三个 worker 各读 main 的 instruction，各产出一条 256-token 回复（在本轮释放的 codebase stage 上） |
| 4 | main.r2 | round 2：main 读三个 worker 的 256-token 回复，发下一条 128-token instruction |
| 5 | w0.r2、w1.r2、w2.r2 | round 2：三个 worker 各读 main 的 instruction，各产出一条 256-token 回复（在本轮释放的 codebase stage 上） |
| 6 | main.r3 | round 3：main 读三个 worker 的 256-token 回复，发下一条 128-token instruction |
| 7 | w0.r3、w1.r3、w2.r3 | round 3：三个 worker 各读 main 的 instruction，各产出一条 256-token 回复（在本轮释放的 codebase stage 上） |
| 8 | main.r4 | round 4：main 读三个 worker 的 256-token 回复，发下一条 128-token instruction |
| 9 | w0.r4、w1.r4、w2.r4 | round 4：三个 worker 各读 main 的 instruction，各产出一条 256-token 回复（在本轮释放的 codebase stage 上） |

### pipeline-repair

ChatDev / MetaGPT 式 waterfall。一个 **architect**（300-token system prompt、200-token task、256-token plan）开链，之后 **engineer**（256-token patch）与 **reviewer**（128-token review）交替 5 个 cycle，各自保留 history；共享 50-chunk（12,800-token）codebase，沿 12 个链位释放。最后一个 history-free 的 **tester** 读全部 50 chunk。最深 context 40,668 token。**layer = 12 个链位**（architect、engineer.c0、reviewer.c0、…、tester），每层 1 个 agent。

**分 tier**（tier map 取自 `20260828-130630_workload_pipeline_repair_c5k50_LLAMA3-8B_k2/dag_A2.json` 的 `workload.tiers`；agent ID 为 workload 原样字段）:

| tier | agents | 干什么 |
|---|---|---|
| 0 | architect | 开链：把 200-token task 变成 256-token plan |
| 1 | engineer.c0 | cycle 0：读 plan 和上一条 review，写 256-token patch |
| 2 | reviewer.c0 | cycle 0：读 patch，写 128-token review |
| 3 | engineer.c1 | cycle 1：读 plan 和上一条 review，写 256-token patch |
| 4 | reviewer.c1 | cycle 1：读 patch，写 128-token review |
| 5 | engineer.c2 | cycle 2：读 plan 和上一条 review，写 256-token patch |
| 6 | reviewer.c2 | cycle 2：读 patch，写 128-token review |
| 7 | engineer.c3 | cycle 3：读 plan 和上一条 review，写 256-token patch |
| 8 | reviewer.c3 | cycle 3：读 patch，写 128-token review |
| 9 | engineer.c4 | cycle 4：读 plan 和上一条 review，写 256-token patch |
| 10 | reviewer.c4 | cycle 4：读 patch，写 128-token review |
| 11 | tester | history-free：读全部 50 chunk，测试成品代码 |

### debate

multiagent debate / Mixture-of-Agents。**三个对称 debater** 就一个 100-token 问题辩论，共享 49-chunk（12,544-token）文档分 5 个 stage 释放；每轮各 debater 重读自己的 history 与两个对手的 256-token 答案，再答 256 token。最后一个 history-free 的 **judge** 只读三份终答、出 128-token 裁决。最深 context 41,616 token。**layer = 辩论轮次**（每轮一个 3-debater 层，最后一个 judge 层）。

**分 tier**（tier map 取自 `20260828-173206_workload_debate_d3r5k49_LLAMA3-8B_k2/dag_A2.json` 的 `workload.tiers`；agent ID 为 workload 原样字段）:

| tier | agents | 干什么 |
|---|---|---|
| 0 | d0.r0、d1.r0、d2.r0 | round 0：三个 debater 读 100-token 问题与文档，各答 256 token |
| 1 | d0.r1、d1.r1、d2.r1 | round 1：三个 debater 重读各自 history 与两个对手的 256-token 答案，再各答 256 token |
| 2 | d0.r2、d1.r2、d2.r2 | round 2：三个 debater 重读各自 history 与两个对手的 256-token 答案，再各答 256 token |
| 3 | d0.r3、d1.r3、d2.r3 | round 3：三个 debater 重读各自 history 与两个对手的 256-token 答案，再各答 256 token |
| 4 | d0.r4、d1.r4、d2.r4 | round 4：三个 debater 重读各自 history 与两个对手的 256-token 答案，再各答 256 token |
| 5 | judge | history-free：只读三份终答，出 128-token 裁决 |

### map-reduce  *（此 k 下尚未跑完，不进下方数值表）*

map-reduce 摘要，**低复用对照（low-reuse control）**。8 个并发 **mapper** 各读一段私有、不重叠的 24,576-token 切片 + 共享的 300-token system prompt，各出 200-token 摘要；一个 **reducer**（1,900-token context）汇成 256-token 终稿。只有 system prompt 和一次性摘要可共享，共享比例很低。**layer = map 层，然后 reduce 层。**

**分 tier**（tier map 取自 `20260828-125509_workload_mapreduce_sum_m8_LLAMA3-8B_k2/dag_A2.json` 的 `workload.tiers`；agent ID 为 workload 原样字段）:

| tier | agents | 干什么 |
|---|---|---|
| 0 | map0、map1、map2、map3、map4、…（共 8 个） | map 层：8 个 mapper 各读一段私有、不重叠的 24,576-token 切片 + 共享 300-token system prompt，各出 200-token 摘要 |
| 1 | reduce | reduce 层：reducer 读 8 份摘要，出 256-token 终稿 |

### multi-source RAG

12 个独立单轮 RAG 查询。每个查询用滑窗取 96 个不同的 256-token source chunk，窗口每次滑 1 个 source，所以相邻查询共享 96 中的 95 个 source（24,976-token 输入、64-token 答案）。全部共享是内容重叠，无 history、无 output 复用。**单 layer**，12 个查询。

**分 tier**（tier map 取自 `20260828-130628_workload_multisource_rag_n12s96_LLAMA3-8B_k2/dag_A2.json` 的 `workload.tiers`；agent ID 为 workload 原样字段）:

| tier | agents | 干什么 |
|---|---|---|
| 0 | q0、q1、q2、q3、q4、…（共 12 个） | 12 个独立单轮查询；每个用滑窗取 96 个 256-token source chunk，窗口每次滑 1 个 source，相邻查询共享 96 中的 95 个 |

## 名词定义

- **makespan**（计算机语言）：整个 workload 的 **schedule length**（调度长度），即从第一个 request 到达、到最后一个 request 完成的墙钟时间 $\max_i C_i$（所有 request 完成时间的最大值）。这是多处理器 scheduling 的经典目标（minimize makespan）；越小=整批 agent 越早全部跑完。它计入了 overlap 与 queueing，**≠ 各 request latency 之和**。来源字段 `makespan_s`。

- **layer**：agentic workflow 的一层（一轮 / 一个 role stage），即报告的 `tier`。各 workload 的 layer 见上表。prefill placement 是 per request 决定的，同一 request 的 32 个 transformer layer 走同一侧，所以随 workflow layer 变、不随 transformer layer 变。

- **latency**（per request，在一个 layer 内对该层 agent 取平均）：**prefill** = 从 request 到达到它 prefill 结束；**decode** = 从它首个 output token 到最后一个；**e2e = prefill + decode**。到达取该 layer 的释放时刻（`summary.tiers[tier].start_s`），故不含等待上游 layer 的时间。来源字段 `summary.requests` 的 `prefill_end_s` / `first_token_s` / `end_s`。

- **power / energy**：轨迹只按 phase（`decode*` 事件 vs 其余一次性 prefill 事件）、按 unit（GPU/PIM/LINK/DIE）、按整个 run 打标，**不带 layer 标签**，所以 energy 与 average power（energy / makespan）**只给 per rung**。来源字段 `energy_breakdown_nj`、`energy_nj`。

- **rung（档）**：**A1** AttAcc dense 无复用（decode 在 bank、prefill 在 GPU）；**A2** GPU-only 软件复用；**A3** decode 进 bank、跑 append-order 布局；**A3a** + 陈旧行 write mask；**A4** + split-channel master-diff 布局；**A5** + 所有 prefill attention 进 bank；**A6 = Fugue**，A5 + 动态 per-request placement rule。

## 总览：整条阶梯

各 workload 的数取自其小节标注的 run 目录下 `dag_A1.json … dag_A6.json`。


**Makespan (s)**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 53.25 | 70.19 | 38.13 | 27.86 | 13.99 | 11.32 | 11.32 |
| pipeline-repair | 32.83 | 37.91 | 20.21 | 19.57 | 13.46 | 12.24 | 12.16 |
| debate | 43.84 | 62.86 | 26.95 | 24.09 | 11.61 | 9.23 | 9.22 |
| multi-source RAG | 86.69 | 23.44 | 27.07 | 13.10 | 9.96 | 8.82 | 8.82 |

**总能量 total energy (kJ)**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 148.0 | 2.9 | 5.0 | 4.5 | 4.7 | 4.8 | 4.8 |
| pipeline-repair | 76.1 | 2.1 | 2.8 | 2.8 | 2.9 | 3.0 | 2.8 |
| debate | 124.5 | 2.5 | 4.1 | 3.9 | 4.3 | 4.7 | 4.7 |
| multi-source RAG | 281.7 | 1.0 | 2.0 | 1.3 | 1.4 | 1.5 | 1.5 |

**平均功率 average power (W)**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 2779 | 41 | 131 | 160 | 338 | 423 | 423 |
| pipeline-repair | 2318 | 55 | 140 | 143 | 219 | 245 | 227 |
| debate | 2839 | 40 | 151 | 162 | 367 | 508 | 508 |
| multi-source RAG | 3250 | 44 | 75 | 103 | 143 | 175 | 175 |

**prefill attention 落 PIM 的 agent 占比**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 0% | 0% | 0% | 0% | 0% | 95% | 90% |
| pipeline-repair | 0% | 0% | 0% | 0% | 0% | 92% | 58% |
| debate | 0% | 0% | 0% | 0% | 0% | 81% | 75% |
| multi-source RAG | 0% | 0% | 0% | 0% | 0% | 92% | 92% |

**A6（Fugue）相对 A1、A2 的加速**

| workload | A6 vs A1 | A6 vs A2 | energy A6 vs A1 |
|---|---|---|---|
| star-repair | 4.7x | 6.2x | 31x |
| pipeline-repair | 2.7x | 3.1x | 28x |
| debate | 4.8x | 6.8x | 27x |
| multi-source RAG | 9.8x | 2.7x | 182x |

---

## star-repair

**原始数据目录** `20260828-125506_workload_star_repair_r5w3k47_LLAMA3-8B_k2/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 155.2 | n/a | 147.9 | 147.9 | 147.8 | 147.8 | 147.8 |
| 1 | 301.1 | n/a | 28.1 | 28.1 | 27.9 | 25.7 | 25.6 |
| 2 | 657.8 | n/a | 374.0 | 374.0 | 373.7 | 210.9 | 210.9 |
| 3 | 1151.0 | n/a | 30.1 | 30.1 | 29.4 | 14.6 | 14.6 |
| 4 | 1764.4 | n/a | 739.0 | 739.0 | 738.3 | 345.7 | 345.7 |
| 5 | 3135.9 | n/a | 57.3 | 57.3 | 56.0 | 24.3 | 24.3 |
| 6 | 3386.2 | n/a | 1078.9 | 1078.9 | 1077.8 | 485.8 | 485.8 |
| 7 | 6070.4 | n/a | 95.8 | 95.8 | 93.6 | 35.2 | 35.2 |
| 8 | 5857.6 | n/a | 1559.9 | 1559.9 | 1558.3 | 700.2 | 700.2 |
| 9 | 10820.2 | n/a | 150.4 | 150.4 | 147.3 | 52.3 | 52.3 |
| **全部** | 3812.9 | n/a | 249.2 | 249.2 | 247.9 | 117.3 | 117.3 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 550.1 | n/a | 617.6 | 617.6 | 550.3 | 547.4 | 547.4 |
| 1 | 1130.4 | n/a | 2375.6 | 1618.6 | 1154.9 | 1135.9 | 1135.9 |
| 2 | 558.3 | n/a | 782.3 | 757.4 | 561.0 | 552.8 | 552.8 |
| 3 | 1176.0 | n/a | 3681.1 | 2320.3 | 1210.9 | 1175.6 | 1175.6 |
| 4 | 569.6 | n/a | 960.6 | 937.0 | 574.9 | 558.7 | 558.7 |
| 5 | 1239.2 | n/a | 5385.4 | 3325.0 | 1302.1 | 1249.7 | 1249.7 |
| 6 | 584.1 | n/a | 1202.0 | 1177.0 | 594.4 | 567.3 | 567.3 |
| 7 | 1323.9 | n/a | 7382.2 | 4739.3 | 1414.3 | 1346.0 | 1346.0 |
| 8 | 601.9 | n/a | 1493.2 | 1468.2 | 617.4 | 577.2 | 577.2 |
| 9 | 1428.3 | n/a | 9663.1 | 6350.8 | 1535.6 | 1451.2 | 1451.2 |
| **全部** | 1087.9 | n/a | 4525.9 | 3001.0 | 1137.6 | 1093.9 | 1093.9 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 705.3 | n/a | 765.5 | 765.5 | 698.0 | 695.1 | 695.1 |
| 1 | 1431.5 | n/a | 2403.7 | 1646.8 | 1182.8 | 1161.5 | 1161.4 |
| 2 | 1216.1 | n/a | 1156.3 | 1131.4 | 934.7 | 763.7 | 763.7 |
| 3 | 2326.9 | n/a | 3711.2 | 2350.4 | 1240.3 | 1190.1 | 1190.1 |
| 4 | 2334.0 | n/a | 1699.6 | 1676.0 | 1313.2 | 904.5 | 904.5 |
| 5 | 4375.1 | n/a | 5442.7 | 3382.3 | 1358.0 | 1274.0 | 1274.0 |
| 6 | 3970.3 | n/a | 2280.9 | 2255.9 | 1672.2 | 1053.1 | 1053.1 |
| 7 | 7394.3 | n/a | 7477.9 | 4835.1 | 1508.0 | 1381.2 | 1381.2 |
| 8 | 6459.4 | n/a | 3053.1 | 3028.1 | 2175.7 | 1277.4 | 1277.4 |
| 9 | 12248.5 | n/a | 9813.5 | 6501.2 | 1682.9 | 1503.6 | 1503.6 |
| **全部** | 4900.7 | n/a | 4775.1 | 3250.2 | 1385.5 | 1211.3 | 1211.2 |

> A2 是 GPU-only software baseline，此 sim path 把 prefill 与 decode 合成一个时间戳（`prefill_end==end`、`first_token=0`），所以它的 per-phase latency 记 `n/a`；其 makespan 见总览。


**每层 prefill attention 放置（HBM vs PIM），每个 agent 一个点**  来源 `dag_A5.json` / `dag_A6.json` 的 `prefill_attention_sides`（A1–A4 按构造 prefill 全在 GPU/HBM）

| tier(layer) | agents | A5 PIM | A5 HBM | A6 PIM | A6 HBM | A6 PIM% |
|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 1 | 0 | 1 | 0% |
| 1 | 3 | 3 | 0 | 2 | 1 | 67% |
| 2 | 1 | 1 | 0 | 1 | 0 | 100% |
| 3 | 3 | 3 | 0 | 3 | 0 | 100% |
| 4 | 1 | 1 | 0 | 1 | 0 | 100% |
| 5 | 3 | 3 | 0 | 3 | 0 | 100% |
| 6 | 1 | 1 | 0 | 1 | 0 | 100% |
| 7 | 3 | 3 | 0 | 3 | 0 | 100% |
| 8 | 1 | 1 | 0 | 1 | 0 | 100% |
| 9 | 3 | 3 | 0 | 3 | 0 | 100% |
| **全部** | 20 | 19 | 1 | 18 | 2 | 90% |

**每档 energy 与 average power**  来源 `dag_A*.json` 的 `energy_breakdown_nj`（by_event / by_class）、`energy_nj`、`makespan_s`、`link_bytes`（power = energy / makespan，不分 layer）

| rung | prefill E (mJ) | decode E (mJ) | total E (mJ) | avg power (W) | GPU | PIM | LINK | KV over link (GiB) |
|---|---|---|---|---|---|---|---|---|
| A1 | 143787217.8 | 4189573.0 | 147976790.8 | 2779 | 2261738.2 | 145713279.9 | 1772.68 | 19.84 |
| A2 | 413331.1 | 2476456.3 | 2889787.4 | 41 | 2774762.5 | 0.0 | 115024.89 | 1287.56 |
| A3 | 299537.4 | 4706550.0 | 5006087.4 | 131 | 1312967.4 | 3690977.3 | 2142.48 | 23.98 |
| A3a | 299537.4 | 4170827.2 | 4470364.5 | 160 | 1312967.4 | 3155254.6 | 2142.48 | 23.98 |
| A4 | 299537.4 | 4425637.0 | 4725174.3 | 338 | 1312967.4 | 3410064.4 | 2142.48 | 23.98 |
| A5 | 1827521.5 | 2965372.3 | 4792893.8 | 423 | 1162618.9 | 3630046.2 | 228.66 | 2.56 |
| A6 | 1822103.2 | 2965372.3 | 4787475.5 | 423 | 1163105.9 | 3624127.2 | 242.31 | 2.71 |

---

## pipeline-repair

**原始数据目录** `20260828-130630_workload_pipeline_repair_c5k50_LLAMA3-8B_k2/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 70.8 | n/a | 63.0 | 63.0 | 63.0 | 63.0 | 63.0 |
| 1 | 144.2 | n/a | 68.5 | 68.5 | 68.4 | 84.7 | 68.4 |
| 2 | 230.4 | n/a | 85.1 | 85.1 | 85.0 | 129.1 | 85.0 |
| 3 | 507.1 | n/a | 116.7 | 116.7 | 116.4 | 108.9 | 116.4 |
| 4 | 744.5 | n/a | 143.3 | 143.3 | 142.9 | 113.1 | 113.1 |
| 5 | 1178.2 | n/a | 205.9 | 205.9 | 205.3 | 141.9 | 141.9 |
| 6 | 1824.2 | n/a | 308.4 | 308.4 | 307.7 | 202.1 | 202.1 |
| 7 | 2484.5 | n/a | 327.1 | 327.1 | 326.1 | 191.3 | 191.3 |
| 8 | 3344.6 | n/a | 384.7 | 384.7 | 383.5 | 198.6 | 198.6 |
| 9 | 4325.0 | n/a | 476.5 | 476.5 | 475.1 | 251.3 | 251.3 |
| 10 | 5591.9 | n/a | 548.0 | 548.0 | 546.4 | 274.0 | 274.0 |
| 11 | 2086.0 | n/a | 246.1 | 246.1 | 245.6 | 266.3 | 245.6 |
| **全部** | 1877.6 | n/a | 247.8 | 247.8 | 247.1 | 168.7 | 162.5 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1101.5 | n/a | 1184.3 | 1184.3 | 1102.0 | 1098.3 | 1098.3 |
| 1 | 1104.5 | n/a | 1271.7 | 1245.6 | 1112.3 | 1106.2 | 1106.2 |
| 2 | 551.2 | n/a | 654.6 | 641.0 | 559.6 | 555.3 | 555.3 |
| 3 | 1118.6 | n/a | 1553.5 | 1489.1 | 1133.0 | 1117.9 | 1117.9 |
| 4 | 559.6 | n/a | 802.8 | 780.2 | 567.5 | 557.8 | 557.8 |
| 5 | 1135.5 | n/a | 1869.9 | 1785.4 | 1156.1 | 1128.4 | 1128.4 |
| 6 | 570.8 | n/a | 993.6 | 961.8 | 582.3 | 565.0 | 565.0 |
| 7 | 1162.1 | n/a | 2315.5 | 2211.3 | 1190.5 | 1143.0 | 1143.0 |
| 8 | 585.1 | n/a | 1242.6 | 1202.5 | 600.2 | 572.1 | 572.1 |
| 9 | 1189.9 | n/a | 2791.7 | 2662.9 | 1231.0 | 1162.6 | 1162.6 |
| 10 | 602.0 | n/a | 1523.9 | 1473.8 | 623.0 | 581.9 | 581.9 |
| 11 | 564.8 | n/a | 940.1 | 870.1 | 582.9 | 569.2 | 569.2 |
| **全部** | 853.8 | n/a | 1428.7 | 1375.7 | 870.0 | 846.5 | 846.5 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1172.2 | n/a | 1247.4 | 1247.4 | 1165.0 | 1161.3 | 1161.3 |
| 1 | 1248.7 | n/a | 1340.2 | 1314.1 | 1180.7 | 1190.9 | 1174.6 |
| 2 | 781.6 | n/a | 739.8 | 726.1 | 644.5 | 684.4 | 640.3 |
| 3 | 1625.7 | n/a | 1670.2 | 1605.8 | 1249.4 | 1226.8 | 1234.3 |
| 4 | 1304.1 | n/a | 946.1 | 923.6 | 710.4 | 670.9 | 670.9 |
| 5 | 2313.8 | n/a | 2075.8 | 1991.3 | 1361.4 | 1270.3 | 1270.3 |
| 6 | 2395.0 | n/a | 1302.1 | 1270.2 | 890.0 | 767.1 | 767.1 |
| 7 | 3646.6 | n/a | 2642.6 | 2538.3 | 1516.6 | 1334.3 | 1334.3 |
| 8 | 3929.7 | n/a | 1627.2 | 1587.2 | 983.7 | 770.6 | 770.6 |
| 9 | 5514.9 | n/a | 3268.3 | 3139.4 | 1706.2 | 1413.9 | 1413.9 |
| 10 | 6193.8 | n/a | 2071.9 | 2021.8 | 1169.3 | 856.0 | 856.0 |
| 11 | 2650.8 | n/a | 1186.2 | 1116.2 | 828.4 | 835.5 | 814.8 |
| **全部** | 2731.4 | n/a | 1676.5 | 1623.4 | 1117.1 | 1015.2 | 1009.0 |

> A2 是 GPU-only software baseline，此 sim path 把 prefill 与 decode 合成一个时间戳（`prefill_end==end`、`first_token=0`），所以它的 per-phase latency 记 `n/a`；其 makespan 见总览。


**每层 prefill attention 放置（HBM vs PIM），每个 agent 一个点**  来源 `dag_A5.json` / `dag_A6.json` 的 `prefill_attention_sides`（A1–A4 按构造 prefill 全在 GPU/HBM）

| tier(layer) | agents | A5 PIM | A5 HBM | A6 PIM | A6 HBM | A6 PIM% |
|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 1 | 0 | 1 | 0% |
| 1 | 1 | 1 | 0 | 0 | 1 | 0% |
| 2 | 1 | 1 | 0 | 0 | 1 | 0% |
| 3 | 1 | 1 | 0 | 0 | 1 | 0% |
| 4 | 1 | 1 | 0 | 1 | 0 | 100% |
| 5 | 1 | 1 | 0 | 1 | 0 | 100% |
| 6 | 1 | 1 | 0 | 1 | 0 | 100% |
| 7 | 1 | 1 | 0 | 1 | 0 | 100% |
| 8 | 1 | 1 | 0 | 1 | 0 | 100% |
| 9 | 1 | 1 | 0 | 1 | 0 | 100% |
| 10 | 1 | 1 | 0 | 1 | 0 | 100% |
| 11 | 1 | 1 | 0 | 0 | 1 | 0% |
| **全部** | 12 | 11 | 1 | 7 | 5 | 58% |

**每档 energy 与 average power**  来源 `dag_A*.json` 的 `energy_breakdown_nj`（by_event / by_class）、`energy_nj`、`makespan_s`、`link_bytes`（power = energy / makespan，不分 layer）

| rung | prefill E (mJ) | decode E (mJ) | total E (mJ) | avg power (W) | GPU | PIM | LINK | KV over link (GiB) |
|---|---|---|---|---|---|---|---|---|
| A1 | 73509961.6 | 2578608.8 | 76088570.5 | 2318 | 1904698.0 | 74182854.6 | 1017.94 | 11.39 |
| A2 | 277147.2 | 1823454.0 | 2100601.2 | 55 | 2050395.0 | 0.0 | 50206.20 | 562.00 |
| A3 | 227766.6 | 2595868.3 | 2823634.9 | 140 | 1413056.2 | 1409426.6 | 1152.10 | 12.90 |
| A3a | 227766.6 | 2564269.5 | 2792036.1 | 143 | 1413056.2 | 1377827.8 | 1152.10 | 12.90 |
| A4 | 227766.6 | 2717364.2 | 2945130.8 | 219 | 1413056.2 | 1530922.5 | 1152.10 | 12.90 |
| A5 | 1410433.6 | 1588732.5 | 2999166.2 | 245 | 1309974.3 | 1689011.0 | 180.86 | 2.02 |
| A6 | 1176862.9 | 1588732.5 | 2765595.4 | 227 | 1326460.2 | 1438845.6 | 289.61 | 3.24 |

---

## debate

**原始数据目录** `20260828-173206_workload_debate_d3r5k49_LLAMA3-8B_k2/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 298.8 | n/a | 46.6 | 46.6 | 46.6 | 46.6 | 46.6 |
| 1 | 1343.8 | n/a | 406.1 | 406.1 | 405.3 | 270.0 | 270.0 |
| 2 | 3580.4 | n/a | 743.7 | 743.7 | 742.3 | 399.5 | 399.5 |
| 3 | 7114.1 | n/a | 1183.7 | 1183.7 | 1181.4 | 528.7 | 528.7 |
| 4 | 12252.7 | n/a | 1584.8 | 1584.8 | 1581.4 | 701.4 | 701.4 |
| 5 | 42.0 | n/a | 26.5 | 26.5 | 26.5 | 44.1 | 26.5 |
| **全部** | 4613.2 | n/a | 745.1 | 745.1 | 743.6 | 367.7 | 366.6 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1130.5 | n/a | 1531.7 | 1531.7 | 1132.8 | 1114.1 | 1114.1 |
| 1 | 1176.2 | n/a | 3091.0 | 2384.1 | 1242.2 | 1206.4 | 1206.4 |
| 2 | 1247.8 | n/a | 4182.5 | 3470.0 | 1327.4 | 1274.9 | 1274.9 |
| 3 | 1337.4 | n/a | 5685.2 | 4991.4 | 1449.9 | 1380.0 | 1380.0 |
| 4 | 1445.1 | n/a | 7498.6 | 6766.6 | 1590.8 | 1504.1 | 1504.1 |
| 5 | 547.9 | n/a | 582.0 | 577.4 | 550.5 | 549.2 | 549.2 |
| **全部** | 1222.4 | n/a | 4159.3 | 3625.5 | 1298.7 | 1249.2 | 1249.2 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1429.4 | n/a | 1578.3 | 1578.3 | 1179.4 | 1160.7 | 1160.7 |
| 1 | 2520.1 | n/a | 3497.0 | 2790.2 | 1647.5 | 1476.4 | 1476.4 |
| 2 | 4828.2 | n/a | 4926.2 | 4213.7 | 2069.7 | 1674.5 | 1674.5 |
| 3 | 8451.5 | n/a | 6868.9 | 6175.1 | 2631.3 | 1908.7 | 1908.7 |
| 4 | 13697.8 | n/a | 9083.4 | 8351.4 | 3172.3 | 2205.4 | 2205.4 |
| 5 | 590.0 | n/a | 608.5 | 603.9 | 577.0 | 593.3 | 575.7 |
| **全部** | 5835.7 | n/a | 4904.4 | 4370.6 | 2042.3 | 1616.9 | 1615.8 |

> A2 是 GPU-only software baseline，此 sim path 把 prefill 与 decode 合成一个时间戳（`prefill_end==end`、`first_token=0`），所以它的 per-phase latency 记 `n/a`；其 makespan 见总览。


**每层 prefill attention 放置（HBM vs PIM），每个 agent 一个点**  来源 `dag_A5.json` / `dag_A6.json` 的 `prefill_attention_sides`（A1–A4 按构造 prefill 全在 GPU/HBM）

| tier(layer) | agents | A5 PIM | A5 HBM | A6 PIM | A6 HBM | A6 PIM% |
|---|---|---|---|---|---|---|
| 0 | 3 | 0 | 3 | 0 | 3 | 0% |
| 1 | 3 | 3 | 0 | 3 | 0 | 100% |
| 2 | 3 | 3 | 0 | 3 | 0 | 100% |
| 3 | 3 | 3 | 0 | 3 | 0 | 100% |
| 4 | 3 | 3 | 0 | 3 | 0 | 100% |
| 5 | 1 | 1 | 0 | 0 | 1 | 0% |
| **全部** | 16 | 13 | 3 | 12 | 4 | 75% |

**每档 energy 与 average power**  来源 `dag_A*.json` 的 `energy_breakdown_nj`（by_event / by_class）、`energy_nj`、`makespan_s`、`link_bytes`（power = energy / makespan，不分 layer）

| rung | prefill E (mJ) | decode E (mJ) | total E (mJ) | avg power (W) | GPU | PIM | LINK | KV over link (GiB) |
|---|---|---|---|---|---|---|---|---|
| A1 | 120844372.6 | 3633801.8 | 124478174.3 | 2839 | 1751913.2 | 122724832.6 | 1428.49 | 15.99 |
| A2 | 399386.8 | 2085836.5 | 2485223.3 | 40 | 2380263.7 | 0.0 | 104959.62 | 1174.90 |
| A3 | 295258.9 | 3768094.7 | 4063353.5 | 151 | 1045493.7 | 3016173.1 | 1686.72 | 18.88 |
| A3a | 295258.9 | 3619161.8 | 3914420.7 | 162 | 1045493.7 | 2867240.2 | 1686.72 | 18.88 |
| A4 | 295258.9 | 3962664.8 | 4257923.7 | 367 | 1045493.7 | 3210743.2 | 1686.72 | 18.88 |
| A5 | 1845197.4 | 2846226.7 | 4691424.1 | 508 | 897526.9 | 3793672.5 | 224.64 | 2.51 |
| A6 | 1837059.5 | 2846226.7 | 4683286.3 | 508 | 897988.6 | 3785073.3 | 224.34 | 2.51 |

---

## multi-source RAG

**原始数据目录** `20260828-130628_workload_multisource_rag_n12s96_LLAMA3-8B_k2/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 46562.0 | n/a | 7862.3 | 7862.3 | 7855.8 | 7454.0 | 7454.0 |
| **全部** | 46562.0 | n/a | 7862.3 | 7862.3 | 7855.8 | 7454.0 | 7454.0 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 755.9 | n/a | 17692.9 | 3943.3 | 865.1 | 708.8 | 708.8 |
| **全部** | 755.9 | n/a | 17692.9 | 3943.3 | 865.1 | 708.8 | 708.8 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 47317.9 | n/a | 25555.2 | 11805.6 | 8720.9 | 8162.9 | 8162.9 |
| **全部** | 47317.9 | n/a | 25555.2 | 11805.6 | 8720.9 | 8162.9 | 8162.9 |

> A2 是 GPU-only software baseline，此 sim path 把 prefill 与 decode 合成一个时间戳（`prefill_end==end`、`first_token=0`），所以它的 per-phase latency 记 `n/a`；其 makespan 见总览。


**每层 prefill attention 放置（HBM vs PIM），每个 agent 一个点**  来源 `dag_A5.json` / `dag_A6.json` 的 `prefill_attention_sides`（A1–A4 按构造 prefill 全在 GPU/HBM）

| tier(layer) | agents | A5 PIM | A5 HBM | A6 PIM | A6 HBM | A6 PIM% |
|---|---|---|---|---|---|---|
| 0 | 12 | 11 | 1 | 11 | 1 | 92% |
| **全部** | 12 | 11 | 1 | 11 | 1 | 92% |

**每档 energy 与 average power**  来源 `dag_A*.json` 的 `energy_breakdown_nj`（by_event / by_class）、`energy_nj`、`makespan_s`、`link_bytes`（power = energy / makespan，不分 layer）

| rung | prefill E (mJ) | decode E (mJ) | total E (mJ) | avg power (W) | GPU | PIM | LINK | KV over link (GiB) |
|---|---|---|---|---|---|---|---|---|
| A1 | 280923451.2 | 794271.3 | 281717722.6 | 3250 | 2439518.2 | 279274925.4 | 3278.92 | 36.70 |
| A2 | 634449.8 | 406653.6 | 1041103.3 | 44 | 1014512.4 | 0.0 | 26590.95 | 297.65 |
| A3 | 554011.5 | 1480061.2 | 2034072.7 | 75 | 624964.6 | 1407446.9 | 1661.09 | 18.59 |
| A3a | 554011.5 | 791554.0 | 1345565.5 | 103 | 624964.6 | 718939.8 | 1661.09 | 18.59 |
| A4 | 554011.5 | 873390.2 | 1427401.7 | 143 | 624964.6 | 800776.0 | 1661.09 | 18.59 |
| A5 | 1292527.6 | 251692.0 | 1544219.7 | 175 | 552770.7 | 991236.8 | 212.19 | 2.38 |
| A6 | 1292527.6 | 251692.0 | 1544219.7 | 175 | 552770.7 | 991236.8 | 212.19 | 2.38 |
