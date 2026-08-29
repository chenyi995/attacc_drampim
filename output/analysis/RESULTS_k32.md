# k=32 阶梯结果（LLAMA3-8B，MQ PIM @ 1.30 GHz）

由 `make_results_tables.py` 自动生成。k 是 reuse policy 对每个 shifted chunk 重算的 token 数（此处 32）。每个 workload 取最新的完整 run；某 workload 在此 k 下不足 7 档则跳过。**所有数值均为仿真实测**，来源文件与字段在每处标注。

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
| star-repair | 53.25 | 74.47 | 141.40 | 32.46 | 18.85 | 14.88 | 14.87 |
| pipeline-repair | 32.83 | 39.47 | 30.68 | 21.14 | 15.20 | 13.71 | 13.52 |
| debate | 43.84 | 64.03 | 56.45 | 25.20 | 12.94 | 10.21 | 10.20 |
| multi-source RAG | 86.69 | 33.22 | 109.98 | 23.13 | 20.08 | 17.89 | 17.89 |

**总能量 total energy (kJ)**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 148.0 | 3.2 | 11.1 | 4.9 | 5.1 | 7.1 | 7.0 |
| pipeline-repair | 76.1 | 2.2 | 3.3 | 2.9 | 3.1 | 3.8 | 3.5 |
| debate | 124.5 | 2.6 | 5.8 | 4.0 | 4.4 | 5.3 | 5.3 |
| multi-source RAG | 281.7 | 1.6 | 6.9 | 2.0 | 2.1 | 6.6 | 6.6 |

**平均功率 average power (W)**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 2779 | 42 | 78 | 149 | 272 | 474 | 473 |
| pipeline-repair | 2318 | 56 | 108 | 138 | 203 | 280 | 258 |
| debate | 2839 | 40 | 102 | 159 | 338 | 522 | 522 |
| multi-source RAG | 3250 | 49 | 63 | 87 | 105 | 367 | 367 |

**prefill attention 落 PIM 的 agent 占比**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 0% | 0% | 0% | 0% | 0% | 95% | 80% |
| pipeline-repair | 0% | 0% | 0% | 0% | 0% | 92% | 58% |
| debate | 0% | 0% | 0% | 0% | 0% | 81% | 75% |
| multi-source RAG | 0% | 0% | 0% | 0% | 0% | 92% | 92% |

**A6（Fugue）相对 A1、A2 的加速**

| workload | A6 vs A1 | A6 vs A2 | energy A6 vs A1 |
|---|---|---|---|
| star-repair | 3.6x | 5.0x | 21x |
| pipeline-repair | 2.4x | 2.9x | 22x |
| debate | 4.3x | 6.3x | 23x |
| multi-source RAG | 4.8x | 1.9x | 43x |

---

## star-repair

**原始数据目录** `20260828-235623_workload_star_repair_r5w3k47_LLAMA3-8B_k32/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 155.2 | n/a | 147.9 | 147.9 | 147.8 | 147.8 | 147.8 |
| 1 | 301.1 | n/a | 59.1 | 59.1 | 58.9 | 68.0 | 58.9 |
| 2 | 657.8 | n/a | 412.8 | 412.8 | 412.5 | 259.4 | 259.4 |
| 3 | 1151.0 | n/a | 155.8 | 155.8 | 155.1 | 136.6 | 136.6 |
| 4 | 1764.4 | n/a | 811.2 | 811.2 | 810.5 | 408.5 | 408.5 |
| 5 | 3135.9 | n/a | 411.4 | 411.4 | 410.2 | 315.9 | 315.9 |
| 6 | 3386.2 | n/a | 1192.3 | 1192.3 | 1191.2 | 563.6 | 563.6 |
| 7 | 6070.4 | n/a | 839.6 | 839.6 | 837.6 | 548.3 | 548.3 |
| 8 | 5857.6 | n/a | 1724.3 | 1724.3 | 1722.6 | 801.4 | 801.4 |
| 9 | 10820.2 | n/a | 1493.5 | 1493.5 | 1490.5 | 878.3 | 878.3 |
| **全部** | 3812.9 | n/a | 658.3 | 658.3 | 657.1 | 401.1 | 399.7 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 550.1 | n/a | 616.0 | 616.0 | 550.3 | 547.4 | 547.4 |
| 1 | 1130.4 | n/a | 9279.5 | 1626.1 | 1193.0 | 1174.0 | 1174.0 |
| 2 | 558.3 | n/a | 1102.0 | 756.3 | 567.3 | 559.1 | 559.1 |
| 3 | 1176.0 | n/a | 16918.9 | 2350.2 | 1282.3 | 1247.0 | 1247.0 |
| 4 | 569.6 | n/a | 1286.7 | 934.7 | 581.2 | 565.1 | 565.1 |
| 5 | 1239.2 | n/a | 25157.6 | 3386.2 | 1418.6 | 1366.3 | 1366.3 |
| 6 | 584.1 | n/a | 1518.6 | 1173.2 | 600.7 | 573.6 | 573.6 |
| 7 | 1323.9 | n/a | 33436.6 | 4833.8 | 1564.6 | 1496.3 | 1496.3 |
| 8 | 601.9 | n/a | 1817.6 | 1456.8 | 623.8 | 583.5 | 583.5 |
| 9 | 1428.3 | n/a | 41001.9 | 6494.7 | 1715.1 | 1630.8 | 1630.8 |
| **全部** | 1087.9 | n/a | 19186.2 | 3050.5 | 1222.2 | 1178.6 | 1178.6 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 705.3 | n/a | 763.8 | 763.8 | 698.0 | 695.1 | 695.1 |
| 1 | 1431.5 | n/a | 9338.6 | 1685.2 | 1251.9 | 1242.0 | 1232.9 |
| 2 | 1216.1 | n/a | 1514.8 | 1169.1 | 979.8 | 818.5 | 818.5 |
| 3 | 2326.9 | n/a | 17074.6 | 2505.9 | 1437.5 | 1383.6 | 1383.6 |
| 4 | 2334.0 | n/a | 2097.9 | 1745.9 | 1391.7 | 973.5 | 973.5 |
| 5 | 4375.1 | n/a | 25569.0 | 3797.6 | 1828.8 | 1682.2 | 1682.2 |
| 6 | 3970.3 | n/a | 2710.9 | 2365.5 | 1791.9 | 1137.2 | 1137.2 |
| 7 | 7394.3 | n/a | 34276.2 | 5673.5 | 2402.2 | 2044.6 | 2044.6 |
| 8 | 6459.4 | n/a | 3541.9 | 3181.1 | 2346.4 | 1384.9 | 1384.9 |
| 9 | 12248.5 | n/a | 42495.4 | 7988.2 | 3205.6 | 2509.1 | 2509.1 |
| **全部** | 4900.7 | n/a | 19844.6 | 3708.8 | 1879.3 | 1579.7 | 1578.3 |

> A2 是 GPU-only software baseline，此 sim path 把 prefill 与 decode 合成一个时间戳（`prefill_end==end`、`first_token=0`），所以它的 per-phase latency 记 `n/a`；其 makespan 见总览。


**每层 prefill attention 放置（HBM vs PIM），每个 agent 一个点**  来源 `dag_A5.json` / `dag_A6.json` 的 `prefill_attention_sides`（A1–A4 按构造 prefill 全在 GPU/HBM）

| tier(layer) | agents | A5 PIM | A5 HBM | A6 PIM | A6 HBM | A6 PIM% |
|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 1 | 0 | 1 | 0% |
| 1 | 3 | 3 | 0 | 0 | 3 | 0% |
| 2 | 1 | 1 | 0 | 1 | 0 | 100% |
| 3 | 3 | 3 | 0 | 3 | 0 | 100% |
| 4 | 1 | 1 | 0 | 1 | 0 | 100% |
| 5 | 3 | 3 | 0 | 3 | 0 | 100% |
| 6 | 1 | 1 | 0 | 1 | 0 | 100% |
| 7 | 3 | 3 | 0 | 3 | 0 | 100% |
| 8 | 1 | 1 | 0 | 1 | 0 | 100% |
| 9 | 3 | 3 | 0 | 3 | 0 | 100% |
| **全部** | 20 | 19 | 1 | 16 | 4 | 80% |

**每档 energy 与 average power**  来源 `dag_A*.json` 的 `energy_breakdown_nj`（by_event / by_class）、`energy_nj`、`makespan_s`、`link_bytes`（power = energy / makespan，不分 layer）

| rung | prefill E (mJ) | decode E (mJ) | total E (mJ) | avg power (W) | GPU | PIM | LINK | KV over link (GiB) |
|---|---|---|---|---|---|---|---|---|
| A1 | 143787217.8 | 4189573.0 | 147976790.8 | 2779 | 2261738.2 | 145713279.9 | 1772.68 | 19.84 |
| A2 | 688454.5 | 2476456.3 | 3164910.8 | 42 | 3049885.9 | 0.0 | 115024.89 | 1287.56 |
| A3 | 574700.7 | 10491706.7 | 11066407.4 | 78 | 1588090.8 | 9476132.9 | 2182.40 | 24.43 |
| A3a | 574700.7 | 4277125.8 | 4851826.5 | 149 | 1588090.8 | 3261553.3 | 2182.40 | 24.43 |
| A4 | 574700.7 | 4557182.6 | 5131883.3 | 272 | 1588090.8 | 3541610.1 | 2182.40 | 24.43 |
| A5 | 3965423.6 | 3096918.0 | 7062341.5 | 474 | 1278770.9 | 5783182.3 | 388.31 | 4.35 |
| A6 | 3940485.5 | 3096918.0 | 7037403.4 | 473 | 1280759.6 | 5756217.6 | 426.09 | 4.77 |

---

## pipeline-repair

**原始数据目录** `20260828-200549_workload_pipeline_repair_c5k50_LLAMA3-8B_k32/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 70.8 | n/a | 63.0 | 63.0 | 63.0 | 63.0 | 63.0 |
| 1 | 144.2 | n/a | 78.1 | 78.1 | 78.0 | 100.4 | 78.0 |
| 2 | 230.4 | n/a | 96.9 | 96.9 | 96.8 | 150.8 | 96.8 |
| 3 | 507.1 | n/a | 164.7 | 164.7 | 164.4 | 171.9 | 164.4 |
| 4 | 744.5 | n/a | 182.4 | 182.4 | 182.0 | 155.1 | 155.1 |
| 5 | 1178.2 | n/a | 311.8 | 311.8 | 311.2 | 243.3 | 243.3 |
| 6 | 1824.2 | n/a | 404.6 | 404.6 | 403.9 | 284.6 | 284.6 |
| 7 | 2484.5 | n/a | 539.8 | 539.8 | 538.9 | 355.5 | 355.5 |
| 8 | 3344.6 | n/a | 574.4 | 574.4 | 573.3 | 323.6 | 323.6 |
| 9 | 4325.0 | n/a | 836.1 | 836.1 | 834.7 | 493.5 | 493.5 |
| 10 | 5591.9 | n/a | 877.2 | 877.2 | 875.6 | 474.3 | 474.3 |
| 11 | 2086.0 | n/a | 404.6 | 404.6 | 404.1 | 507.3 | 404.1 |
| **全部** | 1877.6 | n/a | 377.8 | 377.8 | 377.2 | 276.9 | 261.4 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1101.5 | n/a | 1182.7 | 1182.7 | 1102.0 | 1098.3 | 1098.3 |
| 1 | 1104.5 | n/a | 1653.5 | 1246.5 | 1118.6 | 1112.5 | 1112.5 |
| 2 | 551.2 | n/a | 842.6 | 641.2 | 562.7 | 558.4 | 558.4 |
| 3 | 1118.6 | n/a | 2453.5 | 1490.3 | 1150.7 | 1135.6 | 1135.6 |
| 4 | 559.6 | n/a | 1111.1 | 778.9 | 573.2 | 563.5 | 563.5 |
| 5 | 1135.5 | n/a | 3080.0 | 1787.9 | 1178.9 | 1151.2 | 1151.2 |
| 6 | 570.8 | n/a | 1422.2 | 960.3 | 590.6 | 573.2 | 573.2 |
| 7 | 1162.1 | n/a | 3802.0 | 2217.7 | 1219.4 | 1172.0 | 1172.0 |
| 8 | 585.1 | n/a | 1810.3 | 1200.9 | 610.8 | 582.7 | 582.7 |
| 9 | 1189.9 | n/a | 4527.6 | 2673.5 | 1267.7 | 1199.3 | 1199.3 |
| 10 | 602.0 | n/a | 2201.8 | 1465.1 | 636.1 | 595.1 | 595.1 |
| 11 | 564.8 | n/a | 1922.1 | 876.9 | 602.9 | 589.3 | 589.3 |
| **全部** | 853.8 | n/a | 2167.4 | 1376.8 | 884.5 | 860.9 | 860.9 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1172.2 | n/a | 1245.7 | 1245.7 | 1165.0 | 1161.3 | 1161.3 |
| 1 | 1248.7 | n/a | 1731.6 | 1324.6 | 1196.7 | 1212.9 | 1190.5 |
| 2 | 781.6 | n/a | 939.5 | 738.2 | 659.5 | 709.3 | 655.2 |
| 3 | 1625.7 | n/a | 2618.1 | 1654.9 | 1315.1 | 1307.5 | 1299.9 |
| 4 | 1304.1 | n/a | 1293.4 | 961.2 | 755.2 | 718.6 | 718.6 |
| 5 | 2313.8 | n/a | 3391.8 | 2099.8 | 1490.2 | 1394.5 | 1394.5 |
| 6 | 2395.0 | n/a | 1826.9 | 1364.9 | 994.5 | 857.9 | 857.9 |
| 7 | 3646.6 | n/a | 4341.8 | 2757.5 | 1758.3 | 1527.5 | 1527.5 |
| 8 | 3929.7 | n/a | 2384.7 | 1775.3 | 1184.1 | 906.4 | 906.4 |
| 9 | 5514.9 | n/a | 5363.7 | 3509.6 | 2102.5 | 1692.7 | 1692.7 |
| 10 | 6193.8 | n/a | 3079.0 | 2342.4 | 1511.7 | 1069.4 | 1069.4 |
| 11 | 2650.8 | n/a | 2326.7 | 1281.5 | 1007.0 | 1096.6 | 993.4 |
| **全部** | 2731.4 | n/a | 2545.2 | 1754.6 | 1261.6 | 1137.9 | 1122.3 |

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
| A2 | 378465.1 | 1823454.0 | 2201919.1 | 56 | 2151712.9 | 0.0 | 50206.20 | 562.00 |
| A3 | 329100.3 | 2998400.9 | 3327501.2 | 108 | 1514374.1 | 1811959.1 | 1167.88 | 13.07 |
| A3a | 329100.3 | 2591037.6 | 2920137.9 | 138 | 1514374.1 | 1404595.9 | 1167.88 | 13.07 |
| A4 | 329100.3 | 2757438.8 | 3086539.1 | 203 | 1514374.1 | 1570997.1 | 1167.88 | 13.07 |
| A5 | 2216088.2 | 1628807.1 | 3844895.3 | 280 | 1355190.9 | 2489460.4 | 244.00 | 2.73 |
| A6 | 1856953.0 | 1628807.1 | 3485760.1 | 258 | 1379366.7 | 2106054.9 | 338.52 | 3.79 |

---

## debate

**原始数据目录** `20260829-025941_workload_debate_d3r5k49_LLAMA3-8B_k32/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 298.8 | n/a | 46.6 | 46.6 | 46.6 | 46.6 | 46.6 |
| 1 | 1343.8 | n/a | 483.7 | 483.7 | 483.0 | 362.4 | 362.4 |
| 2 | 3580.4 | n/a | 887.3 | 887.3 | 885.9 | 520.9 | 520.9 |
| 3 | 7114.1 | n/a | 1413.2 | 1413.2 | 1411.0 | 677.2 | 677.2 |
| 4 | 12252.7 | n/a | 1919.1 | 1919.1 | 1915.7 | 897.5 | 897.5 |
| 5 | 42.0 | n/a | 28.3 | 28.3 | 28.3 | 47.3 | 28.3 |
| **全部** | 4613.2 | n/a | 892.4 | 892.4 | 890.9 | 472.6 | 471.4 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1130.5 | n/a | 1526.0 | 1526.0 | 1132.8 | 1114.1 | 1114.1 |
| 1 | 1176.2 | n/a | 10297.1 | 2386.5 | 1282.0 | 1246.2 | 1246.2 |
| 2 | 1247.8 | n/a | 11400.8 | 3468.3 | 1367.2 | 1314.7 | 1314.7 |
| 3 | 1337.4 | n/a | 12590.1 | 4980.2 | 1489.7 | 1419.7 | 1419.7 |
| 4 | 1445.1 | n/a | 14335.9 | 6723.8 | 1630.6 | 1543.8 | 1543.8 |
| 5 | 547.9 | n/a | 638.8 | 577.4 | 551.2 | 550.0 | 550.0 |
| **全部** | 1222.4 | n/a | 9443.0 | 3614.5 | 1328.6 | 1279.1 | 1279.1 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1429.4 | n/a | 1572.6 | 1572.6 | 1179.4 | 1160.7 | 1160.7 |
| 1 | 2520.1 | n/a | 10780.7 | 2870.1 | 1764.9 | 1608.6 | 1608.6 |
| 2 | 4828.2 | n/a | 12288.1 | 4355.6 | 2253.1 | 1835.6 | 1835.6 |
| 3 | 8451.5 | n/a | 14003.3 | 6393.4 | 2900.7 | 2096.9 | 2096.9 |
| 4 | 13697.8 | n/a | 16254.9 | 8642.9 | 3546.3 | 2441.3 | 2441.3 |
| 5 | 590.0 | n/a | 667.2 | 605.7 | 579.5 | 597.2 | 578.3 |
| **全部** | 5835.7 | n/a | 10335.4 | 4506.9 | 2219.5 | 1751.6 | 1750.5 |

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
| A2 | 476420.1 | 2085836.5 | 2562256.6 | 40 | 2457297.0 | 0.0 | 104959.62 | 1174.90 |
| A3 | 372303.4 | 5382119.7 | 5754423.2 | 102 | 1122527.0 | 4630197.8 | 1698.01 | 19.01 |
| A3a | 372303.4 | 3640916.8 | 4013220.2 | 159 | 1122527.0 | 2888995.2 | 1698.01 | 19.01 |
| A4 | 372303.4 | 4000467.7 | 4372771.2 | 338 | 1122527.0 | 3248546.1 | 1698.01 | 19.01 |
| A5 | 2446130.4 | 2884029.6 | 5330160.1 | 522 | 930389.1 | 4399501.1 | 269.79 | 3.02 |
| A6 | 2437197.8 | 2884029.6 | 5321227.4 | 522 | 930883.3 | 4390075.0 | 268.99 | 3.01 |

---

## multi-source RAG

**原始数据目录** `20260828-174235_workload_multisource_rag_n12s96_LLAMA3-8B_k32/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 46562.0 | n/a | 12778.0 | 12778.0 | 12772.1 | 11641.9 | 11641.9 |
| **全部** | 46562.0 | n/a | 12778.0 | 12778.0 | 12772.1 | 11641.9 | 11641.9 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 755.9 | n/a | 89688.4 | 4194.6 | 1204.9 | 1048.6 | 1048.6 |
| **全部** | 755.9 | n/a | 89688.4 | 4194.6 | 1204.9 | 1048.6 | 1048.6 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 47317.9 | n/a | 102466.4 | 16972.6 | 13977.0 | 12690.5 | 12690.5 |
| **全部** | 47317.9 | n/a | 102466.4 | 16972.6 | 13977.0 | 12690.5 | 12690.5 |

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
| A2 | 1233837.3 | 406653.6 | 1640490.8 | 49 | 1613899.9 | 0.0 | 26590.95 | 297.65 |
| A3 | 1153484.5 | 5744948.5 | 6898433.0 | 63 | 1224352.1 | 5672333.3 | 1746.56 | 19.55 |
| A3a | 1153484.5 | 864459.5 | 2017944.0 | 87 | 1224352.1 | 791845.3 | 1746.56 | 19.55 |
| A4 | 1153484.5 | 947632.9 | 2101117.4 | 105 | 1224352.1 | 875018.7 | 1746.56 | 19.55 |
| A5 | 6239579.7 | 325934.8 | 6565514.5 | 367 | 797383.7 | 5767576.6 | 554.06 | 6.20 |
| A6 | 6239579.7 | 325934.8 | 6565514.5 | 367 | 797383.7 | 5767576.6 | 554.06 | 6.20 |
