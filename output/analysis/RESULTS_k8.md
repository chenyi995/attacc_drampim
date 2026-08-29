# k=8 阶梯结果（LLAMA3-8B，MQ PIM @ 1.30 GHz）

由 `make_results_tables.py` 自动生成。k 是 reuse policy 对每个 shifted chunk 重算的 token 数（此处 8）。每个 workload 取最新的完整 run；某 workload 在此 k 下不足 7 档则跳过。**所有数值均为仿真实测**，来源文件与字段在每处标注。

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
| star-repair | 53.25 | 70.96 | 66.93 | 28.57 | 14.86 | 11.93 | 11.93 |
| pipeline-repair | 32.83 | 38.23 | 22.32 | 19.81 | 13.80 | 12.49 | 12.40 |
| debate | 43.84 | 63.09 | 34.83 | 24.18 | 11.86 | 9.41 | 9.39 |
| multi-source RAG | 86.69 | 25.40 | 62.55 | 15.08 | 11.98 | 10.07 | 10.07 |

**总能量 total energy (kJ)**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 148.0 | 2.9 | 6.5 | 4.5 | 4.8 | 5.2 | 5.2 |
| pipeline-repair | 76.1 | 2.1 | 2.9 | 2.8 | 3.0 | 3.2 | 2.9 |
| debate | 124.5 | 2.5 | 4.5 | 3.9 | 4.3 | 4.8 | 4.8 |
| multi-source RAG | 281.7 | 1.2 | 4.0 | 1.5 | 1.6 | 2.5 | 2.5 |

**平均功率 average power (W)**

| workload | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| star-repair | 2779 | 42 | 97 | 159 | 323 | 439 | 439 |
| pipeline-repair | 2318 | 55 | 131 | 142 | 215 | 253 | 234 |
| debate | 2839 | 40 | 128 | 163 | 361 | 512 | 512 |
| multi-source RAG | 3250 | 46 | 63 | 98 | 130 | 247 | 247 |

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
| star-repair | 4.5x | 5.9x | 28x |
| pipeline-repair | 2.6x | 3.1x | 26x |
| debate | 4.7x | 6.7x | 26x |
| multi-source RAG | 8.6x | 2.5x | 114x |

---

## star-repair

**原始数据目录** `20260828-173145_workload_star_repair_r5w3k47_LLAMA3-8B_k8/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 155.2 | n/a | 147.9 | 147.9 | 147.8 | 147.8 | 147.8 |
| 1 | 301.1 | n/a | 33.6 | 33.6 | 33.3 | 32.1 | 31.6 |
| 2 | 657.8 | n/a | 381.9 | 381.9 | 381.5 | 216.8 | 216.8 |
| 3 | 1151.0 | n/a | 52.1 | 52.1 | 51.5 | 34.4 | 34.4 |
| 4 | 1764.4 | n/a | 753.4 | 753.4 | 752.7 | 354.1 | 354.1 |
| 5 | 3135.9 | n/a | 117.5 | 117.5 | 116.3 | 72.3 | 72.3 |
| 6 | 3386.2 | n/a | 1101.7 | 1101.7 | 1100.6 | 497.6 | 497.6 |
| 7 | 6070.4 | n/a | 225.8 | 225.8 | 223.6 | 123.6 | 123.6 |
| 8 | 5857.6 | n/a | 1592.9 | 1592.9 | 1591.3 | 716.5 | 716.5 |
| 9 | 10820.2 | n/a | 393.9 | 393.9 | 390.8 | 194.8 | 194.8 |
| **全部** | 3812.9 | n/a | 322.3 | 322.3 | 321.0 | 165.2 | 165.2 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 550.1 | n/a | 616.1 | 616.1 | 550.3 | 547.4 | 547.4 |
| 1 | 1130.4 | n/a | 4350.4 | 1612.7 | 1161.5 | 1142.5 | 1142.5 |
| 2 | 558.3 | n/a | 850.1 | 754.1 | 562.1 | 553.9 | 553.9 |
| 3 | 1176.0 | n/a | 7367.4 | 2309.6 | 1223.0 | 1187.6 | 1187.6 |
| 4 | 569.6 | n/a | 1023.7 | 932.7 | 576.0 | 559.8 | 559.8 |
| 5 | 1239.2 | n/a | 10845.1 | 3305.3 | 1321.5 | 1269.2 | 1269.2 |
| 6 | 584.1 | n/a | 1264.7 | 1170.8 | 595.5 | 568.4 | 568.4 |
| 7 | 1323.9 | n/a | 14925.7 | 4729.6 | 1443.3 | 1375.0 | 1375.0 |
| 8 | 601.9 | n/a | 1549.7 | 1454.8 | 618.5 | 578.3 | 578.3 |
| 9 | 1428.3 | n/a | 18668.2 | 6363.7 | 1570.5 | 1486.2 | 1486.2 |
| **全部** | 1087.9 | n/a | 8688.7 | 2994.6 | 1153.1 | 1109.5 | 1109.5 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 705.3 | n/a | 763.9 | 763.9 | 698.0 | 695.1 | 695.1 |
| 1 | 1431.5 | n/a | 4384.0 | 1646.3 | 1194.8 | 1174.5 | 1174.1 |
| 2 | 1216.1 | n/a | 1232.0 | 1136.0 | 943.6 | 770.7 | 770.7 |
| 3 | 2326.9 | n/a | 7419.5 | 2361.7 | 1274.4 | 1222.1 | 1222.1 |
| 4 | 2334.0 | n/a | 1777.1 | 1686.1 | 1328.7 | 913.9 | 913.9 |
| 5 | 4375.1 | n/a | 10962.6 | 3422.9 | 1437.8 | 1341.5 | 1341.5 |
| 6 | 3970.3 | n/a | 2366.4 | 2272.5 | 1696.1 | 1066.0 | 1066.0 |
| 7 | 7394.3 | n/a | 15151.5 | 4955.4 | 1666.9 | 1498.6 | 1498.6 |
| 8 | 6459.4 | n/a | 3142.6 | 3047.7 | 2209.8 | 1294.8 | 1294.8 |
| 9 | 12248.5 | n/a | 19062.1 | 6757.6 | 1961.3 | 1680.9 | 1680.9 |
| **全部** | 4900.7 | n/a | 9011.1 | 3316.9 | 1474.1 | 1274.7 | 1274.6 |

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
| A2 | 469218.8 | 2476456.3 | 2945675.1 | 42 | 2830650.2 | 0.0 | 115024.89 | 1287.56 |
| A3 | 355433.0 | 6162881.9 | 6518315.0 | 97 | 1368855.1 | 5147308.9 | 2150.47 | 24.07 |
| A3a | 355433.0 | 4182612.4 | 4538045.5 | 159 | 1368855.1 | 3167039.9 | 2150.47 | 24.07 |
| A4 | 355433.0 | 4451844.8 | 4807277.8 | 323 | 1368855.1 | 3436272.3 | 2150.47 | 24.07 |
| A5 | 2245127.3 | 2991580.1 | 5236707.5 | 439 | 1186890.5 | 4049556.3 | 260.59 | 2.92 |
| A6 | 2238469.6 | 2991580.1 | 5230049.8 | 439 | 1187466.8 | 4042309.2 | 273.71 | 3.06 |

---

## pipeline-repair

**原始数据目录** `20260828-170034_workload_pipeline_repair_c5k50_LLAMA3-8B_k8/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 70.8 | n/a | 63.0 | 63.0 | 63.0 | 63.0 | 63.0 |
| 1 | 144.2 | n/a | 71.1 | 71.1 | 71.0 | 88.1 | 71.0 |
| 2 | 230.4 | n/a | 88.2 | 88.2 | 88.0 | 133.6 | 88.0 |
| 3 | 507.1 | n/a | 125.3 | 125.3 | 125.0 | 118.8 | 125.0 |
| 4 | 744.5 | n/a | 150.4 | 150.4 | 150.0 | 119.2 | 119.2 |
| 5 | 1178.2 | n/a | 227.1 | 227.1 | 226.5 | 158.9 | 158.9 |
| 6 | 1824.2 | n/a | 328.6 | 328.6 | 327.9 | 217.4 | 217.4 |
| 7 | 2484.5 | n/a | 369.6 | 369.6 | 368.6 | 219.7 | 219.7 |
| 8 | 3344.6 | n/a | 422.7 | 422.7 | 421.5 | 220.1 | 220.1 |
| 9 | 4325.0 | n/a | 548.6 | 548.6 | 547.2 | 293.3 | 293.3 |
| 10 | 5591.9 | n/a | 613.9 | 613.9 | 612.2 | 309.9 | 309.9 |
| 11 | 2086.0 | n/a | 277.0 | 277.0 | 276.5 | 306.1 | 276.5 |
| **全部** | 1877.6 | n/a | 273.8 | 273.8 | 273.1 | 187.3 | 180.2 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1101.5 | n/a | 1182.7 | 1182.7 | 1102.0 | 1098.3 | 1098.3 |
| 1 | 1104.5 | n/a | 1347.4 | 1243.9 | 1113.3 | 1107.2 | 1107.2 |
| 2 | 551.2 | n/a | 689.6 | 639.9 | 560.0 | 555.8 | 555.8 |
| 3 | 1118.6 | n/a | 1744.0 | 1483.9 | 1136.6 | 1121.4 | 1121.4 |
| 4 | 559.6 | n/a | 862.1 | 776.8 | 568.3 | 558.7 | 558.7 |
| 5 | 1135.5 | n/a | 2108.2 | 1777.7 | 1160.2 | 1132.5 | 1132.5 |
| 6 | 570.8 | n/a | 1082.9 | 957.3 | 583.8 | 566.5 | 566.5 |
| 7 | 1162.1 | n/a | 2622.8 | 2200.4 | 1195.9 | 1148.5 | 1148.5 |
| 8 | 585.1 | n/a | 1358.3 | 1196.0 | 601.9 | 573.8 | 573.8 |
| 9 | 1189.9 | n/a | 3153.1 | 2653.6 | 1237.0 | 1168.5 | 1168.5 |
| 10 | 602.0 | n/a | 1650.1 | 1458.3 | 625.3 | 584.3 | 584.3 |
| 11 | 564.8 | n/a | 1132.1 | 864.7 | 586.4 | 572.8 | 572.8 |
| **全部** | 853.8 | n/a | 1577.8 | 1369.6 | 872.6 | 849.0 | 849.0 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1172.2 | n/a | 1245.7 | 1245.7 | 1165.0 | 1161.3 | 1161.3 |
| 1 | 1248.7 | n/a | 1418.5 | 1315.0 | 1184.3 | 1195.3 | 1178.2 |
| 2 | 781.6 | n/a | 777.8 | 728.1 | 648.1 | 689.4 | 643.8 |
| 3 | 1625.7 | n/a | 1869.3 | 1609.2 | 1261.6 | 1240.2 | 1246.4 |
| 4 | 1304.1 | n/a | 1012.5 | 927.2 | 718.3 | 677.9 | 677.9 |
| 5 | 2313.8 | n/a | 2335.3 | 2004.8 | 1386.8 | 1291.4 | 1291.4 |
| 6 | 2395.0 | n/a | 1411.5 | 1285.9 | 911.7 | 783.8 | 783.8 |
| 7 | 3646.6 | n/a | 2992.4 | 2570.0 | 1564.5 | 1368.2 | 1368.2 |
| 8 | 3929.7 | n/a | 1781.0 | 1618.7 | 1023.4 | 794.0 | 794.0 |
| 9 | 5514.9 | n/a | 3701.7 | 3202.2 | 1784.2 | 1461.8 | 1461.8 |
| 10 | 6193.8 | n/a | 2264.0 | 2072.2 | 1237.6 | 894.2 | 894.2 |
| 11 | 2650.8 | n/a | 1409.1 | 1141.7 | 862.9 | 878.9 | 849.3 |
| **全部** | 2731.4 | n/a | 1851.6 | 1643.4 | 1145.7 | 1036.4 | 1029.2 |

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
| A2 | 297483.9 | 1823454.0 | 2120937.9 | 55 | 2070731.7 | 0.0 | 50206.20 | 562.00 |
| A3 | 248106.4 | 2665083.6 | 2913190.0 | 131 | 1433392.9 | 1478641.8 | 1155.25 | 12.93 |
| A3a | 248106.4 | 2564345.8 | 2812452.2 | 142 | 1433392.9 | 1377904.1 | 1155.25 | 12.93 |
| A4 | 248106.4 | 2725233.2 | 2973339.6 | 215 | 1433392.9 | 1538791.5 | 1155.25 | 12.93 |
| A5 | 1568706.4 | 1596601.5 | 3165307.9 | 253 | 1319131.4 | 1845983.1 | 193.49 | 2.17 |
| A6 | 1310866.5 | 1596601.5 | 2907468.0 | 234 | 1337164.7 | 1570003.9 | 299.39 | 3.35 |

---

## debate

**原始数据目录** `20260829-001648_workload_debate_d3r5k49_LLAMA3-8B_k8/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 298.8 | n/a | 46.6 | 46.6 | 46.6 | 46.6 | 46.6 |
| 1 | 1343.8 | n/a | 421.4 | 421.4 | 420.7 | 285.7 | 285.7 |
| 2 | 3580.4 | n/a | 772.1 | 772.1 | 770.7 | 421.0 | 421.0 |
| 3 | 7114.1 | n/a | 1229.1 | 1229.1 | 1226.8 | 555.6 | 555.6 |
| 4 | 12252.7 | n/a | 1651.6 | 1651.6 | 1648.2 | 738.5 | 738.5 |
| 5 | 42.0 | n/a | 26.9 | 26.9 | 26.8 | 44.7 | 26.8 |
| **全部** | 4613.2 | n/a | 774.3 | 774.3 | 772.9 | 386.7 | 385.6 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1130.5 | n/a | 1526.1 | 1526.1 | 1132.8 | 1114.1 | 1114.1 |
| 1 | 1176.2 | n/a | 4954.2 | 2367.7 | 1249.4 | 1213.6 | 1213.6 |
| 2 | 1247.8 | n/a | 6204.0 | 3448.6 | 1334.5 | 1282.1 | 1282.1 |
| 3 | 1337.4 | n/a | 7578.5 | 4960.2 | 1457.1 | 1387.1 | 1387.1 |
| 4 | 1445.1 | n/a | 9329.9 | 6703.4 | 1598.0 | 1511.2 | 1511.2 |
| 5 | 547.9 | n/a | 594.9 | 577.0 | 550.5 | 549.2 | 549.2 |
| **全部** | 1222.4 | n/a | 5585.8 | 3599.7 | 1304.1 | 1254.6 | 1254.6 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 1429.4 | n/a | 1572.8 | 1572.8 | 1179.4 | 1160.7 | 1160.7 |
| 1 | 2520.1 | n/a | 5375.6 | 2789.1 | 1670.1 | 1499.3 | 1499.3 |
| 2 | 4828.2 | n/a | 6976.1 | 4220.7 | 2105.2 | 1703.1 | 1703.1 |
| 3 | 8451.5 | n/a | 8807.6 | 6189.3 | 2683.9 | 1942.7 | 1942.7 |
| 4 | 13697.8 | n/a | 10981.6 | 8355.0 | 3246.2 | 2249.7 | 2249.7 |
| 5 | 590.0 | n/a | 621.8 | 603.9 | 577.3 | 593.9 | 576.1 |
| **全部** | 5835.7 | n/a | 6360.2 | 4374.0 | 2077.0 | 1641.3 | 1640.2 |

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
| A2 | 414678.9 | 2085836.5 | 2500515.4 | 40 | 2395555.8 | 0.0 | 104959.62 | 1174.90 |
| A3 | 310553.2 | 4159251.2 | 4469804.4 | 128 | 1060785.8 | 3407329.5 | 1688.98 | 18.91 |
| A3a | 310553.2 | 3619616.3 | 3930169.5 | 163 | 1060785.8 | 2867694.7 | 1688.98 | 18.91 |
| A4 | 310553.2 | 3969924.3 | 4280477.5 | 361 | 1060785.8 | 3218002.7 | 1688.98 | 18.91 |
| A5 | 1964198.5 | 2853486.2 | 4817684.6 | 512 | 904045.0 | 3913406.0 | 233.67 | 2.62 |
| A6 | 1955940.6 | 2853486.2 | 4809426.8 | 512 | 904513.3 | 3904680.2 | 233.27 | 2.61 |

---

## multi-source RAG

**原始数据目录** `20260828-150654_workload_multisource_rag_n12s96_LLAMA3-8B_k8/`。下面各表的列 **A1…A6 分别取自该目录下的 `dag_A1.json` … `dag_A6.json`**。


**每层 latency [prefill]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 46562.0 | n/a | 8845.6 | 8845.6 | 8839.2 | 8011.5 | 8011.5 |
| **全部** | 46562.0 | n/a | 8845.6 | 8845.6 | 8839.2 | 8011.5 | 8011.5 |

**每层 latency [decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 755.9 | n/a | 50693.6 | 3966.5 | 928.0 | 771.7 | 771.7 |
| **全部** | 755.9 | n/a | 50693.6 | 3966.5 | 928.0 | 771.7 | 771.7 |

**每层 latency [e2e = prefill+decode]（ms）**  来源 `dag_A*.json` 的 `summary.requests` + `summary.tiers[tier].start_s`

| tier(layer) | A1 | A2 | A3 | A3a | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 47317.9 | n/a | 59539.2 | 12812.1 | 9767.2 | 8783.3 | 8783.3 |
| **全部** | 47317.9 | n/a | 59539.2 | 12812.1 | 9767.2 | 8783.3 | 8783.3 |

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
| A2 | 754822.2 | 406653.6 | 1161475.8 | 46 | 1134884.9 | 0.0 | 26590.95 | 297.65 |
| A3 | 674401.1 | 3289870.3 | 3964271.4 | 63 | 745337.1 | 3217255.6 | 1678.18 | 18.79 |
| A3a | 674401.1 | 805632.3 | 1480033.4 | 98 | 745337.1 | 733018.1 | 1678.18 | 18.79 |
| A4 | 674401.1 | 888281.1 | 1562682.3 | 130 | 745337.1 | 815667.0 | 1678.18 | 18.79 |
| A5 | 2215220.3 | 266583.0 | 2481803.4 | 247 | 602284.6 | 1879238.1 | 280.56 | 3.14 |
| A6 | 2215220.3 | 266583.0 | 2481803.4 | 247 | 602284.6 | 1879238.1 | 280.56 | 3.14 |
