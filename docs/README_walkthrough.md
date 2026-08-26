# 全链路走读(walkthrough):从跑法与输入,到周期数出炉

目标读者:用户逐文件审读用;也适合接手人。规矩:**一站一个文件**,每站
讲清"它收什么变量、算什么、把什么传给谁",配一条可跑的验证命令。顺序
沿数据流:输入 → 解析 → 建模 → 定价 → 仿真内核 → 报告。

## 路线图(12 站)

| 站 | 文件 | 一句话职责 |
|---|---|---|
| 0 | (跑法)`main.py` 的命令行 | 怎么把仿真跑起来:两条路径的入口命令与全部旋钮 |
| 0.5 | 上游数据源仓库(`/data2/chenyi9/KV-PIM/workload/`,链接清单 `external/README.md`) | **软件侧的源头**:Mooncake/ShareGPT/MultiHop-RAG 等 GitHub/HF 仓库,各自的原始输出格式,及我们的转换器 `convert_*.py` 把它改写成 workload JSON |
| 1 | `experiments/paper_ladder/workloads/*.json` | **输入**:软件侧的输出=仿真的输入;两种 schema(rag 列表 / v2-dag) |
| 2 | `src/workload.py` | 收 JSON → `Workload/Request/Segment`;`build_reuse_plan` 把策略变成"哪些行复用、哪些行重算" |
| 3 | `src/config.py` | 收 `--model/--gpu` → 模型维度表与设备参数(带宽/能量表/功耗档) |
| 4 | `src/model.py` | 把维度铸成 **`Layer` 对象**(m/n/k/numOp/dbyte)——全系统的变量载体 |
| 5 | `src/system.py` | 组装 `System`:GPU/PIM 设备 + `Ramulator(…, "ramulator2", "ramulator.out")` |
| 6 | `src/ablation.py` | 解析路径主干:A1–A6 preset、prefill 定价 `_prefill_batch`(含 A6 dynamic 块)、decode 布局 `_batch_scan_profile` 与逐步定价 |
| 7 | `src/devices.py` | 设备模型:`PIM.get_time_and_energy(layer)` 把 score 层转交 Ramulator,其余按公式 |
| 8 | `src/ramulator_wrapper.py` | 把 Layer 贴片翻译成 trace 参数 + **YAML nCCDAB 覆盖**(mq 时序在此注入);形状缓存与签名缓存 |
| 9 | `ramulator2/trace_gen/gen_trace_attacc_bank.py` | 生成命令流 `.trace`(WRGB/MAC_AB/MVSB/SFM/MVGB/BARRIER;`--mq` 折叠、`--phase` 切相) |
| 10 | `ramulator2/src/dram/impl/HBM3-PIM.cpp` | 周期级内核:命令时序表、preset、移动总线掉头(nRTW/nWTRL) |
| 11 | `src/workload_runner.py` | 物理事件 DAG 路径:TLB、事件流、gpu/pim/dynamic 三分支、校验器 |
| 12 | 报告 JSON(`results/*.json`) | **输出**:三维指标、breakdown、memory、`pim_prefill_sides` 的读法 |

每站讲完后,A1–A6 六条路在第 6/11 站分岔处各走一遍(对应
`README_A1.md`…`README_A6.md` 的代码定位表)。

## 变量流(总图,细节在各站)

```
CLI(main.py 收) ──→ Workload(第2站) ──→ ReusePlan(第2站)
      │                    │
      └── AblationConfig(第6站)/pim 旋钮(第11站)
                           │
     model.build(batch,lin,lout) → Layer(第4站)
                           │  ← 贴片:pim_kv_runs / pim_shared_queries /
                           │          pim_batch_command / pim_pe_freq_ghz / pim_phase
      devices(第7站) → wrapper(第8站) → trace(第9站)+YAML → C++(第10站)
                           │
                 (time, energy) 逐层返 → 报告 JSON(第12站)
```

## 快速跑法(第 0 站的实体)

```bash
# 解析路径(A 阶梯):
python3 main.py --system dgx-attacc --model LLAMA-7B \
  --workload experiments/paper_ladder/workloads/workload_multihoprag_n32_o0.json \
  --reuse epic --epic-prefix-recompute-tokens 8 \
  --ablation A6 --history-len 3 --pipeopt \
  --ramulator-workers 8 --workload-report /tmp/a6.json
# 物理事件路径(同一编排):
python3 main.py … --pim-prefill-mode dynamic --workload-report /tmp/dag.json
# 回归:
python3 -m unittest discover -s tests    # 41/41
```

审读辅助:每站验证命令与更深的机制文档索引见 `docs/README.md` §7;
逐档代码定位表 `README_A1.md`–`README_A6.md`;问题台账
`README_manual_audit_findings.md`。
