# C 系列:微架构选择与消融实验(严格论文模式,2026-08-21 定)

论文:`/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027`(Fugue)。本目录是本仓库
**C 系列**实验的全部内容;A 系列(放置消融)见 `../GPU_PIM_vs_GPU_prefill/` 与
`docs/EXPERIMENTS.md`。设计文档:`DATAFLOW.md`(硬件增量、全数据流、三路审计、
裁决记录);零基础数据流讲解:`docs/README_fugue_dataflow.md`。

## C 编号定义(宸逸 2026-08-21 定名)

| 编号 | 含义 | 存储 | 频率 |
|---|---|---|---|
| **C1** | 一份 compact 共享拷贝、无微架构设计:N 个 agent 逐个做单 Q 全 sweep(原版 AttAcc 行为) | 1× | 原版(PE 666 MHz) |
| **C2** | 多 channel 复制 k 份、k 路并发(延迟 = ⌈N/k⌉×t1,由实测 t1 解析合成;并发前提是有空闲 channel,头数≥通道数时退化为 C1) | k× | 原版 |
| **C3** | 非对称 MQ 加速:MQ-MAC 命令(一条 `MAC_AB` 读一次列、服务 buffer 内全部驻留 Q)+ 相位非对称驻留(score 相 n_q 条 Q 一遍;context 相 n_c 条 P × ⌈n_q/n_c⌉ 遍)+ PE 全流水可提频 | 1× | 0.666(原版)/1.3/2.08/3.2 GHz |

**MQ-MAC 语义与时序**:`ACT_AB` 不变;相邻 `MAC_AB` 有效间隔 =
max(IDD7 功耗拉伸, PE 吞吐, DRAM 通路下限),见
`src/ramulator_wrapper.py::mq_interval_cycles`,经 Ramulator2 原生 YAML `nCCDAB`
覆盖注入(不改 C++、命令集不变,n 走 AttAcc 自有的 `PIM_SET_CONFIG`)。
驻留上限:每 bank 一条 Q 切片 64 B、一条 P 切片 L/8 B;(16,2) 需 buffer 1 KiB
(×2,die +12.2%→审计修正后 ≈13.5%),(32,4) 需 2 KiB(×4,≈17%)。

## C 系列实验清单(有且仅有以下各项)

| 项 | 内容 | driver | 结果 |
|---|---|---|---|
| **C-main** | C1 / C2 / C3 三方案对比,(16,2) 与 (32,4) 两点 × PE 频率档 | `run_c_points.py` | `results_c_points.json`,表见下 |
| **C-abl-1(命令方案消融)** | MQ-MAC vs 旧 ×B 命令复制 vs dense,96 点(间隔模型校验 / decode 共享 / prefill 多 Q) | `run_mq_study.py --workers 48` | `results_mq_study.json` |
| **C-abl-2(总线转向)** | 搬运总线方向转向惩罚 + 同 channel 两头流水合成 trace | `run_pipeline_overlap.py` | `results_pipeline_overlap.json`:JEDEC 默认转向代价 ≤0.84%、×4 夸大 ≤3.8% → 错峰调度与专用窄下行均不做 |
| **C-abl-3(微架构 RTL sweep)** | in-bank PE(基线/(8,1)/(16,2)/(32,4) × 667 MHz–1.3 GHz)与 logic die(AGENTS 8/16/32)的 N28 综合面积/时序,对照 AttAcc 基线复现 | `/data2/chenyi9/KV-PIM/fugue-logic-die-rtl`:`syn/run_mq_sweep_all.sh`(nohup 断点续跑)、汇总 `syn/collect_mq_results.py` | `syn/MQ_MICROARCH.md`(sweep 完成后生成) |
| **C-impl(机制实装)** | ① D_i 位图 master 写过滤;② bank-whole 因果丢弃 prefill(`--pim-prefill-mode`) | 单测 `tests/test_workload.py`(32/32)+ e2e 冒烟 | 见 `DATAFLOW.md` §6 |

## C-main 实测(L=4096,PC 档,每 channel 单头视角;t1 = 6.22 µs;C1 = n_q×t1)

| C3 配置 | PE | score | context×8趟 | 合计 | 每 agent | vs C1 | 列读/ACT | C2 等延迟需 k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (16,2) S=1KB | 0.666 GHz | 14.6 µs | 3.6 µs×8 | 43.5 µs | 2.72 µs | 2.29× | ÷3.6 | 3 |
| (16,2) | 1.3 GHz | 8.5 | 3.6×8 | 37.4 | 2.34 | **2.66×** | ÷3.6 | 3 |
| (16,2) | 2.08 GHz | 6.5 | 3.6×8 | 35.4 | 2.21 | 2.81× | ÷3.6 | 3 |
| (32,4) S=2KB | 0.666 GHz | 29.6 | 4.9×8 | 68.6 | 2.14 | 2.90× | ÷7.1 | 3 |
| (32,4) | 1.3 GHz | 19.0 | 4.5×8 | 54.8 | **1.71** | **3.63×** | ÷7.1 | 4 |
| (32,4) | 2.08 GHz | 14.8 | 4.5×8 | 50.6 | 1.58 | 3.93× | ÷7.1 | 4 |
| (32,4) | 3.2 GHz | 12.3 | 4.5×8 | 48.1 | 1.50 | 4.13× | ÷7.1 | 5 |

要点:①(32,4) 全面优于 (16,2);②PE 频率回报递减(context 占 60–85% + MVSB
串行地板 256·n_q),**1.3 GHz(命令时钟同频)是性价比点,0.666 GHz 行是论文原版
频率**;③C2 要在延迟上胜过 C3 需 k≥8 份拷贝且读能耗不降,C3 另有 ÷3.6–7.1 的
列读/ACT;④C-abl-1 显示旧 ×B 命令仅 0.90×dense(只省 ACT)——C3 的收益主体是
去掉冗余列读。

## 复现

```bash
python3 -m unittest tests.test_workload            # 32/32
python3 experiments/mq_command/run_c_points.py     # C-main
python3 experiments/mq_command/run_mq_study.py --workers 48    # C-abl-1
python3 experiments/mq_command/run_pipeline_overlap.py         # C-abl-2
# C-abl-3(RTL):cd /data2/chenyi9/KV-PIM/fugue-logic-die-rtl/syn && nohup setsid ./run_mq_sweep_all.sh &
# e2e(main.py 默认 --pim-batch-command mq;--pim-prefill-mode bank-whole 可选):
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json --reuse epic \
  --epic-prefix-recompute-tokens 1 --cacheblend-batch-size 4 \
  --ramulator-workers 24 --pipeopt --workload-report /tmp/e2e_mq.json
```

历史注记:C 编号前身的 "B1/B2 baseline" 临时命名已废止(与论文 outline 的
B0–B4 阶梯撞名);本仓库实验编号**有且仅有 A 系列与 C 系列**。
