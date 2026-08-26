# C 系列:微架构选择与消融实验(严格论文模式,2026-08-21 定)

论文:`/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027`(Fugue)。本目录是本仓库
**C 系列**实验的全部内容;A 系列(放置消融)见 `../GPU_PIM_vs_GPU_prefill/` 与
`docs/EXPERIMENTS.md`。设计文档:`DATAFLOW.md`(硬件增量、全数据流、三路审计、
裁决记录);零基础数据流讲解:`docs/README_fugue_dataflow.md`。

## C 编号定义(用户 2026-08-21 定名)

| 编号 | 含义 | 存储 | 频率 |
|---|---|---|---|
| **C1** | 一份 compact 共享拷贝、无微架构设计:N 个 agent 逐个做单 Q 全 sweep(原版 AttAcc 行为) | 1× | 原版(PE 666 MHz) |
| **C2** | 多 channel 复制 k 份、k 路并发(延迟 = ⌈N/k⌉×t1,由实测 t1 解析合成;并发前提是有空闲 channel,头数≥通道数时退化为 C1) | k× | 原版 |
| **C3** | MQ 加速(2026-08-24 流式 P 修订):MQ-MAC 命令(一条 `MAC_AB` 读一次列、服务 buffer 内全部驻留 Q)+ Q 驻留(容量轴)+ **P 流式**(score 相与 context 相各一遍、同用 n_q;P 不驻留,以 MV_GB 流经 TSV,由移动总线带宽+方向转向计价)+ PE 全流水可提频 | 1× | 0.666(原版)/1.3/2.08/3.2 GHz |

**MQ-MAC 语义与时序**:`ACT_AB` 不变;相邻 `MAC_AB` 有效间隔 =
max(preset 地板 6 PC/4 NPC, PE 吞吐 ceil(n/(f·tCK)))——计算永不拉长
DRAM 节拍,PE 功率单独记账(`mq_pe_power_w`);见
`src/ramulator_wrapper.py::mq_interval_cycles`,经 Ramulator2 原生 YAML `nCCDAB`
覆盖注入(不改 C++、命令集不变,n 走 AttAcc 自有的 `PIM_SET_CONFIG`)。
驻留上限只在 Q 侧:每 bank 一条 Q 切片 64 B,n_q=8 用原装 512 B buffer,
16→1 KiB(×2)、32→2 KiB(×4);P(每查询每 bank L/8 B)不驻留,
其上限是 TSV 移动总线(当前命令序下 n ≤ interval 不拖慢 V 扫描,
见 `docs/README_mq_design_space.md` §4)。

## C 系列实验清单(有且仅有以下各项)

| 项 | 内容 | driver | 结果 |
|---|---|---|---|
| **C-main** | C1 / C2 / C3 三方案对比,n_q ∈ {8,16,32} × PE 频率档(流式 P) | `run_c_points.py` | `results_c_points.json`,表见下 |
| **C-abl-1(命令方案消融)** | MQ-MAC vs 旧 ×B 命令复制 vs dense,96 点(间隔模型校验 / decode 共享 / prefill 多 Q) | `run_mq_study.py --workers 48` | `results_mq_study.json` |
| **C-abl-2(总线转向)** | 搬运总线方向转向惩罚 + 同 channel 两头流水合成 trace | `run_pipeline_overlap.py` | `results_pipeline_overlap.json`:JEDEC 默认转向代价 ≤0.84%、×4 夸大 ≤3.8% → 错峰调度与专用窄下行均不做 |
| **C-abl-3(微架构 RTL sweep)** | in-bank PE(基线/(8,1)/(16,2)/(32,4) × 667 MHz–1.3 GHz)与 logic die(AGENTS 8/16/32)的 N28 综合面积/时序,对照 AttAcc 基线复现 | `/data2/chenyi9/KV-PIM/fugue-logic-die-rtl`:`syn/run_mq_sweep_all.sh`(nohup 断点续跑)、汇总 `syn/collect_mq_results.py` | `syn/MQ_MICROARCH.md`(sweep 完成后生成) |
| **C-impl(机制实装)** | ① D_i 位图 master 写过滤;② bank-whole 因果丢弃 prefill(`--pim-prefill-mode`) | 单测 `tests/test_workload.py`(32/32)+ e2e 冒烟 | 见 `DATAFLOW.md` §6 |

## C-main 实测(L=4096,PC 档,每 channel 单头视角;t1 = 6.22 µs;C1 = n_q×t1;
2026-08-24 流式 P 重测)

| n_q | PE | 间隔 | score | context | 合计 | 每 agent | vs C1 | 列读/ACT | C2 等延迟需 k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 (S=512B 原装) | 0.666 GHz | 16 | 7.3 µs | 9.9 µs | 17.2 µs | 2.15 µs | 2.89× | ÷8 | 3 |
| 8 | 1.3 GHz | 9 | 4.6 | 6.9 | 11.5 | 1.44 | 4.32× | ÷8 | 5 |
| 8 | ≥2.08 GHz(匹配) | 6 | 3.1 | 5.7 | 8.8 | **1.11** | **5.63×** | ÷8 | 6 |
| 16 (S=1KB) | 0.666 GHz | 32 | 14.6 | 19.6 | 34.2 | 2.14 | 2.91× | ÷16 | 3 |
| 16 | 1.3 GHz | 17 | 8.5 | 13.6 | 22.0 | 1.38 | 4.52× | ÷16 | 5 |
| 16 | 3.2 GHz | 7 | 5.6 | 9.4 | 15.0 | **0.94** | **6.63×** | ÷16 | 7 |
| 32 (S=2KB) | 1.3 GHz | 33 | 19.0 | 26.9 | 45.9 | 1.44 | 4.33× | ÷32 | 5 |
| 32 | 3.2 GHz | 14 | 12.3 | 19.1 | 31.4 | 0.98 | 6.35× | ÷32 | 7 |

要点:①context 一遍完成后,**n_q=32 不再优于 16**——P 流受 TSV 线
(n ≤ interval 才不拖慢 V 扫描)、score 受 PE Fmax,n 超过后每 agent 延迟
贴平,收益只剩列读/ACT ÷n_q 继续涨;②n=8 在 2.08 GHz 到匹配点
(f\*(8)=1.73 GHz),再提频无收益;③C2 要在延迟上胜过 C3 需 k≥5–7 份拷贝
且读能耗不降;④C-abl-1 显示旧 ×B 命令仅 0.90×dense(只省 ACT)——C3 的
收益主体是去掉冗余列读。

## 复现

```bash
python3 -m unittest tests.test_workload            # 全套 38/38
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
