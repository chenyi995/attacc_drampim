# C 系列实验汇总:共享 KV 下 bank-PIM 的批处理微架构

目标读者:计算机专业学生/接手人,不预设了解本项目或 DRAM/PIM;
概念首现即释,数字均为实测并标出处,本页自足。
读完应能回答:C 系列每个实验是什么、机制在哪、头条数字与出处、
支撑论文哪个论点、怎么复现。

## 0. 三十秒背景

n 个 agent 共享同一份 KV 缓存时,decode 每步有 n 条查询要对同一 K/V
做注意力(GEMM 视角:Q[n×d_head]·K^T[d_head×L])。bank 级 PIM
(bank PE,DRAM bank 旁的乘加单元)的原生命令一次只服务一条查询:
n 条查询 = 把 KV 从 DRAM 列读 (column read) n 遍——列读是最贵的动作。
C 系列回答:**怎么让一次列读服务 n 条查询、代价是什么、微架构怎么配**。
查证过的空白:bank 级 PIM 文献无人做多查询共享列读(LongSight,
MICRO'25:"batching…no reuse due to lack of shared KV")。

## 1. 主实验:C1 / C2 / C3

| 编号 | 是什么 | 机制 |
|---|---|---|
| **C1 compact** | 基线一:AttAcc 原样,KV 存一份,单查询串行扫 | 上游 replicate 命令流 |
| **C2 多通道** | 基线二:KV 复制 k 份到 k 组通道并行(容量换时间) | 解析上界 `ceil(N/k)×t1`(对 C2 有利的估计) |
| **C3 MQ** | 论文主张:MQ-MAC 批命令 + Q 驻留(容量轴)+ P 流式(TSV 移动总线计价)+ PE 提频;score/context 两相各一遍、同用 n_q(2026-08-24 流式 P 修订) | `--pim-batch-command mq`、`--phase`、`mq_interval_cycles` |

**MQ-MAC 命令**:一条 `PIM_MAC_AB` 列读一次,bank PE 对 n 条驻留查询各做
一次 16 路 FP16 乘加;只有查询私有搬运(载 Q/搬 score/softmax/搬回 P)
仍每查询一份。命令间隔 `mq_interval_cycles` = max(preset 地板 6(功耗
受限)/4(不受限), PE 吞吐 ceil(n/(f·tCK)))——**计算永不拉长 DRAM
节拍**(FIMDRAM 先例),PE 功率单独记账(`mq_pe_power_w` 对照 116 W
预算线)。设计上有**两个独立的轴**:GEMV buffer 容量决定驻留几条查询
(64 B/条,面积大头;**只约束 Q**——context 相的 P 流式,不驻留,
2026-08-24 裁决),PE 频率决定每次列读间隔能服务几条(面积小头);
每档容量 n 有匹配频率 f\*(n)=n/(地板·tCK):n=8/16/32 →
1.73/3.47/6.93 GHz(6.93 超出 MAC 树流水实测 Fmax≈2.67 GHz,被封顶)。
P 流的第三条线:context 相 TSV 移动总线,当前命令序下 n ≤ interval
即不拖慢 V 扫描(推导+实测闭合,见 `README_mq_design_space.md` §4)。

**头条数字**(L=4096、功耗受限,2026-08-24 流式 P 重测,
`experiments/mq_command/results_c_points.json`):

- C3 n_q=16@1.3 GHz:**每 agent 1.38 µs,4.52× vs C1,列读/行激活 ÷16,
  KV 容量 1×**;n_q=8@2.08 GHz(匹配点)5.63×,n_q=16@3.2 GHz 6.63×;
  同频对照行(0.666 GHz=AttAcc 原生 PE):n=8/16/32 → 2.89/2.91/2.87×
  ——这是纯命令/微架构收益,提频部分另有面积代价(C-abl-3);
- C2 需 k≥5–7 份拷贝才在延迟上追平,且读能耗不降(排除项)。

## 2. 消融与实装

| 编号 | 内容 | 结论/落点 | driver |
|---|---|---|---|
| **C-abl-1** | 命令方案消融:MQ vs ×B 复制 vs dense,96 点 | 支撑"为什么是 MQ 命令" | `run_mq_study.py` |
| **C-abl-2** | 搬运总线方向转向(nRTW/nWTRL 约束)+ 同通道两头流水 | 流水收益 ≤0.84% → **关闭窄下行方案**(设计裁决) | `run_pipeline_overlap.py` |
| **C-abl-3** | 微架构 RTL sweep:bank PE(基线/(8,1)/(16,2)/(32,4)×频点)+ logic die(AGENTS 8/16/32),N28/Genus 12 点 | 论文 die 面积口径(只报相对 AttAcc bank PE 的倍数);按 AttAcc §7.7 的 in-bank 面积预算,(16,2) 各频点在预算内,(32,4) 从 1.0 GHz 起越线。**注**:build 名沿用旧 (n_q,n_c) 驻留设计;流式 P 后 n_c 不再是设计参数,面积点待按新结构重扫 | `fugue-logic-die-rtl/syn/run_mq_sweep_all.sh` → `collect_mq_results.py` |
| **C-impl** | 机制实装:D_i 位图 master 写过滤(到达顺序无关);bank-whole 因果丢弃 prefill | 对应论文 §4.3.2 / §4.5.2 | `--pim-prefill-mode bank-whole`;单测 |

## 3. 在论文中的意义

C 系列支撑论文的**微架构与 die 面积章节**(E4 方向)。公平性声明
(进正文必写):C3 头条须并排同频行与提频面积代价;C2 是解析上界
(对被排除方有利,排除结论因此保守);PE 提频只提 PE、DRAM 时序地板
(功耗/通路 nCCDAB)不放松。

## 4. 实现位置(结合代码)

- 时序模型:`src/ramulator_wrapper.py`(`mq_query_capacity`/
  `mq_interval_cycles`/YAML `nCCDAB` 覆盖/签名缓存键);
- trace:`ramulator2/trace_gen/gen_trace_attacc_bank.py`(`--mq`/`--phase`);
- C++:`ramulator2/src/dram/impl/HBM3-PIM.cpp`(MVSB↔MVGB/WRGB 转向
  约束,唯二 +7 行);
- 集成:`src/workload_runner.py`(层标记、按 GEMV 容量拆 sweep、D_i
  位图事件、bank-whole)、`main.py`(CLI,**mq 为默认**;代码内部 Layer
  默认 replicate 保回归);
- 测试:`tests/test_workload.py` `MQBatchCommandTests` 5 例;
- 设计与三路审计:`experiments/mq_command/DATAFLOW.md`。

## 5. 复现

```sh
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
python3 experiments/mq_command/run_c_points.py          # C1/C2/C3 十二点
python3 experiments/mq_command/run_mq_study.py --workers 48   # C-abl-1
python3 experiments/mq_command/run_pipeline_overlap.py  # C-abl-2
# C-abl-3(RTL 仓库):
cd /data2/chenyi9/KV-PIM/fugue-logic-die-rtl/syn && python3 collect_mq_results.py
```
