# 实验总纲(严格论文模式):有且仅有 A 系列与 C 系列

论文:`/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027`。
本分支(chenyi-experiment-821)的论文实验编号**只有两个系列**;
其余目录一律在 `experiments/_archive/`,不入论文。

---

## A 系列:放置消融(简单消融实验)

**问题**:prefill 注意力、decode 注意力、KV cache 各放在哪(GPU/PIM/布局),
每次只改一件事。实现:`src/ablation.py`(legacy 代价模型 + Ramulator 扫描计时),
入口 `main.py --ablation A1..A6`。

| 编号 | prefill attn | decode attn | KV 布局 | 含义 |
|---|---|---|---|---|
| **A1** | GPU | PIM | private | 原版 AttAcc(无复用,参照点) |
| **A2** | GPU | GPU | none | 纯 GPU 跑 CacheBlend/EPIC |
| **A3** | GPU | PIM | naive | 软件 prefill + PIM decode,无 PIM 感知重映射 |
| **A4** | GPU | PIM | master-diff | 分池布局(master dense + diff 紧凑) |
| **A5** | PIM | PIM | master-diff | prefill 注意力也全进 PIM |
| **A6** | split | PIM | master-diff | GPU 算新行、PIM 扫复用 KV |

**已有 A 系列研究**:`experiments/GPU_PIM_vs_GPU_prefill/`——A4 vs A6(vs A5)的
协同 prefill 拐点:EPIC 每段重算 token 上限 p*(NVLink3 22–35 / PCIe4 89–210,
p* 几乎不随 L 变),CacheBlend 重算比例上限 r(0.4–2.7%)。结果:该目录
`RESULTS.md`;复现:`run_one.sh <dir> <wl> <A4|A5|A6> <link> <policy> "<extra>"`。

配套开关:`--kv-pool-split`、`--master-shadow`、`--split-attn`、
`--pim-prefill-query-batch`、`--tier-batch-size`、`--gpu-model`、`--pim-link`。

---

## C 系列:微架构选择与消融实验

**问题**:共享 KV 下 bank 级 PIM 的批处理微架构——一次列读服务几条 Q、
buffer/PE 频率怎么配、die 侧怎么合并。全部内容与实测表:
`experiments/mq_command/README.md`;设计与审计:`experiments/mq_command/DATAFLOW.md`。

| 编号 | 内容 | driver / 落点 |
|---|---|---|
| **C1** | compact 一份、无微架构设计、单 Q 串行(基线一) | `run_c_points.py`(t1 实测) |
| **C2** | 多 channel 复制 k 份并发(基线二,容量换时间) | 同上(解析合成) |
| **C3** | 非对称 MQ 加速:MQ-MAC 命令 + (n_q,n_c) 驻留 + PE 提频 | 同上((16,2)/(32,4) × 4 频点实测) |
| **C-abl-1** | 命令方案消融:MQ vs ×B 复制 vs dense(96 点) | `run_mq_study.py` |
| **C-abl-2** | 搬运总线方向转向 + 同 channel 两头流水(≤0.84% → 关闭窄下行方案) | `run_pipeline_overlap.py` |
| **C-abl-3** | 微架构 RTL sweep:in-bank PE(基线/(8,1)/(16,2)/(32,4) × 频率)+ logic die(AGENTS 8/16/32),N28 综合对照 AttAcc 基线复现 | `fugue-logic-die-rtl/syn/run_mq_sweep_all.sh` → `collect_mq_results.py` → `MQ_MICROARCH.md` |
| **C-impl** | 机制实装:D_i 位图 master 写过滤;bank-whole 因果丢弃 prefill | `--pim-prefill-mode bank-whole`;单测 32/32 |

**C-main 头条数字**(L=4096,PC):C3 (32,4)@1.3 GHz = 每 agent 1.71 µs,
**3.63× vs C1**,列读/ACT **÷7.1**,容量 1×;C2 需 k≥8 份拷贝才在延迟上胜出且
能耗不降。PE 频率回报递减(context 占比 + MVSB 串行地板 256·n_q),
1.3 GHz 为性价比点,0.666 GHz 行 = AttAcc 原版频率。

---

## 与论文的对应(非实验编号,仅口径映射)

论文 outline 的五级阶梯(GPU-only / PIM-append / PIM-split / PIM-static / Fugue)
≈ A2 / A3 / A4 / A5 / (动态选边未实装,见 `docs/README_design_check.md` §3.1);
AttAcc 参照 = A1。C 系列支撑论文的微架构与 die 面积章节(E4 方向)。
仿真器与论文正文的全部差距清单:`docs/SIM_VS_PAPER_AUDIT_0821.md`。
