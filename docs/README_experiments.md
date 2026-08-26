# 我们的实验是怎么做的(822-dirty 实验方法总述)

目标读者:接手人;概念首现即释。本页讲**流程与口径**;矩阵轴的定义在
`../experiments/paper_ladder/MATRIX.md`,claim 核对在同目录
`CLAIMS_CHECK.md`,问题台账在 `README_manual_audit_findings.md`。

## 1. 两个问题,两套证据

- **问题 1(放置)**:共享 KV 下 prefill 注意力 / decode 注意力 / KV
  各放哪,何时逐请求动态选边 → **A 矩阵**(本分支
  `experiments/paper_ladder/`);
- **问题 2(微架构)**:bank 级 PIM 怎么让一次列读服务 n 条查询 →
  **C 系列**(experiment 分支 `experiments/mq_command/`,A5/A6 消费其
  机制,不重测)。

## 2. A 矩阵的做法(固定流程)

1. **轴**:同一批 workload(4 真实 + 1 合成,阶梯间完全相同)×
   3 个模型(LLAMA-7B/65B/GPT-175B)× A1–A6;软件上游**不是实验轴**
   (2026-08-25 裁决:仿真代价只由重算 token 数量决定,选择算法身份不
   进模型);阶梯行固定 EPIC k=8 使档间差只归因放置;
2. **跑**:`python3 run_matrix.py`(断点续跑;32 并发 × 每作业 2 个
   Ramulator worker ≈ 64 核;每作业钉 `OMP/OPENBLAS/..._NUM_THREADS=1`
   防 BLAS 线程池超订);物理 DAG 作业只出 summary 报告(全事件转储
   过大),选边比例在 `pim_prefill_sides`;
3. **指标**:TTFT = `prefill_s`;TBT = 各 tier `decode_per_token_s` 按
   步数加权;压缩率 = `memory.kv_bytes_vs_no_reuse`(owner-copy 修复
   口径);A6 另报 **PIM/GPU 选边比例**(解析=时间份额,DAG=请求/事件
   份额);`collect_results.py` 一键汇总;
4. **结果保留纪律(chenyi9 2026-08-25)**:模型修复后**只保留重跑的新
   结果**,旧 run 一律作废删除;修复不便重跑的纯报告字段用等价算术
   修补并打标(`repair_memory_column.py`,`owner_copy_fix`);
5. **核对**:每个 claim 在 `CLAIMS_CHECK.md` 里对着数据形状过一遍,
   不达标的明确写"不支撑/需改口径";数据暴露的问题进审计台账
   (`README_manual_audit_findings.md`),修复归因到人和 commit。

## 3. C 系列的做法(指针)

C1/C2/C3 定义、MQ 命令消融、总线掉头、RTL sweep 见 experiment 分支
`docs/README_c_series.md` 与 `experiments/mq_command/README.md`;
微架构参数(平衡点 2.6 GHz / 768 B,PROVISIONAL)由 C 系列推导、
A5/A6 preset 继承。

## 4. 复现入口

```bash
cd experiments/paper_ladder
python3 run_matrix.py                     # 全矩阵,断点续跑
python3 repair_memory_column.py           # 兜底修补(幂等)
python3 collect_results.py > results/summary.json
python3 -m unittest discover -s tests     # 回归(仓库根目录)
```

外部库与数据源(只留链接):`../external/README.md`;真实 workload 的
获取与转换:`README_workloads.md` 与
`/data2/chenyi9/KV-PIM/workload/SOURCES.md`。
