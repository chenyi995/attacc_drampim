# CLAIMS_CHECK:数据是否撑得住文章的两个问题(核对页,随矩阵更新)

文章的两个问题与其证据体系:

- **问题 1(放置)**:请求共享 KV 后,prefill 注意力 / decode 注意力 /
  KV 缓存各放哪,何时值得逐请求动态选边?——证据 = 本目录 A1–A6 矩阵
  (TTFT / TBT / 压缩率三维 + dynamic 选边比例)。
- **问题 2(微架构)**:bank 级 PIM 怎么让一次列读服务 n 条共享 KV 的
  查询?——证据 = experiment 分支 C 系列(C3 vs C1/C2、容量×速率两轴、
  流式 P 的 TSV 平衡线、n_q=16@3.2 GHz 6.63× 等,2026-08-24 重测)。

## 1. 每个 claim 需要的数据形状(核对标准)

| Claim | 需要的数据 | 维度 | 来源 |
|---|---|---|---|
| C1a 软件复用本身有限(A2 vs A1) | 同 workload 同模型两档对比 | TTFT/TBT | ladder_* |
| C1b PIM decode + KV 驻留是 TBT 的主收益(A3 vs A2) | 同上 | **TBT** 为主 | ladder_* |
| C1c PIM 感知布局消除碎裂(A4 vs A3) | 同上 | TTFT+TBT | ladder_* |
| C1d prefill 上 PIM + batching 拿 TTFT(A5 vs A4) | 同上 | **TTFT** 为主 | ladder_* |
| C1e 动态规则两头都不吃亏(A6 ≥ max(A4,A5) 侧) | A6 vs A4/A5 + 选边比例 | TTFT + 比例 | ladder_* + dag_* |
| C1f 复用不牺牲容量(压缩率 <1 且随共享度变化) | kv_vs_no_reuse 全矩阵 | 压缩率 | ladder_* |
| C1g 放置结论对选择规则不敏感 | A6 × 4 个选择变体 | TTFT/TBT | select_* |
| C2(微架构) | C 系列实测表 | 加速比/列读/ACT | experiment 分支 `results_c_points.json` |

## 2. 核对结果(矩阵完成后填;当前为部分数据)

**已知风险与提醒(先记录)**:

1. **小模型上阶梯差异被 decode 稀释**:CACHEBLEND-TINY + ShareGPT 的
   六档 makespan 只差 ~1%(A1 1.799 / A3 1.804 / A4 1.794 / A5 1.795 /
   A6 1.795;A2 2.222 除外)——TINY 的 prefill 占比太小。**结论必须报
   TTFT 维而不是 makespan,且主表用真实模型**(LLAMA-65B/GPT-175B)。
   A2 慢 24% 倒是干净地支撑 C1b(decode 回 GPU 立刻付带宽代价)。
2. **压缩率维**:`kv_bytes_vs_no_reuse` 在 A1(private)恒 1,A3+ 随
   共享度下降;注意它衡量**容量**,与 C2(多通道复制换时间)的对照要
   引用 C 系列的 storage 列,不在本矩阵。
3. **dynamic 选边比例**:物理 DAG 的 `pim_prefill_sides`(请求份额)与
   解析路径的时间份额都会给出;若某 workload 全偏一侧,要报出"为什么"
   (复用占比/模型规模),这本身是论文的分析点而非缺陷。
4. **oracle 口径**:A6 决策输入是模型实价(上界口径);换论文闭式
   Eq.(placement) 的次优差距是待补数据点(TODO 已在代码注释)。

(矩阵跑完后:`python3 collect_results.py > results/summary.json`,把
关键表填进本节,不达标的 claim 明确标"数据不支撑/需要改口径"。)
