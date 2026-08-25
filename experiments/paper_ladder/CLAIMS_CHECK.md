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
| C2(微架构) | C 系列实测表 | 加速比/列读/ACT | experiment 分支 `results_c_points.json` |

## 2. 核对结果(2026-08-25,145/145 全矩阵,修复后数据;`results/summary.json`)

### 逐 claim 判定

| Claim | 判定 | 关键数据 |
|---|---|---|
| C1a 软件复用改善 TTFT | **支撑,幅度=复用占比的函数** | vs A1:relay 2.6×、sharegpt 1.77×、mooncakemt 1.05×、multihop 1.07×、mooncake 1.03×(压缩率 0.47→0.88 同序)——报数须按复用占比分层 |
| C1b PIM decode 是 TBT 主收益 | **强支撑(最硬的一条)** | A3 vs A2 的 TBT:mooncake/175B 858→97.8 ms(**8.8×**)、/65B 431→55.6(7.8×)、multihop/175B 192.8→42.1(4.6×);全 15 格方向一致 |
| C1c PIM 感知分池布局 | **支撑(带一处已解释的反例)** | A4 vs A3 TBT:multihop/65B −9.1%、/175B −9.0%、mooncake/175B −5.2%;**反例** mooncake/7B +13%(32 头 % 15 channel 条带不均,头数被 15 整除性问题;头多的模型不受影响) |
| C1d A5 = 强制全 PIM 对照 | **按新口径支撑** | A5 只在真多轮/小中模型赢 TTFT(mooncakemt/7B **−11.4%**、/65B −1.0%、multihop/7B −1.3%),大模型最多 +22%(mooncakemt/175B)——"固定放置两头都错"成立 |
| C1e A6 贴住便宜侧 | **强支撑(修复后零反例)** | 15/15 格 A6 = min(A4, A5),且 mooncakemt/65B A6 47.86 < min(48.38, 47.88)(逐类混选优于两个极端) |
| C1f 复用不牺牲容量 | **支撑(owner-copy 修复后)** | 压缩率全 <1:relay 0.473 / sharegpt 0.601 / multihop 0.798 / mooncake 0.825 / mooncakemt 0.875,随共享度单调 |
| C2 微架构 | 由 C 系列支撑(experiment 分支,流式 P 重测) | n=16@3.2 GHz 6.63× 等;A5/A6 消费其平衡点 2.6 GHz/768 B |

### A6 的 prefill 给 PIM 的份额(问题 1 的核心分析点)

解析路径(时间份额):

| workload | 7B | 65B | 175B |
|---|---|---|---|
| relay | 100% | 100% | 27% |
| multihop | 100% | 0% | 0% |
| sharegpt | 35% | 34% | 5% |
| mooncake | 0% | 0% | 0% |
| mooncakemt | 100% | 98% | 1% |

物理 DAG(逐请求份额,平衡点旋钮):mooncakemt/7B 14%(6/43)、sharegpt/7B 17%(9/52)、mooncake 5%、multihop/relay 0%。

读法:**没有固定放置在所有格子上正确**——份额横跨 0–100%,模型越大越偏 GPU,真多轮/高复用越偏 PIM,同一 run 内部逐请求/逐类真混选(sharegpt 34%、relay/175B 27%)。这正是 dynamic 的立论数据;两条路径口径(逐类时间份额 vs 逐请求计数)并列报告。

### 撤销记录(宸逸 2026-08-25)

原 C1g("放置结论对选择规则不敏感")及其 A6 × 选择变体扫**撤销**:
仿真器的代价只由重算 token **数量**决定,选择算法的身份本来就不进模型,
该"检验"是构造使然而非实验发现;上游软件不构成实验轴。select_* 结果
已从仓库移除,矩阵驱动不再生成。

### 残留注意事项

1. mooncake/7B 的 A4 反例来自 15 通道条带的头数不整除,属布局细节而非方向问题——论文若报 A3/A4 对比,用 65B/175B 行;
2. A6 决策仍是模型实价(oracle)口径,论文闭式 Eq.(placement) 的次优差距是待补数据点(代码 TODO);
3. DAG 与解析两口径的份额数值不同(粒度与旋钮不同),定性一致;正文选一个口径主报、另一个进附录。
