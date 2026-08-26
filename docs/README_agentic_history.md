# agentic 多轮驻留 KV(history_len)功能说明

目标读者:计算机专业学生/接手人,不预设了解本项目或 DRAM/PIM;
概念首现即释,数字均为实测并标出处,本页自足。
读完应能回答:`history_len` 的语义是什么、两条延迟模型路径各怎么实现、
怎么用、验证到什么程度、还悬置什么。实现 commit:`882887b`(2026-08-22)。

## 1. 语义(一句话 + 三条纪律)

多轮 agent 场景:一个 agent 第 k 轮开始时,自己前 k−1 轮的 KV 缓存
(KV cache,注意力为每个历史 token 存的 K/V 向量对)**已经算好、
驻留在 PIM 内存里**。每请求字段 `history_len`(记 H)声明这份驻留行数:

1. **只被注意力读 (attended),永不重算**——prefill 和 decode 的每次
   注意力都要扫它,但 QKV 投影/FFN 的行数不含它;
2. **私有**——不参与任何共享/复用,不属于任何段 (segment),
   不计入 `total_length`;
3. **不过链路**——它本来就在 PIM 里,不因本轮运行产生 GPU↔PIM 传输
   (验证见 §4)。

算子维度(GEMM 记号:C[M×N]=A[M×K]·B[K×N],K 是被乘加消掉的累加维):
注意力的 score 一步是 Q[M×d_head]·K^T[d_head×L],context 一步是
P[M×L]·V[L×d_head];驻留历史让 L 变成 L+H——即 score 的列数 N 与
context 的收缩维 K 各加宽 H,而 M(查询条数)与投影/FFN 行数不变。

## 2. 用法

- JSON:RAG legacy-list 与 supervisor v2-dag 两种格式的每请求条目都可加
  `"history_len": H`(缺省 0);
- CLI:`--history-len H` 把所有请求的值整体覆盖为 H;
- 路径支持:**解析式 A 系列路径**(`--ablation A1..A6`)与
  **物理事件 DAG 路径**(physical)都支持;两条 legacy 解析路径
  (`--no-reuse-latency-model legacy`、`--cacheblend-latency-model analytic`)
  **不建模**,H>0 显式报 `WorkloadValidationError`(拒绝而非静默算错)。

## 3. 实现(两半,结合代码)

### 3.1 解析式半(`src/ablation.py`)

- `run_ablation_report`:对每批取 H=批内最大 `history_len`,把
  score/softmax 的 `n` 与 context 的 `k` 各 +H(prefill 与逐步 decode
  的算子序列都改);
- `_prefill_batch(..., history_rows)`:全重算类里 GPU 重建新行、PIM 同时
  扫驻留历史(H=0 退化为普通 GPU 块);GPU-prefill 放置(A3/A4)下历史行
  计入 `kv_pim_to_gpu` 回读链路量;
- `_batch_scan_profile(..., history_rows)`:decode 扫描画像在
  private/naive/master-diff 三种 KV 摆法下都把历史作为**一个前置连续
  extent**;
- `_memory_report`:`history_rows` 计入存储与 no-reuse 基线字节。

### 3.2 物理事件 DAG 半(`src/workload_runner.py`)

核心设计:**历史行 = 负位置的常驻绑定**,让既有机制免改兼容——

- TLB 每层每请求预留一个 `"<id>::history"` 私有指纹块(master 池;
  no-reuse 物理路径为 private 池),`_history_tlb_rows` 把 H 行绑定在
  查询位置 −H..−1(反正历史永远在"过去",因果过滤 `pos <= position`
  与 bank-whole 因果比较器对每条查询恒可见);
- 不进 `compute_positions` → 不重算、不过链路、不写回;位移 delta=0 →
  不产生 Q 旋转变体;
- 分支条件 `full or not reusable` → `not reusable`(H=0 逐位等价;
  有历史时全重算层走 split 机制:GPU 矩形块 + PIM 扫历史 + DIE 合并);
- 批量 decode 的共享/私有分组把各 agent 的私有历史自动归入**私有扫描**,
  不污染共享 master 流;
- 报告:顶层 `history_rows`;`workload_summary` 增 `total_history_tokens`。

## 4. 验证(实测,2026-08-22)

- 单测 **38/38**(新增 `AgenticHistoryTests` 6 例:两种 JSON 解析、
  split prefill+decode 扫描行数算例、全重算层仍扫历史、物理 no-reuse
  加宽、legacy 拒绝、ablation 时间/内存入账);
- 端到端(CACHEBLEND-TINY+真 Ramulator,relay 负载,EPIC):
  H=0 makespan 0.13646 s → H=256 0.13825 s,`history_rows=1280`,
  重叠契约校验通过;
- 反事实核对:`kv_gpu_to_pim` 链路字节 H=0/H>0 **完全相同**——
  没有 KV 行因历史过链路;增量只在 Q 下行/LSE 元组/context 回传
  (新出现 PIM 分支的层所必需的合并流量)。

## 5. 在论文中的位置与悬置

- **暂无论文实验编号**(严格论文模式下 A/C 之外开新系列须chenyi9 裁决);
  属能力储备,现有 workload JSON 全为 H=0,行为逐位不变;
- 悬置:历史块受 TLB 单块 8 MiB 上限约束(LLAMA-7B 口径单请求
  H ≤ 32768 行),未做多块分裂。
