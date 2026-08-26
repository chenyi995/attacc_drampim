# 块 05:agentic 多轮驻留 KV(history_len)

归属:用户+助手,2026-08-22,**工作区未提交**(前半是被打断会话的遗留,
后半是 08-22 会话续完)。改动文件:`src/workload.py`、`src/ablation.py`、
`src/workload_runner.py`、`main.py`、`tests/test_workload.py`、
`docs/LOG.md`、`docs/HANDOFF.md`。

## 1. 语义(零基础)

多轮 agent 场景:一个 agent 在第 k 轮开始时,自己前 k−1 轮的 KV
**已经算好、驻留在 PIM 内存里**。每请求新字段 `history_len`(H):
- 这 H 行**只被注意力读 (attended),永不重算**;
- 私有(不参与任何共享/复用),不属于任何段,不计入 `total_length`;
- CLI `--history-len` 可整体覆盖 JSON 逐请求值(`main.py:377-382`)。
算子维度:prefill/decode 的 score 列数 n 与 context 收缩维 k 各**加宽 H**
(多读 H 列/多收缩 H 行),而 qkv/投影/FFN 的行数不变(不重算)。

## 2. 前半:解析式(A 系列)路径(`src/ablation.py`)

- `run_ablation_report`:对每批取 `hist_rows=max(history_len)`,把
  score/softmax 的 `n` 与 context 的 `k` 各 +H(sum_decoder 与逐步
  gen_decoder 都改)。
- `_prefill_batch(..., history_rows)`:全重算类里 GPU 算新行、PIM 同时扫
  驻留历史(H=0 时退化为普通 GPU 块);GPU-prefill 放置(A3/A4)下历史行
  计入 `kv_pim_to_gpu` 回读链路量。
- `_batch_scan_profile(..., history_rows)`:decode 扫描画像在三种 KV 摆法
  下都把历史作为**一个前置连续 extent**(master-diff 下=master 池私有行)。
- `_memory_report`:`history_rows` 计入存储与 no-reuse 基线字节。

## 3. 后半:物理事件 DAG 路径(`src/workload_runner.py`)

一个设计决定让所有既有机制免改自然兼容——**历史行=负位置的常驻绑定**:
- TLB 预留:每层每请求一个 `"<id>::history"` 私有指纹块,master 池
  (no-reuse 物理路径下 private 池),`_prepare_cacheblend_tlb` 内。
- 绑定:`_history_tlb_rows`(`_contiguous_no_reuse_tlb_rows` 定义后)
  在段绑定**之后**追加 H 条 `(position=-H..-1, reused=True,
  corrected=False)`。负位置的效果:
  - 因果过滤(split 分支 `pos <= position`、bank-whole 的 DIE 比较器)
    对每条查询**恒可见**(历史永远在过去);
  - 不进 `compute_positions`(不重算、不过链路、不写回);
  - `_prefill_location_deltas` 按索引 0..L−1 读段绑定,追加在尾不受扰,
    历史地址查不到 delta → 位移 0 → 不产生 Q 旋转变体(自己算的 KV
    没有位置移位)。
- 分支条件唯一改动:`full or not reusable` → `not reusable`(H=0 逐位
  等价;有历史时全重算层/无复用请求走 split 机制:GPU 矩形块 + PIM 扫
  历史 + DIE 合并)。
- decode(单请求与批版)自动继承:`old` 来自 prefill 绑定表,历史在内;
  批版中历史是私有 master 行,取交集后自动落入**私有扫描**分支,
  不污染共享 master 流。
- 物理 no-reuse 层:扫描 op 的 n 与行数 +H,KV 链路与写回仍 L 行。
- legacy 解析两路(`run_no_reuse_report`/`run_cacheblend_analytic_report`)
  **不建模**:H>0 显式 `WorkloadValidationError`(老 AttAcc 模拟器保持
  一行不动)。
- 报告:顶层 `history_rows`;`workload_summary` 加 `total_history_tokens`。

## 4. 验证记录(2026-08-22 实测)

- 单测 38/38(含新增 `AgenticHistoryTests` 6 例:两种 JSON 解析、
  split prefill+decode 扫描行数(4+3 例算例)、全重算层仍扫历史、
  物理 no-reuse 加宽(4 查询×6 行、2 个物理 run)、legacy 拒绝、
  ablation 时间/内存入账)。
- 端到端(CACHEBLEND-TINY 模型+真实 Ramulator,relay 负载,EPIC):
  H=0 makespan(全负载完成时刻)0.13646 s → H=256 0.13825 s,
  `history_rows=1280`,重叠契约校验通过。
- 链路核对:`kv_gpu_to_pim` 字节 H=0/H>0 **完全相同**(147,712 B,
  mock 设备(固定时延的假设备,用于隔离事件结构)+relay 负载)——
  没有 KV 行因历史过链路;增量只在
  `q_gpu_to_pim`/`gpu_partial_lse_to_pim`/`ctx_pim_to_gpu`
  (新出现 PIM 分支的层所必需的合并流量)。

## 5. 在论文中的意义

把论文场景从"单轮多 agent 共享 KV"扩展到"**多轮** agent 自有历史
驻留"的建模能力。**目前论文没有对应的实验编号**(严格论文模式下
A/C 之外不设新系列须用户裁决),也没有带 `history_len` 的 workload
JSON——属于能力储备,不入当前正文。

## 6. 悬置

- 尚无带 `history_len` 的实验系列与 workload JSON(现有 JSON 全 H=0,
  行为逐位不变);
- 未提交——提交时机与拆分待用户裁决;
- 历史块的 8 MiB 单块上限(TLB 分配器约束)⇒ 单请求 H ≤ 8 MiB/stride
  行(LLAMA-7B 口径 32768 行),超限报 `WorkloadValidationError`,未做
  多块分裂。
