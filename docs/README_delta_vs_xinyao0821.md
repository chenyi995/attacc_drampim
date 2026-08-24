# 相对 xinyao_0821 基线的全部改动(chenyi-822 / chenyi-822-dirty)

基线:`origin/xinyao_0821`(d6fc5a8)——Xinyao 的复用栈(CacheBlend PIM
仿真:TLB、master/diff 双池、split prefill、事件 DAG、批 decode)。本页
按主题列出我们在其上的全部改动;干净分支 `chenyi-822` 按逐步审阅移植,
快速集成分支 `chenyi-822-dirty` 先行(本页描述 dirty 的当前全量)。

## 1. MQ 批命令与时序模型(阶段 1–3;C3 微架构)

- `src/ramulator_wrapper.py`:MQ 常数(带出处注释)、
  `mq_query_capacity = S/64`(**只约束 Q**;P 流式)、
  `mq_interval_cycles = max(preset 地板 6 PC/4 NPC, ceil(n/(f·tCK)))`
  ——计算永不拉长 DRAM 节拍(FIMDRAM 先例);`mq_pe_power_w` PE 功率
  单独记账(116 W 线,近似口径 TODO 见
  `README_manual_audit_findings.md`);缓存键/YAML nCCDAB 覆盖/trace
  标志管道。
- `ramulator2/trace_gen/gen_trace_attacc_bank.py`(+ 种子副本
  `pim_ramulator_src/` 逐字节同步):`--mq`(一列一条 MAC 服务全部驻留
  Q)、`--shared-kv`、`--phase` 切相、流式 P 注释口径。
- C++ `HBM3-PIM.cpp`:移动总线方向转向约束(MVSB↔MVGB/WRGB 用
  nRTW/nWTRL),两处 +7 行;重编译。

## 2. 事件 DAG 集成(阶段 4–5)

- `src/workload_runner.py`:`_apply_pim_batch` 打批戳;批 decode 按
  GEMV 容量拆波(每波 die_qs/TLB/共享扫描;逐波准入审计);prefill 同
  口径拆波;**D_i 位图**事件(master 写过滤,`di_bitmap_gpu_to_die` +
  `die_load_di_bitmap`,EPIC 每 agent 一次/CacheBlend 每 partial 层
  一次);bank-whole prefill(着陆序落库 + 全范围扫描 + DIE 因果丢弃
  `die_score_assembly`)。
- `src/ablation.py`:MQ 三旋钮进解析模型(容量封顶、批戳、两处调用点)。
- `main.py`:`--pim-batch-command/--pe-freq-ghz/--gemv-buffer-bytes/
  --pim-prefill-mode` 等 CLI。

## 3. 多轮 agentic 历史(阶段 6 内容,dirty 已含)

- `src/workload.py`:`Request.history_len` 字段与解析;
- 解析路径:score/softmax n、context k 各 +H;全重算类"GPU 重建 ∥ PIM
  扫历史";`_batch_scan_profile` 三种映射下前置每请求驻留 extent;占用
  报告含历史行;
- 物理路径:`_history_fingerprint`/`_history_tlb_rows`(负位置
  -H..-1 的私有 extent)、TLB 预留与绑定、no-reuse 物理扫描加宽;
  legacy no-reuse 报告显式拒绝 history;
- `main.py --history-len`、`workload_summary.total_history_tokens`。

## 4. A1–A6 阶梯重定义(2026-08-24 裁决;dirty)

- **废除 split**(基线的 GPU/PIM 混合 prefill,0aced82 引入):解析模式
  只剩 gpu/pim/dynamic,`split_overlap`/`--split-attn` 删除;
- preset 携带批命令(A1–A4 replicate、A5/A6 mq),
  `--pim-batch-command` 默认改为"跟随阶梯";
- **A6 = dynamic**:逐层类/逐请求现算 bank 路 vs xPU 路取小(oracle
  口径;论文闭式 Eq.(placement) 换入留 TODO);
- **物理 DAG 对齐**:`--pim-prefill-mode ∈ {gpu, pim, dynamic}`(默认
  dynamic):gpu = 回读 + GPU 全上下文块(新分支);pim = bank-whole;
  dynamic 决策记入报告 `pim_prefill_sides`;split 事件与 LSE 缝合从
  prefill 路径删除(decode 侧在途行合并保留)。

## 5. 软件上游扩充(2026-08-24;dirty)

`--reuse` 新增 promptcache / cachecraft / cachetune,策略族机制
(`CACHEBLEND_FAMILY`/`EPIC_FAMILY`)贯通两条路径;详见
`README_software_upstream.md`。

## 6. 测试与文档

- 测试:基线 27 → 当前 40(MQ 4、位图/bank-whole、历史 6、阶梯耦合、
  dynamic 逐类取小、放置菜单事件契约、策略族);
- 文档:本 docs/ 全套(总览、A1–A6、软件上游、workload、审计发现、
  本差异页);人工审计发现单独成页
  (`README_manual_audit_findings.md`:功耗分账为近似、TLB 5 ns 未溯源
  两个 TODO)。

## 7. 与基线行为的兼容性说明

- 基线 27 个测试在阶段 0–5 期间始终保持绿色;A 阶梯重定义与 split 废除
  是**语义变更**(2026-08-24 裁决),涉及的旧测试按新菜单重写;
- `pim_ramulator_src/` 种子副本与 `ramulator2/` 保持逐字节一致;
- C1/C2 实验级对照按裁决**不移植**(留在 experiment 分支);仿真器内部
  的 replicate 路径是回归基线,与 C1 无关,保留。
