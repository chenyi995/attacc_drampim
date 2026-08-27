# 相对 xinyao_0821 基线的全部改动(chenyi-822 / chenyi-822-dirty)

基线:`origin/xinyao_0821`(d6fc5a8)——Xinyao 的复用栈(CacheBlend PIM
仿真:TLB、master/diff 双池、split prefill、事件 DAG、批 decode)。本页
按主题列出我们在其上的全部改动;干净分支 `chenyi-822` 按逐步审阅移植,
快速集成分支 `chenyi-822-dirty` 先行(本页描述 dirty 的当前全量)。
**更新至 2026-08-27**(§8 = 08-25 修复波;§9 = 08-26 引擎波
`4e582cb`/`8b58fe7`/`c8e39c9`;§10 = 08-26 晚–08-27 语义波
`94f8f46`/`6b05a22`/`c163936` + recompute 提交)。

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
- C++ `HBM3-PIM.cpp`:**新增移动总线掉头 (direction turnaround) 开销**
  ——xinyao/attacc 原模型里,半双工 TSV/GBUS 通路上 MVSB(bank→die,
  读向)与 MVGB/WRGB(die→bank,写向)之间的方向切换是**零代价**的;
  我们在 pseudochannel 级加上 JEDEC 口径的转向约束(MVSB→MVGB/WRGB 收
  nRTW,反向收 nWTRL,preset 值、YAML 可覆盖),两处 +7 行,重编译,
  种子副本同步。这是对我们自己的设计**从严**的改动:流式 P 与分数上行
  共用这条半双工通路,方向切换的真实代价必须计入。实测量级
  (experiment C-abl-2,`run_pipeline_overlap.py`):JEDEC 默认转向代价
  ≤0.84%,×4 夸大也只 ≤3.8%——搬运命令大多藏进 MAC 间隔空档,由此
  裁决关闭了错峰调度与专用窄下行两个备选设计。

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
  prefill 路径删除(decode 侧在途行合并保留)。2026-08-26 进一步扩到
  **六档全覆盖**,见 §9。

## 5. 软件上游扩充(2026-08-24;dirty)

`--reuse` 新增 promptcache / cachecraft / cachetune,策略族机制
(`CACHEBLEND_FAMILY`/`EPIC_FAMILY`)贯通两条路径;详见
`README_software_upstream.md`。

## 6. 测试与文档

- 测试:基线 27 → 当前 **41**,2026-08-26 引擎波之后全绿(MQ 4、位图/
  bank-whole、历史 6、阶梯耦合、dynamic 逐类取小、放置菜单事件契约、
  策略族、占用报告单测等);
- 文档(2026-08-26 重组):`docs/intro/` 收 A1–A6 六份逐档定位;
  `docs/archived/` 收弃用件(旧 workload 文档、走查稿、被合并的三份
  audit);**`README_manual_audit_findings.md` 是唯一审计总台账**
  (已解决 R1–R17、未解决 U1–U6、流程性裁决记录、阶梯诊断与 workload
  有效性两个附录)。

## 7. 与基线行为的兼容性说明

- 基线 27 个测试在阶段 0–5 期间始终保持绿色;A 阶梯重定义与 split 废除
  是**语义变更**(2026-08-24 裁决),涉及的旧测试按新菜单重写;
- `pim_ramulator_src/` 种子副本与 `ramulator2/` 保持逐字节一致;
- C1/C2 实验级对照按裁决**不移植**(留在 experiment 分支);仿真器内部
  的 replicate 路径是回归基线,与 C1 无关,保留;
- 2026-08-26 新增的 `--engine` 默认 `analytic`:所有既有命令行为逐字节
  不变;`--engine dag` 才启用 §9 的六档物理路径。

## 8. 审计修复波(2026-08-25 裁决落地;台账 R4–R8)

首批矩阵数据审计出的五题全部修复(诊断原文见总台账附录 A):

- **A6 估价平价**(R4):dynamic 的 xPU 侧估价改用 gpu 档同一批级折算
  口径,估=入账对称,A6 = min(A4,A5) 全 15 格验证;
- **分池 15/1**(R5):物理 `_KV_CHANNELS` 与解析默认同步从 8/8 改
  master 0–14 / diff 15("channel 划分不是对半分");
- **naive 冲突模型**(R6):解析路径逐 chunk 顺序分配 channel 并追踪,
  同 channel 串行化(`_naive_channel_pools`);
- **A5/A6 挂微架构参数**(R7):preset 绑定平衡点 2.6 GHz / 768 B
  (PROVISIONAL),mq 下每波跟随 `mq_query_capacity`;
- **占用报告属主副本双计**(R8):`_memory_report` 去重 + 单测 +
  存量修补脚本;
- 配套:`experiments/paper_ladder/` 论文证据矩阵(5 workload × A1–A6 ×
  3 模型,145/145 重跑,`CLAIMS_CHECK.md` 逐 claim 判定);legacy A1
  decode 形状缓存 32 线程并行预热(§9 提至 64)。

## 9. 引擎全覆盖与运行基建(2026-08-26;`4e582cb`/`8b58fe7`/`c8e39c9`;台账 R10–R13)

- **`--engine {analytic,dag}`,物理事件 DAG 覆盖 A1–A6 全六档**(R11):
  A1 接通既有 private 物理路径(replicate);**A2 新 GPU-only 事件路径**
  ——按裁决 KV 全放**远端哑存储**,链路字节定义 = GPU↔远端存储经
  NVLink/PCIe(R10:prefill 写出/读回,decode 每步整上下文×全层拖回,
  wl_tiny 实测链路 42.1 GB);**A3 新 `NaiveKVLayout`**(块按软件序轮转
  16 个单 channel 池、无 master/diff 之分,碎片惩罚由排程涌现,TLB
  描述符数同步上升可见);A4–A6 preset 布线。附带修复:整层复用零修正
  请求的 qkv m=0 除零护栏(两分支);
- **归边统计统一口径**(R12):`prefill_attention_rows/_sides`
  (pim/gpu/mixed/none 普查,分母恒为全部请求)+ 行加权
  `pim_prefill_share`;报告新增 `energy_breakdown_nj`(设备类 + 事件名
  两粒度);
- **Ramulator"先建缓存再跑"**(R13):签名缓存落盘
  `ramulator2/signature_cache.jsonl`(载入/追加),`--ramulator-workers
  64`,解析预热池 32→64;实测六档首建 ~32 min → 暖缓存复跑 2 min;
- **一键阶梯**:`experiments/run_dag_ladder.sh <workload.json> [模型]`
  (A1 no-reuse、A2–A6 EPIC k=8)→ 六份报告 +
  `collect_dag_ladder.py` 汇总 CSV(每部件能耗、归边普查);
- **目录约定**:自造 workload 入 `workload/`(`wl_tiny.json` 已入);
  输出入 `output/<时间戳>_<负载>_<模型>/`;
  `/data2/chenyi9/KV-PIM/workload/` 仅作真实负载源数据暂存区;
- 已知未对齐(总台账 U3/U4):解析引擎 A2 仍是"KV 在 GPU 本地"旧口径
  (矩阵 A2 列同);DAG pim 档对全新请求走普通 GPU prefill,与解析 A5
  "全部上 PIM"语义分叉——两条均待裁决。

## 10. fig:motiv 负载工程与七档语义波(2026-08-26 晚–08-27;台账 R14–R17)

- **naive 布局按真实软件语义重建**(R14):256-token 自然块、页化
  (大段先切页再 append 轮换)、共享内容分轮引入——row conflict 只由
  轮换与负载自己产生;
- **A3/A3a 语义拆分**(R15):读掩是 Fugue die 侧硬件特性——A3 无掩
  (重算行处 master run 劈开),新档 **A3a** 可掩不断流;gpu-prefill
  读回补 DRAM 侧读事件;**阶梯七档**;
- **decode 服务批化 + MQ 顶格 8**(R16):全档批宽 8(GPU 权重一遍服务
  全波、KV 每查询各自拉;A2 重写为同构波结构);A5/A6 = 512 B/1.3004 GHz
  (PC 能量钳位 8 tCK,R19),
  MQ 仅在"prefill attention in PIM"档启用;
- **通用 `recompute` 策略**(R17):位移块内均匀随机抽 k 个 token 重算
  (EPIC 等原有分支不动);ladder A2–A6 用之,比例轴 k∈{2,4,8,16,32};
- 五个文献引用的理论负载(星型/流水线/辩论/map-reduce 薄共享对照/
  多来源单块 RAG)+ 单段 8-MiB = bank 级硬限制(上下文扩展走加 HBM
  数量,拆段作废);
- 相对 xinyao 基线的性质:以上全部为**新增语义/新档/新策略**,
  0aced82/47ae0c3 引入的原有路径(split 已废除者除外)行为不变,
  `--engine` 默认 analytic 保历史命令逐字节兼容。
