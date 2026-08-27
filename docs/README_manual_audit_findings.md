# 手动审计总台账(唯一合并版;chenyi9 2026-08-26 指令合并)

目标读者:接手人/审稿人;概念首现即释,本页自足。
本页是**全项目人工审计的唯一文档**。2026-08-26 起,原三份 audit 合并于此:

- 原 `README_audit_ladder_issues.md`(阶梯五题诊断)→ 本页附录 A;
- 原 `README_audit_workload_validity.md`(workload 有效性)→ 本页附录 B
  (**归档**:其审计对象数据源已按 2026-08-25 指令删除);
- 原 `README_audit_0825.md`(bank 级数据流口径一致性核对,结论全一致)
  → 其唯一遗留(softmax buffer 第二容量上限)立为本页 U6;
- 原文件存根/全文均移入 `archived/`(2026-08-26 目录重组),git 历史
  保留。

结构:一、已解决(R);二、未解决(U);三、流程性裁决记录;附录 A/B。

## 一、已解决(17 条)

| # | 问题 | 修复 | 位置 / commit |
|---|---|---|---|
| R1 | **计算能量摊进 DRAM 列命令间隔**(结构性混账:每条驻留查询的计算能量拉长列读节拍) | C 模型分账:interval = max(preset 地板 6/4, PE 项),计算永不拉长 DRAM 节拍(FIMDRAM 先例),PE 功率单独记账 `mq_pe_power_w` | `src/ramulator_wrapper.py`;experiment `cdf8f9a` / 822 `e81c2e4`。**定量预算闭合仍开 → U2** |
| R2 | **diff/master 写序竞争**(diff 段短常先到,master 后写反盖) | per-agent **D_i 位图 master 写过滤**,到达顺序无关;`di_bitmap_gpu_to_die`/`die_load_di_bitmap` 事件 | `src/workload_runner.py`;experiment `3e338e6`(822 移植 5.L1) |
| R3 | **bank 整段 prefill 批内下三角归属**无定义 | bank-whole:K/V 先落地、每查询扫全范围、DIE 因果丢弃(一个比较器;上三角被扫即计费) | `--pim-prefill-mode pim` 分支;experiment `3e338e6` |
| R4 | **A6 dynamic 估价口径不对称**(xPU 路按每请求算子估价,比 gpu 档实际入账贵 → 偏选 PIM) | 估价改为与 gpu 档同一口径(顶层 scale 折算),估=入账两侧对称;multihop/65B 上 A6 = min(A4,A5) 验证 | `src/ablation.py::_prefill_batch`;`b649674` |
| R5 | **分池 8/8**(diff 行极少却独占一半带宽,A4 反劣于 A3)——即"channel 划分不是对半分"裁决 | 两条路径同步改 **15/1**(物理 `_KV_CHANNELS` master 0–14 / diff 15 + 解析默认 + `--kv-pool-split`);注释留"后续可按 diff 密度 ρ_b 自动定宽" | `b649674` |
| R6 | **naive 布局无 channel 冲突模型**(每 run 摊满 16 channel 理想并行,乱序零代价) | 解析:逐 chunk **顺序分配 channel 并追踪**,同 channel 冲突**串行化**(`_naive_channel_pools`,decode 取池间 max);物理:见 R11 的 `NaiveKVLayout` | `src/ablation.py`;`b649674` |
| R7 | **A5/A6 没挂微架构参数**(PE 0.666 GHz / 512 B / 每波 4) | preset 绑定平衡点 **2.6 GHz / 768 B(=12 驻留)**,标 PROVISIONAL;mq 下每波跟随 `mq_query_capacity` | `src/ablation.py::PRESETS`;`b649674`。**2026-08-26 修订:MQ max n_cap=8 → 512 B/1.733 GHz(R16)** |
| R8 | **压缩率列低估**:`_memory_report` 把共享 chunk 的属主副本双计(multihop 报省 4.7%,实际 20.3%) | 公式去重 + 单测锁死 + `owner_copy_fix` 标记;存量结果 `repair_memory_column.py` 修补(multihop → 0.798 ✓) | `src/ablation.py::_memory_report`;`0305d4c` |
| R9 | **多轮历史口径**(名义 `--history-len` 一开始就满长) | Mooncake conversation 去尾块链轮,history 逐轮从 0 累积;进矩阵为 `mooncakemt` | `convert_mooncake_multiturn.py`;`0305d4c`。(注:mooncakemt 后判不可用,见附录 B) |
| R10 | **A2 的 KV 驻留口径**:GPU-only 档 KV 原实现留在 GPU 本地 HBM,链路字节=0 | 裁决(chenyi9 2026-08-26):链路字节 = **GPU↔远端存储**(HBM 或哑 DRAM)经 NVLink/PCIe 的流量;**A2 的 KV 全放远端哑存储**——prefill 写出/复用与 history 行读回,decode 每步整上下文×全层拖回再写回(wl_tiny 实测链路 42.1 GB、makespan 1.671→1.811 s)。**解析引擎 A2 未对齐 → U3** | `src/workload_runner.py::_run_gpu_software_only`;`8b58fe7` |
| R11 | **物理 DAG 引擎只覆盖 ≈A4/A5/A6**(decode 事件词汇只有 PIM;KV 布局只有 master-diff 与 private;`--ablation` 强制走解析引擎) | 新增 `--engine {analytic,dag}`:`--ablation Ax --engine dag` 六档全通——A2 新 GPU-only 事件路径(含 R10 远端模型)、A3 新 `NaiveKVLayout`(块按软件序轮转 16 个单 channel 池,碎片惩罚由排程涌现)、A1 接通已有 private 物理路径、A4–A6 preset 布线(A1–A4 replicate,A5/A6 mq+2.6 GHz/768 B)。顺带修:整层复用零修正请求致 qkv m=0 除零(两分支加护栏);A2 不再套 split 时代事件命名校验器。wl_tiny 六档验证:A6=min(A4,A5) 逐字节成立;**A5 语义分叉 → U4** | `main.py`、`src/workload_runner.py`;`8b58fe7` |
| R12 | **prefill attention 归边统计口径混用**(A6 列只数走到动态判决器的请求,分母与 A1–A5 不一致;A5 的"4/4"也虚——全新请求实际走 GPU 分支;chenyi9 发现) | 统一为**事件流实际行数**统计:每请求记 PIM/GPU 注意力行数,share = PIM 行/(PIM+GPU) 行;普查列 pim/gpu/mixed/**none**(none=整层复用零修正、prefill attention 未发生),分母恒为全部请求 | `_prefill_side_summary` + `collect_dag_ladder.py`;`8b58fe7`/`c8e39c9` |
| R13 | **Ramulator 签名缓存纯进程内存**("先 cache 再跑"跨进程不成立,每次运行重付全部仿真) | 裁决(chenyi9 2026-08-26):cache 阶段最多 64 核。签名缓存落盘 `<ramulator_dir>/signature_cache.jsonl`(append-only JSONL,启动即载);`--ramulator-workers 64`;解析路径预热池 32→64。实测:六档首建 ~32 min,暖缓存复跑 2 min 11 s | `src/ramulator_wrapper.py`、`src/ablation.py`;`4e582cb` |
| R14 | **naive 布局三处失真**(块数可调造惩罚;大段整段挤单 channel;共享块一次性连续引入抹掉混叠) | 裁决落地(chenyi9 2026-08-26):①共享内容按**自然 256-token 块**;②`NaiveKVLayout` **页化**——一切预留(含 history/live 大段)先切 256-token 页再按 append 序轮换,大段 30k 行实测摊满 16 channel;③共享内容**分轮引入**(首用定 append 位,混叠只由负载产生) | `94f8f46` |
| R15 | **naive 白得 Fugue 的读掩**(mask 是 die 侧硬件特性) | `shadow_reads` 拆分:**A3 无掩断流**(重算行处 master run 劈开:act 段-act 行-act 段)/**新档 A3a 可掩不断流**(`NaiveMaskKVLayout`);gpu-prefill 读回补 **DRAM 侧读事件**(散页在 prefill 也收费);阶梯变七档 | `6b05a22` |
| R16 | **decode batch=1 埋没全部布局差异**(权重流按并发数虚增,PIM 扫描全藏其下;AttAcc 原版本就重叠) | **全档 decode 服务批宽 8**(GPU 权重一遍服务全波,KV 每查询各自拉;A2 重写为同构波结构);**MQ n_cap 顶格 8**(512 B/1.733 GHz 配平,修订 R7 的 12 档;MQ 仅 A5/A6,其余档连 prefill 也不开);wl_tiny 实测 A2 1.81→1.51 s、A4 1.29→0.98 s | `c163936` |
| R17 | **重算 token 取块头与"随机抽取"裁决不符**(EPIC 前缀是该策略本义,不许动) | **新增独立策略 `--reuse recompute`**:每位移块内**均匀随机抽 k 个 token**(种子可复现;归 EPIC_FAMILY 复用逐段记账/diff/位图机制);随机位置在物理上只影响 A3(断口散布);ladder A2–A6 改用之,比例轴 k∈{2,4,8,16,32} | 工作树(2026-08-27 提交) |

另有设计裁决(非 bug,已落地):老 A6"split 混合 prefill"废除、物理 DAG 与 A 阶梯同菜单(`654aeee`/`0755694`);TSV 窄下行经实测关闭(转向代价 ≤0.84%,experiment C-abl-2)。

## 二、未解决(6 条)

### U1:TLB 描述符 5 ns / 0.1 pJ —— 未溯源常数 + 重复收费;时序可 overlap(已分析,未修改)

- **位置**:`src/workload_runner.py::_TLB_DESCRIPTOR_S = 5e-9`、
  `_TLB_DESCRIPTOR_ENERGY = 0.1`(pJ/条;2026-08-26 补记:**能耗常数与
  时延常数同样无出处**)。`_tlb_plan_cost` 每次 PIM 扫描按物理连续段数
  收费,五处调用:prefill sweep、decode 单/批路径(两变体)、A6 动态
  估价器 bank 侧。
- **问题**:(a) 两常数无测量、无文献、无推导;(b) 论文口径是 attach 时
  一次性装载 decoder metadata,扫描查常驻表不另收费,仿真器却每扫描每
  run 收一次,拆趟后重复收。
- **时序 overlap 分析(chenyi9 2026-08-26 要求;只分析未修改)**:
  当前依赖链 `die_query_transform → tlb_lookup_and_bank_plan → pim_scan`
  把 TLB 串在关键路径上。但描述符规划只依赖**绑定元数据**(扫描位置
  集,KV 落位时即已知),不依赖 Q 的内容——原则上可提前到与
  `q_gpu_to_pim` 链路/DIE 变换并行;decode 相邻步的扫描集只差一行
  (尾部追加),描述符可在**上一步扫描期间增量更新**,摊薄到近零。
  量级:wl_tiny/A4 全程 TLB 能耗 2.0416 nJ ÷ 0.0001 nJ/条 = 20,416 条
  × 5 ns ≈ **102 µs 未重叠时间,对 1.29 s makespan ≤0.008%**——修与
  不修都不影响既有结论,但口径上应当能整段藏掉。
- **关闭条件**(待裁决):保留+标注(现状);改 attach 一次性装载事件
  /实装 overlap;或用 `kvpim-rtl` 的 `mq_diff_decoder`/metadata 路径
  综合反标出有依据的数。

### U2:总能量预算的定量闭合(R1 的遗留半条)

- **位置**:`src/ramulator_wrapper.py::MQ_POWER_BUDGET_W = 116`。
- **问题**:分账结构对,但 (a) 116 W 从 AttAcc Fig.7(a) 人工读图,IDD7
  绝对电流不公开;(b) cell 侧微观能量对宏观 116 W 两口径不能闭环;
  (c) preset=6(PC) 来源未推导。**chenyi9 2026-08-23:近似方案,不算解决。**
- **关闭条件**:JESD238 IDD7 环定义或厂商电流值,或 HBM-PIM/Newton 功耗
  拆分,把预算按"列读/激活/背景/PE"分项立账。

### U3:解析引擎的 A2 未对齐"KV 远端"裁决(R10 的另一半)

- 解析路径 A2(`kv_mapping=none`)仍是"KV 在 GPU 本地、零链路"旧口径;
  已跑矩阵的 A2 列同此口径。**处置待裁决**:解析引擎同步改远端模型并
  重跑 A2 列,或矩阵 A2 列重标注口径。

### U4:DAG 引擎 pim 档的"全新请求走 GPU"与解析 A5 语义分叉

- 物理引擎老规则:某层**无驻留行可扫**即走普通 GPU prefill("不伪造
  PIM 流量"),故全新请求在 `--pim-prefill-mode pim` 下整个 prefill 在
  GPU;解析 A5 语义是"全部 prefill attention 按 PIM 计价"。R12 的新
  统计使其显形:wl_tiny/A5 的 PIM 行占比仅 4.8%(普查 pim=2, gpu=1,
  none=1 / 4)。**处置待裁决**:pim 档对全新请求也走 bank-whole
  (先落 K/V 再扫),或维持现状并注明两引擎口径差。

### U5:workload 相关两条待裁决(原 W1/W2,详见附录 B)

- **W1**:矩阵中 mooncake/mooncakemt 行标 "EPIC k=8" 实为零重算,应改标
  "前缀共享对照"并从重算类 claim 引用中剔除;
- **W2**:`parent_out ⇒ shifted` 一刀切在会话续写下强制多算(保守方向),
  处置:维持+注明 / 细化条件 / 归入 history。
- (原 W3"过真实复用软件"方向已由 2026-08-25 修订作废并关闭:复用是
  编排层复用,无须真跑复用软件,详见附录 B。)

### U6(条件性):softmax buffer 未作第二容量上限

- 出处:`archived/README_audit_0825.md`(bank 级数据流口径核对,其余
  逐点全一致)。仿真器里驻留查询数 n_q 的上限只有 GEMV buffer 一条
  (`mq_query_capacity = gemv_buffer_bytes/64`);论文 §4.4.3 还有
  die 侧 softmax buffer 这条 "secondary limit"(n_q=16 双头 ≈256 KB <
  AttAcc 512 KB)。当前扫过的 n_q 档均先被 GEMV buffer 绑定,结果不受
  影响。**触发条件**:未来扫大 `gemv_buffer_bytes`/n_q 时须补
  softmax-buffer 封顶,届时再裁决。

## 三、流程性裁决记录(chenyi9;账户名记名规则)

| 日期 | 裁决 |
|---|---|
| 2026-08-25 | 分池 15/1(R5);A5/A6 绑平衡点 2.6 GHz/768 B PROVISIONAL(R7);一个任务 ≤32 核 |
| 2026-08-26 | **真实 workload 运行默认走 DAG 事件依赖路线,不能纯走解析模型出数**(`--engine dag`,R11 后六档全可走 DAG;解析模型只作快速预估与两引擎交叉校验) |
| 2026-08-26 | **Ramulator 先用 ≤64 核把 cache 建好再跑**(签名缓存落盘;替代 32 核上限,仅此阶段) |
| 2026-08-26 | **链路字节定义 = GPU↔远端存储(HBM 或哑 DRAM)经 NVLink/PCIe 的流量**;GPU↔GPU allreduce 不计入 |
| 2026-08-26 | **A2 的 KV 放远端哑存储**(R10;解析侧遗留 U3) |
| 2026-08-26 | 自造 workload 放 `<repo>/workload/`;输出放 `<repo>/output/<时间戳>_<负载>_<模型>/`;`/data2/chenyi9/KV-PIM/workload/` 仅作真实负载源数据暂存区(准入标准见其 README.md) |
| 2026-08-26 | prefill 归边统计统一分母(R12);TLB overlap 只分析不修改(U1) |
| 2026-08-26 | **256-token 自然块**;naive 冲突只许轮换自撞;共享内容分轮引入;**8-MiB K 窗口 = bank 级硬限制不许动,上下文扩展走加 HBM 数量**(拆段方案作废) |
| 2026-08-26 | **七档一起跑**(A3a 加档);**MQ max=8 且仅 A5/A6**;decode 服务批宽 8 全档默认;重算比例轴 k∈{2,4,8,16,32},A2–A6 用 `recompute` 策略 |
| 2026-08-27 | **口径注记:A2 的远端布局语义属 A3a 类**(GPU 可掩,无断流问题),但建模上远端为哑存储、读写仅按字节÷链路带宽计价,页布局/行激活未建模(瓶颈在互连)——A2 收的是逐 token 整上下文过链路的账 |

---

## 附录 A:阶梯五题的完整诊断(原 README_audit_ladder_issues.md,2026-08-25)

对象:`experiments/paper_ladder/` 首批 ~49 个作业核对(`CLAIMS_CHECK.md`)中
不支撑/反向的条目。每条:现象 → 用户口径 → 代码实际 → 归因(attacc /
xinyao(xw338)/ chenyi(Allan)账户)→ 修复方向。**全部已按 2026-08-25
裁决落地**(对应 R4–R8),此处保留诊断原文以备追溯。

### A.1 A6 dynamic 没贴住便宜侧(→R4,已修)

- 现象:multihop/65B TTFT:A4 98.3 s、A5 149.3 s、A6 134.5 s;7B 上 A6
  整体选了 PIM。
- 代码实际:dynamic 块 t_bank 估价与 A5 入账一致,但 t_xpu 按每请求算子
  构造、与 gpu 档批级折算口径不一致,系统性把 xPU 路估贵。
- 归因:chenyi `654aeee`(自造 xPU 估价而非复用 gpu 档入账函数);物理
  DAG 估计器无此问题。
- 修复:t_xpu 改调 gpu 档同一定价路径,估=入账对称;A6 全列重跑。

### A.2 A5 定位=强制全上 PIM 的对照档(叙述修正,非 bug;已落)

- A5 在全部已完成 workload 上 TTFT 劣于 A4(multihop/7B +42%)。用户
  口径:prefill attention 也 memory-bound,但全压 PIM 也不行——正是要
  dynamic 的原因;A4/A5/A6 三档并排即"prefill 放哪"对比,A6 报选边
  比例即可。CLAIMS C1d 已改写;比例字段双路径提取。

### A.3a 分池 8/8(→R5,已修)

- 现象:A4 TBT 反劣于 A3(multihop/65B 22.3→28.5 ms)——master 只有
  8 条 channel,主扫描流带宽减半;diff 行(EPIC k=8)极少却独占 8 条。
- 代码实际:物理 `_KV_CHANNELS` 固定 8/8(无旋钮);解析
  `master_pool_channels` 默认 8。
- 归因:xinyao——物理 8/8 在 `0aced82`,解析默认 8 在 `47ae0c3`;
  chenyi 矩阵驱动沿用未调。
- 修复:15/1 两路同步;池宽纳入矩阵旋钮。

### A.3b naive 布局看不到碎片惩罚(→R6,已修)

- 现象:A3 TBT ≈ A1(22.335 vs 22.393 ms),乱序零代价。
- 两层原因:①段边界拆 run 的 ACT/PRE 代价天然小(几十 ns 对 22 ms);
  ②真正该疼的 **channel 内冲突没建模**——`_runs_from_lengths(...,
  channels=16)` 让每个 run 摊满 16 channel 理想并行;legacy trace 的
  满宽批抽象是根源(attacc),naive profile 构造沿用(xinyao
  `47ae0c3`)。
- 修复:逐 chunk 顺序分配 channel 并追踪,同 channel 串行化(解析);
  物理侧 2026-08-26 由 `NaiveKVLayout` 实装(R11)。

### A.4 A5/A6 用老 PE 频率与老 buffer(→R7,已修)

- 现象:矩阵 A5/A6 bank 路 PE 0.666 GHz、buffer 512 B、每波 4。
- 三处叠加:preset 未耦合频率/buffer(chenyi `654aeee`);矩阵驱动未传
  旋钮(chenyi `81eedc9`);`pim_prefill_query_batch` 默认 4(xinyao
  `47ae0c3`)。
- 修复:preset 绑平衡点 2.6 GHz/768 B(PROVISIONAL);mq 下每波跟随
  容量;A5/A6 重跑。

### A.5 压缩率列低估——属主副本双计(→R8,已修)

- 现象:multihop `kv_bytes_vs_no_reuse` 报 0.953,实际可去重 20.3%。
- 代码实际:`private_rows` 已含属主副本,`reuse_layer_rows` 又加一遍
  `shared_rows`。归因:xinyao `47ae0c3`。
- 处置:公式去重 + 单测 + `owner_copy_fix: native` + 存量
  `repair_memory_column.py` 修补(multihop → 0.798 ✓)。

### A.6 修后单点验证(2026-08-25)

multihop/LLAMA-7B:A3 TBT 6.24 > A4 5.77(naive 惩罚显现、分池不再
反向);A5 TTFT 17.85 < A4 18.09(平衡点生效);A6 = min 侧且
pim_share=1.00。全量矩阵按新代码重跑(145/145)。

---

## 附录 B:workload 有效性审计(原 README_audit_workload_validity.md;**归档**)

**归档说明(2026-08-26)**:本节审计对象中被判不可用的数据源
(mooncake 三 trace、sharegpt、wildchat、azure、burstgpt 及 relay/
mooncakemt 产物)已按 2026-08-25 指令从
`/data2/chenyi9/KV-PIM/workload/` 删除;判定结论与待办(U5)保留于此。
新 workload 的准入标准定稿于 `/data2/chenyi9/KV-PIM/workload/README.md`
(A 真实性一票否决 / B 场景覆盖(multi-turn agent 重点)/ C 编排层复用
无 GPU / D 可核查 / E 实用性 / F 模型覆盖 MHA+GQA)。

**判据(chenyi9 2026-08-25 收紧)**:可用,当且仅当**真实 workload** 的
数据自身产生内容位移复用(同一 chunk 不同请求不同位置);规则强制重算
与合成设计品都不算。只有两档:可用 / 不可用。

| case | 内容位移 / 决策数 | 判定 |
|---|---|---|
| multihop | 13 / 48(doc 段,真实位移) | **可用**(gold evidence 充当检索结果,模拟器层面合理——chenyi9 确认) |
| relay | 0 / 8(位移 Δ−150 手写;合成品) | 不可用 |
| sharegpt | 1 / 43(唯一条=3-token 重复短句) | 不可用 |
| mooncake | 0 / 167(前缀链哈希,构造上无位移) | 不可用 |
| mooncakemt | 0 / 42(自行拼接;重算全为规则产物) | 不可用 |

三个 Mooncake trace 实测:所有块 id 前驱与位置均唯一 → 只能表达严格
前缀共享(脱敏所致,不可恢复)。不依赖位移的结论(decode 灾难、布局、
容量/压缩、前缀共享放置)在五 case 上仍有效。

**口径记录**:文本类 workload 的 token 计数当时用 tiktoken cl100k_base
代理(比值不受影响);2026-08-26 起新体检一律用真实模型 tokenizer
(D2 修订),cl100k 口径淘汰。

**待办去向**:W1/W2 → 本页 U5;W3(2026-08-25 修订:复用是编排层复用,
"哪一段复用什么"由编排/数据决定,复用软件真跑只决定哪几个 token 被
重算、不决定哪里被复用,具体 token 需要时按比例或个数随机抽取即可,
**无须真跑复用软件、无须 GPU**)→ 已关闭,并入
`workload/README.md` C0;2026-08-25 起的 14 源重找与体检见
`/data2/chenyi9/KV-PIM/workload/` 各文件夹 `BODY_CHECK.md`。
