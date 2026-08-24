# 块 04:C 系列——MQ-MAC 批命令、非对称相位、D_i 位图、bank-whole、总线转向

归属:宸逸+助手会话(2026-08-21,分支 chenyi-experiment-821),四个 commit:
`9d6fc7b`(MQ-MAC + 相位 + 审计文档)、`264d14a`(总线转向 C++)、
`3e338e6`(D_i 位图 + bank-whole prefill)、`711ae25`(严格论文模式)。
设计计划见 `docs/PLAN_mq_command.md`,数据流与三路审计见
`experiments/mq_command/DATAFLOW.md`。

## 1. 动机(零基础,算子维度)

多 agent 共享一份 KV 时,decode 每步有 n 条查询要对同一 K/V 算
score/context(GEMM 视角:Q[n×d_head]·K^T[d_head×L])。上游命令集只有
"一条查询一次扫"(replicate):n 条查询就把 KV 读 n 遍——列读 (column
read,一次从 bank 行缓冲读出 32 B) 是 DRAM 里最贵的动作,白读 n−1 遍。
**MQ (multi-query) 批命令**:一次列读,bank PE 对 n 条驻留查询各做一次
16 路 FP16(半精度浮点,2 字节/数)乘加 (16-lane MAC)——
列读×1,乘加×n。查证:bank 级 PIM 文献无人做多查询共享列读
(LongSight, MICRO'25 明写 "batching…no reuse due to lack of shared KV";
检索记录见 `docs/LOG.md` 2026-08-21 条 3)。

## 2. `9d6fc7b`:MQ-MAC 三层实装

### 2.1 trace 层(`ramulator2/trace_gen/gen_trace_attacc_bank.py`)

`--mq` 旗标(`:557`):共享扫描时 `PIM_MAC_AB` 只发一次,查询私有的
搬运/写命令(`PIM_WR_GB` 写 Q 切片、`PIM_MV_SB` 搬 score、`PIM_MV_GB`
搬回 P、`PIM_SFM`)仍每查询一份。测试
`test_mq_trace_reads_each_column_once`:replicate 的 MAC 数 = 4×mq(4 查询),
共享外命令逐条相等。`--phase {full,score,context}`(`:552`)把 trace 切成
两相(§3)。

### 2.2 时序层(`src/ramulator_wrapper.py:32-50`)

`mq_interval_cycles(n, power_constraint, f)`:一次 MQ 列读命令的有效
nCCDAB = max(功耗拉伸(由 IDD7——DRAM 规格书里全 bank 激活工况的电流
预算参数——推出的最小间隔), PE 吞吐 ceil(n/(f·tCK)), 通路下限 4)。
微基准实测定标:70 cycles/行平台期 → 免费 MAC 槽 4/5、基准 nCCDAB 6/4
(功耗受限/不受限;`docs/README_design_check.md`)。该值经 YAML `nCCDAB`
覆盖写进 Ramulator 预设,C++ 不改语义。两轴(容量 vs 速率)的完整展开
见 `docs/README_mq_design_space.md` 与本目录块 06。

### 2.3 集成层(`src/workload_runner.py` / `main.py`)

批 decode 与 PIM prefill 的共享扫描给 Layer 打 `pim_shared_queries`/
`pim_batch_command` 标记;sweep 超过 GEMV buffer 驻留容量
(`mq_query_capacity`)时**拆成连续多趟**(同一批行重扫,测试
`test_batched_decode_splits_sweeps_at_the_gemv_buffer_capacity`)。
CLI:`--pim-batch-command {mq,replicate}`,**mq 是 main.py 默认**;
代码内部 Layer 默认仍是 replicate 保回归(`HANDOFF.md` §3.5)。

## 3. C 编号与非对称相位

- **C1 compact**:AttAcc 原样(单查询扫描);
- **C2 多通道**:复制 KV 到 k 个通道换并行(存储 k 倍,等延迟拷贝数
  实测记录在 `results_c_points.json` 的 `c2_*` 字段);
- **C3 非对称 MQ**:score 相位(K 流读,间隔=PE 吞吐主导)与 context
  相位(V 流读,间隔≈通路/功耗下限)**分开定间隔**——
  `--phase score|context` 两段 trace 拼接,测试
  `test_phase_slices_partition_the_full_trace` 保证两相并集=全 trace。
实测(`experiments/mq_command/run_c_points.py` →
`results_c_points.json`):C3@1.3 GHz 相对 C1,(16,2) 2.66×、(32,4)
3.63×;全 bank 行激活 512→144 次(见 json `c1/c3_act_allbank`)。

**在论文中的意义**(C1/C2/C3 与消融,映射见 `docs/EXPERIMENTS.md`):
C 系列支撑论文的**微架构与 die 面积章节**(E4 方向)。C1 是微架构基线,
C2 证明"复制换并行"要 k≥8 份拷贝才在延迟上追平且能耗不降(排除项),
C3 是论文主张的微架构,头条数字(L=4096,功耗受限):
**每 agent 1.71 µs、3.63× vs C1、列读/行激活 ÷7.1、容量 1×**。
C-abl-1(`run_mq_study.py`,96 点)是命令方案消融(MQ vs ×B 复制 vs
dense),支撑"为什么是 MQ 命令"的论证。

## 4. `264d14a`:搬运总线方向转向(C++ 唯二改动)

MVSB(bank→die,读向)与 WRGB/MVGB(die→bank,写向)共享**半双工**
(同一时刻只能一个方向)的 TSV(through-silicon via,3D 堆叠层间的
竖直互连)/全局总线,方向掉头有代价。补丁在 `ramulator2/src/dram/impl/
HBM3-PIM.cpp:421` 约束表加两条:`MVSB→{MVGB,WRGB}` 计 `nRTW`、
`{MVGB,WRGB}→MVSB` 计 `nWTRL`(复用 JEDEC——DRAM 标准组织——的
读写转向时序参数,YAML 可覆盖;`pim_ramulator_src/` 种子同步)。
配套实验 `run_pipeline_overlap.py`(**C-abl-2**):两个注意力头
(attention head) 在同一通道上流水合成,JEDEC 默认转向值下收益 ≤0.84%。
**在论文中的意义**:这是一次**设计裁决**——"窄下行总线"方案被该实验
定量关闭,论文不再提该方案;转向约束保留在 C++ 里保证 C 系列计时保守
(裁决记录 `DATAFLOW.md` §6)。

## 5. `3e338e6`:两个裁决项实装

这两项合称 **C-impl**(机制实装),分别对应论文 §4.3.2(写口的 D_i
过滤,到达顺序无关论证)与 §4.5.2(bank-whole 的因果丢弃)——机制已
实装,正文各欠一句措辞(`HANDOFF.md` §4.1)。

- **D_i 位图 master 写过滤**:agent i 的修正行集合 D_i 以位图
  (每 token 1 bit)传到 DIE(事件 `di_bitmap_gpu_to_die`/
  `die_load_di_bitmap`,`src/workload_runner.py:2339` 附近);master 侧
  对 D_i 位置的 score 写被丢弃,于是 diff/master **到达顺序无关**。
  EPIC 修正集层不变→每 agent 载一次;CacheBlend 逐层抽样→每层载。
- **bank-whole 因果丢弃 prefill**(`--pim-prefill-mode bank-whole`,
  `:2441` 起):本批 K/V 先落 bank(着陆序),每条查询扫**全范围**
  (含尚属"未来"的行),DIE 装配 score 时按"key 位置>query 位置"丢弃
  非因果项——一个比较器换掉 GPU 上的三角块和 LSE 元组;被丢的上三角
  照样被扫、照样计费。测试
  `test_bank_whole_prefill_lands_kv_first_and_loads_di_bitmap`。

## 6. `711ae25`:严格论文模式

实验编号收敛为**有且仅有 A 系列与 C 系列**;历史实验目录移入
`experiments/_archive/`;`run_asym_points.py` 由 C 版 `run_c_points.py`
取代;根部设计文档移入 `docs/` 并立四件套(README/EXPERIMENTS/
HANDOFF/LOG)与仓库根 `CLAUDE.md`(工作守则)。

## 7. 微架构对应(RTL 侧,交叉引用)

`fugue-logic-die-rtl`:`mq_bank_pe.sv`(列锁存/Q 槽轮转/per-Q 状态/
行界暂存)、`mq_diff_decoder.sv`(D_i 位图写过滤+因果比较器)、
`mq_score_store.sv`、12 点综合 sweep(**C-abl-3**,Genus——Cadence 的
逻辑综合工具——在 N28(台积电 28 nm 工艺)上出面积/时序/功耗,对照
本仓库复现的 AttAcc 基线)。**在论文中的意义**:C-abl-3 提供论文
die 面积章节的全部硬件数字口径(倍数/占比有效,绝对值须按 AttAcc 的
"DRAM 工艺疏 10×"折算,见块 06)。三路独立复核结论(`DATAFLOW.md` §5):
数值逐位等价;MVSB 串行地板 256·n_q cycles 是 PE 提频回报递减的第二
原因;寄存器开销 ~0.9–1.4 KB/bank。

## 8. 测试覆盖与悬置

- 覆盖:`MQBatchCommandTests` 5 例(间隔/容量、MAC 计数、相位切分、
  sweep 拆分、bank-whole+D_i)+ 全套 38/38 回归。
- 悬置:论文正文欠两句措辞(D_i 写口、因果丢弃)与 §4.5.3 "列访问随
  n_r 增长"的口径张力(MQ 下列访问不随 n 长,PE 操作才随 n 长)——
  见 `HANDOFF.md` §4.1。
