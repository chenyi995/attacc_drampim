# MQ 多 Q 批处理:分层硬件增量与全数据流(C3,2026-08-21,已过三路独立审计)

依据:Fugue 正文(§4.1.3 两侧拼装 / §4.3 die 四件 / §4.4 转 Q / §4.5 执行模型)+
AttAcc 原文(§4.2/§5.1/§7.7)+ 本仓库实测(results_c_points.json)。
默认配置:d_head=128、BF16、L=4096、(n_q,n_c)=(16,2)、每 channel 64 bank、命令时钟
1.3 GHz(0.769 ns)。**§5 为三路审计(带宽/数值/时序-buffer)结论;正文已按其修正。**

---

## ⚠️ 2026-08-24 设计修订:P 流式(streaming P,用户裁决)——本节取代下文的 n_c 驻留描述

本页以下正文与三路审计是 **2026-08-21 的 (n_q, n_c) 驻留设计**的存档记录。
2026-08-24 裁决:**P 不驻留**——P 的一个条目在本 bank 只被两个输出趟各用一次,
复用近零(Q 则被每个 K 列复用,必须驻留),所以"多条 P 对一段 V"与"一条 P 对
一段 V"同样都是流;驻留 buffer 买不到东西。由此:

1. **context 相一遍完成**,与 score 相同用 n_q;不再有 n_c 档与 ⌈n_q/n_c⌉ 趟
   重扫 V(§2.B 步骤 11–14、§1 表 GEMV buffer/累加器行、§3 表 context 行、
   步骤 60 的"故驻留、不流式"结论均被取代)。
2. **容量轴只约束 Q**:`mq_query_capacity = S/64`;P 以 MV_GB 流经半双工 TSV
   进双缓冲半区(现役块 32 B/查询),上限是移动总线带宽+方向转向
   (nRTW/nWTRL),不是 buffer 容量。平衡条件(推导+实测闭合,误差 3–5%):
   当前命令序(同 BG 连发,nCCDL=4 tCK)下 **n ≤ interval** 即不拖慢 V 扫描;
   跨 BG 交错可回到 nBL=2 tCK、翻倍为 n ≤ 2·interval(开点,未动)。
3. **context 侧硬件**:n_c 条 P 驻留区 + 2×n_c×64 B 累加器 → 每查询一组
   16×FP16 累加寄存器(n_q×32 B)+ P 现役块 n_q×32 B×双缓冲;buffer 需求
   与 score 区同为 n_q×64 B,两相统一。
4. **实现与数据**:`run_c_points.py`(context 单遍,n_q ∈ {8,16,32})、
   `results_c_points.json`(2026-08-24 重测,加速比全线上修,如 n=16@1.3 GHz
   2.90×→4.52×)、`src/ramulator_wrapper.py::mq_query_capacity` docstring、
   trace 生成器 phase 注释。汇总见 `docs/README_mq_design_space.md`
   §3–§5、§7 第 6–7 条(含"P 跨两输出趟的送法"未裁决项)。

---

## 1. 分层硬件增量

| 层 | AttAcc 原有 | MQ 之后 | 增量 |
|---|---|---|---|
| **bank PE 数据通路** | 16×FP16 乘 + 加法树/累加器,666 MHz | 算术**不变**;全流水后提频(1.3–2.08 GHz 档)。轮转结构天然无 RAW 冒险:同一 Q 的同一累加器两次更新相隔一整个命令间隔(≥5.4 ns),环延迟极宽松,无需前递 | 重定时(RTL 待证) |
| **bank PE 操作数** | 每条 MAC-AB 直接消费列数据 | **列操作数锁存器 32 B** | +32 B |
| **bank PE 状态**(审计后修正) | 1 组部分和 | ① score 运行部分和 **2×n_q×4 B**(ping-pong,行界搬运期不可覆盖);② **行界分数暂存 n_q×32 B**(每 Q 该行 16 个 FP16 分数,候 MVSB 排空;含排空重叠再 ×2);③ context 累加器 **2×n_c×64 B** | (16,2) 合计 ≈ **0.9–1.4 KB/bank**(原稿 192 B 为低估) |
| **GEMV buffer** | 双缓冲 16×256-bit(有效 512 B) | score 区 n_q×64 B / context 区 n_c×L/8 B 分相复用;**双缓冲另一半用于趟间 P 预装载**(消 8–13% 串行,见 §5-C) | 有效 ×2((16,2))/×4((32,4)) |
| **PE 控制** | 单流 FSM | Q 槽轮转计数(mod n);n_q/n_c 走 `PIM_SET_CONFIG`(原本就带 batch size) | 极小;**命令集不变** |
| **BG 累加器 / GBUS** | 空间归约 4 bank | 不变;(Q,token) 时分复用。**MVSB 输出次序=Q 槽固定序,die 侧按序落位(或加 Q-id tag,见 §5-B 风险)** | 无 |
| **TSV / 上下行** | 每 sweep 1 份 | 分数上行 ×n_q、P 下行合计 ×n_q | 无新硬件;两头流水叠加时合计 ≈47–50%(§5-A 风险) |
| **die:softmax buffer** | 每 head 一条向量 | 驻 n_q 条 per-agent 向量(=Fugue §4.5.3 的 batch 上界):n_q=16 双头流水 256 KB<512 KB ✓;**n_q=32 双头流水 512 KB 顶格** | 容量诉求归 E4 |
| **die:softmax 单元 / Fugue 四件** | 按 AttAcc/Fugue | 同一套单元轮流服务 n 个 agent;metadata 驻 ≥n_q(16 agent ≈14.7 KB) | 结构不变 |
| **链路 / GPU** | 每 agent Q 下行 + ctx 上行 | 每 agent 每 head 每步 256 B + 256 B,与单 Q 逐字节相同 | 无 |
| **DRAM 命令集** | 六条 PIM 命令 | **全部不变** | 无 |

面积(按 AttAcc §7.7 口径折算,含审计修正的寄存器):(16,2) ≈ **13.5% die**
(buffer ×2 + ~1 KB 寄存器/bank);(32,4) ≈ **17%**(超过论文 FP32 变体的 14.59%,
需 E4 权衡)。

## 2. 全数据流(一个 KV head 的完整旅程)

### 2.0 attach(一次性,per agent)
1. 上游策略给出复用 chunk 与重算集 D_i(Fugue §4.5.1)。
2. diff K/V 按 token-wise 位置 DMA 进 diff 通道;新 token 追加进 master(行放置表)。
3. driver 把逻辑位置装进 die 的 decoder metadata buffer(不进 DRAM)。
4. `PIM_SET_CONFIG` 写 n_q、n_c 与 Q 槽↔agent 映射。

### 2.A score 相(每 sweep 服务 n_q 个 agent;master 与 diff 通道并行)
5. GPU 每 agent 每 chunk 旋转 Q′ 变体(§4.4),n_q 份下行。
6. WRGB 装 n_q 条 Q 切片进 GEMV buffer score 区(128 条命令,被 MAC 间隔空槽吸收)。
7. 扫描:每行一次 `ACT_AB`;每列一条 `MAC_AB`——列数据读一次进锁存器,PE 在命令
   间隔内对 Q₁…Q_{n_q} 各做一次 16 路点积,入各自部分和。间隔 =
   max(功耗拉伸, PE 吞吐, 通路下限)。一个 token 的 K 由某 BG 的 4 bank
   按 head 维分摊,该 BG 独立产出该 token 分数。
8. **每个 bank 行界**(每 bank 16 token = 全 channel 256 token):各 (Q,token)
   分数经 BG 累加器上 TSV;MVSB ×n_q,**串行地板 ≈256·n_q cycles/sweep**
   (第二资源,见 §4)。
9. die 拼装:master 分数按 token 序写入该 agent 的 softmax-buffer 向量;diff
   通道同机制扫紧凑段,decoder 驱动写口按逻辑位置**覆盖** D_i(**需 diff-priority
   位或强制 diff 后于 master 过点**,§5-B 风险①)。
10. SFM ×n_q:每 agent 分数凑齐后**一次完整 softmax**(顺序流水,无 LSE),
    合计 ≈0.6 µs ≪ sweep。

### 2.B context 相(⌈n_q/n_c⌉ 趟,每趟 n_c 个 agent)
11. MVGB 装本趟 n_c 条 P 切片(每 BG 每 Q L/8 B);mask gate 在下行处按各
    agent 的 D_i 把 master 位置归零、改道 diff(Fugue §4.3③)。
    **双缓冲另一半可与上一趟扫描重叠预装载。**
12. V 扫描(修正后的口径):V 列向分割 + 累加器模式,**一个 32 B 列 = 一个
    token 的 16 个 d_head 维分量**,P[t] 标量广播给 16 lane;k_idx 沿 token 轴扫,
    n_idx(=2)选 32 个输出维中的哪 16 个,P 值跨两组输出复用(故驻留、不流式)。
    PE 对 n_c 条 P 轮乘,入 per-Q 的 32 元素累加器。
13. 趟末:per-Q 部分 context 经 BG 累加器 → die 级 adder(§4.3④)跨 bank、
    跨 channel、跨 master/diff 两侧求和 → 每 agent 每 head 的 context。
14. 换下 n_c 个 agent 的 P,重扫 V(K 读 ÷n_q、V 读 ÷n_c 的来源)。

### 2.C 收尾与续行
15. ctx 上行;GPU 投影/FFN 跨 agent 批权重,与下一 head 扫描重叠(AttAcc 流水)。
16. 新 token K/V 追加进 master(行放置表),decoder 位置表延长;回到步骤 5。
17. prefill:n_q 换成同一 agent 的 n_r 条计算 Q,分段落地保因果;重算 token 写
    diff 通道。**bank 整段 prefill + MQ 批时,批内 token 相互的下三角不在扫描里,
    归属需裁决**(§5-B 风险②;仓库现行 split 模式由 GPU 算 fresh×fresh,无此洞)。

## 3. buffer 需求表((16,2)、L=4096;审计修正后)

| 结构 | 需求 | 结论 |
|---|---|---|
| 列锁存 | 32 B/bank | 微小 |
| score 运行部分和(ping-pong) | 2×16×4 = 128 B/bank | 微小 |
| 行界分数暂存(MVSB 排空) | 16×32 = 512 B/bank(重叠排空再 ×2) | **主要新增寄存器** |
| context 累加(ping-pong) | 2×2×64 = 256 B/bank | 微小 |
| GEMV buffer 有效 | max(1 KB, 1 KB) = 1 KB(×2) | die +1.4%((16,2)) |
| softmax buffer(die) | n_q×L×2 B/head(P 原位覆盖分数):16→128 KB,双头流水 256 KB ✓;32→顶格 512 KB | n_q=32 需 E4 定容或弃双头流水 |
| decoder metadata | 16 agent × ⌈0.15L⌉×12 bit ≈ 14.7 KB | ✓ |

## 4. 带宽 / 第二资源核对表(审计修正后)

| 项 | 值 | 判定 |
|---|---|---|
| 列读(MAC 流) | **非"定义性满载"**:实测效率 = interval 限速值的 76–79%(MVSB/MVGB 突发 + barrier + 刷新挤占);score(16)@1.3 有效 124 GB/s(上限 157) | 已按实测 |
| **MVSB 串行地板** | ≈**256·n_q cycles/sweep**(nBL=2/条 × 128·n_q 条);score 相时间 ≈ max(512·interval(n_q), 256·n_q)+开销。n_q=32、PE≥2 GHz 时 **MVSB 主导**(实测 (32,4)@3.2 超纯 MAC 链 +123% 的来源);**PE 提频回报递减的第二原因** | 新增关键行 |
| MVSB 上行流量 | 128 KB/8.5 µs ≈ 15 GB/s ≈ 36%/channel | ✓ |
| MVGB 下行 | ≈4.4 GB/s(11%) | ✓ |
| SFM | n_q 次 ≈ 0.6 µs(7% of sweep) | 富余 |
| 两头流水叠加 | head i 的 MVGB + head i+1 的 MVSB ≈ 20 GB/s ≈ **47–50%** | TSV 上下行是否分离 → E4 |
| 链路 | 每 agent 不变(256 B + 256 B /head/步) | ✓ |
| 刷新 | REFab 每 5070 cyc、~+5.3%,已含实测(nRFC 换算口径与 tCK 存既有矛盾,~2% 级,非 MQ 引入) | 备注 |

## 5. 三路独立审计结论(2026-08-21;三个并行 agent,只读复核)

### A. 带宽审计
- 确认:MVSB 36% / MVGB 11% / WRGB 被遮 / SFM 7% / 每 agent 链路不变 /
  C++ 时序表证实搬运命令与 MACAB 无互约束(不同数据通路)。
- 修正(已入正文):"列读定义性满载"→ 实测 76–79%;"16-token 行界"→ 每 bank 行界。
- 风险:①两头流水 TSV 合计 47–50%,上下行是否分离待 E4;②MVSB 依赖 BG 交错
  布局(退化则 nCCDL=4/条);③decode 新 token KV 追加与 P 下行同窗(量小)。

### B. 数值正确性审计
- 确认:**每 agent 与单 Q AttAcc 逐位等价**(批只在其命令间插入他人 op;同槽
  累加间距反而更宽松);Eq.(step2)/(step4) 的 per-agent 语义成立;P 驻留
  L/8 B 复核一致;仓库 prefill 分组不越因果。
- 修正(已入正文):①score 侧寄存器补行界暂存 n_q×32 B;②V 列内容口径
  (一列=一个 token 的 16 维,非 16 token 各一维)。
- 风险(Fugue 级,n 批放大):①**diff/master 写序竞争**——diff 段短可能先到,
  master 后写会盖掉 diff;需写口 diff-priority 位或调度强制 diff 后行;
  ②bank 整段 prefill + MQ 的**批内下三角归属**(GPU/die 补小三角,或批界对齐段界);
  ③MVSB 的 Q 槽识别要么固定序要么带 tag,实现须写明。

### C. 时序与 buffer 审计
- 确认:interval(n) 作为 MAC 链模型无漏项(nFAW 不挂 ACTAB;行切换非瓶颈;
  刷新已含);PE 轮转天然无 RAW 冒险,无需前递;GEMV 双缓冲恰好支撑趟间预装载;
  metadata 14.7 KB ✓。
- 修正(已入正文):①**MVSB 串行地板 256·n_q**(整 sweep 的第二资源;"interval
  逐点吻合"只对 cycles/row 微基准成立);②相位相加式偏保守——趟间 MVGB 可被
  双缓冲重叠,实测 C3 为上界,还有 **~8–13% 余量**;③per-bank 寄存器 ping-pong ×2。
- 风险:①**甜点在 (16,2)–(32,4) 之间**:n_q≥32 时 MVSB 地板与 softmax buffer
  512 KB 顶格同时出现,再大需 MVSB 打包(一条携多 Q)或 buffer 加倍——新硬件;
  ②PE >2 GHz 在 1z DRAM 工艺的可综合性(且其价值随 MVSB 地板出现而贬值);
  ③nRFC/tCK 换算口径矛盾(全局 ~2%,既有问题,建议一并修)。

## 6. 三个裁决项的处置建议(2026-08-21 查证,待用户裁决)

1. **diff/master 写序 —— 已实装(2026-08-21,用户裁决采纳)**:per-agent D_i 位图
   做 **master 端写过滤**(命中 D_i 即丢),diff 经写口直写,到达顺序无关;与
   mask gate 同源信息。仿真器:attach/prefill 时新增 `di_bitmap_gpu_to_die`
   (LINK,⌈context/8⌉ B)与 `die_load_di_bitmap`(DIE)事件,EPIC 每 agent 一次、
   CacheBlend 每 partial 层一次,层内含掩码的扫描依赖位图装载;报表含
   `di_bitmap_bytes` 与机制说明。RTL 落点:E4 位图 + master 写路查询口;
   论文 §4.3.2 一句(待写)。
2. **prefill 批内下三角 —— 已实装(2026-08-21,用户裁决采纳)**:新模式
   `--pim-prefill-mode bank-whole`(默认仍 split):本层 fresh/corrected K/V
   **先落地**(store 事件为扫描的前置依赖),每个 ≤cap 的 Q 子批扫**全落地范围**
   (read-mask 的 master + 全部 fresh 行,行数 +s,上三角被扫即被计费),DIE 以
   `die_score_assembly` 事件按 (position ≤ q_pos) 因果丢弃后逐 Q 单次 softmax——
   **无 GPU 三角、无 LSE tuple**。校验器新增 assembly 规则并将其纳入 context
   返回检查;单测
   `test_bank_whole_prefill_lands_kv_first_and_loads_di_bitmap` 锁住
   落地顺序/无 GPU 三角/位图计数三件事。RTL 落点:E4 一个位置比较器;
   论文 §4.5.2 一句(待写)。
3. **TSV 上下行 —— 已实测,结论:不是墙,窄下行不需要**(2026-08-21,
   chenyi-experiment-821 分支)。查证:WRGB/MVSB/MVGB/SFM/RD/WR 在 pCH 级本就
   互相按 nBL=2 串行(共享半双工,与真实 TSV 一致)。已给模型补上方向转向约束
   (MVSB→MVGB/WRGB = nRTW,反向 = nWTRL,复用 JEDEC preset 值、YAML 可覆盖;
   HBM3-PIM.cpp 与 pim_ramulator_src 同步)。两头同 channel 流水合成实验
   (run_pipeline_overlap.py,L=4096、MQ、n∈{1,4,8,16}):
   **JEDEC 默认转向 (3/11) 代价 ≤0.84%,×4 夸大 (12/44) 也只 ≤3.8%**——
   搬运命令藏进 MAC 间隔空槽,方向切换次数天然少。附带发现:同一 channel 上
   两头流水对 MQ sweep 基本中性(0.98–1.03×),因为 channel 命令总线本身把两头
   的 MAC 流串行化;head 流水的价值在跨 channel(AttAcc 正常布局)。
   处置定案:模型保留转向约束;**错峰调度与专用窄下行均无数据支持,不做**。
