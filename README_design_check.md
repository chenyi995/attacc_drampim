# Fugue 设计 vs kvpim-sim 仿真器:时序算术与差距清单(自足版)

写给**有计算机背景、但不熟悉 DRAM-PIM / KV cache 细节**的读者。所有设计表述以论文正文
(`KVPIM-1Fugue-ASPLOS2027/main.tex` + `sections/`,2026-08-21 版)为准;所有数字来自
本仓库 `xinyao_0821` 分支的代码与其携带的 Ramulator2 时序表。逐条的代码行号在配套的
`SIM_VS_PAPER_AUDIT_0821.md`;本文自成一体,不看代码也能读懂。

---

## 1. 名词与符号(先读这节,后文不再重复解释)

### 1.1 系统与角色

| 术语 | 解释 |
|---|---|
| **LLM 推理的两个阶段** | **prefill**:一次吃进整段输入,产出每个 token 的 KV;**decode**:每步生成 1 个新 token,要读之前全部 KV。 |
| **KV cache** | 注意力 (attention) 为每个 token 缓存的 key/value 向量,按层、按 KV 头存;decode 每步都要整读一遍,是容量与带宽的大头。 |
| **agent** | 一个推理请求方(多智能体工作流中的一个成员)。多个 agent 大量共享相同上下文,是本文场景。 |
| **chunk** | 非前缀复用 (non-prefix reuse) 的单位:几百个连续 token 的 KV 作为一个对象缓存(CacheBlend 用 512)。 |
| **CacheBlend / EPIC** | 两种上游"复用+选择性重算"策略。CacheBlend:每层挑 KV 偏差最大的 5–15% token 重算(**deviation policy**,重算点孤立分布);EPIC:每个 chunk 边界固定重算 k 个 token(**chunk-boundary policy**,重算点成簇)。 |
| **ρ_b (diff density)** | 被重算 token 占共享上下文的比例。 |
| **master / diff** | Fugue 的存储两分:**master** = 每个 chunk 的一份密集共享拷贝;**diff** = 每个 agent 自己重算的那些 token 的 KV,紧凑打包成该 agent 的一段 (segment)。 |
| **AttAcc** | ASPLOS'24 的 bank 级 PIM 注意力加速器,是 Fugue 的衬底 (substrate):GPU 管线性层/FFN,PIM-HBM 管 decode 注意力。本仓库就是 AttAcc 官方模拟器的改造版。 |
| **kvpim-sim** | 论文方法学里我们的事件流编排器 (event-stream orchestrator) 的名字,即本仓库:每个操作是一个带依赖的事件,无依赖即可重叠;注意力命令流交给 Ramulator2 逐周期仿真。 |
| **Ramulator2** | CMU-SAFARI 的周期级 DRAM 模拟器;本仓库带一份加了 HBM3-PIM 命令集的补丁版(`ramulator2/`)。 |
| **TTFT** | time to first token,prefill 的延迟指标。 |
| **RoPE** | rotary position embedding:把 token 的绝对位置旋转进 Q 和 K。因此**存下来的 K 字节里带着位置**;Fugue 的做法是不动 K、把位置差旋进 Q(GPU 侧完成)。 |
| **GQA** | grouped-query attention:g 个 query 头共享 1 个 KV 头(g=1 退化为普通多头 MHA)。 |

### 1.2 DRAM / PIM 硬件(HBM3 的组织,自顶向下)

| 术语 | 解释 |
|---|---|
| **HBM** | 高带宽堆叠内存。1 颗 HBM = 16 个 **channel**(通道);每 channel = 2 个 **pseudo-channel (pCH)** × 2 个 **rank** × 4 个 **bank group (BG)** × 4 个 **bank** = **64 个 bank**。 |
| **bank / row / column** | bank 是能独立开行的存储阵列。**行 (row)** = 1 KB(每 bank);读数据前要先**行激活 (row activation, ACT)**,把整行拉进感放 (sense amplifiers);之后按**列 (column)** 访问,每列 **32 B**;一行 = 32 列。 |
| **bank 级 PIM** | 每个 bank 旁放一个只会乘加的 **PE (processing element)**:16 路 16-bit **MAC**(乘加)。一条命令全通道 64 个 bank 同时执行。 |
| **MAC-AB** | "MAC all-bank"命令:64 个 bank 各读自己开着的行里的一列(32 B)并与各自 **GEMV buffer** 里的输入向量做乘加。一条 MAC-AB 共读 64×32 B = **2 KB**。 |
| **GEMV buffer** | 每个 bank PE 旁的小 SRAM,存"输入向量":score 相存 Q 的切片,context 相存概率 P 的切片。**WRGB** = 从外部写入它;**MVGB** = 从 die 的 softmax buffer 搬 P 进它。 |
| **softmax buffer / MVSB / SFM** | 逻辑 die(堆叠底部的 buffer die)上的分数缓冲;**MVSB** = 把 bank 算好的部分分数搬上去;**SFM** = die 上做 softmax。 |
| **ACT 相关时序** | **nRC**:同一 bank 两次 ACT 的最小间隔(row cycle);**nRCDRD**:ACT 到第一条读/MAC 的间隔;**nCCDAB**:两条 MAC-AB 之间的最小间隔(即本文的"列到列切换")。单位都是 DRAM 命令时钟周期 (cycle)。 |

### 1.3 数值符号(论文公式用到的)

| 符号 | 含义 | 本文取值/量纲 |
|---|---|---|
| d_head | 每个注意力头的向量维度 | 128 |
| b | 每元素字节数(BF16) | 2 B |
| L(论文写 L_ctx) | 被复用的缓存上下文长度 | token 数 |
| n_r(代码里叫 q 或 n_new) | 一次 prefill 实际计算的 token 数(重算的+新增的) | token 数 |
| g | GQA 组大小(每 KV 头对应的 query 头数) | MHA=1 |
| N_ag | 同时在场的 agent 数 | — |
| B | **一次扫描里共享同一份 KV 流的 query 条数(本文要定的"Q batch")** | — |
| d | 一段连续 KV run 落在一个 bank 行内的 token 数 | 1…16 |
| B_tok | 一个 token 全模型的 KV 字节数 = 2·L_layer·H·d_head·b(层数×KV头数×两个向量) | B |
| η | bank 侧注意力的实测效率(实测周期数 vs 理想 MAC 数) | 论文 \TBDnum |
| BW_link / BW_int | GPU↔PIM 链路带宽 / bank 级聚合内部带宽 | B/s |

### 1.4 本仓库特有的名字

| 名字 | 解释 |
|---|---|
| **两条代价路径** | ① legacy 消融路径(`src/ablation.py`):沿用 AttAcc 的矩形批公式模型,静态放置预设 **A1–A6**;② 物理 DAG 路径(`src/workload_runner.py`):逐事件、逐物理地址,交 Ramulator2 计时。 |
| **A1–A6** | A1=原版 AttAcc(私有 KV、无复用);A2=纯 GPU 跑复用软件;A3=PIM decode+朴素布局;A4=PIM decode+master/diff 分池;A5=prefill 注意力也全进 PIM;A6=split(GPU 算新行、PIM 扫复用行)。 |
| **论文的五级阶梯 (ladder)** | B0 GPU-only → B1 PIM-append → B2 PIM-split → B3 PIM-static → B4 Fugue(+AttAcc 参照)。与 A 系对应:B0≈A2、B1≈A3、B2≈A4、B3≈A5、**B4 无对应实现**(见 §3.1)、参照=A1。 |
| **TLB(本仓库语境)** | 软件侧"逻辑位置→物理 HBM 地址"的表(不是 CPU 的 TLB):master 固定放 channel 0–7,diff 放 channel 8–15,V 在 K 之上 8 MiB。 |
| **read-mask** | 被某 agent 重算覆盖的 master 行照常读出、但其分数被屏蔽(masked)、概率也不回流给它——与论文 Eq.(step2)/(step4) 的"diff 覆盖、master 归零"语义一致。 |

---

## 2. 问题一:列切换几个 cycle?计算几个 cycle?Q batch 该是多少?

背景一句话:AttAcc 原设计是 **GEMV**——GEMV buffer 驻**一条** Q,对整段 K 逐列乘加
(Q 复用)。Fugue 让多个 agent 的 Q 复用同一份 K,于是一行 K 开着的时候要给 **B 条** Q
轮流做乘加,变成小 **GEMM**;B 就是"Q batch"。

### 2.1 列到列切换 (column-to-column delay)

PIM 扫描由 MAC-AB 命令组成。相邻两条 MAC-AB 的最小间隔 = **nCCDAB**:

| 模式 | nCCDAB | 换算(1 cycle = 0.769 ns,1.3 GHz 命令时钟) |
|---|---|---|
| 有功耗约束 (power-constrained, `--powerlimit`,preset `HBM3_5.2Gbps`) | **6 cycles** | ≈ 4.6 ns |
| 无功耗约束 (NPC,preset `HBM3_5.2Gbps_NPC`) | **4 cycles** | ≈ 3.1 ns |

(对照:普通读命令的列间隔 nCCDS/nCCDL = 2/4 cycles;MAC-AB 是 64 bank 同时读+算,
受功耗墙限制所以更长。trace 生成器头部注释写的 "MACAB: 8tCK" 是旧注释,生效的是
Ramulator preset 里的 6/4。另注:preset 的 `tCK_ps=1300` 字段与包装器换算用的
0.769 ns/cycle 不一致——周期数不受影响,换算秒数以包装器 0.769 ns 为准,这是
沿袭 AttAcc 的口径。)

### 2.2 计算需要几个 cycle

**0 个额外 cycle**——AttAcc 的前提是 PE 吞吐匹配列带宽:一条 MAC-AB 的 6-cycle 槽内,
16 路 FP16 乘加流水完成,计算被列访问完全遮住。因此有用的换算只剩几何:

- 1 个 token 的 key 在 1 个 bank 里 = 64 B = **2 列** → 每条 Q 每 token **12 cycles**(PC);
- 1 个 bank 行 = 1 KB = 16 token = 32 列 → 每条 Q 每行 **32×6 = 192 cycles** 的 MAC,
  外加行激活到首条 MAC 的 **nRCDRD = 19 cycles**;
- 换行:受两条约束的较大者——同 bank 两次 ACT 至少隔 nRC = 63 cycles;走关行路径
  (末条 MAC →(nRTPL=8)→ PREA 关行 →(nRP=19)→ 下次 ACT,且 ACT→PREA ≥ nRAS=45)
  则至少 45+19 = **64 cycles**,后者是实际下限(实测含 refresh 摊销约 70)。

(几何依据,与论文 §4.1.1 一致:d_head=128、b=2 → 每 token 每 KV 头的 key 256 B,
BG 内 4 个 bank 按头维各拿 64 B;一条 MAC-AB 读 2 KB = 8 个 token;一条 channel 行
64 KB = 256 个 token。)

### 2.3 Q batch B 应该是多少

**下界(越过 ACT-bound 拐点)**。ACT 全速节奏的下限 = max(nRC, nRAS+nRP) = 64 cycles/行;
一行里能"白插"的 MAC-AB 数 = 1 + ⌊(64 − nRCDRD − nRTPL − nRP)/nCCDAB⌋ = 1 + ⌊18/6⌋ =
**4 条**(PC;NPC 为 1+⌊18/4⌋ = **5 条**)——插到这个数行切换节奏不变,再多每条加一个
nCCDAB。微基准实测(每行 k 条 MAC-AB,512/1024 行差分,含 refresh 的地板 70):
k=1…4 都是 70 cycles/行(NPC 到 k=5),之后斜率 ≈6.5(PC)/≈4.3(NPC)每条。

一段 run 在一个 bank 行里有 d 个 token(每 token 2 列),B 条 Q 轮流用这行 = B·2d 条
MAC-AB;越过拐点(不再被行激活卡住)的条件是

> 2·B·d ≥ 4,即 **B·d ≥ 2**(PC;NPC 为 B·d ≥ 3)。

| 场景 | d | 越过拐点需要的 B |
|---|---|---|
| master 整行扫描(chunk 的 256-token 行) | 16 | **1 就够**(B>1 的收益只剩 ACT 次数/能耗 ÷B) |
| EPIC 边界簇(k 个连续 token) | k | B ≥ ⌈2/k⌉ |
| CacheBlend 孤立重算 token | 1 | **B ≥ 2**(NPC 则 ≥3) |

(2026-08-21 更正:本节旧版写 "B·d ≥ 4",漏算了行尾 nRTPL+nRP 与 ACT 窗口的重叠;
按模型显式约束并经微基准验证,拐点如上。)

**上界(buffer 装得下)**。B 条 Q 在扫描期间要同时驻留:

- bank 侧 GEMV buffer,score 相:每条 Q 的切片 = **64 B/bank** → 共 64·B B;
- bank 侧 GEMV buffer,context 相:每条 Q 的概率向量 P 切片,一次性装载口径下
  = **L/8 B**(每 BG,4 bank 共享;L=4096 时 512 B/条)——这是 sizing 大头;
  若改成"开哪行喂哪行"的流式装载,可降到每行在飞一小片。**流式与否是必须写死的
  微架构口径,目前论文和 RTL 计划(E4)都没定**;
- die 侧:softmax buffer 要驻 B 份长度为 L 的分数向量 + decoder 的 metadata buffer
  驻 B 个 agent 的逻辑位置(每 agent ≈ ρ_b·L 条 × ⌈log₂L⌉ bit)。论文 §4.5.3 把
  batch 上界归给这两个 die 侧结构,数值 \TBDnum,由 RTL 实验(E4)给出。

**系统级(凑批要等 Q 到齐)**。`experiments/cacheblend_tier_batch` 的实测:在被测的
Q 到达模式下 B=2 makespan 最优(24.72 µs),B=4 接近(24.83 µs),B=8 因等待反而变差。

**结论**:最坏情形 (d=1) 的算术拐点是 **B=2**(PC)/**3**(NPC);仓库默认 B=4
(`--pim-prefill-query-batch=4`;decode 侧 `--cacheblend-batch-size` 实验常用 4)
= 每行 8 个 MAC 槽,稳稳在 MAC-bound 一侧;凑批实验的甜点在 2–4。**硬上界必须由
buffer 容量(E4 的 RTL sizing)给出,论文里对应 \TBDnum,按数据纪律先不填数。**

### 2.4 这套机制在仿真器里的现状(已实测验证)

- **执行语义已实现且与上述算术一致**:trace 生成器在 `shared_queries=B>1` 时把每条
  非同步命令原位重复 B 次(行/列在外层、Q 在内层);ACT 没有显式命令,由控制器在
  行缺失 (row miss) 时自动补发——因此一行只激活一次,B 条 Q 的 MAC 背靠背发,
  PE 每个命令槽都在干活,单条 Q 的复用(它仍扫完自己的全部 K 列)不被破坏。
  实测:L=512、1 头,B=1:930 cycles;B=4 共享:3553 cycles(对照串行 4×930=3720;
  MAC 数严格 ×4——印证"整行扫描本来就不缺乘法,共享省的是 ACT")。
- **三个缺口**:
  1. **论文正文没有这套机制的硬件承载**。§4.5.3 的执行描述逐句能对上(“the queries
     of several agents follow one another while the row stays open”“amortizes the
     same activations over its n_r queries, while the column accesses its MACs
     consume grow with n_r”),但 batch 上界只写了 die 侧,没写 bank 侧 GEMV buffer
     要驻 B 份 Q/P;而 §5.1 断言 “The bank PE keeps its MAC as the only arithmetic,
     the DRAM command set is unchanged / Everything else stays as AttAcc built it”——
     多 Q 至少需要 buffer 分槽 + 每条 MAC 选槽(不改命令集就得在 PE 里做 mod-B
     轮转计数),这半句和多 Q 批处理存在张力,需要一个明确口径。
  2. **仿真器没有容量约束**:B 是命令行旋钮,trace 里 B 次 WRGB 写的是同一地址
     (纯计时模型,没有分槽概念);论文说的 “beyond which the sweep splits”
     (超界自动拆扫)没有对应实现。
  3. trace 生成器里有个**无人调用的旧版共享 Q 函数**(`_shared_query_attention_commands`,
     interleave 方式与生效路径不同),属于死代码,易误导。

---

## 3. 问题二:仿真器与论文正文的差距清单

每条格式:**论文正文怎么说 → 仿真器现状 → 差距**。

### 3.1 ❌ 动态选边规则 Eq.(placement) ——B4/Fugue 的本体,未实现

- **论文**(§4.5.2):每个 prefill 按它的 (n_r, L) 现场比较两个时间——GPU 路
  t_xPU = max(L·B_tok/BW_link, n_r(F_tok+g·L·B_tok)/P_xpu)(传输与算力取大者,F_tok
  是每 token 线性层运算量、P_xpu 是 GPU 吞吐),bank 路 t_bank = n_r·F_tok/P_xpu +
  n_r·g·(L+n_r)·B_tok/(η·BW_int)(线性层加扫描,因为 Q 是扫描的输入,两段相加)——
  t_bank ≤ t_xPU 就把这个 prefill 的注意力放进 bank,平局归 bank;decode(n_r=1)
  恒在 bank,冷启动归 GPU。§5.4 的第五级 “Fugue adds the placement rule”,§6.1/§6.3
  的 Fugue 数字全依赖它。
- **仿真器**:全库没有任何 t_bank/t_xPU 比较。legacy 路径只有静态放置(A1–A6 里
  prefill 固定 gpu / pim / split);物理 DAG 路径永远是 split 形态。
- **差距**:B4 无法按正文口径跑。另注:`experiments/GPU_PIM_vs_GPU_prefill` 扫出的
  拐点(EPIC 每段重算 token 数 p ≤ 22–35 @NVLink3、≤ 89–210 @PCIe4;CacheBlend
  重算比例 r 上限 0.4–2.7%)是 **A4 对 A6** 的交点,对象是"协同 split",不是正文
  "整段二选一"的规则;规则实装后要按正文口径重扫。

### 3.2 ❌ 放置表 (placement table) 与 256-token 行粒度布局,未实现

- **论文**(§4.1.1+§4.2):chunk 写入时切成 256-token 的行,每行独立放置;软件维护一张
  行→channel 表,把"同一轮会被一起读的行"(包括同一 chunk 的两行)放到不同 channel,
  使它们的扫描并行;写入和发命令时都查这张表。动机(§3.1):一条 channel 一次只能
  服务一行,撞在同一 channel 的两行只能排队(row conflict)。
- **仿真器**:沿用 AttAcc 的按头条带 (head striping)——第 h 个注意力头放进池内第
  (offset+h) mod 8 条 channel,一个头的整段上下文顺序放在**一条** channel 里;分配是
  线性游标,没有任何防冲突逻辑;一次扫描以头并行占满整个池,于是**同池两个 chunk 的
  扫描永远串行**。
- **差距**:§4.2 整节在仿真器里没有对应物。多头、带宽饱和时两种布局总时间可能接近,
  但少 KV 头(GQA)时差距会拉大,且论文的 row-conflict 论证无法用本仿真器复现。

### 3.3 ❌ B0 (GPU-only) 的语义不符

- **论文**(§5.4):B0 把软件 master–diff 压缩存储放在**内存池里**(池只当内存用)、
  按追加顺序 (append order) 摆放,prefill 和 decode 的注意力都在 GPU 上算,**KV 每次
  过链路**。B0→B1 只改"算在哪"。
- **仿真器**(A2):KV 驻 GPU 自己的 HBM、零链路流量、无乱序惩罚;校验还强制
  decode-gpu 只能配 "无 PIM 映射"。
- **差距**:A2 是"KV 本地的纯 GPU",比论文的 B0 强得多(少了每步整读 KV 的链路
  代价)。按正文跑 B0 需要:KV 驻池 + 每层注意力的读回链路事件 + 池侧 append-order
  访问开销。

### 3.4 ❌ diff 通道配比 n_d ≈ ρ_b·f·C,未落地

- **论文**(§4.1.1, Eq.(ratio)):C 条 channel 里给 diff 的条数 n_d ≈ ρ_b·f·C
  (f = 共享 chunk 占已存 KV 的比例),"most channels serve masters and one or a few
  serve diffs"。
- **仿真器**:物理路径硬编码 master=ch0–7 / diff=ch8–15;legacy 路径可调但默认 8/8。
- **差距**:8/8 不但比例不对,还直接影响计时(diff 池带宽被高估、master 减半)。

### 3.5 ❌ GQA / 组大小 g 完全未建模

- **论文**:Eq.(placement) 里带 g;工作负载是 LLaMA-3-8B(GQA)+ GPT-175B 形状
  (§5.3);§7 有专段讨论 GQA 把 bank 侧优势缩小 g 倍。
- **仿真器**:模型表里 `gqa_size` 恒为 1 且没有任何代码使用它;没有 LLaMA-3-8B 条目
  (现有 LLAMA-7B/65B 是多头模型)。
- **差距**:g>1 时每 token KV 缩小 g 倍、每 KV 头 g 条 Q,放置公式、trace 形状、
  容量结论都会变;目前一概按 MHA 算。

### 3.6 ❌ 行激活数指标(Eq.(actcost))断在半路

- **论文**(§4.5.3+§5.4):decode 每 KV 头每步的行激活数 n_act ≈ n_row(1+ρ_b·N_ag)
  (n_row = 一份密集上下文的行数),对比密集布局的 N_ag·n_row,是正式指标之一。
- **仿真器**:本分支给 Ramulator 控制器加了 `pim_activations` 计数器(数实际下发的
  各类 ACT),但实测它**不出现在运行输出里**,Python 包装器也不解析;而且
  `pim_ramulator_src/` 这份"种子拷贝"没有同步该补丁——运行 `set_pim_ramulator.sh`
  会用旧文件把它覆盖掉(该脚本还有一个坑:本分支 `ramulator2/` 已是父仓库的普通
  目录,脚本里的 `git reset --hard` 会作用到父仓库上)。
- **差距**:指标链路 计数→输出→解析→报表 只有第一环。

### 3.7 ⚠️ die 上的合并机制不同(数值近似等价,硬件故事不同)

- **论文**(§4.3;2026-08-20/21 定稿口径):master 各 bank 的分数按 token 序写进
  softmax buffer,diff 的分数经 decoder 从**外部写口**按逻辑位置**覆盖**同位置的
  master 分数;凑齐后做**一次** softmax;明确"顺序流水、不需要 online softmax、
  die 不加 max/sum 寄存器";die 新增只有四样(decoder+metadata buffer、写口、
  概率回流上的 mask gate、context 加法器)。
- **仿真器**(物理 DAG):每个物理 run(master 流、diff 流)各自在 trace 内做完
  score+softmax+PV,输出局部 (max, sum, output) 三元组,die 上用 **LSE merge**
  (log-sum-exp 合并,FlashAttention 的在线合并方式)拼起来;decode 还有一条
  论文没有的 **GPU 本地分支**——当前新 token 的自注意在 GPU 算好,以三元组过链路
  进 die 一起合并(论文说 decode 全在 bank)。
- **差距**:两种合并在数学上等价,但 die 需要的硬件不同(写口+覆盖 vs LSE 合并单元),
  E4 的 RTL 对象、§4.3 的四样新增清单都取决于选哪个口径;必须二选一对齐。

### 3.8 ⚠️ bank 侧 prefill 的形态不同

- **论文**(§4.5.2):选边到 bank 的 prefill,其注意力**整段**进 bank:新 KV 在 GPU
  分段产出、逐段落进堆栈,之后的 query 在 bank 里扫"缓存上下文+已落地的前面各段"
  (扫描长度 L+n_r),段搬运与前段扫描重叠,因果性由顺序天然保证。
- **仿真器**:物理 DAG 永远是 split——新行×新行的注意力留在 GPU 上算,PIM 只扫
  复用的旧行(query 只依赖位置≤自己的旧行,因果 ✅),两边 LSE 合并;legacy 的 A5
  虽然"全 PIM prefill",但其扫描地址用的是 16 通道**私有连续布局**,不是 master/diff
  的 8 通道分池地址——B3 (PIM-static) 的 PIM prefill 带宽因此偏乐观约 2 倍。
- **差距**:正文的"整段进 bank+分段落地流水"在两条路径里都没有。

### 3.9 ⚠️ CacheBlend 的选点是随机采样,不是偏差准则

- **论文**(§5.3.2):CacheBlend 选"KV 偏差最大"的 token(5–15%)。
- **仿真器**:每个 partial 层均匀随机采样 ceil(ratio·N) 行(带种子可复现)。
- **差距**:孤立分布的形状一致,但选点准则不同;随机位置与真实偏差位置在计时上
  是否等价,取决于真实选点的空间聚集性——在用真 trace 做实之前是个假设。

### 3.10 其他小项

| 条目 | 论文 | 仿真器 |
|---|---|---|
| 转 Q 变体计数 | "one variant per chunk"(§4.4) | 按**不同位置偏移**去重(同偏移的 chunk 共用一份,更省;措辞对不上) |
| rotate 位置 | 只保留 GPU 侧(8-18 已删 die 上的 rotation unit) | `--cacheblend-rotate-mode` 仍有 die/bank 两档残留;实验须保持默认 gpu |
| attach 时装载位置元数据 | driver 一次装进 die(§5.1) | 无 attach 装载事件;代之以每次扫描 5 ns/run 的软件查表 + 每条 Q 的 die 变换事件 |
| 阶梯的 legacy 路径 | 每个 PIM 级都转 Q、共享存储批处理 | A1–A6 路径不含转 Q 流量,decode 逐 agent 独立扫(共享扫描只在物理 DAG 路径有) |
| 平台 | 8×H100(§5.1) | H100 配置存在,但现有全部实验数据跑的是 A100a;进正文前须统一 |
| 死代码 | — | 旧版共享 Q 函数无人调用(见 §2.4) |

### 3.11 ✅ 抽查一致的部分

Ramulator2 周期级命令流/HBM3-PIM 时序/按命令能耗表;A1 与原版 AttAcc 的逐事件回归
测试;master/diff 分池与 diff 紧凑段;**diff 通道里只有 K/V、没有元数据字节**(与
8-21"逻辑位置不进 DRAM"的裁决一致);read-mask 语义(Eq.(step2)/(step4));GPU 侧
转 Q 且只计链路增量;EPIC"每个移位段重算前缀 k、位置稳定段不修";decode 跨 agent
共享 master 扫描 + 按真实 Q 到达凑批(含审计器);"rows outer, agents inner" 的命令
序;事件依赖/重叠契约校验;TTFT 与逐步 decode 延迟、压缩率报表;FFN 并行 (`--ffopt`)
与流水 (`--pipeopt`) 沿 AttAcc。全套 27 个单元测试在新编译的 Ramulator2 上通过。

---

## 4. 数字速查(来源:`ramulator2/src/dram/impl/HBM3-PIM.cpp` preset 与 trace 生成器几何)

| 量 | 值 |
|---|---|
| 列大小 / 行大小(每 bank) | 32 B / 1 KB(32 列) |
| 每 channel bank 数 | 64(2 pCH × 2 rank × 4 BG × 4 bank) |
| 每 token 每 KV 头 key 字节(d_head=128, b=2) | 256 B(BG 内 4 bank 各 64 B = 2 列) |
| 一条 MAC-AB 读的数据 / token 数 | 2 KB / 8 个 token |
| 一条 channel 行 | 64 KB = 256 token(= 论文的放置单位"行") |
| nCCDAB(列→列,MAC-AB) | 6 cycles(PC)/ 4(NPC) |
| nRCDRD(ACT→首条 MAC) | 19 cycles |
| nRC(ACT→ACT,同 bank) | 63 cycles |
| nRAS / nRP / nRTPL(ACT→PREA / PREA→ACT / 末条 MAC→PREA) | 45 / 19 / 8 cycles(行切换实际下限 nRAS+nRP = 64) |
| 周期换算 | 0.769 ns/cycle(1.3 GHz,包装器口径) |
| 每行免费 MAC-AB 槽(不拖慢 ACT 节奏) | 4(PC)/ 5(NPC);实测平台期 70 cycles/行(含 refresh) |
| ACT-bound → MAC-bound 拐点 | 2·B·d ≥ 4,即 B·d ≥ 2(PC)/ ≥ 3(NPC) |
