# 仿真器(kvpim-sim / attacc_drampim_xinyao@xinyao_0821)与论文正文(Fugue main.tex@b3a38cb)校对报告

日期:2026-08-21。校对基准:论文正文 `sections/02–07`(以正文为主,outline 仅作佐证);
仿真器 `xinyao_0821` 分支。代码引用为 `文件:行号`。

---

## 0. 结论速览

| 状态 | 条目 |
|---|---|
| ✅ 一致 | Ramulator2 周期级 AttAcc 命令流;链路模型;FFN/pipeopt;master/diff 分池;diff 通道无 metadata(8-21 裁决);read-mask;GPU 侧转 Q;EPIC 前缀规则;"rows outer, agents inner";decode 跨 agent 共享扫描(物理 DAG);TTFT/decode 指标;压缩率报表 |
| ⚠️ 机制不同 | die 合并方式(LSE merge vs 写口覆盖+单次 softmax);decode 的 GPU 本地分支;bank 侧 prefill 的形态(split vs 整段进 bank);CacheBlend 选点(随机 vs deviation) |
| ❌ 缺失 | **Eq.(placement) 动态选边(B4/Fugue 本体)**;**§4.2 放置表与 256-token 行粒度跨通道布局**;B0(GPU-only)的"KV 在池 + 过链路"语义;n_d≈ρfC 配比;GQA/g;Eq.(actcost) 行激活数指标输出;bank 侧多 Q buffer 的容量约束 |

---

## 1. chenyi9 提出的专项:AttAcc 单 Q GEMV → Fugue 多 Q GEMM(bank 侧 buffer)

**用户观点**:AttAcc 一个 bank 的 GEMV buffer 驻一个 Q、对多行 K 做 GEMV;Fugue 批多个
agent 的 Q 复用同一行 K,必须"看两次 ACT 之间能做多少次乘法 → 决定塞几个 Q 进 buffer →
行开着做多 Q×多 K 的 GEMM"(decode 跨 agent 共享 chunk 与 prefill 多 Q 都适用),
这样 PE util 仍 100%,也不破坏 AttAcc 的 Q 复用假设。

**仿真器现状(已实测)**:

- 执行语义已实现。`gen_trace_attacc_bank.py:475-487`:`shared_queries>1` 时把
  两头流水命令流中每条非 BARRIER 命令**原位 ×B**(WRGB/MAC/MVSB/SFM/MVGB 都重复
  B 次、barrier 全批共享)。ACT 无显式命令,由 controller 在 row miss 时自动补
  (`ramulator2/src/dram/lambdas/preq.h:50`),因此同一行只激活一次,B 个 Q 的
  MAC 在行开着时背靠背发——正是 rows(列)outer、queries inner,PE 每个命令槽
  都在做真实 MAC,Q 复用(每个 Q 仍扫完它的所有 K 列)不被破坏。
- 两个用点都接上了:decode 跨 agent 共享 master 扫描
  (`workload_runner.py:1912`,`pim_shared_queries=len(group)`,B 即
  `--cacheblend-batch-size`)与 prefill 多 Q 分组扫描
  (`workload_runner.py:2385`;legacy 路径 `--pim-prefill-query-batch`,默认 4)。
- 实测(HBM3_5.2Gbps PC preset,本机新编译的 ramulator2):L=512、1 head,
  q=1:930 cyc;q=4 共享:3553 cyc(vs 串行 4×930=3720)。MAC 数 128→512 严格 ×4。
  整行扫描本来就不 ACT-bound(nRCDRD+32×nCCDAB = 19+192 = 211 ≥ nRC=63),
  所以共享的收益主要是 **ACT 次数/能耗**与**短 run**:d 个 token 的 run 只有
  2d 条 MACAB,d≤3 时 max(nRC, 19+12d) 被 nRC 卡死;B 个 Q 后 19+B·12d ≥ 63,
  d=1 时 B=4 恰好补满——与仿真器默认 `pim_prefill_query_batch=4` 巧合一致。

**缺口(paper 正文 + 仿真器各一半)**:

1. **正文没有这个机制的硬件承载**。§4.5.3 的执行语义是对的("the queries of
   several agents follow one another while the row stays open"、"amortizes the
   same activations over its nr queries, while the column accesses its MACs
   consume grow with nr"——与 trace 行为逐句吻合),但 batch 上界只归给 die 侧
   ("bounded by the per-agent state **the die holds**…the softmax buffer entries
   and the decoder's metadata buffer")。bank 侧 GEMV buffer 要同时驻:
   score 相 B 份 Q 切片(每 bank 64 B/Q,d_head=128、b=2)+ context 相 B 份
   P 切片(随 L 增长,是 sizing 大头;流式按行喂 P 可以压小,但这是要写死的口径)。
   而 §5.1 又断言 "The bank PE keeps its MAC as the only arithmetic, the DRAM
   command set is unchanged"、"Everything else stays as AttAcc built it"——
   多 Q 至少需要 buffer 分槽 + MAC 按 Q 轮转选槽(命令集不变的话须在 PE 内做
   mod-B 计数器轮转,trace 里 B 次 WRGB 写的是同一地址,纯 timing,没有分槽概念)。
   用户的"两次 ACT 之间的乘法数 → B → buffer 条数"就是缺的那段 sizing 论证,
   E4(RTL)目前口径也只测 die 侧结构。
2. **仿真器没有容量约束**:B 是 CLI 旋钮,超过任何真实 buffer 都照样跑;正文说的
   "beyond which the sweep splits" 在仿真器里没有对应的自动拆分。
3. **死代码**:`gen_trace_attacc_bank.py:210 _shared_query_attention_commands`
   (另一种按 K-row 粒度的 interleave)无人调用,与实际生效的 ×B 展开(475 行)
   语义不同,易误导后续改动。

---

## 2. 缺失的核心机制(仿真器侧)

### 2.1 Eq.(placement) 动态选边 —— B4/Fugue 本体未实现
正文 §4.5.2:"The event stream evaluates them per prefill from (nr, Lctx) and
places the attention on the banks when t_bank ≤ t_xPU";§5.4 的第五级
"Fugue adds the placement rule";§6.1/§6.3 的 Fugue 结果全部依赖它。
仿真器里没有任何 t_bank/t_xPU 比较逻辑(全库 grep 无):
- legacy 路径只有静态 `--prefill-attn gpu|pim|split`(`ablation.py:70-77`,A1–A6);
- 物理 DAG 路径永远是 split 形态(fresh×fresh 在 GPU + PIM 扫 reused,
  `workload_runner.py:2296` 起)。
`outline/README_experiments.md` 也标着"待做:kvpim-sim 接入自适应选边"。
另注:`experiments/GPU_PIM_vs_GPU_prefill` 得到的拐点(EPIC p*≈22–35 NVLink3 /
89–210 PCIe4;CacheBlend r 上限 0.4–2.7%)是 **A4 vs A6** 网格扫描的交点,
对象是"协同 split prefill",不是正文 Eq.(placement) 的"整段二选一";B4 实装后
这些拐点要按正文口径重扫。

### 2.2 §4.2 放置表 + 行粒度跨通道布局未实现
正文:chunk 切成 256-token 的行,"the driver appends each new row to a channel
that holds none of the rows it will be read alongside",一张 row→channel 软件表,
同轮同读的行(包括同一 chunk 的两行)放不同 channel、"their scans run at once"。
仿真器:沿用 AttAcc 的 head-striping——head h 进 pool 内第 (offset+h)%8 条
channel,一个 head 的整段上下文顺序放在**一条** channel 的分区里
(`CacheBlendTLB.finalize`,`workload_runner.py:683+`;分配是线性游标,无任何
防冲突放置)。一次扫描以 head 并行占满整个 pool,因此**同 pool 两个 chunk 的扫描
永远串行**(每个 pool 是一条串行资源,`_schedule_cacheblend`,
`workload_runner.py:1006+`)。"行切分/放置表/跨 chunk 并行"三件事都不存在。
注意:head-striping 下"一条扫描占满 pool 带宽"与正文"行分散+并行扫"在**多 head、
带宽饱和**时总时间可能接近,但正文描述的机制(§4.2 全节 + §3.1 的 row conflict
动机)在仿真器里没有对应物;且 GQA 少 head 时两种布局的带宽差距会拉大(见 2.5)。

### 2.3 B0(GPU-only)语义不符
正文 §5.4:"GPU-only keeps the software master–diff store **in the memory pool**
in append order and runs every attention…on the GPU **with the KV crossing the
link**"。仿真器 A2:KV 驻 GPU HBM、attention 全 GPU、**不过链路**,校验还禁止
decode-gpu 配任何 PIM mapping(`ablation.py:157`),也没有 append-order 惩罚。
B0→B1 应只改"算在哪"(outline 明确),现在 A2→A3 同时改了"存哪+算哪"。
按正文跑 B0,需要:KV 驻 Acc 池 + 每层 attention 的 KV 读回链路事件 + 池侧
乱序(append-order)访问开销。

### 2.4 diff 通道配比 n_d ≈ ρ_b·f·C 未落地
正文 §4.1.1 Eq.(ratio):"most channels serve masters and one or a few serve
diffs"。仿真器:物理 DAG 硬编码 master=ch0–7 / diff=ch8–15
(`workload_runner.py:503-506`);legacy 可调 `--kv-pool-split` 但默认 8/8。
8/8 也影响 timing(diff pool 8 条 channel 的带宽被高估、master 减半)。
无按 ρf 推导配比的逻辑。

### 2.5 GQA / g 完全未建模
正文:Eq.(placement) 含 g;§5.3 workloads 是 LLaMA-3-8B(GQA)+ GPT-175B 形状;
§7 有专段 GQA 讨论("evaluation includes one model of each kind")。
仿真器:`config.py` 模型表 `gqa_size` 恒为 1 且**无任何消费者**;没有
LLaMA-3-8B 条目(LLAMA-7B/65B 是 MHA);attention 各层按 num_heads 全 KV 头建。
g>1 时 KV/token 缩小 g 倍、每 KV 头 g 个 Q——放置公式、trace 形状、容量都会变。

### 2.6 Eq.(actcost) 行激活数指标断在半路
正文 §5.4 metrics 把 "the row activation count of Equation (actcost)" 列为指标。
分支在 controller 里加了 `pim_activations` 计数
(`ramulator2/src/dram_controller/impl/hbm3_pim_controller.cpp`,统计实际下发的
ACT/ACTAB/ACTSB/ACTPB),但:
(a) 实测运行输出里**没有**这行(finalize 只打印 frontend/memory-system 统计);
(b) `src/ramulator_wrapper.py` 的解析器不认它(只解析 mac/sfm/mvgb/mvsb/wrgb/cycles);
(c) `pim_ramulator_src/hbm3_pim_controller.cpp` **没有**这个 patch——
跑一次 `set_pim_ramulator.sh` 会把它覆盖丢失(该脚本还有一个坑:本分支
`ramulator2/` 是父 repo 追踪目录而非 submodule,脚本里的
`git reset --hard b7c7027…` 会作用到父 repo 上)。

---

## 3. 机制不同(数值可能接近,硬件故事不一致)

### 3.1 die 合并:LSE merge vs 写口覆盖 + 单次顺序 softmax
正文(§4.3 + outline 8-20/21 裁决):master 分数按 token 序写 softmax buffer,
diff 分数经 decoder 走**外部写口按逻辑位置覆盖**,凑齐后**一次** softmax;
明确"不需要 running/online softmax,die 不加 max/sum 寄存器";die 新增只有四样。
仿真器(物理 DAG):每个物理 run(master run、diff run)各自在 trace 里做完
score+softmax+PV,输出 (m,l,o) 局部 tuple,DIE 上 `die_lse_merge` 事件按
FlashAttention 方式合并(`workload_runner.py:1645/1970`)。数学等价,但:
- die 需要 LSE 合并单元(正文已删掉的 online-softmax 机制);
- decode 还有一条 **GPU 本地分支**:当前 token 的自注意在 GPU 算
  (`decode_gpu_local_score` m=1,n=1,`workload_runner.py:1578`)、
  partial-LSE tuple 过链路进 die 合并——正文说 decode 全在 bank,无此分支。
两边必须选一个口径:要么正文承认跨 run/跨设备的 LSE 合并,要么仿真器改成
分数覆盖后单次 softmax(对 timing 影响很小,但 die 面积/E4 的 RTL 对象不同)。

### 3.2 bank 侧 prefill 的形态
正文 §4.5.2:选边到 bank 时**整段** prefill attention 进 bank,新 KV 分段落地,
同一 prefill 的后续 Q 在 bank 里扫已落地的段(扫描长度 L+nr),段搬运与前段扫描
重叠。仿真器:
- 物理 DAG:永远 split——fresh×fresh 矩形块在 GPU,PIM 只扫 reused 旧行
  (Q 只依赖 pos≤自己 的 reused 行,因果 ✅),两分支 LSE 合并。fresh 行不进 bank。
- legacy A5(全 PIM prefill)存在,但其 PIM 扫描用 `_private_runs`
  (`ablation.py:652`)——16 条 channel 的 **private 连续布局**,不是 master/diff
  的 8 通道分池地址 → B3(PIM-static)的 PIM prefill 带宽偏乐观约 2×。
正文的"分段落地 + 后续 Q 扫到新段"的流水在两条路径里都没有(物理 DAG 的新 KV
落地是 per-layer 一笔、异步到 decode 前 join)。

### 3.3 CacheBlend 选点准则
正文 §5.3.2:"selects the tokens with the largest KV deviation, at 5–15%"。
仿真器:每 partial 层均匀随机采样 ceil(ratio·N) 行(seeded,
`workload.py:324`)。孤立-token 的结构形状一致,选点准则不同
(outline 已有"真 HKVD 选择待做"条目);随机位置对 timing 的行为与真实
deviation 位置是否等价,取决于真实选点的空间聚集性——用 trace 做实前是个假设。

### 3.4 其它小项
- 转 Q 变体计数:正文 §4.4 "one variant per chunk";仿真器按 **distinct delta**
  去重(同偏移的 chunk 共享一份变体,`_append_q_rotate_distribution`,
  `workload_runner.py:1305+`)——更省,严格说与正文措辞不同。
- die/bank 两种 rotate 模式还在代码里(`--cacheblend-rotate-mode`);正文 8-18
  已删 die rotation unit,实验须保持默认 `gpu`。
- attach 时"logical positions 一次装进 die":仿真器无 attach 装载事件,代之以
  每 run 5 ns 的软件 TLB descriptor + 每 Q 的 `die_query_position_transform`。
- ladder 的 legacy 路径(A1–A6)不含 Q 旋转流量、也不含 decode 跨 agent 共享
  扫描(decode 以 numOp=heads×batch 逐 agent 扫)——正文说 "every PIM step of
  the ladder" 转 Q、共享 store 批处理;这两点只有物理 DAG 有。
- 平台:正文 §5.1 是 8×H100;config 支持 H100,但现有实验(E2/E3 初版、
  GPU_PIM_vs_GPU_prefill)都跑的 A100a。数字进正文前要统一。

---

## 4. 一致的部分(抽查通过)

- **Ramulator2 周期级**:每个物理 extent 一次 trace 运行,签名缓存;HBM3-PIM
  时序 preset(5.2 Gbps PC/NPC)、ACTAB/MACAB/MVSB/SFM 命令集与 AttAcc 一致;
  能耗按命令计数 × AttAcc/HBM-PIM 表(`ramulator_wrapper.py`,`config.py`)。
- **A1 = 原版 AttAcc**:有逐事件回归测试
  (`test_ablation_a1_reproduces_the_original_attacc_legacy_report` 等)。
- **master/diff 分池 + diff 紧凑段 + 无 DRAM metadata**:TLB 每 vector 只有
  K/V 各 256 B(d_head=128、b=2),逻辑位置在软件侧/die decoder——与 8-21
  "逻辑位置不进 DRAM" 裁决一致;diff 占用 = ρ_b·N_ag·L ✅。
- **read-mask**:shadow master 行照读、`masked_rows` 从分数里排除、P 由 mask
  决定 master/diff 归属(Eq. step2/step4 的语义)✅;`--master-shadow skip`
  作为对照保留。
- **decode 共享扫描**:同批 agent 的 master 地址交集走一次共享 scan
  (`workload_runner.py:1879-1913`),diff/私有行逐 agent 扫——Eq.(actcost)
  的结构;admission 按真实 Q 链路到达排序(global-q-ready-queue),有审计器。
- **EPIC**:每个 shifted 段前缀 k 行、位置稳定前缀不修 ✅;CacheBlend
  full/partial 层结构、逐层采样 ✅。
- **事件流**:named dependencies、per-device 资源时间线(pipe)、AttAcc
  overlap contract 校验器、TTFT(prefill_end/first_token)与 decode per-step ✅。
- **FFN/pipeopt**:`_ff_parallel`、`apply_attacc_pipeline` 按 AttAcc ✅。

---

## 5. 环境/工程备忘

- 本分支 `ramulator2/` 已入库;EL8 系统 gcc 8.5 编不过(`<ranges>`),用
  `CC/CXX=/opt/rh/gcc-toolset-11/...` 编译通过,二进制在 `ramulator2/ramulator2`。
- `pim_ramulator_src/` 与 `ramulator2/src/` 不同步(controller 的
  pim_activations patch 只在后者);`set_pim_ramulator.sh` 对入库后的布局不再安全。
- 本地未提交改动(README.md、ramulator.out 等)已 stash:
  `WIP before switching to xinyao_0821 (0821 session)`;
  旧 `ramulator2/` 整目录备份在 `ramulator2.local_backup_0821/`
  (含旧二进制与 E1 分析脚本 analyze_trace.py / audit_bank_dist.py,
  branch 里没有这两个脚本)。
