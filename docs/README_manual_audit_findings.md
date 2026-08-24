# 手动审计发现清单(chenyi-822 移植过程,宸逸逐步审查)

目标读者:接手人/审稿人;概念首现即释,本页自足。
本分支的移植方式是**每一步人工核对**(规则见 `PORTING_PLAN.md`)。这种
逐行过目在参考实现(experiment 分支,原自动化会话产出)里抓出了真问题
——本页记录这些**人工审计发现**:是什么、错在哪、怎么处置、状态。

## 发现 1:计算能量被摊进 DRAM 列命令间隔(结构已修;定量为近似,TODO 未关闭)

- **位置**:`src/ramulator_wrapper.py` 的 `mq_interval_cycles`。
- **原模型**(参考实现最初版本):功耗受限时把列命令间隔按每命令能量
  等比例拉伸——`interval = ceil(nCCDAB_PC × (e_col + n·e_q)/(e_col + e_q))`,
  即每多一条驻留查询的**计算**能量 (e_q) 都会拉长 **DRAM** 列读节拍。
- **审计论证**(宸逸,2026-08-23):(a) "6 cycle 恰好打满功耗预算"是
  无证据的紧性假设(AttAcc 自己的数字说 16 bank 全开只用预算的 16/18);
  (b) 计算能量与列读电流不该混一个预算池——nCCDAB 的构成里没有计算项
  (NPC=4 可从 AttAcc 的 9×内部带宽/18 units 精确闭合推出;Samsung
  FIMDRAM 先例:PIM 计算严格由命令流驱动,官方 PIMSimulator 无任何
  PIM 专属时序);设计意图本就是"列流全速、PE 在空档计算"。
- **处置**(已落地,两分支同步):`interval = max(preset 6/4, ceil(n/(f·tCK)))`,
  计算功率**另算**——`mq_pe_power_w()` 独立记账,对照 AttAcc Fig.7(a)
  的 116 W IDD7 预算线(n=32 全速 37.1 W,远在预算内)。
- **TODO(重要限定,宸逸 2026-08-23):这只是近似方案,不算解决**。
  分账的**结构**是对的,但**定量口径没有闭合**,因为查不到真实的总能量
  预算:(a) IDD7 的绝对电流/预算构成不公开(JESD238 与厂商 datasheet),
  116 W 是从 Fig.7(a) 图上人工读的;(b) 功率检查拿 cell 侧微观能量
  (FGDRAM 口径,PIM 流仅 32.4 W)去对宏观 116 W——宏观值含背景/外围/IO,
  两个口径不能闭环校验,"37.1 W 在预算内"只是量级判断;(c) PC preset=6
  的来源至今未闭合(只有 NPC=4 能从 9×带宽/18 units 精确推出)。
  关闭此 TODO 需要:JESD238 的 IDD7 环定义或厂商电流数值,或
  HBM-PIM/Newton 的功耗拆分数据,把预算按"列读/激活/背景/PE"分项立账。
- **状态**:结构修正已提交(experiment `cdf8f9a`、本分支 `e81c2e4`);
  C 点已按新口径重测(加速比全线上修,如 C3 (32,4)@1.3 GHz
  3.63×→4.06×)——这些数字继承上述近似口径。

## 发现 2:TLB 描述符 5 ns 是未溯源常数,且与论文口径重复收费(TODO)

- **位置**:`src/workload_runner.py` 的 `_TLB_DESCRIPTOR_S = 5e-9`
  (代码处已留 TODO 注释)。
- **背景**:TLB(逻辑 KV 位置→物理地址映射)不是 AttAcc 原版的东西——
  原版私有 KV 连续摆放、地址仿射计算;TLB 机制是复用栈(0aced82)引入,
  这个计时常数是 47ae0c3 加的。语义:每个连续物理 run 收一份
  "描述符"(基址/长度/池/掩码)的下发代价,5 ns/run。
- **审计发现**(宸逸,2026-08-23):(a) **5 ns 无出处**——无测量、无文献、
  无推导,纯建模假设;(b) 与论文口径**重复收费**:Fugue §5.1 的口径是
  attach 时 driver 把位置元数据一次性装进 die 的 decoder metadata
  buffer,扫描时查常驻元数据不另收费;仿真器却每次扫描每 run 收 5 ns,
  且 sweep 拆趟后每趟重复收。
- **量级**:保守方向(多收时间),<1% decode 步(每步每层 2–3 run
  ×5 ns vs 扫描本体 ~2.4 µs @L=4096)——不影响既有结论,但来历不明。
- **处置选项**(待裁决):(a) 保留+标注(现状,已留 TODO);
  (b) 改为 attach 一次性装载事件模型(忠实论文,动事件结构);
  (c) 用 RTL 侧 decoder(`kvpim-rtl` 的 `mq_diff_decoder`/metadata 路径)
  综合反标一个有依据的数。
- **状态**:TODO 已留;两分支模型暂维持 5 ns 原样(保守方向)。

## 附:同批手动审计的其他记录(已在别处存档,此处索引)

- 批量>容量的静默封顶、拆趟"行交错"替代设计:experiment
  `docs/README_mq_design_space.md` §7(第 4、5 条,均标暂不动);
- BG 级归约层缺口与口径:experiment `docs/README_bg_reduction.md`;
- V 矩阵列内容口径、per-Q 寄存器清单等 8-21 三路审计修正:experiment
  `experiments/mq_command/DATAFLOW.md` §5。
