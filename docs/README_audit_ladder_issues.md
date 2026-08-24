# AUDIT:A 矩阵首批数据暴露的四个问题——代码定位、归因与修复方向(2026-08-25,只诊断不动手)

对象:`experiments/paper_ladder/` 首批 ~49 个作业的核对结论(见
`CLAIMS_CHECK.md`)中不支撑/反向的四条。每条给:现象 → 应该的样子
(宸逸口径)→ 代码里实际的样子(文件:函数)→ **谁在哪个 commit 弄成
这样**(attacc / xinyao(xw338)/ chenyi(Allan))→ 修复方向。
**均未动手,待逐条裁决。**

---

## 问题 1:A6 dynamic 没贴住便宜侧(该修的 bug)

- **现象**:multihop/65B TTFT:A4 98.3 s、A5 149.3 s、A6 134.5 s——
  A6 只优于 A5,没贴住 A4;7B 上 A6 甚至整体选了 PIM(=A5)。
- **应该的样子**:A6 逐类比价后走便宜侧,数值上应 ≈ min(A4 侧, A5 侧)。
- **代码实际**:`src/ablation.py::_prefill_batch` 的 dynamic 块里,
  **t_bank 的估价与 A5 实际入账逐项一致**(同一 `_pim_scan`、同 sfm、
  同链路),但 **t_xpu 的估价与 A4(gpu 档)实际入账口径不一致**:
  gpu 档的注意力在顶部 `sum_decoder` 循环按**批级**折算定价
  (`op.m = layer.m × scale`,scale=有效行/填充行,模板自带 batch 形状,
  回读链路每层记一次);而估计器按**每请求**行数构造算子
  (`op.m = queries = rows_batch/batch`、`n = 全上下文`)再乘层类数
  ——batch 因子与折算口径都对不上,系统性把 xPU 路估贵,于是偏向选
  PIM。
- **归因**:chenyi,`654aeee`(dynamic 块引入时自造了 xPU 估价,而不是
  复用 gpu 档的入账函数)。物理 DAG 路径的估计器没有此问题(它直接用
  gpu 分支同一套事件定价)。
- **修复方向**:把解析路径的 t_xpu 估价改为**调用与 gpu 档完全相同的
  定价路径**(把 gpu 侧类定价抽成共享函数,估价=试算入账),保证
  "估=入账"两侧对称;修后 A6 全列重跑。

## 问题 2:A5 的定位——不是"优势档",是"强制全上 PIM 的对照档"(叙述修正,非 bug)

- **现象**:A5 在全部已完成 workload 上 TTFT 都劣于 A4(multihop/7B
  +42%,mooncake/7B +65%)。
- **宸逸口径**:prefill attention 也是 memory-bound,但**全都压进 PIM
  也不行**——这正是要 dynamic 的原因。A4/A5/A6 三档并排就是"prefill
  放哪"的对比:A4=全 GPU、A5=全 PIM、A6=逐请求选。**不需要**再造
  "交叉点以下"的专门设计点;A6 选完后报一个比例(多少 prefill 事件走
  GPU、多少走 PIM)即可。
- **落点**:`CLAIMS_CHECK.md` 的 C1d 从"A5 赢 TTFT"改写为"A5 与 A4
  构成两个强制极端,A6 的选边比例 + 贴住便宜侧是主张本体";比例字段
  两条路径都已有(解析:breakdown 时间份额;物理:`pim_prefill_sides`
  请求份额 + 事件计数份额,`collect_results.py` 已提取)。
- **注意**:问题 1 修复前,A6 的比例数字不可用(偏 PIM 是估价偏差
  造成的)。

## 问题 3a:分池为什么把 master 砍到 8 条 channel

- **现象**:A4 TBT 反而劣于 A3(multihop/65B 22.3→28.5 ms;
  mooncake/7B 10.5→17.1 ms)。直接原因:master 池只有 8 条 channel,
  **主扫描流带宽减半**;而 diff 行(EPIC k=8)极少,却独占 8 条。
- **应该的样子**(宸逸口径):16 条 channel 里 diff 池应该**很少**
  (够放紧凑 diff 即可),其余全归 master。
- **代码实际**:两处硬口径——
  1. 物理 TLB:`src/workload_runner.py` 固定
     `master = range(0,8)`、`diff = range(8,16)`(第 ~513 行的
     channel_sets 与 `_prepare_cacheblend_tlb` 的池划分),**无旋钮**;
  2. 解析路径:`src/ablation.py::AblationConfig.master_pool_channels`
     默认 8,`main.py --kv-pool-split` 默认 8;矩阵没改默认。
- **归因**:xinyao——物理 8/8 划分在 `0aced82`(CacheBlend 栈引入),
  解析默认 8 在 `47ae0c3`(A1–A6 消融引入);chenyi 的矩阵驱动
  (`81eedc9`)沿用默认未调。
- **修复方向**:diff 池收窄(如 15/1 或 14/2;或按 diff 密度 ρ_b 自动
  定宽),两条路径同步;分池宽度纳入矩阵为一个扫描旋钮。**注意**:
  修完后 A4 对 A3 的对比才公平——现在 A4 的劣势主要是池宽人为造成。

## 问题 3b:naive 布局为什么看不到碎片惩罚

- **现象**:A3 的 TBT ≈ A1(22.335 vs 22.393 ms)——乱序布局零代价。
- **宸逸的问题**:Ramulator 不是按命令流仿真、乱序读和顺序读时间不同
  吗?为什么没体现?
- **代码实际**(两层原因):
  1. **碎片的物理代价本来就小而模型也只收这部分**:
     `src/ablation.py::_naive_run_lengths` 确实把每个段边界、每个重算
     行都拆成独立 run;`_pim_scan` 把 run 串行送 Ramulator,每个 run
     自带行激活开销——但一次行切换只有 ACT/PRE 量级(几十 ns),对
     22 ms 的整段流扫描完全不可见。碎片惩罚"有建模、但天然量级小"。
  2. **真正该疼的地方没建模——channel 内冲突**:
     `_runs_from_lengths(..., channel_base=0, channels=16)` 让 naive 的
     **每个 run 都摊满全部 16 条 channel 并行流出**(理想并行);真实
     乱序布局的代价是不同 chunk 落在同一 channel 时**排队串行**
     (row conflict),以及一条 channel 一次只能开一行。这一层在整个
     仿真栈里都不存在:legacy trace 的抽象是"一个批一个 run 形状、
     跨池满宽并行"(attacc 原版 trace 生成器的批抽象),xinyao 的
     profile 构造沿用,已有记录见
     experiment 分支 `docs/README_design_check.md` §3.2(论文 §4.2 的
     行粒度放置表/防冲突在仿真器里没有对应物)。
- **归因**:抽象根源 attacc(legacy trace 的满宽批抽象);naive profile
  构造 xinyao(`47ae0c3`);chenyi 未补冲突模型。
- **修复方向**(选项,待裁决):(a) 给 naive 加冲突代价模型(按 chunk
  →channel 均匀散列,冲突段串行化,期望冲突因子可解析算出);
  (b) 或承认口径:A3 vs A4 的对比只在"含行冲突的物理模型"下有意义,
  解析路径不报这条 claim;(c) 物理路径实装 §4.2 的放置表后用事件路径
  出这条数据。

## 问题 4:A5/A6 为什么用了老 PE 频率与老 buffer

- **现象**:矩阵里 A5/A6 的 bank 路是 PE 0.666 GHz、GEMV buffer 512 B、
  每波 4 条查询——没吃满 C3 微架构。
- **宸逸口径**:prefill attention 上 PIM 与微架构、attention batching
  是**一起使用**的,A5/A6 应该带上匹配的微架构参数。
- **代码实际**(三处叠加):
  1. `654aeee`(chenyi)把批命令耦合进 preset(A5/A6 = mq),但
     **没有耦合 PE 频率与 buffer**——`--pe-freq-ghz` 默认 0.666、
     `--gemv-buffer-bytes` 默认 512(沿 8-24"先不假设频率/buffer"的
     指示保守处理,现口径已更新);
  2. `experiments/paper_ladder/run_matrix.py`(chenyi,`81eedc9`)没有
     给 A5/A6 传这两个旋钮;
  3. `AblationConfig.pim_prefill_query_batch` 默认 4(xinyao,
     `47ae0c3`)——prefill 每波被 min(4, 容量 8) 卡在 4,即使 buffer
     容量是 8;batching 开启时这个独立旋钮应让位于容量。
- **修复方向**:A5/A6 preset(或矩阵)绑定微架构档——buffer 与匹配
  频率取哪一档(512 B/1.73 GHz?768 B/2.6 GHz 平衡点?)由宸逸定;
  `pim_prefill_query_batch` 在 mq 下默认改为跟随 `mq_query_capacity`。
  修后 A5/A6 两列重跑。

---

## 汇总表

| # | 问题 | 归因(commit) | 性质 | 修后需重跑 |
|---|---|---|---|---|
| 1 | A6 估价口径不对称 | chenyi `654aeee` | bug | A6 全列 |
| 2 | A5 定位=强制对照,报选边比例 | (叙述口径) | claim 改写 | 无 |
| 3a | 分池 8/8,diff 独占一半带宽 | xinyao `0aced82`/`47ae0c3` | 设计参数错 | A4(及 A5/A6 池宽相关) |
| 3b | naive 无 channel 冲突模型 | attacc 抽象 + xinyao `47ae0c3` | 模型缺口 | A3(或改口径) |
| 4 | A5/A6 未挂微架构参数 | chenyi `654aeee`/`81eedc9` + xinyao 默认 4 | 耦合缺口 | A5/A6 全列 |
