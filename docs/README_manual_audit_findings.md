# 手动审计总台账(chenyi-822 / chenyi-822-dirty,宸逸逐步审查)

目标读者:接手人/审稿人;概念首现即释,本页自足。
本页是**全项目人工审计发现的唯一总台账**:已修复的列"已解决"(带
commit 与位置),未修复的列"未解决"(带关闭条件)。逐项技术细节分别在
`README_audit_ladder_issues.md`(阶梯五题)与本页下方两节(未解决项)。

## 一、已解决(9 条)

| # | 问题 | 修复 | 位置 / commit |
|---|---|---|---|
| R1 | **计算能量摊进 DRAM 列命令间隔**(结构性混账:每条驻留查询的计算能量拉长列读节拍) | C 模型分账:interval = max(preset 地板 6/4, PE 项),计算永不拉长 DRAM 节拍(FIMDRAM 先例),PE 功率单独记账 `mq_pe_power_w` | `src/ramulator_wrapper.py`;experiment `cdf8f9a` / 822 `e81c2e4`。**定量预算闭合仍开 → 未解决 U2** |
| R2 | **diff/master 写序竞争**(diff 段短常先到,master 后写反盖) | per-agent **D_i 位图 master 写过滤**,到达顺序无关;`di_bitmap_gpu_to_die`/`die_load_di_bitmap` 事件 | `src/workload_runner.py`;experiment `3e338e6`(822 移植 5.L1) |
| R3 | **bank 整段 prefill 批内下三角归属**无定义 | bank-whole:K/V 先落地、每查询扫全范围、DIE 因果丢弃(一个比较器;上三角被扫即计费) | `--pim-prefill-mode pim` 分支;experiment `3e338e6` |
| R4 | **A6 dynamic 估价口径不对称**(xPU 路按每请求算子估价,比 gpu 档实际入账贵 → 偏选 PIM,没贴住便宜侧) | 估价改为与 gpu 档同一口径(顶层 scale 折算),估=入账两侧对称;multihop/65B 上 A6 = min(A4,A5) 验证 | `src/ablation.py::_prefill_batch`;`b649674` |
| R5 | **分池 8/8**(diff 行极少却独占一半带宽,A4 反劣于 A3) | 两条路径同步改 **15/1**(物理 `_KV_CHANNELS` + 解析默认 + `--kv-pool-split`) | `b649674` |
| R6 | **naive 布局无 channel 冲突模型**(每 run 摊满 16 channel 理想并行,乱序零代价) | 逐 chunk **顺序分配 channel 并追踪**,同 channel 冲突**串行化**(每占用 channel 一个单通道池,decode 取池间 max) | `src/ablation.py::_naive_channel_pools`;`b649674` |
| R7 | **A5/A6 没挂微架构参数**(PE 0.666 GHz / 512 B / 每波 4,与"上 PIM 即带 batching"的口径矛盾) | preset 绑定平衡点 **2.6 GHz / 768 B(=12 驻留)**,注释标 PROVISIONAL(后续还会调);mq 下每波跟随 `mq_query_capacity` | `src/ablation.py::PRESETS`;`b649674` |
| R8 | **压缩率列低估**:`_memory_report` 把共享 chunk 的属主副本双计(multihop 报省 4.7%,实际可去重 20.3%) | 公式去掉 `shared_rows` 重复项 + 单测锁死 + `owner_copy_fix` 标记;存量结果 `repair_memory_column.py` 纯算术修补(multihop → 0.798 ✓) | `src/ablation.py::_memory_report`;`0305d4c` |
| R9 | **多轮历史口径**(名义 `--history-len` 一开始就满长,不符多轮语义) | Mooncake conversation 按去尾块哈希前缀链轮,**history 逐轮从 0 累积**(= 上一轮完整 input);进矩阵为 `mooncakemt` | `workload/convert_mooncake_multiturn.py`;`0305d4c` |

另有两项**设计裁决**(非 bug,已落地):老 A6"split 混合 prefill"废除、物理 DAG 与 A 阶梯同菜单(`654aeee`/`0755694`);TSV 窄下行经实测关闭(转向代价 ≤0.84%,experiment C-abl-2)。

## 二、未解决(2 条)

### U1:TLB 描述符 5 ns —— 未溯源常数 + 与论文口径重复收费

- **位置**:`src/workload_runner.py::_TLB_DESCRIPTOR_S = 5e-9`(代码有
  TODO 注释;至今仍是该值)。
- **问题**:(a) 5 ns 无出处——无测量、无文献、无推导;(b) 论文口径是
  attach 时一次性装载 decoder metadata,扫描查常驻表不另收费,仿真器却
  每次扫描每 run 收 5 ns,拆趟后每趟重复收。
- **量级**:保守方向(多收),<1% decode 步——不影响既有结论。
- **关闭条件**(三选一,待裁决):保留+标注(现状);改 attach 一次性
  装载事件;用 `kvpim-rtl` 的 `mq_diff_decoder`/metadata 路径综合反标出
  有依据的数。

### U2:总能量预算的定量闭合(R1 的遗留半条)

- **位置**:`src/ramulator_wrapper.py::MQ_POWER_BUDGET_W = 116`。
- **问题**:R1 的**分账结构**是对的,但定量口径未闭合——(a) 116 W 是从
  AttAcc Fig.7(a) 图上人工读的,IDD7 绝对电流/构成不公开;(b) 功率检查
  拿 cell 侧微观能量(FGDRAM 口径)对宏观 116 W,两个口径不能闭环,
  "37.1 W 在预算内"只是量级判断;(c) preset=6(PC)的来源未推导(只有
  NPC=4 可从 9× 内部带宽精确闭合)。**宸逸 2026-08-23 裁决:这是近似
  方案,不算解决。**
- **关闭条件**:JESD238 的 IDD7 环定义或厂商电流数值,或 HBM-PIM/Newton
  的功耗拆分数据,把预算按"列读/激活/背景/PE"分项立账。

## 三、索引

- 阶梯五题的完整诊断(现象/代码/归因/修法):`README_audit_ladder_issues.md`
- C 系列与设计空间口径:experiment 分支 `docs/README_mq_design_space.md`、
  `docs/README_c_series.md`、`docs/audit/`(全项目公平性/完整性审计)
- 论文表述审计:`KVPIM-1Fugue-ASPLOS2027/audit/`(01 过时表述清单等)
