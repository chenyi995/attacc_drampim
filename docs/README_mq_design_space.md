# MQ bank-PE 设计空间:容量轴 × 速率轴(零基础版)

目标读者:学过计算机、会矩阵乘法,但**不了解**本项目、LLM serving、DRAM 内部
结构或 PIM 的人。读完应能独立回答:MQ bank-PE 的性能由哪**两个互相独立**的
旋钮决定、每个旋钮在算子/数据摆放/时序/面积四个维度上各对应什么、两个旋钮的
"匹配点"怎么算、以及在 in-bank 面积预算下当前哪些配置装得下。
本文遵守 `docs/OUTPUT_SPEC.md`;所有数字均为实测,来源逐一标注。

**矩阵乘法记号**(与 `README_fugue_dataflow.md` 一致):C[M×N]=A[M×K]×B[K×N],
K 是收缩维 (contraction dim)——被乘加消掉、累加发生的维度。

---

## 0. 三十秒背景

大模型生成每个新词(decode)时,注意力 (attention) 要把整段历史的 KV 缓存
(KV cache,每个历史 token 存一条 K 向量和一条 V 向量)从头到尾读一遍。
AttAcc/Fugue 把这一步搬进 HBM 内存的每个 bank(DRAM 的独立读写单元)旁边的
小计算单元(bank PE, processing element)里做,省去把 KV 搬去 GPU 的带宽。
Fugue 的多 agent 场景里,**n 个 agent 共享同一份 KV**:MQ(multi-query)批命令
让一次 KV 读出被 n 条查询共用——本文回答"这个 n 由什么决定、代价是什么"。

## 1. 结论先行

1. MQ bank-PE 的并发查询数由**两个独立旋钮的较小者**决定:
   **容量轴**(GEMV buffer 能驻留几条查询,面积大头)和
   **速率轴**(两次列读之间 PE 吞吐能服务几条查询,由 PE 频率决定,面积小头)。
   任一轴超出另一轴都是浪费(buffer 白塞 / 频率白提)。
2. 每档容量 n 有一个**匹配频率 f\* = n/(nCCDAB·tCK)**,把 PE 推到 f\* 即
   吃满列读间隔的全部空档(2026-08-23 模型修订:DRAM 节拍恒为 preset
   nCCDAB=6/4,**永不被计算拉长**——FIMDRAM 先例;计算功率单独记账):
   **n=8 ↔ 1.73 GHz;n=16 ↔ 3.47 GHz;n=32 ↔ 6.93 GHz**(6.93 超出
   MAC 树流水后的实测 Fmax≈2.67 GHz,n=32 实际由 PE 频率上限封顶)。
3. 面积上(Genus 实测):沿速率轴提频便宜((16,2) 从 667 MHz 到 1.3 GHz
   仅 +19%),沿容量轴翻倍很贵((16,2)→(32,4) 同频 +73~87%)。
4. 以 AttAcc 论文 §7.7 的 in-bank 面积口径为预算线(见 §6):
   (16,2) 已综合的三个频点都在预算内,(32,4) 从 1.0 GHz 起越线。
   **悬置**:(16,2) 推到其匹配点 2.08 GHz、以及 (8,1) 高频路线的面积,
   RTL sweep 尚无该综合点(§7 缺口清单)。

## 2. 算子维度:MQ 把 GEMV 摞成小 GEMM

单 agent decode 的 score 一步是 GEMV(矩阵×向量):
q[1×d_head]·K^T[d_head×L] → s[1×L]。其中 d_head=每个注意力头的向量宽度
(模型常数,如 128),L=上下文长度(历史 token 数)。

MQ 把 n 个 agent 的查询摞起来:

| 步 | A[M×K] | B[K×N] | M | K(累加维) | N |
|---|---|---|---|---|---|
| ① score | Q[n×d_head] | K^T[d_head×L] | **n = 并发查询数(本文主角)** | d_head | L |
| ③ context | P[n×L] | V[L×d_head] | n | L | d_head |

K/V 只读一遍、n 条查询共用——**KV 读出量不随 n 涨,PE 乘加量随 n 线性涨**。
所以 n 越大,越是"用便宜的 PE 算力换昂贵的 DRAM 读带宽"。n 的上限就由
下面两个旋钮的较小者决定。

## 3. 数据摆放维度:谁驻在哪

- **K/V**:驻在 bank 的 DRAM 阵列里,按 32 B 一次的列读 (column read) 流出,
  不搬家。
- **查询 Q**:每条查询在每个 bank 只需自己对应的 64 B 切片 (query slice),
  驻在该 bank PE 的 **GEMV buffer**(SRAM)里。
- **每条驻留查询**另需少量 per-Q 状态(累加寄存器、行界暂存等,
  见 `experiments/mq_command/DATAFLOW.md` 的寄存器清单)。

**容量轴**由此而来:`mq_query_capacity = gemv_buffer_bytes / 64`
(`src/ramulator_wrapper.py:32`)。实测:512 B→8 条,1024 B→16 条,2048 B→32 条。

## 4. 时序维度:速率轴与列读间隔

DRAM 相邻两次全 bank 列读的最小间隔叫 nCCDAB
(column-to-column delay, all-bank,单位:命令周期,1 命令周期 tCK=0.769 ns,
`src/ramulator_wrapper.py:68`)。MQ 命令的有效间隔由三项取最大
(`mq_interval_cycles`,`src/ramulator_wrapper.py:37-50`):

```
interval = max( nCCDAB_preset, pe_cycles )       (2026-08-23 修订)
pe_cycles = ceil( n / (f · tCK) )
```

符号:n=本次列读要服务的查询条数(个);f=PE 频率(GHz);tCK=0.769 ns;
f·tCK=一个命令周期里 PE 自己跑过的周期数,每个 PE 周期完成一条查询的
16 路 FP16 乘加 (16-lane MAC);nCCDAB_preset=6(功耗受限)/4(不受限),
是纯 DRAM 读侧约束(构成里无计算项)。**计算永不拉长 DRAM 节拍**
(Samsung FIMDRAM 先例:PIM 计算由命令流驱动,官方 PIMSimulator 无任何
PIM 专属时序);PE 功率单独记账:`mq_pe_power_w`,对照 AttAcc Fig.7(a)
的 116 W IDD7 预算线(n=32 全速增量 37.1 W,远在预算内)。
早先的"等功率拉伸"模型已废除(把计算能量摊进列命令间隔,与设计意图
和 FIMDRAM 先例相悖——宸逸 2026-08-23 裁决)。

**算例**(实测值,`results_c_points.json`):n=16、f=1.3 GHz 时
pe_cycles=ceil(16/(1.3×0.769))=ceil(16.004)=17——16 条查询要 17 个命令周期
才服务完,列读只能每 17 周期发一次,**频率是瓶颈,buffer 里的 16 条没喂饱**。

## 5. 两轴的匹配点(实测)

f 提高,pe_cycles 下降,直到撞上功耗地板;地板一撞,再提频无收益。
每档 n 的功耗地板与匹配频率 f\* = n/(地板×tCK)(模型直接取值):

| n(容量档) | 间隔地板 (cycles, preset) | 匹配频率 f\* = n/(6·tCK) | 对应已综合配置 |
|---:|---:|---:|---|
| 8 | 6 | **1.73 GHz** | (8,1) buf×1(仅综合了 667 MHz 点) |
| 16 | 6 | **3.47 GHz** | (16,2) buf×2(已综合 667 M/1.0 G/1.3 G) |
| 32 | 6 | **6.93 GHz(被 PE Fmax≈2.67 GHz 封顶)** | (32,4) buf×4(已综合 1.0 G/1.3 G) |

MAC 树频率不是墙:每级 FP16 算子都可再流水(T-cube/CVFPU 风格 retiming,
`MACTREE_FMAX.md` 实测:单级 1.30 GHz、2 级子流水 2.16 GHz、3 级 2.67 GHz)。

C 系列仿真的 8 个点印证(C3 相对 C1 的加速比,`results_c_points.json`;
配置记号 (n_q, n_c) = 驻留查询数、context 累加通道数):

| 配置 | 0.666 GHz | 1.3 GHz | 2.08 GHz | 3.2 GHz |
|---|---:|---:|---:|---:|
| (16,2) 间隔/加速 | 32 / 2.46× | 17 / 2.90× | 11 / 3.08× | 7 / 3.16× |
| (32,4) 间隔/加速 | 63 / 2.90× | 33 / 4.06× | 21 / 4.45× | 14 / 4.70× |

(2026-08-23 重测,`results_c_points.json`;相对旧"等功率拉伸"模型全线
上修,主要来自 context 相位间隔回到地板 6。)读法:两配置在 3.2 GHz 仍
未到匹配点(f\*(16)=3.47、f\*(32) 被 Fmax 封顶),PE 频率越高越接近
吃满列读空档;0.666 GHz 的 (16,2) 间隔 32≫6(buffer 白塞大半)。
**此前的错读**:把"(配置, 频率)"当一维绑定 sweep,掩盖了两轴各自的
瓶颈归属——详见 `audit/06_area_balance_0822.md` §7。

## 6. 面积维度与 in-bank 预算线

Genus 实测(N28,SS/0.72 V/125 °C;`fugue-logic-die-rtl/syn/collect_mq_results.py`):

| 配置 | 面积 (µm²) | vs AttAcc bank PE (83,563) |
|---|---:|---:|
| MQ (8,1) buf×1 @667 MHz | 75,825 | 0.91× |
| MQ (16,2) buf×2 @667 MHz | 129,369 | 1.55× |
| MQ (16,2) @1.0 GHz | 143,311 | 1.71× |
| MQ (16,2) @1.3 GHz | 154,511 | 1.85× |
| MQ (32,4) @1.0 GHz | 248,121 | 2.97× |
| MQ (32,4) @1.3 GHz | 267,571 | 3.20× |

两轴的面积斜率(实测):速率轴便宜——(16,2) 667 MHz→1.3 GHz +19.4%,
(32,4) 1.0→1.3 GHz +7.8%;容量轴贵——(16,2)→(32,4) 同频 +73~87%
(buffer、per-Q 状态、context 通道都翻倍)。

**预算线**(AttAcc 论文 §7.7,原文数字):每张 121 mm² 的 HBM3 DRAM die 上
128 个 GEMV 单元(1z-nm DRAM 工艺下每个 0.094 mm²)+ 32 个累加器
(每个 0.036 mm²),合计 13.12 mm² = **10.84%**;按更早 in-DRAM PIM 工作的
单元面积保守估则约 **25%**——这是 AttAcc 自认还站得住的上限。
die 占比只依赖面积**倍数** k(分子分母同工艺同流程,缩放因子相消):
die% = (12.03·k + 1.15)/121。按 25% 线反解 **k\* = 2.42×**。
结论:(16,2) 三个已综合频点 16.3%–19.3%,在线内;(32,4) 从 1.0 GHz 起
30.5%–32.8%,越线。换算链的完整审计(含 N28→7 nm 口径、两条方向性
caveat)见 `audit/06_area_balance_0822.md` §5–6。

## 7. 设计空间覆盖缺口(悬置清单)

现有 12 个综合点只覆盖了 (配置,频率) 网格的**对角线**,两轴解耦视角下缺:

1. **(16,2) @≈2.08 GHz**(其匹配点)——超过 1.3 GHz 需 MAC 子流水
   (`MACTREE_FMAX.md` 实测 2 级子流水 Fmax≈2.16 GHz,恰好覆盖);
   缺此点无法回答"(16,2) 推满后是否仍在预算内"。
2. **(8,1) @1.3 GHz**(其匹配点)——最省面积的路线,只综合过 667 MHz。
3. **(8,1) 的 C3 仿真行**——`results_c_points.json` 只有 (16,2)/(32,4)。

## 8. 数据来源与复现

- 时序模型:`src/ramulator_wrapper.py:32-50`;地板与 f\*:
  `python3 -c "from src.ramulator_wrapper import mq_interval_cycles; ..."`
  (f 取大数即得纯功耗/通路地板)。
- 仿真 8 点:`python3 experiments/mq_command/run_c_points.py` →
  `experiments/mq_command/results_c_points.json`。
- 面积 12 点:`cd /data2/chenyi9/KV-PIM/fugue-logic-die-rtl/syn &&
  python3 collect_mq_results.py`。
- AttAcc §7.7 原文:`KVPIM-1Fugue-ASPLOS2027/ref/attacc.pdf`。
