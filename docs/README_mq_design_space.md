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
   **容量轴只约束 Q**(2026-08-24 裁决):context 相的分数向量 P 复用近零,
   不驻留、改流式,其上限是 TSV 移动总线(见 §4 末与 §7 第 6 条)。
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
  驻在该 bank PE 的 **GEMV buffer**(SRAM)里——Q 要被本 bank **每一个** K 列
  复用(L/256 次),必须驻留。
- **分数向量 P**(context 相输入,每 bank 每查询 L/8 B):**不驻留,流式**
  (2026-08-24 裁决)。P 的一个条目在本 bank 只被两个输出趟各用一次,复用近零,
  驻留买不到东西;以 `PIM_MV_GB` 经 TSV 送入双缓冲半区、边送边耗,
  上限是移动总线带宽与方向转向,不是 buffer 容量(§4 末)。
- **每条驻留查询**另需少量 per-Q 状态(累加寄存器、行界暂存等,
  见 `experiments/mq_command/DATAFLOW.md` 的寄存器清单)。

**容量轴**(只含 Q)由此而来:`mq_query_capacity = gemv_buffer_bytes / 64`
(`src/ramulator_wrapper.py` 同名函数)。实测:512 B→8 条,1024 B→16 条,
2048 B→32 条。

## 4. 时序维度:速率轴与列读间隔

DRAM 相邻两次全 bank 列读的最小间隔叫 nCCDAB
(column-to-column delay, all-bank,单位:命令周期,1 命令周期 tCK=0.769 ns)。
MQ 命令的有效间隔由两项取最大(`src/ramulator_wrapper.py` 的
`mq_interval_cycles`):

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
和 FIMDRAM 先例相悖——chenyi9 2026-08-23 裁决)。

**算例**(实测值,`results_c_points.json`):n=16、f=1.3 GHz 时
pe_cycles=ceil(16/(1.3×0.769))=ceil(16.004)=17——16 条查询要 17 个命令周期
才服务完,列读只能每 17 周期发一次,**频率是瓶颈,buffer 里的 16 条没喂饱**。

**context 相的第三条线:TSV 移动总线(P 流的上限,2026-08-24)**。
P 流式后,context 相里每查询每 channel 有 L/32 条 MV_GB(每条 32 B)走
半双工 TSV 移动通路;当前命令序同 bank 组连发,受同组列间隔
nCCDL=4 tCK 限速(= 原始 TSV 1024 bit @ 5.2 Gbps 的一半),另有
MVSB↔MVGB 方向转向 (nRTW/nWTRL)。平衡条件(推导并经实测闭合,误差 3–5%):
**P 流不拖慢 V 扫描 ⟺ n ≤ interval**;超过后 context 时间随 n 线性涨
(总线主导)。跨 BG 交错可回到 nBL=2 tCK、把平衡点翻倍为 n ≤ 2·interval
——记 §7 第 6 条,未动。

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

C 系列仿真的 12 个点印证(C3 相对 C1 的加速比,`results_c_points.json`;
2026-08-24 流式 P 重测:配置记号只剩 **n_q**——context 相与 score 相同用
n_q、一遍完成,不再有 n_c 驻留档与 ⌈n_q/n_c⌉ 趟重扫;score+context 的
列读与行激活均 **÷n_q**):

| n_q | 0.666 GHz | 1.3 GHz | 2.08 GHz | 3.2 GHz |
|---:|---:|---:|---:|---:|
| 8 间隔/加速 | 16 / 2.89× | 9 / 4.32× | 6 / 5.63× | 6 / 5.63× |
| 16 间隔/加速 | 32 / 2.91× | 17 / 4.52× | 11 / 5.73× | 7 / 6.63× |
| 32 间隔/加速 | 63 / 2.87× | 33 / 4.33× | 21 / 5.40× | 14 / 6.35× |

(相对 (n_q,n_c) 驻留模型全线上修,主因是去掉了 context 的多趟重扫;
n=8 在 2.08 GHz 已到匹配点,再提频无收益 ✓ 与公式一致。)读法:
16/32 在 3.2 GHz 仍未到匹配点(f\*(16)=3.47、f\*(32) 被 Fmax 封顶);
n=32 各频点不再优于 n=16——context 相被 TSV/PE 线压住后,n 超过
2·interval 的部分只摊薄 score,收益贴平(§4 末)。
**此前的错读**:把"(配置, 频率)"当一维绑定 sweep,掩盖了两轴各自的
瓶颈归属——详见 `audit/06_area_balance_0822.md` §7。

## 6. 面积维度与 in-bank 预算线

Genus 实测(N28,SS/0.72 V/125 °C;`fugue-logic-die-rtl/syn/collect_mq_results.py`)。
注意:下表 build 名沿用**旧 (n_q, n_c) 驻留设计**的记号;流式 P(2026-08-24)
之后 n_c 不再是设计参数(context 侧改为每查询一组 16×FP16 累加寄存器 +
双缓冲半区里 32 B/查询的 P 现役块,buffer 需求与 Q 侧同为 n_q×64 B),
**面积点待 RTL 按新结构重扫**——在此之前下表仅作旧结构的量级参考:

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
4. **批量>容量的静默封顶**(2026-08-23 记录,chenyi9 提出,暂不动):
   `query_batch`/`n_q` 配置超出 `mq_query_capacity` 时两条路径都静默降级
   多趟,报告不暴露生效值——待裁决:prefill 配置矛盾改报错、decode 调度
   拆趟保留、报告加 effective_shared_queries/sweep_passes 字段。
5. **拆趟的行交错替代设计**(2026-08-23 记录,暂不动):现行"波外层"
   每趟重做全部 ACT(passes−1 倍额外行激活);"行外层、波内层"可让
   ACT 不随趟数涨,代价是每行重装 Q(64 B/条)。context 相 8 趟一类
   配置下值得算账;AttAcc 与本仓库 trace 均未采用。
6. **MV_GB 跨 BG 交错**(2026-08-24 记录,暂不动):当前 trace 的 P 流
   同 bank 组连发、吃 nCCDL=4 tCK,只用到 TSV 原始速率的一半;把 MVGB
   循环序改为块号在外、BG 在内即可回到 nBL=2 tCK,context 相的
   TSV 平衡点由 n ≤ interval 翻倍为 n ≤ 2·interval。一行生成序改动,
   等裁决。
7. **P 跨两个输出趟的送法**(2026-08-24 记录,未裁决):V 按输出趟为主
   序摆放时,严格流式意味着第二趟要重送 P(MV_GB 流量 ×2);替代是 V 改
   token 主序摆放(同 token 两列相邻,P 块跨两趟就地复用,per-Q 累加
   寄存器 32→64 B)。当前 trace 按"送一遍、块跨趟保留"计价(等效
   token 主序摆放),两者取舍未裁决。

## 8. 数据来源与复现

- 时序模型:`src/ramulator_wrapper.py`(`mq_query_capacity`/
  `mq_interval_cycles`/`mq_pe_power_w`);地板与 f\*:
  `python3 -c "from src.ramulator_wrapper import mq_interval_cycles; ..."`
  (f 取大数即得纯 preset 地板)。
- 仿真 12 点:`python3 experiments/mq_command/run_c_points.py` →
  `experiments/mq_command/results_c_points.json`。
- 面积 12 点:`cd /data2/chenyi9/KV-PIM/fugue-logic-die-rtl/syn &&
  python3 collect_mq_results.py`。
- AttAcc §7.7 原文:`KVPIM-1Fugue-ASPLOS2027/ref/attacc.pdf`。
