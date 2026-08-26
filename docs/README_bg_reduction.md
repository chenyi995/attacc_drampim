# BG 级归约层:两相的累加/缓冲配置与面积约束(2026-08-23 定稿待审)

目标读者:计算机专业学生/接手人,不预设了解本项目或 DRAM/PIM;
概念首现即释,数字标出处;推导与悬置明确区分。
读完应能回答:AttAcc 的层级归约里 BG 级到底加什么、MQ 化后 score/context
两侧各要配几套什么 buffer、频率约束是什么、面积怎么入账、
现在仓库缺什么。本页由用户 2026-08-23 的层级审查讨论固化。

## 0. 记号与背景(一段)

一个注意力头的两步:score S=Q·K^T(K^T 形状 [d_head×L],其行=维度 dim)
与 context O=P·V(V 形状 [L×d_head],其**行=token**)。AttAcc_bank 把
计算放在每个 bank 的 GEMV 单元(16 条乘加 lane),往上是层级归约:
bank → BG(bank group,4 bank 一组,归约点在 DRAM die 的 GBUS CTRL)
→ rank/pCH(pseudo-channel,半宽子通道)→ buffer/logic die。
**切分类型**(AttAcc §5.1 原文口径):row-wise partitioning=沿被乘矩阵
的行切,各单元出**同一输出的部分和,必须相加**;column-wise=沿列切,
各单元出**不同输出,只需拼接**,该级累加器旁路 (bypass)。

## 1. 谁在哪级切哪个轴(全表,推导自 trace_gen 循环边界)

dim 轴只有 d_head=128,只够 bank 级分一次(4 bank×32 维,再分喂不满
16 lane);token 轴有 L,其余层级全用它:

| 层级 | score(K^T) | context(V) |
|---|---|---|
| bank(BG 内 4 个) | **dim 切**(row-wise of K^T):树加出 token 分数部分和 | **dim 切**(column-wise of V):16 累加器出 32 维段部分和 |
| **BG 之间(4 个)** | token 切(column-wise of K^T):不同 token 的分数,**拼接** | **token 切(row-wise of V):同维部分和,真加法** |
| BG 内 bank→BG | **同 token 部分和,真加法**(4 入树加) | 不同维,**拼接/bypass** |
| rank / pCH / die | 拼接(score);**继续归约**(context,per-pCH accu) | |

原文锚:*"hierarchical accumulators gather data at the GBUS CTRL per
BG"*;*"when row-wise partitioning is employed for the V_i matrix, the
per-BG accumulators reduce the partial result"*;*"the accumulator ...
is simply bypassed when a column-wise partitioning is used"*;归约放在
DRAM die 的理由:*"decreases the amount of overall data transfer"*
(段尾先加成一份再上行,流量 ÷4)。

## 2. BG 级归约层的 MQ 化配置(本页核心)

单查询时 BG 级归约是同拍空间加法,几乎无状态。MQ 后 n 条查询分时
流过,归约逻辑本身仍可流水(同一查询的 4 份部分和同拍——`MAC_AB`
全 bank 广播保证步调一致),**新增的是每侧的暂存 SRAM**:

| 侧 | 归约逻辑 | 配套 SRAM buffer | 套数 |
|---|---|---|---|
| score(bank→BG) | 4 入 FP 加法树,流水化 | 对齐/排空缓冲:MVSB 按 bank 串行排空,树加要收齐同查询同 token 的 4 份部分和 | **n_q 套**(每驻留查询一套在途上下文) |
| context(BG 之间) | 部分和累加器(FP32) | 部分和向量缓冲:每查询 32 维段和(在途)+ 排空重叠的 ping-pong | **n_c 套**(context 相驻留数,非 n_q——非对称设计的又一收益) |

**频率约束**:BG 级归约层的处理速率必须 ≥ bank PE 的结果产出速率,
否则它成为新瓶颈——**GBUS CTRL 归约层与 bank PE 同频**(0.666–3.2 GHz
随 C3 频点走)。频率本身不是墙:归约的 FP 加法链与 MAC 树同款,每级
可流水(T-cube/CVFPU 风格 retiming,`MACTREE_FMAX.md` 实测 2 级 2.16 GHz、
3 级 2.67 GHz)。推论:PE 提频的时序收敛面积代价**不只发生在 bank PE**,
BG 级也要按同频综合——C-abl-3 现有频点面积(只含 bank PE 与 logic die)
在这一项上**偏乐观**。

**功率口径**(2026-08-23 模型修订联动):计算功率与 DRAM 列流分账——
DRAM 节拍恒为 preset nCCDAB,计算(bank PE 与本页的 BG 归约层)的功率
增量单独对照 AttAcc Fig.7(a) 的 116 W IDD7 预算线(`mq_pe_power_w`;
bank PE n=32 全速 37.1 W,BG 归约层增量待 RTL 综合后并入同一本账)。

## 3. 面积约束

### 3.1 基线(AttAcc §7.7 原文数字)

- 每 pCH:16 个 GEMV 单元 + **4 个 accumulator**(即每 BG 一个);
- 每个 accumulator:**0.036 mm²**(1z-nm DRAM 工艺);
- 每 die 32 个 → 32×0.036 = **1.152 mm²/die**——正是 in-bank 面积
  预算公式 die% = (12.03·k + **1.15**)/121 里的第二项。

### 3.2 MQ 化增量(参数化,数值待综合——不编数)

BG 级增量 = 流水寄存器 + 两侧 SRAM:

```
ΔA_BG ≈ A_pipe(f) + n_q·B_score + n_c·B_ctx
B_score = score 侧每查询在途对齐缓冲(∝ 一次 MVSB 突发量)
B_ctx   = context 侧每查询部分和向量(32 维 × FP32 = 128 B)+ ping-pong
```

其中只有 B_ctx 的 128 B/查询是可推导值(32×4 B);B_score 与 A_pipe(f)
**必须综合才有数**(标 TBD)。入账方式:die% 公式的 1.15 项变为
1.15·(1+ΔA_BG/0.036·32),与 bank PE 的倍数 k 一起受 25% 预算线约束。

### 3.3 约束结论(定性,待 C-abl-3 扩点后定量)

n_c 侧 SRAM 是小头(n_c=4 → 512 B/accu 量级);风险项是**同频综合**:
1.3 GHz 以上 BG 归约层的时序面积与 bank PE 同样受 Fmax 制约
(`MACTREE_FMAX.md`:FP 加法链单级 ~1.3 GHz,再高须子流水)。

## 4. 缺口记录(截至 2026-08-23,须进审计)

1. RTL 无 BG 级模块(现有:`mq_bank_pe.sv`/`mq_diff_decoder.sv`/
   `mq_score_store.sv`/`fugue_mq_logic_die.sv`)——C-abl-3 需新增
   BG 归约层模块并入 12 点 sweep;
2. `DATAFLOW.md` §3 寄存器清单全部是 per-bank 项,无 BG 级行;
3. 解析时序模型未建 BG 归约速率项(现由"同频"假设吸收;若 BG 级
   降频设计,须加显式瓶颈项);
4. 面积账:C-abl-3 现值不含 BG 级同频综合代价,高频点偏乐观
   (方向:MQ 面积被低估,与 audit 06 §5.3 caveat (a) 同向)。

## 5. 本页结论一句话

BG 级:score 侧 4 入树加 + n_q 套对齐缓冲,context 侧累加器 + n_c 套
128 B 部分和缓冲,整层与 bank PE 同频;基线面积 0.036 mm²×32/die
(原文),MQ 增量待综合;这一层目前不在 RTL/清单/时序模型中,是
已登记的审计缺口与 E4 项。
