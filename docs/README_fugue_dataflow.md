# Fugue 全数据流:每一步的矩阵形状、切分、累加、合并、轮转(零基础版)

目标读者:学过计算机、会矩阵乘法,但**不了解** LLM serving / DRAM / PIM 的人。
读完应能独立回答:数据长什么形状、摆在哪、每一步谁乘谁、在哪个维度切、在哪里加起来、
在哪里拼起来、什么在"转"。全文以 Fugue 论文正文为准;具体数字用
GPT-175B 形状(d_head=128)、BF16(每数 2 字节)、上下文 L=4096、张量并行 TP=8。

**矩阵乘法记号(全文统一)**:C[M×N] = A[M×K] × B[K×N]。
- **M** = A 的行数 = 独立的输出行数(本文里永远是"查询条数");
- **K** = 收缩维(contraction dim)= 被乘加**消掉**的维度——**累加发生在 K 维**;
- **N** = B 的列数 = 输出宽度。

---

## 0. 三十秒背景

大模型每层的注意力 (attention) 对每个 **head**(独立的注意力子空间,GPT-175B 有 96 个,
每个宽 d_head=128)做三步:

```
① S = Q · K^T     (score:查询和每个历史 token 的相关度)
② P = softmax(S)  (归一化成权重)
③ O = P · V       (context:按权重加权求和历史信息)
```

K/V 是**每个历史 token 都存一份**的缓存(KV cache)。生成每个新 token(decode)都要把
整个 KV 读一遍——这是带宽瓶颈,所以 AttAcc/Fugue 把 ①②③ 搬进 HBM 内存里算(PIM),
GPU 只留权重层。Fugue 的场景:**多个 agent 共享同一段上下文的 KV**(存一份,大家用)。

---

## 1. 每一步的 M/K/N 到底是什么(核心表)

### 1.1 注意力三步(每 head、每 agent)

| 步 | 算式 | A[M×K] | B[K×N] | M 是什么 | K(累加维)是什么 | N 是什么 |
|---|---|---|---|---|---|---|
| ① score | S=Q·K^T | Q[M×128] | K^T[128×L] | **查询条数**(见下) | **d_head=128,模型常数** | **L=上下文 token 数,随生成增长** |
| ② softmax | 按行归一化 | S[M×L] | — | 查询条数 | —(每行对 N=L 归一) | L |
| ③ context | O=P·V | P[M×L] | V[L×128] | 查询条数 | **L=上下文长度!** | **d_head=128** |

**M(查询条数)在不同模式下**:
- 单 agent decode:M=1(每步一条新查询)——这就是"GEMV"(矩阵×向量),PIM 的原生形态;
- **Fugue MQ 批**:M=n_q(如 16)——n_q 个 agent 各出一条查询,共享同一份 K/V → 变成小 GEMM;
- prefill:M=n_r(该 agent 本轮要计算的 token 数)。

**最重要的观察:①和③里,"head 维(128)"和"token 维(L)"互换了 K/N 角色**——
score 在 head 维累加、沿 token 维出一排分数;context 在 token 维累加、沿 head 维出一条向量。
但硬件里**数据不搬家**(见 §2):变的只是"哪个维被加掉"。这就是 AttAcc 把 V **转置存放**的原因。

### 1.2 GPU 上留下的权重层(对照)

| 层 | A[M×K] | B[K×N] | K/N 是什么 |
|---|---|---|---|
| QKV 投影 | [token 数×12288] | [12288×3·12288/TP] | 模型宽度(常数) |
| 输出投影 | [token 数×12288/TP] | [12288/TP×12288] | 模型宽度 |
| FFN 两层 | [token 数×12288]·[12288×4·12288/TP] 等 | | 模型宽度 |

**分界线一句话**:权重层的 K、N 全是**模型常数**(数据=权重,固定不涨),
attention 的 K 或 N 里有 **L**(数据=KV cache,随上下文线性涨)——
所以权重层留在 GPU(权重驻 GPU HBM),attention 下到 KV 所在的 PIM。

---

## 2. 数据摆在哪:每一级切的是哪个维(切分总表)

HBM 的层级:1 颗 HBM = 16 个 **channel**;每 channel = 2 个 **pCH**(伪通道)× 2 个
**rank** × 4 个 **bank group (BG)** × 每组 4 个 **bank** = 64 个 bank。bank 里按
**行 (row)**(1 KB)组织,行内 32 个**列 (column)**(每列 32 B = 16 个 BF16 数)。

### 2.1 K^T 和 V 的物理切分(它们摆定了,Q/P 来找它们)

| 层级 | 份数 | 切的是哪个维 | 这个维是什么 | 每份多大 |
|---|---|---|---|---|
| **channel(Fugue 行放置)** | ≤16 | **token 维** | 上下文长度 L | 每"行"256 token;放置表把同轮同读的行放不同 channel;**master/diff 所有权切分也在这级**(共享 token→master 通道;各 agent 重算的 token→diff 通道的紧凑段) |
| **channel 内:pCH×rank×BG** | 2×2×4=16 | **token 维** | L | 每份 L/16 token(L=4096 → 256 token/BG) |
| **bank 行内** | — | token 维 | L | 每 bank 行 16 token(每 token 在此 bank 占 64 B) |
| **BG 内 4 个 bank** | 4 | **head 维** | d_head=128 | 每 bank 32 个元素(64 B/token) |
| **bank 内 16 lane** | 16 | head 维 | 128 | 每列 16 元素,一个 token 占 2 列 |

- **K^T 视角**(score):token 切分 = 它的 **N**;head 切分 = 它的 **K**。
- **V 视角**(context,转置存):token 切分 = 它的 **K**;head 切分 = 它的 **N**(输出维)。
- **同一套摆放,两相通用**——token 永远按 channel/pCH/BG 摆,head 维永远按 bank/lane 摆。

### 2.2 Q 和 P 不摆、只驻留 + 轮转(M 维)

**M(查询维)从不切分到存储上**——查询是小向量,驻留在每个 bank 旁的
GEMV buffer(小 SRAM)里:
- score 相:每条 Q 在每 bank 驻 64 B(与该 bank 的 32 个 head 维元素对齐);n_q 条共驻;
- context 相:每条 P 在每 BG 驻 L/8 B(与该 BG 的 token 份对齐);n_c 条共驻;
- **MQ 轮转**:一列 K/V 数据从 DRAM 读**一次**进列锁存器,PE 在 n 个 PE 周期里对
  Q₁…Qₙ(或 P₁…Pₙ)各乘加一次——**M 维靠时分轮转覆盖,不靠复制数据**。

---

## 3. 全流程走一遍(decode 一步,一个 head,n_q=16 个 agent)

### 相 A:score(S=Q·K^T,K=128 维累加,N=L 维摆放)

| # | 在哪 | 干什么 | 形状片段 |
|---|---|---|---|
| 1 | GPU | **RoPE 旋转 Q**(轮转之一,数学的):每 agent 每 chunk 按位置偏移转一份 Q′ 变体 | Q[1×128]/agent |
| 2 | 链路 | n_q 条 Q′ 下行(每 agent 每 head 256 B) | |
| 3 | bank | WRGB:每条 Q 的 32 元素切片写进 GEMV buffer(2 列×32 B) | Q 切片[32]/bank |
| 4 | bank | **MAC-AB per 列**:列=某 token 的 16 个 head 维元素;**列锁存一次**,PE 轮转 n_q 槽,每槽 16 lane 乘 + **树内累加 16→1**(K 维内第一级) | 部分点积 |
| 5 | bank | **per-Q 部分和**:同一 token 的 2 列相加(K 维内 32 元素完成);行界暂存 | s 的 1/4 |
| 6 | BG | **BG 累加器**:同 token 在 4 个 bank 的部分和相加(32×4=128,**K=d_head 累加至此完成**)→ 完整分数 s_{q,t} | s[1] 标量 |
| 7 | TSV | MVSB:每 (q, t) 一个标量上 logic die | |
| 8 | die | **合并之一(分数拼装)**:master 通道的 s 按 token 序写入该 agent 的分数向量;diff 通道同机制扫该 agent 紧凑段,decoder 查 **D_i 位图**——master 在被覆盖位置的写被**过滤**、diff 直写(顺序无关);prefill 再加 **因果丢弃**(token 位置 > 查询位置的丢) | S[1×L]/agent ×n_q |
| 9 | die | softmax:每 agent 一次完整归一(对 N=L),**无 online/LSE 合并** | P[1×L]/agent |

### 相 B:context(O=P·V,K=L 维累加,N=128 维输出)

| # | 在哪 | 干什么 | 形状片段 |
|---|---|---|---|
| 10 | TSV | MVGB:本趟 n_c(如 2)条 P 的切片下行进 GEMV buffer(每 BG 每条 L/8 B);**mask gate**(合并之二的反向):D_i 位置的 P 对 master 归零、改道 diff 通道 | P 切片 |
| 11 | bank | V 扫描:列=某 token 的 16 个**输出维**分量;P_t 标量广播 16 lane,轮转 n_c 槽;**per-Q 累加器**沿本 bank 的 token 份累加(**K=L 维累加第一级,bank 内**) | o 切片[32]/bank |
| 12 | BG→pCH→die | **K=L 累加爬层级**:不同 BG 持不同 token 份、产同一输出维的部分和 → pCH 累加器 + **die 级 adder** 跨 BG、跨 channel、**跨 master/diff 两侧**求和(合并之三) | O[1×128]/agent |
| 13 | 链路 | 每 agent 的 O 上行(256 B/head);换下 n_c 条 P,重扫 V,共 ⌈n_q/n_c⌉ 趟 | |
| 14 | GPU | 输出投影 + FFN(跨 agent 批,M=agent 数);新 token 的 K/V 下行追加进 master;回到 1 | |

**为什么 score 的累加止步于 BG、context 的要爬到 die**:score 消的是 head 维(128),
它整个就摆在一个 BG 的 4 个 bank 里;context 消的是 token 维(L),它摆满了所有
BG/channel/两个通道池——**累加层级 = 被消掉的那个维的摆放层级**。这一句是整个
硬件结构的钥匙(AttAcc 为何在 BG 和 buffer die 两级都放累加器、Fugue 为何再加
跨两侧的 die adder)。

---

## 4. 三个"轮转",不要混

| 名字 | 是什么 | 在哪 | 转的是什么 |
|---|---|---|---|
| **RoPE 旋转** | 数学:把位置编码转进 Q(2 维一对的旋转矩阵) | GPU(Fugue 裁决) | Q 向量的数值,每 chunk 偏移一份变体 |
| **MQ Q 槽轮转** | 微架构:一次列读,PE 按 mod-n 计数器轮流用 n 条驻留 Q/P | bank PE | **M 维的时分复用** |
| **乒乓/双缓冲** | 结构:装载与使用两半交替(行界分数暂存、趟间 P 预装载) | bank buffer | 缓冲的角色 |

## 5. 累加与合并,一张总账

**累加(都发生在 K 维,五级)**:
1. lane 树 16→1(bank 内,一列);
2. per-Q 部分和(bank 内,token 的 2 列);
3. BG 累加器(4 bank,score 的 K=128 到此完成);
4. per-Q context 累加器(bank 内,自己 token 份);
5. pCH 累加器 + die 级 adder(跨 BG/channel/master-diff 两侧,context 的 K=L 到此完成)。

**合并(不是加,是"按位置拼")**:
1. die 分数拼装:master ∪ diff,按逻辑位置,D_i 位图过滤 master 写(顺序无关)+ 因果丢弃;
2. mask gate(反向):P 按 D_i 分流回 master/diff 两侧——保证每个位置恰好一侧服务。

## 6. 尺寸速查

| 量 | 值 |
|---|---|
| d_head / BF16 / 每 token 每 head 的 K | 128 / 2 B / 256 B |
| 列 / bank 行 / channel 行 | 32 B(16 数)/ 1 KB(16 token/bank)/ 64 KB(256 token) |
| 一条 MAC-AB 读 | 64 bank×32 B = 2 KB = 8 个 token 的 K 片 |
| Q 切片 / P 切片(每 bank·BG,每条) | 64 B / L/8 B(4096→512 B) |
| MQ 驻留 | (n_q, n_c)=(16,2) 需 buffer 1 KiB;(32,4) 需 2 KiB |
| 每 agent 每 head 每步链路 | Q′ 256 B 下 + O 256 B 上(与单 agent 相同) |
| 命令时钟 / AttAcc PE / MAC 间隔 | 1.3 GHz / 666 MHz / 6 cycles(PC)·4(NPC) |

---

配套文档:`experiments/mq_command/DATAFLOW.md`(MQ 的硬件增量、buffer/带宽核对、
三路审计与裁决记录);`README_design_check.md`(时序算术与论文-仿真器差距);
`PLAN_mq_command.md`(实现计划);RTL 微架构在 `/data2/chenyi9/KV-PIM/fugue-logic-die-rtl`。
