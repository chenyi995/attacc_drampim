# A3b–A6 的数据布局走查（GPT-13B / baseline）

一个具体请求、一个具体 head，逐 token 说清楚 **KV 摆在哪条 channel、哪个
stripe unit、行内第几个槽**，以及**一次 ACT 到底打开了哪些 token**。

本页所有数字都由脚本从仓库代码和 workload JSON 生成，不是手推。走查对象：

| 项 | 值 |
|---|---|
| 模型 | **GPT-13B**（40 层，40 Q head = 40 KV head，`dhead` 128，bf16）|
| 配置 | `--ngpu 2 --num-hbm 10`（`output/_orch2/common.sh` 的模型表）|
| workload | `wl_baseline_alltoall_N16_C32_D2.json`，`--epic-prefix-recompute-tokens 8` |
| 请求 | **`t1n0`**（tier-1）。选它是因为 tier-0 的复用段位置没有位移、**重算行为 0**，讲不出 master/diff |
| 代码 | `src/workload_runner.py` 的 `_striped_append_channel_rows`（striped-append 布局，2026-09-03）与 `ramulator2/trace_gen/gen_trace_attacc_bank.py`（AttAcc 的行内摆放）|

选 GPT-13B 是因为它每个 head 正好分到 **4 条 channel**，四档的差别最容易看清。

---

## 1. 三层切分：head → 堆栈 → channel → unit

```
40 个 KV head
  ├─ 张量并行 --ngpu 2  →  每 GPU 20 个本地 KV head
  ├─ 堆栈也按 GPU 切：--num-hbm 10 / 2 = 每 GPU 5 个 HBM 堆栈
  ├─ heads_per_hbm = ceil(20 / 5) = 4        ← 一个堆栈上挤 4 个 KV head
  └─ 一个堆栈 16 条 channel，stripe = 16 / 4 = 4  ← 一个 head 独占 4 条
```

四个 head 在这个堆栈上的 channel 归属（`base = (head × stripe) % 16`）：

| head | A3b 的 4 条 channel | A4 的 master 池（`stripe_m = 15/4 = 3`）|
|---|---|---|
| head 0 | ch0 ch1 ch2 ch3 | ch0 ch1 ch2 |
| head 1 | ch4 ch5 ch6 ch7 | ch3 ch4 ch5 |
| head 2 | ch8 ch9 ch10 ch11 | ch6 ch7 ch8 |
| head 3 | ch12 ch13 ch14 ch15 | ch9 ch10 ch11 |

A4 让出 ch15 给 diff 池，剩 15 条 master channel；4 × 3 = 12 条被占，**ch12/13/14 空着**
（后面第 6 节会看到这正是 A4b 要修的）。

## 2. 硬件几何：一次 ACT 恰好覆盖 256 个 token

`ramulator2/trace_gen/gen_trace_attacc_bank.py` 的常数：column 32 B、32 column/row、
`n_bank` 4、`n_bg` 4、`n_rank` 2、`n_pch` 2、`n_mac` 16。

```
一个 token 的 K 向量 = dhead 128 x 2 B          = 256 B
  切到一个 bank group 的 4 个 bank             =  64 B / bank / token
一个 DRAM row = 32 col x 32 B                  = 1024 B
  -> 一个 bank 的一行装 1024 / 64              =  16 个 token 槽
并行分区 = pCH x rank x BG = 2 x 2 x 4         =  16 个
=> 一次 ACT（每个分区的每个 bank 各开一行）覆盖 16 x 16 = 256 个 token
```

**所以 256-token 的 chunk 不是随便取的单位，它就是"一次 ACT 打开的量"。**
本文后面把它叫 **stripe unit**。

一条 channel = 1 GiB（`n_pch × n_rank × n_bg × n_bank × n_row × 1024 B`），
V 向量固定在 K 的地址 + 8 MiB 处。

## 3. 被摆放的东西：`t1n0` 一个 head 的 KV 流

### t1n0 的流（history 在前，然后是 context）

| 段 | role | token 数 | context token | **流下标 s** | 复用? | 重算(diff)行 |
|---:|---|---:|---|---|---|---|
| — | history | 256 | — | s 0..255 | — | 0 |
| 0 | sys | 16 | 0..15 | s 256..271 | 否 | 0 |
| 1 | parent_out | 256 | 16..271 | s 272..527 | **是** | **8**（该段前 8 个 token，s 272..279） |
| 2 | user | 256 | 272..527 | s 528..783 | 否 | 0 |
| 3 | user | 256 | 528..783 | s 784..1039 | 否 | 0 |
| … | … | … | … | … | … | … |
| 16 | user | 256 | 3856..4111 | s 4112..4367 | 否 | 0 |
| 17 | doc | 256 | 4112..4367 | s 4368..4623 | **是** | **8**（该段前 8 个 token，s 4368..4375） |
| 18 | doc | 256 | 4368..4623 | s 4624..4879 | **是** | **8**（该段前 8 个 token，s 4624..4631） |
| … | … | … | … | … | … | … |
| 47 | doc | 256 | 11792..12047 | s 12048..12303 | **是** | **8**（该段前 8 个 token，s 12048..12055） |
| 48 | doc | 256 | 12048..12303 | s 12304..12559 | **是** | **8**（该段前 8 个 token，s 12304..12311） |

合计 **12560 行/head**（history 256 + context 12304），其中 diff **264** 行（33 段 x 8）。


---

`sys` 和 15 个 `user` 段是这个 agent 自己算的（fresh，全是 master）；
`parent_out` 和 32 个 `doc` 段是**复用**的，每段头 8 个 token 因为位置位移要重算，
成为该段的 **diff 行**。

**四档摆的是同一条流、同样的顺序。** 变的只有两件事：

1. **unit u 落到哪条 channel**（下面每档的第 4 列）；
2. **diff 行走不走单独的池**（A3b 不走，A4 起走 ch15）。

> 一个口径差别，影响流的长度：A3b 用 `NaiveKVLayout`（`shadow_reads = False`），
> 被改写的 master 行**直接跳过**，流就是 12560 行、其中 264 行是 diff，就地内联。
> A4/A4b/A5/A6 用 `CacheBlendTLB`（`shadow_reads = True`），陈旧的 master 副本**照读
> 再掩掉**，所以 master 流仍是完整的 12560 行，diff 的 264 行**另起一条流**放 ch15，
> 物理读一共 12824 行。

---
## 4. A3b —— `slice-append`

**新增的一件事**：head 的 chunk 在自己那 4 条 channel 上轮转（A3/A3a 是全压一条）。
diff 行不分池，就地内联在 master 流里。

### A3b（slice-append，head 0 独占 ch0..ch3） —— head 0 的前 10 个 stripe unit

| unit u | 流下标 s | 该 unit 装的是 | **落到** | 该 channel 上的第几个 unit | channel 内行号 |
|---:|---|---|---|---:|---|
| 0 | 0..255 | history → history | **ch0** | 第 0 个 | 0..255 |
| 1 | 256..511 | sys#0 → parent_out#1 | **ch1** | 第 0 个 | 0..255 |
| 2 | 512..767 | parent_out#1 → user#2 | **ch2** | 第 0 个 | 0..255 |
| 3 | 768..1023 | user#2 → user#3 | **ch3** | 第 0 个 | 0..255 |
| 4 | 1024..1279 | user#3 → user#4 | **ch0** | 第 1 个 | 256..511 |
| 5 | 1280..1535 | user#4 → user#5 | **ch1** | 第 1 个 | 256..511 |
| 6 | 1536..1791 | user#5 → user#6 | **ch2** | 第 1 个 | 256..511 |
| 7 | 1792..2047 | user#6 → user#7 | **ch3** | 第 1 个 | 256..511 |
| 8 | 2048..2303 | user#7 → user#8 | **ch0** | 第 2 个 | 512..767 |
| 9 | 2304..2559 | user#8 → user#9 | **ch1** | 第 2 个 | 512..767 |

读法，以 **unit 2** 为例：它装的是流下标 512..767，也就是 `parent_out#1` 的尾部
接着 `user#2` 的开头 —— **一个段被切开、跨到下一条 channel，这正是 striped-append
"不凑整、不留空"的意思**。它落在 **ch2**，是 ch2 上的第 0 个 unit，占 ch2 打包区的
第 0..255 行。

`parent_out#1` 的 8 个重算 token（流下标 272..279）在 **unit 1**（s 256..511）内，
偏移 o = 16..23 → 落在 **ch1 的第 0 个 unit，行内槽 1**，跨 8 个分区
（pCH0/rank0/BG0 一直到 pCH0/rank1/BG3）。**A3b 里 diff 行就躺在 master 行中间，
没有单独的池。**

每条 channel 的载荷（4 个 head 一起）：

```
A3b   ch0=3328  ch1=3088  ch2=3072  ch3=3072   (head 0)
      ch4=3328  ch5=3088  ch6=3072  ch7=3072   (head 1)
      ch8=3328  ch9=3088  ch10=3072 ch11=3072  (head 2)
      ch12=3328 ch13=3088 ch14=3072 ch15=3072  (head 3)
      16/16 条活跃，最忙 3328 行 = 13 次 ACT
```

三条 channel 拿 3072/3088、一条拿 3328，差的就是 50 个 unit 除以 4 除不尽的零头
（50 = 4x12 + 2）和那个 16 行的尾巴。

---

## 5. A4 —— `master-diff-slice-append`

**新增的一件事**：把重算行搬到独立的 diff 池 ch15，master 行留在该 head 的
`15 // 4 = 3` 条 master channel 上。master 流仍是 12560 行（陈旧副本照读再掩掉），
50 个 unit 在 3 条 channel 上轮转。

### A4（master-diff-slice-append，head 0 的 master 池 = ch0..ch2） —— head 0 的前 10 个 stripe unit

| unit u | 流下标 s | 该 unit 装的是 | **落到** | 该 channel 上的第几个 unit | channel 内行号 |
|---:|---|---|---|---:|---|
| 0 | 0..255 | history → history | **ch0** | 第 0 个 | 0..255 |
| 1 | 256..511 | sys#0 → parent_out#1 | **ch1** | 第 0 个 | 0..255 |
| 2 | 512..767 | parent_out#1 → user#2 | **ch2** | 第 0 个 | 0..255 |
| 3 | 768..1023 | user#2 → user#3 | **ch0** | 第 1 个 | 256..511 |
| 4 | 1024..1279 | user#3 → user#4 | **ch1** | 第 1 个 | 256..511 |
| 5 | 1280..1535 | user#4 → user#5 | **ch2** | 第 1 个 | 256..511 |
| 6 | 1536..1791 | user#5 → user#6 | **ch0** | 第 2 个 | 512..767 |
| 7 | 1792..2047 | user#6 → user#7 | **ch1** | 第 2 个 | 512..767 |
| 8 | 2048..2303 | user#7 → user#8 | **ch2** | 第 2 个 | 512..767 |
| 9 | 2304..2559 | user#8 → user#9 | **ch0** | 第 3 个 | 768..1023 |

每条 channel 的载荷：

```
A4    ch0=4352  ch1=4112  ch2=4096   (head 0 的 master)
      ch3=4352  ch4=4112  ch5=4096   (head 1)
      ch6=4352  ch7=4112  ch8=4096   (head 2)
      ch9=4352  ch10=4112 ch11=4096  (head 3)
      ch12 ch13 ch14 = 闲置
      ch15 = 1056 行 = 4 个 head x 264 个 diff 行
      13/16 条活跃，最忙 4352 行 = 17 次 ACT
```

**A4 在这个配置上比 A3b 慢**（4352 vs 3328 行）。原因不是分池有害，是**取整**：
让出 ch15 之后 master 池只剩 15 条，`stripe_m = 15 // 4` 从 4 掉到 3，
一个 head 的 50 个 unit 从摊 4 条变成摊 3 条，同时 **ch12/13/14 三条完全空着**。

---

## 6. A4b / A5 / A6 —— `master-diff-table-append`

**新增的一件事**：丢掉"每个 head 固定占哪几条"的切片，改成一张**全局表**，
把所有 head 的所有 master unit 顺次摊到 15 条 master channel 上
（`channel = (head x unit数 + u) % 15`）。diff 仍在 ch15。

### A4b / A5 / A6（master-diff-table-append，全局轮转 ch0..ch14） —— head 0 的前 10 个 stripe unit

| unit u | 流下标 s | 该 unit 装的是 | **落到** | 该 channel 上的第几个 unit | channel 内行号 |
|---:|---|---|---|---:|---|
| 0 | 0..255 | history → history | **ch0** | 第 0 个 | 0..255 |
| 1 | 256..511 | sys#0 → parent_out#1 | **ch1** | 第 0 个 | 0..255 |
| 2 | 512..767 | parent_out#1 → user#2 | **ch2** | 第 0 个 | 0..255 |
| 3 | 768..1023 | user#2 → user#3 | **ch3** | 第 0 个 | 0..255 |
| 4 | 1024..1279 | user#3 → user#4 | **ch4** | 第 0 个 | 0..255 |
| 5 | 1280..1535 | user#4 → user#5 | **ch5** | 第 0 个 | 0..255 |
| 6 | 1536..1791 | user#5 → user#6 | **ch6** | 第 0 个 | 0..255 |
| 7 | 1792..2047 | user#6 → user#7 | **ch7** | 第 0 个 | 0..255 |
| 8 | 2048..2303 | user#7 → user#8 | **ch8** | 第 0 个 | 0..255 |
| 9 | 2304..2559 | user#8 → user#9 | **ch9** | 第 0 个 | 0..255 |

head 0 的 50 个 unit 现在铺满 ch0..ch14，head 1 从 slot 50 接着往下排
（`50 % 15 = 5`，所以 head 1 的第 0 个 unit 落 ch5）—— **不同 head 的 unit 交错**，
没有哪条 channel 专属于哪个 head，也没有 channel 空着。

```
A4b   ch0..ch3   = 3584   ch4 = 3104   ch5..ch8 = 3328
      ch9 = 3088  ch10..ch13 = 3328    ch14 = 3088
      ch15 = 1056（diff）
      16/16 条活跃，最忙 3584 行 = 14 次 ACT
```

**A5 和 A6 的布局与 A4b 逐位相同**，三档的差别只在 prefill 注意力走哪边：
A4b 走 GPU、A5 全部走 PIM（带 MQ 批命令）、A6 逐请求动态选边。
decode 的摆放、ACT 次数、扫描时间三者在这三档上一模一样。

---

## 7. ACT 的时候，找的是哪些 token？哪些行？

### 7.1 一次 decode 扫描要开多少次 ACT

一次 decode 扫描要读**整段上下文**（12560 行/head）。**ACT 次数不是我们算的** ——
一条 channel 的全部 extent 作为**一次** Ramulator 仿真提交，由它的行缓冲决定
（`pim_kv_extent_groups`，2026-09-03）。下表的 ACT 列是按同一套几何数出来的
行数（每个 extent 行对齐，`ceil(行数 x 4 B / 1024 B)`），用来读懂量级；
真正进结果的是 Ramulator 的时间。

| ch | A3b 行 / extent / **ACT** | A4 行 / extent / **ACT** | A4b/A5/A6 行 / extent / **ACT** |
|---:|---|---|---|
| ch0 | 3152 / 22 / **22** | 4352 / 17 / **17** | 3584 / 14 / **14** |
| ch1 | 3136 / 20 / **20** | 4112 / 17 / **17** | 3584 / 14 / **14** |
| ch2 | 3136 / 20 / **20** | 4096 / 16 / **16** | 3584 / 14 / **14** |
| ch3 | 3136 / 20 / **20** | 4352 / 17 / **17** | 3584 / 14 / **14** |
| ch4 | 3152 / 22 / **22** | 4112 / 17 / **17** | 3104 / 14 / **14** |
| ch5 | 3136 / 20 / **20** | 4096 / 16 / **16** | 3328 / 13 / **13** |
| ch6 | 3136 / 20 / **20** | 4352 / 17 / **17** | 3328 / 13 / **13** |
| ch7 | 3136 / 20 / **20** | 4112 / 17 / **17** | 3328 / 13 / **13** |
| ch8 | 3152 / 22 / **22** | 4096 / 16 / **16** | 3328 / 13 / **13** |
| ch9 | 3136 / 20 / **20** | 4352 / 17 / **17** | 3088 / 13 / **13** |
| ch10 | 3136 / 20 / **20** | 4112 / 17 / **17** | 3328 / 13 / **13** |
| ch11 | 3136 / 20 / **20** | 4096 / 16 / **16** | 3328 / 13 / **13** |
| ch12 | 3152 / 22 / **22** | — 闲置 | 3328 / 13 / **13** |
| ch13 | 3136 / 20 / **20** | — 闲置 | 3328 / 13 / **13** |
| ch14 | 3136 / 20 / **20** | — 闲置 | 3088 / 13 / **13** |
| ch15 | 3136 / 20 / **20** | 1056 / 4 / **8** | 1056 / 4 / **8** |
| **最忙** | **3152 / 22 / 22** | **4352 / 17 / 17** | **3584 / 14 / 14** |
| 活跃 channel | 16/16 | 13/16 | 16/16 |

**扫描时间 = 最忙那条 channel 的时间**（16 条 channel 并行流）。

**关键在 extent 列**：A3b 的 22 次里只有 13 次是 master 的连续 unit，
**另外 9 次是 9 个 repair group** —— 每个只用掉那一行 1024 B 里的 32 B。
A4/A4b 把各 head 的 repair 收进 ch15 打包，所以那 9 次消失了。
master/diff 分离的收益因此是 **22 → 14**，不是早先模型给的 13 → 14（倒退）。

同一条 channel、同样 3144 行，只改 repair 的摆法，真机 Ramulator 实测：

| 场景 | extent | 时间 |
|---|---:|---:|
| 1 段连续 3072 行 | 1 | 0.004744 ms |
| 12 个相邻的 256 行 unit | 12 | 0.005113 ms |
| **12 unit + 9 个散落的 8 行 repair** | 21 | **0.006431 ms** |
| **12 unit + 1 个 72 行 repair（集中）** | 13 | **0.005310 ms** |

**差 21%，全部来自 repair 是散落还是集中。**

### 7.2 一次 ACT 打开的那一行里，装的是哪 256 个 token

在 channel 内，第 k 个 unit 覆盖该 channel 打包区的第 `k x 256 .. k x 256 + 255` 行。
一次 ACT 把这 256 个 token **同时**摊在 16 个分区 x 4 个 bank 上：

| unit 内偏移 o | pCH | rank | BG | 行内槽 | bank 行内字节 | 例：属于哪个段 |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0 | 0 | 0 | 0 | 0..63 | 该 unit 第 1 个 token |
| 1 | 0 | 0 | 1 | 0 | 0..63 |  |
| 15 | 1 | 1 | 3 | 0 | 0..63 |  |
| 16 | 0 | 0 | 0 | 1 | 64..127 | 第 17 个 token |
| 17 | 0 | 0 | 1 | 1 | 64..127 |  |
| 23 | 0 | 1 | 3 | 1 | 64..127 |  |
| 255 | 1 | 1 | 3 | 15 | 960..1023 | 该 unit 最后一个 token |

每个 token 槽发 **2 条 `PIM_MAC_AB`**（`ceil(dhead / n_bank / n_mac)` = 128/4/16），
一条 MAC_AB 同时驱动一个 bank group 的 4 个 bank。


举个具体的：`t1n0` 的 `parent_out#1` 段有 8 个重算 token（流下标 272..279）。
在 **A3b** 里它们在 unit 1 内偏移 o = 16..23：

- **在同一次 ACT 里**（unit 1 就是一次 ACT）；
- 分散在 **8 个不同的分区**（pCH0/rank0/BG0 … pCH0/rank1/BG3）；
- 每个分区里都在**行内槽 1**，即 bank 行的第 64..127 字节；
- 每个 token 的 128 维被切到该分区的 4 个 bank，每 bank 64 B。

在 **A4/A4b** 里，同样这 8 个 token 不在 master 那次 ACT 里了 —— 它们在 **ch15**
的 diff 流上，和另外 32 个 doc 段的 8 个、以及另外 3 个 head 的，一共 1056 行，
连续打包，占 **5 次 ACT**（4 次满 + 1 次 32 行）。master 那边对应位置读到的是
**陈旧副本，读出来后被掩掉**（`masked_rows`）。

### 7.3 每次 ACT 之后发的命令

每个 token 槽发 **2 条 `PIM_MAC_AB`**（`ceil(dhead / n_bank / n_mac) = ceil(128/4/16)`），
一条 MAC_AB 同时驱动一个 bank group 的 4 个 bank。所以一次 ACT（256 token）之后是
`256 x 2 = 512` 条 MAC_AB，按 16 个分区并行发出。A5/A6 的 MQ 批命令把
"每 (列, query) 一条" 压成 "每列一条服务全部驻留 query"，改的是命令数和命令间隔，
**不改上面任何一条摆放规则**。

---

## 8. 四档并排

| | unit → channel 的规则 | repair 行怎么放 | 活跃 ch | 最忙 ch | extent | ACT |
|---|---|---|---:|---:|---:|---:|
| **A3b** | `base + (u % 4)`，head 独占 4 条 | **各占一个行对齐的槽**（8 行用掉一整行）| 16/16 | 3152 行 | 22 | **22** |
| **A4** | `base_m + (u % 3)`，head 独占 3 条 master | 各 head 打包进 **ch15** | 13/16 | 4352 行 | 17 | **17** |
| **A4b** | 全局表 `(head x 50 + u) % 15` | 各 head 打包进 **ch15** | 16/16 | 3584 行 | 14 | **14** |
| **A5** | 同 A4b | 同 A4b | 16/16 | 3584 行 | 14 | **14** |
| **A6** | 同 A4b | 同 A4b | 16/16 | 3584 行 | 14 | **14** |

A3b 的行数最少却 ACT 最多 —— 22 次里 9 次是 9 个 repair group，每个只装 8 个
token。这就是 master/diff 分离要买下的东西：**22 → 14**。

A5 = A4b 的布局 + prefill 注意力全上 PIM（MQ 批命令）；A6 = A5 + 逐请求选边。
**三者的 decode 布局逐位相同。**

（A1/A3/A3a 不在本页：它们按老的"每段补齐到一个 256-token chunk"计价，
2026-09-03 起只有 A3b 及之后走 striped-append。见 `sessions/2026-09-03.md`。）

---

## 9. 这个模型没有建的东西

> 2026-09-03 更新：原先列在这里的两条 ——"每条 channel 一个合成 run"和"跨 run 的
> 行缓冲命中抓不到"—— **已经不再成立**。一条 channel 的真实 extent 列表现在整体
> 提交给一次 Ramulator 仿真，ACT 由它的行缓冲决定，回到了原版 AttAcc 的记账口径。
> 见 `sessions/2026-09-03.md` §10。

写正文引用前必须知道的边界：

1. **一条 channel 上多个 head 的 extent 是顺次排布的**：模型把 head 0 的全部
   extent 排完再排 head 1，没有建模它们在 append 时间上的交错。
2. **同一请求的 cached chunk 仍假定彼此打包连续**：真实存储里，其他请求的 append
   会插在中间，把 master 流也切碎。建这个需要全局 append 序，**尚未裁决**；
   方向上它会进一步加大 A3b 的代价（也就是继续低估 master/diff 分离的收益）。
3. **绝对 row 号不进 Ramulator 的签名缓存**（`_address_mapping_signature` 原注释：
   "the model has no row-number-dependent timing"）。所以地址只通过
   channel / pCH / rank / BG / bank 字段、行内偏移、以及 **extent 的切法**影响结果 ——
   承载物理的是后者。
4. **`rows=0` 的报告 bug**：上下文行数少于活跃 channel 数时，报告侧的轮转会给尾部
   channel 记 0 行，校验器会拒绝。baseline 这种 12560 行的上下文不触发。
   见 `sessions/2026-09-03.md` §6.2。
5. **能量的 `num_attacc` 恒为 8**：与 `--ngpu` 无关，GPT-13B（`--ngpu 2`）的 PIM 能量
   偏大 4 倍。只影响能量/功率，不影响本页的行数与 ACT 次数。
   见 `sessions/2026-09-03.md` §9。

## 10. 怎么复现本页的数字

```bash
export PYTHONPATH=$PWD
# 段表与重算行：先导出 reuse plan（不跑仿真）
python3 main.py --system dgx-attacc --model GPT-13B \
  --workload workload/sweep/wl_baseline_alltoall_N16_C32_D2.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --validate-workload --workload-plan plan.json --num-hbm 10 --ngpu 2

# 每条 channel 的真实 extent（行数 / extent 数 / ACT）
python3 -c "
from types import SimpleNamespace as NS
from src.workload_runner import (_striped_append_channel_extents as EX,
                                 _heads_per_hbm, _GEN_BYTES_PER_TOKEN,
                                 _GEN_ROW_BYTES)
H = _heads_per_hbm(40//2, 10, 2)              # = 4
def loc(o,f,k): return NS(owner=o,fingerprint=f,kind=k)
def build(shadow):                            # t1n0 一个 head 的读列表
    r = [loc('t1n0','hist','master')]*256 + [loc('t1n0','sys','master')]*16
    for i in range(33):
        fp = 'reuse%02d' % i
        r += [loc('t1n0',fp,'diff')]*8
        r += [loc('t0n0',fp,'master')]*(256 if shadow else 248)
    for i in range(15):
        r += [loc('t1n0','user%02d' % i,'master')]*256
    return r
for tag, pol, sh in (('A3b','slice-append',False),
                     ('A4','master-diff-slice-append',True),
                     ('A4b','master-diff-table-append',True)):
    g = EX(build(sh), policy=pol, heads_per_hbm=H)
    b = max(g, key=lambda x: sum(-(-n*_GEN_BYTES_PER_TOKEN//_GEN_ROW_BYTES)
                                 for _,_,n in x[2]))
    print(tag, 'ch%d' % b[0], sum(n for _,_,n in b[2]), '行',
          len(b[2]), 'extent',
          sum(-(-n*_GEN_BYTES_PER_TOKEN//_GEN_ROW_BYTES) for _,_,n in b[2]), 'ACT')
"
```

真机 Ramulator 那四个数（§7.1 末表）：把 `pim_kv_extent_groups` 直接交给
`Ramulator.output_runs`，一条 channel 一组，改 repair 的摆法即可复现。
