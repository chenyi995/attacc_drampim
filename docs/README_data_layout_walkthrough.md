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

一次 decode 扫描要读**整段上下文**（12560 行/head），所以每条 channel 上它自己那
一段被**从头扫到尾**：`ACT 次数 = ceil(该 channel 的行数 / 256)`。

| ch | A3b 行 / ACT | A4 行 / ACT | A4b/A5/A6 行 / ACT |
|---:|---|---|---|
| ch0 | 3328 / 13 | 4352 / 17 | 3584 / 14 |
| ch1 | 3088 / 13 | 4112 / 17 | 3584 / 14 |
| ch2 | 3072 / 12 | 4096 / 16 | 3584 / 14 |
| ch3 | 3072 / 12 | 4352 / 17 | 3584 / 14 |
| ch4 | 3328 / 13 | 4112 / 17 | 3104 / 13 |
| ch5 | 3088 / 13 | 4096 / 16 | 3328 / 13 |
| ch6 | 3072 / 12 | 4352 / 17 | 3328 / 13 |
| ch7 | 3072 / 12 | 4112 / 17 | 3328 / 13 |
| ch8 | 3328 / 13 | 4096 / 16 | 3328 / 13 |
| ch9 | 3088 / 13 | 4352 / 17 | 3088 / 13 |
| ch10 | 3072 / 12 | 4112 / 17 | 3328 / 13 |
| ch11 | 3072 / 12 | 4096 / 16 | 3328 / 13 |
| ch12 | 3328 / 13 | — 闲置 | 3328 / 13 |
| ch13 | 3088 / 13 | — 闲置 | 3328 / 13 |
| ch14 | 3072 / 12 | — 闲置 | 3088 / 13 |
| ch15 | 3072 / 12 | 1056 / 5 | 1056 / 5 |
| **最忙** | **3328 / 13** | **4352 / 17** | **3584 / 14** |
| 活跃 channel | 16/16 | 13/16 | 16/16 |

**扫描时间 = 最忙那条 channel 的时间**（16 条 channel 并行流），所以这一列
（A3b 13 次、A4 17 次、A4b/A5/A6 14 次）就是三档 decode 时间的比值来源。

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

| | unit → channel 的规则 | diff 行 | 活跃 channel | 最忙 channel | ACT |
|---|---|---|---:|---:|---:|
| **A3b** | `base + (u % 4)`，head 独占 4 条 | 内联在 master 流里 | 16/16 | 3328 行 | 13 |
| **A4** | `base_m + (u % 3)`，head 独占 3 条 master | 独立池 **ch15** | 13/16 | 4352 行 | 17 |
| **A4b** | 全局表 `(head x 50 + u) % 15` | 独立池 **ch15** | 16/16 | 3584 行 | 14 |
| **A5** | 同 A4b | 同 A4b | 16/16 | 3584 行 | 14 |
| **A6** | 同 A4b | 同 A4b | 16/16 | 3584 行 | 14 |

A5 = A4b 的布局 + prefill 注意力全上 PIM（MQ 批命令）；A6 = A5 + 逐请求选边。
**三者的 decode 布局逐位相同。**

（A1/A3/A3a 不在本页：它们按老的"每段补齐到一个 256-token chunk"计价，
2026-09-03 起只有 A3b 及之后走 striped-append。见 `sessions/2026-09-03.md`。）

---

## 9. 这个模型没有建的东西

写正文引用前必须知道的边界：

1. **喂给 Ramulator 的是"每条 channel 一个 run"**：
   `_placement_channel_runs` 发出的是 `(channel x 1 GiB, +8 MiB, 行数, channel, 1)` ——
   起点是 channel 基址、长度是该 channel 的真实行数。上面第 3–6 节的 unit/槽是
   **布局模型**，它决定行数；行数之内 token 怎么摆是 **AttAcc 的 trace 生成器**
   按第 2 节的几何铺的。两者是两层，不要当成一层读。
2. **跨 run 的行缓冲命中抓不到**：`ramulator_wrapper.run()` 对每个 run 起一次独立的
   Ramulator 仿真（"restarts Ramulator for every run"），开局没有 open row。
   段越碎、cold-start ACT 越多；striped-append 把同一条 channel 上的 unit 打包成
   一段连续地址，正是为了少切几刀。
3. **同一条 channel 上的多个 head 的交错没有单独建模**：`heads_per_hbm > 16`
   时（本例不触发）多个 head 共用一条 channel，模型把它们的 unit 当成交错打包后的
   一整段。
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

# 载荷向量
python3 -c "
from src.workload_runner import _striped_append_channel_rows as R, _heads_per_hbm
H = _heads_per_hbm(40//2, 10, 2)              # = 4
for p in ('slice-append','master-diff-slice-append','master-diff-table-append'):
    m = 12560 - (264 if p=='slice-append' else 0)
    print(p, [int(x) for x in R(m, 264, policy=p, heads_per_hbm=H)])
"
```
