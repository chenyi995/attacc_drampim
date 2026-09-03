# 2026-09-03 attacc_drampim 代码 review —— 完整对话记录与结论

> 状态：**to be reviewed by me tomorrow**（xinyao，2026-09-03）
>
> 本文件记录 2026-09-02～09-03 两轮 review 对话的全部问题、回答与验证数据。
> 第一轮对话因为 claude 崩溃丢失，由 xinyao 手动搬运到第二轮，故本文件是唯一的完整记录。
>
> 术语约定（第一次出现处解释）：
> - **DAG engine / 物理事件引擎**：`src/workload_runner.py`，把一次运行展开成
>   逐事件（event）的有向无环图，每个事件有 device、时长、能耗、依赖。
> - **analytic engine / 解析引擎**：`src/ablation.py`，闭式公式模型，不展开事件。
> - **master / diff**：CacheBlend 语义下，可复用且不变的 KV 行叫 master，
>   被本 agent 重算覆盖的修正行叫 diff。
> - **chunk**：256 个 token 的 KV 块（`_NAIVE_PAGE_ROWS = 256`），本文里
>   `c_master` / `c_diff` 指 chunk 数。
> - **heads_per_hbm (h)**：一个 HBM stack 上承载的本地 KV head 数
>   = `ceil(kv_heads_local / num_hbm)`。
> - **ACT**：DRAM 的 row activation 命令；一个断开的读流每断一次要多付一次
>   ACT（+ 可能的 PRE）。

---

## 0. 本次 session 的问题清单

第一轮（2026-09-02，已丢失原文，问题为 xinyao 手动搬运）：

| # | 问题 |
|---|---|
| Q1 | 所有 supervisor json，最后的 agg 到底有没有收集 workers 的结果 |
| Q2 | 目前的 diff layout 是什么（nhead/hbm=1 与 >1 时）；nhead/channel>1 时 master 如何 layout；nhead 与 layer 谁的优先级更高 |
| Q3 | nhead/hbm>1、master/diff 分 channel 时，diff channel 的 latency 如何建模 |
| Q4 | 功耗建模有没有问题 |
| Q5 | `validate_cacheblend_events()` 的依赖关系、`_schedule_*` 的串并行关系是否正确；同 tier 内不同 agent 是串行吗 |
| Q6 | 分 channel 跑多个 Ramulator 实例 vs 一个实例跑多 trace，误差多大 |

第二轮（2026-09-03）追问：

| # | 追问 |
|---|---|
| Q1.1 | 目前 parent 只能设置 1 个 worker 吗 |
| Q1.2 | tier 串行 ⇒ agg 必须等所有 worker 输出完才能 prefill 吗 |
| Q1.3 | 多对多（非 fully-connect）时，粗粒度 tier 串行是否应该改为 parent 串行 |
| Q2.1 | nhead 与 layer 的优先级；layout 是否根本没考虑 nhead，只在 latency 建模里变成串行 |
| Q2.2 | 用真实 workload 的 k、heads/hbm、chunk 数，算出 15 master vs 1 diff 的带宽差，找到 diff channel latency 更高的点 |
| Q2.3 | nhead/hbm 与 nhead/channel 的影响究竟在哪里被纳入 |
| Q3.1 | 256 行是 Ramulator 仿真本身的粒度问题吗 |
| Q3.2 | 现在 workload 的 diff 到底有多少行；改成 16 行粒度会慢多少 |
| Q3.3 | heads/hbm=4 时 diff latency 翻 4 倍，是必要开销吗 |
| Q3.4 | 我们认为 TLB 成本低廉，但断流导致的 ACT 代价很高，这点没考虑吗 |
| Q4.x | 4 个功耗问题详细阐述 |
| Q5.x | 两个函数的 event 关系和依赖，详细过一遍 |

---

## Q6（先答，已有实测）分 channel 跑多个 Ramulator 实例的误差

**结论：误差 = 0，周期数逐位相同。** 现有做法（`src/ramulator_wrapper.py:701` 的
`execute(job)`，每个 run 一次 ramulator2 调用，DAG 里取 max）在时序上是**精确的**，
不是近似。

| L（总 token） | 每 channel | 16-channel 单实例 | 单 channel 独立实例 |
|---|---|---|---|
| 4 096 | 256 | 465 cycles | 465 cycles |
| 32 768 | 2 048 | 3 810 | 3 810 |
| 131 072 | 8 192 | 16 644 | 16 644 |

命令数也守恒。原因：Ramulator 的 HBM3-PIM 里 channel 之间没有任何共享资源
（各自的 controller、各自的 AllBank refresh、没有共享数据总线）。

两点注意：

1. 如果将来真要合成"一个实例跑多 trace"，必须按 channel **交错（interleave）**
   而不是拼接。前端 `pim_loadstore_trace.cpp` 是严格顺序发射、队列满就
   head-of-line 阻塞，拼接会人为把 channel 之间串行化。
2. 芯片级功耗预算跨 channel 没建模，但 MQ 的功耗钳位（`mq_interval_cycles`）
   本来就是 device-wide 口径，不构成额外误差。

**真正的误差不在"拆不拆实例"，在于喂给 Ramulator 的 run 内容是合成的（见 Q3）。**

---

## Q1 supervisor JSON 与 agg 收集

### Q1.0 第一轮结论（复述，本轮未推翻）

分三层：**文本上收了、时序上收了、KV 上没收。**

(a) **论文主矩阵的四个 workload 里根本没有 agg 节点。**
`experiments/paper_ladder/workloads/`：
- `workload_relay_s400w4t1.json`：tier 结构 `{0:1, 1:4}`，纯 fan-out，4 个 worker
  的输出没有下游消费者；
- `workload_sharegpt_*` / `workload_mooncakemt_*`：tier 是"轮次"，每个 agent 只有
  1 个 parent，是链不是汇聚；
- `mooncake_toolagent` / `multihoprag`：legacy RAG 列表，无 DAG。

(b) 有 agg 的是 `workload/sweep/` 里的 reduce / supervisor / alltoall。这些 agg 的
segment 确实枚举了全部上游输出（`wl_reduce.json` 的 `t1n0` 有 49 个 segment，
1 个 `parent_out` + 15 个携带 `sha16("out-t0nX")` 的 user segment，16 个 worker 一个不少）。

(c) **但只有 1/16 真的复用了 producer 的 KV。** 根因在 `src/workload.py:459-464`：
只有 `role == "parent_out"` 才被强制指向 producer；其它 segment 走
`owners.get(fingerprint)` 的"先到先得"。而 producer 自己从不注册自己的输出指纹——
`_parent_output_fingerprints()`（`src/workload_runner.py:2243`）只从**子节点的
`parent_out` 声明**里反推 parent 的输出 sha，decode 输出保留在 `"{rid}::output"` 下。
于是 `sha16("out-t0n1")` 在整个 workload 里第一次出现就是在 agg 自己身上 → 无 owner →
agg 把 15 个 worker 的输出当全新文本重新 prefill 一遍。

实测（`build_reuse_plan`）：

```
wl_reduce.json        16 个 worker-output sha, 只有 1 个进了 reuse 决策(owner=parent)
wl_supervisor_D4.json 33 个, owner 都是 hub 自己(t0n0/t1n0/t2n0)
alltoall N16 D2       240 个决策: 16 个 owner=producer(parent_out),
                                224 个 owner=**同层的另一个 consumer**
```

`workload/gen_sweep.py:83-86` 的注释写着 "the remaining upstream outputs are ordinary
fingerprinted segments carrying the producer's sha, so cross-request reuse still
resolves through them" —— **这句话是错的**，reuse 解析不到 producer。

> xinyao 裁定（2026-09-03）：**supervisor json 视为正确**，要改的是代码而不是 json。
> 修法：在 `_parse_supervisor` 之后给每个 request 注册一条
> `output_fingerprint → (request, -1)` 的 owner 记录；`_prepare_cacheblend_tlb`
> 用同一个指纹 reserve decode 输出行。

### Q1.1 目前 parent 只能设置 1 个 —— 对

`src/workload.py:77-78`：

```python
tier: int
parent_id: Optional[str]        # 单值，不是列表
```

`_parse_supervisor`（`src/workload.py:285-287`）读 `agent["parent"]`，
`_string()` 强制是单个字符串；`:327-331` 再强制"有 parent 的 agent 必须**恰好**
有 1 个 `parent_out` segment"。

所以 **schema 层面 DAG 其实是一棵树（结构边 in-degree ≤ 1）**。多对一的汇聚只能
用"额外 segment 携带别的 producer 的 sha"来表达，而那条路径正是 Q1.0(c) 里
解析不到 producer 的路径。**这两件事是同一个根因的两面：**
`parent` 单值 ⇒ 只有一条边能被强制解析 ⇒ 其余上游只能靠指纹表 ⇒ 指纹表里没有 producer。

### Q1.2 tier 串行 ⇒ agg 必须等所有 worker —— 对，而且比这更强

`src/workload_runner.py:3452` 与 `:3901`：

```python
request_ready: Tuple[str, ...] = previous_tier_done      # 每个 request 的起点
...
previous_tier_done = tuple(tier_done)                    # 整个 tier 的终点
```

`tier_done` 收集了本 tier **每一个 request 的最后一个事件**。所以：
tier *t* 的**每个** request 依赖 tier *t-1* 的**全部** request —— 这是**全 tier barrier**，
不是 parent 边。

- 对 relay（1 → 4）：等价于 parent 边，无害。
- 对 `mooncakemt`（8 个互不相干的会话被塞进 8 个 tier）：**是伪依赖**——
  会话 A 的第 5 轮要等会话 B 的第 4 轮。
- 对 reduce/supervisor：agg 等所有 worker，语义正确，但粒度过粗（见 Q1.3）。

### Q1.3 多对多时应该用 parent 串行 —— 对，而且数据已经齐了

粒度阶梯：

| 粒度 | 约束 | 现状 |
|---|---|---|
| tier barrier | tier *t* 全体 ← tier *t-1* 全体 | **现在就是这个** |
| parent 边 | child ← 它自己的 parent | `Request.parent_id` 已存在，**调度器完全没读** |
| 数据边 | consumer ← 它复用的每个 owner 的 KV 写事件 | `ReuseDecision.owner_request_id` 已存在，**调度器完全没读** |

`parent_id` 目前只被用于两处：(a) `_parse_supervisor` 的合法性校验；
(b) `build_reuse_plan` 里把 `parent_out` segment 的 owner 钉到 parent。
**调度里一次都没用。**

所以"改成 parent 串行"不需要新增任何 workload 字段，只需要把
`request_ready` 从 `previous_tier_done` 换成
`deps_of(request.parent_id) ∪ {owner 的 KV 写事件}`。

⚠️ **但这件事和 Q5 的缺失边是耦合的**：一旦放开 tier barrier，
Q5 里"consumer 读 owner 还没写的 KV"就会变成物理上不可能的 schedule
（现在只是被 append 顺序偶然掩盖，而且 mooncake/multihop 上已经掩盖失败，见 Q5.3）。
**必须先补 owner→consumer 边，再放开 tier barrier。**

---

## Q2 diff layout、nhead 与 layer 的优先级

### Q2.1 代码里有**两套互不相干的 layout**

这是理解 Q2/Q3 全部问题的关键。

#### 布局 A —— 物理字节地址（`CacheBlendTLB.finalize`，`workload_runner.py:938-1003`）

```python
for index, key in enumerate(sorted(self._reserved_rows)):
    layer, owner, fingerprint, kind = key
```

- 排序 key = `(layer, owner, fingerprint, kind)` ⇒ **layer 优先级最高**，然后
  owner、fingerprint、kind。
- `channel = channels[pool["offset"]]`，而 `pool["offset"]` **只有在一整条 channel 的
  1 GiB 用完时才 +1**（`:987-995`）。
- `_KV_CHANNELS`（`:524`）写死 master = ch0..14、diff = ch15，**与 `heads_per_hbm` 无关**。

**head 在这里根本不是地址维度**：`report()`（`:1078`）写明
`head h → channel_base + ((channel_offset + h) % channel_count)`，
即 head 决定**哪条 channel**，`(layer, owner, fp, row)` 决定**channel 内的字节偏移**。
两者正交，互不争抢。

> 直接推论（新发现 **N5**）：**这个字节布局只能物理实现 `single` 策略**——
> 一个 head 的全部 chunk 都在同一条 channel 上（chunk 只推进偏移，不换 channel）。
> `slice`（A3b/A4）和 `table`（A4b）在字节布局里**没有对应物**，
> 它们只活在计时模型 `_layout_channel_loads` 里。

> 新发现 **N4**：`NaiveKVLayout.finalize`（`:1258-1300`）反而是真的做了
> per-channel 字节布局——每 256 行一页，`channel = rotation % 16` 轮转 16 条 channel。
> 但 A3/A3a 的计时策略是 `single`（"一个 head 的所有 chunk 堆在一条 channel"），
> **与它自己的字节布局互相矛盾**：字节布局说页是轮转的，计时说全堆在一条上。

#### 布局 B —— 计时用的 per-channel 负载（`_layout_channel_loads`，`:570-630`）

```python
def _layout_channel_loads(policy, master_chunks, diff_chunks,
                          heads_per_hbm, master_channels=15) -> List[float]:
```

**输入只有 `(policy, c_master, c_diff, heads_per_hbm)`**，
没有 layer、没有 request、没有真实地址；slot 计数器每次 scan 从 0 重启。

所以：**优先级是 head > chunk，layer 这个维度完全不参与。**
同一个 `(head, chunk)` 在每一层都映射到同一条 channel。因为层是串行扫的，
这不造成建模冲突，但也意味着模型表达不了任何跨层交错的布局。

xinyao 追问"还有没有别的维度"——**有：request / agent 维度也不在里面。**
多个 agent 并发扫描时它们的 chunk 会不会撞同一条 channel，模型里看不到
（每次 scan 独立从 slot 0 开始铺）。

### Q2.2 nhead/hbm > 1 时 master 怎么摆（计时模型）

| policy | master | diff |
|---|---|---|
| `single` (A3/A3a) | head h → channel `h % 16`，该 head 全部 chunk 堆在上面 | 没有 diff pool，diff chunk 并进 head 自己那条 |
| `slice` (A3b) | head h → `max(1, 16 // h)` 条 channel 的条带，chunk 在条带内轮转 | 同上 |
| `master-diff-slice` (A4) | head h → `stripe_m = max(1, 15 // h)` 条 master channel | **恒定 1 条（ch15）**，承载全部 head 的 correction ⇒ `loads[15] = h × c_diff` |
| `master-diff-table` (A4b) | 全局 slot 轮转 15 条，`(head, chunk)` 打散 ⇒ 每条 ≈ `h × c_master / 15` | 同上 |

两点直接后果：

1. **diff channel 的负载随 `heads_per_hbm` 线性增长，master 每条不变。**
2. `stripe_m = 15 // h` 在 h ≥ 8 时钳到 1，A4 退化成"一 head 一 channel"，
   还要撞 `16 ∤ 15` 的绕回（已记录于
   `experiments/channel_parallel_validation/RESULTS.md §5.2`，该分析正确）。

### Q2.3 **量化**：diff channel 什么时候成为瓶颈（本轮新算）

先给论文矩阵的 `heads_per_hbm`（`--system dgx-attacc`, TP=8, `--num-hbm 5`）：

| model | Q heads | 本地 Q heads | gqa | 本地 KV heads | **h = heads_per_hbm** |
|---|---|---|---|---|---|
| LLAMA-7B | 32 | 4 | 1 | 4 | **1** |
| LLAMA-65B | 64 | 8 | 1 | 8 | **2** |
| GPT-175B | 96 | 12 | 1 | 12 | **3** |
| LLAMA3-8B | 32 | 4 | 4 | 1 | **1** |

再给真实 workload 的 chunk 数（EPIC k=8，即论文矩阵的 `LADDER_POLICY`）：

| workload | 每 request master 行 | 每 request **diff 行** | c_master | c_diff |
|---|---|---|---|---|
| relay (worker) | 700 | **8** | 3 | 1 |
| sharegpt | 34 – 1300 | **8** | 1 – 5 | 0 – 1 |
| mooncakemt | 6 059 – 15 172 | **8** | 24 – 60 | 0 – 1 |
| mooncake (RAG) | 2 290 – 7 322 | 0 | 9 – 29 | 0 |
| wl_reduce | 8 464 – 12 560 | 0 – 264 | 34 – 49 | 0 – 2 |

**注意：diff 只有 8 行**（EPIC 对每个"发生位移的可复用 segment"重算前 k=8 个 token）。

`_layout_channel_loads("master-diff-slice", c_master, 1, h)` 的
`max(master 15 条)` / `diff 1 条`：

```
  h \ c_master:    1     2     3     5     8    15    26    34    60
  h=1            1/1   1/1   1/1   1/1   1/1   1/1   2/1   3/1   4/1
  h=2            1/2   1/2   1/2   1/2   2/2   3/2   4/2   5/2   9/2
  h=3            1/3   1/3   1/3   1/3   2/3   3/3   6/3   7/3  12/3
  h=4            1/4   1/4   1/4   2/4   3/4   5/4   9/4  12/4  20/4
  h=8            1/8   2/8   3/8   5/8   8/8  15/8  26/8  34/8  60/8
```

**交叉点（c_diff = 1 时，master 重新成为瓶颈所需的最小 c_master）：**

| heads/hbm | stripe_m | diff 负载 | master 反超所需 c_master | 折合上下文 |
|---|---|---|---|---|
| 1 | 15 | 1 | ≥ 1 | 256 tok |
| 2 | 7 | 2 | ≥ 8 | **2 048 tok** |
| 3 | 5 | 3 | ≥ 11 | **2 816 tok** |
| 4 | 3 | 4 | ≥ 10 | 2 560 tok |
| 8 | 1 | 8 | ≥ 8 | 2 048 tok |
| 12 | 1 | 12 | ≥ 12 | 3 072 tok |

**⇒ 经验规则：上下文短于 ~2 000–3 000 token 时，单条 diff channel 就是关键路径。**

**用真实 workload 落地（本轮实测，`_placement_channel_runs` 直接算）：**

```
relay（论文的 DAG headline workload，c_master=3, c_diff=1）
  A3b LLAMA-7B  busiest=1        A4 LLAMA-7B  busiest=1  (diff lane 1)   -> 打平
  A3b LLAMA-65B busiest=1        A4 LLAMA-65B busiest=2  (diff lane 2)   -> A4 慢 2x
  A3b GPT-175B  busiest=1        A4 GPT-175B  busiest=3  (diff lane 3)   -> A4 慢 3x

wl_reduce（c_master=50, c_diff=2）
  A3b LLAMA-65B busiest=7        A4 LLAMA-65B busiest=8  (diff lane 4)   -> diff 不是瓶颈
  A3b GPT-175B  busiest=11       A4 GPT-175B  busiest=10 (diff lane 6)   -> diff 不是瓶颈
```

**这就是 xinyao 要的那个点**：relay + LLAMA-65B/GPT-175B 上，
**diff lane 的负载恰好等于 master 的最大负载**，A4 相对 A3b 的
全部"退步"来自 diff channel，而那条 lane 的真实内容只有 **8 行**。

> `experiments/channel_parallel_validation/RESULTS.md` 里"瓶颈是 master pool，
> 不是 diff channel"的结论**只对该文档那个 workload 成立**（重算率 1.59%、
> tier0 的 `c_diff = 0`）。短上下文 / 高重算率下会翻过来。

### Q2.4 nhead 到底在哪里被纳入 —— 穷举

| 位置 | 文件:行 | 纳入的是 nhead/hbm 还是 nhead/channel | 说明 |
|---|---|---|---|
| `_heads_per_hbm` | `workload_runner.py:656` | **nhead/hbm** | `ceil(kv_heads_local / num_hbm)` |
| `_layout_channel_loads` | `:570-630` | **nhead/channel**（间接） | head 决定条带/slot，从而决定每条 channel 的 chunk 数 |
| `_placement_channel_runs` | `:1428-1451` | 两者 | `loads[c] × 256` = 该 channel 上**所有 head 折叠后**的行数 |
| `_append_placement_pim_scan` `num_hbm_used` | `:1467` | **nhead/hbm** | `ceil(kv_heads / heads_per_hbm)`，**只用于能耗倍乘，不用于时间** |
| `scan_op.numOp = 1` | `:1541` | —— | **head 被折叠进行数后，trace 里的 nhead 强制为 1** |
| `Ramulator.run` `num_ops_per_hbm` | `ramulator_wrapper.py:556` | nhead/hbm | `ceil(numOp / num_hbm)`；placement 路径下恒为 1 |
| 字节布局 | `CacheBlendTLB.finalize` | **都没有** | head 只出现在 `report()` 的文字说明里 |
| 解析引擎 `_batch_scan_profile` | `ablation.py:598-655` | **都没有** | 完全没有 head 维度 |

**回答 Q2.1 的原话**：是的，**字节 layout 里没有 nhead**（head 只是 channel 索引，
不参与地址分配）；nhead 只在**计时模型**里通过 `_layout_channel_loads` 的
条带/slot 分配和 `loads × 256` 的行折叠进入，表现为"同一条 channel 上多个 head
的 chunk 串行"。

---

## Q3 diff channel 的 latency 建模

### Q3.1 256 行的量化是从哪来的 —— **不是 Ramulator 的问题**

Ramulator 本身没有 256 的粒度。256 是**我们自己**加的，有两处，语义不同：

**(1) legacy 连续 run 的向上取整**（`ramulator_wrapper.py:626-633`）

```python
if channel_base is None:
    # Ruling (chenyi9 2026-08-28): legacy contiguous runs quantize UP to
    # the natural 256-token chunk so the per-step unique-length explosion
    # of no-reuse decode collapses to ~L/256 buckets.
    run_length = ((run_length + 255) // 256) * 256
```

目的是**签名缓存去重**（否则 decode 每步一个新长度、每步一次 Ramulator）。
方向保守（最多多算 255 行）。这一条只作用于 **A1 / no-reuse** 的连续 run。

**(2) placement 路径把 chunk 数直接乘回行数**（`workload_runner.py:1440-1450`）

```python
c_master = -(-master_rows // 256)          # 向上取整成 chunk
c_diff   = -(-diff_rows   // 256)
loads = _layout_channel_loads(policy, c_master, c_diff, heads_per_hbm, master_channels)
runs = tuple((channel << 30, (channel << 30) + 8MiB,
              max(1, int(round(loads[channel] * 256))),   # <-- 再乘回 256
              channel, 1) for channel in active)
```

**这才是问题所在**：`diff_rows = 8` → `c_diff = 1` → diff lane 被计价
`1 × 256 × h` 行。然后 `_append_placement_pim_scan:1543`

```python
scan_op.pim_kv_runs = runs        # 覆盖掉调用点刚算好的 tlb.scan_runs(reads)
```

**真实地址被丢掉了。** `tlb.scan_runs()` 的结果唯一去处是 `_tlb_plan_cost`
（`:734`，5 ns/run）。

### Q3.2 现在 workload 的 diff 到底有多少行 / 高估多少

实测（EPIC k=8）：

| workload | 真实 diff 行 | 计价 diff 行（h=1 / 2 / 3） | 高估倍数 |
|---|---|---|---|
| relay worker | **8** | 256 / 512 / 768 | **32× / 64× / 96×** |
| sharegpt | **8** | 256 / 512 / 768 | 32× / 64× / 96× |
| mooncakemt | **8** | 256 / 512 / 768 | 32× / 64× / 96× |
| wl_reduce（有 diff 的 request） | ≤ 264 | 512 / 1024 / 1536 | ~2× / 4× / 6× |

整条 scan 的总计价行数 vs 真实行数：

```
relay:      priced 1024 / 2048 / 3072  vs real 700–708   (1.4x / 2.9x / 4.3x)
wl_reduce:  priced 13312/26624/39936   vs real 12560–12824 (1.0x / 2.1x / 3.1x)
```

（其中 h 倍是**应该有的**——h 个 head 都要扫；超出 h 的部分是 256 量化的虚高。）

### Q3.2b 改成 16 行粒度会慢多少

代价**不在 Ramulator 的仿真速度**，在**签名缓存的命中率**：

- Ramulator 一次 run 的墙钟时间 ≈ O(run_length)，256 → 16 反而**更快**。
- 但 `_run_signature` 里 `run_length` 是签名的一部分。粒度 256 时
  一个 request 的 decode 全过程只产生 ~`L/256` 个不同长度；粒度 16 时
  变成 ~`L/16` 个，**唯一签名数 × 16 ⇒ Ramulator 调用数 × 16**。
- 对 `wl_reduce` / GPT-175B 这种 L≈12 k、96 层的 run，这是从"分钟"到"小时"。

**建议的折中（成本几乎为零）：diff lane 不做 chunk 取整，直接用真实 `diff_rows`。**
理由：
- diff 行数**本来就少且离散**（EPIC 下恒为 `8 × 可复用 segment 数`），
  唯一取值极少，签名爆炸不会发生；
- master lane 保持 256 粒度不变，缓存命中率不受影响。

即把 `_placement_channel_runs` 改成：master 用 chunk 计价，
diff lane 用 `heads_per_hbm × diff_rows`（真实行）计价。

### Q3.3 heads/hbm=4 时 diff latency ×4，是必要开销吗

**要拆成两半看：**

| 成分 | 是否物理必要 |
|---|---|
| ×h（h 个 head 的修正行都堆在 ch15 这一条上） | **必要**——只要 diff pool 宽度恒为 1，h 个 head 的 correction 就只能串行流过那一条 channel |
| ×(256 / 真实行数) | **不必要**——纯量化虚高，relay 上是 32× |

所以 h 倍**是模型如实反映了"diff pool 宽度写死为 1"这个设计选择**，
而不是建模错误。真正该问的是**设计问题**：

> `_KV_CHANNELS = {"master": range(0,15), "diff": range(15,16)}`（`:524`）
> 是 2026-08-25 的裁定（"diff 行很少，给 1 条就够"）。
> 该裁定在 `heads_per_hbm = 1` 下成立；**在 h > 1 下不成立**，
> 因为 diff 的负载按 h 增长而 master 每条不变。
> 至少应该让 diff pool 宽度随 h 走（例如 `min(h, 4)` 条），
> 或者按 `rho_b`（密度）驱动——代码注释里 `PROVISIONAL` 已经预告了这一点。

### Q3.4 断流的 ACT 代价 —— **完全没建模**，TLB 5 ns 是唯一的代理

xinyao 的直觉是对的。逐条说明：

**(a) placement 路径下，断流信息根本到不了 Ramulator。**
调用点算出的 `op.pim_kv_runs = tlb.scan_runs(reads)`（真实、可能碎片化的 run 列表）
在 `_append_placement_pim_scan:1543` 被 `scan_op.pim_kv_runs = runs` **整体覆盖**，
`runs` 是每条 channel 一条**合成的连续 run**。Ramulator 看到的永远是连续流。

**(b) 于是 A3（skip 语义）与 A3a（read-mask 语义）的差别只剩两项：**
`_tlb_plan_cost` 的 5 ns/descriptor，以及 ±1 个 chunk 的取整。实测：

```
wl_reduce:  A3  runs/req = 18.4   tlb_cost/scan = 92 ns
            A3a runs/req = 17.4   tlb_cost/scan = 87 ns     -> 差 5 ns
relay:      A3  runs/req = 5      A3a runs/req = 5          -> 差 0 ns
            A3 rows=700           A3a rows=708（多读被 mask 的 master 行）
```

而一次 scan 本身 ≈ 34 chunk × ~605 ns ≈ **20 µs**。5 ns 是它的 **0.025%**。

**(c) 一次真实的多余 ACT 值多少：** HBM3-5.2 Gbps（`ramulator2/src/dram/impl/HBM3.cpp:32`）
`tCK = 1.3 ns`，`nRCD = 19`（24.7 ns）、`nRP = 19`（24.7 ns）、`nRC = 63`（**81.9 ns**）。

> **⇒ `_TLB_DESCRIPTOR_S = 5e-9` 比一次 same-bank 的 ACT+PRE（tRC = 81.9 ns）
> 低 16×，比单纯的 tRCD（24.7 ns）低 5×。而且 PIM 的 `MAC_AB` 是 all-bank 命令，
> 一次断流要付的是**全 bank** 的 PRE+ACT，代价只会更高。**

**(d) 所以 xinyao 的判断"A3a 显著快于 A3 是不可能的，只能说 TLB cost 被低估了"
——结论对，但机制要修正：**不是"低估了"，而是**碎片代价根本没进时间轴**。
`NaiveKVLayout` docstring 里说的"碎片惩罚由 rotation 涌现"在 placement 计时路径上
**不存在**；A3 断流的 `act-段 / act-行 / act-段` 代价一分钱没收。

**修法建议（按代价从小到大）：**
1. 把 `_TLB_DESCRIPTOR_S` 换成一个有出处的 per-run 惩罚
   （例如 `tRC = 81.9 ns` 或 `tRCD + tRP`），并在台账里注明来源；
2. 更正确：**不要覆盖 `pim_kv_runs`**——把 `tlb.scan_runs(reads)` 的真实 run
   按 channel 分组后交给 Ramulator，让它自己算 ACT。代价是 Ramulator 调用数上升。

**(e) 顺带：`_append_physical_pim_scan`（`:1383`，唯一会用真实 run 地址的函数）
是死代码**，全仓库没有调用点（只有注释引用）。warm reprice 里的 `"runs"` 分支
（`:4192`）因此也是死路径。

### Q3.5 报告口径不一致

事件的 `rows` 用 round-robin 分配的**真实**行数（`per_channel_rows`，`:1508-1524`），
而 `time_s` 用 `loads[c] × 256` 的**量化**行数。下游按 `rows` 做 J/token、GB/s
的统计会和时间对不上。

---

## Q4 功耗建模（4 个问题，按严重性排）

### Q4-(1) 🔴 PIM scan 能耗被 `num_hbm` **重复乘了一次**

三处乘法：

```
ramulator_wrapper.py:789   postprocess:            traffic = [i * self.num_hbm for i in traffic]
devices.py:526             get_time_and_energy_runs: [value * self.num_attacc for value in energy]
workload_runner.py:1549    _append_placement_pim_scan: energy * num_hbm_used
```

**为什么是重复的：**

- legacy 路径：`num_ops_per_hbm = ceil(layer.numOp / num_hbm)`，trace 里只放
  **一个 stack 的 head 数**，所以 postprocess 里 `× num_hbm` 是把全部 stack 补回来 ——
  **legacy 路径正确**。
- placement 路径：`scan_op.numOp = 1`（`:1541`，head 已经折进行数）
  ⇒ `num_ops_per_hbm = ceil(1/5) = 1`，trace 只含 **1 个 head 的 1 条 channel**。
  此时 postprocess 的 `× num_hbm = 5` 是**凭空造出 5 个 head**；
  而 `num_hbm_used = ceil(kv_heads / heads_per_hbm)`（`:1467`）才是真正的
  "其余 stack 上并发 head"的乘子。**两者重复。**

**精确倍数（LLAMA-7B, TP=8, num_hbm=5）：**

```
现状     : base × 5 (postprocess) × 4 (num_hbm_used) × 8 (num_attacc) = 160×
应该     : base ×                  4                × 8              =  32×
虚高     : 恰好 num_hbm = 5×        （--num-hbm 16 时就是 16×）
```

**不对称性：** A3/A4 的 prefill 在 GPU 上，A1 的 prefill 走
`_append_physical_no_reuse_prefill_layer`（聚合 API，`numOp = kv_heads_local = 4`
⇒ `ceil(4/5)×5 = 5` vs 真实 4，只虚高 **1.25×**）。
**decode 侧则是 A1 和 A3–A6 都走 placement 路径，都虚高 5×。**
所以受影响最不均衡的是 **A5/A6 的 prefill-on-PIM**（5×）
vs **A1 的 prefill-on-PIM**（1.25×）—— 同样的物理工作，A5/A6 被多收 **4×**。

**实测确认（CACHEBLEND-TINY + wl_tiny, `--engine dag --ablation A5`）：**

```
energy_breakdown_nj.by_class:
  PIM   1.933e9 nJ   62.9%     (其中 pim_kv_scan = 1.770e9 = 57.6%)
  GPU   1.138e9 nJ   37.0%
  LINK  8.70e5 nJ     0.03%
total = 3.072 J,  makespan = 0.1309 s,  GPU busy 0.1273 s (97.2%)
```

扣掉重复的 5×：

| | 现状 | 修正后 |
|---|---|---|
| 总能耗 | 3.072 J | **1.526 J**（2.01× 低） |
| PIM 占比 | 62.9% | **25.3%** |
| GPU 占比 | 37.0% | **74.7%** |

**⇒ "能耗主要花在 PIM 扫描上"这个结论会直接翻转成"主要花在 GPU 上"。**

**⚠️ 新发现（本轮）：解析 A1 模型 `src/a1_dag_free.py` 忠实复制了同一个 bug**
（`:204` `traffic = [value * num_hbm ...]` + `:212` `× energy_replicas=hbm_used`）。
它是照着 DAG 引擎写的、并以"能耗完全一致"为验收标准，所以**修 bug 必须两边同时改**，
否则 `docs/` 里那份"energy exact"的一致性结论会失效。

### Q4-(2) 🔴 完全没有静态 / 背景功耗

`workload_runner.py:4010`：

```python
"energy_nj": sum(event.energy_nj for event in scheduled),
```

全是 per-event 的**动态**能量。全仓库 grep 不到 idle / static / background / leakage / TDP。

量级：
- 8×A100 idle ≈ 500 W；
- 40 个 HBM stack（8 AttAcc × 5 stack）的 background + refresh ≈ 150 W；
- 实测 `dag_relay_LLAMA-7B` 报 331 J / 3.13 s = 平均 **106 W**；
- 上面那个 A5 小验证：3.072 J / 0.1309 s = **23.5 W**（修正后 11.7 W）。

**⇒ 静态项比动态项还大 1–2 个数量级。**

这项缺失**系统性偏袒 A3–A6**：它们的收益正是"时间短"，而静态功耗恰好按时间收费。
**跨档比能耗前必须补上**，最低限度是在 report 里加一个显式项：

```python
"static_energy_nj": P_idle_w * makespan_s * 1e9
```

### Q4-(3) 🟡 KV 写回能耗的两套口径互不一致

| 事件 | 文件:行 | 字节口径 | 时间口径 | 缺的乘子 |
|---|---|---|---|---|
| `dram_store_master` / `dram_store_diff_and_live`（prefill） | `:1591` | `2 × dhead × dbyte`（**一个 head**） | `peak_bw / num_hbm × ch/16`（**一个 head、一条 channel**） | `× kv_heads_local × num_attacc` |
| `decode_dram_store_master` | `:2415` | `2 × local_hidden × dbyte`（**全部本地 head**） | `peak_bw`（**全部 stack、16 条 channel**） | `× num_attacc` |

两者时间/能量各自内部自洽，但**彼此口径不同**，且都少乘 `num_attacc = 8`。
LLAMA-7B/TP8 下 prefill store 少算约 `4 × 8 = 32×`。

**实测量级：** `dram_store_diff_and_live = 1.98e4 nJ`、
`decode_dram_store_master = 2.6e3 nJ`，比 `pim_kv_scan`（1.77e9 nJ）低 **5 个数量级**。
补上 32× 仍可忽略。**按"口径不一致、优先级低"处理，不要和 (1)(2) 并列。**

（另：`decode_dram_store_master` 用 `local_hidden`（Q 的宽度）算 KV 字节，
GQA 模型下多算 `gqa_group` 倍。同样可忽略。）

### Q4-(4) 🟡 `energy_nj` 单位在两条路径上不一致

```python
_event            (:303-308)  → SplitEvent(..., sum(energy), ...)        # pJ
_cacheblend_event (:1358-1372)→ SplitEvent(..., sum(energy)/1000.0, ...) # nJ
warm reprice else 分支 (:4234)→ sum(energy)   # 跟着 _event 走，注释已承认"unit question tracked separately"
```

`_run_legacy_reuse_prefill`（`:372-502`）用 `_event`，它返回的 report **顶层没有
energy 字段**，所以目前不会暴雷。但只要有人 `--reuse cachetune/cachecraft
--decode-attn pim` 再去 sum events，就会拿到 **1000×** 的数。

### Q4-(5) 🟢 次要：`Layer.get_flops()` 不含 `pim_shared_queries`

`src/model.py:36-58` 的 MATMUL flops = `2·m·n·k·numOp`，
不含 `op.pim_shared_queries`。GQA / MQ 的 n 倍 MAC 在 `cal_energy` 里没算
（LLAMA3-8B 少 4×）。相对 `dram_energy` 占比小，且 LLAMA3-8B 不在论文矩阵里。

---

## Q5 `validate_cacheblend_events()` 与 `_schedule_*` 的依赖 / 串并行

### Q5.1 事件与依赖全表

#### A. prefill —— 三条分支（`_run_cacheblend_prefill`, `:3357-3901`）

**分支 0：`physical_no_reuse`（A1，`--reuse no-reuse`）** —— `_append_physical_no_reuse_prefill_layer`, `:2865`

```
qkv (GPU)                    ← request_ready
 ├─ q_gpu_to_pim (LINK)      ← qkv
 │   └─ contiguous_address_plan (ADDR, 0 s)   ← q_link
 │       └─ pim_kv_scan_score_softmax_pv (PIM, 聚合API, ceil(rows/cap) 次 sweep)
 │           └─ ctx_pim_to_gpu (LINK)
 │               └─ post-attention GPU 链 (proj/FF/…)   =: post_last
 └─ kv_gpu_to_pim (LINK)     ← qkv
     └─ dram_store_master (PIM)                          =: store
返回 (post_last, store)
```

> **新发现 N3**：A1 的 preset 写着 `prefill_attn: "gpu"`，但 DAG 引擎里
> `physical_no_reuse = (plan.config.policy == "no-reuse")`（`:4068`），
> 走的是**上面这条 PIM 扫描**分支。这符合 AttAcc 基线的语义（attention 在 PIM 上），
> 但和 preset 字段字面矛盾，而且 `side_rows["gpu"] += request.total_length`（`:3470`）
> 把它记成了 GPU 侧行数。**报告里的 prefill side 统计对 A1 是错的。**

**分支 1：`not reusable`（这一层没有任何可复用行）** —— `:3527-3576`

```
qkv (GPU) ← request_ready
 ├─ kv_gpu_to_pim (LINK) ← qkv
 │    └─ dram_store_master (per-channel PIM:poolX-Y)
 └─ gpu_score → gpu_softmax → gpu_context (GPU 链) → post-attention → post_last
```

**分支 2a：`prefill_side == "pim"`（A5，或 A6 判定为 pim）** —— `:3714-3818`

```
（可选）di_bitmap_gpu_to_die (LINK) ← request_ready
        └─ die_load_di_bitmap (DIE)          ← 并入 request_ready
qkv (GPU) ← request_ready
 ├─ q_gpu_to_pim (LINK)  ← qkv
 ├─ kv_gpu_to_pim (LINK) ← qkv
 │    └─ dram_store_diff_and_live (per-channel PIM)      =: store
 └─ 每个 sweep（cap = min(batch, mq_query_capacity/gqa)）:
      每个 query:
        (可选) q rotate 分发事件 ← q_link
        └─ die_query_position_transform (DIE) ← (rotate_ready, *store)
      tlb_lookup_and_bank_plan (TLB, 5ns×run) ← 全部 die_q
      └─ pim_kv_scan_score_softmax_pv (每条活跃 channel 一个 PIM:poolC-C 事件) ← tlb
          └─ die_score_assembly (DIE, 每 query 一个) ← 全部 scan lane
      → ctx_pim_to_gpu (LINK) ← 全部 die_score_assembly
         └─ post-attention GPU → post_last
```

**分支 2b：`prefill_side == "gpu"`（A3/A4，或 A6 判定为 gpu）** —— `:3819-3874`

```
qkv (GPU) ← request_ready
 ├─ q_gpu_to_pim (LINK) ← qkv          （即使不用 PIM 也发，见下）
 ├─ kv_gpu_to_pim (LINK) ← qkv
 │    └─ dram_store_diff_and_live (per-channel PIM)
 └─ dram_read_resident (per-channel PIM) ← request_ready
     └─ kv_pim_to_gpu (LINK) ← (qkv, *dram_reads)
         └─ gpu_prefill_score → gpu_prefill_softmax → gpu_prefill_context (GPU)
             → post-attention → post_last
```

#### B. decode —— 两种模式

**B1：`batch_size == 1`（默认）** —— `_append_cacheblend_decode`, `:2260-2426`

循环嵌套：`for output_row: for layer_index:`（**逐 request 完整跑完**）

```
decode_qkv (GPU) ← layer_deps
 ├─ decode_q_gpu_to_pim (LINK) ← qkv
 ├─ decode_kv_gpu_to_pim (LINK) ← qkv
 │    └─ decode_dram_store_master (PIM)                       =: store
 ├─ decode_gpu_local_score → _softmax → _context (GPU 链) ← qkv   =: local_last
 │    └─ decode_gpu_partial_lse_to_pim (LINK) ← local_last     =: tuple_link
 └─ (若 old 非空)
      q rotate 分发 ← q_link
      └─ decode_die_query_position_transform (DIE)
          └─ decode_tlb_lookup_and_bank_plan (TLB) [no-reuse 时是 decode_contiguous_address_plan/ADDR]
              └─ decode_pim_kv_scan_score_softmax_pv (每条活跃 channel 一个 PIM:poolC-C)
                  └─ decode_die_lse_merge (DIE) ← (*scan, tuple_link)
                      └─ decode_ctx_pim_to_gpu (LINK)
                          └─ post-attention GPU → post_last
layer_deps(下一层) = (post_last, store)
```

**B2：`batch_size > 1`** —— `_append_cacheblend_decode_batched`, `:2428-2864`

循环嵌套：`for output_row: for layer_index: for request in active`
⇒ **同 tier 的 agent 在 decode 阶段真正交错**。
额外机制：Stage A 先发完所有 Q/KV link；然后用
`_schedule_cacheblend_incremental` 做一次**试排**，按 Q link 的实际完成时间
（`_q_key`）重排 `ready`，再按 `batch_size` 分组做 PIM 批。
这是"全局 Q ready queue"admission（`batch_records[*].admission = "global-q-ready-queue"`）。

### Q5.2 `validate_cacheblend_events()` 检查了什么 / 漏了什么

**检查了（`:1839-1970`）：**

| # | 检查 | 行 |
|---|---|---|
| 1 | event id 唯一 | 1849 |
| 2 | `rows > 0`、`time_s ≥ 0`、`energy_nj ≥ 0` | 1853 |
| 3 | `0 ≤ masked_rows ≤ rows` | 1855 |
| 4 | LINK 事件必须有 `link_bytes`；非 LINK 事件不许有 | 1857-1862 |
| 5 | 依赖必须是**已出现过的**事件（拓扑序 = list 序） | 1863-1867 |
| 6 | `parent.tier ≤ event.tier` | 1868 |
| 7 | Q/ctx link 字节 = `rows × local_hidden × dbyte` | 1875 |
| 8 | KV link 字节 = `rows × 2 × local_hidden × dbyte` | 1880 |
| 9 | LSE tuple link：`rows == 1` 且字节 = `heads × (dhead+2) × dbyte` | 1884 |
| 10 | **每个 PIM scan 必须依赖一个 address plan**（TLB / contiguous） | 1888-1896 |
| 11 | DIE merge 必须同时依赖 ≥1 个 scan 和 ≥1 个 partial-LSE link | 1897 |
| 12 | `die_score_assembly` 必须依赖 ≥1 个 scan | 1904 |
| 13 | context 回传必须等齐同 query position 的全部 merge（或 tuple、或 scan） | 1911-1940 |
| 14 | KV link 必须排在同 position 的 QKV 之后 | 1941-1955 |
| 15 | KV store 必须排在它的 KV link 之后 | 1955 |
| 16 | 每个 request 的 `decode_dram_store_master` 数 == `lout × ndec` | 1958-1969 |

**漏了（全部是跨 request 的数据依赖）：**

| # | 缺失的检查 | 后果 |
|---|---|---|
| M1 | 不要求 child 依赖它 parent 的**输出 KV 写事件** | 目前被 tier barrier 掩盖 |
| M2 | **不要求 consumer 依赖 owner 的 master 写事件** | 见 Q5.3，**已经出问题** |
| M3 | 不检查 `PIM:poolX-Y` 设备名与事件里 `dram_addresses` 的 channel 是否一致 | placement 路径下两者本来就不一致（Q3.5） |
| M4 | 不检查 `rows` 与 `time_s` 的口径一致性 | Q3.5 |

M2 的严重性由复用的**同 tier 比例**决定，实测：

```
workload      reusable决策  同tier owner   同tier复用行占比
relay             8            0 (0.0%)         0.0%
sharegpt         43            0 (0.0%)         0.0%
mooncakemt       42            7 (16.7%)       20.5%
mooncake        167          167 (100%)       100.0%     <- legacy RAG, 全在 tier0
multihop         48           48 (100%)       100.0%     <- 同上
wl_reduce       528          495 (93.8%)       93.8%
alltoall N16   1247          719 (57.7%)       57.7%
```

`_cacheblend_tlb_rows`（`:2096-2140`）明确把复用行 bind 到
`decision.owner_request_id` 的物理块上，即 consumer **真的**去读 owner 写的那些行。
而 owner 的写事件在 `PIM:pool0-14`、consumer 的 scan 在 `PIM:pool{c}-{c}`，
**是两个不同的 resource，既没有边也没有资源序。**

### Q5.3 🔴 **新发现 N2：owner 的写事件排在 consumer 的读之后**

第一轮说"今天不出错，纯粹因为 append 序把 agent 串死了"。**本轮实测发现：
在 mooncake 和 multihop 上，这个掩盖已经失效。**

根因：两个顺序不一致。
- `build_reuse_plan` 按 `sorted(requests, key=(tier, request_id))` 决定 owner
  （`workload.py:456`），`request_id` 是**字符串**；
- DAG 按 `workload.tiers` 的**文件顺序**追加事件（`_tier_shapes`, `:86`）。

legacy RAG 的 request_id 是 `"0"`,`"1"`,…,`"39"`，
字典序是 `"0","1","10","11",…,"19","2","20",…` ≠ 文件序。

实测"owner 被追加在 consumer 之后"的决策数：

```
relay        0/8
sharegpt     0/43
mooncakemt   0/42
mooncake    34/167  (20.4%)   例: request "4" 复用 owner "13" 的 KV
multihop     7/48   (14.6%)   例: request "4" 复用 owner "31"/"17"
wl_reduce    0/528
alltoall     0/1247
```

**含义：** request `4` 在模拟时间上比 request `13` 先完成 prefill，
却直接扫了 `13` 还没写进去的 master 行，**并且省掉了那部分 prefill**。
这是"免费的复用"。校验器不报错，因为跨 request 依赖根本不在检查项里（M2）。

**影响范围：** 论文矩阵里的 `mooncake` 和 `multihop` 两个 workload
（占 5 个 workload 中的 2 个），A3–A6 全部 rung。
`relay` / `sharegpt` / `mooncakemt` 不受影响。

**最小修法：** 让 `build_reuse_plan` 的 owner 选择顺序与 DAG 的追加顺序
一致（都用 `workload.tiers` 的文件序），并在 `validate_cacheblend_events`
里加上 M2 的边检查。

### Q5.4 `_schedule_cacheblend` 的串并行

```python
resource = event.device if pipe else "SERIAL"
start = max([availability.get(resource, 0.0)] + [finish[dep] for dep in deps])
availability[resource] = start + event.time_s
```

- **`pipe=False`**：所有事件共享一条 `"SERIAL"` 时间线（AttAcc 的保守约定），
  唯一的例外是 placement scan 的多条 channel lane（`:1626-1653`）——
  同一 scan 的连续 `PIM:poolX-X` 事件被识别成一组，**同时开始**，
  该组对 SERIAL 只收 `max`（不是 `sum`）。
- **`pipe=True`（论文矩阵用的 `--pipeopt`）**：每个 `device` 字符串是一条独立时间线。
  device 取值：`GPU`、`LINK`、`DIE`、`TLB`、`ADDR`、`PIM`、`PIM:pool{a}-{b}`。
  **⇒ 每条 channel 是一条独立时间线，16 条 channel 天然并行；
  master pool（`PIM:pool0-14`）与每条 channel lane 也是不同的时间线。**

**关键性质：这是一个 list scheduler with fixed priority = append order，
非抢占、不可重排。** `availability[resource]` 是一个标量，
所以同一 device 上的执行次序**恒等于 append 次序**。

后果：**同 tier 内不同 agent 目前是完全串行的，而且这是调度器的产物，
不是依赖决定的。** 实测 `dag_relay_LLAMA-7B_dynamic.json`：

```
tier1  t1w0 end=1.2544   t1w1 end=1.8789   t1w2 end=2.5034   t1w3 end=3.1279
       first_token:0.641 /  1.266         /  1.890         /  2.515
```

每个 worker 恰好 0.6243 s，**零重叠**：w1 的第一个 token 在 w0 全部 decode 完之后
才出来。原因是 append 顺序是"request A 的全部 32 层 prefill + 全部 decode →
request B …"，于是 B 的第 0 层 GPU 算子排在 A 的第 31 层 GPU 算子后面。

**对 makespan 的影响很小**（实测 GPU 占用 96%–99.6%）：

| run | makespan | GPU unoverlapped | GPU 占用 | PIM pool |
|---|---|---|---|---|
| dag_relay_LLAMA-7B | 3.128 s | 3.104 s | 99.2% | 0.024 s (0.8%) |
| dag_mooncake_LLAMA-7B | 217.5 s | 211.2 s | 97.1% | 6.17 s (2.8%) |
| dag_mooncakemt_LLAMA-7B | 97.66 s | 95.74 s | 98.0% | 1.93 s (2.0%) |
| dag_multihop_LLAMA-7B | 21.77 s | 21.68 s | 99.6% | 0.018 s (0.08%) |

**但对每-request 延迟、TTFT、tier 内并发度的任何结论都是无效的。**

**顺带（第一轮结论，本轮复核确认）：** `--cacheblend-batch-size` 默认 1
（`main.py:252-255`），ablation preset 不覆盖它，
`_append_cacheblend_decode_batched` 走不到。加上 LLAMA-7B/65B/GPT-175B 的
`gqa_size = 1`（`config.py:322-325`），单请求 decode 的 `pim_shared_queries = 1`
⇒ `mq_command = False`（`ramulator_wrapper.py:593`）。
**A5/A6 的 decode 侧 MQ 在论文矩阵的模型上一条命令都没发出来**
（prefill 侧的 MQ 是有的，`:3781`）。
`experiments/channel_parallel_validation` 用了 `--cacheblend-batch-size 8`，
那批 run 不受影响。

---

## 跨切面新发现

### N1 🔴 `experiments/paper_ladder/results/ladder_*.json` 里没有 placement 信息

`run_matrix.py:107-118` 构造的 ladder 任务**没有传 `--engine dag`**，
而 `--engine` 默认 `"analytic"`（`main.py:353`）。
`src/ablation.py:615-623` 的 docstring 自己写着：

> NOTE (chenyi9 2026-08-29): the head-aware channel-placement ladder
> (`config.channel_placement` single/slice/table -> A3/A3b, A4/A4b) is
> modelled only by the PHYSICAL EVENT ENGINE …
> This analytic profile … does NOT distinguish single vs slice vs table.
> **Do not read A3/A3b/A4/A4b differences off the analytic engine.**

所以 `ladder_relay_GPT-175B_A3/A4/A5/A6` 的
8.479 / 8.474 / 8.471 / 8.470 s 这些差（0.1%）**不含任何 head-placement 信息**。
（解析引擎确实区分 A3 的 naive 池 vs A4 的 master+diff 池，但不区分
single/slice/table，也**完全没有 head 维度**。）

顺带：解析引擎的 diff 池用 `_runs_from_lengths([diff_rows], …)`（`ablation.py:574`），
**用的是真实 `diff_rows`，没有 256 量化，也没有 ×h**。
⇒ **两个引擎在 Q2/Q3 问的这件事上结构性地互相矛盾**：

| | diff 行数 | head 维度 |
|---|---|---|
| analytic (`ablation.py`) | 真实行数 | **无** |
| DAG (`workload_runner.py`) | 256 量化 | ×`heads_per_hbm` |

### N3 A1 的 prefill side 统计是错的

见 Q5.1 分支 0。

### N4 / N5 字节布局与计时策略互相矛盾

见 Q2.1。

### N6 `_append_physical_pim_scan` 是死代码

见 Q3.4(e)。

---

## 建议的处理顺序

| 优先级 | 条目 | 改动量 | 影响 |
|---|---|---|---|
| **P0** | Q4-(1) `num_hbm` 双乘（`ramulator_wrapper.py:789` 或 `workload_runner.py:1549`，**外加 `a1_dag_free.py:204/212` 同步改**） | 1–2 行 ×2 处 | 所有 A1/A3–A6 的 PIM 能耗 ÷ num_hbm；结论方向会翻转 |
| **P0** | N2 owner/consumer 顺序反转（`workload.py:456` 的排序键） | 1 行 | mooncake / multihop 上 15–20% 的复用决策不再"免费" |
| **P1** | Q4-(2) 静态功耗：report 里加 `static_energy_nj` 显式项 | ~10 行 | 跨档能耗比较才成立 |
| **P1** | Q5-M2 owner→consumer 边（`_cacheblend_tlb_rows` 记 owner，scan deps 加 owner 的 `dram_store_master`） | ~20 行 | 是"以后能做 tier 内并发"的前提 |
| **P1** | N1 ladder 结果的引擎归属：要么补跑 `--engine dag`，要么在文档里明确 ladder 数不含 placement | 文档 or 补跑 | 直接影响论文里 A3→A4→A5→A6 的叙事 |
| **P2** | Q3 diff 量化：diff lane 用真实 `diff_rows`（不做 chunk 取整）；`heads_per_hbm > 1` 时 diff pool 宽度跟着走 | ~15 行 | relay/sharegpt + 65B/175B 上 A4 的 diff 关键路径消失 |
| **P2** | Q3.4 断流 ACT：`_TLB_DESCRIPTOR_S` 换成有出处的 tRC/tRCD，或不覆盖 `pim_kv_runs` | 1 行 / ~30 行 | A3 vs A3a 的差才有物理含义 |
| **P2** | Q1 producer output 指纹注册（`workload.py` + `_prepare_cacheblend_tlb`） | ~15 行 | reduce/supervisor/alltoall 三类拓扑的复用率显著变化 |
| **P3** | Q1.3 parent 边调度（替换 tier barrier） | ~40 行 | 必须在 P1 的 owner→consumer 边之后做 |
| **P3** | Q4-(3)(4)(5) 口径统一；删 `_append_physical_pim_scan` 死代码 | 小 | 清理，无数值影响 |
| —— | Q6 | 无需处理 | 现有做法精确 |

---

## 附：本次 review 复现用的命令

```bash
# Q2.2/Q2.3 的 diff-vs-master 负载表
python3 -c "
import sys; sys.path.insert(0,'.')
from src.workload_runner import _layout_channel_loads as L
for h in (1,2,3,4,8):
    print(h, [(max(L('master-diff-slice',cm,1,h)[:15]), L('master-diff-slice',cm,1,h)[15])
              for cm in (1,2,3,5,8,15,26,34,60)])
"

# Q3.2 真实 diff 行数
python3 -c "
import sys; sys.path.insert(0,'.')
from src.workload import load_workload, build_reuse_plan
from src import workload_runner as wr
w=load_workload('experiments/paper_ladder/workloads/workload_relay_s400w4t1.json')
p=build_reuse_plan(w,'epic',epic_prefix_recompute_tokens=8)
for r in w.requests: print(r.request_id, len(wr._policy_corrected_rows(p,0,r)))
"

# N2 owner/consumer 顺序反转
python3 -c "
import sys; sys.path.insert(0,'.')
from src.workload import load_workload, build_reuse_plan
w=load_workload('experiments/paper_ladder/workloads/workload_mooncake_toolagent_n40_o0.json')
p=build_reuse_plan(w,'epic',epic_prefix_recompute_tokens=8)
o={}; i=0
for t,rs in w.tiers.items():
    for r in rs: o[r.request_id]=i; i+=1
print(sum(1 for d in p.reusable if o[d.owner_request_id]>o[d.request_id]), '/', len(p.reusable))
"
```
