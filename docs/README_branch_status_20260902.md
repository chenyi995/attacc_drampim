# 分支现状（2026-09-02）：哪个分支该出数，哪个还在验证

> 本文只记录状态与判据；未执行 merge / rebase / cherry-pick / reset。
> 与 `README_branch_audit_20260902.md` 的关系：那份是**提交前**的工作树盘点，
> 结论是「未提交改动不属于任何 branch，因此现在不该称任一分支为最新版」。
> 那批改动此后已经落到两个 branch 上，本文取代它的分支结论部分。

## 结论

| 分支 | 角色 | 现在可以拿来做什么 |
|---|---|---|
| **`xinyao_0902`** | **当前正确基线** | **出论文数、跑 sweep、被别人 fork** |
| `xinyao_0902_Analysis_Test` | 解析模型（analytic model）的验证分支 | 只做方法验证；**不要用它出论文数** |

`xinyao_0902_Analysis_Test` 是 `xinyao_0902` 的**超集**（从 `083218e` 分出，
只多一个 commit `5227883`），不是竞争实现。它引入的两个解析替代已经有相当强的
逐层证据，但**闭环还没有合上**——具体缺什么见第 3 节。在那之前，两个分支各司其职。

---

## 1. `xinyao_0902` —— 正确基线

已推送，HEAD `083218e`。相对 `chenyi-822-cppcore-exp`（`61f32dd`）新增：

- `75da860` 同一逻辑 PIM 扫描的 channel lanes 在 `pipe=False` 下并行
  （`src/workload_runner.py`、`src/cpp_eventcore.py`、`src/cppcore/eventcore.cpp`；
  `ec_add` 增加了 `pool_scan` 参数，gitignore 掉的 `.so` **需要重编**）
- `ce0cde6`、`872150f` `experiments/channel_parallel_validation/`
- `083218e` 让退化的 A3b 不可能被无声地跑掉或记录下来

这条线上的每个数都来自 event DAG + Ramulator 子进程，即**没有任何拟合参数参与
定价**。这是它可以直接出数的唯一理由，也是它必须保持这样的理由。

## 2. `xinyao_0902_Analysis_Test` —— 引入了解析模型

一个 commit `5227883`，77 个文件。两个**互相独立**的替代：

**`src/analytic_pim.py`** 替掉 Ramulator 子进程。三层，各有各的真值：

| 层 | 内容 | 拟合参数 | 对什么校验 | 结果 |
|---|---|---|---|---|
| L1 | 命令计数 | **0** | Ramulator 自己的计数器 | 12,945 / 12,945 精确 |
| L2 | barrier 组数、DRAM row openings | **0** | **重新生成的真 trace** | 120 / 120 精确 |
| L3 | cycle | 5–7（MAC 系数钉在 datasheet `nCCDAB`） | 留出交叉验证 | 见下 |

**`src/a1_dag_free.py`** 替掉 event DAG 作为 A1 的 PIM 调用来源。decode 枚举是
对 256 行 chunk band 的闭式走查，`wl_N64`/GPT-175B 从 352 s 降到 0.078 s，
与被它取代的循环在 112/112 组（14 workload × 4 model × 2 num_hbm）**逐位相同**。

实测代价（冷缓存，`experiments/analytic_a1_0902/matrix2x2/matrix_2x2.csv`）：

| | `wl_pipeline_D4` | `rag_shared` |
|---|---|---|
| PIM scan 能量误差 | 0.0000%（枚举器侧 −0.061%，见 3.1） | **0.0000%** 四格全同 |
| PIM latency 误差 | +2.49% ~ +3.68% | +4.02% |
| wall clock | 1360.7 s → **4.7 ms** | 1319.7 s → **16.7 ms** |

---

## 3. 闭环还差什么

按「会不会影响论文数」排序。

### 3.1 阻断级：DAG 侧的 allocator 缺陷未裁决

枚举器对 DAG 的 10 组比对里 7 组精确吻合、3 组不吻合，**三组都归到同一个 DAG
缺陷**，并且重放 allocator 逐条复现：

`NoReuseKVLayout.finalize`（`src/workload_runner.py`）是全局 bump allocator，
游标换 8 MiB tile 时若正好落在某请求的 `::history` 与 `::no-reuse-input` 之间，
两段不再地址相邻，`CacheBlendTLB.scan_runs` 就拆成两个 run 而不是一个。

| 用例 | 层数 | 非相邻层 | 预测多出 run | 实测 |
|---|---:|---:|---:|---:|
| `wl_pipeline_D4` / LLAMA-7B | 32 | 10+11+11 | 33,856 | 33,856 |
| `workload_llama7b_small` / LLAMA-65B | 80 | 1 | 16 | 16 |
| `workload_llama7b_small` / GPT-175B | 96 | 1 | 16 | 16 |

**这意味着 A1 的模拟 baseline 成本是 allocator 游标和层数的函数**——同一个算法
扫描因为无关的分配历史被定成两种价。`workload_rag_shared_p24_s8` 没有 history，
缺陷咬不到，枚举器就 245,504/245,504 精确吻合，这反证了枚举器本身是对的。

需要的裁决：run 的拆分应当跟随**逻辑 extent 结构**（history 与 prefill 恒分或恒合），
还是保持按字节相邻？改了之后**十组 ground truth 全部要重录**，七组 MATCH 也会变。
在裁决之前，`Analysis_Test` 上任何带 history 的 workload 的枚举器结果都带着一个
已知错误的 placement 模型。

### 3.2 阻断级：枚举器没有调度器

`a1_dag_free.py` 产出的是 `pim_cycles_unordered`——**未排程的工作量**，不是
makespan。所以 2×2 表里比较的是「所有 PIM run 时长之和」，`makespan` 被显式排除。
在补上一个宏观调度器之前，**解析引擎无法替代 DAG 出端到端延迟**，只能替代
「PIM 侧总工作量与能量」。

### 3.3 需要注意：latency 偏差方向不利

解析定价器在 A1 工作点上系统性**高估** PIM 时间 +2.5% ~ +4.0%。A1 是 baseline，
高估 baseline 会让 Fugue 的 speedup 偏大——**方向对论文结论不利**。
引用这些数时必须一起说这句话。

误差来源已经定位并量化：65 个不同模型输入里只有 5 个是工作点那种超长 prefill
sweep（拟合对所有输入等权），以及 62 个输入对应多个不同 Ramulator 结果
（p90 展宽 5.95%）因为特征里带行内字节偏移、不带 bank / bank-group。

### 3.4 已答但需复核：误差随规模的走向

`experiments/analytic_a1_0902/error_scaling/README.md`。结论是**不发散**：
更多 run → 聚合收敛（+2.8% 在千条以后不动，一直到一百万条）；更长上下文 → 压平成
渐近线（Ramulator 实测到 262,144 行，标定边界的 6.7 倍，decode 停在 −5.3%、
prefill −2.2%）。但**误差会变号**：短 run 偏高、超长上下文偏低。
「模型大约高 4%」这句话在 A1 当前工作点对，在 10 万级上下文上错。

### 3.5 覆盖面：只标定了 A1 会走的那一小块

| 维度 | 已标定 | 未标定（`estimate()` 会 **raise**，不会猜） |
|---|---|---|
| `pim_type` | BA | BG、BUFFER |
| regime | `chunkstripe1\|{replicate,mq}`、`legacy\|replicate` | `legacy\|mq` 等 |
| ablation | **只有 A1** | A2–A6 从未用解析模型跑过 |
| `dbyte` / `dhead` | 2 / 128 | 其他 |

留出交叉验证（按**模型输入**切分，8 个种子）：

| regime | 不同输入 | 有效参数 | 留出输入 | 留出配置 |
|---|---:|---:|---|---|
| `chunkstripe1\|replicate`（A1 用这个） | 63 | 6 | **2.98% ± 0.61** | 5.86% ± 3.26 |
| `chunkstripe1\|mq` | 66 | 8 | 5.63% ± 0.57 | 6.84% ± 1.64 |
| `legacy\|replicate` | 49 | 7 | 6.81% ± **9.19** | 3.76% ± 2.75 |

`legacy|replicate` 的 ±9.19 是单个坏切分拉出来的，49 个输入太少，**这一格的数
不该单独引用**。

### 3.6 尚未做的闭环实验

1. **全 sweep 复现**：用解析引擎重跑一遍已入库的 A1 sweep，逐 cell 比对
   `output/analysis/` 里的表。这是真正的闭环，目前只做了 2 个 workload。
2. **A2–A6**：`a1_dag_free.py` 的枚举规则是 A1 专用的，硬编码了
   `gemv_buffer_bytes`、batch size、`pim_batch_command`；换档位会**静默给出错的
   多重集**，而不是报错。要么把这几个参数接出来，要么让它在不匹配时 raise。
3. **能量口径的一处疑似缺陷**：`_append_placement_pim_scan` 对 decode 能量重复
   乘了一次 `num_hbm`（`Ramulator.run` 的 `postprocess` 已经乘过）。枚举器为了
   对齐**照抄了这个乘法**。需要裁决哪边是对的。
4. **`heads_per_hbm > 1` 的 prefill 半边**：decode 半边已验证精确，prefill 半边
   被 3.1 的 allocator 缺陷挡住，裁决后要重验。

## 4. 测试现状

`python3 -m unittest tests.test_analytic_pim tests.test_a1_dag_free` → 19/19 通过。

两个**既有**失败，与本次工作无关，但会让 `discover` 变红：

- `tests/test_qbatch_pim.py`（未跟踪，故意没入库）：按旧的
  `gen_trace_attacc_bank.run_attention(q_batch=…)` API 写的，在 HEAD 上 9 个错 8 个；
- `test_cacheblend_emits_trace_ordered_tlb_and_physical_addresses`：
  `xinyao_0902` 的 channel-parallel 调度改了 `overlap_validation.contract` 字符串，
  测试里的期望值没跟着改。

## 5. 合并判据

`Analysis_Test` 可以并回 `xinyao_0902` 的条件，按依赖顺序：

1. 3.1 的 allocator 语义裁决，并重录全部 ground truth；
2. 3.6.1 的全 sweep 复现通过，或明确把解析引擎的适用范围限制成
   「PIM 工作量与能量的快速预估，不出端到端延迟」；
3. 3.3 的 +2.5~4% 偏差要么修掉，要么写进每一处引用它的地方。

在此之前，两个分支并存是**正确**的状态，不是待办事项。
`xinyao_0902` 出数，`Analysis_Test` 验证方法。
