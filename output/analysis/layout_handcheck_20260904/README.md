# 布局手算校验：LLAMA3-8B × baseline 七档（2026-09-04）

目的：**把每一块 KV chunk 的真实地址、以及每一档的扫描布局，都摊到能用纸笔重算的程度。**
所有数字都能从本页给出的规则和参数自己算一遍；算不出来的地方，本页明说是实测还是推导。

术语（首次出现即解释）：
- **档（rung）** = 消融阶梯（ablation ladder）上的一级，A1…A6。
- **k** = 每个复用块重算的 token 数（`--epic-prefix-recompute-tokens`），本次 k=8。
- **channel（通道）** = HBM 堆栈里的一条独立通道，本仿真固定 16 条，编号 0–15。
- **ACT（activation）** = 打开一个 DRAM row 的命令；一条扫描的代价主要由 ACT 次数决定。
- **extent** = 一段连续的物理 KV，是交给 Ramulator 的最小地址单位。
- **stripe unit** = 256 个 token，正好一个 DRAM row（见 §1）。

---

## 0. 跑了什么，怎么复现

| 项 | 值 |
|---|---|
| 模型 | **LLAMA3-8B**（32 层，32 个 Q head，GQA group 4 → **8 个 KV head**）|
| 配置 | `--ngpu 1 --num-hbm 1`，所以 `tp=1`、一个 HBM 堆栈、`heads_per_hbm = ceil(8/1) = 8` |
| workload | `workload/sweep/wl_baseline_alltoall_N16_C16_D2.json`（32 agents，C=16 复用块，块长 256 token）|
| k | 8 |
| 档 | A1 A2 A3b A4 A4b A5 A6 |
| 引擎 | `--engine dag`，无 `--pipeopt` |
| 资源 | 7 档并行 × 8 个 Ramulator worker + 7 个构建进程 = **63 核**（上限 64）；峰值内存约 90 GB（上限 500 GB）|

```bash
bash experiments/run_layout_handcheck.sh
# 出报告
python3 output/analysis/layout_handcheck_report.py output/layout_handcheck_20260904/LLAMA3-8B
# 手算 vs 引擎，逐扫描对账
python3 output/analysis/layout_handcheck_theory.py output/layout_handcheck_20260904/LLAMA3-8B
```

**为什么选 LLAMA3-8B。** 它是六个 baseline 模型里唯一**既没有张量并行、又只有一个堆栈**的，
不用绕 `num_hbm // tp` 那一层；8 个 KV head 整除 16 条通道，A3b 的 stripe = `16//8 = 2`，
不像 LLAMA-7B（32 个 KV head）那样 stripe 钳到 1、A3b 退化成 A3。

> ⚠️ **但 A4 在这个配置上是退化的。** A4 让出 ch15 给 diff 池，master 只剩 15 条，
> `stripe_m = 15 // 8 = 1`。引擎自己在 stderr 报了警
> （`placement_degeneracy_warning`，见 `dag_A4.log` 末尾）。这不是 bug，是这个配置的真实结果，
> 而且正是 2026-09-03 记录的「A4 在 `num_hbm=1` 的模型上是负收益」那条的机制 ——
> §3 的实测把它量化了。**但这一格不能用来论证 master/diff 分池本身。**

数据放哪（按仓库约定，原始产物**留本机、不入库**，见 `docs/RAW_DATA_MANIFEST.md`）：
`output/layout_handcheck_20260904/LLAMA3-8B/`
- `dag_<档>.json` / `.log` —— 常规运行报告
- `layout_<档>.jsonl` —— 布局探针（layout probe）的 dump，共约 30 MB，见 §1.3

**入库的**在本目录（`output/analysis/layout_handcheck_20260904/`）：本页 + `scans_excerpt.jsonl`
（本页引用到的那几条扫描记录的原文，几十 KB，可直接对照）。

---

## 1. 地址空间：**两把尺子**，别用错

### 1.1 Ramulator 侧：1 token = 4 B（**已验证**）

`PIM_MAC_AB` 是 all-bank 广播，一个地址同时点名 `n_pch × n_rank × n_bg = 2×2×4 = 16` 个分区。
`ramulator2/trace_gen/gen_trace_attacc_bank.py` 的 `score_mac` 里，每 16 个 token 地址前进
`ceil(dhead / n_bank / n_mac) = ceil(128/4/16) = 2` 个 column：

| 单位 | 字节 | token 数 | 来源 |
|---|---:|---:|---|
| 1 column | `HBM_GS['col'] = prefetch_size` = **32 B** | **8** | `gen_trace_attacc_bank.py:37` |
| 1 row | `HBM_GS['row'] = n_col × 32` = **1024 B** | **256** | `gen_trace_attacc_bank.py:38`，`n_col = 2^5` |
| 1 token | 64 B / 16 token = **4 B** | 1 | `score_mac` 的地址步长 |

和引擎侧的 `_GEN_ROW_BYTES = 1024` / `_GEN_BYTES_PER_TOKEN = 4` / `_STRIPE_UNIT_ROWS = 256`
（`src/workload_runner.py:751-752`、`:667`）**逐个对上**。
这一条此前标注为「未验证」，2026-09-04 核对 trace 生成器后**确认成立**。

**推论（后面处处要用）：ACT 只在跨 row 边界时增加。** 在一行内多占 column 只增加 RD/CAS 次数
（tCCD 量级），不增加 ACT（tRC 量级）。k 从 8 涨到 32 时，如果 diff 总量还在同一行里，ACT **不变**。

### 1.2 TLB 侧：1 token = 256 B —— **和上面不是一个刻度**

KV 存储（`CacheBlendTLB` / `NaiveKVLayout` / `NoReuseKVLayout`）按
`vector_stride = ceil(dhead × dbyte / 32) × 32 = 128 × 2 = 256 B` 每 token 分配地址。
探针 dump 里 `blocks[].key_base` 就是这个空间的地址。

**关键：这些地址不进 Ramulator。** `_channel_extent_addresses`（`src/workload_runner.py:774`）
按 `base = channel × 2^30`、`span = tokens × 4 B` **另行合成**一套地址，
`_append_placement_pim_scan` 再用 `scan_op.pim_kv_extent_groups = extent_groups` 覆盖掉
`tlb.scan_runs(reads)` 的真实地址。TLB 的地址只喂 `_tlb_plan_cost` 和 JSON 报告。

> **所以：`blocks[].key_base` 和 scan 记录里的 `key_address` 不是同一把尺子，不要直接比。
> 能比的是 channel 和相对顺序。**

### 1.3 探针（layout probe）

`src/layout_probe.py`，环境变量 `KVPIM_LAYOUT_DUMP` 指到文件才开启，不开启时零开销。
每行一个 JSON：

| `kind` | 内容 |
|---|---|
| `blocks` | 每一块**缓存 chunk** 的真实地址：`(layer, owner, fingerprint, kind)` → `key_base` / `value_base` / `channel_base` / token 行号 |
| `scan` | 一次 PIM 扫描：放置参数、**交给 Ramulator 的逐通道 extent**、逐通道的 rows/ACT/时间/能量，以及两个归约（reduction）的每一项 |

其它开关：`KVPIM_LAYOUT_DUMP_LAYER`（默认只留第 0 层）、`KVPIM_LAYOUT_DUMP_MAX`（默认 400 条）、
`KVPIM_LAYOUT_DUMP_REQUEST`（只留指定的 request id）。

---

## 2. 一个扫描的完整手算（A3b）

取 dump 里最大的一次 A3b 扫描：`request = batch:t0:o0:l0:g0`，**master = 4352 token，diff = 0**。

**输入**：`heads_per_hbm = 8`，policy = `slice-append`（A3b = `kv_mapping=naive` + `channel_placement=slice`）。

**手算**：

```
1) 切 unit：4352 / 256 = 17，整除 → units = [256] × 17
2) stripe = 16 // heads = 16 // 8 = 2
3) head h 的基址通道 base = (h × 2) % 16 → 0, 2, 4, 6, 8, 10, 12, 14
4) 第 u 个 unit 落在 (base + u % 2) % 16
   17 个 unit 里 u 为偶数的有 9 个 → 落 base
                u 为奇数的有 8 个 → 落 base+1
5) 于是 8 个 head 铺满 16 条通道：
     偶数通道（0,2,…,14）各 9 个 unit = 9 × 256 = 2304 token
     奇数通道（1,3,…,15）各 8 个 unit = 8 × 256 = 2048 token
6) ACT = 每个 unit 一次（unit 正好一个 row）= 17 × 8 = 136
```

**实测 dump**（`layout_A3b.jsonl`）：

```
loads  = [2304,2048,2304,2048,2304,2048,2304,2048,
          2304,2048,2304,2048,2304,2048,2304,2048]
units/channel = {0:9, 1:8, 2:9, 3:8, ..., 14:9, 15:8}
scan_acts = 136
```

**逐格相同。**

**归约（reduction）**：
- 时间 = **max** over 通道 = 25.220 us（在 9-unit 的通道上；8-unit 的是 22.578 us）
- 能量 = **sum** over 通道 × `num_hbm_used`（这里 = `ceil(8/8)` = 1）= 5 057 004 nJ

### 2.1 逐扫描对账（自动化）

`output/analysis/layout_handcheck_theory.py` 把上面的规则**独立重写了一遍**
（不 import 引擎的放置函数 —— 否则就是拿引擎验引擎），对每一条扫描比对逐通道的 slot 列表：

| 档 | 扫描数 | 相符 | 不符 | ACT 手算 | ACT 引擎 |
|---|---:|---:|---:|---:|---:|
| A3b | 400 | **400** | 0 | 22592 | 22592 |
| A4 | 400 | **400** | 0 | 22592 | 22592 |
| A4b | 400 | **400** | 0 | 22592 | 22592 |
| A5 | 400 | **400** | 0 | 57600 | 57600 |
| A6 | 400 | **400** | 0 | 22592 | 22592 |

> ⚠️ **这 2000 条扫描全部 `diff = 0`**（都是 tier-0 生产者，自己的 KV 没有复用、没有修正）。
> 所以规则里的 **repair 分支还没有被这批数据检验到** —— A3b「每 head 自己一段」
> 与 A4/A4b「所有 head 打包进 ch15」的差别，在这 2000 条里体现不出来。
> 探针的 400 条上限在 tier-0 就用完了。已给探针加 `KVPIM_LAYOUT_DUMP_REQUEST`，
> 第二遍指定一个 tier-0 生产者 + 一个 tier-1 消费者即可覆盖，**本页尚未跑**。

---

## 3. 七档对照：同一个扫描，布局差在哪

同一次扫描（master = 4352 token，8 个 head，17 个 unit），**总 ACT 一律是 136 = 17 × 8**。
变的是这 136 个 unit 怎么摊到通道上：

| 档 | policy | 活跃通道 | 每通道 unit 数 | 最忙通道 (token) | 扫描时间 | 扫描能量 (nJ) |
|---|---|---:|---|---:|---:|---:|
| **A3b** | `slice-append` | **16** | 偶 9 / 奇 8 | **2304** | **25.220 us** | 5 057 004 |
| **A4** | `master-diff-slice-append` | **8** | 各 17 | **4352** | **48.453 us** | 5 055 887 |
| **A4b** | `master-diff-table-append` | **15** | ch0 是 10，其余 9 | **2560** | **28.545 us** | 5 056 865 |
| **A6** | `master-diff-table-append` | **15** | ch0 是 10，其余 9 | **2560** | **6.806 us** | 664 589 |

读法，逐条都能手算：

**A3b → A4：慢了 1.92×。** A4 让出 ch15 给 diff 池，master 只剩 15 条，
`stripe_m = 15 // 8 = 1` —— stripe 钳到 1，一个 head 的 17 个 unit **全堆在一条通道上**，
而且只用了 8 条通道（head 0..7 的 base = 0..7），ch8–ch15 全闲。
最忙通道从 2304 涨到 4352（正好 ×17/9 ≈ 1.89），时间 25.220 → 48.453 us（1.92×）。
**这就是「A4 在 `num_hbm=1` 的模型上是负收益」的机制，第一次逐通道量出来。**

**A4 → A4b：救回 1.70×。** 全局放置表不再按 head 分片，而是用一个 slot 计数器
把 8×17 = 136 个 unit 依次丢到 `slot % 15`：136 = 15×9 + 1，所以 ch0 拿 10 个、其余 9 个。
最忙 2560，时间 28.545 us。

**A4b 仍比 A3b 慢 1.13%×3 ≈ 13%。** 不是放置更差，是**通道少了一条**：
A3b 用 16 条，A4b 只用 15 条（ch15 留给 diff 池，而这次扫描没有 diff 行，ch15 全闲）。
`ceil(136/15)=10` vs `ceil(136/16)=9`，2560 / 2304 = 1.111，与 28.545/25.220 = 1.132 同向。

**A4b → A6：快了 4.19×，布局完全没变。** 两者 `loads` 逐格相同、ACT 相同、活跃通道相同。
差的是 **batch command**：A1–A4b 用 `replicate`（每个 (column, query) 一条 MAC），
A5/A6 用 `mq`（一条 `MAC_AB` 服务所有驻留 query）+ `pim_pe_freq_ghz = 1.3004`。
能量同时降到 1/7.6。**旁证**：A5 的 prefill 扫描（4608 token，144 unit，`ceil(144/15)=10` → 最忙也是 2560）
时间**同样是 6.806 us** —— 同样的最忙载荷 + 同样的命令 = 同样的时间，不是巧合。

**A1**（`slice`，旧 chunk 计数模型）：master = 4608 token → `ceil(4608/256) = 18` 个 chunk，
stripe = 2，每 head 9+9，16 条通道各 9 个 chunk × 256 = **2304**，与 dump 的 `loads` 逐格相同。

**A2**：**没有 PIM 扫描**。实测 `dag_A2.json` 的 `energy_breakdown_nj.by_class`
只有 `GPU` 和 `LINK` 两项，没有 PIM。这一档没有布局可查。

---

## 4. 逐档差异：具体到函数、具体到 sum/max 的哪一项

先说两个归约在哪：

| 归约 | 位置 | 做法 |
|---|---|---|
| **latency（延迟）取 max** | `src/workload_runner.py:1983-2007`，`_schedule_cacheblend` | 一次扫描的每条活跃通道是一个独立事件 `PIM:pool{c}-{c}`；`pipe=False` 下这些 lane 被识别成一组，**共享同一个 start**，`availability["SERIAL"] = group_end = max(lane.end)`。所以扫描时间 = **最忙通道**，不是各通道之和（这是 `75da860` 修的） |
| **energy（能量）取 sum** | `src/workload_runner.py:4376`，`sum(event.energy_nj for event in scheduled)` | 每条 lane 事件的 `energy_nj` 在 `_cacheblend_event`（`:1650`）里 = `sum(energy_vec)/1000 × num_hbm_used`；报告把所有事件相加 |

逐档：

| 转换 | 变的是什么 | 哪个函数 | max 里哪一项变了 | sum 里哪一项变了 |
|---|---|---|---|---|
| **A1 → A2** | `decode_attn` pim → gpu | `run_cacheblend_dag:4419` 分派到 `_run_gpu_software_only:3475` | **整个 PIM lane 组消失**，没有 max 可取；时间轴变成 GPU + LINK 串行 | PIM 项整个消失；`kv_remote_to_gpu` 链路项暴涨（实测 A2 `link_bytes` = 7.31e12 B）|
| **A2 → A3b** | 回到 PIM 路径，`kv_mapping=naive` + `channel_placement=slice` | `_layout_policy:880` → `slice-append`；`_striped_append_channel_extents:838-851` | max 的**候选集**变成 16 条通道的 lane 时间 | sum 的**项数**变成活跃通道数（16） |
| **A3b → A4** | `kv_mapping` naive → master-diff | 同函数的 else 分支 `:852-870` | 通道池 16 → **15**，且 `stripe_m = 15//heads` 代替 `16//heads`：**最忙通道的 token 数变了**（本例 2304 → 4352） | 项数 16 → 8（本例）；每项的能量随其载荷变 |
| **A4 → A4b** | `channel_placement` slice → table | 同函数 `:857-862`：`slot` 计数器代替 `base + unit%stripe_m` | 最忙通道 4352 → 2560 | 项数 8 → 15 |
| **A4b → A5** | ① `prefill_attn` gpu → pim ② `pim_batch_command` replicate → mq | ① `pim_prefill_mode` 走 `:4076` 的 `prefill_side = pim_prefill_mode`；② `_apply_pim_batch:996` 改 `op.pim_batch_command` / `op.pim_pe_freq_ghz` | **放置一字不变**，`loads` 逐格相同；变的是**每条 lane 的 `time_s`**（Ramulator 对同一批 extent 用不同命令定价）：本例 28.545 → 6.806 us | 每项能量降到 ~1/7.6；另外 prefill 从 GPU 事件变成 PIM 事件，换了 `by_class` 的归属 |
| **A5 → A6** | `prefill_attn` pim → dynamic | 选边器 `:4002-4075`，`prefill_side = "pim" if t_bank <= t_xpu else "gpu"`，**每个 request 判一次**，跨层稳定 | 被判给 GPU 的 request，其 prefill 的 PIM lane 组不存在；被判给 PIM 的与 A5 相同 | 同上，按 request 在 GPU 项与 PIM 项之间搬 |

补充两条：
- `num_hbm_used = ceil(kv_heads / heads_per_hbm)`（`:1802`）。本配置 `= ceil(8/8) = 1`，
  所以能量没有额外的堆栈倍数；`--num-hbm` 更大的模型上这一项会放大 sum。
- A5/A6 还带 `gemv_buffer_bytes = 512`（MQ 驻留 8 个 query），影响 `mq_query_capacity`，
  进而影响 prefill 的 sweep 次数。

---
