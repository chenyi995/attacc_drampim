# sweep 实测结果（从原始 JSON 提取的 CSV）

本页描述 `output/analysis/` 下三个 CSV，以及它们说了什么。**数字全部来自
committed 的 CSV**，不是另算一遍；要复算就重新生成，不要手抄本页。

## 0. 三个文件

| 文件 | 一行代表 | 行数（本快照） |
|---|---|---|
| `sweep_rungs.csv` | (模型, 配置, k, 档) —— **主结果** | 693 |
| `sweep_tiers.csv` | (模型, 配置, k, 档, tier) —— 逐层时序 | 1635 |
| `sweep_completeness.csv` | (模型, 配置, k) —— 哪些档在、哪些缺 | 84 |

重新生成：

```bash
bash output/analysis/extract_sweep.sh          # 默认 8 进程
python3 output/analysis/extract_sweep_csv.py --self-check 12   # 与 json.load 对账
```

**来源是 `dag_<档>.json` 本身**，不是 `ladder.log`、不是 claim 目录。提取用逐字段
正则，锚定到缩进层级（`main.py` 输出是缩进 + 键排序，这是前提）；`--self-check`
会用 `json.load` 全量重解析逐字段比对。冷启动约 55 分钟（101 GB，NFS 约 78 MB/s
是下限），之后按 `(路径, mtime, 大小)` 缓存，补跑后重跑约 2 分钟。

`extract_sweep.sh` 结尾会抽样重读 JSON 比对 CSV，不符就非零退出。

> **只有两列是算出来的**：`sweep_tiers.csv` 的 `duration_s`（end − start），和
> `sweep_completeness.csv` 的计数。其余原样拷贝。
>
> `workload_kind_reported` **不是 ground truth** —— 那是仿真器自己的拓扑猜测
> （sweep 的 workload JSON 根本没有 `kind` 字段，两层 alltoall 会被它标成
> `supervisor`）。以 `config` 和 `workload` 两列为准。

---

## 1. 快照与完整度

本页写作时**补跑仍在进行**，所以这是快照，不是终版。

| claim 状态 | 数量 |
|---|---|
| done（9/9） | **69** |
| damaged（缺档，补跑中） | 8 |
| excluded（N-hi，已放弃） | 6 |
| in-flight | 1 |
| **合计** | **84** |

下面所有统计**只用 69 个九档齐全的任务**，残缺任务一律不参与，避免拿半个梯子
和整个梯子比。按模型：GPT-13B 12、LLAMA-33B 12、LLAMA-7B 12、LLAMA3-8B 12、
LLAMA-65B 11、GPT-175B 10。

**补跑结束后必须重跑提取和本页的统计。**

---

## 2. 阶梯（69 个任务的中位值）

| 档 | makespan (s) | 能量 (kJ) | KV link (GiB) | decode | prefill | kv_mapping | PIM 占能量 |
|---|---:|---:|---:|---|---|---|---:|
| A1 | 95.0 | **71.5** | 209.1 | pim | gpu | private | 91% |
| A2 | **163.2** | 4.8 | **21201.1** | gpu | gpu | none | — |
| A3 | 44.8 | 12.1 | 112.7 | pim | gpu | naive | 76% |
| A3a | 41.7 | 12.1 | 112.7 | pim | gpu | naive-mask | 77% |
| A3b | 43.9 | 12.1 | 112.7 | pim | gpu | naive | 76% |
| A4 | 41.3 | 12.2 | 112.7 | pim | gpu | master-diff | 77% |
| A4b | 41.3 | 12.2 | 112.7 | pim | gpu | master-diff | 77% |
| A5 | 28.0 | 6.1 | 15.5 | pim | pim | master-diff | 61% |
| **A6** | **27.5** | 5.9 | 18.4 | pim | dynamic | master-diff | 59% |

**A6 相对每一档的中位改善（正 = A6 更好）：**

| 基线 | makespan | 能量 | KV link |
|---|---:|---:|---:|
| A1 | +68.0% | +91.2% | +90.8% |
| A2 | +84.3% | **−29.6%** | +99.9% |
| A3 | +25.3% | +46.1% | +82.6% |
| A3a | +22.9% | +46.0% | +82.6% |
| A3b | +24.9% | +46.1% | +82.6% |
| A4 | +22.3% | +46.1% | +82.6% |
| A4b | +22.3% | +46.1% | +82.6% |
| A5 | +0.0% | +0.0% | +0.0% |

**按模型看 A6 相对 A1：**

| 模型 | 任务 | makespan | 能量 | KV link |
|---|---:|---:|---:|---:|
| LLAMA3-8B | 12 | +86.2% | +97.4% | +67.4% |
| GPT-13B | 12 | +70.8% | +89.0% | +92.9% |
| LLAMA-33B | 12 | +69.5% | +89.4% | +91.9% |
| LLAMA-7B | 12 | +68.0% | +95.3% | +92.9% |
| GPT-175B | 10 | +61.3% | +84.7% | +91.9% |
| LLAMA-65B | 11 | +59.1% | +89.8% | +92.9% |

---

## 3. 三件数据说了、但需要小心处理的事

### 3.1 A6 不是能量最低的档 —— A2 才是，低 19%

| | 总 | GPU | PIM | LINK |
|---|---:|---:|---:|---:|
| A2 | **4.79 kJ** | 4.15 | **0.00** | 0.641 |
| A6 | 5.88 kJ | 1.59 | **4.45** | 0.001 |

原因不含糊：**A2 的 `decode_attn=gpu`，它根本不开 PIM**，能量里没有 PIM 这一项。
A6 把 GPU 能量从 4.15 降到 1.59 kJ，但 PIM 加了 4.45 kJ 回来。

代价那边同样不含糊：A2 比 A6 **慢 5.9 倍**（163.2 s vs 27.5 s），KV 链路流量是
**1150 倍**（21,201 GiB vs 18.4 GiB）。

所以"能量更低"这个说法**对 A1（no-reuse）成立且幅度很大（+91%），对 A2（软件复用
基线）不成立**。论文里把 A2 当基线时，卖点必须是时间和链路流量，不能是能量。
这是审稿人会第一个找的地方。

### 3.2 A3 / A3a / A3b / A4 / A4b 五档的结果几乎完全相同

上表里这五行的 makespan 在 41.3–44.8 s，能量 12.1–12.2 kJ，**KV link 全部
112.7 GiB**。A6 相对它们的改善也几乎一样（makespan +22.3%～+25.3%，能量 +46%，
链路 +82.6%）。

也就是说**整条放置阶梯（naive → naive-mask → head 切片 → master/diff →
全局 co-read 表）在本轮 workload 上没有产生可报告的效应**。逐对的严格检验见
`README_rung_analysis.md` §1：A3 vs A3a、A3b vs A4b、A4 vs A4b 三对在全 sweep
的**任何一个**配置上都分不开（最大差异 8.9% / 13.8% / 28.8%，且 KV link 处处
完全相同）。

### 3.3 A6 的选边在一个格子上破了它不该破的不变量

A6 逐 request 选 GPU 或 PIM prefill，所以它**永远不该输给 A4（全 GPU）或
A5（全 PIM）**。60 个有这三档的任务里破了 2 个，实质违反 1 个：
`LLAMA-7B / k-hi` 比 A4 慢 **23.3%**。诊断见 `README_rung_analysis.md` §3。

`prefill_requests_pim` 列可以直接看到选边行为：A4 中位 0%（强制 GPU）、
A5 中位 97%、**A6 中位 94%** —— A6 绝大多数情况下与 A5 一致，只在 LLAMA3-8B 上
真正改选 GPU。

---

## 4. 已知缺口

1. **N-hi（64 agent）整行放弃**，六个模型一个都不跑 —— N 轴只剩 N=4 与 N=16
   两点，没有第三点显示曲率。理由与代价见 `README_sweep_design.md` §6.1。
2. **受损任务仍在补跑**；`sweep_completeness.csv` 的 `rungs_missing` 列逐任务
   列明当前状态。缺档绝大多数由节点磁盘/内存耗尽造成，**但不是全部** —— 见 §5。
3. **A2 没有 `overlap_validation`**：`overlap_passed` 列在 A2 的 84 行上为空，
   其余 610 行全为 `True`。所以"重叠自检全过"这个说法只对 A2 以外的档成立。
4. `dag_ladder.csv` 有三个任务缺失（`LLAMA-33B/private`、`LLAMA-33B/C-hi`、
   `LLAMA3-8B/private`）—— 我在作业运行中改了 `run_dag_ladder.sh`，触发 NFS
   `Stale file handle`。**九档 JSON 全部完好，本页数据不受影响**，已补生成。
   经过见 `sessions/2026-09-01.md`。

---

## 5. 已确认的引擎缺陷：`LLAMA-7B × pipeline` 的 A3 / A3a / A3b 确定性崩溃

**这是本轮唯一一处不能归因于基础设施的失败。** 之前几版文档写过"没有一档死于
引擎缺陷"，那个说法**已被证伪**，此处更正。

### 影响面：一个格子，不多不少

| 检验 | 结果 |
|---|---|
| `wl_pipeline_D4.json` 在其他五个模型上 | **全部 9/9** |
| `LLAMA-7B` 在其余 12 个配置上 | **A3/A3a/A3b 全部正常** |
| 受影响 | **仅 `LLAMA-7B / pipeline_k8`,3 档** |

所以既不是"A3 这一档坏了"，也不是"pipeline 这个 workload 坏了"，更不是
"LLAMA-7B 这个模型坏了" —— 是**这三者的交集**。A4/A4b/A5/A6 在同一个格子上
正常产出，A1/A2 也正常。

### 复现证据

两次运行，不同节点，字节级一致：

| | 第一次 | 第二次 |
|---|---|---|
| 时间 / 节点 | 2026-08-31 02:07 / **node6** | 2026-09-01 19:41 / **node1** |
| 失败档 | A3, A3a, A3b | **同样三档** |
| 日志结尾 | 启动横幅之后无任何输出 | **同样** |
| 日志字节数 | 25863 / 25821 / 25863 | **完全相同** |
| 节点内存占用 | 14% | 8% |

两个不同节点、两次独立运行、**日志大小逐字节相同**，说明是在同一个点确定性
崩溃，不是环境抖动。

### 已排除

- **不是 OOM**：两次的节点内存占用分别是 14% 和 8%。
- **不是 ENOSPC**：无 `Errno 28`，`/localdata` 有 TB 级空间。
- **不是 Python 异常**：`run_dag_ladder.sh` 对每一档都做 `2>&1`，有异常必然进日志。
  日志停在启动横幅，之后什么都没有。**非零退出 + 无 traceback = 致命信号。**
- **不是节点特有**：node6 与 node1 都复现。

### 尚未确定

**具体是哪个信号、崩在哪里。** 假设是原生代码崩溃（`src/cppcore/eventcore.cpp`
或 ramulator2 子进程），但**这是假设，未验证**。

一个可能相关的结构性事实：`wl_pipeline_D4.json` 只有 **4 个 agent**（每层 1 个，
共 4 层，是全 sweep 最小的 workload），而 LLAMA-7B 是 `num_hbm=1` 且
`kv_heads=32`，即 `heads_per_hbm=32` —— 六个模型里通道最拥挤的一个
（每通道 2 个 head，次差的 GPT-175B 只有 0.62，见 `README_rung_analysis.md` §3）。
A3/A3a/A3b 恰好是**不走 master/diff 分池**的三种放置，其 `stripe = 16 // 32 = 1`。
**"请求极少 × 通道极挤 × 非 master-diff 放置"这个组合值得先查**，但在信号被抓到
之前，它只是一条线索。

### 现状

这三档**补不回来** —— 补跑于 2026-09-01 19:27 在 node1 上重试，三档一个都没产出
（`6/9 -> 6/9`）。它们要等这个 bug 修掉。所以本轮的终态预计是
**77 个任务完整 + `LLAMA-7B/pipeline` 缺 3 档**。

单档复现作业已提交（node1，4 核 64G），专门捕获退出码与信号名；结果出来后更新
本节。诊断全文另见 `claims/LLAMA-7B__pipeline_k8/damaged`。
