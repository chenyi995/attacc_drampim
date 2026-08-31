# 原始数据本地清单 (RAW DATA MANIFEST)

**口径（2026-08-29 起）**：所有仿真原始 run 数据（`dag_A*.json` 事件轨迹 +
`.log`）**一律不入 git、只在本机**（体积大、GitHub 拒 >100 MB）。git 里只留
**`output/analysis/`**（分析脚本与 RESULTS 表）。

本机根路径 `/data2/chenyi9/KV-PIM/attacc_drampim_xinyao`。

## 原始 run 数据在哪
- **新参数化 sweep**：`output/sweep_<时间戳>/<config>_k<k>/dag_A*.{json,log}`
  （由 `experiments/run_sweep.sh` 产出；workload 定义 `workload/sweep/`，入库）；
- **归档的旧手调结果**：`output/archived/2026-08-29_pre-unify/`（21 个 run，
  9.6 GB，含旧 workload 备份与 RESULTS 快照；见该目录 README）。

## 需要这些原始数据时
本机直接取；异地复现用 `workload/`（入库的 workload 定义）+
`experiments/run_sweep.sh` / `run_dag_ladder.sh` 重跑即可（RESULTS 表由
`output/analysis/make_results_tables.py` 从 `dag_A*.json` 重算）。

## 口径修订(2026-08-31,用户裁决):脚本入库,产物仍只在本机

`output/` 下**不是所有东西都是数据**。原口径把整个 `/output/*` 排除,
连带把两套**脚本**也挡在库外;它们丢了要重写,而且丢的是方法不是数据:

- **`output/_orch2/`** — 集群编排与资源 governor。它带着**实测标定的模型**:
  `建图秒数 ≈ 122 + 2135 × W`、`常驻内存 ≈ 40 + 25 × W`
  (`W = 输出 token 数 × 层数 × 每 agent token 数 / 1e9`),以及按这两个模型
  做的放置与限流。这些系数是跑出来的,不是拍的;
- **`output/_verify/`** — 逐字节等价验证基建。参照矩阵、严格比对器、
  `rows` 不变量检查,以及**用 FIFO 流式 sha256 比对 25 GB 事件流**的做法
  (报告不落盘,适用于共享卷紧张时)。对以后任何引擎改动都能直接复用。

**仍然排除的**(和原口径一致):这两个目录的**运行产物** ——
`_verify/{before,after,after2}/` 的报告、`pristine/`、大 JSON;
以及所有 `dag_A*.json` 事件轨迹、`ramulator2/` 的构建与 trace。
`_verify/` 里保留的 `.sha` / `.cmp` / `.inv` / `.out` 是**验证证据**(几十 KB),
`snapshot.jsonl`(98 KB)是冻结的缓存种子,留着才能复现那次比对。
