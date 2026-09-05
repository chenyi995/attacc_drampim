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
- **手算校验（2026-09-03）**：`output/handcheck_20260903/`（172 MB）——
  `{A3b,A4,A4b,A5,A6}.json`（`--workload-report-events full` 的完整事件流）+
  `.log` + 复跑脚本 `run.sh`。这是 `workload/handcheck/README.md` 里
  「理论 vs 实测」那两张表的**唯一原始出处**。
  **提取出来的证据已入库**：`workload/handcheck/results_handcheck.csv`
  （109 行，逐档 × 逐扫描 × 逐 channel 的 手算行数/extent/ACT 与实测行数/时间，
  以及 `agree` 列 —— 全部 `yes`）。所以即使这 172 MB 丢了，结论仍可查；
  要重算就用 `workload/handcheck/compare_theory_vs_measured.py`。

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

## 2026-09-03 column-packed 一轮(2026-09-04 归档)

**本机路径**:`output/archived/2026-09-03_colpack/` —— 2.7 GB,
144 个 `dag_A*.json` + 334 个日志 + `claims_rung/` + `slurm/` + 当时的队列文件,
**全部保留在本机,不入库**(带自己的 README,写明里面有什么、口径是什么、怎么续跑)。

**入库的分析产物**:`output/analysis/colpack_20260903/` ——
`baseline_ladder.csv`(42 行,六模型 × 七档)、
`points_partial.csv`(102 行,从各 rung 日志的 `REPORT_SUMMARY` 抽出)、
`README.md`(结果与口径)。

**完成度**:baseline **42/42 跑满**;其余十二个 sweep 点 **102/288,且只有 A5 和 A6**
—— 队列按档排(A5 → A6 → A4 → A3b),在 A4 开始前按用户要求停机,所以
**没有任何一个 (模型, 点) 拿到完整四档**。

**口径提醒**:这一轮是 `33331b7`(PIM 能量不再被多乘 `num_hbm`)之后的第一批数,
**与 2026-09-03 之前的任何一轮能量数都不可比** —— 之前 GPT-175B / LLAMA-65B 虚高
40 倍、GPT-13B / LLAMA-33B 10 倍。makespan 不受该修复影响,但布局模型
(column-packed extent)和 workload(C=16 / sys=256)都换过,同样不可跨轮比。

## 2026-09-04 一轮(同日归档)

**本机路径**:`output/archived/2026-09-04_final/` —— 4.5 GB,289 个 `dag_A*.json` +
522 个日志 + `claims_rung/` + `slurm/` + 队列文件,**保留在本机,不入库**。

**入库的分析产物**:`output/analysis/final_20260904/` ——
`baseline_ladder.csv`(41 行)、`points.csv`(248 行)、`README.md`(结果与口径)。

**完成度**:289/330。baseline **41/42**(只缺 LLAMA-7B 的 A3b);其余十二个点
248/288(A3b 56、A4b 62、A5 65、A6 65,各 /72)。失败 0。缺的集中在
`GPT-175B`/`LLAMA-65B` 的 `private`、`D-hi`、`C-hi`。

**与 0903 轮共享 inode**:本轮开跑时把 0903 轮里不受 `09b50fb` 影响的 138 个 rung
**硬链接**进来了,所以两个归档目录的这部分文件是同一份数据。删其中一个目录不会
损坏另一个,但**不要 `cp -r` 之后删原件**,那会把共享变成两份实拷。

**一个未解的问题**:A3b 重跑后比 0903 轮**快 4–7%**,而 `09b50fb` 预测的是变慢。
TLB、放置模型、`masked_rows`、GPU 时间、事件数、ablation 配置均已排除;决定性实验
(用修复前代码同机重跑 GPT-13B A3b)已提交但随停机取消。**查清之前不要把两轮的
A3b 混比,也不要把差异归因于 `09b50fb`。** 详见两份 README。
