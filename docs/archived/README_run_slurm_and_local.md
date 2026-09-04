# 怎么跑：Slurm 集群 与 本机（无 Slurm）

> **已归档 2026-09-03。** 本页把**两台机器**的跑法写在了一起,而它们的 scratch
> 路径、编译器、并行度、有没有 Slurm 全都不同。合在一页读的人很容易把 squire 的
> `/data2` 和 `gcc-toolset-11` 套到 athena 上 —— athena 上两者都不存在,
> `src/cppcore/Makefile` 里写死的那条 gcc-toolset-11 路径就是这么来的。已拆成
> **两份**:
>
> - `../README_run_athena_slurm.md` —— athena 集群,Slurm,活干在 node1–node6
> - `../README_run_squire_local.md` —— squire,无 Slurm,`/data2`
>
> 本页保留作历史记录。页内的 §2 "本机" 指的是 **squire**,不是 athena。

两套路径。**集群那套是历史上产出 `baseline_20260902_postfix` 那 54 个点的方式**，
脚本还在 `output/_orch2/`；**本机这套是 2026-09-03 新加的**，因为当前这台机器上
没有 Slurm。

> 术语：下文 "rung / 档" 指 A1…A6 消融阶梯（定义见 `README.md` §3）。

---

## 0. 先决条件（两套都要）

### 0.1 编 Ramulator2

```bash
bash set_pim_ramulator.sh
cd ramulator2 && mkdir -p build && cd build && cmake .. && make -j
cp ramulator2 ../ramulator2 && cd ../../
```

### 0.2 编 C++ event core（`KVPIM_CPPCORE=1` 要用）

```bash
cd src/cppcore && make && cd ../..
```

**必须用 gcc-toolset-11**。`Makefile` 里已经写死成 `CXX = /opt/rh/gcc-toolset-11/root/usr/bin/g++`。

> ⚠️ **2026-09-03 修过的坑**：这一行原本是 `CXX ?= ...`。GNU make **内置**就定义了
> `CXX = g++`，所以 `?=` 永远不生效，`make` 一直在用系统编译器（本机 gcc 8.5.0）。
> 编出来的 `.so` 在运行时抛
> `terminate called after throwing an instance of 'std::length_error' / vector::_M_range_insert`
> 直接崩掉整个 run。仓库里若带着一个来历不明的 `libeventcore.so`，
> **先 `make clean && make` 重编一遍**。命令行 `make CXX=...` 仍可覆盖。

自检（本机可用 `/opt/rh/gcc-toolset-{9,11,14}`）：

```bash
cd src/cppcore && make clean && make        # 应打印 gcc-toolset-11 的绝对路径
```

### 0.3 验证环境

```bash
export PYTHONPATH=$PWD
python3 tests/test_placement.py     # 34 个，全过
python3 tests/test_workload.py      # 41 个，全过
```

`test_workload.py` 在 `KVPIM_CPPCORE=1` 下崩，基本就是 0.2 那个坑。

---

## 1. Slurm 集群（athena）

### 1.1 脚本在哪

| 文件 | 干什么 |
|---|---|
| `output/_orch2/common.sh` | 模型表（`NGPU` / `NHBM`）、节点本地 scratch、环境变量 |
| `output/_orch2/baseline_submit.sh` | 一个模型一个作业，提交 baseline 配置的整条阶梯 |
| `output/_orch2/rungs5_submit.sh` | 只跑 A3/A4/A4b/A5/A6 五档的批次 |
| `output/_orch2/slot_submit.sh` / `worker.sh` / `governor.py` | 槽位调度、认领队列 |
| `output/_orch2/status.sh` / `progress_table.py` / `eta.py` | 看进度 |
| `experiments/run_dag_ladder.sh` | 真正的跑档入口（一个 workload → 若干档并行）|

`common.sh` 里的 `REPO` 是**集群上的路径**（`/home/cw636/chenyi/attacc_drampim`），
换机器要改。

### 1.2 提交

```bash
bash output/_orch2/baseline_submit.sh <MODEL> <NODE>
# 例：bash output/_orch2/baseline_submit.sh GPT-13B node2
```

它生成一个 sbatch 脚本（`--cpus-per-task=12 --mem=420G --time=2-00:00:00`），
节点与分区的对应写在脚本里的 `PART` 表（node1→athena-mini、node2/3→athena、
node4→athena-small、node5/6→athena-genai），然后 `sbatch` 出去。

### 1.3 看进度 / 续跑

```bash
squeue -u $USER                      # 队列
bash output/_orch2/status.sh         # 汇总
python3 output/_orch2/progress_table.py
tail -f <ROOT>/slurm/bl_<MODEL>_<JOBID>.out
```

续跑：`run_dag_ladder.sh` 认 `RUNGS` 环境变量，只跑缺的档，
`collect_dag_ladder.py` 会把 `OUT` 目录里现有的 `dag_A*.json` 一起收成 CSV。

### 1.4 集群上的资源纪律（`2026-08-30-HANDOFF.md` 与 `common.sh` 里的教训）

- **scratch 用节点本地盘**（`/localdata/kvpim_$USER`，node5 例外落回 `/tmp`）。
  单档 trace 可能要 128 GB，`/tmp` 在 `/` 上装不下 —— 2026-08-31 有三个档因此 ENOSPC 死掉。
- **signature cache 的 publish 已被禁用**（`common.sh` 的 `publish_loop` /
  `publish_final` 直接 `return 0`）：每次 publish 会把本次 seed 的内容重写回去，
  分片复利膨胀，1 MB 两小时后变 193 GB、全池 499 GB，把 98% 满的共享卷打爆。
  只做进程内 memoisation，**结果不受影响**。
- **大模型的槽把 `KVPIM_NOGC` 设 0**：不回收引用环会多占约 100 GB，
  换来的 16% 提速抵不过并行度损失。

---

## 2. 本机（这台机器，无 Slurm）

`which sbatch` 为空 —— 这台机器上**没有 Slurm**，`output/_orch2/*.sh` 全部用不了。

### 2.1 硬约束

| 项 | 值 |
|---|---|
| 核数 / 内存 | 128 核 / 754 GB（约 690 GB 可用）|
| `/`（`/tmp` 在上面）| 50 GB，**95% 满，只剩约 3 GB** |
| `/localdata` | **本机不可写** |
| `/data2` | 7 TB，剩约 1.9 TB |

> **scratch 必须放 `/data2`。** 放 `/tmp` 会在跑到一半时 ENOSPC。
> 本仓库的脚本用 `/data2/chenyi9/kvpim_run_scratch`。

另外：不设 `ATTACC_RAMULATOR_DIR` 的话，trace 和 yaml 会直接落进
`ramulator2/`，把工作区弄脏（`git status` 里那一堆
`attacc_l*_run*.trace` 就是这么来的）。**始终设它。**

### 2.2 跑 A3b–A6（本次用的）

```bash
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
setsid nohup bash experiments/run_local_a3b_a6.sh \
  > /data2/chenyi9/kvpim_run_scratch/driver.log 2>&1 < /dev/null &
```

- 一次一个模型，**模型内 5 档并行**；核预算 `5 × RAMU_WORKERS(20) + 5 = 105 ≤ 128`。
- 输出 `output/sweep_a3b_a6_20260903_append/<MODEL>/baseline_k8/dag_<RUNG>.json`。
- **可续跑**：某个模型的目录里已经有 5 个 `dag_A*.json` 就跳过。
- 环境：`KVPIM_CPPCORE=1`、`KVPIM_NOGC=0`、各种 `*_NUM_THREADS=1`。
- 改跑哪些模型：`MODELS="GPT-13B LLAMA-7B" bash experiments/run_local_a3b_a6.sh`
- 改并行宽度：`RAMU_WORKERS=14 bash experiments/run_local_a3b_a6.sh`

看进度：

```bash
tail -f /data2/chenyi9/kvpim_run_scratch/driver.log
pgrep -c -f "main.py --system dgx-attacc"          # 还有几个 run 在跑
ls output/sweep_a3b_a6_20260903_append/*/baseline_k8/dag_A*.json | wc -l
grep -h REPORT_SUMMARY output/sweep_a3b_a6_20260903_append/*/baseline_k8/dag_*.log
```

停：`pkill -f run_local_a3b_a6.sh; pkill -f "main.py --system dgx-attacc"`

### 2.3 单档手跑

```bash
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RD=/data2/chenyi9/kvpim_run_scratch/manual
mkdir -p $RD
ln -sf $PWD/ramulator2/ramulator2 $RD/ ; ln -sf $PWD/ramulator2/trace_gen $RD/
cp ramulator.out $RD/
export ATTACC_RAMULATOR_DIR=$RD ATTACC_RAMULATOR_LOG=$RD/ramulator.out

python3 main.py --system dgx-attacc --model GPT-13B \
  --workload workload/sweep/wl_baseline_alltoall_N16_C32_D2.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --ablation A3b --engine dag \
  --workload-report out.json --workload-report-events none \
  --cacheblend-batch-size 8 --num-hbm 10 --ngpu 2 --ramulator-workers 20
```

A1 是唯一用 `--reuse no-reuse` 且**不带** `--epic-prefix-recompute-tokens` 的档。
`--num-hbm` / `--ngpu` 必须按模型表取（下节）。

### 2.4 模型表（两套路径通用，来自 `output/_orch2/common.sh`）

| 模型 | `--ngpu` | `--num-hbm` |
|---|---:|---:|
| LLAMA-7B | 1 | 1 |
| LLAMA3-8B | 1 | 1 |
| GPT-13B | 2 | 10 |
| LLAMA-33B | 2 | 10 |
| LLAMA-65B | 8 | 40 |
| GPT-175B | 8 | 40 |

---

## 3. 出数

```bash
python3 output/analysis/extract_sweep_csv.py \
        --root output/<SWEEP_ROOT> --outdir output/analysis/<TAG> --jobs 8
python3 output/analysis/extract_sweep_csv.py \
        --root output/<SWEEP_ROOT> --self-check 6      # 与 json.load 逐字段对账
```

产出 `sweep_rungs.csv`（一个数据点一行）、`sweep_tiers.csv`、`sweep_completeness.csv`。

---

## 4. 排错

| 现象 | 原因 / 处理 |
|---|---|
| `std::length_error: vector::_M_range_insert`，进程 abort/segv | `libeventcore.so` 陈旧或用系统 gcc 编的。`cd src/cppcore && make clean && make`（见 §0.2）。临时绕开：`KVPIM_CPPCORE=0`（纯 Python，慢但结果逐位相同）|
| **退出码 2，日志里只有 banner、看不到任何错误** | `main.py:588` 把 `WorkloadValidationError` 交给 `parser.error()`，argparse `sys.exit(2)`，真实异常被吞。想看真因，用 `runpy` 包一层并 `except SystemExit` 打 traceback |
| `CacheBlend event has an invalid shape or cost` | `validate_cacheblend_events` 里 `event.rows <= 0`。tiny 模型 / 短上下文下 `_append_placement_pim_scan` 的 report 轮转会给尾部 channel 分到 0 行。**已知问题，未修**，见 `sessions/2026-09-03.md` §6.2。baseline sweep（C=32）不触发 |
| ENOSPC / 跑一半死掉 | scratch 落到 `/`（只剩 3 GB）。设 `ATTACC_RAMULATOR_DIR` 到 `/data2` |
| `ramulator2/` 里冒出一堆 `.trace` / `.yaml` | 没设 `ATTACC_RAMULATOR_DIR`，trace 落进了工作区。删掉即可，它们不该进 git |
| 内存吃爆 | 大模型槽把 `KVPIM_NOGC=0`（本机脚本已经这么设）|
