# 怎么跑:athena 集群(Slurm)

**这台机器**(`athena.egr.duke.edu`)是**带 Slurm 的集群登录节点**,活干在
`node1`–`node6` 上。跑法只有一条:写 sbatch,`sbatch` 出去,别在登录节点上跑真实
负载。

> squire 那台(128 核、`/data2`、**没有 Slurm**)的跑法是**另一份文档**:
> `README_run_squire_local.md`。两台机器的 scratch 路径、编译器、并行度都不一样,
> 不要交叉套用。

---

## 0. 这台机器的硬约束

| 项 | 值 |
|---|---|
| 登录节点 | 48 核 / 251 GB —— **只用来提交和看进度** |
| 计算节点 | `node1`–`node6`,各 96–128 核 / 1 TB |
| `/`(`/tmp` 在上面) | 214 GB,**95% 满,只剩约 12 GB** |
| `/localdata` | 计算节点上 42 TB(node6 剩 15 TB);**登录节点上不存在** |
| `/data2` | **本机不存在**(那是 squire 的盘) |
| 编译器 | 系统 `g++` **11.4.0**;**没有 `/opt/rh/gcc-toolset-*`** |
| 仓库路径 | `/home/cw636/chenyi/attacc_drampim`(= `/zpool-00/home/...`,同一个 inode) |

分区与节点的对应(写死在各 submit 脚本的 `PART` 表里):

| 节点 | 分区 |
|---|---|
| node1 | athena-mini |
| node2 / node3 | athena |
| node4 | athena-small |
| node5 / node6 | athena-genai |

---

## 1. 先决条件

### 1.1 Ramulator2

```bash
bash set_pim_ramulator.sh
cd ramulator2 && mkdir -p build && cd build && cmake .. && make -j
cp ramulator2 ../ramulator2 && cd ../../
```

### 1.2 C++ event core —— **本机不要 `make`**

`src/cppcore/Makefile` 里写死的是

```make
CXX = /opt/rh/gcc-toolset-11/root/usr/bin/g++
```

那是 **squire 的路径**,本机(登录节点和 node6 都确认过)**没有 `/opt/rh`**,
直接 `make` 会失败。本机系统 `g++` 本身就是 11.4.0,树里带的
`src/cppcore/libeventcore.so` 可以正常加载,而且 `eventcore.cpp` 自 2026-09-02
起没有再改过,**所以本机跑不需要重编**。

真要重编,显式给编译器:

```bash
cd src/cppcore && make clean && make CXX=g++ && cd ../..
```

### 1.3 验证环境(在登录节点上跑就行,很快)

```bash
export PYTHONPATH=$PWD
python3 tests/test_placement.py                    # 46 个,全过
KVPIM_CPPCORE=1 python3 tests/test_workload.py     # 41 个,全过(约 130 s)
```

`test_workload.py` 在 `KVPIM_CPPCORE=1` 下崩(`std::length_error` /
`vector::_M_range_insert`)= `.so` 和源码对不上,按 1.2 重编。

---

## 2. 资源纪律(踩过的坑,别再踩)

- **scratch 必须落 `/localdata`。** `common.sh` 的 `scratch_root()` 已经这么做了
  (`/localdata/kvpim_$USER`,不可写时退回 `/tmp`)。单档 trace 可能要 128 GB,
  `/` 只剩 12 GB —— 2026-08-31 有三个档因此 ENOSPC 死掉。
- **一定要设 `ATTACC_RAMULATOR_DIR`。** 不设的话 trace 和 yaml 直接落进
  `ramulator2/`,把工作区弄脏(`git status` 里那一堆 `attacc_l*_run*.trace`
  就是这么来的)。`make_ramdir` 会替你设好。
- **signature cache 的 publish 已被永久禁用**(`common.sh` 里 `publish_loop` /
  `publish_final` 直接 `return 0`)。每次 publish 会把本次 seed 的内容重写回去,
  分片复利膨胀:1 MB 两小时后变 193 GB、全池 499 GB,把 98% 满的共享卷打爆。
  现在只做进程内 memoisation,**结果不受影响**,只是首跑慢些。
- **大模型的槽把 `KVPIM_NOGC` 设 0**。不回收引用环会多占约 100 GB,
  换来的 16% 提速抵不过并行度损失。

---

## 3. 当前这轮:column-packed 重跑(2026-09-03)

引擎口径:`_striped_append_channel_extents` —— 每条 channel 的**真实 extent**
(一个 cached chunk 一个,消费者补的那几行**另算一个**)交给 Ramulator 跑**一次**
仿真,由它的 row buffer 决定 ACT 数;一个 1024 B 的 DRAM row 装 256 个 token,
一个 col(32 B)装 8 个 —— k=8 的修正就占一个 col,自己占掉一整行。A3b 让每次修正
各占一行,master/diff 各档把所有 head 的修正打包进 diff 通道共享行。
workload 是 C=16、sys=256 的新一批(baseline 上下文 `18×256 = 4608`,不带零头;
4 个 head × 16 个 chunk × 8 token = 512 token = **恰好 2 行**打包在 ch15)。

跑什么(chenyi9 裁决):

| 配置 | 档 |
|---|---|
| baseline | **A1 A2 A3b A4 A4b A5 A6**(没有 A3/A3a) |
| 其余每个 sweep 点 | **A3b 和 A6** 两档 |

六个模型 × (1 个 baseline + 12 个点) = **78 个任务**,队列在
`output/sweep_colpack_20260903/tasks.txt`,**baseline 排在最前面**,先被认领。
`N-hi`(`wl_N64.json`,128 个 agent)仍然放弃,和 rungs5 那轮一致。

### 3.1 提交 —— 三个槽位 class,三条队列

**队列是分开的**,这样 72 个两档的 point 任务和 6 个七档的 baseline **同时跑**,
而不是排在 baseline 后面;而且每个 class 只要它真正需要的内存。

| class | 队列 | 任务 | `--cpus-per-task` | `--mem` | `--time` |
|---|---|---|---:|---:|---:|
| `big` | `tasks_baseline_big.txt` | GPT-175B / LLAMA-65B 的 baseline,7 档 | 24 | 300G | 24h |
| `base` | `tasks_baseline.txt` | 另外四个 baseline,7 档 | 24 | 170G | 16h |
| `points` | `tasks_points.txt` | 一个 sweep 点的 A3b+A6,2 档 | 6 | 64G | 6h |

```bash
bash output/_orch2/colpack_tasks.sh          # 写三条队列
bash output/_orch2/colpack_submit.sh node3 big
bash output/_orch2/colpack_submit.sh node4 big
bash output/_orch2/colpack_submit.sh node2 base
for n in node1 node1 node6 node6 node6 node5; do
  bash output/_orch2/colpack_submit.sh $n points
done
```

槽是**互相独立的认领者**(claim 目录 `claims_bf/` 三个 class 共用,所以不会重复领),
加槽就是加吞吐。

**内存怎么定的**:2026-09-02 那轮**九档 + C=32(8464 token)**的实测峰值是
GPT-175B 461 GB、LLAMA-65B 386、LLAMA-33B 297、GPT-13B 194、LLAMA3-8B 169、
LLAMA-7B 170(`sacct -j <id> --format=MaxRSS`)。这轮是**七档 + 上下文 4608**,
所以上表的要价按比例留了余量。

> ⚠️ **要价过大会让作业根本排不上。** 第一次提交时六个槽统一要 420G,
> 结果三个卡在 `(Priority)` 上进不去 —— 而实测一个 GPT-13B 的七档槽当时只用
> 34 GB。**Slurm 的 backfill 只会塞得下的作业**,所以要价要贴着实际需求。

**要价小只是一半,`--time` 是另一半。** Slurm 的 backfill 只会启动那些**能证明在
最高优先级作业的预约开始之前跑完**的作业。最初这几个槽写的是 `--time=3-00:00:00`,
在一个前面排着 55 个作业的分区上**几乎永远 backfill 不进去** —— 即使核数和内存都
压小了也一样。把 walltime 按实际工作量改成 24h / 16h / 6h 之后,原本一直
`(Priority)` 的 node2/3/4 立刻排上了:**在跑的槽从 12 个变成 22 个,六个节点全用上,
同时在跑的 rung 进程 60 个。**

**短 walltime 要配回收器。** `backfill.sh` 用 `mkdir claims_bf/<tid>` 原子认领,
只在跑完时写 `done`;槽要是被 walltime 杀掉,claim 目录会**留着**,别人 `mkdir`
失败,这个任务就再也没人碰。`output/_orch2/colpack_reap.sh` 负责把
"owner 的 job 已经不在队列里、又没有 done" 的 claim 释放掉:

```bash
bash output/_orch2/colpack_reap.sh                      # 跑一次
setsid nohup bash output/_orch2/colpack_reap.sh \
  $PWD/output/sweep_colpack_20260903 --loop 600 &       # 每 10 分钟一次
```

重跑一个被回收的任务是便宜且正确的:`backfill.sh` 在跑每个档之前都对着磁盘重新
核对,已经跑完的档直接跳过。

**分区拥挤仍然是真的**:`athena`(node2/3)和 `athena-small`(node4)前面压着别人的
高优先级作业(`squeue -h -t PD -o "%P" | sort | uniq -c` 看得到 55/23/11),
`athena-mini`(node1)与 `athena-genai`(node5/6)通常更快。但短 walltime 让拥挤分区
也能塞进去了,所以现在往哪个节点加都行。

### 3.2 看进度

```bash
squeue -u $USER
tail -f output/sweep_colpack_20260903/slurm/cp_*_*.out
ls output/sweep_colpack_20260903/claims_bf | wc -l        # 已认领的任务数
find output/sweep_colpack_20260903 -name 'dag_A*.json' | wc -l
grep -rh REPORT_SUMMARY output/sweep_colpack_20260903/*/*/dag_*.log
```

### 3.3 续跑 / 停

- **续跑**:直接再提一个槽。`backfill.sh` 在跑每个档之前会**对着磁盘重新核对**,
  已经有 `dag_<档>.json` 的档跳过,所以死在半路的任务只会补上缺的那几档。
- **加并行度**:再提几个 `points` 槽(8 核 / 90G,最容易 backfill)。
- **停某个 job**:`touch output/sweep_colpack_20260903/drain/<JOBID>`,
  它跑完手上这个任务就退出(比 `scancel` 干净,不会留半个 json)。
- **硬停**:`scancel -u $USER -n cpp_node6`(job 名是 `cpb_`/`cpp_` + 节点)。

---

## 4. 单档手跑(调试用)

```bash
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RD=/localdata/kvpim_$USER/manual; mkdir -p $RD
ln -sf $PWD/ramulator2/ramulator2 $RD/ ; ln -sf $PWD/ramulator2/trace_gen $RD/
cp ramulator.out $RD/
export ATTACC_RAMULATOR_DIR=$RD ATTACC_RAMULATOR_LOG=$RD/ramulator.out

python3 main.py --system dgx-attacc --model GPT-13B \
  --workload workload/sweep/wl_baseline_alltoall_N16_C16_D2.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --ablation A3b --engine dag \
  --workload-report out.json --workload-report-events none \
  --cacheblend-batch-size 8 --num-hbm 10 --ngpu 2 --ramulator-workers 20
```

`/localdata` 只有计算节点上有,所以这条要在 `srun` 到节点上之后再跑:

```bash
srun --partition=athena --nodelist=node2 --cpus-per-task=24 --mem=180G \
     --time=8:00:00 --pty bash
```

A1 是唯一用 `--reuse no-reuse` 且**不带** `--epic-prefix-recompute-tokens` 的档。

### 模型表(`output/_orch2/common.sh` 里的 `NGPU` / `NHBM`)

| 模型 | `--ngpu` | `--num-hbm` |
|---|---:|---:|
| LLAMA-7B | 1 | 1 |
| LLAMA3-8B | 1 | 1 |
| GPT-13B | 2 | 10 |
| LLAMA-33B | 2 | 10 |
| LLAMA-65B | 8 | 40 |
| GPT-175B | 8 | 40 |

---

## 5. 出数

```bash
python3 experiments/collect_dag_ladder.py <OUT_DIR> <workload.json> <MODEL>
python3 output/analysis/extract_sweep_csv.py \
        --root output/sweep_colpack_20260903 --outdir output/analysis/colpack --jobs 8
python3 output/analysis/extract_sweep_csv.py \
        --root output/sweep_colpack_20260903 --self-check 6   # 与 json.load 逐字段对账
```

`collect_dag_ladder.py` 现在**按磁盘上实际存在的档出 CSV**(2026-09-03 修):
它以前写死七档、漏掉 A3b 和 A4b,所以 9-02 那轮的每个 `dag_ladder.csv` 都少两行,
而 `dag_A3b.json` / `dag_A4b.json` 其实都在。这轮 baseline 是七档、其余点是两档,
都能正常收。

---

## 6. 排错

| 现象 | 原因 / 处理 |
|---|---|
| `make` 报 `/opt/rh/gcc-toolset-11/...: No such file` | 那是 squire 的路径。本机用 `make CXX=g++`,或者干脆别重编(见 §1.2) |
| `std::length_error: vector::_M_range_insert`,进程 abort | `libeventcore.so` 和源码对不上。`make clean && make CXX=g++`。临时绕开:`KVPIM_CPPCORE=0`(纯 Python,慢但结果逐位相同) |
| **退出码 2,日志里只有 banner** | `main.py:588` 把 `WorkloadValidationError` 交给 `parser.error()`,argparse `sys.exit(2)`,真实异常被吞。想看真因用 `runpy` 包一层并 `except SystemExit` 打 traceback |
| `CacheBlend event has an invalid shape or cost` | `validate_cacheblend_events` 里 `event.rows <= 0`。上下文比活跃 channel 数还短时,`_append_placement_pim_scan` 的 report 轮转会给尾部 channel 分到 0 行。**已知未修**,见 `sessions/2026-09-03.md`;本轮 sweep 的上下文都够长,不会触发 |
| ENOSPC / 跑一半死掉 | scratch 落到 `/` 了(只剩 12 GB)。确认 `ATTACC_RAMULATOR_DIR` 指向 `/localdata` |
| `ramulator2/` 里冒出一堆 `.trace` / `.yaml` | 没设 `ATTACC_RAMULATOR_DIR`。删掉即可,它们不该进 git |
| 内存吃爆 | 大模型槽用 `big`(`KVPIM_NOGC=0`) |
| 登录节点卡死 | 有人在登录节点上跑了真实负载。所有活都要 `sbatch` / `srun` 到 node 上 |
