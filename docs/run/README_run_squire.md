# 怎么跑：squire（本机直跑，无 Slurm）—— `chenyi-0905` 分支

跑什么、怎么跑以 [运行协议](../README_run_protocol.md) 为准（每 GPU 5 个 HBM 栈、W1 baseline 与 W1_S* sweep）；指标定义见 [实验指导](../experiments/README.md)。本页只讲这台机器的环境、预算和起停。

squire（`squire.ece.uw.edu`）是 128 核、754 GB、带 `/data2`、**没有 Slurm** 的机器。
本页只讲这台机器上的跑法；athena 集群（Slurm，`node1`–`node6`）见同目录的
`README_run_athena.md`。两台机器的 scratch、编译器、并行度都不同，不要交叉套用。
选项与公平规则的定义在 `../README_run_guide.md`，本页不重复。

## 0. 硬约束

| 项 | 值 |
|---|---|
| 核数 / 内存 | 128 核 / 754 GB |
| **本项目的预算（chenyi9 裁决 2026-09-05）** | **最多 64 核、最多 500 GB**；监视器按整机实际占用 10 s 均值 > 700 GB 才 kill 本批最大进程 |
| `/`（`/tmp` 在上面） | 50 GB，接近满，**不能放 scratch** |
| `/data2` | 7 TB，scratch 放这里 |
| 编译器 | 系统 gcc 8.5；`src/cppcore` 必须用 `/opt/rh/gcc-toolset-11`（`Makefile` 已写死） |
| Python | 3.10 |

## 1. 先决条件（2026-09-05 在本机验证过）

**Ramulator2。** 本分支的 `ramulator2/` 目录只有一个旧二进制（8-22）和 `CMakeFiles`，
**不是 git checkout**，不能在里面跑 `set_pim_ramulator.sh`（它会 `git reset --hard` 到父仓库）。
两条路：

- 直接用 xinyao 树里 8-27 编的二进制：`/data2/chenyi9/KV-PIM/attacc_drampim_xinyao/ramulator2/ramulator2`。
  它与本分支提交的 `pim_ramulator_src/{hbm3_pim_controller.cpp,HBM3-PIM.cpp,trace_gen/}` 逐字节一致，
  当天的所有结果都用它。
- 自己编：拿一份干净的 ramulator2 `b7c7027` 源码（本机没有 git checkout，需 clone），
  按 `set_pim_ramulator.sh` 的 cp/patch 列表覆盖，然后
  `cd build && CC=/opt/rh/gcc-toolset-11/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-11/root/usr/bin/g++ cmake .. && make -j8`。

**事件核。** `cd src/cppcore && make clean && make`（会打印 gcc-toolset-11 的绝对路径）。
用系统 gcc 8.5 编出来的 `.so` 运行时抛 `std::length_error`；`KVPIM_CPPCORE=0` 可临时退回纯 Python（结果逐位相同，慢）。

**scratch 目录**（一次性）：

```bash
export KVPIM_SCRATCH=/data2/chenyi9/KV-PIM/scratch_0905          # 任何 /data2 下的目录
mkdir -p $KVPIM_SCRATCH
ln -sfn /data2/chenyi9/KV-PIM/attacc_drampim_xinyao/ramulator2/ramulator2 $KVPIM_SCRATCH/ramulator2
ln -sfn $PWD/pim_ramulator_src/trace_gen $KVPIM_SCRATCH/trace_gen
```

签名缓存 `signature_cache.jsonl` 会落在这里；键里带二进制 + trace_gen 的指纹，换工具链自动失效。

**验证环境。**

```bash
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
python3 -m unittest discover -s tests          # 2026-09-05：121/121，约 130 s
```

## 2. 跑之前：先起监视器

```bash
setsid nohup experiments/mem_guard.sh $KVPIM_SCRATCH/guard.log > /dev/null 2>&1 &
tail -2 $KVPIM_SCRATCH/guard.log      # 每 30 s 一行：整机占用、本批 RSS、本批核数
```

它只 kill 本批（`main.py … scratch_0905` 与 `ramulator2` 进程）里 RSS 最大的一个，其他人的进程不碰。
核数超 64 只告警不 kill，核数由下面的并行度参数保证。

## 3. 跑

**规则：没有明确指令不跑（`agent.md` §1.7）。** 以下命令都要用户说"跑"才执行。

一个 baseline（七个 combo 并行；6 个 PIM combo × 8 worker + 7 个构建进程 = 55 核）：

```bash
export ATTACC_RAMULATOR_DIR=$KVPIM_SCRATCH ATTACC_RAMULATOR_LOG=$KVPIM_SCRATCH/ramulator.out
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
setsid nohup bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_w1_LLAMA3-8B '^W1_turns' LLAMA3-8B \
    > $KVPIM_SCRATCH/proto_w1_LLAMA3-8B.out 2>&1 < /dev/null &
```

`run_sweep.sh` 按 model 自动设 `NGPU`/`NUM_HBM`（协议 §1 的表），默认 flash、pipeopt、k=8、batch 8、`EVENTS=none`。
`setsid nohup … < /dev/null &` 是必须的：会话的超时会连带杀掉子进程。

一条 sweep 轴（每点 A3b + A6，三点并行 54 核）或整套协议（按 manifest 顺序分批）：

```bash
bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_w1_LLAMA3-8B 'W1_S3_' LLAMA3-8B
bash experiments/run_sweep.sh $KVPIM_SCRATCH/proto_w1_LLAMA3-8B '.' LLAMA3-8B
```

单档手跑（调试）：

```bash
python3 main.py --system dgx-attacc --model LLAMA3-8B --workload workload/probe/sweep/W1_turns.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 --ablation A6 --engine dag --pipeopt --gpu-model flash \
  --workload-report out.json --workload-report-events none --cacheblend-batch-size 8 --num-hbm 5 --ngpu 1 --ramulator-workers 8
```

看进度：

```bash
tail $KVPIM_SCRATCH/proto_w1_LLAMA3-8B/sweep.log
grep -h "done\|FAILED" $KVPIM_SCRATCH/proto_w1_LLAMA3-8B/W1_turns.log
ps -eo args --no-headers | grep 'python3 main.py' | grep -c scratch_0905     # 还在跑的档
tail -1 $KVPIM_SCRATCH/guard.log
```

停（按 PID，别用会匹配到自己 shell 的 `pkill -f`）：

```bash
for p in $(ps -eo pid,args --no-headers | grep -E "run_dag_ladder.sh|run_sweep.sh|python3 main.py|ramulator2 " \
           | grep -E "scratch_0905|attacc_l[0-9]" | awk '{print $1}'); do kill $p; done
```

## 4. 出数

```bash
python3 experiments/extract_protocol.py $KVPIM_SCRATCH/proto_w1_LLAMA3-8B --ref A3b     # protocol.csv + protocol.md：baseline 全表、sweep 的 A6/A3b
python3 experiments/summarize_ladder.py $KVPIM_SCRATCH/proto_w1_LLAMA3-8B/C1_turns workload/probe/sweep/W1_turns.json A3b
#   E2E = makespan；TTFT = 首 token − release；TBT 两种口径（论文用加权）；scan 三列；能量与平均功率；相对某档的比值
cat $KVPIM_SCRATCH/proto_w1_LLAMA3-8B/W1_turns/dag_ladder.csv     # collect_dag_ladder.py 自动生成
cat $KVPIM_SCRATCH/proto_w1_LLAMA3-8B/W1_turns.sides.jsonl        # A6 每个请求的 t_xpu / t_bank / side
```

结果目录不进仓库；汇总表进 `output/analysis/`，数字只能由脚本复制和计算（`agent.md` §3）。

## 5. 模型表

| 模型 | `--ngpu` | `--num-hbm` | 备注 |
|---|---:|---:|---|
| CACHEBLEND-TINY | 1 | 5 | 4 层 8 头，只证明流程能跑通；C1 七档 20 分钟 |
| LLAMA-7B / LLAMA3-8B | 1 | 5 | LLAMA3-8B 是 GQA（8 KV 头，每头 8 通道）；LLAMA-7B 每头 2 通道 |
| GPT-13B | 2 | 10 | 每头 4 通道 |
| LLAMA-33B | 4 | 20 | 每头 5 通道 |
| LLAMA-65B / GPT-175B | 8 | 40 | 每头 8 / 5 通道；大模型每档内存 300–460 GB（9-02 实测，旧引擎），本机预算下一次只能跑一档 |

## 6. 排错

| 现象 | 处理 |
|---|---|
| `std::length_error: vector::_M_range_insert` | `.so` 用错编译器。`cd src/cppcore && make clean && make`；临时 `KVPIM_CPPCORE=0` |
| 退出码 2、日志只有 banner | `WorkloadValidationError` 被 argparse 吞掉；用 `python3 -c "import runpy; runpy.run_path('main.py', run_name='__main__')"` 看 traceback |
| `master rows of one channel exceed the diff region base` | workload 太大，单通道 master 超过 4 MiB 的 diff 区起点；缩小 workload 或改 `_DIFF_REGION_BYTES` |
| ENOSPC | scratch 落到 `/` 了；`ATTACC_RAMULATOR_DIR` 必须指向 `/data2` |
| `ramulator2/` 里冒出 `.trace` / `.yaml` | 没设 `ATTACC_RAMULATOR_DIR` |
| 监视器日志出现 `KILL` | 整机 10 s 均值超过 700 GB；看被杀的是哪一档，缩小并行度后重跑该档 |
