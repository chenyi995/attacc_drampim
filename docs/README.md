# Fugue（`chenyi-0905`）：文档索引

**本轮 audit 只需看 [CURRENT_ISSUES.md](../audit/2026-09-05/CURRENT_ISSUES.md)。** 每个 case 都解释论文声明、实际行为、AttAcc 来源和相对影响；旧报告与证据已归档，不再作为并列的“最新报告”。[Audit 目录入口](../audit/2026-09-05/README.md)；[整理 session](sessions/2026-09-05-audit-docs-cleanup.md)。

本仓库在原始 AttAcc 上实现 Fugue 的七档比较。当前口径接受共同模型限制，FlashAttention 必须启用，候选问题由用户逐项裁决；设计说明见下方，历次记录见 [session 索引](sessions/README.md)。

## 运行指南 —— `README_run_guide.md`，分机器的跑法在 `run/`

`run/README_run_squire.md`（本机直跑，64 核 / 500 GB 预算、监视器、scratch 在 /data2）与
`run/README_run_athena.md`（Slurm，scratch 在 /localdata，一档一核的要价规则）。

跑论文数据必须开的选项（`--gpu-model flash`、`--pipeopt`、`--powerlimit`、k=8、batch 8）、公平规则、
baseline + 八个 sweep 轴、跑法与汇总；workload 矩阵在 `workload/probe/sweep/`（`gen_sweep.py` 生成，42 个文件）。

## 设计阶梯 —— `README_design_ladder.md`

| 档 | 论文里的角色 | 相对上一档的变化 |
|---|---|---|
| A1 | 硬件 baseline（AttAcc 原样） | — |
| A2 | 软件 baseline | 复用有了，decode 搬回 GPU，KV 在远端哑存储 |
| A3b | 朴素 PIM 存储 | decode 回 PIM；chunk 与修正混在一条 append 流里按写入序轮转 |
| A4c | claim 1：per-agent diff 行 | 该 head 的修正聚到它自己通道上的几行；master 一条通道都不让 |
| A4e | claim 2：placement table | master chunk 的通道由写入时的冲突感知表决定 |
| A5 | claim 3：prefill 进 PIM | prefill 注意力进 PIM + MQ 批命令 |
| A6 | Fugue | prefill 逐请求动态选边 |

每档相对上一档的差别说到函数、参数、归约里哪一项，以及实测证据与证据等级，都在那一页。

## 相对上游 AttAcc 改了什么

| 位置 | 内容 |
|---|---|
| `src/workload_runner.py`（新） | 物理事件 DAG 引擎：workload → 每请求每层的事件图，PIM 扫描按通道 lane 发给 Ramulator，`--pipeopt` 下各设备各自的资源时间轴 |
| `src/ablation.py`（新） | 七档 preset、`resolve_config` |
| `src/workload.py`（新） | workload 文件格式、复用计划（`recompute` 策略：每个位移段重算 k 个 token） |
| `src/layout_probe.py`（新） | `KVPIM_LAYOUT_DUMP=<file>` 时逐扫描 dump 交给 Ramulator 的 extent 与两个归约的每一项 |
| `src/cppcore/`（新） | 事件核的 C++ 实现（`make`，gcc-toolset-11） |
| `src/ramulator_wrapper.py`、`src/devices.py`、`src/system.py`、`src/config.py`、`main.py` | 在上游基础上加：真实 extent 列表进 Ramulator、`pim_activations` 统计、MQ 命令定价、`--num-hbm` / `--ngpu` / `--pipeopt`（默认 ON） |
| `pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py` | `--kv-extents-file`：一条通道的全部 extent 作为一次仿真，ACT 由行缓冲决定；MQ / shared-kv 命令 |
| `pim_ramulator_src/hbm3_pim_controller.cpp` | `pim_activations` 统计（ACT / ACTAB / ACTSB / ACTPB） |
| `pim_ramulator_src/HBM3-PIM.cpp` | MVSB ↔ MVGB/WRGB 方向切换时序 |
| `tests/` | 当前 108 个单测，包括放置/事件路径、DIE/TLB 计量与 Python/C++ metadata 调度一致性 |
| `output/analysis/` | 布局手算 CSV（`layout_grid_csv.py`）、放置规则独立重写 vs 引擎 dump 对账（`layout_handcheck_theory.py`）、同分配器只改 diff 落点的公平对照（`diff_gather_effect.py`） |

## 怎么跑

```bash
# 1. ramulator2：上游的方式（submodule + pim_ramulator_src 覆盖 + patch），再编译
git submodule update --init ramulator2
bash set_pim_ramulator.sh
(cd ramulator2 && mkdir -p build && cd build && cmake .. && make -j && cp ramulator2 ..)

# 2. 事件核
(cd src/cppcore && make)                      # Makefile 写死 gcc-toolset-11

# 3. 跑一档（scratch 必须放 /data2；ATTACC_RAMULATOR_DIR 里要有 ramulator2 二进制与 trace_gen/ 的软链）
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
RD=/data2/<you>/scratch; mkdir -p $RD; ln -sf $PWD/ramulator2/ramulator2 $RD/; ln -sf $PWD/ramulator2/trace_gen $RD/
export ATTACC_RAMULATOR_DIR=$RD ATTACC_RAMULATOR_LOG=$RD/ramulator.out
python3 main.py --system dgx-attacc --model LLAMA3-8B --workload <wl.json> \
  --reuse recompute --epic-prefix-recompute-tokens 8 --ablation A4e --engine dag \
  --gpu-model flash --pipeopt \
  --workload-report out.json --workload-report-events none \
  --cacheblend-batch-size 8 --num-hbm 1 --ngpu 1 --ramulator-workers 8

# 4. 七档一键
GPU_MODEL=flash RUNGS="A1 A2 A3b A4c A4e A5 A6" NUM_HBM=1 NGPU=1 bash experiments/run_dag_ladder.sh <wl.json> LLAMA3-8B <outdir>

# 5. 测试
python3 -m unittest discover -s tests         # 最近验证：108/108
```

模型 ↔ `--ngpu` / `--num-hbm`：LLAMA-7B、LLAMA3-8B 1/1；GPT-13B、LLAMA-33B 2/10；LLAMA-65B、GPT-175B 8/40。
A1 是唯一用 `--reuse no-reuse` 且不带 `--epic-prefix-recompute-tokens` 的档。
`--pipeopt` 默认 ON；`--no-pipeopt` 是串行对照，不能替代本轮要求的正式配置。
直接运行 `main.py` 必须显式传 `--gpu-model flash`；只设置同名环境变量不会改变 CLI 默认。
以上命令仅补齐配置，不代表当前实现和结果已通过审计。
