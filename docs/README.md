# Fugue（`chenyi-0905`）：文档索引

本分支只放论文 `KVPIM-1Fugue-ASPLOS2027` 用到的东西：根基是 **AttAcc 原版**（上游 `c600051`），
之上是 Fugue 的引擎与七档设计阶梯。消融用的其余档、sweep 编排与旧 session 记录在
`chenyi-0904-test` 分支。当前分支的新会话从 [session 文档索引](sessions/README.md) 归档。
当前分支已包含 `workload/probe/` 及其 42 个 sweep 输入。

最新审计：[存储与扫描专项](../audit/2026-09-05/STORAGE_SCAN_CONSISTENCY.md)，补充 [cdd89db 七档公平性复审](../audit/2026-09-05/REAUDIT_cdd89db.md)。A3b–A6 的写入与扫描仍使用两套映射；A1/A2 的抽象和数量边界已分别检查。A6 按用户澄清接受简单逐 request 选边，不要求两套候选 DAG。主审与独立 agent 均已复核；仅修改文档，见 [最新 session](sessions/2026-09-05-storage-scan-and-request-choice.md) 与 [前轮 session](sessions/2026-09-05-cdd89db-fairness-reaudit.md)。

最近修改：[2026-09-05（晚）：审阅计量改动并修复 F01 / F02 / F04](sessions/2026-09-05-ladder-fixes-f01-f02-f04.md)
（A1 prefill 回到 GPU、A3b 持久写入序放置、fresh prefill 按档选边）；之前是
[2026-09-05：AttAcc 计量口径与 GPU query 旋转](sessions/2026-09-05-attacc-accounting-and-rotation.md)。

## 运行指南 —— `README_run_guide.md`

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
  --workload-report out.json --workload-report-events none \
  --cacheblend-batch-size 8 --num-hbm 1 --ngpu 1 --ramulator-workers 8

# 4. 七档一键
RUNGS="A1 A2 A3b A4c A4e A5 A6" NUM_HBM=1 NGPU=1 bash experiments/run_dag_ladder.sh <wl.json> LLAMA3-8B <outdir>

# 5. 测试
python3 -m unittest discover -s tests         # 最近验证：108/108
```

模型 ↔ `--ngpu` / `--num-hbm`：LLAMA-7B、LLAMA3-8B 1/1；GPT-13B、LLAMA-33B 2/10；LLAMA-65B、GPT-175B 8/40。
A1 是唯一用 `--reuse no-reuse` 且不带 `--epic-prefix-recompute-tokens` 的档。
`--pipeopt` 默认 ON（`--no-pipeopt` 是 AttAcc 的 serial 保守约定，会抹掉布局收益）。
