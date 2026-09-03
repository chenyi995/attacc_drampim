# A1:AttAcc 原样、无复用(阶梯参照点)

> **已归档 2026-09-03。** 本档(AttAcc 原样、无复用的参照点)的逐档说明,写于 2026-08-24～27。
> `src/ablation.py` 的 PRESETS 没变,所以**这一档"多做了哪一件事"仍然准确**;
> 归档是因为 `intro/` 整个目录写在 2026-09-03 三处引擎修正(heads-per-HBM `d3a3c4c`、striped-append 布局 `84f87f5`、真实 extent 进 Ramulator `897c294`) 之前、
> 页内的代码指路与"结果预期"都没有在新引擎上复核过。姊妹页
> `README_A3b.md` / `README_A4b.md` 已于本日先行归档(它们写的
> `_layout_channel_loads` 分支已不是那两档实际走的 policy)。
> **当前口径:阶梯定位见 `../../README.md` §3;逐 token 的落点与逐档 ACT
> 见 `../../README_data_layout_walkthrough.md`;手算与实测的逐格对照见
> `../../../workload/handcheck/README.md`。**


一句话:**原版 AttAcc 行为**——prefill 注意力在 GPU、decode 注意力在
PIM、每个请求的 KV 私有连续摆放 (private)、无任何跨请求复用、无批命令
(replicate,一条 MAC 命令服务一条查询)。A1 是整个 A 系列的公平参照:
后面每一档的收益都以它为基准衡量。

## 1. 配置(preset 定义处:`src/ablation.py` 的 `PRESETS["A1"]`)

| 旋钮 | 值 | 含义 |
|---|---|---|
| prefill_attn | gpu | prefill 注意力整块在 GPU 上算 |
| decode_attn | pim | decode 注意力在 bank PE 上扫 |
| kv_mapping | private | 每请求 KV 连续私有摆放,无共享 |
| pim_batch_command | replicate | 无 attention batching |

约束:A1 只与 `--reuse no-reuse` 搭配(校验在 `src/ablation.py::resolve_config`:
`kv_mapping private` 要求 no-reuse 策略)。多轮 agentic 的历史
(`history_len`)照常参与:历史行驻留在 PIM、被扫描、不被重算。

## 2. 模型的每一步在哪里做(解析路径,`--ablation A1`)

入口:`main.py`(`--ablation` 分支)→ `src/ablation.py::resolve_config`
展开 preset → `run_ablation_report` 逐依赖层级 (tier) 定价。

| 步 | 做什么 | 代码位置 |
|---|---|---|
| 1 | 解析 workload(段、指纹、history_len) | `src/workload.py::load_workload` |
| 2 | 复用计划 = 空(no-reuse) | `src/workload.py::build_reuse_plan`(policy=="no-reuse" 提前返回) |
| 3 | 按 tier 组批、构建模型形状(batch, lin, lout) | `src/ablation.py::run_ablation_report` 的 `_tier_shapes` 循环 |
| 4 | 多轮历史加宽注意力(score/softmax 的 n、context 的 k 各 +H) | `run_ablation_report` 中 `hist_rows` 加宽块 |
| 5 | prefill 投影/FF/注意力全部按 GPU 设备定价(scale=1,无节省) | `src/ablation.py::_prefill_batch` 顶部 `sum_decoder` 循环(prefill_attn=="gpu" 时注意力不跳过) |
| 6 | prefill 产生的 KV 写入 PIM(store 语义在解析路径中并入布局/占用) | `_memory_report`(容量);物理细节仅在事件路径 |
| 7 | decode 每步:score 扫描按私有连续 KV 的 Ramulator 实测定价 | `src/ablation.py::_decode_block_time`(`profile.legacy_shape` 走 AttAcc 原始 shape 缓存;历史行由 `_batch_scan_profile(history_rows)` 前置一段驻留 extent) |
| 8 | decode 其余层(投影/FF)按 GPU 定价、AttAcc 流水合成 | `_decode_block_time` + `src/system.py::apply_attacc_pipeline` |
| 9 | 报告(时间/能量/占用分解) | `run_ablation_report` 返回 dict;`main.py` 落盘 |

其中"Ramulator 实测"指:扫描算子带上物理 run 列表
(`op.pim_kv_runs`),经 `src/ramulator_wrapper.py::Ramulator.run` 生成
trace(`ramulator2/trace_gen/gen_trace_attacc_bank.py`)、跑
Ramulator2(HBM3-PIM 命令集)得到周期数。

## 3. 物理事件路径的对应物

A1 的物理对应 = `--reuse no-reuse --no-reuse-latency-model physical`:
`src/workload_runner.py::run_no_reuse_report` /
`_append_physical_no_reuse_prefill_layer`(连续地址规划
`contiguous_address_plan` + 整段扫描;历史行加宽扫描,见
`README_A5.md` §3 的同名机制)。注意:该 legacy 入口对 history 的支持有
显式校验(`run_no_reuse_report` 拒绝 legacy 模型 + history 的组合,
物理模型支持)。

## 4. 怎么跑

```bash
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json \
  --reuse no-reuse --ablation A1 --history-len 3 \
  --pipeopt --workload-report /tmp/a1.json
```

报告里 `ablation.pim_batch_command == "replicate"`、
`ablation.kv_mapping == "private"` 可作自检。

## 5. 与相邻档的关系

A1 → A2:把 decode 也搬回 GPU、KV 不进 PIM,但打开软件复用——两者
差异隔离出"PIM decode 本身"与"复用本身"的贡献;A1 → A3:保持放置,打开
复用与 naive 布局。


---

## 状态更新(2026-08-27)

- **物理 DAG 引擎为默认出数路径**(R11 接通 private 布局;
  `--engine dag`);解析表仅预估/校验;
- **R18 条带映射**:private 连续 extent 的扫描按 head→HBM 条带语义
  执行(一个 KV 头独占 HBM、16 channel 载其 token 条带);
  `--num-hbm` > 头数时头内序列切分同样适用;
- 功耗约束 (PC) 默认开(R19);decode 服务批宽 8(R16)。
