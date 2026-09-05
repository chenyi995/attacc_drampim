# kvpim-sim：GPU–PIM 共享 KV 服务仿真器 —— 文档索引

目标读者：有计算机背景、不了解本项目 / LLM serving / DRAM-PIM 细节的人。
概念首次出现即解释，关键术语标注英文。

## 这个项目是什么

大语言模型（LLM）推理分 **prefill**（把整段输入一次算完、写出 KV 缓存）和 **decode**
（逐词生成，每个新词都要把历史 KV 从头读一遍做注意力，attention）两段。KV 缓存（KV cache）
是每个历史 token 存下的一条 K 向量、一条 V 向量；decode 的瓶颈是把它从内存搬到计算单元的**带宽**。

AttAcc（ASPLOS'24）把 decode 注意力搬进 HBM 每个 bank 旁的小计算单元执行 —— 存内计算
（PIM，processing-in-memory）。本仓库在它的开源仿真器上扩展，研究 **Fugue**：多智能体
（multi-agent）、多轮（multi-round）场景下，多个请求**共享**同一份 KV 时，GPU 与 PIM 怎么分工、
KV 怎么摆、修正行（diff）放哪。

出数一律走物理事件引擎：`main.py --ablation <档> --engine dag`（`src/workload_runner.py`）。
**`--pipeopt` 默认 ON**（裁决 2026-09-04：必须常开；`--no-pipeopt` 是 AttAcc 的 serial 保守约定，会抹掉布局收益）。
解析引擎（`--engine analytic`）只作预估与交叉校验，不单独出数（裁决 2026-08-26）。

## 设计阶梯（2026-09-04 定）

**`README_design_ladder.md`** —— 六档，每档只比上一档多一件事，说到函数、参数、归约里哪一项变了：

| 档 | 角色 | 相对上一档的变化 |
|---|---|---|
| A1 | 硬件 baseline（AttAcc 原样） | — |
| A2 | 软件 baseline | 复用有了，decode 搬回 GPU，KV 在远端哑存储 |
| A4c | 布局设计 1 | decode 回 PIM；master 按 head 切片铺满 16 条；每个 head 的 diff 聚到它自己通道的几行 |
| A4d | 布局设计 2 | 各 head 的 diff 合成一段放 ch15（溢出到邻通道）；master 不变 |
| A4e | 布局设计 3（论文的 placement table） | master chunk 的通道由写入时的冲突感知表决定；diff 同 A4c |
| A5 | prefill 加速 | prefill 注意力进 PIM + MQ 批命令（建在 A4e 上）|
| A6 | 最终设计（Fugue） | prefill 逐请求动态选边 |

A3 / A3a / A3b / A4 / A4b 保留在代码里作消融，为什么淘汰见该页 §9。

## 怎么跑

- `README_run_athena_slurm.md`：athena（有 Slurm，scratch 在 `/localdata`，系统 g++ 11.4）
- `README_run_squire_local.md`：squire（无 Slurm，scratch **必须** `/data2`，cppcore 要 gcc-toolset-11）
- `RAW_DATA_MANIFEST.md`：原始数据在本机哪里（>50 MB 的事件轨迹不入库）
- `../experiments/run_layout_handcheck.sh`：一个模型 × 七档，带布局探针（`KVPIM_LAYOUT_DUMP`）
- `../workload/handcheck/gen_multiround.py`：多轮 workload 生成器（`ROUNDS` / `CHUNKS` / `CONSUMERS` / `LOUT`）
- `../output/analysis/layout_grid_csv.py`、`layout_interleave_csv.py`：布局手算 CSV（第一列物理行号，之后 ch0…15）
- `../output/analysis/layout_handcheck_theory.py`：放置规则独立重写 vs 引擎 dump 逐通道对账
- `../output/analysis/diff_gather_effect.py`：同分配器、只改 diff 落点的公平对照

```bash
python3 -m unittest discover -s tests     # 99/99
```

## 时间线与归档

- `sessions/`：每日裁决记录。**最新 `sessions/2026-09-04.md`**：`_pool_reads` 一表三用、
  `sorted()` 全知分配、master/diff 段求和坍缩三处修正；A4c / A4d 的诞生；GPU 时间的构成；
  A5/A6 改建在 A4d 上。
- `archived/`：每份都带归档说明（何时、为什么、被什么取代）。**2026-09-03 及之前的所有结果页
  都在这里，不要引用** —— 当天三处引擎修正加上 2026-09-04 的三处，让所有档的绝对值全变。
  2026-09-04 新归档：`README_data_layout_walkthrough.md`（走的是被淘汰的 A4/A4b 布局线）、
  `README_run_sweep_guide.md`（9 档 sweep 流程，阶梯已重定）。

分支：当前工作在 `chenyi-0904-test`（基于 `chenyi-0903-result`）。
