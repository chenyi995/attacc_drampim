# 2026-09-05：Fig. 3b 改为固定八列的行分散扫描

## 用户裁决和修改原因

chenyi9 明确要求同样读取八个 column，分别放在一行到八行；两行时均分到两行，依此类推，并确认“对，改成我这样的方法”。本次按 1、2、4、8 行落实。图内只用读者容易理解的行数、列数和扫描时间，不出现尚未定义的硬件命令或布局标签。

旧曲线在八轮到十六轮略降，因为分子 A3b 扫描周期和分母 A4e 扫描周期同时增长：1402/342 变成 2842/710，分母增长略快。这不等于扫描时间下降，但复用轮数同时改变数据量和参考布局，增加了动机图的解释负担。新图固定读取量、通道、初始状态和归一化分母，只改变八份数据占据的行数。

## 修改和数据

论文路径：`/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027`。

- `sections/03-motivation.tex`：图注、无障碍描述、Problem 2 量化段统一为同等八列扫描；保留多 agent 数据分散的上下文，不由该微基准推导真实工作流加速。未新增 citation key。
- `fig/plots/motiv/plot_motiv.py`、`fig/ver1/fig1_motiv.pdf`：右侧换成四根柱，标 `Same 8 columns`；左侧 roofline 定义保持。
- `fig/plots/motiv/experiments/04_fixed_columns/`：输入 fixture、生成/模拟/核验/导出/绘图脚本、完整原始数据、新目录复现、独立报告和验证证据。
- `data_allbank.csv`、`scan_numbers.tex`、`scan_summary.json` 和结果表从真实 stdout 自动导出；索引、旧实验 README 与旧 session 明示历史状态。
- 旧正式图、正文、脚本和输入复制到 `04_fixed_columns/archive/previous_reuse/`；02、03 历史实验原始数据继续保留。

| 触及行数 | 每行列数 | 读取总列数 | Ramulator cycles | 相对一行 |
|---|---:|---:|---:|---:|
| 1 | 8 | 8 | 62 | 1.0000× |
| 2 | 4 | 8 | 102 | 1.6452× |
| 4 | 2 | 8 | 218 | 3.5161× |
| 8 | 1 | 8 | 468 | 7.5484× |

表格由本 session 的生成脚本读取原始 CSV 产生。每个 case 实际启动一个新的 Ramulator 进程，采用同一二进制、动态库、HBM 配置和冷启动。每行内依次读需要的列，再进入下一行。输入每例恰好八条物理列扫描请求，激活和预充电由控制器自行产生，未拟合罚时。

该实验直接构造物理列地址，不声称经主程序 allocator 分配或完成某个 head 的整个 attention。采用 AttAcc 原有 `memory_system_cycles` 口径；结果只代表此内存扫描片段，不覆盖 Q、softmax/PV、链路、GPU、TBT 或 E2E。

## 验证和复现

完整命令与口径见论文 `fig/plots/motiv/experiments/04_fixed_columns/README.md`。四例真实模拟共执行两遍，第二遍在新目录运行；CSV、四组 trace、命令日志和地址映射与第一次逐字节一致。独立 agent 复核输入、实际执行顺序、相同配置、冷启动、周期和归一化，通过后留下脚本、JSON 和报告。

独立审查发现首次脚本的一个源文件归档路径不存在。已补齐实际 HBM3-PIM、controller 和 mapper 的快照与 hash，并在首次 manifest 的 `review_source_supplement` 说明；首次运行脚本原样保留。当前脚本修正路径且缺源会报错，新目录复现使用修正后的归档逻辑。模拟输入、二进制和原始结果没有改变；源码快照也不被用来冒充二进制编译 revision。

论文编译命令：

```bash
make BUILD=/tmp/fugue-fixed-columns-build TEXBIN=/data2/chenyi9/TinyTeX/bin/x86_64-linux
```

TinyTeX 编译通过，共 17 页，最终日志没有 warning、未定义引用或 overfull。第 4 页已渲染并人工查看，图内四柱、图注和正文数值一致且无重叠；证据见实验目录 `validation/`。最终安装使用修改前 SHA256 防止覆盖同期论文改动，安装清单和收据位于模拟器 `audit/2026-09-05/archive/fig3b_fixed_columns_revision/`。

本 session 未改动模拟器实现，未 commit 或 push；工作区中其他执行者正在进行的代码修改不属于本次换图。数据由 Codex 按用户授权执行，不标为用户本人手跑确认。运行时具体源码、二进制及动态库来源以原始 manifest 的 hash 为准。
