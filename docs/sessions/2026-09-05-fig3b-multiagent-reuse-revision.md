> 后续修订：正式 Fig. 3b 已按用户要求改为固定八列的行分散对照，见 [新 session](2026-09-05-fig3b-fixed-columns-revision.md)。本记录及五点曲线保留为历史。

# 2026-09-05：Fig. 3b 改为 multi-agent 复用增长的扫描开销

## 用户裁决和为什么再次修改

chenyi9 裁决：motivation 不应出现尚未解释的 ACTAB、Packed diffs / Scattered diffs；要跑一个同时有 row conflict 和跨轮修正分散的 case。主图和正文只表达复用增多时扫描 overhead 变大，构造细节与机制归因放入复现材料。

上一版单 channel 两柱只展示行局部性，且图内硬件命令让读者过早接触实现。本次改为四 agent 循环写入/复用的五点扫描曲线，仍用当前真实 allocator 和 Ramulator，不改模型时序或添加人工罚时。每轮引入不同共享块，仍有效的历史修正继续留在上下文；没有累计同一块的失效版本。

具体 fixture 和归一化由 Codex 在用户已授权的“冲突 case + 复用增长”范围内确定，不标成用户逐参数裁决。consumer 1 与全部五个轮数在第一轮测量前固定，后续没有更换或删点。测量由 Codex 按用户明确要求执行，不标成用户本人手跑确认。

## 实际结果

| 复用轮数 | A3b cycles | A4c cycles | A4e cycles | A3b / A4e |
|---|---:|---:|---:|---:|
| 1 | 206 | 206 | 206 | 1.0000× |
| 2 | 322 | 206 | 206 | 1.5631× |
| 4 | 682 | 438 | 206 | 3.3107× |
| 8 | 1402 | 902 | 342 | 4.0994× |
| 16 | 2842 | 1830 | 710 | 4.0028× |

每个格都是实跑，时间取该档所有 channel 的最大值；主图只绘 A3b / A4e，A4c 留给复现审查。八轮之后十六轮略降，正文明确两个点分别多少，未写成严格单调增长或理论渐近上限。这是一个 consumer 的共享 key 扫描，不是整体 attention 或四个 agent 的 E2E。

## 公平性和独立复核

- 同轮数三档的逻辑/物理读集、extent 长度、MAC 数相同；master 完整物理读，覆盖位置逻辑 mask。
- 写入顺序自然映射出两个热点 channel 与跨轮不同行。A4e 使用当前 co-read 软件表；没有手工指定更有利的表或合并其 diff extent。
- 各 channel 独立实例只减去 channel 基地址，row/column 不变；最终取 max，不取平均。
- Ramulator 原始命令与输入 MAC 地址逐一核对，未拟合 ACT/PRE 成本。原始二进制和实际加载动态库都有 hash。
- 独立 agent 重建所有账本，确认不同轮数的旧对象地址不变、历史 diff 有效、扫描地址和执行命令一致。证据在实验目录 independent_audit。
- 独立 agent 发现 consumer 0 的参考布局在部分列边界上与其他 consumer 不同。因此补跑其余三个 consumer，完整结果在 sensitivity；增长趋势仍成立，差异保留。图注明确只观察一个 consumer，不声称四个 consumer 完全等价。

## 文件和复现入口

论文目录：`/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027`。

- `sections/03-motivation.tex`：Problem 2 压缩为问题和增长观察，提到 row conflict / 分散更新；图注定义分子、分母和观测范围，不讲命令机制。现有 citation keys 保留，没有新增引用。
- `fig/plots/motiv/plot_motiv.py`、`fig/ver1/fig1_motiv.pdf`：右侧改为复用轮数曲线；左侧 roofline 的数据与定义保持。
- `fig/plots/motiv/experiments/03_multiagent_reuse/README.md`：完整实验指南，说明 workload、物理地址、计时、范围、运行/验证/导出/编译命令。
- 同目录 `fixture.json`、`gen.py`、`run.py`、`verify.py`、`sensitivity.py`、`export.py`、`plot.py`：生成、真实模拟、核验、其他 consumer 检查和导出。
- 同目录 `raw/`、`reproduction/`、`sensitivity/`：全部原始 trace、配置、命令日志、结果、源文件快照和来源。数值宏、CSV、Markdown 表均由脚本产生。
- 同目录 `archive/previous_two_bar/`：原正式图、正文与输入备份；02 实验仍保留，更早未对齐案例也不删除。
- 当前图索引、实验索引、历史实验提示与 outline 中旧实验提示同步，避免读者混用旧倍率。

## 验证结果

在新目录重新生成与重跑，全部扫描 trace 和结果 CSV 与首次一致。TinyTeX 编译成功，17 页；Fig. 3 所在第 4 页已渲染检查，图中文字和正文数值一致，无新增未定义引用或溢出版面。构建命令：

```bash
make BUILD=/tmp/fugue-fig3-reuse-build TEXBIN=/data2/chenyi9/TinyTeX/bin/x86_64-linux
```

复现命令以实验 README 为准。validation 保存编译日志、页面图与源码变更脚本。模拟器实现未修改，没有 commit 或 push。论文目录在当前可写范围之外，最终安装按已授权范围通过文件系统提权执行，并用修改前 hash 防止覆盖同期改动。

运行模拟器 revision：`167fe08e608b89e32f57402953314ced0b1194c6`。独立审查覆盖 `15` 个布局点和 `50` 次 channel 运行。
