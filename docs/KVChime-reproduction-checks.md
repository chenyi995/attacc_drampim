# KVChime 多模型实验完成记录

2026-09-07 已完成本轮仿真结果的最终校验和目录整理，并按用户原定方案恢复 Experiment 1 的 2×3 六联图：四个模型各一张，后续十张 performance/capacity 图仍各有一个响应轴。正式入口为 [Fugue-paper](../Fugue-paper/README.md)，机器可读记录为 [finalization.json](../Fugue-paper/provenance/finalization.json) 和 [KVChime-verification.json](../Fugue-paper/KVChime-verification.json)。

## 本轮运行与收尾

原始运行目录是 `output/KVChime-multi-model-TP4-20260907`。以下时间均为 2026-09-07、America/Los_Angeles（PDT）：

| 时间 | 已完成工作 |
| --- | --- |
| 11:58 | 四模型 Q/cache/link 扫描，8,320 行结果 |
| 12:27 | 32 个模型/workload 组合、160 行 F0–F4 结果；48 行共享 MQ 和 96 行选边结果 |
| 12:28 | 第一轮 26 张图的独立重画校验 |
| 12:42 | 更新绘图后的 26 个图包 |
| 12:59 | 最新图包校验、正式目录整理、旧结果归档和回归检查全部通过 |

上表的 26 张图是版式修正前的历史记录。当前图包为 4 张 Experiment 1 六联图加 10 张后续单轴图，共 14 个 PDF/PNG 图包。Experiment 1 的六个配置和位置见 [实验说明](../Fugue-paper/KVChime-experiment-1-prefill-boundary/README.md)；当前校验检查每张六联图的 a–f 顺序、模式、cache 和 link 配置。

本次收尾使用已经完成的 TP4 仿真结果，没有重新执行全量仿真。七个测量模块的 SHA-256 与该轮 sweep/workload 阶段记录一致；绘图和校验模块按更新后的版本重新检查。正式目录与运行目录的比较验证了整理过程没有改变结果，不能把它称为第二次独立全量仿真。

## 校验结果

- 当前 14 个图包在临时目录中仅使用 CSV、配置和独立绘图脚本重画，PNG 全部逐字节一致；Experiment 1 每张六个面板，后续图每张一个响应轴。
- 四张主结果表共核对 238,384 个数值字段，最大相对差为 0。表分别为 `summary.csv`、`sweep.csv`、`selection.csv` 和 `shared-mq.csv`。
- Experiment 1 图包文件与最新重画输出逐字节一致，后续十张图的正式图包保留版式修正前的文件字节；原始仿真表没有修改。完整事件表的 gzip 解压字节校验和其余归档检查见 [上一轮完成记录](../Fugue-paper/provenance/finalization-before-six-panel.json)。
- 模型几何、GQA 算术/容量、TP4 容量、共享替换视图、MQ 和校准选择器的 12 项测试通过。
- 旧版 106 个结果/来源文件与前一提交逐字节一致；迁移只调整 README 导航和清单。历史重画命令生成的 13 张 PNG 与归档原图一致。
- 当前及历史目录清单、仓库文档的本地链接检查通过。

复核本次保留的运行结果：

```bash
python3 -m fugue all-models-verify --jobs 8 --output output/KVChime-multi-model-TP4-20260907
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest fugue.kvchime_model_tests fugue.kvchime_tests
```

新克隆没有旧 `output` 时，按 [README](../README.md) 运行 `all-models`，选择一个新的输出目录。单张正式图可直接在对应 `paper/<figure>/` 中运行 `python3 plot.py --output-dir redraw`。

## 历史失败与解释边界

`output/KVChime-20260907` 是主动中止、被多模型套件替代的早期运行。`output/KVChime-multi-model-20260907` 在 LLAMA-65B、TP2、B4 上触发容量断言；其日志保留。后续 TP4 运行全部通过，这两轮无需续跑。

文档中 B2 的并发数已从误写的 4 修正为 2；输入和结果原本就是 2，没有修改仿真数据。四个模型统一为 LLAMA-7B、GPT-13B、LLAMA-65B（TP4）和 LLAMA3.1-8B（GQA），详细指标口径见 [多模型说明](KVChime-multi-model.md)。

96 个选边样本中仍保留 6 个误选；最大 regret 为 244.60%，是已完成实验的实际结果。数值 LLM 精度、RoPE 张量相似度和论文大纲中额外规划的 motivation/消融证据，不属于本次命令时序仿真已验证的范围。频率和面积证据仍由 `kvpim-rtl` 仓库负责。
