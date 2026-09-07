# GPT-13B-C1024-link450

本图只表示 **Attention service (µs / layer)**。交点标注为相邻采样 Q 的区间；没有交点时不制造交点。

[PDF](figure.pdf) · [PNG](figure.png) · [原始结果](raw-data.csv) · [绘图脚本](plot.py)

`raw-data.csv` 是未归一化的原始结果行；`raw-profiles.csv` 是该实验的命令 profile 原值。`plot-config.json` 记录取数/分组/归一化，`models.json` 记录模型几何。全量采样在实验上层 `data/`。

```bash
python3 plot.py
# 输出到其它目录：
python3 plot.py --output-dir redraw
```

只需 Python、numpy 和 matplotlib；把本文件夹单独复制出去仍能重画，不需要仿真器代码或旧 output。
