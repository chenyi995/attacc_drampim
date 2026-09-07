# Experiment 2：MQ 对 link 与 cache 的敏感性

统一模型：LLAMA-7B, GPT-13B, LLAMA-65B, LLAMA3.1-8B.

每张图只保留一个纵轴指标；多模型/多 case 用 AttAcc Fig. 13 式横向分组条形图。每个 `paper/` 子文件夹均包含最终 PDF/PNG、原始数据与独立 `plot.py`。

- [link-sensitivity](paper/link-sensitivity/README.md)
- [cache-sensitivity](paper/cache-sensitivity/README.md)

完整采样和绝对值见 `data/`。默认 A100a、NVLink 3；模型几何与 GQA 假设见每图 `models.json` 及仓库 `docs/KVChime-multi-model.md`。

## 解读

| Model | 全网格最小 GPU/MQ | 全网格最大 GPU/MQ |
| --- | --- | --- |
| LLAMA-7B | 0.240 | 315.762 |
| GPT-13B | 0.254 | 385.467 |
| LLAMA-65B | 0.243 | 169.727 |
| LLAMA3.1-8B | 0.134 | 72.493 |

>1 表示 MQ PIM 更快，<1 表示 GPU 更快。图中的所有样本来自同一个完整 grid；link 图固定 C=1024，cache 图固定单向 300 GB/s。Link 扫描仅重新计算传输及暴露时间，复用同一形状本次生成的 native scan；没有修改 scan 以制造交点。完整 latency 各组成项见 `data/raw-sweep.csv`。
