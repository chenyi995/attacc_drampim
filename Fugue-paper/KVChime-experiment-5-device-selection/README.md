# Experiment 5：四种 prefill 选边策略

统一模型：LLAMA-7B, GPT-13B, LLAMA-65B, LLAMA3.1-8B.

每张图只保留一个纵轴指标；多模型/多 case 用 AttAcc Fig. 13 式横向分组条形图。每个 `paper/` 子文件夹均包含最终 PDF/PNG、原始数据与独立 `plot.py`。

- [four-policies](paper/four-policies/README.md)

完整采样和绝对值见 `data/`。默认 A100a、NVLink 3；模型几何与 GQA 假设见每图 `models.json` 及仓库 `docs/KVChime-multi-model.md`。

## 选择误差

| Model | 样本 | 与 oracle 不同 | 平均 regret | 最大 regret |
| --- | --- | --- | --- | --- |
| LLAMA-7B | 24 | 1 | 0.30% | 7.08% |
| GPT-13B | 24 | 0 | 0.00% | 0.00% |
| LLAMA-65B | 24 | 2 | 0.15% | 1.81% |
| LLAMA3.1-8B | 24 | 3 | 26.02% | 244.60% |

Regret = (算法实际 service / 两设备实际 service 的较小值 − 1)。朴素算法只用 N=1024/4096 的两条 8-query profile 拟合每对象 scan，然后加 Q/descriptor、输出、softmax 和全量新 KV 写入；不读当前样本的实际 PIM 时间来决策。Oracle 使用独立模拟所得的实际较小值，仅作上界。对象边界、padding 和保守的写入估计会引起错误选边；这里保留全部 96 个样本，图仅选 Q=32/128/512 展示。算法伪代码见仓库 `docs/KVChime-multi-model.md`。

最差样本是 LLAMA3.1-8B、C=0、Q=8：GPU 0.825 µs，实际 MQ PIM 2.842 µs，算法只估计 0.549 µs。两点校准来自 N=1024/4096，向很短对象外推会漏估行/列粒度和 query 移动命令的成本；GQA 又放大了 query 组数。这是朴素估计器的失效场景，实际仿真时间没有被删去，也没有按 oracle 重选后改写数据。
