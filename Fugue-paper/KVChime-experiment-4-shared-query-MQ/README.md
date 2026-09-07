# Experiment 4：共享 query 与跨 agent MQ

统一模型：LLAMA-7B, GPT-13B, LLAMA-65B, LLAMA3.1-8B.

每张图只保留一个纵轴指标；多模型/多 case 用 AttAcc Fig. 13 式横向分组条形图。每个 `paper/` 子文件夹均包含最终 PDF/PNG、原始数据与独立 `plot.py`。

- [scan_us](paper/scan_us/README.md)
- [TBT_ms](paper/TBT_ms/README.md)

完整采样和绝对值见 `data/`。默认 A100a、NVLink 3；模型几何与 GQA 假设见每图 `models.json` 及仓库 `docs/KVChime-multi-model.md`。

## 完全共享文档的结果

| Model | Ready agents | Scan 降低 | TBT 降低 |
| --- | --- | --- | --- |
| LLAMA-7B | 1 | 0.0% | 0.0% |
| LLAMA-7B | 2 | 38.1% | 2.8% |
| LLAMA-7B | 4 | 63.1% | 8.6% |
| LLAMA-7B | 8 | 73.3% | 16.7% |
| GPT-13B | 1 | 0.0% | 0.0% |
| GPT-13B | 2 | 38.1% | 2.0% |
| GPT-13B | 4 | 63.1% | 6.4% |
| GPT-13B | 8 | 73.3% | 13.5% |
| LLAMA-65B | 1 | 0.0% | 0.0% |
| LLAMA-65B | 2 | 38.1% | 2.3% |
| LLAMA-65B | 4 | 63.1% | 7.0% |
| LLAMA-65B | 8 | 73.4% | 14.6% |
| LLAMA3.1-8B | 1 | 64.9% | 8.7% |
| LLAMA3.1-8B | 2 | 74.8% | 17.6% |
| LLAMA3.1-8B | 4 | 74.6% | 27.8% |
| LLAMA3.1-8B | 8 | 74.6% | 40.3% |

取 EPIC 4K 文档作为共享池，每个 agent 独立保留重算项和新 token。MQ 复用共享列操作数，不合并 query 的累加器或 softmax。MHA 单 agent 只有一条 Q，MQ 必须退化为相同 decode；GQA 单 agent 的同一 KV head 已服务多个 Q heads，因此可能出现 MQ 收益。额外的 0/50% 共享控制保留在图和原始数据，实际共享 token 数单列，不能把按 chunk 选出的比例误称为精确 token 比例。这里假设 agent 已就绪，不含组批等待或线上排队。
