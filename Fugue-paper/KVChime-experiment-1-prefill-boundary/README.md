# Experiment 1：Prefill attention 的 GPU/PIM/MQ 边界

统一模型：LLAMA-7B, GPT-13B, LLAMA-65B, LLAMA3.1-8B.

每张图只保留一个纵轴指标；多模型/多 case 用 AttAcc Fig. 13 式横向分组条形图。每个 `paper/` 子文件夹均包含最终 PDF/PNG、原始数据与独立 `plot.py`。

- [LLAMA-7B-C0-link300](paper/LLAMA-7B-C0-link300/README.md)
- [LLAMA-7B-C1024-link300](paper/LLAMA-7B-C1024-link300/README.md)
- [LLAMA-7B-C1024-link32](paper/LLAMA-7B-C1024-link32/README.md)
- [LLAMA-7B-C1024-link450](paper/LLAMA-7B-C1024-link450/README.md)
- [GPT-13B-C0-link300](paper/GPT-13B-C0-link300/README.md)
- [GPT-13B-C1024-link300](paper/GPT-13B-C1024-link300/README.md)
- [GPT-13B-C1024-link32](paper/GPT-13B-C1024-link32/README.md)
- [GPT-13B-C1024-link450](paper/GPT-13B-C1024-link450/README.md)
- [LLAMA-65B-C0-link300](paper/LLAMA-65B-C0-link300/README.md)
- [LLAMA-65B-C1024-link300](paper/LLAMA-65B-C1024-link300/README.md)
- [LLAMA-65B-C1024-link32](paper/LLAMA-65B-C1024-link32/README.md)
- [LLAMA-65B-C1024-link450](paper/LLAMA-65B-C1024-link450/README.md)
- [LLAMA3.1-8B-C0-link300](paper/LLAMA3.1-8B-C0-link300/README.md)
- [LLAMA3.1-8B-C1024-link300](paper/LLAMA3.1-8B-C1024-link300/README.md)
- [LLAMA3.1-8B-C1024-link32](paper/LLAMA3.1-8B-C1024-link32/README.md)
- [LLAMA3.1-8B-C1024-link450](paper/LLAMA3.1-8B-C1024-link450/README.md)

完整采样和绝对值见 `data/`。默认 A100a、NVLink 3；模型几何与 GQA 假设见每图 `models.json` 及仓库 `docs/KVChime-multi-model.md`。

## 交点与成本

| Model | C | GB/s 单向 | PIM/GPU Q 区间 | MQ/GPU Q 区间 |
| --- | --- | --- | --- | --- |
| LLAMA-7B | 0 | 300 | 无交点；GPU 较快 | 无交点；GPU 较快 |
| LLAMA-7B | 1024 | 300 | 48–64 | 512–768 |
| LLAMA-7B | 1024 | 32 | 192–256 | 512–768 |
| LLAMA-7B | 1024 | 450 | 32–48 | 512–768 |
| GPT-13B | 0 | 300 | 无交点；GPU 较快 | 无交点；GPU 较快 |
| GPT-13B | 1024 | 300 | 48–64 | 512–768 |
| GPT-13B | 1024 | 32 | 192–256 | 512–768 |
| GPT-13B | 1024 | 450 | 48–64 | 512–768 |
| LLAMA-65B | 0 | 300 | 无交点；GPU 较快 | 1024–1536, 1536–2048 |
| LLAMA-65B | 1024 | 300 | 32–48 | 无交点；PIM 较快 |
| LLAMA-65B | 1024 | 32 | 128–192 | 512–768 |
| LLAMA-65B | 1024 | 450 | 24–32 | 无交点；PIM 较快 |
| LLAMA3.1-8B | 0 | 300 | 无交点；GPU 较快 | 无交点；GPU 较快 |
| LLAMA3.1-8B | 1024 | 300 | 4–8 | 48–64 |
| LLAMA3.1-8B | 1024 | 32 | 24–32 | 128–192 |
| LLAMA3.1-8B | 1024 | 450 | 4–8 | 48–64 |

GPU service = max(GPU QK + softmax + PV + cached KV 读回, 新 KV 写入)；PIM service = Q 输入 + QK/PV scan + softmax + 输出返回 + max(0, 新 KV 写入 − overlap 窗口)。窗口取第一组 scan 启动到首次消费新 K 的原生命令时间；C=0 时窗口为零。

小 Q 的 GPU 成本可能主要是旧 KV 读回；普通 PIM 的逐 query 扫描随 Q 增长，MQ 让一列读取服务最多 8 个 query，推迟 PIM 算力成为限制的区间。Q 较大时 GPU 的矩阵吞吐可能占优，因此出现交点。缓存大小同时改变计算和读取，但 padding、固定传输项及设备带宽不同，交点不保证与 C 无关；C=0 没有旧 KV 回读项，也不保证存在 PIM 优势区。范围是实际相邻采样，不能解释为已逐整数测量。
