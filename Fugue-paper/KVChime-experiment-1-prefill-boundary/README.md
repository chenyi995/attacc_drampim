# Experiment 1：Prefill attention 的 GPU/PIM/MQ 边界

统一模型：LLAMA-7B, GPT-13B, LLAMA-65B, LLAMA3.1-8B.

每个模型一张原定的 2×3 六联图，共四张图、24 个面板；保留原 Experiment 1 的留白、配色与分面顺序。每个 `paper/` 子文件夹均包含最终 PDF/PNG、原始数据与独立 `plot.py`。

- [LLAMA-7B-six-panel](paper/LLAMA-7B-six-panel/README.md)
- [GPT-13B-six-panel](paper/GPT-13B-six-panel/README.md)
- [LLAMA-65B-six-panel](paper/LLAMA-65B-six-panel/README.md)
- [LLAMA3.1-8B-six-panel](paper/LLAMA3.1-8B-six-panel/README.md)

完整采样和绝对值见 `data/`。默认 A100a、NVLink 3；模型几何与 GQA 假设见每图 `models.json` 及仓库 `docs/KVChime-multi-model.md`。

## 六个面板与交点

上排 a/b/c、下排 d/e/f，每格保留 GPU 与指定 PIM 的两条曲线。默认窗口：C=0 使用全部 Q；b 聚焦 Q=16–192；c/e/f 聚焦 Q=192–1536。若某模型交点超出窗口，自动扩展以显示全部交点区间。

| Model | 面板 | PIM 模式 | C | GB/s 单向 | 交点 Q 区间 |
| --- | --- | --- | --- | --- | --- |
| LLAMA-7B | a | 普通 PIM | 0 | 300 | 无交点；GPU 较快 |
| LLAMA-7B | b | 普通 PIM | 1024 | 300 | 48–64 |
| LLAMA-7B | c | MQ PIM | 1024 | 450 | 512–768 |
| LLAMA-7B | d | MQ PIM | 0 | 300 | 无交点；GPU 较快 |
| LLAMA-7B | e | MQ PIM | 1024 | 300 | 512–768 |
| LLAMA-7B | f | MQ PIM | 1024 | 32 | 512–768 |
| GPT-13B | a | 普通 PIM | 0 | 300 | 无交点；GPU 较快 |
| GPT-13B | b | 普通 PIM | 1024 | 300 | 48–64 |
| GPT-13B | c | MQ PIM | 1024 | 450 | 512–768 |
| GPT-13B | d | MQ PIM | 0 | 300 | 无交点；GPU 较快 |
| GPT-13B | e | MQ PIM | 1024 | 300 | 512–768 |
| GPT-13B | f | MQ PIM | 1024 | 32 | 512–768 |
| LLAMA-65B | a | 普通 PIM | 0 | 300 | 无交点；GPU 较快 |
| LLAMA-65B | b | 普通 PIM | 1024 | 300 | 32–48 |
| LLAMA-65B | c | MQ PIM | 1024 | 450 | 无交点；PIM 较快 |
| LLAMA-65B | d | MQ PIM | 0 | 300 | 1024–1536, 1536–2048 |
| LLAMA-65B | e | MQ PIM | 1024 | 300 | 无交点；PIM 较快 |
| LLAMA-65B | f | MQ PIM | 1024 | 32 | 512–768 |
| LLAMA3.1-8B | a | 普通 PIM | 0 | 300 | 无交点；GPU 较快 |
| LLAMA3.1-8B | b | 普通 PIM | 1024 | 300 | 4–8 |
| LLAMA3.1-8B | c | MQ PIM | 1024 | 450 | 48–64 |
| LLAMA3.1-8B | d | MQ PIM | 0 | 300 | 无交点；GPU 较快 |
| LLAMA3.1-8B | e | MQ PIM | 1024 | 300 | 48–64 |
| LLAMA3.1-8B | f | MQ PIM | 1024 | 32 | 128–192 |

GPU service = max(GPU QK + softmax + PV + cached KV 读回, 新 KV 写入)；PIM service = Q 输入 + QK/PV scan + softmax + 输出返回 + max(0, 新 KV 写入 − overlap 窗口)。窗口取第一组 scan 启动到首次消费新 K 的原生命令时间；C=0 时窗口为零。

小 Q 的 GPU 成本可能主要是旧 KV 读回；普通 PIM 的逐 query 扫描随 Q 增长，MQ 让一列读取服务最多 8 个 query，推迟 PIM 算力成为限制的区间。Q 较大时 GPU 的矩阵吞吐可能占优，因此出现交点。缓存大小同时改变计算和读取，但 padding、固定传输项及设备带宽不同，交点不保证与 C 无关；C=0 没有旧 KV 回读项，也不保证存在 PIM 优势区。范围是实际相邻采样，不能解释为已逐整数测量。
