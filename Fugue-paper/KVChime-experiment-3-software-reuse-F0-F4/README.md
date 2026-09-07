# Experiment 3：CacheBlend/EPIC 的 F0–F4

统一模型：LLAMA-7B, GPT-13B, LLAMA-65B, LLAMA3.1-8B.

每张图只保留一个纵轴指标；多模型/多 case 用 AttAcc Fig. 13 式横向分组条形图。每个 `paper/` 子文件夹均包含最终 PDF/PNG、原始数据与独立 `plot.py`。

- [TTFT_ms](paper/TTFT_ms/README.md)
- [TBT_ms](paper/TBT_ms/README.md)
- [E2E_ms](paper/E2E_ms/README.md)
- [simultaneous_peak_KV_GiB](paper/simultaneous_peak_KV_GiB/README.md)
- [decode_scan_ms_per_token](paper/decode_scan_ms_per_token/README.md)

完整采样和绝对值见 `data/`。默认 A100a、NVLink 3；模型几何与 GQA 假设见每图 `models.json` 及仓库 `docs/KVChime-multi-model.md`。

能耗采用 AttAcc 原生动态能耗口径，F4 相对 F0 的 E2E 能耗降低范围为 68.2%–95.6%（负值为增加）；完整数值保留在 `data/summary.csv`，不单独绘图。

## Workload、对照与收益

八种输入分别是 CacheBlend demo 2/3、EPIC LongContext 4K/8K/16K，以及同一 8K 文档被 1/2/4 个已就绪 reader 复用。准确 token 数、chunk 顺序、重算位置和输出截断见 `data/workload.json`；逐模型复用冻结形状，不重新分词，也不测量 LLM 输出精度。

F0 全 GPU 重算；F1 软件 reuse + GPU attention/decode；F2 软件 reuse + GPU 拼出完整私有 KV 后写回原生 PIM decode；F3 只存私有重算/生成 KV、GPU prefill、共享视图单查询 PIM decode；F4 保持共享视图，增加 MQ decode 与估计成本 prefill 选边。

性能柱状图使用加速比以展示各档位差距，容量图使用相对峰值占用；每张图只表示一个指标。下表给出每个模型八个 workload 的降低比例范围，负值就是衰退，没有删去；逐 case 的 F0/F1 等全部比较在 `data/reductions.csv`。

| Model | 比较 | TTFT 降低 | TBT 降低 | E2E 降低 | KV 峰值降低 |
| --- | --- | --- | --- | --- | --- |
| LLAMA-7B | F2 → F3 | 7.5%–28.8% | -15.3%–-0.4% | 4.4%–14.1% | 25.8%–76.2% |
| LLAMA-7B | F3 → F4 | -2.7%–28.4% | -0.0%–14.9% | 0.0%–6.9% | 0.0%–9.6% |
| GPT-13B | F2 → F3 | 7.5%–28.8% | -11.5%–-0.3% | 4.2%–14.7% | 26.2%–76.5% |
| GPT-13B | F3 → F4 | 0.0%–28.7% | -0.0%–11.5% | 0.0%–6.8% | 0.0%–7.7% |
| LLAMA-65B | F2 → F3 | 3.9%–22.5% | -18.1%–-0.3% | 1.9%–5.6% | 27.0%–77.3% |
| LLAMA-65B | F3 → F4 | 0.0%–16.6% | -0.0%–12.5% | 0.0%–7.3% | 0.0%–1.1% |
| LLAMA3.1-8B | F2 → F3 | 1.2%–10.3% | -71.4%–-1.8% | -27.0%–0.8% | 25.8%–76.2% |
| LLAMA3.1-8B | F3 → F4 | 0.0%–0.0% | 7.9%–40.5% | 2.6%–25.0% | 0.0%–0.0% |

TTFT/TBT/E2E 包含正常 Transformer 层的 projection、FFN 和 attention 服务，因此 scan 下降不等于 TBT 按同样比例下降。容量取同时刻 GPU + remote pool + private 的总峰值；完整分项和 snapshot 在 `data/storage.csv`。完整事件表以无损 `data/events.csv.gz` 保存，解压字节哈希已核对原始 CSV；共享池预热单列 `data/warmup.csv`，F2 导出 overlap 对照单列 `data/F2-overlap-control.csv`。共享块中已失效的旧项仍可能被物理扫描，且共享视图带来分段和额外 Q/descriptor 成本，所以 F3 的 scan/TBT 不保证比 F2 小；它主要减少完整物化和私有副本。F1→F3 不是独立 RoPE kernel 消融，旋转算术沿用 native 未单列计时的范围。
