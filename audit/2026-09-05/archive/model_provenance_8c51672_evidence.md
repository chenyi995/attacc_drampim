> 历史记录：保留当时的技术证据和评价，当前待审事项统一看 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md)。

# 模型复审的轻量验证数据

来源：`model_provenance_8c51672_probe.txt` 自动生成；设备成本为桩，不能用作性能数据。

| 档次 | MHA 控制组 | GQA | STORE 全零 |
|---|---|---|---|
| A1 | ok | WorkloadValidationError: CacheBlend KV link byte count is invalid | True |
| A2 | ok | ok:  | True |
| A3b | ok | WorkloadValidationError: CacheBlend KV link byte count is invalid | True |
| A4c | ok | WorkloadValidationError: CacheBlend KV link byte count is invalid | True |
| A4e | ok | WorkloadValidationError: CacheBlend KV link byte count is invalid | True |
| A5 | ok | WorkloadValidationError: CacheBlend KV link byte count is invalid | True |
| A6 | ok | WorkloadValidationError: CacheBlend KV link byte count is invalid | True |

| 模型 | QKV 宽度 | K/V 单向宽度 | 4 行 prefill KV bytes |
|---|---:|---:|---:|
| LLAMA-7B | 12288 | 4096 | 65536 |
| LLAMA3-8B | 6144 | 1024 | 16384 |

| A3b GQA 事件 | 实际 bytes/row | 应有 bytes/row | 倍率 |
|---|---:|---:|---:|
| kv_gpu_to_pim | 4096.0 | 4096.0 | 1.0 |
| decode_kv_gpu_to_pim | 16384.0 | 4096.0 | 4.0 |

| 检查 | 输出 |
|---|---|
| GQA MAC group 能量倍率 | `4.0` |
| 只改共享库后 fingerprint 不变 | `true` |
| 只改 bank generator 后 fingerprint 改变 | `true` |
| 原 extent / 改 K+V row / 只改 V row 的累计模拟桩调用 | `[1, 2, 2]` |
| 连续 scan 传给模拟桩的实际 L | `257` |
| TBT 请求均值 / 最大值 / step 加权 | `[1.5, 2.0, 1.1818181818181819]` |
| 理论 step 加权 | `1.1818181818181819` |
| tier CSV 实际档次 | `["A3b"]` |
| tier CSV 输入档次 | `["A2", "A3b"]` |
| tier_total_s / cum_end_s | `[["2.0", "10.0"]]` |
