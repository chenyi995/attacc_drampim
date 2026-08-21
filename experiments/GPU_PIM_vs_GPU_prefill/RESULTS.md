# GPU+PIM 协同 prefill（A6）什么时候比纯 GPU prefill（A4）快

实验：同一个 8192-token 的共享 prompt 被第二次 prefill（K/V 已在 AttAcc HBM 里，只差一个位置偏移，没有新 query）。比较第二次 prefill 的延迟：A4 = 纯 GPU（把复用 K/V 经链路读回 GPU 再算），A6 = 协同（PIM 就地扫描复用 K/V，GPU 只算新行）。GPU 按 FlashAttention 建模，bank PIM 用 Ramulator 计时。

**拐点** = 两者延迟相等的重算量；重算量小于拐点 → 协同更快，大于 → 纯 GPU 更快。

## 1. CacheBlend：协同更快所要求的 recompute ratio r 上限（r ≤ 表中值）

| 模型 | NVLink3（600 GB/s） | PCIe4（64 GB/s） |
|---|---:|---:|
| LLAMA-7B | r ≤ 0.39 % | r ≤ 1.18 % |
| LLAMA-65B | r ≤ 0.55 % | r ≤ 1.92 % |
| GPT-175B | r ≤ 0.55 % | r ≤ 2.69 % |

CacheBlend 论文默认 r = 15 %，高出这些上限 1–2 个数量级，所以 CacheBlend 下协同 prefill 总是比纯 GPU 慢。（r 上限 = 拐点重算行数 / 8192；上下文越长，r 上限按比例越小。）

## 2. EPIC：协同更快所要求的重算 token 总数上限（每段重算 p 个 × 复用段数 ≤ 表中值）

| 模型 | NVLink3（600 GB/s） | PCIe4（64 GB/s） |
|---|---:|---:|
| LLAMA-7B | ≤ 23 个 token | ≤ 90 个 token |
| LLAMA-65B | ≤ 30 个 token | ≤ 145 个 token |
| GPT-175B | ≤ 36 个 token | ≤ 214 个 token |

EPIC 默认每段重算 1 个 token：单段时远在上限内，协同更快（NVLink3 下快 1.3–4×，PCIe4 下快 2–11×，上下文越长优势越大）；10 个复用段时每段最多重算 2–3 个（NVLink3）或 9–21 个（PCIe4）。这个 token 数上限基本不随上下文长度变化（2k–32k 下都在 20–25 / 90–220 附近）。

如果这一轮还有新追加的 query，新 query 的 token 数要从上限里扣掉（它们同样要对复用 K/V 做 attention）。

---
数据来源、网格点、夹逼区间、理论核对、上下文长度扫描等全部细节在 `RESULTS_details.md`；原始 JSON 在 `results/`。
