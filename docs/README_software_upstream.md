# 软件上游:chunk 复用 + 选择性重算的策略族(`--reuse`)

背景一句话:跨请求复用 KV chunk(位置无关缓存, position-independent
caching)后,chunk 边界处的交叉注意力是错的,主流软件方案的分歧点只在
**"重算哪些 token 来修正、按什么规则选"**。本仓库把这一维做成可插拔的
策略族,让放置消融(A 系列)在不同软件上游下都能跑。

## 1. 已实装的六个策略(`src/workload.py`)

两族共享锚定策略的计划机制(`CACHEBLEND_FAMILY` / `EPIC_FAMILY`):

| 策略 | 族 | 重算规则 | 旋钮 | 出处 |
|---|---|---|---|---|
| `no-reuse` | — | 无复用(参照) | — | AttAcc 原样 |
| `cacheblend` | ratio 族 | 指定选择层全重算,partial 层按偏差采样 r 比例的行(仿真中为按种子均匀采样) | `--cacheblend-recompute-ratio/-full-layers/-partial-layers` | EuroSys'25(arXiv:2405.16444) |
| `cachetune` | ratio 族 | 同 r 比例,但**离线选行**(FFT 低频能量式):无全重算选择层,不付在线选择代价 | 同上,full-layers 必须为空 | arXiv:2605.24022 风格 |
| `epic` | prefix 族 | 每个**位移过**的复用段重算前 k 个 token(边界修正) | `--epic-prefix-recompute-tokens` | ICML'25(arXiv:2410.15332,算法名 LegoLink) |
| `promptcache` | prefix 族 | **零重算**,chunk 原样复用(精度换速度的端点基线) | — | MLSys'24(arXiv:2311.04934) |
| `cachecraft` | prefix 族 | 前缀长度逐 chunk 变化:k_i = ceil(α·(1−overlap)·长度),overlap = 消费者与属主前文指纹集的 Jaccard 相似度;位移边界至少 1 行 | `--cachecraft-alpha` | SIGMOD'25 风格(arXiv:2502.15734) |

实现要点:prefix 族的重算行落在每个复用决策的
`ReuseDecision.epic_prefix_rows`(校验器只要求"前导前缀"),ratio 族落在
`ReusePlan.cacheblend_partial_rows`;下游(解析与物理两条路径)一律按
**族**分支,不再逐策略特判。

## 2. 文献地图(2026-08-24 调研,供扩展与对照)

同族可继续加的(与 cacheblend 同代价结构、只换选择评分):ProphetKV
(查询相关性,arXiv:2602.02579)、CacheClip(辅助小模型注意力、选择
挪出关键路径,arXiv:2510.10129)、A³(query→doc 注意力,
arXiv:2511.17560)、QCFuse(关键层注意力,arXiv:2604.08585)、KVShare
(语义差分编辑 + 双阶段高偏差,arXiv:2503.16525)。

与 epic 同代价结构(每段前缀/块):MPIC(多模态,每图像 chunk 前 k,
arXiv:2502.01960)、MEPIC(前缀量化到 KV page 粒度,arXiv:2512.16822)、
KVLink(可训练 link token,arXiv:2502.16002)。

零重算近族(r=0 基线,修正走别的机制):APE(注意力温度重标定,
ICLR'25)、TurboRAG/Block-Attention(微调适配)、KVCOMM(锚点偏移
近似,NeurIPS'25)、CacheFocus(重定位+剪枝)。基础设施向:CacheGen
(KV 压缩流送)、RAGCache(多级缓存)、HCache(存 hidden state、只重算
KV 投影——恢复代价建模可借鉴)。

**实验口径裁决(chenyi9 2026-08-25,当日晚更正)**:上游软件**不构成
实验轴**——仿真器的代价只由重算 token 的**数量**决定,选择算法选了哪些
token 本来就不进模型,所以"换选择规则对比"没有信息量,原 A6 选择变体扫
已撤销。策略实现保留,仅作为配置重算数量/结构的入口(cacheblend 族给
按层比例,epic 族给每段前缀数)。历史记录:矩阵曾只用**保证重算**的
选择族——每个成员都重算一部分 token,彼此只差"选哪些"(cacheblend =
偏差采样 r 比例;**EPIC 算该族的特例:选每个位移段的前 k 个**;
cachecraft = 重叠度缩放前缀;cachetune = 离线选行 r 比例);零重算端点
(promptcache)保留实现、**不进实验矩阵**。矩阵见
`experiments/paper_ladder/`。

选型理由:promptcache/cachecraft/cachetune 三个先进仓库,是因为它们在
**仿真器可见的代价结构**上与两个锚各有一处真实差异(零修正端点/逐 chunk
变长前缀/无在线选择层),而不只是精度曲线不同;精度维不在本仿真器口径
内,论文引用时须注明。

## 3. 怎么跑

```bash
# 三个新策略各过一遍 A6(小模型冒烟)
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json --history-len 3 \
  --ablation A6 --pipeopt --reuse promptcache --workload-report /tmp/p.json
python3 main.py ... --reuse cachecraft --cachecraft-alpha 0.1 ...
python3 main.py ... --reuse cachetune --cacheblend-recompute-ratio 0.15 \
  --cacheblend-partial-layers 0,1,2 ...
```

单测:`test_software_upstream_policy_family_enrichment`(零前缀/α 单调/
cachetune 拒绝选择层)。
