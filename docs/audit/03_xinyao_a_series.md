# 块 03:A 系列放置消融(A1–A6)与 GPU prefill 拐点研究

归属:xinyao,commit `47ae0c3`("Placement ablation (A1-A6), refined/flash
GPU models, GPU+PIM vs GPU prefill study",2026-08-21,分支 xinyao_0821);
结果整理 `11745e1`/`9467de5`/`34d3cd7`。新增 `src/ablation.py`(1018 行)、
`src/gemm_table.py`(187 行),改 `src/devices.py`(+139)、`main.py`。

## 1. 要回答的问题(零基础)

复用场景下,注意力的两相各放哪个设备、共享 KV 在 PIM 里怎么摆,
一共有多少种合理组合?各自的时间/能耗/存储是多少?这是论文的
**放置消融 (placement ablation)**。三个开关(`src/ablation.py:65-68`):
- `prefill_attn ∈ {gpu, split, pim}`:prefill 注意力在哪算;
- `decode_attn ∈ {gpu, pim}`:decode 注意力在哪算;
- `kv_mapping ∈ {none, private, naive, master-diff}`:KV 在 PIM 的摆法
  (none=不进 PIM;private=每请求私有全份,即上游 AttAcc;naive=按软件
  chunk 布局塞一个池;master-diff=块 02 的双池)。

## 2. A1–A6 预设(`src/ablation.py:70-86`)

| 预设 | prefill | decode | KV 摆法 | 一句话 |
|---|---|---|---|---|
| A1 | gpu | pim | private | 原版 AttAcc,无复用(基线锚) |
| A2 | gpu | gpu | none | 纯 GPU 跑 CacheBlend/EPIC |
| A3 | gpu | pim | naive | 软件 prefill + PIM decode,朴素映射 |
| A4 | gpu | pim | master-diff | 同上,双池映射 |
| A5 | pim | pim | master-diff | prefill 注意力也进 PIM |
| A6 | split | pim | master-diff | GPU/PIM 协同 prefill(论文主方案) |

`resolve_config`(合法性检查在 `AblationConfig` 校验)拒绝不自洽组合
(如 decode=pim 却 kv_mapping=none),测试
`test_ablation_rejects_incoherent_placement_switches` 盯守。

## 3. 解析式代价模型:`run_ablation_report`(`src/ablation.py:943`)

与块 02 的逐事件模拟不同,A 系列是**解析式** (analytic):按 tier 分批,
每批一次性算 prefill/decode 的时间、能耗、分项 breakdown。关键函数:
- `_prefill_batch`(`:563`):一个批的 prefill。GPU 侧逐层查
  `system.model.sum_decoder`;PIM/split 侧把"扫描行数 vs GPU 行数"
  按 kv_mapping 分类;A3/A4(GPU prefill+PIM decode)时,复用行要
  **从 PIM 读回 GPU**(`prefill_kv_readback`,链路层 `kv_pim_to_gpu`)——
  这是"GPU prefill 不白吃"的关键代价项;A6 的 GPU/PIM 两支按
  `split_overlap` 取 max 而非相加(测试
  `test_split_prefill_overlaps_its_gpu_and_pim_branches`)。
- `_batch_scan_profile`(`:406`):decode 每步的 PIM 扫描画像——把批平均的
  run 长度序列变成物理 run 列表(private/naive/master-diff 三种摆法的
  run 结构不同:naive 的修正行会打碎流、master-diff 不会,测试
  `test_naive_mapping_fragments_the_scan_that_master_diff_keeps_whole`),
  喂给 Ramulator(带签名缓存)。
- `_memory_report`(`:844`):存储账——共享行存一份、diff 按层摊、
  与 no-reuse 基线的字节比。
- 回归锚:A1 必须复现上游 legacy 报告(时间/能耗逐位,链路层除外,
  测试 `test_ablation_a1_reproduces_the_original_attacc_legacy_report`)。

## 4. GPU 模型精化:`--gpu-model {legacy,refined,flash}`

纯 roofline(见块 01 §3)会高估 GPU 注意力效率,把对比拉偏向 GPU 一侧,
反而让 PIM 的收益显得虚高——所以要给 GPU 配更真实的模型。两档精化
(`src/devices.py:52/:57`):
- `refined`:GEMM 效率查表(`src/gemm_table.py`,`gemm_tflops` `:154`,
  按 m/n/k 对数插值的实测 TFLOPS(每秒 10¹² 次浮点运算)表)+
  注意力按 key 长度的效率曲线(`attention_efficiency`,`:184`);
- `flash`:FlashAttention(GPU 上按块分片 (tile)、中间矩阵不落显存的
  注意力内核)风格的显式 tile 流量/算力模型
  (`_flash_traffic` `:106`、`_flash_compute_time` `:129`)。
A 系列实验默认用 `flash`(对 GPU 最有利,结论保守)。

## 5. 实验:`experiments/GPU_PIM_vs_GPU_prefill/`

问题:**协同 prefill(A6)什么时候赢过软件 prefill(A4)?**
(A4 在 GPU 上重算,但复用行要过 GPU–PIM 互连链路读回;A6 让 PIM 就地扫。)
`run_one.sh` 一点一跑(模型×链路×policy×参数),`analyze_grid.py` 汇总。
两张拐点表(`RESULTS.md`,汇总数字亦录入 `docs/EXPERIMENTS.md`):
- EPIC:每段重算前缀 token 数 p 低于阈值 p\* 时 A6 胜;
  **p\* = 22–35(NVLink3,快互连)/ 89–210(PCIe4,慢互连)**,
  且 `34d3cd7` 证明 **p\* 几乎不随共享段长度 L 变**(results/pgridL 网格);
- CacheBlend:重算比例 r 上限 **0.4–2.7%** 时 A6 才胜。

**在论文中的意义**:这是论文"协同 prefill 何时开启"的设计规则来源——
拐点表直接支撑 §6 的选边论述;p\* 与 L 无关意味着规则可以按 policy
一次定死、不必按负载在线调。

## 6. 在论文中的意义(A 系列整体)

论文 outline 的**五级阶梯**(GPU-only / PIM-append / PIM-split /
PIM-static / Fugue)≈ **A2 / A3 / A4 / A5 / 动态选边**,AttAcc 参照=A1
(`docs/EXPERIMENTS.md` 尾节)。即:A 系列就是论文主对比的骨架;
其中第五级(按请求在 A4/A6 间动态选边,B4)**未实装**,是论文 §6
Fugue 行的已知缺口(`docs/README_design_check.md` §3.1)。

## 7. 测试覆盖与悬置

- 覆盖:A1 逐位复现、非法组合拒绝、naive vs master-diff 的 run 结构、
  A6 重叠语义(时间下降、能耗不变、A5 两设置逐位同)。
- 悬置(见 `SIM_VS_PAPER_AUDIT_0821.md`):B4 动态选边(论文 §6 的
  Fugue 行按请求在 A4/A6 间在线选择)未实装;GQA 群组、n_d≈ρfC 配比、
  行激活计数输出链路等 11 条差距,均记录在该审计与 `HANDOFF.md` §4。
