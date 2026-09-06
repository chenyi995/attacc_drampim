# 探针 workload（2026-09-05，CACHEBLEND-TINY，1 GPU + 1 HBM）

都是"多轮多 agent 复用"的最小构造：`a0_owner` 声明共享 chunk（256 token = 1 DRAM 行），
每个 reuser 的上下文按轮交替"共享 chunk × C | 自己写的 block"，共享 chunk 有位移，每块 k=8 行修正，
修正 per-agent 私有。没有任何惩罚项。生成器：`gen_multiround.py`（xinyao 分支同名脚本的副本）、
`gen_a6.py`、`gen_a6_chat_mix.py`。

| 文件 | 构造 | 用途 | 状态（2026-09-05 04:00，legacy GPU 模型） |
|---|---|---|---|
| `wl_mr_R8C2N8_L128.json` | R=8 轮 × C=2 chunk，8 个 reuser，lout 128 | 基线七档：TBT / E2E / 能量 | 重跑完成 |
| `wl_mr_R16C2N16_L128.json` | R=16，16 个 reuser | 更多 agent / 更多轮 → 布局收益（A3b→A4c→A4e）是否增长 | 重跑完成 |
| `wl_mr_R8C2N8_L1024.json` | 同基线，lout 1024 | decode 占比大 → 布局收益进 E2E | 重跑完成 |
| `wl_a6_R8C2N8_own16_256.json` | reuser 每轮只写 16 / 256 token 两种 | 试探 A6 按"新写多少"分裂（结论：分不开，比例恒为 2） | 已跑 |
| `wl_a6_crossover.json` | + 8…1024 token 的独立新 prompt | 找选边器交叉点（legacy GPU 模型下 ≤512 GPU、≥1024 PIM） | 已跑（仅 A6） |
| `wl_a6_split.json` | 4 个长复用 agent + 32 个 512-token 独立短请求（lout 16） | 让 A6 在同一 workload 里两边都选 | 完成（A6 选边 GPU 32 / PIM 5，比 A5 多 1.0% E2E） |

跑法见 `experiments/run_queue.sh`、`experiments/run_after_queue.sh`（≤ 64 核），
内存监视 `experiments/mem_guard.sh`，汇总 `experiments/summarize_ladder.py <outdir> <wl.json> [ref]`。
`sweep/*_turns.json` 自 2026-09-05 起每轮重新列出该 agent 的全部早期上下文（无 `history_len`），后一轮继承前一轮写过的修正（C8）。
`sweep/` 是 2026-09-05 的运行协议（`docs/README_run_protocol.md`）：baseline `C1_turns.json`（混合型 agent 会话，41 个请求）与 `C2_turns.json`（纯对话型，65 个请求）跑全部七个 combo，`C1_S*_turns.json` 各动 C1 的一个变量、只跑 A3b 与 A6。`gen_sweep.py --all <dir>` 生成整套，`--preset C1|C2|B1 --agents ...` 生成单个变体；旧的 B0/S/B1/T 集合用 `LEGACY_MATRIX=1` 复现。`output/analysis/b1_levers.py` 是结构探针（`LEVERS_HEADS_PER_HBM=2` 对应每头 8 通道）。
结果目录在 /data2 的 scratch 里，不进仓库。
