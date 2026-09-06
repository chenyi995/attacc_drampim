# workload/probe：运行协议的 workload

**协议只跑一个 workload 家族：`sweep/W1_turns.json`（baseline，七个 combo）和它的一轴 sweep（`sweep/W1_S*_turns.json`，只跑 A3b 与 A6）。**
怎么跑见 `docs/README_run_protocol.md`。

## W1：main agent 每轮总结它的 worker

`gen_main_workers.py` 生成。一个会话 = 一个 main agent + `workers` 个 worker；`sessions` 个会话并排跑，共用设备和 decode batch。

- `a0_corpus`：语料 owner，一次导入全部文档（每篇 256 token = 一个 DRAM 行），是共享的 master。
- worker w 第 r 轮：读**新**文档 `r*workers + w`（从 owner 复用，偏移不同，产生 k 个修正行），写 16 token 笔记，答 `worker_lout` token。
  上下文每轮重列它之前读过、写过的一切，所以早先轮次的修正被继承，不重算。
- main 第 r 轮（r ≥ 2）：总结 worker 第 r−2 轮的回答——`workers` 个回答块作为复用段进入 main 的上下文（偏移不同，每轮新增 `workers × k` 个修正），
  加 16 token 指令，答 `main_lout` token。早先轮次的修正全部继承。
- 两个会话的同号 worker 每轮读同一篇文档：decode batch 里有共读行（MQ 的材料），放置表看到跨会话共读。

为什么是 r−2：loader 把 (tier, id) 排序中第一个列出某指纹的请求当作 owner；worker 自己的下一轮（tier r−1）以 `parent_out` 列出它的回答，
main 必须在更晚的 tier 引用，否则会被当成写者、得不到修正。

每一档拿到什么（结构探针 `output/analysis/b1_levers.py`，`LEVERS_HEADS_PER_HBM=2` = 每 KV head 8 通道）：

| 相邻档 | 机制 | W1 上的结构杠杆 |
|---|---|---|
| A3b → A4c | main 每轮的修正落在朴素写入流的不同行（断开的 diff），紧凑的 diff 行把它们收拢，diff 行在 head 的通道上轮转 | 修正行 3412 → 940（少 72%）；最忙 lane 的 DRAM 行 A3b 1704 → A4c 1569 |
| A4c → A4e | worker 的回答被 main 共读，表把它们分到不同通道；main 的修正被表分成一组、放到 main 所读行最少的通道 | 最忙 lane 行数少 34%；最忙 lane 的 DRAM 行 1569 → 957 |
| A4e → A5 | 每一轮都是 decode 形状（m 为几十、上下文几千）；MQ 合并 batch 里两个会话的共读 sweep | 两会话同号 worker 的文档历史相同 |
| A5 → A6 | 语料导入是唯一的大 fresh prefill，A6 留在 GPU | owner 12288 token 的导入 |

参数：`rounds=24 workers=2 sessions=2 worker_lout=128 main_lout=128 doc_tokens=256`，145 个请求，main 末轮上下文 9k。

## sweep 轴（各动一个变量，跑 A3b + A6）

| 轴 | 文件 | 值 | 动什么 |
|---|---|---|---|
| S1 轮数 | `W1_S1_rounds_{12,48}` | 12, 48 | 断开的 diff 积累多少 |
| S2 每 main 的 worker 数 | `W1_S2_workers_{1,4}` | 1, 4 | main 每轮的修正量、共读宽度 |
| S3 会话数 | `W1_S3_sessions_{1,4}` | 1, 4 | 每 tier 就绪请求数（batch、MQ 共享） |
| S4 worker 回答长度 | `W1_S4_worker_lout_{32,256}` | 32, 256 | main 共读的块大小 |
| S5 main 回答长度 | `W1_S5_main_lout_{32,512}` | 32, 512 | decode 在 E2E 里的份额 |
| S6 文档长度 | `W1_S6_doc_tokens_{512,1024}` | 512, 1024 | 每轮驻留上下文 |

`manifest.csv` 列出每个文件的参数、请求数、prefill/decode token 数；`experiments/extract_protocol.py` 按它出表。
重新生成：`python3 workload/probe/gen_main_workers.py --all workload/probe/sweep`；单个变体：`gen_main_workers.py --rounds 24 --workers 2 --sessions 2 > wl.json`。

## 其他目录

- `archive/2026-09-05-C-protocol/`：9-05 的 C1/C2 baseline 与 C1_S* sweep（多 agent RAG 会话）及其生成器 `gen_sweep.py`，已退役；
  TINY 上的结果在 session 记录 §20、§25。
- `targeted/`：另一会话 9-05 的受控设计（D 持续汇总、E 周期共读、P 低 query）与手算，不在协议里。
- `archive/2026-09-05-probes/`：9-05 之前的探针生成器与 workload（A6 分流、多轮）。

结果目录在 /data2 的 scratch 里，不进仓库。
