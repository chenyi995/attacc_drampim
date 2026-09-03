# 手算校验：A3b–A6 的布局，理论对实测

一个**小到可以用笔算出全部布局**的 workload，跑完 A3b / A4 / A4b / A5 / A6，
把手算的逐 channel 行数、extent 数、ACT 次数，和仿真器真正吐出来的逐 channel
事件并排放。**结论：公共扫描与 private 扫描，逐格全部一致。**

- workload：`wl_handcheck.json`（本目录，脚本生成，见 §6）
- 引擎：striped-append 布局 + 真实 extent 进 Ramulator（`897c294`）
- 记账：**ACT 由 Ramulator 的行缓冲决定**，不是公式算的

> ⚠️ 这是**机制校验**，不是证据级结果。模型只有 4 层 8 head、上下文 4096 token，
> 两个 agent。不要拿这里的时间做任何论文结论。

---

## 1. 配置：为什么这些数字是圆的

| 项 | 值 |
|---|---|
| 模型 | `CACHEBLEND-TINY`（4 层，8 Q head = 8 KV head，`dhead` 128，bf16）|
| 系统 | `--ngpu 1 --num-hbm 2` |
| → 每 GPU 堆栈 | `2 // 1 = 2` |
| → **heads_per_hbm** | `ceil(8 / 2) = 4` |
| → **每个 head 的 channel 数** | `16 // 4 = 4`（A3b）；`15 // 4 = 3`（A4 的 master 池）|
| 复用 | `--reuse recompute --epic-prefix-recompute-tokens 8` |

选 `--num-hbm 2` 就是为了凑出 **heads_per_hbm = 4、每 head 4 条 channel**，
和 baseline 上的 GPT-13B 一样，四档的差别最容易看清。

## 2. workload：谁拥有、谁复用

```
A_owner   pad(256) + doc00..doc14 (15 x 256)          = 4096 token
B_reuser            doc00..doc14 (15 x 256)           = 3840 token
```

归属由 `sorted(requests, key=(tier, id))` 决定，字母序在前的先声明，所以
**`A_owner` 拥有那 15 个块，`B_reuser` 复用它们**。`A_owner` 把它们放在一个
256-token 的 pad 之后（偏移 256, 512, …），`B_reuser` 从 0 开始 —— 于是
**每一块都位置位移，各要 8 行重算**。

```
B_reuser 每个 head：15 个复用块 x 8 = 120 个 diff 行
```

尺寸是特意选的：`B_reuser` 的流是 **15 x 256 整除无余**，15 组 repair 分到
4 条 channel 正好是 **4/4/4/3**。

`--validate-workload` 确认：`(复用者 B_reuser, 拥有者 A_owner, 重算 8 行) x 15`。

## 3. 手算

一次 ACT 覆盖 **256 个 token**（AttAcc 的 all-bank 广播：一个 token 占 4 B
地址空间，一个 1024 B 的 DRAM 行装 256 个）。所以
`ACT = ceil(该 extent 行数 x 4 B / 1024 B)`，**一个 8 行的 repair 也要一次**。

真实的 decode 被拆成两次扫描，因为两个 agent 共享那 15 个块：

- **公共扫描**：15 个共享块（batch 事件），
- **private 扫描**：`B_reuser` 自己的 120 个重算行（消费者私有，不可写回共享副本）。

### 3.1 公共扫描

`A3b` 的 `shadow_reads = False`，被改写的 master 行**直接跳过**，所以每块只剩
248 行：`15 x 248 = 3720` → unit 切成 `[256] x 14 + [136]`，共 15 个。
`A4/A4b` 的 `shadow_reads = True`，陈旧副本照读再掩掉，所以是 `15 x 256 = 3840`
→ 15 个满 unit。

| 档 | unit → channel | 手算逐 channel（一个 head 的 4 条）|
|---|---|---|
| **A3b** | `base + (u % 4)` | ch0 = 单元 0,4,8,12 = **4x256 = 1024**；ch1 同 = 1024；ch2 = 单元 2,6,10,14 = 3x256+136 = **904**；ch3 = 单元 3,7,11 = **768** |
| **A4** | `base_m + (u % 3)`，`base_m = 3h % 15` | 每条 = 单元 5 个 x 256 = **1280**；head 占 ch0-2 / ch3-5 / ch6-8 / ch9-11 → **ch12,13,14 闲置** |
| **A4b** | 全局表 `(h x 15 + u) % 15` | 4 head x 15 unit = 60 槽 / 15 条 = 每条 4 槽 = **1024**，15 条全用 |

### 3.2 private 扫描

`A3b` 没有 diff 池：15 组 repair 按 `i % 4` 摊到该 head 的 4 条 channel 上，
**每组各占一个行对齐的槽**（8 行用掉一整行）→ ch0/ch1/ch2 各 4 组 = 32 行，
ch3 得 3 组 = 24 行。加上第一个 decode 步的 1 个自有行（落 ch0）。

`A4/A4b` 有 diff 池：**4 个 head 的 120 行全部打包进 ch15**，一段 480 行。

---

## 4. 实测：逐格对照

脚本从报告的 `PIM:pool*` 事件里读逐 channel 行数。**行数一栏是"手算 / 实测"。**

### 4.1 公共扫描（15 个共享块）

| 档 | 逐 channel（手算/实测）| 一致 |
|---|---|:--:|
| **A3b** | ch0 1024/1024 ch1 1024/1024 ch2 904/904 ch3 768/768 （×4 head，16 条全活跃）| ✅ |
| **A4** | ch0–ch11 全是 1280/1280，**ch12/13/14 闲置** | ✅ |
| **A4b** | ch0–ch14 全是 1024/1024 | ✅ |

| 档 | 手算最忙 channel | extent | ACT | **实测扫描时间** |
|---|---|---:|---:|---:|
| **A3b** | 1024 行 | 4 | 4 | **2.8622 us** |
| **A4** | 1280 行 | 5 | 5 | **3.5697 us** |
| **A4b** | 1024 行 | 4 | 4 | **2.8622 us** |
| **A5 / A6** | 1024 行（布局同 A4b）| 4 | 4 | **1.8641 us** |

- **A4 比 A3b 慢**：让出 ch15 给 diff 池后 master 池只剩 15 条，
  `stripe_m = 15 // 4` 从 4 掉到 3，一个 head 的 15 个 unit 从摊 4 条变成摊 3 条，
  同时 **ch12/13/14 完全空着**。手算 5 个 unit vs 4 个，实测 3.57 vs 2.86 us。
- **A4b 把它修回来**：全局表铺满 15 条，回到每条 4 个 unit。
- **A5/A6 布局与 A4b 逐位相同**，快下来的 1.0 us 全部来自 **MQ 批命令**
  （一条 `MAC_AB` 服务全部驻留 query，不再按 query 复制）—— 这正是 A5 该买到的东西。

### 4.2 private 扫描（重算行，两个 decode 步之和）

| 档 | 逐 channel（手算/实测）| 一致 |
|---|---|:--:|
| **A3b** | ch0 65/65 ch1 64/64 ch2 64/64 ch3 48/48（×4 head，16 条全活跃）| ✅ |
| **A4** | ch0 1/1 ch3 1/1 ch6 1/1 ch9 1/1 **ch15 960/960** | ✅ |
| **A4b** | ch0 1/1 ch1 1/1 ch2 1/1 ch3 1/1 **ch15 960/960** | ✅ |

| 档 | 手算最忙 | extent | ACT | 实测 |
|---|---|---:|---:|---:|
| **A3b** | ch0 65 行 | **9** | **9** | 0.6667 us |
| **A4 / A4b** | ch15 960 行 | **2** | **4** | 0.6790 us |

**这一格就是 master/diff 分离要买的东西**：A3b 用 **9 次 ACT 读 65 行**
（每个 8 行的 repair 各占一整行，利用率 3.1%），A4/A4b 用 **4 次 ACT 读 960 行**。
每行的激活代价差 **约 30 倍**。

> 但要如实说：**在这个小 workload 上，两者的时间几乎打平（0.667 vs 0.679 us）。**
> 因为 diff 池把 4 个 head 的 480 行全压到 **一条** channel 上，
> 集中省下的 ACT 被"只用一条 channel 流"抵掉了。
> 重算量再大（baseline 的 33 组、或 head 更多）时散落的一侧才会明显吃亏 ——
> GPT-13B 上最忙 channel 的 ACT 是 A3b 22 次对 A4b 14 次。
> **这个 workload 证明的是机制被正确建模了，不是它的量级。**

## 5. 整体阶梯

| 档 | makespan (s) | energy (nJ) | prefill 放置 | 事件数 |
|---|---:|---:|---|---:|
| **A3b** | 0.026658 | 4.5519e8 | GPU | 882 |
| **A4** | 0.026614 | 4.5514e8 | GPU | 558 |
| **A4b** | 0.026609 | 4.5514e8 | GPU | 582 |
| **A5** | 0.025706 | 5.1261e8 | PIM | 2546 |
| **A6** | 0.025706 | 5.1261e8 | dynamic | 2546 |

单调改善，但幅度很小 —— 这个 workload 里 decode 只有 2 步，makespan 被 prefill
和 GPU 侧主导，PIM 扫描的差异被稀释。**A6 = A5 逐位相同**，说明选边器把两个
请求都判给了 PIM（没有混合的余地：只有两个 agent）。

## 5.1 产物在哪

| | |
|---|---|
| 入库 | `wl_handcheck.json`、`README.md`（本页）、`compare_theory_vs_measured.py`、**`results_handcheck.csv`** |
| 只在本机 | `output/handcheck_20260903/`（172 MB）：五档的完整事件流 `{A3b,A4,A4b,A5,A6}.json`（`--workload-report-events full`）+ `.log` + `run.sh` |

`results_handcheck.csv` 是本页两张对照表的**机器可读版**：109 行，逐档 × 逐扫描
（common / private）× 逐 channel，列为 `theory_rows / theory_extents /
theory_acts / measured_rows / measured_time_s / agree`，外加该档的
`makespan_s` 与 `energy_nj`。**`agree` 列 109 行全部 `yes`。**
原始事件流按 `docs/RAW_DATA_MANIFEST.md` 的口径不入 git。

## 6. 复现

```bash
export PYTHONPATH=$PWD KVPIM_CPPCORE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 复用计划（不跑仿真，确认 15 段各 8 行重算）
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/handcheck/wl_handcheck.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --validate-workload --workload-plan plan.json --num-hbm 2 --ngpu 1

# 五档（scratch 必须放 /data2，见 docs/README_run_slurm_and_local.md）
for A in A3b A4 A4b A5 A6; do
  RD=/data2/chenyi9/kvpim_run_scratch/hc_$A; rm -rf $RD; mkdir -p $RD
  ln -sf $PWD/ramulator2/ramulator2 $RD/; ln -sf $PWD/ramulator2/trace_gen $RD/
  cp ramulator.out $RD/
  ATTACC_RAMULATOR_DIR=$RD ATTACC_RAMULATOR_LOG=$RD/ramulator.out \
  python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
    --workload workload/handcheck/wl_handcheck.json \
    --reuse recompute --epic-prefix-recompute-tokens 8 \
    --ablation $A --engine dag --workload-report out_$A.json \
    --workload-report-events full --cacheblend-batch-size 8 \
    --num-hbm 2 --ngpu 1 --ramulator-workers 8
done
```

手算一侧：

```bash
python3 -c "
from types import SimpleNamespace as NS
from src.workload_runner import (_striped_append_channel_extents as EX,
                                 _GEN_BYTES_PER_TOKEN, _GEN_ROW_BYTES)
def loc(o,f,k): return NS(owner=o,fingerprint=f,kind=k)
def act(n): return -(-n*_GEN_BYTES_PER_TOKEN//_GEN_ROW_BYTES)
common = lambda sh: [loc('A_owner','doc%02d'%i,'master')
                     for i in range(15) for _ in range(256 if sh else 248)]
for tag,pol,sh in (('A3b','slice-append',False),
                   ('A4','master-diff-slice-append',True),
                   ('A4b','master-diff-table-append',True)):
    g = EX(common(sh), policy=pol, heads_per_hbm=4)
    print(tag, [(c, sum(n for _,_,n in p), len(p),
                 sum(act(n) for _,_,n in p)) for c,_,p in g][:4])
"
```

## 7. 这个校验证明了什么、没证明什么

**证明了**：

1. `_striped_append_channel_extents` 算出来的逐 channel 行数，和仿真器真正发给
   Ramulator、并回写到报告里的**逐格相同**（公共扫描 16/12/15 条 channel，
   private 扫描 16/5/5 条，无一例外）。
2. A4 的 `stripe_m` 取整退化是真的：**ch12/13/14 确实空着**，实测比 A3b 慢。
3. 重算行的两种摆法确实走了不同的路径：A3b 16 条 channel 全沾，A4/A4b 只有
   ch15 + 每 head 一行。
4. A5/A6 的加速与布局无关 —— 布局逐位同 A4b，快的 1.0 us 来自 MQ 批命令。

**没证明**：

- 任何量级上的结论。两个 agent、2 个 decode 步，makespan 被 prefill 主导。
- 散落 vs 集中的**代价差**：这里几乎打平（见 §4.2 的说明）。
- `A6 = A5` 只说明这个 workload 没有给选边器混合的余地。

**已知未建模**（见 `docs/sessions/2026-09-03.md` §10.7）：一条 channel 上多个
head 的 extent 顺次排布、不建 append 时间上的交错；同一请求的 cached chunk 仍
假定彼此打包连续。
