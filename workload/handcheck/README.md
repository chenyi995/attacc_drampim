# 手算校验：A3b–A6 的布局，理论对实测

> **2026-09-03 更新**:A3b 的 repair 打包口径按 chenyi9 裁决改过 ——
> 一个 head 自己的 repair 是**一条连续 append**(head 内共享行),A3b 买不到的
> 只是**跨 head 共享行**。本页的数字与 `results_handcheck.csv` 都已按改后的
> 代码**重跑重生成**,`agree` 列 100 行全 `yes`。

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

`A3b` 没有 diff 池，但**一个 head 自己的 15 组 repair 仍然是一条连续的 append**
（裁决 chenyi9 2026-09-03：它们由该 head 的同一次 prefill 一起产生、接连写下,
所以**在 head 内部**共享行）。15 x 8 = 120 行放不满一整行(256)，所以**每个 head
留一个半行**；A3b 买不到的是**跨 head 共享行**，而那正是 diff 池唯一多出来的东西。
加上第一个 decode 步的 1 个自有行(落该 head 的第一条 channel)。

`A4/A4b` 有 diff 池：**4 个 head 的 120 行全部打包进 ch15**，一段 480 行 ——
四个半行合并成两行。

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
| **A3b** | ch0 121/121 ch1 120/120（×4 head，落 ch0/1、ch4/5、ch8/9、ch12/13）| ✅ |
| **A4** | ch0 1/1 ch3 1/1 ch6 1/1 ch9 1/1 **ch15 960/960** | ✅ |
| **A4b** | ch0 1/1 ch1 1/1 ch2 1/1 ch3 1/1 **ch15 960/960** | ✅ |

| 档 | 手算最忙 | extent | ACT | 实测 |
|---|---|---:|---:|---:|
| **A3b** | ch0 121 行 | **2** | **2** | 0.2099 us |
| **A4 / A4b** | ch15 960 行 | **2** | **4** | 0.6790 us |

**这一格就是 master/diff 分离要买的东西**,但要按行数、而不是按这个小负载的时间读：
A3b 的 120 行 repair **每个 head 留一个半行**,四个 head 就是 **4 行**;
A4/A4b 把 4 x 120 = 480 行打包成 **2 行**。裁决里的算式是同一件事:
`4 个 head x C 个 chunk x k=8` 打包进 diff 池,C=24 时 `768 = 恰好 3 行`、
C=16 时 `512 = 恰好 2 行`,而 A3b 恒付 4 行 —— **所以 sweep 用 C=16,收益翻倍**。

> 但要如实说：**在这个小 workload 上,A3b 的 private 扫描反而更快
> (0.210 vs 0.679 us)。** 因为 diff 池把 4 个 head 的 480 行全压到 **一条**
> channel 上串行流,而 A3b 的 4 个半行分散在 8 条 channel 上并行。
> 省下的行要在**行数本身成为瓶颈**时才兑现。
> **这个 workload 证明的是机制被正确建模了(理论与实测逐格一致),不是它的量级。**

## 5. 整体阶梯

| 档 | makespan (s) | energy (nJ) | prefill 放置 | 事件数 |
|---|---:|---:|---|---:|
| **A3b** | 0.026655 | 4.5492e8 | GPU | 802 |
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

`results_handcheck.csv` 是本页两张对照表的**机器可读版**：100 行，逐档 × 逐扫描
（common / private）× 逐 channel，列为 `theory_rows / theory_extents /
theory_acts / measured_rows / measured_time_s / agree`，外加该档的
`makespan_s` 与 `energy_nj`。**`agree` 列 100 行全部 `yes`。**
它由 `compare_theory_vs_measured.py --csv <路径>` 重生成（2026-09-03 加的开关）：
A3b 的 repair 打包口径改过一次，那次就是靠"CSV 只能手工誊抄"发现它落后了代码，
所以现在证据跟着代码一起重生成。
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

# 五档（scratch:squire 放 /data2、athena 放 /localdata,见 docs/README_run_athena_slurm.md
#       与 docs/README_run_squire_local.md）
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


---

## 2026-09-04 补充：多轮 workload 生成器

`gen_multiround.py`：消费者上下文交替 `sys | shared | own | shared | own | …`，每轮取一个（或 `CHUNKS` 个）
shared chunk 修一次、再写一块自己的 KV，于是各轮的修正被自己的新 KV 隔开 —— 这是 baseline sweep 的
workload 里没有的结构（它的消费者一次 prefill 把修正连着写完）。环境变量 `ROUNDS` / `CHUNKS` / `CONSUMERS` / `LOUT`。
修正保持 per-agent，不跨 agent 共享。用途与结果见 `../../docs/README_design_ladder.md` §8。
