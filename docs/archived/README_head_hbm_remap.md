# head→HBM 重映射记录(2026-08-27,裁决:chenyi9)

> **已归档 2026-09-02。** 本页记录 2026-08-27 的一次放置修正
> (从「一个 head 住一个 channel」改为「一个 head 住一个 HBM」)。
> 该放置模型已被两轮改动取代:2026-08-29 重切为 **head-aware 通道放置阶梯**
> (A3b 切片 / A4b 全局 co-read 表,见 `README_sweep_design.md`),
> 2026-09-02 又修了通道 lane 的并行调度(`75da860`)。
> **当前放置语义以 `src/workload_runner.py` 的 `_layout_channel_loads`
> docstring 与 `README_rung_analysis.md` §4 为准。** 本页保留作那次修正的记录。


**一句话**:此前所有 PIM 侧数字建立在"一个 attention head 住一个
**channel**"的放置假设上;在本项目的配置(LLAMA-7B、TP8、5 HBM)下该
假设退化为**单 channel 扫全部上下文、其余 15 个 channel 闲置**,多
channel 池还把同一份列样式按"幻影 head"逐 channel 复制。本次改为
"一个 head 住一个 **HBM**,run 覆盖的各 channel 承载该 head **自己的
token 条带 (stripe)**",并同步修正带宽记账、签名缓存与验证。**此前
跑出的全部 PIM 侧结果作废,套件已按新语义重跑。**

术语:channel = HBM 内的独立通道(本模型 1 个 HBM 16 个 channel);
head = attention 头;HBM = 高带宽内存堆叠 (stack);MAC_AB =
channel 级全 bank 乘加命令(64 bank × 32 B = 2 KB/命令);
条带化 (striping) = 把一段 token 序列按 channel 轮流切开存放。

---

## 1. 旧语义是什么错(代码证据)

### 1.1 放置规则:head = channel

trace 生成器把"第 lch 个 head"直接放到"第 lch 个 channel":

- 初始版(上游)`pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py`
  (commit `c1540de`):
  `for lch in range(valid_channel): addr = base + lch * HBM_GS['ch'] + ...`
  ——同一份 row/col 命令样式对每个 lch(= head)平移一个 channel 的
  地址增量后**整份复制**;一个 head 的全部 L 个 token 全部落在**它自己
  那一个 channel** 里(在 channel 内部沿 pCH/rank/BG/bank 展开)。
- 迭代结构:`num_itr = ceil(n_head_per_hbm / n_channel)`,余数轮
  `remainder = n_head_per_hbm % n_channel` 决定 `valid_channel`
  ——语义都是"**head 数决定用几个 channel**"。

### 1.2 head 数的来源:wrapper 的 ceil 除法

`src/ramulator_wrapper.py`(同为 `c1540de` 引入,现 418/603 行):

```python
num_ops_per_attacc = layer.numOp            # 本加速器上的 head 数
num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
```

`num_ops_per_hbm` 以 `--nhead` 传给 trace 生成器,变成上面的
`n_head_per_hbm`。

### 1.3 本项目配置下的退化(定量)

LLAMA-7B、TP8(张量并行 8 路):每加速器 `numOp = 32/8 = 4` 个
head;`num_hbm = 5`(`src/config.py:266`)→
`n_head_per_hbm = ceil(4/5) = 1`。于是:

- **legacy(no-reuse)路径**:`remainder = 1 % 16 = 1` →
  `valid_channel = 1`——一个 channel 扫全部 L 个 token,**15/16 的
  channel 与 60/64 的 bank 级 PIM 单元闲置**;实测 trace(L=256,
  dhead=128,BF16):64 条 MAC_AB **全部落在 channel 0**。
- **池化(TLB/复用)路径**:run 给定 `--channels 15`(master 池)时,
  `Attention()` 的 lch 循环仍按"每 channel 一个 head"复制同一份列
  样式——15 个 channel 各扫一遍**同一段 L**,等于虚构了 15 个
  "幻影 head":能耗虚增 ~15×,时间也不随 channel 数下降(每
  channel 命令数不变)。
- 上层再叠一层不自洽:workload_runner 的 KV 布局(TLB)明明把
  token 块**按 channel 轮转条带化**(256-token 页轮换 16 channel),
  而 trace 层却按"head 复制"解释同一批 channel——**两层各说各话**。

### 1.4 影响面

所有经 Ramulator 的 PIM 侧读数(扫描时间、能耗、A3/A3a 断流惩罚、
A5/A6 MQ 收益、fig:motiv 套件、wl_tiny 阶梯、star r3 全表)在
2026-08-27 之前的输出目录里都是旧语义,**全部作废**;旧签名缓存
`ramulator2/signature_cache.jsonl`(~75 万行)一并弃用(留盘归档,
不再载入)。GPU 侧与链路侧事件不受影响。

---

## 2. 归因:错误是哪个 commit、谁的(git 考古)

`git log --follow` + `git blame` 结论,三层叠加:

| 层 | commit | 账户 | 日期 | 内容 |
|---|---|---|---|---|
| ① 放置规则本体 | `c1540de`(initial commit,文件当时在 `pim_ramulator_src/trace_gen/`) | jwchoi(AttAcc 上游) | 2024-05-06 | `lch * HBM_GS['ch']` 的 head=channel 复制、`num_itr`/remainder 结构、wrapper 的 `ceil(numOp/num_hbm)`。对上游自身评估配置(head 数多、能塞满 16 channel)是自洽设计,**不是他们场景下的 bug** |
| ② 带进复用/池化路径 | `47ae0c3`(A1–A6 placement ablation) | xw338 | 2026-08-21 | `ch_delta()`/`pool_base` 让池 run 跨多 channel,但保留了"lch=head"的复制语义 → 幻影 head 复制从此进入共享 KV 路径(注:`0aced82` 2026-08-17 加 CacheBlend 支持时尚无 ch_delta) |
| ③ 在退化配置下出数 | `fb9fabe` 起本仓实验链 | 本线(chenyi9) | 2026-08-22 起 | LLAMA-7B/TP8 把 `n_head_per_hbm` 压到 1,触发 §1.3 两种退化;未察觉,fig:motiv 等全部套件按此出数 |

直接回答"谁留下的":**根源假设是 ① `c1540de`(jwchoi,AttAcc
上游)的 head-per-channel 设计;把它带进共享 KV 池化路径的是 ②
`47ae0c3`(xw338);在退化配置下跑出错误数字并沿用的是 ③ 本线
(chenyi9)的实验 commit 链。**单独指认任何一个 commit 都不完整——
①在上游场景自洽,②③各自继承时都未重新审视该假设。

---

## 3. 新映射规范(head→HBM 条带)

裁决(chenyi9 2026-08-27):"按照 head 一个 hbm 的重映射改好"。

- **一个 head 住一个 HBM**;head 间并行是 **HBM 间并行**,发生在
  trace 之外(各 HBM 同时各跑自己的 trace,墙钟不乘 head 数)。
- 一次 run 覆盖的 channel(legacy 16 个;池 run 的 `--channels` 个)
  承载**该 head 自己的 token/chunk 条带**:
  `stripe_width = channels // heads_per_hbm`,
  每 channel 命令按 `L_per_channel = ceil(L / stripe_width)` 生成。
- **通式**(head 数多于 HBM 数时):
  `heads_per_hbm = ceil(local_heads / num_hbm)`,同一 HBM 内的几个
  head 平分 channel(`stripe_width` 随之变窄)。本项目 7B/TP8:
  `heads_per_hbm = 1`,`stripe_width = channels`。
- `num_itr = 1`(不再按 head 分轮),remainder 轮作废。
- **256-token chunk 的几何不变**:一个 chunk 仍是每 bank 1 个
  DRAM row × 64 bank(K 侧;V 镜像 +8 MiB),扫一个 chunk 仍是
  32 条 MAC_AB(score)+ 32 条(context);变的只是这些命令现在
  **摊在 16 个 channel 上**(每 channel 串行深度 64→4),而不是全部
  压在一个 channel。8-MiB K 窗口是 bank 级硬限制,不动;更长上下文
  加 HBM(既有裁决)。

## 4. 代码改动清单(全部在工作树,未 commit)

| 文件 | 改动 |
|---|---|
| `ramulator2/trace_gen/gen_trace_attacc_bank.py` | 新增 `--head-hbm-stripe` 开关(全局 `head_hbm_stripe`,默认 False——不带开关时旧语义逐字节保留);`run_attention()` **顶部**按 `stripe_width` 缩 L(命令生成与拼装环用同一个 L);`num_itr` 两站点在开关下恒 1;两处 remainder 计算在开关下关闭;`n_head_per_hbm` 提为模块全局 |
| `src/ramulator_wrapper.py` | `run_ramulator()` 恒定追加 `--head-hbm-stripe`;`_run_signature` 元组首元素加版本记号 `"hbmstripe1"`;签名缓存文件轮换为 `signature_cache_v2_headhbm.jsonl`(旧文件留盘归档、不载入) |
| `src/workload_runner.py` | ① `_append_channel_kv_stores`:落 KV 事件带宽从"聚合 × channels/16"改为"**聚合 / num_hbm** × channels/16"——一个池事件现在只含**一个 head** 的字节,其余 HBM 上其他 head 的并发拷贝是 head 并行维度;② `die_load_di_bitmap`:D_i 位图是**广播**(每个堆叠的 die 各存同一份),时间按单 die 带宽份额、能量按 num_hbm 份拷贝计;③ die 侧 `q_bytes`/`tuple_bytes` 事件**不改**——字节是全部本地 head 的聚合,`SOFTMAX_MEM_BW = 670.4 GB/s × num_hbm` 也是聚合秤,聚合/聚合与"每堆叠 1/num_hbm 字节 ÷ 1/num_hbm 带宽"数值相同,口径自洽 |

实现教训(留档):第一版把缩 L 放在 `Attention()` **内部**,而
`run_attention` 的拼装环仍用原始 L 推
`length = ceil(L/n_pch/n_rank/n_bg/16)`,大 L 时列表越界
(IndexError)。修正为在 `run_attention` 顶部缩(生成与拼装看同一个
L)。中间版本成功持久化的 v2 缓存条目与最终代码同语义(拼装长度
一致才可能成功),无需清洗。

## 5. 验证

- **trace 层对照**(dhead=128、BF16、nhead=1、L=256):旧语义 64 条
  MAC_AB 全在 channel 0;新语义 4 条 × 16 channel,总量守恒 64。
- **长度扫**:L ∈ {256, 1024, 8192, 30000} 生成通过,MAC_AB 总数 =
  旧语义值(64/256/2048/7552),仅分布摊开。
- **路径扫**:池 run(`--channels 15 --pool-base 0` + TLB 地址)、
  MQ(`--shared-kv --shared-queries 8 --mq`)、相位切分
  (`--phase score`)全部通过。
- **单测**:41/41 OK(测试桩 PIM 无 `num_hbm` 属性,记账处一律
  `getattr(..., "num_hbm", 1)` 兜底)。
- **wl_tiny 冒烟**(A1/A4/A5,`--engine dag`):见
  `docs/sessions/2026-08-27.md` 当日记录。

## 6. 台账与重跑

- 台账条目:R18(`docs/README_manual_audit_findings.md`)。
- 套件重跑:`bash experiments/run_dag_suite.sh LLAMA-7B`
  (5 workload × k∈{2,4,8,16,32},N_PAR=3,RAMU_WORKERS=4,96 核
  预算不变),v2 缓存从零暖起。2026-08-27 之前的 `output/` 目录一律
  视为旧语义存档,不再引用。
