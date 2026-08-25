# AUDIT(2026-08-25):chenyi-822-dirty 与统一口径的一致性核对

**范围。**把本仿真器（分支 `chenyi-822-dirty`）建模里涉及 AttAcc bank 级
数据流的假设，逐点核对论文仓库 `audit/2026-08-25_bank_dataflow_reuse.md`
定下的统一口径：**一 channel 一 head（一个 HBM 打包 16 个 head）；每 bank
$d_{head}/4$；Q 复用 $L/16$ → 驻留；P 复用 2 → 流式；logic-die softmax =
buffer（按 $n_q$）、不是流式。**

**结论:全部承重点一致,无冲突。** 代码里 2026-08-24 的 rulings 和口径
逐字对得上。仅一处**简化**（softmax buffer 未作第二容量上限），下附。

> **来源澄清（2026-08-25 下午）**：早先口径文档把 head 映射误写成"一 HBM 一
> head（1024 banks / L/256）"。核对原版 AttAcc 仿真器
> `pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py`（初始 commit `c1540de`，
> `score_mac` 按 `L/(n_pch·n_rank·n_bg)=L/16` 切、`run_attention` 用
> `num_itr=ceil(n_head_per_hbm/n_channel)`）确认：**一 head 一 channel（64 banks）
> 是原版 AttAcc 建模，我们没改过**（只在其上加了 `_shared_query_attention_commands`
> 的 MQ 路径，复用同一结构）。故**仿真器保持现状不动**，改的是文档口径：L/256 → L/16。

## 逐点核对

| 口径 | 仿真器落点 | 判定 |
|---|---|---|
| **一 channel 一 head（64 banks）** | `gen_trace_attacc_bank.py:226` `num_itr = ceil(n_head_per_hbm / n_channel)`（`n_channel=16`，一 head 一 channel）；`score_mac` 按 `L/(n_pch·n_rank·n_bg)=L/16` 切 token、`d_head/(n_bank·n_mac)` 切 $d_{head}$ | ✓ 一致（原版 AttAcc 建模） |
| **每 bank $d_{head}/4$ = 64 B/query** | `ablation.py:154` “one Q slice is 64 B, so this caps the queries resident”；`ramulator_wrapper.py:56` `n_q = S/64` | ✓ 一致（64 = $d_{head}/4{\times}2$，与 head 铺多少 bank 无关） |
| **Q 复用 $L/16$ → 驻留** | `ramulator_wrapper.py:54` “Q is the ONLY capacity-bound operand (ruling 2026-08-24): a Q slice is reused across **every K column of the bank**, so it must stay resident” | ✓ 一致（“every K column” = 本 bank 的 $L/16$ 个 token；**代码本来就对，早先口径列写错成 L/256**） |
| **P 复用低 → 流式** | `ramulator_wrapper.py:57` / `trace_gen:479` “A P entry has (almost) no per-bank reuse … streams through the double-buffered GEMV-buffer halves via MV_GB；bound = movement-bus BW + MVSB↔MVGB turnaround (nRTW/nWTRL), not buffer capacity” | ✓ 一致 |
| **softmax = per-query buffer,非流式** | `trace_gen:218` “Q, score/softmax state, and PV results remain **private to each query**”；`:266` “**Softmax is query-private**”；每 query 一条 `PIM_SFM`（AttAcc buffer 式 softmax 命令，非 flash） | ✓ 一致（buffer 存 $n_q$ 条，不是 online/流式） |
| **MQ:一次列读服务全部驻留 Q** | `A5` preset `pim_batch_command="mq"`；`_apply_pim_batch`；`mq_interval_cycles = max(地板, ceil(n/(f·tCK)))` | ✓ 一致 |

## 数字细化（不是冲突，是把定性写成定量）

代码把 P 的复用写成“(almost) no per-bank reuse — one scalar per V column
**per output pass**”。统一口径把它**钉成 2**：一个 $P[t]$ 对应 $d_{head}/4=32$
个输出维，一拍只算 16 lane，$32/16 = 2$ 个 output pass → $P[t]$ 被用 **2 次**。
两者同义，口径只是给出确切次数。RTL 侧（kvpim-rtl `ChangeNotes.md` 的 cy 回复）
同此。

## 唯一的简化(需知会,非 bug)

**softmax buffer 没有被当作第二条容量上限建模。** 仿真器里 $n_q$ 的上限只有
一条：GEMV(input-vector) buffer，`n_q = gemv_buffer_bytes/64`
（`ramulator_wrapper.py:mq_query_capacity`）。而论文 §4.4.3 把 **die 上的
softmax buffer 也算一条“secondary limit”**（要放 $n_q$ 条 per-agent 分数向量，
$n_q{=}16$ 双头流水 ≈256 KB < AttAcc 512 KB）。

- 影响：只要 softmax buffer 够放当前 $n_q$（stock 口径 $n_q{=}8$→128 KB、
  $n_q{=}16$→256 KB，都 < 512 KB），GEMV buffer 是**先绑定**的那条，
  结果不受影响。
- 建议：暂不改模型（与论文“secondary”定性一致）；若以后扫**很大的**
  `gemv_buffer_bytes`/$n_q$，需要补一条 `n_q ≤ softmax_buffer_bytes/(2·L·? )`
  的封顶，避免 GEMV buffer 允许的 $n_q$ 超过 softmax buffer 能承载的条数。
  记为 TODO,等真要扫大 $n_q$ 时再定。

## 与既有 audit 的关系

- `README_audit_ladder_issues.md` / `README_manual_audit_findings.md`：ladder
  正确性与手工审计，本文不重复，只补 bank 级 Q/K/P/V 复用口径这一层。
- 论文侧对应：`KVPIM-1Fugue-ASPLOS2027/audit/2026-08-25_bank_dataflow_reuse.md`。
- RTL 侧对应：`kvpim-rtl/ChangeNotes.md` 末尾 “cy 回复（2026-08-25）”。
