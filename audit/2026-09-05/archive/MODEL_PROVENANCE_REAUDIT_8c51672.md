> 历史记录：保留当时的技术证据和评价，当前待审事项统一看 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md)。

# 模型计量整改复审：8c51672

> **后续裁决更新：** 共同沿用的 AttAcc 能量、stack 复制、缓存等限制不再要求精细化；每项先给来源与分档影响，再由 chenyi9 判断。以下“高”“应修”保留为原审计评价，当前分类见 [逐项审阅清单](ATTACC_RELATIVE_FAIRNESS_REVIEW.md)。FlashAttention 仍须共同启用，解析近似本身不等于不公平。

**本轮计量整改多数方向正确，但 GQA 修正没有贯通到验证器与 batched decode，产生了可复现的入口失败；缓存工具链指纹也没有覆盖实际执行的共享库。因此不能判定本轮改动已经全部正确。** 普通 DRAM 读写改为零成本符合 chenyi9 的明确裁决，原先 A3b/A4c 的读写差率不再作为未修问题提出。

审查者：独立 agent `/root/attacc_model_provenance`。源码比较为 `cdd89db..8c51672`，读取了 `agent.md` 以及当前 session §12–13。接受本轮裁决：普通 DRAM 读写只保落地依赖、链路照常计价；A100a 平台；各档共用随机修正计划；A1 精确 L；A6 简单逐 request 选边。以下只评价这些规则下实现是否闭合，不恢复已取消的普通 DRAM 或 DIE/TLB 收费，也不要求候选 DAG 选边。

计量证据来自源码及轻量设备桩、Ramulator 外部调用桩、纯函数输入。数值由 [probe 归档](model_provenance_8c51672_probe.txt) 自动计算，完整输出在 [JSON](model_provenance_8c51672_probe.json)，可读数表在 [自动证据表](model_provenance_8c51672_evidence.md)。这些桩的时间常数不是性能测量，不能用于计算加速比。本报告不借用 session 的历史单测通过数来声称本轮验证完成。

## 已修到位的部分

| 改动 | 判定与证据 |
|---|---|
| 普通 DRAM store/read 不额外计费 | `src/workload_runner.py:2447–2470` 统一生成 `STORE`，single decode `3436`、batched decode `3871`、私有 PIM prefill `4198` 也一致。`src/cpp_eventcore.py:18` 将 STORE 纳入 dependency-only；事件构造 `2154–2163` 强制零费用，Python scheduler `2530–2538` 不占硬件队列。捕获到的各 PIM 档 STORE 均为零；独立依赖链的 STORE 不会因另一个较晚的 STORE 而排队。普通读写的旧带宽/能量乘数缺口按本轮口径关闭。 |
| QKV projection 的 GQA 宽度 | `src/model.py:100–107,115–121,183–188` 改为本地 Q 宽加两份 KV 宽；MHA 保持原尺寸，LLAMA3-8B 的投影和 KV link 模板宽度与模型 Q/KV head 比例一致。具体推导结果见自动证据表。 |
| 部分 GQA KV 链路 | `src/workload_runner.py:3330,3918,3935,4000,4837,5013` 使用 `_kv_hidden`，single decode、GPU prefill 写入/回读、选边回读、共享分支 prefill 等原有多计已修。尚未覆盖的调用见下文。 |
| PIM GQA MAC 工作量 | `src/devices.py:509–516` 从 `pim_shared_queries / m` 还原每 KV head 服务的 Q head 数；`480–481,540–541` 同时应用到 aggregate 和 per-run 两条 ALU 计价路径。桩验证两接口一致，GQA group 的能量倍率与 query 数倍率相同。原 AttAcc 的 QK/PV 合并能量口径仍保留，不将其重新判作本轮引入的错误。 |
| A1 连续 scan 精确 L | wrapper `601–605` 撤销向上量化；外部调用桩实际收到精确非整块长度，见自动证据表。这证明新请求进入 wrapper 的长度正确，不是用较长请求的 cycles 再缩放。 |
| extent 的 K 相对 row | wrapper `622–634` 新键含每个 K extent 对第一段的相对 row。旧报告中“同 row 与不同 row 错共用缓存”的固定 K/V 间距反例现在分为不同调用。 |
| TBT step 加权公式 | `experiments/summarize_ladder.py:25–45` 使用 `sum(end-first_token)/sum(lout-1)`，同时保留请求均值和最大值；打印和相对比较均读新增 weighted 字段。不同 lout 的纯函数算例与独立公式相等，见证据表。 |
| tier 累计结束列 | `experiments/collect_dag_ladder.py:122–125,160` 将 `cum_end_s` 改为该 tier 最晚请求 end；有 batch 记录的控制样例正确。其余 tier 输出仍不完整，见 MR-04。 |

这里的“STORE 计量已修”不等于已经证明 STORE 地址与新物理账本相同；账本地址与 trace 可执行性由另一个独立审查覆盖。

## MR-01：GQA 的正确 KV 字节被旧验证器拒绝（高，新回归）

当前产生 KV 的代码按 KV head 宽度发送，验证器却仍按 Q head 宽度要求流量：

```
# src/workload_runner.py:2756–2767
q_bytes_per_row = local_hidden * dbyte
kv_bytes_per_row = 2 * q_bytes_per_row
```

`_finalize_cacheblend_report` 在 `5138–5140` 无条件调用此验证器，仍只传 Q 侧 `local_hidden` 和 `heads`，没有 GQA/KV 宽度信息。

轻量验证使用同一个短 workload，分别运行 MHA 控制组与保留 LLAMA3-8B head 几何、仅缩减为单层的 GQA 模型。MHA 各档可完成；GQA 的 A1/A3b/A4c/A4e/A5/A6 均抛出 `WorkloadValidationError: CacheBlend KV link byte count is invalid`。A2 采用独立 GPU-only 验证路径，完成构图。没有调用实际 Ramulator。

**影响：** 这是数据通路修正与 invariant 未同步造成的执行失败，不是 GQA 的性能变差；不能将当前单测或 MHA 成功外推为 LLAMA3-8B 阶梯已可运行。验证器应该接收并检查独立的 KV head 宽度。

## MR-02：batched decode 仍按 Q 宽度发送 KV（高，修复遗漏）

`src/workload_runner.py:3473–3474` 仍设置 `q_bytes = local_hidden*dbyte; kv_bytes = 2*q_bytes`，后者用于 `3601–3607` 的每请求 decode KV link。同文件 single decode 已在 `3330` 改为 `_kv_hidden`，所以两条路径不一致。

捕获 GQA 的真实构图（在旧验证器抛异常前保存）显示 prefill KV bytes/row 正确，而 batched decode KV bytes/row 仍乘了 GQA group；具体字节和倍率由自动证据表给出。该问题在默认 batched 入口出现，不能通过修 MR-01 后就视作全部 GQA 修复。

另外几个仍使用 MHA 假设的入口：

- 私有 PIM prefill `src/workload_runner.py:4114` 仍为 `kv_bytes=2*q_bytes`。A1 默认 GPU prefill 不走这里，显式 PIM 对照会走。
- analytic 路径 `src/ablation.py:831,932` 的驻留回读仍为 `2*local_hidden`，没有除 GQA group；不能把 DAG 部分修正描述成整个仓库的 GQA 模型已统一。
- GPU attention 内部的 K/V HBM/L2 流量仍按 layer `numOp`（Q heads）计算，见 `src/devices.py:106–127`、`src/model.py:123–133`；新的 QKV 宽度并未使该部分自动变为 KV heads。这里不是本轮新增的私有优惠，但 GQA 的 GPU/PIM 跨设备收益仍有计量不确定性。

**偏差方向：** batched decode KV 传输多计会增加所有 PIM 档的链路负担；GPU 内部重复计算 KV 流量则可能弱化 GPU attention。两者不能当作相互抵消。

## MR-03：工具链 hash 漏掉 libramulator.so，旧 CSV 也未版本化（高，修复不完整）

`src/ramulator_wrapper.py:277–292` 只 hash `ramulator2` executable 和 bank trace generator。对当前默认 ELF 执行只读 `readelf -d`，可以确认它 `NEEDED libramulator.so`，且携带 RPATH。Ramulator 核心修改可以发生在共享库，而 executable 字节不变。

轻量临时文件验证：修改 bank generator 会改变当前 fingerprint；只修改 `libramulator.so` 不会改变。证据 JSON 保留了真实默认 ELF 的 dynamic section。**因此当前 hash 不能证明 cached cycles 与真正执行的 simulator 核心一致。** 应追踪实际加载的 Ramulator 共享库，而不能只根据所给目录假定加载哪一份库。

另有两个覆盖边界：

- 老 `ramulator.out` CSV 在初始化直接读入（wrapper `164–165`），`output` 仍仅按 shape 等字段匹配，不校验 `_toolchain`（`765–778`）。DAG 的地址/extent 路径绕过它；其他旧入口仍可读到源码升级前的结果。
- 指纹只覆盖 bank generator，BG/buffer 的 generator 不在其中。当前 paper bank 路径得到部分保护，不能扩大成所有 PIM 类型的完整工具链保护。

## MR-04：tier 的 cum_end_s 修正正确，但输出仍会漏档与混用指标（中，部分修复）

collector 从 request summary 得到了所有 tier 的结束时间，却仍只从 `batches` 创建要输出的 `by_tier`（`126–139`），随后只遍历 `sorted(by_tier)`（`148`）。A2 报告没有 `batches`（`src/workload_runner.py:4547–4571`），因而即使 summary 完整，A2 整档仍不会进入 tier CSV；batch size 为单请求时，其他 PIM 档也可能没有 batch 记录。

纯输入验证给 A2 和 A3b 相同的请求 end，只给 A3b batch stamps：CSV 中只出现 A3b。该行 `cum_end_s` 正确取请求结束，但 `tier_total_s` 仍取到最后 attention start，二者明显不同；自动证据表保留了实际输出。

`prefill_s`、`decode_s`、`tier_total_s` 在 `150–157,163` 仍用 attention stamps 和前 tier 最后 attention start 推导。它们可以作为清楚标注的近似，但不能因 `cum_end_s` 已修就称这三个字段也是实际完成耗时。生成 tier 行应直接基于 summary 的 tier 集合，并区分真实完成时间与 batch-start 近似。

## MR-05：相对 row 键修复了固定 K/V 间距反例，但 V 地址仍有通用 API 缺口（低/中）

wrapper `627–633` 加入了 K 的相对 row，却没记录 V 的相对 row。保留同一 K extent，仅将第二段 V 移到另一 row 时，现有 API 仍接受输入并命中前一结果。连续三个输入的累计外部调用数见自动证据表。

**判定要限定：** 当前新账本以固定 K/V 间距生成 extent，K 的相对 row 因而可以推导 V 的相对 row，这个限定下原反例已修。上述残留是 wrapper 通用 extent API 的完整性缺口，不能直接声称当前 ladder 必然触发。可以让缓存包含 K/V 两侧相对 row，或者显式验证固定间距前提；不能一面接受任意 V 地址，一面声称键保留了其全部时序关系。

## 仍需保留的旧计量限制

session §13 已明确未处理 MP-05：`src/workload_runner.py:2313–2317,2439` 仍把最忙 stack 的满载能量按向上取整的 stack 数复制。不满 stack 时能量会多计；单 HBM 控制配置不触发该余数。这项不应被本轮 GQA group ALU 修正掩盖。

MQ 能量间隔公式、GPU flash 的近似校准、普通 AttAcc QK/PV 能量口径在本轮比较中没有修改。关于其来源强弱仍沿用前次专项报告；本轮不能新增“已校准”或“全部由完整命令工作量计价”的认证。此次 ordinary DRAM 零费用已按明确裁决关闭相应旧发现。

## 复核方式与文件覆盖

本轮详细比较 `src/ramulator_wrapper.py`、`src/devices.py`、`src/model.py`、`src/cpp_eventcore.py`、`experiments/summarize_ladder.py`、`experiments/collect_dag_ladder.py` 的改动，追踪了 `src/workload_runner.py` 中对应生产者、验证器、聚合器，以及 `src/ablation.py` 的残留调用。证据 JSON 记录本次 revision 与这些主要源码文件的 SHA-256。

在仓库根目录复现轻量检查：

```bash
python3 -B audit/2026-09-05/model_provenance_8c51672_probe.txt
```

本次实际执行脚本位于 `/tmp/model_provenance_8c51672_probe.py`，与 `.txt` 归档内容一致。probe 所有 Ramulator 入口均替换为返回固定计数的桩；它证明路径、尺寸、计数、缓存和输出规则，不证明实际 cycles 或论文加速比。
