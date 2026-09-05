# 独立 agent 复审：当前阶梯的因果归因仍不能签字

日期：2026-09-05。审查快照：`cdd89db04a85edae029fd3151165f1a488d6139c`；原始 AttAcc 比较点：`c600051`。执行者为独立子 agent `independent_fairness_audit`。本报告由独立阅读当前代码和论文正文得出；没有把旧 audit 的“未修复”直接当作当前事实。

**后续用户口径更新：** A6 的本意是逐 request 估计哪边快就选哪边，正文公式或事件图描述可以不准确。因此本报告撤回“没有构建两份候选 DAG 本身就违规”的判断；I05 只保留两侧估价与实际执行工作不对应的问题。新增的存储专项在相同 HEAD 上独立复查 store 与 scan 是否共享物理布局；仅运行轻量 helper，没有运行性能实验或改实现。

审查开始时 tracked 工作树干净，仅有既存 untracked `experiments/paper_ladder/`。检查 `/`、`/data2`、`/data2/chenyi9`、`/data2/chenyi9/KV-PIM` 和仓库根目录，未发现适用的 `AGENTS.md`。本轮仅新增此审计报告，没有修改实现、参数、测试、workload 或结果。下面的轻量 Python 探针只调用布局、调度和设备桩，不运行真实 Ramulator、不测速度、不运行全矩阵。测试耗时不能解释为硬件 latency。

## 审查口径与结论

按用户确认的口径：A1/A2 是独立 baseline，可以合理自建；A3b 是朴素软硬结合；A4c 增加 head 内 diff 聚集，A4e 增加软件共读 placement table，A5 增加 PIM prefill + MQ，A6 增加自动选边。额外 DIE/TLB latency、energy 不计价；旋转在 GPU。不能把这些已经接受的差异再次当成违规。

**目前不能确认 A3b→A4c→A4e→A5→A6 的所有差异只来自所声明机制，也不能确认所有相对 AttAcc 的扩展都已有充分依据。** A3b→A4c 仍有可直接复现的修正 token 集合和 DRAM read/write 带宽混杂；五档的 store 与 scan 仍使用不一致的物理映射。A6 逐 request 选边本身符合最新口径，但两侧估价与实际执行工作不完全对应。另一方向也有问题：A5/A6 的私有 GQA decode 没有带入 MQ，存在低估收益的路径。没有证据可以据此判断作者主观“刻意”削弱 baseline；能判断的是实现的效果和归因条件尚未满足。

| 比较 | 独立判断 |
|---|---|
| A1 / A2 | 按用户口径接受独立 baseline；A1 GPU prefill 已修，不能再报告成 PIM prefill。仍要把公共 GPU/HBM 扩展与原版区别说清楚。 |
| A3b → A4c | **不通过**：修正集合随档变化、旧 allocator 给 master read/write 不同带宽，且仍有 master extent 几何差异。 |
| A4c → A4e | 主要代码开关确为 append slot → conflict-aware slot；但只持久了 channel slot，没有持久化实际 trace row，不能称完整物理 placement 已核验。 |
| A4e → A5 | 当前代码确实把 fresh 和 reused prefill 都服从 PIM 配置；MQ 是用户接受的套餐。但 GPU 分支多发 Q，私有 GQA decode 漏 MQ，偏差双向存在。 |
| A5 → A6 | 配置层只有 dynamic；逐 request 估计后选较快一侧符合用户口径，不要求构建候选 DAG。剩余问题是两侧估价与实际执行的工作量、传输和 channel 选择不完全对应。 |

## 阻断结论的具体发现

### I01 / P0：默认运行脚本实际改变了 A3b 和 A4c 的修正 token 集合

证据：`main.py:465–480` 设置 `recompute_canonical = (reuse == "recompute" and kv_mapping != "naive")`。`experiments/run_dag_ladder.sh:60` 实际使用 `REUSE=recompute`，尽管文件顶部称 EPIC。因此不是只影响手动特殊参数。

轻量复现：一个 256-token shared 段，另一个请求在它前面加 8-token private 段；同 seed、同 k=8、同 workload。调用实际 `build_reuse_plan` 并使用 CLI 同一 canonical 条件，A3b 修正位置为 `[20,132,155,197,207,215,244,248]`，A4c 为 `[0,1,2,3,4,5,6,7]`。

这不仅改变物理放置，而是改变软件做哪些重计算，因此逻辑工作集合不一致已得到确认。但不能据此直接推断当前 A3b 的 scan 或端到端时间必然受到惩罚：`_pool_reads:1243–1293` 为默认 striped-append 扫描补回完整 master，随机位置主要增加的 `plan_reads` 描述符现在又不收费，因为 DIE/TLB latency 和 energy 均已清零。额外描述符不等于额外计费，其对当前默认 scan/E2E 的净性能影响没有在本轮证明或测量。不能沿用旧描述符收费路径下的速度推断。使用真正 `--reuse epic` 的运行不触发这条逻辑差异，但不能由此豁免默认脚本。

建议：先冻结每个请求/层/段的同一 corrected-row 集合，再让不同布局消费它；若要引入 canonicalization，应单独声明、单独对照。

### I02 / P0：同一 master 写入在 A3b 与 A4c 仍有 15 倍非 scan 价格差

证据：`NaiveKVLayout.finalize` 在 `src/workload_runner.py:1823–1844` 产生 `channel_count=1`；`CacheBlendTLB.finalize` 的 master block 使用 15-channel pool。`_append_channel_kv_stores:2203–2226` 沿用这组旧 TLB 属性，以 `channel_count / 16` 乘带宽；readback 也调用此函数（`3601`、`4648`）。scan 却使用新的 head-local 通道布局（`870–916`）。

独立直接调用：一个无 diff 的 256-row master，`bytes_per_vector=256`、一 HBM、聚合带宽 `10^12 B/s`：

| 档 | store 资源 | store 时间 | energy |
|---|---|---:|---:|
| A3b | `PIM:pool0-0` | `2.097152e-6 s` | `131.072 nJ` |
| A4c | `PIM:pool0-14` | `1.3981013333333334e-7 s` | `131.072 nJ` |

此处完全没有真实 Ramulator 调用；15 倍就是代码里的带宽分母差，不能归因于 diff gather。GPU prefill 的 resident readback 也可能把它放大到关键路径。按实际 head-local placement 为同一 read/write 定价，才可以核验此步公平。

### I03 / P1：master slot 相同，不等于 master 物理 extent 相同

`_place_master_by_slot:890–898` 先调用各自 allocator 的 `tlb.scan_runs`，再把这些 run 转成 row-aligned trace extent。Naive allocator 按 256-token 页切块，LocalDiff allocator 没有这个 master 分页过程。

独立构造同一个 512-token fingerprint，仅扫描 owner rows `[128,384)`，零 diff、同一 append slot、4 heads/HBM：A3b 每个活跃 channel 收到两个 128-token extent，地址分别 row 0 和 row 1；A4c 收到一个 256-token extent，地址 row 0。两者只是换 allocator，已经改变 master row 几何。

触发条件是读到大 fingerprint 的跨页子区间；若所有 master 永远恰好以对齐 256-token chunk 全量读，不能把此探针直接外推到该 workload 的速度。但是仓库接口允许更一般段/历史，因而“所有地方只变 diff”仍不能成立。需将 256-token chunk 的物理 row 身份在所有布局中固定，而非只检查 slot 表相同。

### I04 / P1：A3b 的“同一次产生的同 head 修正连续 append”没有落实

`948–959` 按 `(owner,fingerprint)` 分开给每一修正组 `add`，`793–811` 将每个 extent 起点向新 DRAM row 对齐。两组修正即使来自同一次 GPU prefill、没有 own KV 插入，也不会因此合并。

轻量例子：先保留 c0/c1 两个 master，再连续保留同一 consumer 的 c0/c1 各 8 行 diff；1 head/HBM。A3b 得到 ch2 和 ch3 各 8 行、各一 row；A4c 得到 ch15 的一个 16 行 extent。当前 `918–936` 的注释和设计文档则声称同 head 一次生成可以共享 row。

朴素 page-per-object allocator 本身可以合理，不能仅凭此指控人为构造；但它与当前接受的连续 append 规则不一致，必须明确 baseline 究竟是哪一种。两页可能分到不同 channel 并行，所以此例不证明 latency 必然更差；能量/ACT 以及多轮撞 slot 时才可能抬高 A4c 相对收益。

### I05 / P1：A6 两侧估价与实际执行工作不完全对应

按后续用户澄清，A6 逐 request 估计 GPU/PIM 哪边快就选哪边即可；没有构造候选 DAG、没有读取队列状态、首次选择后跨层保持同一 request 的决策，均不再单独判为违反 claim。论文 `sections/05-execution.tex:39–58` 的更强表述属于待统一文字。实际 `src/workload_runner.py:3640–3737` 仍有以下估价对应性问题：

- `t_xpu` 为 link + GPU score/softmax/context 的加和，没有对应实际分支的 DRAM readback 节点和 I06 所述实际多发的 Q-to-PIM 操作。
- `t_bank` 只按行数选一个 `busiest_run`（`3694`），不是把每条不同碎片程度的 channel 经 Ramulator 后取真正最慢者。
- 所有 sweep 以一个满 sweep 的时间乘次数，尾 sweep 不同 query 数不单独定价；额外旋转 Q variants 也不在这里。

这些问题需要通过比较同一 request 在 GPU/PIM 两个实际分支上的工作与价格来核对，不要求特定选边算法。碎片程度、额外链路传输或尾 sweep 都可能影响选择，净偏差方向没有固定保证。此前设计文档的 `A6 ≤ min(A4e,A5)` 也不能仅由逐 request 选较快一侧推出全局 makespan 保证。

### I06 / P1：A4e GPU prefill 仍发送本不需要的 Q-to-PIM

`src/workload_runner.py:4476–4484` 在选边之前无条件创建 `q_gpu_to_pim`；GPU 分支 `4634–4692` 没有消除它。A4e 和 A6 的 GPU 选择都会多付该 LINK 操作，A5 的 PIM 选择则确实需要 Q。

共享 prefill 的 GPU baseline 因而支付额外链路字节和固定 transfer latency，倾向抬高 A5 相对 A4e 的收益；A6 估计 `t_xpu` 又没有计这笔实际操作。应使 DAG 分支与各自实际输入一致。

### I07 / P1：A5/A6 私有 GQA decode 漏传 MQ，反向低估收益

公有 batch scan 在 `3433–3435` 调用 `_apply_pim_batch`。私有 scan `3466–3492` 只设置 `pim_shared_queries=gqa_group`，没有设置 command/frequency；`src/ramulator_wrapper.py:549` 的缺省值是 `replicate`。batch_size=1 的 `_append_cacheblend_decode` 签名及 `4699–4705` 调用同样没有 MQ 参数，`3057–3065` 只设置 GQA query 数。

独立设备桩运行：两个请求共享 16 行，各有 8 行 private，gqa=4，A5 配置 MQ/1.3004 GHz。

| 路径 | 实际 captured shared_queries | command/frequency |
|---|---:|---|
| batch=2 的 shared decode | 8 | mq / 1.3004 |
| batch=2 的两个 private decode | 每个 4 | 都缺失，wrapper 回退 replicate |
| batch=1 的 decode | 每个 4 | 都缺失，wrapper 回退 replicate |

这是结构事实，没有测真实速度。若这些 query 应按照正文使用 MQ，则会低估 A5/A6 在 private-heavy GQA 上的收益；不能把它描写为 baseline 被削弱。MHA 单 query 私有扫描本来也不需要多 query MQ，影响范围应限定。

## 公共模型与 Ramulator 证据的边界

1. **PIM scan 时间没有发现按档手动乘 speedup 系数。** `2031–2188` 把 extent groups 给真实 accelerator；wrapper `647–654` 调 Ramulator，`699` 以 `tCK × cycle` 转换。MQ 在 `72–99` 计算 `nCCDAB`，经 YAML 注入 timing（`330–334`），并改变 trace 命令扩展，属于机制模型输入而非事后拟合。其具体频率/能量 clamp 依据仍需和 RTL 原始证据配对，代码注释“RTL-backed”不等于本轮重新验证 RTL。

2. **整个 PIM 方面并不全是 Ramulator。** read/write 仍是 `bytes / bandwidth`，其能量为 `bytes × AttAcc mem`（I02）；decode 的 `PIM` store 在 `3538–3544` 也是这种模型。可以采用这种范围有限的 analytic 模型，但不能声称每一项 PIM latency 都来自 Ramulator。

3. **不物理互斥的资源别名。** scheduler `2291–2297`、incremental scheduler `2320–2326` 以整个字符串为资源。轻量构造 `PIM:pool0-14` 和 `PIM:pool0-0` 两个各 1 秒独立事件，二者均安排 `[0,1]`，同一 channel 被同时占用。A4c 后 master store 的 15-channel 资源与 scan 单通道资源尤其会触发，不能将 overlap 全算成公平收益。原生 event core 也需要按相同硬件资源集合核验，而非只做字符串 scheduler 一致性。

4. **trace row 地址不是持久表。** `793–811` 每次 scan 都从 channel base/cursor=0 重新拼 extent；`1551–1600` 持久的是 fingerprint→slot，不是正文的 `(channel,row)`。实际 `_append_placement_pim_scan` 在 `2092` 也承认地址/行映射报告有近似。只能确认真实 Ramulator 给这些合成 extent 定价，不能确认它模拟了贯穿整个 DAG 的同一物理 KV 行。

5. **cache 等价键漏掉 extent 间的 row 等价关系。** `Ramulator._address_mapping_signature:234–260` 删除 absolute row index；`608–612` 对各 extent 独立这么做。`address=64` 与 `1088` 得到完全相同 signature。因此 `(0,64)` 的两 extent 同 row 与 `(0,1088)` 的两 extent 不同 row 可发生键碰撞。单 extent 平移可以等价，多 extent 不能逐个删 row 后假定等价。当前生成器常把每个 extent 行对齐，降低部分触发机会，但 cache key 通用正确性尚不成立；也未绑定 simulator/generator build hash。

6. **多 head 折成长 sequence 缺少等价证明。** `2138–2140` 把 `numOp=1`、heads 折入 rows；generator `241–253` 对整份 extent 只写入一次 Q、做一次 SFM。当 heads/HBM>16，多个不同 head 共处一 channel，Q 和 softmax 状态必须区分。论文的 LLaMA-7B/1HBM 正有此配置。没有运行数值验证，不能断言具体误差大小；但不能凭 MAC 总数大致相等就证明 Q 装载、softmax、输出的 head 边界成本也相等。这是公共模型风险，不是已证明某一档专属优化。

7. **能量常数有 AttAcc 血缘，操作数/维度仍有问题。** `src/config.py:67–118` 保留 AttAcc GPU/PIM 能量表；wrapper `686–699` 从 MAC/WRGB/MVGB/MVSB 统计推 traffic。这不逐项计算真实 ACT/PRE 能量，不应把“Ramulator 能量”与“AttAcc 命令/traffic 模型”混为一谈。用户接受 MQ 按新命令数减少 DRAM 项，该下降本身不是不公平。GQA 的 `op.numOp=KV heads`、`op.m=token/query rows`，query group 另存在 `pim_shared_queries`；但 `src/devices.py:530–531` 的 ALU 项只用 `layer.get_flops()`，还需核对 gqa_group 个 Q 的计算是否完整计数。

8. **GPU 扩展有来源声明，仍不足以称为原版不变。** `src/gemm_table.py:1–27` 的 cuBLAS 表来源是外部 A100 benchmark，FlashAttention 表明确称从图近似读取；`src/devices.py:79–128` 再套 occupancy、short-Q padding/split，`src/config.py:168–170` 将 all-reduce fit 的 6.06 us 截距用于 GPU↔PIM 每次 transfer。统一作用各档并不自动消除偏差，因为 A4e GPU attention 和 A5 PIM attention 暴露程度不同。应保存输入表/源版本/读图点并报告 legacy/flash 敏感性，不能直接宣称 H100 latency 实测校准。本轮没有重新访问外部 benchmark 或测 GPU。

9. **GQA 模型不是只改 PIM head 数就完整。** `src/model.py:174–175` decode QKV 仍为 `3*hdim/tp`，prefill 同理。直接构建 LLAMA3-8B 得到 QKV width=12288，按仓库自身 32 Q/8 KV heads、dhead=128 应为 6144。`3588`、`4488`、`3012` 等 KV link bytes 仍按 `2*local_hidden`，四 Q 一 KV 的配置会高估 KV 流量。GPU-readback 较多的路径受损更多，因此 A5/A6 相对 GPU prefill 的幅度也可能被抬高。

10. **decode 仍有公共 GPU local attention。** `3022–3041`、`3360–3384` 对新 token 做 GPU local score/softmax/context，再送 LSE tuple；旧 KV 则 PIM 扫描。论文说 decode always in banks。此共同成本不是 A4c/A4e 偷加优化，但应作为相对正文/AttAcc 的公共执行模型差异披露。不能把“所有 rung 都一样”误当成“正文已实现”。

## 已独立确认的修复

- A1 正常 GPU prefill 已走 `_append_gpu_prefill_layer`；该 helper 的 GPU context 维度用 `op.k=full_rows`（`3621–3624`），旧 shape 问题不能继续照抄。
- fresh/no-reuse 情况已通过 `_resolve_prefill_side`，A5 fresh prefill 不再无条件落 GPU；A6 fresh 也可选择。
- A3b 的 master channel slot 已调用与 A4c 相同的持久 append table（`937` 对照 `912`），修复了“同一 chunk 随 scan 重转 channel”的旧问题；剩余 row/extent 问题见 I03 和上文。
- DIE/TLB metadata 已统一零 latency/energy、且不占独立队列资源；GPU rotation 的额外 Q traffic 还在，未见残留收费 DIE rotation。
- `main.py:542–548` 已把 `num_attacc=num_gpu` 传入 PIM config，旧 CLI 默认固定 8 导致小机器能量倍率错误已修。不要仅检查 `make_pim_config` 函数默认值就断言未修。
- `_heads_per_hbm:1014–1044` 已按全系统 HBM/TP 推导本 GPU 的 local stacks，避免把所有 HBM 重复送给每个 GPU。

## 推荐的签字条件

先让同一冻结 workload/reuse plan 在每档逐请求逐层核对 corrected rows、logical KV bytes、master read/write physical runs、Q/KV/ctx transfer、head/query 数；A3b→A4c 只允许 diff destination/packing 和明确纳入该 claim 的 mask 机制变化，A4c→A4e 只允许已声明 placement 变化，A4e→A5 只允许 prefill side/MQ，A5→A6 只增加逐 request 选边，并使两侧估价对应同一 request 在各自分支的实际工作。

任何不同都应能指到具体 claim 或明确共同模型修正。再做少量实 Ramulator 的命令计数/地址等价验证，并把 source、binary、cache 版本记录到结果。当前测试通过和既有速度表不足以替代这些归因检查。本轮仅审计；上述建议均未修改实现或启动实验。

## 独立轻量探针的复现方式

在仓库根目录执行下列命令，可复现 I01 的修正位置和 I02 的直接 store 定价。它不构造真实 accelerator，不调用 Ramulator；打印的时间是分析式事件价格，不能称为测得的性能。

```bash
python3 - <<'PY'
from types import SimpleNamespace
from src.workload import Request, Segment, Workload, build_reuse_plan
from src.ablation import PRESETS
import src.workload_runner as w
w._EC = None
wl = Workload('supervisor', (
    Request('a', 0, None, 1, (Segment('doc', 'shared', 256),), 256),
    Request('b', 0, None, 1, (Segment('user', 'private', 8),
        Segment('doc', 'shared', 256)), 264)), {})
acc = SimpleNamespace(peak_memory_bandwidth=1e12, num_hbm=1,
                      energy_table={'mem': 1})
system = SimpleNamespace(devices={'Acc': acc})
for rung, cls in [('A3b', w.NaiveKVLayout), ('A4c', w.LocalDiffKVLayout)]:
    plan = build_reuse_plan(wl, 'recompute', epic_prefix_recompute_tokens=8,
        recompute_canonical=PRESETS[rung]['kv_mapping'] != 'naive')
    print(rung, 'corrected rows:', plan.reusable[0].epic_prefix_rows)
    tlb = cls(256)
    for row in range(256):
        tlb.reserve(0, 'r', 'chunk', row, 'master')
    tlb.chunk_order = ['chunk']
    tlb.finalize()
    locations = [tlb.locate(0, 'r', 'chunk', row, 'master')
                 for row in range(256)]
    events = []
    w._append_channel_kv_stores(system, events, layer=0, tier=0,
        request='r', name='store', locations=locations, dbyte=2,
        deps=(), positions=())
    print(rung, 'store:', [(e.device, e.time_s, e.energy_nj) for e in events])
PY
```

其他已执行探针的最小构造步骤如下，均只调用 Python 布局/调度或固定返回值设备桩：

- **I03：** 将上述 reservation 扩为 512 行；只取 `range(128,384)` 的 locations，调用 `_striped_append_channel_extents(locations, policy=tlb.layout_policy, heads_per_hbm=4, tlb=tlb)`，对比各 channel 的 extent 长度 `[128,128]` 与 `[256]`。
- **I04：** 顺序 reserve 两个 fingerprint `c0,c1` 的 producer master 各 256 行，再连续 reserve 同一 consumer 的 `c0,c1` diff 各 8 行；设 `chunk_order=['c0','c1']`、finalize，仅把两个 diff 组交给上述 extent 函数，`heads_per_hbm=1`。对比两条 8 行通道与一条 16 行通道。
- **I07：** 构建单层 `Transformer(num_heads=8, hdim=1024, gqa_size=4)`；两个请求各含共同 16 行 shared segment 和不同 8 行 private segment，`lout=1`。GPU/PIM 设备的 `get_time_and_energy` 固定返回 `(1e-6,(1.,))`，`get_time_and_energy_runs` 为每个 `op.pim_kv_runs` 返回同值。调用 `run_reuse_prefill(..., warm=False, pipe=True, kv_mapping='master-diff-table-local', pim_prefill_mode='pim', pim_batch_command='mq', pim_pe_freq_ghz=1.3004)`，分别用 `cacheblend_batch_size=1,2`；用 `unittest.mock.patch` 包装 `_append_placement_pim_scan`，捕获 decode 调用的 `op.pim_shared_queries`、`op.pim_batch_command`、`op.pim_pe_freq_ghz` 后转发原函数。缺失字段和 I07 表格一致。此桩只验证参数流转，不验证 MQ 的真实周期。

## 后续存储专项：五档均存在 store 与 scan 两套物理映射

本节独立检查真实布局 helper 的输出，不以 TLB 地址与 trace 地址数值单位不同作为错误证据。**channel 字段和 store 所占资源都不一致**，所以当前确实存在两套布局：store/readback 使用 `KVLocation.key_address/channel_base/channel_count`，scan 则使用 fingerprint slot 重建通道和 extent；A4e 的软件表尚未成为写入端和扫描端共同遵循的物理位置表。

### S01：最小稳定 channel 反例

对五档分别建立实际布局类：A3b=`NaiveKVLayout`，A4c=`LocalDiffKVLayout`，A4e/A5/A6=`TableLocalDiffKVLayout`，全部 `bytes_per_vector=256`。严格按以下顺序 reserve：producer 的 c0 master 256 行 → consumer 的 c0 diff 8 行 → consumer 的 c1 master 256 行。设 `chunk_order=['c0','c1']`，`chunk_coread=[frozenset(['c0','c1'])]` 后 finalize。这与 `_prepare_cacheblend_tlb:2892–2895` 只把非 diff fingerprint 放入 chunk_order 的规则一致。

取 c1 的 256 个 `KVLocation`，分别交给 `_append_channel_kv_stores` 和 `_striped_append_channel_extents`，scan 用 `heads_per_hbm=1`：

| 档 | c1 TLB 地址中的 channel | 实际 store 资源 | 实际 scan channel |
|---|---:|---|---:|
| A3b | 2 | `PIM:pool2-2` | **1** |
| A4c | 0 | `PIM:pool0-14` | **1** |
| A4e | 0 | `PIM:pool0-14` | **1** |
| A5 | 0 | `PIM:pool0-14` | **1** |
| A6 | 0 | `PIM:pool0-14` | **1** |

A3b 的证据最直接：它的 allocator 轮转计入 diff，所以 c1 是第三个对象、写在 ch2；scan 的 master 表过滤 diff，所以 c1 的 slot 是 1、读在 ch1。保持一个 head、没有更换 workload，也没有涉及字节单位转换。c0 的 diff 则写在 ch1、扫描时才被追加成 slot2，因此也不复用写入端决定的 channel。相关源代码：allocator `1823–1844`，master slot `890–898`，diff slot `948–959`，store `2203–2228`。

A4c 之后更强的 diff 反例：同一 reserve 构造，scan 改为 `heads_per_hbm=2`，其 per-head diff 应分别位于 ch7/ch15。实际 helper 确实生成 ch7 和 ch15 各 8 行，但 store 端仍只有 `PIM:pool15-15`，没有占用或写入 ch7。这四档全同：旧 TLB 把 diff 固定在全局 ch15，新 scan 将它分布到各 head 的本地末通道。`LocalDiffKVLayout` 的说明也明确“allocation 不变，改的是 scan placement”（`1595–1602`），这与代码行为一致，而不是 audit 将两个字段误作同一单位。

这些 helper 测试揭示输入映射不一致；没有运行命令级 DRAM 读写仿真，不能将结果说成“真实设备读到错误数据”。但也正因 store 从未经过一个与 scan 共享地址的命令流，现有模拟不能证明写入与读取的是同一块存储。

### S02：相同 channel 上的 trace row 仍随 scan 组合变化

在每个布局中按序 reserve 17 个 master c0…c16，每个 256 行，固定 chunk_order，空 coread，`heads_per_hbm=1`。A3b/A4c 的 append 表令 c0/c16 同 slot0；A4e/A5/A6 的空共读表也令它们同 slot0。比较只扫描 c16 与依次扫描 c0+c16：

| 五档均得到的结果 | c16 的 scan K 地址 | channel |
|---|---:|---:|
| 只扫 c16 | 0 | 0 |
| 扫 c0，再扫 c16 | 1024 | 0 |

同一 TLB、同一对象的 scan 地址改变，TLB 位置没有迁移，也没有 copy 事件。这里比较的是两次 trace 中相同单位的地址，不是拿 TLB 的 vector stride 和 trace 的 token stride 硬作数值相等比较。原因是 `_channel_extent_addresses:802–811` 每次把 cursor 重置为 0，再按本次请求的 extent 顺序重新布局。它可以用于某些地址等价的孤立性能探针，但不能据此宣称实现了贯穿请求/层/scan 的持久 `(channel,row)` 存储。

复现本节不需要创建新源文件：沿用前面的 Python shell 和 `acc/system` 桩，将 reserve 顺序改为 S01；打印 `location.key_address // (1<<30)`、store event 的 `device`、extent helper 返回的 channel 即可。S02 则在同一 finalized TLB 上分别传入 c16 的 locations 和 c0+c16 的 locations，比较返回的 `[(key,value,rows)]`。本轮已经分别对五个档执行这些调用，A5/A6 的类通过其实际 preset mapping 确认与 A4e 相同；未运行选边或计时实验。

**专项结论：** 持久 master slot 的修复是有效进展，但不能替代写入与读取共同使用同一物理地址表。应先规定 `(layer,owner,fingerprint,row,head)` 的唯一物理位置，让分配、store、readback、scan 都消费这个位置；A4c 仅改变 diff 的位置，A4e 才改变 master 的位置。当前不能确认五档存储与扫描已经一致。
