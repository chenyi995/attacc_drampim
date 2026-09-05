# 存储与扫描专项：当前七档没有统一的物理读写账本

日期：2026-09-05。源码对象仍为 `cdd89db04a85edae029fd3151165f1a488d6139c`，没有修改实现。本报告补充 [主审报告](REAUDIT_cdd89db.md)，并按用户最新澄清修订 A6 的判定标准。独立 agent 的交叉检查见 [S01/S02](INDEPENDENT_REAUDIT.md)。

**直接回答：目前不能说每一档都按此前写入的实际存储扫描。** A1 默认 decode 是按 token 数生成的 dense/padded scan；A2 没有物理地址模型；A3b–A6 的 store/readback 与 scan 使用两套通道/地址规则。Ramulator 对收到的 scan extent 计时，并不验证 KV 是否曾写在该位置。

这里的“store”指模拟器登记的写入位置、流量、资源和依赖。本轮没有运行真实 DRAM，也没有数值 KV 数据读写验证；以下反例证明**模型内部不一致**，不冒充实际硬件读错数据的实验。

## 1. 逐档结论

| 档位 | 存储/写入依据 | attention 扫描或回读依据 | 能否确认对应 |
|---|---|---|---|
| A1 | `NoReuseKVLayout` 为 request/layer 分配 private affine block；store 用其位置与 16-channel 元数据 | GPU prefill；默认 PIM decode 的 `slice` 分支按读集总数向上凑 256-token chunk，再生成 channel-base runs | **不是从 private block 地址直接生成 scan。** 可作为独立 dense baseline 抽象；须另证地址与 padding 的时序等价 |
| A2 | 远端 KV 以 `kv_gpu_to_remote` 链路事件和逻辑行数表示，无持久地址 allocator | GPU attention；`kv_remote_to_gpu` 也只按行数收费，无 PIM scan | **没有可核对的物理映射。** 另复现 decode 每步多回读一个尚在本地生成的 token |
| A3b | `NaiveKVLayout.finalize()` 对 master/diff pages 按 reserve 顺序一起轮转，store 用其 channel | master 用过滤掉 diff 的 fingerprint slot 表；diff 在扫描时另追加 slot；row 按本次读集重建 | **明确不一致。** 单 head master 可写 ch2、扫 ch1 |
| A4c | 继承旧 `CacheBlendTLB`：master pool 0–14、diff pool 15；store/readback 按该 pool 计费 | master append slot；diff 扫描分配到每个 head 自己的末通道 | **明确不一致。** 两 head diff 扫 ch7/ch15，store 却只登记 ch15；row 也不持久 |
| A4e | 与 A4c 相同 allocator；软件表不参与实际 store channel/row 的选择 | scan 的 master 改用 co-read table；diff 同 A4c | **表只控制扫描端的新布局，没有打通写入端。** 不能称完整的“写入时放置” |
| A5 | 同 A4e，增加 PIM prefill 前的 KV landing 事件 | prefill/decode 都使用上述重建的扫描布局，MQ 为允许的机制包 | **继承不一致。** 有 landing 依赖不等于 landing 和 scan 位于同一地址 |
| A6 | 同 A5 | 逐 request 选择 GPU/PIM prefill；GPU readback 用旧 pool，PIM scan 用新布局，decode 同 A5 | **继承不一致。** 自动选边本身符合用户澄清，不能修复两套存储模型 |

这张表不要求 A1/A2 与 A3b 单变量比较，也不把“没有命令级 store trace”本身判为不公平。允许用 bytes/BW 建模普通读写，但它必须消费与 scan 一致的物理分配、通道集合和 head 数，并披露简化范围。

## 2. SS01：写入地址与 Ramulator 输入在哪里分开

完整路径中至少有三个容易混淆的地址视图：

1. **存储元数据：** `reserve → finalize → KVLocation → _append_channel_kv_stores`。后者也用于 `dram_read_resident`，按 `KVLocation.channel_base/channel_count` 分组，按每组 bytes/BW 与 bytes×AttAcc 单价计费。见 [allocator](../../src/workload_runner.py:1396)、[store/read helper](../../src/workload_runner.py:2192)。
2. **给计时器的 scan 地址：** `_placement_channel_runs → _striped_append_channel_extents → _channel_extent_addresses`。前两步虽使用 `tlb.scan_runs` 的连续段**长度**，却通过 slot 重选通道，并从每通道 cursor=0 开始重新安排地址。见 [extent 构造](../../src/workload_runner.py:793)、[slot 放置](../../src/workload_runner.py:870)、[scan 定价入口](../../src/workload_runner.py:2138)。
3. **事件报告中的地址：** `_append_placement_pim_scan` 把旧 `KVLocation` 的地址轮流分给活跃 channel，用作 `event.dram_addresses`；源码自己称其为 report approximation。其后更新 per-channel 行数，并没有将这些地址替换为实际计时 extent。见 [report 地址分派](../../src/workload_runner.py:2075)、[数量修补](../../src/workload_runner.py:2091)。

因此，检查报告里“有 K/V 地址”“scan 依赖 TLB plan”或统计 scan 行数，还不能证明真正的计时输入和 store 相同。现有 [validator](../../src/workload_runner.py:2482) 检查图拓扑、地址存在、link bytes 等，没有逐对象/版本/head 对照 store 地址与送给 wrapper 的 extent。

wrapper 优先消费 `op.pim_kv_extent_groups`，不是报告里的 `dram_addresses`。见 [wrapper](../../src/ramulator_wrapper.py:524)。generator 再沿各 extent 的 K/V base 产生 MAC 地址，见 [score](../../pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py:162) 与 [context](../../pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py:216)。`PIM_WR_GB` 在这里是 query 写入 GEMV buffer，不是把此前所有 KV store 纳入同一条 DRAM 写入流的证据。

## 3. SS02：不是地址单位差，而是通道和对象身份对不上

TLB 常用 256 B/head-vector stride，trace 使用分散到 banks 后的 token stride。**不能只因两个地址整数不同就判错。** 本轮因此专门构造不依赖这种数值比较的反例。

按顺序 reserve：producer/c0 master 256 tokens → consumer/c0 diff 8 tokens → consumer/c1 master 256 tokens；`chunk_order=[c0,c1]`，二者在 co-read 集合中。主审与独立 agent 分别调用实际 allocator、store helper 和 placement helper，结果一致：

| 档位 | c1 store 资源 | c1 scan（1 head） | diff store 资源 | diff scan（2 heads） |
|---|---|---|---|---|
| A3b | `PIM:pool2-2` | ch1 | `PIM:pool1-1` | ch2/ch10 |
| A4c | `PIM:pool0-14` | ch1 | `PIM:pool15-15` | ch7/ch15 |
| A4e | `PIM:pool0-14` | ch1 | `PIM:pool15-15` | ch7/ch15 |
| A5 | `PIM:pool0-14` | ch1 | `PIM:pool15-15` | ch7/ch15 |
| A6 | `PIM:pool0-14` | ch1 | `PIM:pool15-15` | ch7/ch15 |

A3b 的单 head 例子最直接：store 轮转计入 diff，c1 是第三个对象，所以在 ch2；scan 的 master 顺序过滤了 diff，所以 c1 是 slot1。此时不存在跨 head/stack 换算。A4c 之后，scan 中每个 head 的本地 diff 通道也没有对应的 store 资源展开。master 的 broad pool 包含 ch1，单凭该集合不能断言 ch1 从未写入；能确认的是 store 没有按 table 的最终通道分配字节和争用。

这也解释主报告 R01 的零 diff master 15 倍读写差率：A3b 用 1-channel 带宽，A4c 用 15-channel 带宽，而零 diff scan 的 master slot 已经相同。这是声明之外的读写模型差异，不是 diff gather 的收益。

## 4. SS03：同一对象的 trace row 随读集改变

每档 reserve 17 个 master c0…c16，每个 256 tokens。固定 chunk_order、空 co-read、1 head。让同一 finalized TLB 先只扫 c16，再扫 c0+c16：

| 五档 A3b/A4c/A4e/A5/A6 的共同结果 | c16 的 scan K 地址 | scan channel |
|---|---:|---:|
| 只扫 c16 | 0 | 0 |
| 扫 c0 后扫 c16 | 1024 | 0 |

比较的是两次 scan **同一单位**的地址，期间没有修改对象、重新分配或迁移。append 表里 c0/c16 同 slot0；空共读关系的 table 也让两者落在 ch0。因此这不是“channel slot 还会变”的旧问题，而是 `_channel_extent_addresses` 没有持久 row。

某些孤立、冷启动、只平移整行的扫描可能时序等价；本反例不声称地址从 1024 变 0 必定减少周期。它否定的是贯穿生命周期的物理身份保证。若要保留规范化地址来减少缓存键，必须先有固定物理账本，再证明转换保持 channel/bank/row 冲突、共享、顺序和实际 MAC 数；不能让各次读集自行成为一份新存储。

同理，A4c+ 的 diff 分支会把本次可见 repair run 长度求和后生成一条新 extent，未保留未被本次读取的其他 diff 所占空隙。跨 request/turn 的布局收益、容量、碎片和 ACT 数不能仅由这种读集打包保证。

## 5. SS04：A1/A2 的存储抽象与数量边界

**A1。** 两个各 16-token 的 private block，allocator K base 分别为 0、8192，`tlb.scan_runs` 分别保留这个 base 和 16 tokens。默认 decode 使用的 `slice` placement 却为二者都返回 `(K=0,V=8388608,rows=256,ch0,count1)`（1 head 控制组）。说明计时按 count/dense chunk 构造，不消费原 private block base。见 [private allocator](../../src/workload_runner.py:1653)、[slice 分支](../../src/workload_runner.py:2014)。

256 是被计入 scan 的 token 数，不是“16 个实际 KV 被写了 256 次”，也不是性能差率。按完整 DRAM row 扫描可以是合理 baseline，但需要说明 padding 的 MAC、mask、分配容量如何对应；不能拿这一路证明逐地址存取。另一个可手动开启的 private PIM-prefill helper 会直接使用 `tlb.scan_runs`（[3810 行](../../src/workload_runner.py:3810)）；它不是 A1 preset 的 GPU-prefill 路径，不能用它替默认 decode 背书。

**A2。** 用真实 DAG 构造器、16-token prompt、2 个 decode step、单层 MHA 小模型，事件依次为：

| 事件 | KV tokens | link bytes |
|---|---:|---:|
| prefill 写远端 | 16 | 65536 |
| decode step 0 从远端读回 | **17** | 69632 |
| step 0 计算之后写远端 | 1 | 4096 |
| decode step 1 从远端读回 | **18** | 73728 |
| step 1 计算之后写远端 | 1 | 4096 |

GPU attention 包含当前 token，所以 attention width 可以是 `L+step+1`；但当前 token 的 KV 在 GPU 本步产生，远端读回应该针对此前驻留的 `L+step`。代码 [4143 行](../../src/workload_runner.py:4143) 把同一个 `+1` 也用在计算之前的 remote read 上；本例每步多计一个 token 的回读流量。此问题倾向弱化 A2，但本轮没有测量 E2E 影响。A2 不具备物理 bank/row 地址模型本身是允许的 baseline 简化；这个额外 token 是可以独立修正的数量问题。

## 6. SS05：地址以外，还缺哪些生命周期证据

主要 fresh/PIM-prefill 路径已经把本请求的 KV store 与 scan 通过依赖连起来，decode 也有新增 KV 的 link/store 事件。这些是有效进展，不能笼统说“没有任何写入”。但它们不能解决 SS01–SS03 的地址来源分离。

复用别的请求的 master 时，same-tier consumer 的 scan 还缺 owner store 依赖；主报告 R05 已在真实 DAG+设备桩中复现先读后写。`history_len` 则在 [2866 行](../../src/workload_runner.py:2866) 为每个新 request 创建独立预置 history extent，没有从 parent 的真实地址继承或迁移。允许声明“历史已预置”的抽象，但它不能验证真实多轮写入后继续扫描同一存储的行为。

容量也需要跟最终 scan 布局一致：不能仅用旧 allocator 的 block 大小解释 scan 新建的 per-head row padding、diff packing 与地址占用。当前没有这份完整守恒证明。

## 7. SS06：A6 按用户最新口径复核

用户澄清：“简单的逐个 request 哪边快选哪个”，文中公式可能不准确。因此接受以下定义：A5 固定 PIM prefill；A6 对每个 request 比较 GPU/PIM 两侧预计 attention 耗时，选择较小的一侧；其余布局、MQ、频点不变。

**撤回前一轮“没有构建两套候选 DAG 就违反 claim”的裁定。** 不要求排队试排、事件队列预测或全局最优。`_resolve_prefill_side` 逐 request 比较 `t_xpu/t_bank`、相等选 PIM、跨层沿用结果，符合这项机制定义。正文的旧 DAG/completion wording 应按该定义同步，本次不修改论文。

仍需要核对估价的工作量和执行器一致，不能把近似估计理解为任意省略成本：当前 [3694 行](../../src/workload_runner.py:3694) 先选 **token 数最多**的 lane，只给该 lane 定价；提交扫描则会对全部 lane 定价再取其完成时间。extent 更碎的少 token lane 可能更慢。尾批使用满批价格、GPU readback 的 DRAM 部分、GPU 分支多余 Q 传输、Q variants 的覆盖也仍需清理。第一层估计跨层复用可以是简化，不再单独判 claim 违规；需说明它适用的层结构和逻辑工作量。

轻量设备桩反例只用于验证估价覆盖：ch0=256 tokens/1 extent，ch1=16 tokens/2 extents；设 GPU 三算子各 50 µs、链路每次 20 µs，PIM 每 extent 100 µs + 每 token 1 ns。现有 chooser 只询价 ch0，得到 PIM 140.256 µs、GPU 170 µs，选择 PIM；同一估价器覆盖本反例构造的全部 scan lanes 会得到 PIM 240.016 µs。**这些是人为可检查的价格，不是 Ramulator 测量，也不证明真实 workload 的具体误判率。** 它只说明当前“估计器与提交路径价格完全相同”的说法过强。

逐 request 的选择也不保证整个 workload 的 makespan 恒满足 `A6 ≤ min(A4e,A5)`：选择会改变共享、批次和资源重叠。移除该全局保证不影响用户认可的逐 request 机制。

## 8. 建议怎样改、怎样验收（本轮不实施）

1. 建立一份持久物理账本：至少记录 `(layer,agent/request,chunk,token,version,KV head,K/V)` 到 `(stack,channel,bank,row,column)` 的映射。可保留逻辑地址，但展开到物理位置的变换必须唯一、与当前读集无关。
2. A3b 写入时按朴素 append 分配；A4c 只改变 diff 存放位置；A4e 才让共读表改变 master 放置；A5/A6 共用 A4e 的 allocator。store、GPU readback、scan、资源争用和容量统计都读取该账本。
3. trace 生成只筛选已存对象及其有效版本，保留 row 边界、空隙、head 和 K/V 身份；需要迁移或 compact 时显式记录数据搬移。masked master 与有效 diff 也要可追溯。
4. 报告同时导出对象 ID、写地址、scan 使用的 extent/地址转换和生产依赖。validator 比较真正交给 wrapper 的输入，不再只核对报告的近似地址列表。普通 DRAM 读写仍可使用已披露的解析成本，不强制为了形式把它们全部改成 Ramulator。
5. 用本报告的通道错位、c16 子集、两个 private block、A2 16-token 小例子验收；增加零 diff 的 master 通道/字节/争用相同、跨 turn 地址不变、未写对象拒绝读的控制组。先完成一致性，再重新判断布局带来的 ACT 与端到端收益。
6. A6 保留简单逐 request 比较：两侧按同一请求输入与实际分支操作估价，PIM 覆盖全部 lane 和真实尾批。日志保留 `t_xpu_s/t_bank_s/side`；不需要升级为双 DAG 调度器。DIE/TLB 继续不额外收费，旋转仍归 GPU。

## 9. 证据与验证边界

主审探针：[文本归档](storage_scan_probe.txt)、[JSON 结果](storage_scan_evidence.json)。复现命令（把归档复制到 `/tmp` 后）：

```bash
cp audit/2026-09-05/storage_scan_probe.txt /tmp/fugue_storage_scan_audit.py
PYTHONDONTWRITEBYTECODE=1 KVPIM_CPPCORE=0 KVPIM_PREFILL_SIDE_LOG= python3 /tmp/fugue_storage_scan_audit.py
```

独立 agent 分别复现五档 channel 与 row 反例并自行修订 A6 结论，未依赖主审 JSON。主审还执行 A1/A2 的真实 DAG 构造和 A6 选择函数探针。设备全为桩；未运行性能矩阵、真实 Ramulator、GPU benchmark、RTL、完整测试套件，也未修改代码、测试、workload 或旧性能结果。文档修改原因和核验记录见 [session](../../docs/sessions/2026-09-05-storage-scan-and-request-choice.md)，文件指纹见 [manifest](storage_scan_manifest.json)。
