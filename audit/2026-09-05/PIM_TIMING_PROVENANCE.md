# PIM 计量来源、公平性与本轮修正

> **版本更新：** 下文为 `8750b5b` 及当时工作区的来源核查，保留历史证据。当前 `cdd89db` 的 A1/fresh prefill 路由已修，最新判断见 [复审主报告](REAUDIT_cdd89db.md) 和 [计量专项报告](MODEL_PROVENANCE_REAUDIT.md)。bank scan 来自 Ramulator、普通 KV 读写另用公式的区别仍成立；15 倍读写差率、缓存、GQA 与能量维度问题尚未解决。

对象：`8750b5b` 及本轮工作区修改，2026-09-05。本次沿实际调用链检查了七档的 bank scan、外围事件、MQ、缓存与能耗。独立探针为 [pim_provenance.py](pim_provenance.py)，输出为 [pim_provenance.json](pim_provenance.json)。

**结论：核心 bank scan 使用 Ramulator cycles，没有发现按 rung 给最终 scan 时间乘人为收益系数；但不能说所有 PIM 相关开销均由 Ramulator 直接输出。** 当前仍有公式计价的 KV store/read 和系统调度假设。能耗本来就是 AttAcc energy model，不是 Ramulator 原生输出；按用户确认的实验口径，这一点可以接受。

用户进一步确认：旋转按正文在 GPU；TLB 是 A3b 起共同需要的机制；额外 DIE/TLB latency/energy 没有原始 AttAcc 依据，不计入。因此本轮已删除 DIE 旋转节点，并将其余 DIE/TLB bookkeeping 改为不计价的依赖节点。下面区分扫描来源、已移除项和仍待修复项。

## 1. 每档 bank attention 时间来自哪里

| 档位 | 实际路径 | 时间来源及处理 |
|---|---|---|
| A1 | private/contiguous scan → `PIM.get_time_and_energy` → wrapper | Ramulator cycles × tCK；当前错误的 PIM prefill 会按合法 buffer 容量重复 sweep。A1 prefill 应在 GPU 的旧问题仍未修复。 |
| A2 | GPU attention，remote KV 经 link 回读 | 没有 PIM attention scan，所以不应要求一个不存在的 bank attention 项来自 Ramulator。远端访问在 link/带宽模型内处理。 |
| A3b | placement scan → 每 channel 的 extents → `get_time_and_energy_runs` | 每个活动 channel 的 extents 放进一次 Ramulator 仿真，cycles 变成该 channel 事件时长。 |
| A4c | 同一 placement-scan/设备/wrapper 链路 | 同样来自各 channel 的 Ramulator cycles；输入布局由上层构造器决定，Ramulator 不会自动修正上层布局错误。 |
| A4e | 同上，软件表改变 extents 放置 | 同样来自 cycles；没有查到为 table 额外乘一个缩时系数。 |
| A5 | 同上，加 MQ trace 与 nCCDAB override | MQ 在生成的命令流和模拟器时序参数中体现，不是把已有结果简单除以 batch。fresh prefill 仍走 GPU 的旧问题未修复。 |
| A6 | 执行时沿用 A5 的 GPU/PIM 候选路径 | 被选中的 bank scan 仍走 Ramulator；**选择过程**是上层成本估算，不能称为 Ramulator 作出的调度决策。 |

入口证据：[PIM.get_time_and_energy / runs](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/devices.py:463>)、[生成 trace 并执行二进制](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ramulator_wrapper.py:374>)、[解析 memory_system_cycles](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ramulator_wrapper.py:464>)、[cycles 转秒](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ramulator_wrapper.py:684>)。

常规 DAG 路径的时间转换为 `cycles × 0.769 ns`。复用多次相同 sweep 时会乘实际次数；同时工作的 channel 由 DAG 取完成时间。这是重复工作与并发建模，不是曲线拟合；是否真实对应硬件并发，仍依赖上层资源模型正确。

探针在 simulator 边界注入固定 1000 cycles，各档 wrapper 均返回 769 ns，转换倍率均为 1。**这验证的是转换代码，不是实际硬件性能**；没有运行旧 Ramulator 二进制冒充当前源码测量。

CLI 默认 `--engine analytic`，而正式 ladder runner 显式指定 `--engine dag`。analytic 使用平均 scan profile、粗 pool、重复次数和原始 pipeline 公式，不能用它证明当前 A3b/A4c/A4e 的精细布局；运行记录必须保留 engine。

## 2. MQ 是怎样进入 Ramulator 的

[真实 generator](</data2/chenyi9/KV-PIM/attacc_drampim_822/pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py:588>) 对 replicate 逐 query 展开 MAC；MQ 保留每列一次 MAC 命令，query 私有的数据移动与 SFM 仍逐 query 展开。

本次直接运行仓库 generator，固定一条 channel、256-token KV、8 个 query slices：

| 命令 | replicate | MQ |
|---|---:|---:|
| MAC_AB | 512 | 64 |
| WR_GB | 64 | 64 |
| MV_SB | 192 | 192 |
| MV_GB | 64 | 64 |
| SFM | 8 | 8 |

MQ 的一次命令耗时并非仍按普通 MAC 不变：wrapper 先按 `max(基础间隔, PE 吞吐约束, 能量窗口约束)` 求出 nCCDAB，再写入 YAML，由 Ramulator 排命令。在 8 slices、1.3004 GHz、PC 条件下 override 为 8 cycles。见 [MQ 参数计算](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ramulator_wrapper.py:72>) 与 [YAML override](</data2/chenyi9/KV-PIM/attacc_drampim_822/src/ramulator_wrapper.py:329>)。

因此它是“带有 MQ 时序模型的 Ramulator 仿真”。模型参数仍需与所声明硬件对应，但不能将其描述成在 Ramulator 结果之后人为打折。

另外，C++ preset 表里的 `tCK_ps=1300` 会在 `set_timing_vals()` 中按传输率重算为 769 ps；它与 wrapper 的 0.769 ns 对应。不能只读 preset 初值就认定两处时钟相差 1.69 倍。

## 3. DIE/TLB 开销原来来自哪里，现已怎样处理

| 项目 | 修改前来源 | 本轮处理 |
|---|---|---|
| TLB descriptor | 新增常数 5 ns/descriptor，能量常数 0.1；不在 `c600051` AttAcc 中 | 所有档位额外 latency/energy 均为 0，A6 成本探针同样为 0 |
| DIE query-position transform | 新增 `Q bytes / softmax BW` 与 `Q bytes × SRAM energy` | 删除节点；正文中的旋转归 GPU |
| 可选 DIE rotate | 新增 1 ns/variant | 删除该路径，CLI/API 拒绝 `die`；`gpu` 为论文默认 |
| DIE bitmap load、score assembly、LSE merge | 将新操作的估计字节乘/除原有 SRAM 能量或带宽 | 保留必要数据依赖，不额外计 latency/energy |
| 原始 AttAcc PIM 命令、通信、energy table | `c600051` 原模型中已有 | 保留；不能因排除新增 bookkeeping 就把原有工作一并置零 |

虽然部分 DIE 公式借用了 AttAcc 的 SRAM 常数，**新增操作次数/字节数与调度延迟不是原始 AttAcc 模型的独立校准结果**。因此按用户要求移除，而不是把“用了同一能量表”当成这些额外操作已有依据。

TLB 是 naive 与 Fugue 共同的寻址需求，不再以 descriptor 数单独制造 A3b→A4c 的收益差异。真正的地址分散造成的 DRAM 命令差异，应由相同的 Ramulator 模型反映。

零成本节点也不再占用 DIE/TLB 资源队列，避免“duration=0 但仍被别的请求阻塞”的隐性附加延迟。Python 全量/增量调度、独立校验器和 C++ event core 已同步。新报表带有 `die_tlb_accounting` 字段，说明这一口径。

## 4. 仍然不公平或不能保证的部分

### 4.1 A3b/A4c 的 KV store/read 有非 Ramulator 的 15 倍计价差异

`_append_channel_kv_stores()` 使用 `bytes / bandwidth`，没有生成普通 DRAM RD/WR trace；GPU prefill 的 resident readback 也调用该 helper。其带宽取 TLB pool 宽度，与 attention scan 的 per-head channel 布局没有统一。

相同的 fresh 256-token master、无 diff、同一设备参数，探针得到：

| 档位 | store 资源 | 公式时长 |
|---|---|---:|
| A3b | `PIM:pool0-0`，1 channel | 2.097152 µs |
| A4c/A4e/A5/A6 | `PIM:pool0-14`，15 channels | 0.139810 µs |

两者相差 **15 倍**，完全出自公式分母；这些绝对值采用设备桩固定带宽，仅用于证明实现差异，**不是 Ramulator 性能测量**。在没有 diff 的场景也出现该差异，不能归因于 diff 集中布局。这一项尚未修改，是仍需修复的公平性问题。

建议统一真实读写/扫描地址，并把 KV RD/WR 按同一物理通道交给 Ramulator；若暂时保留 AttAcc 带宽近似，也必须按实际相同的通道映射计价，不可继承旧 15-channel pool 元数据。

### 4.2 A1 为减少仿真形状数而增加扫描 token

wrapper 对没有 channel_base 的连续 run 将长度向上取整到 256；其他 channel extents 路径不作同样处理。探针原长度 257，A1 交给 simulator 的是 512，其余是 257。结果虽来自 Ramulator，输入工作量却已经不同；不能用“是 simulator 算的”证明公平。需区分硬件实际最小扫描单位与纯缓存分桶优化。

### 4.3 缓存与构建还不能提供完整来源保证

签名缓存复用确定性、相同输入的结果本身合理，不是拟合；但当前 key 不绑定完整 generator / simulator binary hash。extent key 还独立省略每个地址的 row index：探针中同一行的两个不相交 extents 与不同两行的 extents 得到相同地址签名，丢失相对 row 身份。这里证明的是 key 非一一对应，未测出对当前完整矩阵的影响。

当前 `ramulator2` 目录缺少应安装的 PIM source 和 generator，现存旧二进制不能绑定到当前补丁。需要清洁构建、保留 hash 与实际 YAML/trace/原始 cycles，并验证缓存 on/off 一致；本轮没有执行存在父仓库 reset 风险的旧初始化脚本。

### 4.4 端到端调度和能耗不属于 Ramulator 的直接输出

Ramulator 仿真一条 channel 的 scan；channel/stack 并行、GPU/link overlap 和请求完成时间由上层 DAG 组合。先前发现的读前写入、资源集合交叠与持久地址问题仍会影响公平性。

PIM 能耗使用“命令计数派生 traffic × AttAcc 单位能量 + FLOPs × AttAcc ALU 能量”；DIE/TLB 排除后，这个既有模型继续使用。能量不是 cycles 的另一列原生输出。head/stack 复制与 GQA 工作量仍需正确，不应把计数倍数错误解释成能量模型选择。

## 5. 验证与复现

```bash
make -C src/cppcore
python3 -m unittest discover -s tests -p test_attacc_metadata.py
PYTHONPATH=. python3 audit/2026-09-05/pim_provenance.py > /tmp/pim-provenance.json
```

新增回归检查覆盖所有含 PIM attention 的档位、warm/cold 路径、DIE/TLB 零额外成本、移除 DIE 旋转、保留 PIM scan 时间/能量，以及 Python/C++ 调度的一致性。完整测试日志保存于 [unittest_attacc_metadata.log](unittest_attacc_metadata.log)。原始 [evidence.json](evidence.json) 保留为修改前审计快照，不覆盖。

本轮完整测试：**105 tests passed，74.588 秒**；C++ event core 构建成功。

**可接受的描述是：bank scan latency 由 Ramulator 产生，能耗按原始 AttAcc 模型计算，新增 DIE/TLB bookkeeping 不另计成本。当前仍不能保证整条 PIM 相关路径都由 Ramulator 定价，也不能保证所有档位已经公平；上面的 KV store/read 15 倍差异尤其需要修正。**
