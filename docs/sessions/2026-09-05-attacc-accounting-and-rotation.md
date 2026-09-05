# Session：AttAcc 计量口径与 GPU query 旋转

日期：2026-09-05。仓库：`attacc_drampim_822`。会话起始及记录时 HEAD：`8750b5b`；原始 AttAcc 对照 revision：`c600051`。改动保存在工作区，本次没有创建提交。

本文记录整个会话的审计、口径确认、代码修改及原因。**实际实现的修复限于新增 DIE/TLB 成本、其资源排队，以及错误的 DIE query 旋转路径；并未完成七档实验的全部整改。**

## 1. 用户要求与本次工作的范围

用户首先要求以严格审稿标准独立检查仓库，核对论文 `/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027` 的文字、各档差异、公平性和 workload 偏差。随后明确了可接受的实验口径，并要求输出修改建议 README。

本次确认的七档定义：

| 档位 | 本次采用的口径 |
|---|---|
| A1 | 独立硬件侧 baseline，不要求相对 A2 单变量；依当前论文为 GPU prefill / PIM decode，无跨请求软件 KV 复用 |
| A2 | 独立软件侧 baseline，不要求相对 A1 或 A3b 单变量；GPU attention，可使用远端 KV 存储 |
| A3b | 最 naive 的软件复用 + PIM decode 组合，作为后续增量起点 |
| A4c | 集中每个 agent、每个 KV head 的 diff |
| A4e | 加软件 placement table，将可能共同读取的 chunks 分散放置 |
| A5 | PIM prefill 与配套 MQ 作为一组被接受的机制 |
| A6 | 在 A5 基础上增加自动 GPU/PIM 选边 |

用户接受合理的自建 baseline 与合成 workload，以展示机制收益为目标。因此撤回了“强制要求 A1→A2 单变量”“必须把 A5 拆成更多档位”等初始要求；仍保留对逻辑工作量、实际执行设备、有效读写和计量一致性的检查。

后续用户要求检查每档 PIM 结果是否确实来自 Ramulator，并对 DIE/TLB 给出了直接的修改依据：

> “DIE 和TLB的latency 和 energy 是哪里来的？是attacc 吗？如果不是，那么就不要计入……我们只是在attacc 的energy 模型下 算一个energy”

接着明确：

> “根据Fugue 文章正文，旋转不在DIE 上……TLB 如果Fugue（也就是从A4c 开始需要，那么A3b 其实也需要）因此不需要都加”

以上是本次代码变更的授权与计量约束。排除新增成本是遵循用户确认的模型边界，**不表示真实硬件中的 metadata、合并或寻址实际耗时/耗能为零**，也不表示共同成本在任何 speedup 比值中都会严格抵消。

## 2. 会话阶段与产出

| 阶段 | 做了什么 | 产出 / 结论 |
|---|---|---|
| 独立审计 | 对照论文及相对 `c600051` 的 36 个变更文件，追踪逻辑计划、布局、事件、计价与汇总 | [REPORT.md](../../audit/2026-09-05/REPORT.md)、[原始反例](../../audit/2026-09-05/evidence.json)；原有 102 tests 通过但不能覆盖论文约束 |
| 用户确认口径 | 接受 A1/A2 独立 baseline、A5 机制包与合理合成输入 | 更新首页与原报告，创建 [修改建议 README](../README_audit_fixes.md) |
| PIM 来源核查 | 检查 cycles 解析、MQ trace、缓存、带宽公式与 AttAcc 能量表 | [PIM_TIMING_PROVENANCE.md](../../audit/2026-09-05/PIM_TIMING_PROVENANCE.md)、[来源探针](../../audit/2026-09-05/pim_provenance.py) |
| 计量修正 | 排除新增 DIE/TLB 时间与能量，移除其额外资源占用，A6 探针同步 | Python/C++、报表、验证器一致修改 |
| 旋转修正 | 按正文删除 DIE rotate 与额外 position-transform，直接保留 Q 就绪和 KV landing 依赖 | 默认 GPU 路径不再含 DIE 旋转；旧 `die` 模式显式拒绝 |
| 验证与归档 | 构建 native core，新增回归检查，执行完整测试，保留修改前证据 | 105 tests 通过；本 session 文档及索引 |

没有调用其他 agent 代审。没有修改外部论文仓库或外部 workload 文件，没有运行完整论文性能矩阵。

## 3. 为什么排除新增 DIE/TLB latency 和 energy

### 3.1 原来增加了哪些费用

对照 `git show c600051:src/devices.py` 与本分支 runner，发现原始 AttAcc 有自己的 PIM MATMUL、SOFTMAX、通信和 energy table；下列工作却是新 DAG 自行添加的：

| 项目 | 修改前公式 / 行为 | 问题与修改理由 |
|---|---|---|
| TLB descriptor | `count × 5e-9` 秒，`count × 0.1` 的能量项，经 DAG 转成 nJ | descriptor 单价是新增常数；不能以此单独惩罚 A3b 的 descriptor 数，进而宣称是布局硬件收益 |
| DIE rotate | 每个 shifted variant 固定 1 ns | 不属于正文中的 GPU 旋转分工，也不是原始 AttAcc 独立计量项 |
| DIE query-position transform | `Q bytes / softmax_peak_bandwidth`，`Q bytes × SRAM energy` | GPU 产生/发送 variants 后又增加一个 DIE 变换，混入错误设备上的新成本 |
| DIE bitmap load | bitmap 字节、stack 数、SRAM 带宽/单位能量公式 | 新操作的工作量与调度成本没有原始 AttAcc 中的直接对应 |
| DIE score assembly / LSE merge | contribution 数与 tuple 字节乘/除 SRAM energy/BW | 借用了 AttAcc 的常数，但这不意味着新增操作本身已经由 AttAcc 校准 |

**使用同一张 energy table 不足以证明新加操作的能量有依据。** 本次保留原始 AttAcc 模型已经覆盖的工作，排除上述新 bookkeeping 的额外计价。这一点同时适用于 A3b 与 Fugue 各档。

### 3.2 怎样落实到实际计价

在 `src/workload_runner.py` 中：

- 删除 `_DIE_ROTATE_CYCLE_S`、`_TLB_DESCRIPTOR_S`、`_TLB_DESCRIPTOR_ENERGY`。
- `_tlb_plan_cost()` 返回 `(0.0, ())`。这使实际事件和 A6 的成本探针同时排除 TLB 附加项，不会出现执行不收费、选择时还收费的差异。
- `_cacheblend_event()` 和 legacy `_event()` 对 DIE/TLB 统一设置零额外时间、零额外能量，防止另一入口重新加上费用。
- 保留的 bitmap load、score assembly、LSE merge 调用点移除原有字节/带宽/能量公式，显式传入 `time_s=0.0, energy=()`。
- 汇总由实际事件计算；`die_time_s_unoverlapped` 和 DIE/TLB 能量分项因此为零，而不是只在最后总数里减去某个项。
- 增加报表字段 `die_tlb_accounting = "dependency-only; no extra latency or energy beyond AttAcc"`，让新结果声明口径。

TLB descriptor、mask 和地址计划可以继续作为结构信息被记录。**不额外计 TLB 费用，不等于允许丢掉正确寻址，也不等于将散布产生的 DRAM 命令费用一起免除。** 后者仍应由同一 Ramulator 模型对实际地址计价。

## 4. 为什么零时间还必须去掉资源排队

只把 `time_s` 设为零，原来的调度器仍可能通过共享 DIE/TLB 资源给另一个请求制造延迟。例如在一个简化测试中：

1. GPU 工作在时间 10 完成，LINK 工作在时间 1 完成。
2. 先构图的 DIE metadata 等待 GPU，所以就绪时间为 10。
3. 后构图的另一个 DIE metadata 只等待 LINK，本应在时间 1 就绪。
4. 如果两个零时间节点仍共享 DIE 的 `availability`，第二个会被推迟到 10，它后面的 PIM scan 也被推迟。

这会在“显式 DIE latency 为零”的报表里隐藏一笔跨请求排队，不符合本次排除额外开销的要求。

因此，metadata 节点改为：

```text
start = max(自身依赖的完成时间，0)
end = start
不读取、也不推进任何硬件 resource availability
```

GPU、LINK 和 PIM 继续按原有硬件资源规则调度。修改同时覆盖 Python 全量调度、增量调度、独立 overlap 校验器，以及 C++ core；在 `pipe=True` 和 `pipe=False` 下都验证了依赖不丢失、metadata 不创建额外资源等待。

### C++ bridge 与 ABI 为什么也改了

`src/cpp_eventcore.py` 定义 `DEPENDENCY_ONLY_DEVICES = {DIE, TLB}`，将其编码成 device ID `-1`。C++ 对负 ID 不访问/更新资源队列，且在添加事件及重设 duration 时将其时长保持为 0。

旧 `.so` 不支持这个语义，不能继续把负 ID 当作普通资源下标。为避免工作区源码已更新、旧库仍被载入的情况，新增 `ec_abi_version() = 2`；Python loader 若找不到此函数或版本不符，回退到 Python 调度，直到重新构建。这个修改只涉及**模拟程序的调度实现与兼容性**，不是增加被模拟硬件能力。

## 5. 为什么删除 DIE 旋转节点，而不只是把它们置零

按论文正文及用户确认，query variants 在 GPU 上生成。原代码除可选 `die` 模式外，在默认 GPU 路径也无条件加了 `die_query_position_transform` / `decode_die_query_position_transform`。

即便将这些节点计价设为零，保留这个名称与流程仍会声称 DIE 执行了正文没有的旋转，掩盖实际设备分工。因此最终修复删除了这些节点：

- unbatched decode：地址计划直接依赖旋转后的 Q 到达事件。
- batched decode：shared scan 的地址计划依赖本批全部 query 的就绪事件；private scan 依赖对应请求的 query。
- PIM prefill：地址计划同时依赖 GPU variants 到达与本批 KV store 完成，继承旧 transform 节点承载的两类依赖。
- 保留 scan → score assembly/LSE merge → context 返回之间必要的合并依赖；合并节点按上述口径不另计成本。

当前默认路径的结构可理解为：

```text
GPU Q / variants 到达 ─┐
                      ├→ 地址计划（0 额外成本）→ PIM scan
对应 KV 已落地 ───────┘                         ↓
                                  合并依赖（0 额外成本）→ context 返回
```

`main.py` 的 `--cacheblend-rotate-mode` 去掉 `die`；API 也拒绝该值，避免旧命令静默运行成另一个模型。`gpu` 保持论文默认；已有 `bank` 实验选项保留，并在帮助中标为实验性假设，不把它当作正文配置。

**本次没有新增 GPU RoPE 计算时长、没有修复所有 position-delta 推导，也没有完成数值等价性验证。** GPU 路径中已有的额外 Q variant 传输继续按 AttAcc 通信模型计价；删除错误 DIE 工作，不应被解释成完整旋转模型已获验证。

## 6. 文件级修改记录

| 文件 | 实际变化 | 为什么需要 |
|---|---|---|
| [main.py](../../main.py) | CLI 拒绝 `die`；帮助说明 GPU 为正文、bank 为实验选项 | 命令行不能继续允许与本次正文分工冲突的 DIE 旋转 |
| [workload_runner.py](../../src/workload_runner.py) | 清除 DIE/TLB 附加费用；统一零成本事件；去掉 DIE 旋转/transform；转接原有依赖；同步 Python 调度、校验和报表 | 保证实际执行、选边成本、能量汇总与用户确认的 AttAcc 口径一致 |
| [cpp_eventcore.py](../../src/cpp_eventcore.py) | 共享 metadata 设备集合；负 device ID；ABI 检查与回退 | 保持 native/Python 一致，避免加载旧 `.so` 后使用错误资源语义 |
| [eventcore.cpp](../../src/cppcore/eventcore.cpp) | metadata 不预留资源、duration 为零；ABI=2 | 不能让 native fast path 恢复被排除的排队或时长 |
| `src/cppcore/libeventcore.so` | 由 Makefile 重新构建，本地生成产物 | 使当前环境实际运行新语义；源文件才是提交/复现依据 |
| [test_workload.py](../../tests/test_workload.py) | 不再要求 DIE transform 出现；验证 `die` 被拒绝；检查已有 landing 依赖仍存在 | 旧测试将错误节点当作期望，必须按新的正确设备分工调整 |
| [test_attacc_metadata.py](../../tests/test_attacc_metadata.py) | 新增三个回归测试，含多档、warm/cold、两种 pipe 模式及 native/Python 检查 | 防止只改报表、不改时间轴，或只修一条执行路径 |
| [仓库 README](../../README.md) | 修正 A1/A2 与 A5 的消融口径，加入建议文档与本 session 入口 | 首页不再继续宣称所有相邻 baseline 都应只差一个因素 |
| [修改建议 README](../README_audit_fixes.md) | 给出分阶段修法、验收和机制 workload；后续标记本次已落地的计量修正 | 区分建议、用户已确认约束与实际完成项 |
| [审计报告](../../audit/2026-09-05/REPORT.md) | 保存初始发现，补充用户口径和代码修正说明 | 避免把已经接受的实验设计继续当作违规，也不抹掉原始反例 |
| [PIM 来源报告](../../audit/2026-09-05/PIM_TIMING_PROVENANCE.md) | 逐档记录 Ramulator/公式/AttAcc energy 来源与残余问题 | 回答“是不是都来自 Ramulator”时区分 scan、外围与能量 |
| `audit/2026-09-05/reproduce.py`, `evidence.json`, `unittest.log` | 初次审计探针、原始结果和 102-test 日志 | 保存修改前可复核的证据；`evidence.json` 未覆盖 |
| `audit/2026-09-05/pim_provenance.py`, `pim_provenance.json` | 新增计量来源探针及结果 | 验证 cycles 转换、MQ 命令数与公式差异，不冒充真实硬件测速 |
| [manifest.json](../../audit/2026-09-05/manifest.json) | 分别记录初始输入、后续文档与实现修改的 SHA-256 | 区分相同 HEAD 下不同工作区状态，避免旧结果与新代码混用 |
| [session 索引](README.md)、本文、[主文档索引](../README.md) | 新增可发现的 session 记录，链接修改原因与证据 | 用户要求每次改动详细记录“为什么修改” |

本轮没有修改 `src/devices.py`、`src/config.py`、`src/ramulator_wrapper.py` 或 PIM trace generator 的生产实现。原始 AttAcc energy table、Ramulator 命令时序及相关代价没有因 DIE/TLB 修正而被整体替换。

会话开始前已存在的未跟踪目录 `experiments/paper_ladder/` 不是本次生成的结果，没有将其当作当前版本的完成实验。

## 7. PIM 结果来源核查：为何不能宣称全部公平

已确认的正面证据：

- bank scan 经 wrapper 执行 Ramulator 或使用缓存 cycles，通常只乘 `tCK` 转成秒，没有查到按 rung 给 scan 结果乘人为收益系数。
- 在外部 simulator 边界统一注入 1000 cycles，A1/A3b/A4c/A4e/A5/A6 的转换均得到 769 ns，倍率为 1。
- 运行真实 generator 的 8-query 例子，replicate/MQ 的 MAC_AB 为 512/64，其他 query 私有命令数保持一致；MQ 时长约束通过 YAML 的 nCCDAB 进入 Ramulator，并非事后将 cycles 除以 batch。
- 能耗是命令计数派生 traffic 与 FLOPs，再使用 AttAcc 单位能量计算；用户接受的是这个模型口径，不要求 Ramulator 自身输出能量。

仍未修复的项目：

| 项目 | 状态与理由 |
|---|---|
| A3b/A4c KV store/read 带宽差异 | 无 diff 的相同 256-token master，A3b 按 1 channel、A4c 起按旧 15-channel pool 计价，公式时长差 15 倍；不是 Ramulator 给出的收益，仍需统一物理地址与读写计价 |
| A1/A5 prefill 执行设备 | A1 实际跑 PIM prefill、A5 fresh prefill 实际仍跑 GPU 的原有问题未在本次一起修复 |
| 修正集合与 master 地址 | 默认 runner 的 rung-dependent canonicalization、A3b 每次 scan 改地址等初始问题仍未整改 |
| 资源集合与有效数据 | 本次仅取消 metadata 排队；PIM pool 集合重叠、跨请求读前写入等是独立问题，没有因此解决 |
| GQA / GPU-stack 能量倍数 | 能量模型来源可以接受，但工作量和复制倍数错误仍需单独修复 |
| A6 的事件完成时间选择 | 本轮仅移除了 TLB 估计项，没有把静态成本比较改成完整候选 DAG 试排 |
| Ramulator 构建和缓存 | 本地旧二进制/缺少安装源文件、cache 缺少完整构建 hash 等问题仍存在；没有据此做新性能结论 |

上述事项之所以未混入本次代码修正，是因为本轮明确执行的是 DIE/TLB 计量与旋转分工调整。其他发现已记录为后续整改，不能在仅通过本轮测试后宣称整套 ladder 已经公平或论文结果已经重现。

## 8. 验证：执行了什么，能证明什么

| 验证 | 实际结果 | 证明范围 |
|---|---|---|
| 初始 `python3 -m unittest discover -s tests` | 102 tests passed，83.278 秒 | 初始实现的已有测试；不是本轮修正后的结果 |
| `make -C src/cppcore` | 构建成功，使用 Makefile 的 gcc-toolset-11 编译器 | 新 native 资源语义可构建、当前库已更新 |
| `python3 -m unittest discover -s tests -p test_attacc_metadata.py` | 3 tests passed，0.053 秒 | metadata 零成本、不额外排队、Python/native 一致、多档 warm/cold 的结构不变量 |
| 修正后 `python3 -m unittest discover -s tests` | **105 tests passed，74.588 秒** | 新增三项及原有测试全部通过，没有通过删除整个旧测试规避不一致 |
| `PYTHONPATH=. python3 audit/2026-09-05/pim_provenance.py` | 执行成功，结果保存为 JSON | wrapper 注入边界、真实 generator 命令数、store 公式差异；不是真实 Ramulator 性能 |
| `git diff --check`、文档链接检查 | 通过 | 工作区变更格式和本地文档目标有效 |

新回归测试使用不提供 DIE SRAM/bandwidth 标定的设备桩，确保相应执行路径不再依赖这些新成本。各含 PIM attention 的 rung 均检查：DIE/TLB time/energy 为零，PIM scan time/energy 仍为正，默认路径没有 DIE rotation/position-transform。A2 没有 bank attention，不伪造一个 PIM 项要求它通过这类检查。

完整修正后测试日志：[unittest_attacc_metadata.log](../../audit/2026-09-05/unittest_attacc_metadata.log)。测试中的任意时间值只用于依赖与调度验证，不是 GPU/PIM 的真实测量。

没有执行 Ramulator 的干净完整性能复现、没有重跑论文 588 点矩阵、没有做 RTL 综合或数值 attention 等价性测试。本文补写本身仅改文档，未因此再次运行已经通过且未发生行为变化的完整测试。

## 9. 结果、兼容性与后续复现要求

1. 新结果是“bank scan 使用 Ramulator、能耗使用 AttAcc、额外 DIE/TLB bookkeeping 排除”的口径；不能称为测量了 Fugue 新增逻辑的实际物理能耗。
2. 移除附加项会改变 makespan/energy，也可能改变 A6 的选择。没有对完整 workload 测量变化幅度，不保证所有 speedup 朝同一方向变化。
3. 旧 `--cacheblend-rotate-mode die` 明确失败；使用 `gpu` 才与当前正文默认一致。事件数量、ID 和部分依赖边随删除节点改变，旧事件级输出不能当作新版本输出。
4. 旧事件 core 库会回退 Python；重新构建后再使用 native 路径。不要让 source 已更新、binary 仍旧的运行进入最终汇总。
5. 旧 DAG report 含旧 metadata 成本，需要重新生成。Ramulator 的独立 scan cache 不是本轮直接修改对象，但其来源问题仍须在正式重跑前解决。
6. 原始 `evidence.json` 按审计当时的快照保留；当前修改与新增 probe/test 结果由 manifest 的后续记录区分。后续修复其他问题时另写 session 或在本文明确追加，不覆盖为已完成的历史。

本 session 的直接完成条件是：**记录清楚上述修改的用户依据、原有错误、所选修法、接口影响、验证证据与未完成边界。** 整体 ladder 的后续整改仍按 [修改建议 README](../README_audit_fixes.md) 推进。
