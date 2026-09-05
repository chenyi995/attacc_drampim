# AttAcc 计量来源专项复审（当前树 cdd89db）

本报告由独立 agent `/root/attacc_model_provenance` 撰写。审查基点为 `cdd89db04a85edae029fd3151165f1a488d6139c`，比较原始 AttAcc 基点 `c600051`。范围是设备计量、配置、GPU 模型、Ramulator 包装、命令生成与 PIM 读写计费；主审另查执行 DAG 与 workload。读取了两份当前 session，特别是 `docs/sessions/2026-09-05-ladder-fixes-f01-f02-f04.md` §11 的 MQ 能量裁决。

本次只读取源码、diff、现有文档和目录状态，没有运行 simulator、测试、性能实验、矩阵、RTL 或重编译。以下算例是直接代入代码公式的静态推导，不是测量结果。外部 cuBLAS 数据和 FA-2 原图没有重新下载核验；指出的是仓库现有来源记录的强弱，不把注释当成独立验证。

主审另行查阅了 [cuBLAS benchmark 作者仓库](https://github.com/harshithkantamneni/triton-vs-cublas-llm-benchmarks)，确认其 README 声明 A100-SXM4-80GB、76 个 LLM GEMM 形状和 p50；具名原始 CSV 的抓取未成功，故尚未逐项核验本表数值。[FA-2 论文](https://arxiv.org/abs/2307.08691) 存在。这个来源核验支持“有外部实验出处”，不等于本仓库的 H100、短 Q、额外 occupancy 和流量外推已校准。

文件覆盖：完整读取并比较 `src/config.py`、`src/devices.py`、`src/gemm_table.py`、`src/system.py`、`src/ramulator_wrapper.py`、`pim_ramulator_src/HBM3-PIM.cpp` 与 `hbm3_pim_controller.cpp` 的相关改动；追踪了 `pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py` 的 extent、MQ、phase 与 stripe 分支，以及 `src/model.py`、`main.py`、`src/ablation.py`、`src/workload_runner.py` 的调用与乘数。未修改任何生产代码，也不声称覆盖外部 RTL 工程或全部硬件功能。

## 1. 专项结论

**不能确认全部计量改动已有充分依据，也不能确认 A3b 到后续各档只有声明的计量差别。** 可以确认的较窄结论是：核心 bank scan 的生产路径调用 Ramulator，MQ 间隔在输入 YAML 中设定；在检查到的路径中没有发现按 A3b/A4c/A4e/A5/A6 名称对 Ramulator 返回时间再乘经验加速系数。原 AttAcc 的 `ENERGY_TABLE` 数值保留。

但是，**KV store/read 的 15 倍差率仍存在于当前代码**，PIM 读写不是 Ramulator 模拟，缓存不能证明当前物理布局与当前 binary 的对应关系，GPU flash 计价包含尚未充分校准的外推，部分能量乘数与工作量尚不守恒。不能据此判定有人故意削弱 baseline；可以判定当前结果仍不满足严格的归因保证。

接受用户明确允许的口径：A1/A2 是独立 baseline；A5 的 PIM prefill、MQ、PE 频点是一个机制包；DIE/TLB 额外费用排除；GPU 旋转计算尚为零成本简化。**不要求重新加入这些 DIE/TLB 项，也不把 A5 机制包当成多变量违规。**

## 2. 相对 AttAcc 的改动来源台账

| 改动 | 当前证据 | 判定 |
|---|---|---|
| `ENERGY_TABLE` 的 DRAM/ALU/SRAM/link 单价 | `src/config.py:56–118`；对 `c600051` 的 diff 未改变此表 | 保留 AttAcc 模型，可接受；不能因此自动证明新增事件的操作数正确 |
| PC 默认开启 | `main.py:162–169`、`src/config.py:268`；论文 `06-methodology.tex:53–62`、`04-design.tex:357–367` | 全档相同且与论文操作点一致，不构成 A3b 专属惩罚 |
| MQ interval、容量和 PE 频率 | `src/ramulator_wrapper.py:36–99`；`src/ablation.py:153–160`；论文 `04-design.tex:326–367` | 公式与正文和用户裁决一致；输出周期取自带该间隔的 Ramulator 输入 |
| 多 extent 单 channel trace | `src/ramulator_wrapper.py:416–427,524–531`；trace generator `102–118,151–186,206–222` | 机制有依据，改善先前逐 extent 冷启动求和；缓存及物理映射仍限制其证据效力 |
| channel 并行、head/stack 能量归并 | `src/workload_runner.py:2031–2188`；`hbm_replicas` 在 wrapper `117–148` | 最忙 channel 定时间有依据；不满 stack 的能量还会向上取整，见 MP-05 |
| `num_attacc = num_gpu` | `main.py:541–549`、devices `483,533` | 修复原默认 8 与 1-GPU 运行的口径冲突，公平性改善；其他读写路径没有统一乘数 |
| 新增 MVSB 与 MVGB/WRGB 的方向切换约束 | `pim_ramulator_src/HBM3-PIM.cpp:424–430` | 论文确实声明 TSV turnaround；把 DRAM 的 `nRTW/nWTRL` 套到该内部路径仍是建模假设，源码不是该硬件链路的实测标定 |
| 控制器 ACT 计数 | `hbm3_pim_controller.cpp:28,42,131,162,431–437` | 纯观测扩展，没有发现按档改时间；wrapper 不消费此项，不能称 ACT 能量已经按该计数计算 |
| cuBLAS GEMM table | `src/gemm_table.py:1–27,154–181` | 记录了第三方实测来源；M<128、K<4096 和 H100 是外推，且仍乘额外 occupancy；不能称全部形状实测 |
| GPU flash | `src/devices.py:64–146`、`src/gemm_table.py:15–21,117–128` | 全档共享同一开关，有合理算法动机；FA-2 曲线为人工近似读图，缓存/shape/occupancy 模型未等价验证 |
| X2G 增加 6.06 µs 与 far-HBM 限速 | `src/config.py:165–170`、`src/devices.py:382–389` | 新外推假设；6.06 µs 来源是 A100 all-reduce 的截距，不是已验证的 GPU–PIM 单次传输截距 |
| `apply_attacc_pipeline` 抽取、prefill energy 补报告、路径隔离 | `src/system.py:10–77,108–120,253–266` | 抽取部分对照旧代码保持算术；补报告和 host 路径隔离有正当依据。这不证明新的 DAG 排队与旧 pipeline 逐事件等价 |

## 3. 阻止公平性确认的具体问题

### MP-01：A3b 与 A4c 仍以不同 pool 宽度为同一 master 读写定价（高）

`NaiveKVLayout.finalize` 把 `channel_count=1` 存入每个位置（`src/workload_runner.py:1840–1844`）；A4c/A4e 继承 `CacheBlendTLB.finalize`，其 master 位置保存整个 0–14 pool 的宽度 15（`1409–1416,1457–1472,1485–1491`）。A4c 的 scan 放置已经改成每 head 本地通道，但 store/read 函数仍使用原位置的 pool 字段：

```
B = Acc.peak_memory_bandwidth / Acc.num_hbm * channel_count / 16
t = sum(2 * location.bytes_per_vector) / B
```

见 `src/workload_runner.py:2203–2226`。同一份 master、相同字节数、相同硬件，`channel_count=1` 与 15 直接给出 **t_A3b / t_A4c = 15**。这不是 diff 聚合 claim；即使没有修正，差率依然存在。函数还用于 `dram_read_resident`（`3601–3604,4648–4652`），所以不只是新增 KV 写入。

**偏差方向：** 单个事件明显压低 A4c 及后档的 master 读写时间，相对弱化 A3b；最终 makespan 放大多少取决于依赖和争用，不应把事件的 15 倍写成 E2E 的 15 倍。

**建议：** store/read 与 scan 读取同一份最终物理 channel/extent 账本；零 diff 控制组中 A3b/A4c 的 master 地址和费用必须一致。只修改 pool 常数不足以修正地址与 scan 的分离。

### MP-02：PIM 读写仍是解析公式，使用内部 PIM 带宽及不完整能量复制（高）

`src/config.py:247–281` 的 BA/PC `MEM_BW_PER_HBM` 为 `670.4 GB/s * 9`，是内部 all-bank 运算带宽参数。MP-01 的 store/read 直接使用它，未生成普通 DRAM RD/WR trace，也未证明外来 KV 落地/回读能获得该内部并行率。decode store 则使用整个 `Acc.peak_memory_bandwidth`（runner `3111–3113,3540–3542`），又与 prefill 按 channel 分摊不同。

此外，prefill store/read 的 allocator 每行仅有 `dhead * dbyte` 字节（`4291–4303`），`2226` 的能量直接是这些字节乘 `mem` 单价，没有 scan 路径里的 heads/used-HBM/`num_attacc` 复制，也没有完整 IO 层级流量。`num_attacc` 新修复只覆盖 `PIM.get_time_and_energy[_runs]`，未覆盖这些手建事件。

**判定：** 可以称“bank scan 周期由 Ramulator 提供，KV 读写另用解析模型”，不能称“每一档所有 PIM 部分均由 Ramulator 产生”。带宽乐观性和少计复制会低估相关时间/能量；各档 workload 的暴露程度不同，所以不是只改一个公共系数就一定公平。

### MP-03：多 extent 缓存键丢失 row 关系，且没有 simulator/build 指纹（高）

wrapper `233–260` 去掉绝对 row 号；`602–611` 对一个 trace 的每个 extent 独立做同样的删除。对于一个 extent 独立冷启动，平移整个 trace 的 row 号可能等价；对于**同一 trace 的多个 extent**，不能把每个 extent 的 row 关系独立删除。

静态反例：K extent 为 `[(0,16), (64,16)]` 与 `[(0,16), (1088,16)]`，V 均再加 `1<<23`，其 channel/bank/row-offset/length 缓存键相同；前者两段同 row，后者第二段换 row。trace generator `159–172,216–222` 实際发出的 row 访问序列不同。这里没有运行 Ramulator，因而不声称每个具体样例的最终 cycles 必然不同；足以否定“缓存键相同就必然是同一个物理输入”的证明。

持久 `signature_cache.jsonl` 自动载入（wrapper `190–211`），键 `262–273,591–611` 没有 generator、binary、controller/timing source hash。CSV 也只按 shape/power 等字段匹配（`745–758`）。代码改了仍可读旧 cycles。

**偏差方向：** 命中的是哪种布局决定方向，不能保证公平或保守。建议保留相对 row 等价类/完整归一化地址关系，并把 source+binary+timing 配置版本纳入缓存及结果 manifest。

### MP-04：GPU flash 有明确来源说明，但不足以称已校准，论文平台与入口不一致（中/高）

`gemm_table.py:15–21` 明说 FA-2 A100 效率是近似读图，非本仓库实测；`154–175` 把未测形状插值/夹到最近区间。`devices.py:263–273` 在测得效率之后再乘 tile occupancy，FA 路径也再乘 `wave_util`（`135`），需证明没有重复计入样本原有的利用率损失。

Flash HBM 流量把 K/V 无条件视作一次 off-chip 读取，后续 Q block 从 L2 获取（`106–127`），没有容量/竞争判断；L2 带宽仍为无穷大（config `187,215`）。这对大 KV 或多并发可能乐观，倾向强化 GPU。相反，短 Q 默认不启用 split-K（config `159–164`，guide `21`），当 occupancy 小时可能弱化 GPU；读图误差和 H100 外推方向不能统一认定。

论文 `06-methodology.tex:21–24` 写八个 H100，每 GPU 一 stack；`main.py:123` 默认 A100a，ladder 脚本 `63–74` 不传 `--gpu`。run guide 所列 Tiny/LLAMA3-8B 是 1 GPU/1 HBM，其他模型还改变总 stack/GPU 数。A100a 的 HBM3 3352 GB/s 是原 AttAcc 已有的合成平台，不应误报为本轮偷偷增加 GPU 带宽；但这些入口产物不能直接标成正文所述 H100 平台。

`run_sweep.sh:22` 会统一用 flash；`run_dag_ladder.sh:73` 只是可选透传，裸入口仍回落 legacy。应冻结整组配置并拒绝混合旧 legacy 和新 flash 结果。统一采用更合理的 GPU 模型本身是改善 baseline，不能要求为了复现旧数值保留不合理的弱 GPU。

### MP-05：HBM 最忙负载正确用于 latency，却仍向上复制到所有 stack 的 energy（中）

`_heads_per_hbm` 用 `ceil(local_heads/local_stacks)`（runner `1014–1044`），模拟最忙 stack 有合理依据；`_append_placement_pim_scan` 再把这份满负载能量乘 `ceil(kv_heads/heads_per_hbm)`（`2058–2062,2139,2184`），末个不满 stack 也当作满。

按当前 guide 的 LLAMA-33B（52 heads，`--ngpu 2 --num-hbm 10`）：每 GPU 26 heads、5 stacks，最忙 stack 为 6 heads；最终复制为 `6 * ceil(26/6) = 30` heads，较实际 26 多 **15.38%**。wrapper 内层 `hbm_replicas` 修复了另一层 phantom replication，但未消除此处余数。

**偏差方向：** 所有 PIM 档都受影响，A5/A6 新增 PIM prefill 的能量被进一步高估，可能低估其节能收益。建议 latency 取最忙 stack，而 energy 对实际各 stack 的 head 数求和。

### MP-06：MQ 能量公式裁决正确；报告的 MAC 工作量尚未与完整命令工作量对齐（中）

MQ interval 使用 `E_col + n*E_Q`，不会把 8 tCK 直接当成 6 tCK 的 4/3 能量。这与用户 §11 裁决及论文 `04-design.tex:361–367` 一致。本报告不重新质疑这条裁决。

但完整 `full` trace 同时含 QK 和 PV（generator `151–172,206–222`），`PIM.get_time_and_energy_runs` 的 ALU energy 只计算 score `layer.get_flops()/2`（devices `528–532`；model `51–52`）。对单 head、MHA、m=8、L=256、d=128，完整两次 matmul 共 `2*8*256*128=524288` MAC，而这里是 `262144` MAC。原 AttAcc `c600051:src/devices.py:330–350` 也只给 score 计这一份、context 返回零，因此 **QK/PV 少一份是继承局限，不是新添的 A5 私有优惠**；在用户选择“沿用 AttAcc energy 模型”的口径下，应诚实标注，而不私自加入新单价。

新 GQA 扩展还让实际 query 数为 `len(grouped_positions)*gqa_group`（runner `4595–4598`），但 ALU layer 的 m 仍是 `len(grouped_positions)`（`4579–4583`），少了 GQA 组乘数。不能因为单价来自 AttAcc，就称新 GQA 工作量也已正确计入。

此外，ACT 统计未被 wrapper 解析（`465–484`），列访问仍按 `mac*32*64` 乘包含摊销 ACT 的单价（`686–698`、config `96–97`）；因此“布局 row 冲突的 ACT energy 按真实激活次数计入”不成立。是否需要改变继承的 AttAcc 摊销口径应单独决策，不能把该模型称成 RTL/实测总功耗。

## 4. MQ 与内部带宽的证据边界

### MP-07：P 字节计了，论文的有界双缓冲流式 P 还没有对应的命令交错（中）

论文 `04-design.tex:341–350` 写概率随 V 列到达，context 受 TSV 及其方向切换限制。generator 的实际单 head 主路径却先输出全部 `cmd_context_mvgb`，再 barrier，再输出 context MAC（`544–563`）。MQ 扩展仅重复每个 query 私有命令、保留一次 MAC（`588–610`）；未按 512 B 双缓冲容量把 P 块搬运和 V MAC 交错。

因此当前模型确实计入 P 的总 movement 字节和部分方向切换，并非凭空给 A5 零流量；但不能声称已证明这些输入在有界 buffer 中可执行。先搬完再算可能比真实流式重叠更慢，忽略 buffer 容量又是理想化，净偏差在未验证前不定。需要一个 bounded-buffer 命令依赖检查，与真实 RTL 的 producer/consumer 次序对照；不应仅凭 cycles 来证明功能实现成立。

### MP-08：所谓 HBM3 流效率“从同一 timing 推导”还不精确（低）

`config.py:28–32` 保存 `tCK_PS=1300,nRFC=260` 并从此算 0.855。真实 HBM3-PIM 源码在 `304–305` 从 rate 重算 tCK 为 769 ps，在 `334` 从密度的 tRFC 表重算 nRFC；preset 里的 260 实际是 nRFCSB（`35`），不是运行时 nRFC。故该辅助公式不能宣称完全读取同一运行配置。

这**不是** wrapper 的 0.769 ns cycle 转换错误：该转换与真实重算时钟一致。0.85 的 GPU 主带宽利用率还沿用旧值；辅助 stream efficiency 主要进入 refined/flash 的 far-HBM 估算，通常 NVLink 更慢。建议修正文档或从运行配置生成此表，优先级低于 MP-01–06。

## 5. 当前 checkout 的可复现性限制

默认 `ramulator2/ramulator2` 和 `libramulator.so` 存在（文件时间 2026-08-22）；默认 `ramulator2/trace_gen/` 只有 `__pycache__`，没有 wrapper 所需的 `gen_trace_attacc_bank.py`，且没有 `ramulator2/src/dram/impl/HBM3-PIM.cpp`。不能由仓库中的 `pim_ramulator_src` 自动推出默认 binary 已包含新增 turnaround/ACT 统计。`ATTACC_RAMULATOR_DIR`/`KVPIM_SCRATCH` 可以指向外部安装，历史结果须记录那个实际目录、source 与 binary hash。

本审计没有启动默认安装，也没有运行 `set_pim_ramulator.sh`。脚本 `3` 行含 `git reset --hard`，不能在审计中用它代替只读来源检查。当前结论是“源码路径可追踪，但本 checkout 不能确认实际安装与源码对应”，不是“已有结果必定没有运行 Ramulator”。

## 6. 对四个相邻台阶的专项裁定

| 比较 | 本专项可确认的声明差别 | 仍阻止签字的计量问题 |
|---|---|---|
| A3b → A4c | 相同设备单价、相同 replicate command；diff 本地聚合是接受的 claim | MP-01 的零 diff master 15 倍读写差率；MP-02 的写/读物理账本不一致；MP-03 的 row 关系缓存 |
| A4c → A4e | 相同 GPU/PIM 模型、相同 MQ 开关，未发现额外费率切换 | placement 的 row/缓存可复现性仍不成立；表输入是否合法由主审另查 |
| A4e → A5 | PIM prefill+MQ+1.3004 GHz 按已接受机制包切换；interval 与论文一致 | MP-02/04/05/06 的跨设备计量，MP-07 流式 P 尚未验证；不能从相同单价推出全部比较公平 |
| A5 → A6 | 设备模型和配置相同；未发现仅 A6 获得更快 Ramulator 时间的后处理系数；按用户后续澄清接受逐 request 选较小估价 | 主审核对估价/执行的请求工作量、lane、尾批及操作成本是否一致；不要求两套候选 DAG。见 [存储专项 SS06](STORAGE_SCAN_CONSISTENCY.md)；公共 GPU/能量模型不确定性影响选边 |

建议先关闭 MP-01 与物理账本差异，冻结并记录完整 GPU/PIM/缓存版本；然后做授权范围内的轻量不变量验证。修复前不应给“每处相对 AttAcc 的改动都有充分依据、每一级全部公平”的保证，也不应把当前疑点归因于人为动机。
