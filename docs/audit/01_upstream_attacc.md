# 块 01:上游 AttAcc 模拟器基线

归属:AttAcc 官方(scale-snu/attacc_simulator,ASPLOS'24),commit `c1540de`
(+`e7df932` README、`c600051` 缩进修理)。本仓库一切后续层都叠在它上面;
本块讲清它的四件东西:代价模型、设备模型、Ramulator 封装、PIM 命令集。
读者假设:懂矩阵乘法,不了解 LLM/DRAM;概念首现即释。

## 1. 它模拟什么系统

一台 DGX(NVIDIA 的 8 GPU 服务器)+ AttAcc(把注意力算子搬进
HBM (High Bandwidth Memory,3D 堆叠高带宽内存) 的存内计算
(processing-in-memory, PIM) 加速器)。大模型推理分两相:
**prefill**(把输入 prompt 一次算完,矩阵×矩阵为主)与
**decode**(逐 token 生成,每步都要重读整个 **KV 缓存** (KV cache)——
注意力为每个历史 token 存的一条 K 向量和一条 V 向量,矩阵×向量为主、
带宽受限)。注意力 (attention) 每层三步:score(S=Q·K^T,查询与每个
历史 token 的相关度)、softmax(把 S 按行归一化成权重 P)、
context(O=P·V,加权求和)——详见 `../README_fugue_dataflow.md` §0。
AttAcc 的主张:decode 的注意力放 PIM,权重层留 GPU。

**在论文中的角色**:AttAcc 是 Fugue 论文的硬件基线 (baseline)——
实验 A1 与 C1 都是"AttAcc 原样",一切收益都相对它报告
(`docs/EXPERIMENTS.md`)。

## 2. 代价模型:`src/model.py`

- `Layer`(`model.py:7`):一个算子的形状记录。字段:`m/n/k`
  (GEMM 维度,C[m×n]=A[m×k]·B[k×n])、`numOp`(独立重复次数,
  如 头数×batch)、`dtype/dbyte`(数据类型与字节宽)、`stage/name/type`。
  它不算时间——时间由设备模型对着形状算。
- `Transformer`(`model.py:84`):按模型配置(`src/config.py:299`
  `make_model_config`,如 LLAMA-7B/GPT-175B/CACHEBLEND-TINY)生成两串
  Layer 列表:`sum_decoder`(prefill 一层的算子序列:qkv/score/softmax/
  context/proj/ff1/ff2/通信层 comm_x2g)与 `gen_decoder`(decode 每步、
  逐上下文长度增长的序列)。`build(batch, lin, lout, attn_on_hetero)`
  (`model.py:98`)填好全部形状;`attn_on_hetero=True` 表示注意力在 PIM。

## 3. 设备模型:`src/devices.py` 与 `src/system.py`

- `xPU`(`devices.py:8`):GPU 的时间/能耗模型。核心接口
  `get_time_and_energy(layer)`:按 roofline 模型(取"算力受限时间"与
  "访存带宽受限时间"的较大者)给出一个 Layer 的执行时间与分项能耗。
  (47ae0c3 之后它长出 `refined`/`flash` 两档注意力精化,见块 03。)
- `PIM`(`devices.py:406`):AttAcc 侧。GEMV(matrix-vector multiply,
  矩阵×向量)/softmax 类 Layer 转发给 Ramulator 拿实测周期;能耗按
  `ENERGY_TABLE`(每次列读/乘加/SRAM 访问的 pJ 值,1 pJ=10⁻¹² 焦)累加。
- `System`(`system.py:78`):装配 GPU+PIM,`simulate(batch, lin, lout,...)`
  (`system.py:131`)跑一个矩形批(所有请求同形状、padding 到最大),输出
  prefill 总时间与每步 decode 时间/能耗(旧式 CSV 记录)。
  `apply_attacc_pipeline`(`system.py:17`)实现 AttAcc 的算子重叠
  (pipeopt):按论文的事件规则把可并行的层时间合并。

## 4. Ramulator 封装:`src/ramulator_wrapper.py`

`Ramulator` 类(`:53`)把一个注意力扫描变成 trace 文件(逐条 DRAM/PIM
命令的文本流)、调用补丁版 Ramulator2(周期精确的 DRAM 模拟器)二进制、
读回周期数。关键口径:`tCK=0.769 ns`(DRAM 命令时钟周期,`:68`,
秒换算一律以此为准,见 `HANDOFF.md` §3.4)。`run_ramulator`(`:230`)是入口。
(签名缓存 signature cache(`:83-140`)与 MQ 时序模型(`:32-50`)分别是
块 02/块 04 加的,不属于上游。)

## 5. PIM 命令集与 trace 生成

- **命令集**(`ramulator2/src/dram/impl/HBM3-PIM.cpp`,582 行,AttAcc 官方;
  内嵌进本仓库发生在块 02):在 HBM3 之上加 PIM 命令——
  `PIM_ACT_AB`(全 bank 同址行激活 (all-bank row activation))、
  `PIM_MAC_AB`(全 bank 一次列读+乘加)、`PIM_WR_GB`(写 GEMV buffer,
  die→bank 方向)、`PIM_MV_SB`(bank→die 搬运 score)、`PIM_MV_GB`
  (die→bank 搬回权重 P)、`PIM_SFM`(softmax 单元)、`PIM_BARRIER`。
  时序由约束表(preceding/following/latency)驱动,如 nCCDAB
  (相邻全 bank 列命令最小间隔)。
- **trace 生成器**(`ramulator2/trace_gen/gen_trace_attacc_bank.py`):
  按 `--dhead/--nhead/--seqlen/--dbyte` 生成一次注意力扫描的命令流;
  注意力头 (attention head,注意力的独立子空间,每个宽 d_head)沿
  HBM 通道条带化 (striping,头 h 放到第 h mod 通道数 个通道),
  K 与 V 相隔固定 8 MiB 窗口(`_ORIGINAL_KV_GAP_BYTES`,
  见 `src/ablation.py:62`)。
  (`--shared-kv/--shared-queries/--phase/--mq/--pool-base` 等旗标是
  块 02–04 陆续加的,`:548-564`。)

## 6. 上游的边界(后续层为什么要长出来)

上游只会算**矩形批 + 私有 KV**:每个请求自带完整 KV,无共享、无复用、
无按地址的事件模拟(整个批一条时间线)。于是:块 02 造复用执行栈,
块 03 造放置消融,块 04 造 MQ 批命令,块 05 造多轮驻留 KV。

## 7. 回归保证

`tests/test_workload.py` 中 `test_no_reuse_matches_real_attacc_for_a_small_request`、
`test_no_reuse_multi_agent_json_matches_legacy_tier_calls_exactly`、
`test_relay_json_no_reuse_matches_original_attacc_prototype` 等强制:
JSON 工作负载入口在 no-reuse 时与直接调 `System.simulate` **逐位一致**——
上游行为是一切后续层的回归锚点。
