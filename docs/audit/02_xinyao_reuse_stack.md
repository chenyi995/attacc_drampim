# 块 02:KV 复用执行栈(workload / ReusePlan / 物理 TLB + 事件 DAG)

归属:xinyao,commit `0aced82`("Add CacheBlend PIM simulation support",
2026-08-17,分支 xy_0814)。新增 `src/workload.py`(510 行)、
`src/workload_runner.py`(2222 行,现约 2800 行)、测试 601 行;
把 Ramulator2 从 git submodule 改为内嵌源码树并带上 AttAcc 的
HBM3-PIM 补丁;`main.py` +181 行接 CLI。

## 1. 要解决的问题(零基础)

多个请求/agent 的 prompt 里有**相同的文段**(系统提示、共享文档、
上级 agent 的输出)。**KV 复用**:这些段的 KV 缓存只算一次、存一份,
后来者直接读。两种学术方案被建模:
- **CacheBlend**:重用整段,但每层抽一部分"最该修"的行重算
  (修正行 (corrected rows)),弥补跨前缀的注意力误差;
- **EPIC**:只重算每个移位段开头固定几行(前缀行)。
复用带来两个硬件问题:(a) 共享 KV 物理放哪、私有修正放哪;
(b) 事件顺序——A 的 KV 没写完,B 就不能读。上游的矩形批模型表达不了,
所以这一块造了**物理地址级的事件 DAG** (directed acyclic graph)。

## 2. 工作负载与计划层:`src/workload.py`

- 数据类:`Segment`(`:41`,一段文本的指纹 fingerprint+长度+角色)、
  `Request`(`:51`,一个请求=段序列+输出长度 lout)、`Workload`
  (`:66`,请求集合,分**tier**——依赖层级,上级输出是下级输入)。
- 两种 JSON 格式解析(`load_workload`,`:300`):RAG legacy-list
  (RAG=retrieval-augmented generation,检索增强生成——prompt 里拼接
  检索到的共享文档,`_parse_rag`)与 supervisor v2-dag(上级 agent 的
  输出作为下级 agent 输入的多层工作负载,带 parent/tier/段位移 delta,
  `_parse_supervisor`)。所有非法输入给 `WorkloadValidationError`。
- `build_reuse_plan`(`:348`):按指纹配对"谁的段可以复用谁的",产出
  `ReusePlan`(`:103`):每个可复用段的 owner、CacheBlend 每层抽样的
  修正行(`cacheblend_partial_rows`,种子可复现)或 EPIC 前缀行。
  `validate_reuse_plan`(`:451`)查层覆盖与行合法性。

## 3. 物理布局:master/diff 双池 TLB(`src/workload_runner.py`)

- `CacheBlendTLB`(`:674`):逻辑位置→HBM 物理字节地址的映射表
  (借用 TLB (translation lookaside buffer,处理器的地址翻译表) 之名)。
  **master 池**(通道 0–7)放不可变共享行与新算行;**diff 池**
  (通道 8–15)放消费者私有的修正行。`reserve→finalize→locate` 三段式:
  先登记所有行,`finalize`(`:701`)一次性按 AttAcc 原版地址风格
  (1 GiB 通道、8 KiB 头分区、V=K+8 MiB)切块 `KVBlock`(`:543`)。
- **read-mask 语义**(`CacheBlendTLB` 类 docstring,`:677` 附近):被修正行的 master 原行
  **照常顺序流读**、只是从 score 里被掩掉 (masked)——不然 master 流被
  打成一地碎 run,每个都要冷启动。`_physical_reads`(`:637`)给出
  "实际读的行+掩码集合";`scan_runs`(`:789`)把物理相邻的行并成
  连续 run 喂 Ramulator(每个 run 一次实测)。
- `NoReuseKVLayout`(`:845`):no-reuse 物理基线的仿射私有布局
  (一请求一层一条连续 extent),同一 DAG/调度器,用于同秤对比。

## 4. 事件 DAG:prefill/decode 全流程

`_run_cacheblend_prefill`(`:2242`,块 04/05 又扩)按 tier×请求×层生成
`SplitEvent`(`:26`)事件:GPU 算子、LINK 传输(Q 下行/KV 下行/LSE 元组/
context 回传,逐字节记账)、TLB 查询、每池 PIM 扫描、DIE(此处指 HBM
堆叠底部的逻辑晶片 buffer/logic die,放 softmax/合并单元)上的
LSE 合并 (log-sum-exp merge,把 GPU 局部 softmax 与各 PIM run 的部分
softmax 数值等价地合成一个)、DRAM 写回。要点:
- **split prefill**:GPU 把本请求新算的行相互注意(一个矩形块),
  PIM 同时扫已驻留的共享行,DIE 合并——两支无数据依赖,可重叠;
- **decode**(`_append_cacheblend_decode`,`:1595`;批版 `:1755`):
  每 token 每层,PIM 扫"prefill 绑定行+已生成行",批版按
  "全局 Q 就绪队列"admission(Q 传到才准进批);
- 调度器 `_schedule_cacheblend`(`:1025`)在 GPU/LINK/各 PIM 池/DIE/TLB
  资源上排事件;校验器 `validate_cacheblend_events`(`:1176`)与
  `validate_cacheblend_attacc_overlap_contract`(`:1138`)强制拓扑序、
  链路字节数、"context 必须等齐所有本地贡献"等硬件契约。
- 跨 tier 依赖:子 agent 的第一层 qkv 必须等父 agent 最后一个
  `decode_dram_store_master`(`test_cacheblend_emits_trace_ordered_tlb_and_physical_addresses`
  末段断言)。

## 5. Ramulator 侧配套

- 内嵌 `ramulator2/` 源树(HBM3-PIM 命令集见块 01 §5),
  `pim_ramulator_src/` 是种子副本;**不要跑 `set_pim_ramulator.sh`**
  (会 `git reset --hard` 父仓库,见 `HANDOFF.md` §1)。
- `ramulator_wrapper` 加**签名缓存** (signature cache,`:83-140`):
  相同 (run 长度、地址映射签名、命令参数) 的扫描只实测一次——
  decode 逐步只在尾部变化,缓存命中率决定仿真速度。
- trace 生成器加 `--key-addr/--value-addr/--pool-base/--channels`:
  按 TLB 给的真实物理地址出 trace(此前只有固定地址)。

## 6. CLI 与三条延迟模型

`main.py`:`--reuse {no-reuse,cacheblend,epic}`、
`--cacheblend-latency-model {analytic,physical}`、
`--no-reuse-latency-model {legacy,physical}`。三条路:
legacy(上游矩形批,回归锚)、analytic(legacy 上打折算复用节省,
`workload_runner.py:200`)、physical(本块的事件 DAG+Ramulator)。

## 7. 在论文中的意义

物理事件 DAG 是论文方法学(§5 methodology)的**参考模型** (reference
model):A/C 两个系列的所有对比都要求"同一事件路径、同一物理布局"
(同秤原则),CacheBlend/EPIC 的语义(修正行/前缀行)也在这里定义,
供 A2(纯 GPU 跑复用)与 A4–A6 共用。本块自带的两个实验目录
(`experiments/_archive/cacheblend_tier_batch`、`_archive/end_to_end_20260814`)
在严格论文模式下已归档,**不入论文**(`docs/EXPERIMENTS.md`)。

## 8. 测试覆盖与悬置

- 覆盖:TLB 地址自洽(块基址+偏移逐条对)、diff 池只含被修正行、
  master 流 run 数与修正数无关(read-mask 语义)、事件序、重叠契约、
  批 decode 的 Q 到达 admission、池溢出到同池下一通道。
- 悬置(接手须知):`_run_legacy_reuse_prefill`(`:366`)是早期
  逻辑级路径,现 CLI 三种 policy 都走物理 DAG,该函数已不可达。
