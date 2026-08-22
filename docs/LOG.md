# 逐日日志(按 git 历史与会话记录整理;2026-08-21 前的条目为 git 考据)

## 2026-07 上旬 — 仓库起点(xinyao)
- 基于 AttAcc 官方模拟器(scale-snu/attacc_simulator)建仓:`main.py`/`src/`/
  补丁版 Ramulator2;修 trace 生成缩进(c600051)。分支 `xinyao_0707`。

## 2026-07-15/16 — 布局与调度探索(xinyao)
- `xinyao_0715`:block 粒度 diffK 扫描与结果(7196987)。
- `v10-layout-schedule`:v10 block-scatter 共享行 score/context MAC 流(9ef54fe)。

## 2026-07-17/18 — RoPE 仿真(xinyao)
- 0718 KV-PIM 结果(V master+diff,6702cb6);RoPIM 风格 RoPE 仿真模型与重跑
  输出(05681b3、3567810)。分支 `xinyao_0718`。

## 2026-08-14/17 — 复用执行栈(xinyao)
- `xy_0814`;CacheBlend PIM 仿真支持(0aced82,08-17):物理 TLB/事件 DAG、
  master/diff 分池、EPIC/CacheBlend 计划、批 decode、Q 旋转模式等。
- 历史实验:`_archive/cacheblend_tier_batch`(共享 KV 凑批、旋转分布)、
  `_archive/end_to_end_20260814`(no-reuse 基线)。

## 2026-08-18 前后 — RTL 起步(fugue-logic-die-rtl)
- logic die N28/Genus 流程闭合:AttAcc 基线 473,939 µm² @500 MHz;
  Fugue(+TLB+RoPE+diff_decoder)+8.9%;Fugue2(GPU 侧 RoPE)+3.0%;
  分块面积表 `syn/AREA_BREAKDOWN.md`。

## 2026-08-21(上游当日)— A 系列成形(xinyao)
- 放置消融 A1–A6 + refined/flash GPU 模型 + GPU_PIM_vs_GPU_prefill 研究
  (47ae0c3);拐点表:EPIC p*(11745e1/9467de5)与 p*-L 无关(34d3cd7)。
  分支 `xinyao_0821`。

## 2026-08-21(本会话,宸逸 + 助手)— 审计、MQ 微架构、C 系列、严格论文模式
1. **环境**:切 `xinyao_0821`(旧工作区 stash + `ramulator2.local_backup_0821/`
   备份);gcc-toolset-11 重编 Ramulator2;论文仓库 pull 至 b3a38cb。
2. **审计**:通读论文正文与仿真器,产出 `docs/SIM_VS_PAPER_AUDIT_0821.md` 与
   自足版 `docs/README_design_check.md`(11 条差距:B4 选边缺、放置表缺、
   GPU-only 语义、n_d 配比、GQA、行激活指标链路等)。
3. **时序算术**:nCCDAB=6/4、免费 MAC 槽 4/5(微基准实测 70 cyc/行平台期)、
   拐点 B·d≥2/3(修正早先 ≥4 的口径);全网查证:bank-PIM 无人做多 Q/列读
   (LongSight MICRO'25 白纸黑字 "batching…no reuse due to lack of shared KV")。
4. **MQ-MAC 实装**(计划 `docs/PLAN_mq_command.md`):trace `--mq`(MAC×1、
   per-Q 搬运×n)+ wrapper 时序模型(IDD7 拉伸/PE 吞吐/通路下限)经 YAML
   `nCCDAB` 覆盖 + DAG/CLI 集成 + 按 GEMV 容量拆 sweep;31/31 → 提交 9d6fc7b。
5. **C 编号定名 + 实测**:C1 compact / C2 多通道 / C3 非对称 MQ;
   `--phase score|context` 相位切片实装,(16,2)/(32,4)×4 频点实测
   (C3@1.3 GHz:2.66×/3.63× vs C1,列读÷3.6/÷7.1);`xinyao_0821` 上提交
   干净增量层 0fc07bb(默认行为逐位不变)。
6. **数据流与三路审计**:`experiments/mq_command/DATAFLOW.md`(分层硬件增量
   + 17 步全数据流);三个并行 agent 独立复核带宽/数值/时序-buffer:
   数值逐位等价 ✓;发现 MVSB 串行地板 256·n_q(PE 提频回报递减第二原因)、
   实测为上界(趟间可重叠 ~10%)、寄存器清单修正(~0.9–1.4 KB/bank)。
7. **三裁决项处置**:③总线转向——C++ 加 MVSB↔MVGB/WRGB 转向约束
   (nRTW/nWTRL),两头流水合成实验:JEDEC 默认 ≤0.84% → 关闭窄下行方案
   (264d14a);①D_i 位图 master 写过滤 与 ②bank-whole 因果丢弃 prefill
   实装(`--pim-prefill-mode`),32/32,提交 3e338e6。
8. **RTL 微架构(C-abl-3)**:`fugue-logic-die-rtl` 新增 `mq_bank_pe.sv`
   (列锁存/Q 槽轮转/per-Q 状态/行界暂存/buffer 深度参数)+
   `mq_diff_decoder.sv`(AGENTS 路位图 + 写过滤 + 因果比较器)+
   `mq_score_store.sv` + `fugue_mq_logic_die.sv`;12 点综合 sweep 以
   nohup 断点续跑中(`run_mq_sweep_all.sh`,汇总器 `collect_mq_results.py`)。
9. **文档**:零基础全数据流 `docs/README_fugue_dataflow.md`(每步 M/K/N、
   切分/累加/合并/轮转);仓库根 `CLAUDE.md` 工作守则。
10. **严格论文模式整编**(本条目):实验编号定为**有且仅有 A 系列与 C 系列**;
    B1/B2 临时命名废止;历史实验目录移入 `experiments/_archive/`;
    `run_asym_points.py` 由 C 版 `run_c_points.py` 取代;根部设计文档移入
    `docs/`;新增本目录四件套(README/EXPERIMENTS/HANDOFF/LOG)。

### 悬而未决(详见 HANDOFF §4)
论文正文欠两句(写口过滤、因果丢弃)与 §4.5.3 措辞张力;B4 动态选边未实装;
放置表/GQA/n_d 配比/行激活输出链路;C-abl-3 sweep 完成后出 `MQ_MICROARCH.md`。
