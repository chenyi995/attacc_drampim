# 交接文档(chenyi-experiment-821)

接手人从这页开始:环境、构建、怎么跑、坑。索引见 `docs/README.md`,
实验总纲见 `docs/EXPERIMENTS.md`,逐日记录见 `docs/LOG.md`。

## 1. 环境与构建

- **机器**:EL8(Linux 4.18),64 核可用;仓库在
  `/data2/chenyi9/KV-PIM/attacc_drampim_xinyao`。
- **Ramulator2 必须用 gcc-toolset-11 编**(系统 gcc 8.5 编不过 `<ranges>`):
  ```sh
  cd ramulator2 && rm -rf build && mkdir build && cd build
  CC=/opt/rh/gcc-toolset-11/root/usr/bin/gcc \
  CXX=/opt/rh/gcc-toolset-11/root/usr/bin/g++ cmake .. && make -j8
  cp ramulator2 ../ramulator2
  ```
  `source /opt/rh/gcc-toolset-11/enable` 在非登录 shell 可能传不进 cmake,
  显式 CC/CXX 最稳。
- **本分支 `ramulator2/` 是父仓库普通目录(不是 submodule)**:
  **不要跑 `set_pim_ramulator.sh`**——其 `git reset --hard` 会作用到父仓库,
  其文件覆盖会回退分支补丁。`pim_ramulator_src/` 种子副本已与分支同步。
- **每个分支要用自己源码编出的二进制**:本分支的 C++ 含搬运总线转向约束;
  `xinyao_0821` 没有。切分支后重编(增量 make 很快)。
- Python:系统 python3,无 pytest,用 `python3 -m unittest tests.test_workload`。

## 2. 怎么跑(速查)

```sh
# 全部单测(须 32/32)
PYTHONPATH=$PWD python3 -m unittest tests.test_workload
# A 系列(示例:A6 + EPIC)
experiments/GPU_PIM_vs_GPU_prefill/run_one.sh <outdir> <wl> A6 nvlink3 epic "--epic-prefix-recompute-tokens 1"
# C 系列
python3 experiments/mq_command/run_c_points.py
python3 experiments/mq_command/run_mq_study.py --workers 48
python3 experiments/mq_command/run_pipeline_overlap.py
# 端到端(默认已是 mq 命令;bank-whole 可选)
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json --reuse epic \
  --epic-prefix-recompute-tokens 1 --cacheblend-batch-size 4 \
  --ramulator-workers 24 --pipeopt --pim-prefill-mode bank-whole \
  --workload-report /tmp/e2e.json
# RTL sweep(独立仓库,nohup 断点续跑,已有 reports 的 tag 自动跳过)
cd /data2/chenyi9/KV-PIM/fugue-logic-die-rtl/syn && nohup setsid ./run_mq_sweep_all.sh > sweep_nohup.log 2>&1 &
python3 collect_mq_results.py     # 汇总 12 点面积/时序/功耗
```

## 3. 坑(按疼痛排序)

1. **`pkill -f <串>` 若 <串> 出现在你自己的命令行里会把自己杀掉**(exit 144,
   本会话与 RTL 仓库都踩过)。杀 genus 用 `pgrep genus | xargs kill`(按进程名)。
2. 机器上可能有**别人的 genus**(`run_genus.tcl` 风格),清进程前先 `ps` 看清。
3. Genus 许可池按两路并发用;RTL 仓库详见其 `HANDOFF.md`(§7 环境)。
4. Ramulator preset 的 `tCK_ps=1300` 字段与包装器 0.769 ns/cycle 口径不一致
   (nRFC 换算受影响 ~2%),周期数不受影响;秒换算一律以包装器为准(AttAcc 沿袭)。
5. `mq` 是 main.py 的**默认**批命令;要复现旧行为加 `--pim-batch-command replicate`。
   代码内部 Layer 默认仍是 replicate(保回归)。
6. 长任务一律 `nohup setsid ... &`,并把脚本写成按产物跳过的断点续跑式。
7. 论文仓库有自己的 CLAUDE.md(数据纪律:数字须用户手动复现才能进正文,
   否则 `\TBDnum`)——改稿前必读。
8. RTL 数字口径:N28 是 logic 工艺;logic die 直接可用,**in-bank 进论文须按
   AttAcc 的"DRAM 工艺疏 10×"折算**,比例/增量是有效读数。

## 4. 未完成 / 已知开口(接手优先级)

1. **论文正文欠两句**(机制已实装,措辞待chenyi9 定):§4.3.2 写口的 D_i 过滤
   (到达顺序无关);§4.5.2 bank-whole 的因果丢弃。另 §4.5.3 "column accesses
   grow with n_r" 与 MQ 语义有张力(MQ 下列访问不随 n 长,PE op 才随 n 长)。
2. **B4(Eq. placement 动态选边)未实装**——论文 §6 的 Fugue 行依赖它;
   现有拐点表是 A4 vs A6 扫描,实装后要按正文口径重扫
   (`docs/README_design_check.md` §3.1)。
3. §4.2 放置表 / 行粒度跨通道布局、GQA(g 恒 1)、n_d≈ρfC 配比、
   行激活计数输出链路——差距全表见 `docs/SIM_VS_PAPER_AUDIT_0821.md`。
4. C-abl-3 的 sweep 若中断:重跑 `run_mq_sweep_all.sh` 即续;完成后把
   `collect_mq_results.py` 输出整理进 `fugue-logic-die-rtl/syn/MQ_MICROARCH.md`。
5. **agentic 多轮驻留 KV(`history_len` / `--history-len`)已在 ablation 路径
   与物理事件 DAG 路径实装**(LOG 2026-08-22);legacy 解析路径
   (`--no-reuse-latency-model legacy`、`--cacheblend-latency-model analytic`)
   不支持,history>0 会显式报错。尚无带 `history_len` 的实验系列/工作负载
   JSON(现有 JSON 都是 H=0,行为逐位不变)。
