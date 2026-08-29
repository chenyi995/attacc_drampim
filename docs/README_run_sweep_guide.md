# 运行指南：参数化 sweep（给另一台机器上的 AI 照着跑）

本指南面向**在一台全新机器上 clone 了本仓库的 AI**：按下面步骤即可从零把
98-run 的参数化 sweep 跑完、提取数据、独立复核、把结果写进指定 README 并
commit/push。设计规范见 `docs/README_sweep_design.md`。**原始 run 数据不入 git、
只在本机**（见 `docs/RAW_DATA_MANIFEST.md`），git 里只留 `output/analysis/` 的
分析脚本与 RESULTS 表。

---

## 0. 一句话
一个 generator（`workload/gen_sweep.py`）+ (topology, N, C, D, k) → 14 个 config
× 7 档 A1–A6 = **98 run**；跑批 → 提取 → 复核 → 写 `output/analysis/RESULTS_sweep.md`
→ commit。

## 1. Setup（一次性）

```bash
# 1) clone + 切到实验分支
git clone <REPO_URL> attacc_drampim
cd attacc_drampim
git checkout chenyi-822-cppcore-exp

# 2) Python 依赖（stdlib 之外只有 numpy、pandas）
pip install numpy pandas

# 3) 构建 ramulator2（C++，本仓库自带源码；产出 ramulator2/libramulator.so）
cmake -S ramulator2 -B ramulator2/build
cmake --build ramulator2/build -j"$(nproc)"

# 4) 构建 cppcore 事件核（产出 src/cppcore/libeventcore.so）
#    需要 g++ 支持 c++17（Makefile 默认 gcc-toolset-11，可改 CXX=g++）
make -C src/cppcore            # 或： CXX=g++ make -C src/cppcore
#    构建不了就纯 Python 回退：export KVPIM_CPPCORE=0（慢，但结果一致）

# 5) 冒烟自检
python3 -m unittest discover -s tests            # 应 41/41 通过
python3 main.py --system dgx-attacc --model LLAMA3-8B \
  --workload workload/sweep/wl_pipeline_D4.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 \
  --ablation A2 --engine dag --num-hbm 16 --ramulator-workers 8 \
  --cacheblend-batch-size 8 --workload-report-events none \
  --workload-report /tmp/smoke_a2.json           # 几分钟内产出 json 即 OK
```

**首跑会慢**：Ramulator 签名缓存 `ramulator2/signature_cache_v2_headhbm.jsonl`
不入 git，第一次跑各新 shape 时逐个建（用 ≤64 核跑第一遍建缓存，之后复跑秒级
起步）。缓存是确定性的，建一次后整轮 sweep 都快。

## 2. 跑 sweep（98 run）

```bash
bash experiments/run_sweep.sh
```
- 14 个 config **顺序**跑，每个 config 内 **7 档 A1–A6 并行**（安全、内存可控）；
- 产出目录 `output/sweep_<时间戳>/<config>_k<k>/dag_A{1,2,3,3a,4,5,6}.{json,log}`；
- config 清单与含义见 `docs/README_sweep_design.md` §6（N-hi=128 agent 最重、
  pipeline/N-lo 最轻）；
- **内存**：一个 config 的 7 档并峰约 150–300 GB。若机器内存充裕、想更快，可把
  `run_sweep.sh` 的 config 循环改成后台并发 + 加内存门控（`until free -g > 阈值`），
  单机内存用量别超机器上限；否则保持顺序即可。
- **进度**：`ls output/sweep_<ts>/*/dag_*.json | wc -l`（应最终到 98）；
  某 config 目录满 7 个 json 即该 config 完成。

## 3. 提取数据

```bash
python3 output/analysis/make_sweep_tables.py       # 自动取最新 output/sweep_<ts>/
# 或指定： python3 output/analysis/make_sweep_tables.py output/sweep_<ts>
```
产出 **`output/analysis/RESULTS_sweep.md`**：14 config × 7 档的 makespan / 总能量 /
平均功率 / prefill-PIM%，加 OFAT 判别视图（N/C/D/k 各凸显哪对相邻 A 档，见设计
文档 §7）与 topology 对照。全部实测、来源字段已标注。

## 4. 独立复核（必须）

RESULTS 写好后，**另起一个独立 agent**，让它**自己从原始 `dag_A*.json` 重抽数**
（不看 `make_sweep_tables.py`）逐项比对，确认全是实测、无编造。给它这个任务：

> 你是数据完整性审计员。目标：核对 `output/analysis/RESULTS_sweep.md` 里每个
> 数是否都能对到 `output/sweep_<ts>/<config>_k<k>/dag_A*.json` 的某个**实测字段**、
> 且算术正确。**自己写 Python 直接解析原始 JSON，不要用 make_sweep_tables.py 或
> 任何缓存。** 抽查 ≥4 个 config（含最重的 N-hi 和一个 topology 变体）× A1/A3/A6：
> - Makespan (s) = `makespan_s`；
> - 总能量 (kJ) = `energy_nj`/1e12；
> - 平均功率 (W) = `energy_nj`/1e9/`makespan_s`；
> - prefill-PIM% = `prefill_attention_sides` 里 =='pim' 的占比（空则 0%）。
> 舍入容差内即算 MATCH。报告：逐项 (config, rung, metric, 表值, 你重算值,
> MATCH/MISMATCH) + 每个 metric 对应的原始字段 + 是否有非实测/估算值 + 末行
> `VERDICT: all sampled values are measured and correct` 或 `VERDICT: issues: …`。
> 不要改任何文件。

复核不通过就先修 `make_sweep_tables.py` 再重提取，别 commit 带问题的表。

## 5. 结果放哪 + commit/push

- **结果 README**：`output/analysis/RESULTS_sweep.md`（提取脚本已写到这里）。
- **文件夹**：`output/analysis/`（git 跟踪；原始 `dag_A*.json` 与 `.log` 不入 git、
  只在本机，见 `docs/RAW_DATA_MANIFEST.md`）。
- **commit（只提分析产物，不碰原始数据/ramulator2）**：

```bash
git add output/analysis/RESULTS_sweep.md output/analysis/make_sweep_tables.py
git commit -m "Sweep results: RESULTS_sweep.md (98 runs) + independent-check verdict

<把独立复核 agent 的 VERDICT 与关键 MATCH 摘要粘这里>

Co-Authored-By: <你的署名>"
git push origin chenyi-822-cppcore-exp
```
> ⚠️ **不要 `git add -A`**：仓库里有未跟踪的 `ramulator2/`（数十 GB 构建/缓存）与
> 原始 run 数据，`-A` 会把它们塞进暂存并超时/超 GitHub 上限。**只定向 add
> `output/analysis/` 下的产物。** push 前自查：
> `git diff --cached --name-only | grep -E "dag_A|ramulator2" && echo "有脏文件，别提"`。

---

## 附：常见问题
- **cppcore 编不过** → `export KVPIM_CPPCORE=0` 用纯 Python（结果一致、慢些）。
- **首跑特别慢** → 在建签名缓存，正常；用 ≤64 核跑第一个 config 把缓存热起来。
- **某档 A2 的 prefill/decode 显示不了** → A2（GPU-only）此路径把两相位合成一个
  时间戳，正常；它只有 makespan/能量可比。
- **想改 sweep 取值点** → 改 `experiments/run_sweep.sh` 的 CFG 列表 +
  `workload/gen_sweep.py` 重生成 `workload/sweep/`，并同步 `make_sweep_tables.py`
  的 CONFIGS 列表。
