# 运行指南：experiment1 多模型 sweep（给另一台机器上的 AI 照着跑）

本指南面向**在一台全新机器上 clone 了本仓库的 AI**（目标机量级 ~300 CPU core /
~3 TB RAM）。按下面步骤即可从零把 **6 模型 × 14 workload-config × 9 档 = 756 run**
的 sweep 跑完、提取、独立复核、写进 `output/analysis/RESULTS_sweep.md` 并
commit/push。**原始 run 数据不入 git、只在本机**（见 `docs/RAW_DATA_MANIFEST.md`），
git 里只留 `output/analysis/` 的分析脚本与 RESULTS 表。

设计规范见 `docs/README_sweep_design.md`；9 档阶梯语义见 `docs/README.md` §3。

---

## 0. 一句话

预热 Ramulator 缓存 → 跑 756 run → 提取 → 独立复核 → 写
`output/analysis/RESULTS_sweep.md` → commit。

## 1. Setup（一次性）

```bash
git clone <REPO_URL> attacc_drampim && cd attacc_drampim
git checkout chenyi-822-cppcore-exp
pip install numpy pandas                                    # stdlib 之外只有这两个
cmake -S ramulator2 -B ramulator2/build && cmake --build ramulator2/build -j"$(nproc)"
make -C src/cppcore            # 产出 src/cppcore/libeventcore.so；编不过就 export KVPIM_CPPCORE=0
python3 -m unittest discover -s tests                      # 应全绿（61 测试）
```

**换机器构建坑（athena 2026-08-30 实测）**：
- **ramulator2 cmake 需外网**（`FetchContent` 拉 yaml-cpp/spdlog/argparse）。构建节点
  无外网时，先把一个已成功 checkout 的 `ramulator2/ext/` 拷进来再 cmake。
- **`--ramulator-workers` 不能给 1**：`_warm_build_price_finalize` 在 workers≤1 时退回
  冷路径、build-once 暖流程失效；每个 run 至少给 2（脚本默认 8）。
- Ramulator 每个并发 job 要**独立工作目录**（`ATTACC_RAMULATOR_DIR`/`ATTACC_RAMULATOR_LOG`，
  见 `src/system.py`：CSV 整文件重写）；NFS 上别并发 O_APPEND 写签名缓存。

## 2. 模型与 system 配置（3 档）

| 档 | 模型 | `--ngpu` / `--num-hbm` | heads_per_hbm | 出处（每个几何数字都有来源）|
|---|---|---|---|---|
| 小 | LLAMA-7B, LLAMA3-8B | 1 / 1 | 32 / 8 | LLaMA-1 arXiv:2302.13971 T2；Llama-3 arXiv:2407.21783 |
| 中 | GPT-13B, LLAMA-33B | 2 / 10 | 4 / ~5 | GPT-3 arXiv:2005.14165 T2.1；LLaMA-1 T2 |
| 大 | GPT-175B, LLAMA-65B | 8 / 40 | 3 / 2 | GPT-3 T2.1；LLaMA-1 T2 |

模型几何（层/hidden/heads/KV-heads）在 `src/config.py` 的 `model_table` 处逐条标了
论文出处；system 配置在 `experiments/run_sweep_models.sh` 的 `MODELS` 表里。9 档 =
A1 A2 A3 A3a **A3b** A4 **A4b** A5 A6（`docs/README.md` §3）。

## 3. ⚠️ 先预热 Ramulator 缓存（关键，别跳）

**为什么**：全新 clone 的签名缓存是**冷的**。head-aware 放置扫描给每条 channel 的真实
run 计价，**每个新 (模型, 上下文大小, 档) 形状都要跑一遍完整 Ramulator DRAM 仿真**；
1-HBM 小档的单 channel trace 最大、冷启动最慢。若冷缓存下直接开 756 run,大量 run 会
**同时抢建同一个形状**、争写同一签名文件——又慢又浪费。

**怎么做**：分两阶段,让缓存**只建一次、并行、无同形状竞争**。

```bash
# 阶段 1：预热（6 模型并行 = 不同形状不撞；每模型内 9 档串行 = 同形状单写）
bash experiments/warm_cache.sh
# 盯着缓存长大；不再增长即基本建好：
watch -n 20 'ls -la ramulator2/signature_cache.jsonl'   # 代码实际写这个文件
```

`warm_cache.sh` 用几个形状最多样的 config(最大 C64、baseline、最多 agent N64)× 9 档
把主要形状建出来;其余 config 到阶段 2 就是缓存命中。

## 4. 跑 sweep（756 run，阶段 2）

```bash
bash experiments/run_sweep_models.sh
```
- 6 模型顺序、每模型 14 config 顺序、每 config 内 **9 档并行**;缓存已暖 → 命中、快;
- 产出 `output/sweep_models_<时间戳>/<model>/<config>_k<k>/dag_A{1,2,3,3a,3b,4,4b,5,6}.{json,log}`;
- **内存**:大模型(8-GPU)最重;若 3T 充裕想更快,可把 `run_sweep_models.sh` 的模型循环
  改后台并发 + 加内存门控(`until free -g > 阈值`),单机别超上限;
- **进度**:`find output/sweep_models_<ts> -name 'dag_*.json' | wc -l`(应到 6×14×9=756)。

## 5. 提取数据

```bash
python3 output/analysis/make_sweep_tables.py            # 自动取最新 output/sweep_models_<ts>/
```
产出 **`output/analysis/RESULTS_sweep.md`**:**每个模型一节**,含 14 config × 9 档的
makespan / 总能量 / 平均功率 / prefill-PIM% + topology 对照。全部实测,来源字段已标。
脚本会自动识别多模型目录(`<model>/<config>_k<k>`)与单模型目录(向后兼容)。

## 6. 独立复核（必须）

RESULTS 写好后,**另起一个独立 agent**,让它**自己从原始 `dag_A*.json` 重抽数**
(不看 `make_sweep_tables.py`)逐项比对。任务:

> 你是数据完整性审计员。核对 `output/analysis/RESULTS_sweep.md` 里每个数是否都能对到
> `output/sweep_models_<ts>/<model>/<config>_k<k>/dag_A*.json` 的某**实测字段**、且算术
> 正确。**自己写 Python 直接解析原始 JSON,不用任何缓存/脚本。** 抽查 ≥2 个模型(含一个
> 大模型)× ≥3 config × A1/A3/A3b/A4/A4b/A6:makespan=`makespan_s`;能量 kJ=`energy_nj`
> /1e12;平均功率 W=`energy_nj`/1e9/`makespan_s`;PIM%=`prefill_attention_sides` 里
> =='pim' 占比。同时核对每个模型的层/hidden/heads 与 `src/config.py` 标注的论文出处
> (LLaMA arXiv:2302.13971、Llama-3 arXiv:2407.21783、GPT-3 arXiv:2005.14165)一致。
> 报告逐项 (model, config, rung, metric, 表值, 重算值, MATCH/MISMATCH) + 末行
> `VERDICT: all sampled values are measured and correct` 或 `VERDICT: issues: …`。不改文件。

复核不通过就先修 `make_sweep_tables.py` 再重提取,别 commit 带问题的表。

## 7. 结果放哪 + commit/push

- **结果 README**:`output/analysis/RESULTS_sweep.md`(提取脚本已写这里)。
- **文件夹**:`output/analysis/`(git 跟踪;原始 `dag_A*.json`/`.log` 与
  `ramulator2/*.trace`/`.yaml`/签名缓存**不入 git、只在本机**,见 `RAW_DATA_MANIFEST.md`)。

```bash
git add output/analysis/RESULTS_sweep.md output/analysis/make_sweep_tables.py
git commit -m "experiment1 multi-model sweep: RESULTS_sweep.md (756 runs) + independent-check verdict

<把独立复核 agent 的 VERDICT 与关键 MATCH 摘要粘这里>

Co-Authored-By: <你的署名>"
git push origin chenyi-822-cppcore-exp
```
> ⚠️ **绝不 `git add -A`**:仓库有未跟踪的 `ramulator2/`(数十 GB 构建/缓存/trace)与
> 原始 run 数据,`-A` 会把它们塞进暂存并超时/超 GitHub 上限。**只定向 add
> `output/analysis/` 下的产物。** push 前自查:
> `git diff --cached --name-only | grep -E "dag_A|ramulator2|\.trace|\.log" && echo "有脏文件,别提"`。

---

## 附：常见问题
- **首跑特别慢** → 在建签名缓存,正常;**务必先跑第 3 节的 `warm_cache.sh`**。若某模型
  仍太慢(尤其 1-HBM 小档单 channel trace 大),可临时把该档 `--num-hbm` 调到 2(小档
  heads_per_hbm 减半、trace 减半、仍 head/channel≥1),在 `run_sweep_models.sh` 改。
- **cppcore 编不过** → `export KVPIM_CPPCORE=0` 用纯 Python(结果一致、慢些)。
- **某档 A2 的 prefill/decode 显示不了** → A2(GPU-only)把两相位合成一个时间戳,正常;
  只有 makespan/能量可比。
- **想改模型/HBM 取值** → 改 `experiments/run_sweep_models.sh` 与 `warm_cache.sh` 的
  `MODELS` 表(加模型先在 `src/config.py` 的 `model_table` 加条目并标论文出处)。
