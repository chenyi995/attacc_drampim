# 块 08:完整性核查执行记录(防"乱搞",准则见块 00 §2)

核查日期:**2026-08-22**,工作区状态:分支 chenyi-experiment-821,
commit `711ae25` + 未提交改动(清单见 §2)。本页记录**实际执行过**的
核查与结果;§6 是宸逸可自行重跑的命令集。

## 1. 代码引用抽查(I3)——发现并修正一次行号漂移

- **发现**:块 02/03/04 初稿的 `workload_runner.py`/`ablation.py` 行号
  是 history 改动(块 05)插入**之前**的旧值,整体漂移 5–48 行
  (例:`_run_cacheblend_prefill` 写作 `:2194`,实际 `:2242`;
  `run_ablation_report` 写作 `:959`,实际 `:943`——后者系转抄旧视图,
  属抄录错误)。
- **处置**:2026-08-22 以 `grep -n` 逐符号重核,全部修正为当前行号;
  引用规范改为符号名为准(块 00 §3)。
- **残余风险**:提交块 05 或任何后续改动都会再漂。**每次改动
  `src/*.py` 后须重跑 §6.1 的抽查命令**。

## 2. 改动面对账(I4)——通过

`git status --short`(2026-08-22)未提交改动:

| 类别 | 文件 | 与声明的对应 |
|---|---|---|
| 代码/测试 | `main.py`、`src/workload.py`、`src/ablation.py`、`src/workload_runner.py`、`tests/test_workload.py` | =块 05 声明的 history 改动面,无多余代码改动 |
| 文档 | `CLAUDE.md`(历史教训)、`docs/`(HANDOFF/LOG/README + 新增 OUTPUT_SPEC、README_mq_design_space、audit/) | 本审计工程自身 |
| 其他 | `ramulator2.local_backup_0821/`(未跟踪) | 08-21 切分支前备份,`docs/LOG.md` 有记录 |

commit 级归属(块索引表)用 `git log --stat c1540de..HEAD` 逐一核对文件
清单,与各块"归属"节一致;`0aced82` 的 `src/workload{,_runner}.py`
新增、`47ae0c3` 的 `src/ablation.py`/`src/gemm_table.py` 新增、
`264d14a` 仅 C++ 两文件 +7 行,均已验。

## 3. 关键数字双通道交叉核对(I1/I6)——通过

| 数字 | 通道 A | 通道 B | 一致? |
|---|---|---|---|
| C3(16,2)@1.3 GHz = 2.66× | `results_c_points.json` `c3_speedup_vs_c1` | `docs/LOG.md`/`docs/EXPERIMENTS.md` 记录 | ✓ |
| 列读间隔 8 点(32/17/11/10;63/33/21/14) | json `interval_score` | 现场重算 `ceil(n/(f·0.769))` + 地板(§4) | ✓ 逐点 |
| GEMV buffer 容量 512→8/1024→16/2048→32 | `mq_query_capacity` 现场运行 | 测试 `test_interval_carries_power_stretch_and_pe_throughput` 断言 | ✓ |
| 功耗地板 n=8/16/32→8/10/13 cycles | `mq_interval_cycles(n,True,1000)` 现场运行 | json 高频点贴地板行为((16,2)@3.2:interval 10=地板) | ✓ |
| 面积 12 点 | `collect_mq_results.py` 现场重跑(2026-08-22) | `build_*/reports_*` 原始 Genus 报告(collector 解析自其中) | ✓(同源单通道,标注) |
| AttAcc §7.7 口径(0.094/13.12/121/10.84%/25%) | `ref/attacc.pdf` pdftotext 原文抽取 | 复算:128×0.094+32×0.036=13.18≈13.12(舍入),13.12/121=10.84% | ✓ |

## 4. 数字复算记录(I2)——通过

2026-08-22 现场复算(命令见 §6.3):

- 锚定因子:83,563/9,400 = 8.890;
- die 占比:(12.032·k+1.152)/121,k=1 → 10.90%(论文 10.84%,块内差异
  来自 13.18 vs 13.12 的舍入,已注明);k=3.202 → 32.79%;
- 预算反解:k\* = (0.25×121−1.152)/12.032 = 2.418 ≈ 2.42;
- 匹配频率:f\*(n)=n/(地板·0.769 ns) → 1.30/2.08/3.20 GHz(n=8/16/32)。

## 5. 行为不回退证据(I5/I7)——通过

- 全套单测 **38/38 OK**(2026-08-22,`Ran 38 tests in 60.054s / OK`,
  含上游逐位回归三测与 A1 逐位锚);
- history 关闭态等价:分支条件改动 `full or not reusable`→`not reusable`
  在 H=0 时逐位等价(等价论证:force_fresh 下 reusable 仅含 history
  绑定,H=0 则为空,块 05 §3);32 个既有测试全部 H=0 通过;
- 反事实对照:`kv_gpu_to_pim` 字节 H=0/H>0 相同(147,712 B),
  见块 05 §4。

## 6. 核查命令集(宸逸抽查用,不依赖助手)

```sh
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
# 6.1 代码引用抽查(符号名为准;输出行号应与审计块一致)
grep -n "def _run_cacheblend_prefill\|class CacheBlendTLB\|def run_ablation_report\|PRESETS" \
  src/workload_runner.py src/ablation.py
# 6.2 回归与行为锁
PYTHONPATH=$PWD python3 -m unittest tests.test_workload     # 须 38/38 OK
# 6.3 关键数字复算
PYTHONPATH=$PWD python3 - <<'EOF'
from src.ramulator_wrapper import mq_interval_cycles as m, mq_query_capacity as c
assert [c(b) for b in (512,1024,2048)] == [8,16,32]
floors = {n: m(n, True, 1000.0) for n in (8,16,32)}
assert floors == {8:8, 16:10, 32:13}
print({n: round(n/(f*0.769), 2) for n, f in floors.items()})   # f*: 1.3/2.08/3.2
print(round(83563/9400, 2))                                     # 8.89
print(round((0.25*121-1.152)/12.032, 2))                        # 2.42
for k in (1.0, 1.849, 3.202): print(k, round((12.032*k+1.152)/121, 4))
EOF
# 6.4 仿真 8 点与面积 12 点重跑(慢)
python3 experiments/mq_command/run_c_points.py
(cd /data2/chenyi9/KV-PIM/fugue-logic-die-rtl/syn && python3 collect_mq_results.py)
# 6.5 改动面对账
git status --short && git log --stat --format="=== %h %s" -6
```

## 7. 未覆盖项(如实声明)

- 面积数字目前是**单来源**(Genus 报告→collector),无独立第二通道;
  可做的加固:对 `reports_*/ *_qor.rpt` 抽一两个点人工读数比对。
- 结果 json 无哈希清单/时间戳锁定,重跑覆盖后旧值不可追溯
  (git 提交历史部分缓解:已提交的 json 可 `git log -p` 追)。
- 本页自身由助手撰写(自我审计局限,块 00 §5);§6 命令集是唯一
  不依赖信任的核查途径。
