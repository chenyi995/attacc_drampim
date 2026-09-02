#!/usr/bin/env python3
"""Render the per-tier latency and per-run power tables from the extracted CSVs.

Six tables per model, nine columns each (one per rung):

  latency   prefill / decode / makespan   -- PER TIER
  power     prefill / decode / overall    -- PER RUN, not per tier

The asymmetry is not a choice.  The reports break energy down by device class,
by event name and by transformer layer, and by nothing else -- there is no
per-tier energy anywhere in them, and ``batches`` carries a tier field but no
energy.  Attributing energy to tiers would mean tagging it at accounting time
in the engine and re-running the whole sweep.  So latency is per tier because
the data supports it, and power is per run because that is as far as the data
goes.  Saying it plainly beats inventing a split.

Definitions, chosen so the parts add up rather than needing reconciliation:

  prefill_s   a tier's last request finishing prefill, minus the tier's start
  decode_s    the tier's end, minus that same instant
  makespan_s  the tier's end minus its start  ==  prefill_s + decode_s

  prefill power   prefill energy / summed prefill_s over the run's tiers
  decode power    decode energy  / summed decode_s
  overall power   total energy   / makespan

Prefill vs decode energy comes from ``by_event``: the report names every decode
operation as the ``decode_``-prefixed twin of its prefill counterpart
(decode_qkv/qkv, decode_ctx_pim_to_gpu/ctx_pim_to_gpu, ...), so the split is a
naming convention in the writer, not a guess here.

    python3 output/analysis/make_tier_tables.py > docs/README_sweep_tables.md
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = ["A1", "A2", "A3", "A3a", "A3b", "A4", "A4b", "A5", "A6"]
# The order configs vary one axis at a time, then topology -- same as the
# sweep's own design table, so a reader can scan down an axis.
ORDER = ["baseline", "N-lo", "N-hi", "C-lo", "C-hi", "D-lo", "D-hi",
         "k-lo", "k-hi", "broadcast", "reduce", "supervisor", "pipeline",
         "private"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(v, nd):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=HERE)
    a = ap.parse_args()
    rungs_csv = os.path.join(a.indir, "sweep_rungs.csv")
    tiers_csv = os.path.join(a.indir, "sweep_tiers.csv")
    for p in (rungs_csv, tiers_csv):
        if not os.path.exists(p):
            print(f"missing {p}; run output/analysis/extract_sweep.sh first",
                  file=sys.stderr)
            return 1
    R = list(csv.DictReader(open(rungs_csv)))
    T = list(csv.DictReader(open(tiers_csv)))

    # (model, config, tier) -> rung -> value, and (model, config) -> rung -> row
    tier_v = defaultdict(dict)
    for r in T:
        tier_v[(r["model"], r["config"], r["k"], r["tier"])][r["rung"]] = r
    run_v = defaultdict(dict)
    for r in R:
        run_v[(r["model"], r["config"], r["k"])][r["rung"]] = r

    models = sorted({r["model"] for r in R})
    cfg_key = {c: i for i, c in enumerate(ORDER)}

    def cfgs_of(model):
        s = {(r["config"], r["k"]) for r in R if r["model"] == model}
        return sorted(s, key=lambda ck: (cfg_key.get(ck[0], 99), ck[0]))

    o = print
    o("# 逐 tier 延迟表与逐 run 功率表")
    o("")
    o("每个模型六张表，每张表九列（九个档）。**全部数值直接来自")
    o("`output/analysis/sweep_rungs.csv` 与 `sweep_tiers.csv`**，本页不做二次计算；")
    o("重新生成：")
    o("")
    o("```bash")
    o("bash output/analysis/extract_sweep.sh")
    o("python3 output/analysis/make_tier_tables.py > docs/README_sweep_tables.md")
    o("```")
    o("")
    o("## 为什么 latency 是逐 tier、power 不是")
    o("")
    o("**报告里没有逐 tier 的能量。** `energy_breakdown_nj` 只有三种拆分：")
    o("`by_class`（DIE/GPU/LINK/PIM/TLB）、`by_event`（按事件名）、`by_layer`")
    o("（按 transformer 层 0–39）。`batches` 有 `tier` 字段但没有能量字段，而事件流")
    o("被 `--workload-report-events none` 关掉了，无法反推。")
    o("")
    o("把能量归到 tier 需要引擎在记账时打 tier 标签，那要重跑整个 sweep。所以：")
    o("**latency 逐 tier（数据支持），power 逐 run（数据只到这里）**。")
    o("")
    o("## 口径")
    o("")
    o("| 量 | 定义 |")
    o("|---|---|")
    o("| `prefill_s` | 该 tier **最后一个** request 的 `prefill_end_s` − tier 起点 |")
    o("| `decode_s` | tier 终点 − 同一时刻 |")
    o("| `makespan_s` | tier 终点 − 起点 = **prefill_s + decode_s**（严格相加）|")
    o("| prefill 功率 | prefill 能量 ÷ 全 run 各 tier 的 prefill_s 之和 |")
    o("| decode 功率 | decode 能量 ÷ 各 tier 的 decode_s 之和 |")
    o("| 整体功率 | 总能量 ÷ makespan |")
    o("")
    o("prefill / decode 能量取自 `by_event`：报告把每个 decode 操作命名为其 prefill")
    o("对应项的 `decode_` 前缀版（`decode_qkv`/`qkv`、`decode_ctx_pim_to_gpu`/")
    o("`ctx_pim_to_gpu` …），所以这个二分是写入端的命名约定，不是这里的猜测。")
    o("")
    o("`—` 表示该档缺失（见 `sweep_completeness.csv`）。")

    for m in models:
        cks = cfgs_of(m)
        o("")
        o("---")
        o("")
        o(f"# {m}")

        for title, col, nd in (("延迟：prefill（秒，逐 tier）", "prefill_s", 3),
                               ("延迟：decode（秒，逐 tier）", "decode_s", 3),
                               ("延迟：makespan（秒，逐 tier）", "duration_s", 3)):
            o("")
            o(f"## {m} — {title}")
            o("")
            o("| config | tier | " + " | ".join(RUNGS) + " |")
            o("|---|---:|" + "---:|" * len(RUNGS))
            for cfg, k in cks:
                tiers = sorted({r["tier"] for r in T if r["model"] == m
                                and r["config"] == cfg and r["k"] == k},
                               key=lambda x: int(x))
                if not tiers:
                    o(f"| {cfg} | — |" + " — |" * len(RUNGS))
                    continue
                for t in tiers:
                    row = tier_v[(m, cfg, k, t)]
                    cells = [fmt(f(row[r][col]) if r in row else None, nd)
                             for r in RUNGS]
                    o(f"| {cfg} | {t} | " + " | ".join(cells) + " |")

        for title, col, nd in (("功率：prefill（W，整 run）", "power_prefill_w", 0),
                               ("功率：decode（W，整 run）", "power_decode_w", 0),
                               ("功率：整体（W，整 run）", "power_overall_w", 0)):
            o("")
            o(f"## {m} — {title}")
            o("")
            o("| config | " + " | ".join(RUNGS) + " |")
            o("|---|" + "---:|" * len(RUNGS))
            for cfg, k in cks:
                row = run_v[(m, cfg, k)]
                cells = [fmt(f(row[r][col]) if r in row else None, nd)
                         for r in RUNGS]
                o(f"| {cfg} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
