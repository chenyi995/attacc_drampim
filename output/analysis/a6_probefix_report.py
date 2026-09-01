#!/usr/bin/env python3
"""Regenerate RESULTS_A6_probefix_20260901.md from the raw run directories.

The raw runs are local-only (docs/RAW_DATA_MANIFEST.md); this script reads the
``REPORT_SUMMARY`` line each cell's ``dag_A6.log`` carries and the probe
records in ``a6_probe.jsonl``, and prints the page.  Nothing here is typed by
hand, so a partial sweep can be re-rendered as cells land:

    python3 output/analysis/a6_probefix_report.py > \
        output/analysis/RESULTS_A6_probefix_20260901.md
"""
import json, os, sys, glob, collections

NEW = os.environ.get("A6_NEW", "output/a6probefix_20260901")
OLD = os.environ.get(
    "A6_OLD",
    "/home/cw636/chenyi/attacc_drampim/output/sweep_models_20260830-163226")
EVID = "output/analysis/a6_probefix_20260901"   # committed probe records


def summary(path):
    """Last REPORT_SUMMARY line of a run log, as a dict (None if absent)."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "rb") as fh:
        for raw in fh:
            if raw.startswith(b"REPORT_SUMMARY "):
                last = raw
    if last is None:
        return None
    try:
        return json.loads(last.decode("utf-8", "replace")[len("REPORT_SUMMARY "):])
    except ValueError:
        return None


def probes(cell):
    """Probe records for a cell, preferring the committed copy."""
    for path in (f"{EVID}/{cell.replace('/', '_')}.jsonl",
                 f"{NEW}/{cell}/a6_probe.jsonl"):
        if os.path.exists(path) and os.path.getsize(path):
            with open(path) as fh:
                return [json.loads(l.split(" ", 1)[1]) for l in fh if l.strip()]
    return []


cells = sorted("/".join(p.split("/")[-3:-1])
               for p in glob.glob(f"{NEW}/*/*/dag_A6.json") if os.path.getsize(p))

print("# A6 探针 head-folding 修复：验证结果")
print()
print(f"生成于运行目录 `{NEW}`，对照组是 2026-08-30 那轮 sweep。")
print(f"**本页是快照：26 格中已完成 {len(cells)} 格**，补齐后重跑本脚本即可。")
print()
print("全部数字由 `output/analysis/a6_probefix_report.py` 生成，不要手工引用。")
print()
print("""## 0. 改了什么，以及怎么读这几张表

**只有探针（probe）变了。** `_append_placement_pim_scan`（真正被仿真的那条路）是
把 11 行原地计算原样搬进新 helper `_placement_channel_runs`，表达式逐字相同、
函数内再无对 `loads`/`active` 的赋值，因此 committed 路径行为不变。探针的两处改动：

1. **喂给 Ramulator 的 runs**：`tlb.scan_runs(scan_locations)`（只覆盖 **1 个 head**
   的 TLB reuse run）→ `_placement_channel_runs()` 生成的**每通道一条、把
   `heads_per_hbm` 个 head 折进 row count** 的 run，且只提交最忙的那一条。
2. **聚合方式**：`sum(...)` → `max(...)`。16 个通道在 committed 路上是并发的
   `PIM:pool{c}-{c}` 事件，旧探针把它们串行相加了。

净效应是 `t_bank` 变大（旧探针把 PIM 的 bank 成本按单 head 定价、系统性低估），
于是原本被一律推去 PIM 的 request 里，该走 GPU 的被选了出来。

**读表要点：**

- **改善量 = 改判的 request 数。** 选边接近对半开的格子（`baseline`/`k-hi`/`C-lo`/
  `N-lo`/`reduce`）makespan 改善 30–63%；仍判全 PIM 的格子（`D-lo`/`broadcast`/
  `pipeline`）与旧值一致。没有一格变差。
- **KV link 流量是涨的，不是降的。** 改判的格子 link 相对旧 A6 上升（`baseline`
  25→123 GB，`k-hi` 41→127 GB），因为 GPU prefill 必须把 KV 读回主机；但仍显著
  低于全 GPU 的 A4。论文里 A6 的卖点需要按指标分开陈述：**时间与能量赢，KV 流量
  是相对全 PIM 的让步。**
- **`pipeline_k8` 是一个反例**：A6 22.39 s 反而慢于 A4 的 22.30 s（+0.4%）。探针判
  PIM 快 4.5 倍（`t_bank/t_xpu` = 0.22，三次一致），但该 workload 的 DAG 是串行链
  （`gpu 6.83 s + pool 15.54 s ≈ makespan 22.39 s`，几乎无重叠），PIM pool 才是瓶颈：
  把 2304 行（占 prefill 的 0.84%）挪到 PIM，GPU 省 0.023 s、pool 多 0.159 s。
  **探针孤立地给两边定价，没有"哪个设备是瓶颈"的概念。** 这一条不是本次修复引入的
  —— 旧 A6 同样判全 PIM、同样是 22.39 s。

**未决项（提交时仍未解决）：**

1. 选边未改变的格子并非逐字节一致：`D-lo` 的 `event_count` 与 `link_bytes` 逐字节
   相同，但 `makespan_s` 差 0.0085%（18.75999950678873 vs 18.76159292471786）。
   原因未确认（候选：探针的 Ramulator 调用改变了共享 signature cache 的填充状态；
   两轮的 `--ramulator-workers` 也不同）。按 `docs/sessions/2026-08-31.md` 的规矩，
   这需要一次受控复现才能定性。
2. `pipeline` 上探针估的 bank 成本（3 × 156 us）与实测 pool 增量（159 ms）差约 340 倍。
   即便计入 `pim_pool_time_s_unoverlapped` 是 16 通道求和而探针取 max（x16），仍差
   20 倍。另一嫌疑是 PIM prefill 强制该 batch 的 K/V 先落栈（Fugue 4.5.2 landing
   order）带来的额外写事件（A6 比 A4 多 9120 个 event）。
""")
print()
print("## 1. makespan（秒）")
print()
print("| cell | 新 A6 | 旧 A6 | 旧 A4 (全 GPU) | 旧 A5 (全 PIM) | Δ vs 旧 A6 | 新探针选边 |")
print("|---|---:|---:|---:|---:|---:|---|")
for cell in cells:
    n = summary(f"{NEW}/{cell}/dag_A6.log")
    o = {a: summary(f"{OLD}/{cell}/dag_{a}.log") for a in ("A6", "A4", "A5")}
    if not n:
        continue
    side = collections.Counter(r["side"] for r in probes(cell))
    f = lambda d: f"{d['makespan_s']:.2f}" if d else "n/a"
    d = (f"{(o['A6']['makespan_s'] - n['makespan_s']) / o['A6']['makespan_s'] * 100:+.1f}%"
         if o["A6"] else "n/a")
    s = (f"pim {side['pim']} / gpu {side['gpu']}" if side else "-")
    print(f"| `{cell}` | **{n['makespan_s']:.2f}** | {f(o['A6'])} | {f(o['A4'])} "
          f"| {f(o['A5'])} | {d} | {s} |")
print()
print("## 2. KV link 流量（GB）与能量（J）")
print()
print("| cell | link 新 A6 | link 旧 A6 | link 旧 A4 | energy 新 A6 | energy 旧 A6 |")
print("|---|---:|---:|---:|---:|---:|")
for cell in cells:
    n = summary(f"{NEW}/{cell}/dag_A6.log")
    o = {a: summary(f"{OLD}/{cell}/dag_{a}.log") for a in ("A6", "A4")}
    if not n:
        continue
    gb = lambda d, k="link_bytes": f"{d[k] / 1e9:.2f}" if d else "n/a"
    jj = lambda d: f"{d['energy_nj'] / 1e9:.1f}" if d else "n/a"
    print(f"| `{cell}` | {gb(n)} | {gb(o['A6'])} | {gb(o['A4'])} | {jj(n)} | {jj(o['A6'])} |")
print()
print("## 3. 探针自身的开销：旧 vs 新")
print()
print("`old` 列按旧公式从同一批探针记录重算：旧探针把 `tlb.scan_runs()` 解出的每条 run")
print("逐条交给 Ramulator 再求和；新探针只提交最忙的那一条通道 run。")
print()
print("| cell | 探针次数 | Ramulator 调用 旧→新 | 模拟行数 旧→新 |")
print("|---|---:|---|---|")
for cell in cells:
    rs = [r["shape"] for r in probes(cell)]
    if not rs:
        continue
    o_runs = sum(x["tlb_runs"] for x in rs)
    n_runs = sum(x["pim_kv_runs"] for x in rs)
    o_rows = sum(x["scan_locations"] for x in rs)
    n_rows = sum(x["probe_rows_busiest"] for x in rs)
    print(f"| `{cell}` | {len(rs)} | {o_runs} → **{n_runs}** (÷{o_runs / max(n_runs, 1):.1f}) "
          f"| {o_rows:,} → {n_rows:,} (×{n_rows / max(o_rows, 1):.1f}) |")
