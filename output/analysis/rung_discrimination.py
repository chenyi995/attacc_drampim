#!/usr/bin/env python3
"""Does each of the nine A-rungs earn its place, and does A6 choose well?

Three questions, all answered from measured sweep output:

  1. For every pair of rungs, is there ANYWHERE in the sweep where they differ?
     A pair that never separates is an ablation step that demonstrates nothing.
     The test is the MAXIMUM relative difference over all tasks, not the median:
     one config where a mechanism fires is enough to justify the rung.

  2. How much does A6 -- the full design -- beat the best of the A3 family?

  3. A6 picks GPU or PIM prefill per request.  When it picks PIM everywhere it
     is claiming "A5 <= A4"; the sweep contains A4 and A5 run outright, so that
     claim can be checked directly rather than trusted.

Reads the REPORT_SUMMARY lines in each task's ladder.log.  main.py emits that
line and writes dag_<rung>.json from the same values in the same call, so the
two agree by construction -- but --verify re-reads the JSONs and asserts it
rather than assuming, because ladder.log is ten thousand times smaller and the
whole point of using it is speed.

    python3 output/analysis/rung_discrimination.py [--verify N] [--root DIR]
"""
import argparse
import glob
import itertools
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f"{REPO}/output/_orch2")

RUNGS = ["A1", "A2", "A3", "A3a", "A3b", "A4", "A4b", "A5", "A6"]
A3_FAMILY = ["A3", "A3a", "A3b"]
METRICS = [("makespan_s", "makespan"), ("energy_nj", "energy"),
           ("link_bytes", "KV link"), ("event_count", "events")]
# Below this, a pair is reported as never separating anywhere in the sweep.
SEPARATION = 0.30
# num_hbm per model, copied from the NHBM table in output/_orch2/common.sh,
# which is what the runs actually passed as --num-hbm.  Kept here so this script
# stands alone; if common.sh changes, change this with it.
# CORRECTED 2026-09-01: the large models were wrong here.  GPT-175B and
# LLAMA-65B use 40 HBM stacks, not 10, which changes heads_per_hbm from 10 and 7
# to 3 and 2 -- i.e. they are the LEAST channel-crowded models, not the middle
# of the pack.  Any conclusion drawn from the old numbers is void.
NUM_HBM = {"LLAMA-7B": 1, "LLAMA3-8B": 1, "GPT-13B": 10,
           "LLAMA-33B": 10, "LLAMA-65B": 40, "GPT-175B": 40}


def load(root):
    """(model, config) -> {rung: summary dict}, only tasks with all nine."""
    out = {}
    for lg in sorted(glob.glob(f"{root}/*/*_k*/ladder.log")):
        parts = lg.split("/")
        js = {}
        for ln in open(lg, errors="ignore"):
            if "REPORT_SUMMARY" not in ln:
                continue
            try:
                d = json.loads(ln.split("REPORT_SUMMARY", 1)[1].strip())
            except (ValueError, IndexError):
                continue
            if d.get("ablation"):
                js[d["ablation"]] = d
        if all(a in js for a in RUNGS):
            out[(parts[-3], parts[-2])] = js
    return out


def verify(root, tasks, n):
    """Assert ladder.log agrees with the authoritative dag_<rung>.json."""
    keys = sorted(tasks)[:n]
    checked = 0
    for model, cfg in keys:
        for rung in RUNGS:
            p = f"{root}/{model}/{cfg}/dag_{rung}.json"
            if not os.path.exists(p):
                continue
            full = json.load(open(p))
            for k, _ in METRICS:
                a, b = tasks[(model, cfg)][rung].get(k), full.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    assert abs(a - b) < 1e-6 * max(1.0, abs(b)), \
                        f"{model}/{cfg} {rung} {k}: ladder.log {a} != json {b}"
                    checked += 1
    print(f"verify: {checked} values across {len(keys)} tasks match the JSONs\n")


def reldiff(x, y):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if x == 0 and y == 0:
        return 0.0
    return abs(x - y) / max(abs(x), abs(y))


def q1_pairs(tasks):
    print("=" * 78)
    print("1. 每对档的【全 sweep 最大差异】 -- 只要一处分得开，这一档就有存在意义")
    print("=" * 78)
    res = {}
    for a, b in itertools.combinations(RUNGS, 2):
        mx, where = -1.0, None
        per = {k: (-1.0, None) for k, _ in METRICS}
        for key, js in tasks.items():
            ds = []
            for k, _ in METRICS:
                d = reldiff(js[a].get(k), js[b].get(k))
                if d is None:
                    continue
                ds.append(d)
                if d > per[k][0]:
                    per[k] = (d, key)
            if ds and max(ds) > mx:
                mx, where = max(ds), key
        res[(a, b)] = (mx, where, per)
    weak = []
    for (a, b), (mx, where, per) in sorted(res.items(), key=lambda kv: kv[1][0]):
        tag = "分不开" if mx < SEPARATION else ""
        if mx < SEPARATION:
            weak.append((a, b, per))
        w = f"{where[0]}/{where[1]}" if where else "-"
        print(f"  {a+' vs '+b:13s} {100*mx:7.1f}%  {w:26s}{tag}")
    if weak:
        print("\n  分不开的档对，逐指标全 sweep 最大差异：")
        for a, b, per in weak:
            print(f"    {a} vs {b}")
            for k, lab in METRICS:
                d, key = per[k]
                loc = f"{key[0]}/{key[1]}" if key else "-"
                print(f"       {lab:9s} {100*d:9.4f}%   {loc}")
    return weak


def q2_a6_vs_a3(tasks):
    print()
    print("=" * 78)
    print("2. A6 相对【A3 系列中最好的一个】的改善（正 = A6 更好）")
    print("=" * 78)
    rows = []
    for key, js in tasks.items():
        r = {"key": key}
        for k, lab in METRICS[:3]:
            vals = [js[a].get(k) for a in A3_FAMILY]
            vals = [v for v in vals if isinstance(v, (int, float)) and v > 0]
            a6 = js["A6"].get(k)
            if not vals or not isinstance(a6, (int, float)):
                r = None
                break
            best = min(vals)
            r[lab] = (best - a6) / best
        if r:
            rows.append(r)
    for _, lab in METRICS[:3]:
        g = [r[lab] for r in rows]
        neg = [r for r in rows if r[lab] < 0]
        print(f"\n  {lab}: 中位 {100*st.median(g):+.1f}%   "
              f"范围 {100*min(g):+.1f}% ~ {100*max(g):+.1f}%")
        print(f"     A6 反而更差的任务: {len(neg)}/{len(g)}"
              + ("" if not neg else "  -> " + ", ".join(
                  f"{r['key'][0]}/{r['key'][1]} {100*r[lab]:+.1f}%"
                  for r in sorted(neg, key=lambda r: r[lab])[:4])))
    return rows


def q3_chooser(tasks):
    print()
    print("=" * 78)
    print("3. A6 的选边决定，拿 A4 / A5 的实测结果直接检验")
    print("=" * 78)
    same, fired, wrong = [], [], []
    for key, js in tasks.items():
        a4, a5, a6 = js["A4"], js["A5"], js["A6"]
        allpim = all(abs(a5[k] - a6[k]) < 1e-9 for k, _ in METRICS[:3])
        (same if allpim else fired).append(key)
        if allpim and a5["makespan_s"] > a4["makespan_s"]:
            wrong.append((key, a4["makespan_s"], a5["makespan_s"]))
    n = len(same) + len(fired)
    print(f"  A6 与 A5 完全相同（选边器判定全部走 PIM）: {len(same)}/{n}")
    print(f"  A6 实际改选了 GPU 的任务:                 {len(fired)}/{n}")
    if fired:
        print("     " + ", ".join(f"{m}/{c}" for m, c in sorted(fired)[:8]))
    print(f"\n  在判定“全走 PIM”的 {len(same)} 个任务里，"
          f"实测 A5 反而慢于 A4 的: {len(wrong)}")
    for key, a4, a5 in sorted(wrong, key=lambda r: -(r[2] - r[1]) / r[1]):
        print(f"     {key[0]+'/'+key[1]:28s} A4 {a4:8.1f}s -> A5 {a5:8.1f}s "
              f"{100*(a5-a4)/a4:+6.1f}%")
    print("\n  A4（全 GPU prefill）vs A5（全 PIM prefill），按模型：")
    print(f"     {'模型':11s}{'num_hbm':>8s}{'任务数':>7s}{'A5-A4 中位':>12s}"
          f"{'最差格子':>12s}")
    models = sorted({m for m, _ in tasks}, key=lambda m: NUM_HBM.get(m, 99))
    for m in models:
        g = [((tasks[k]["A5"]["makespan_s"] - tasks[k]["A4"]["makespan_s"])
              / tasks[k]["A4"]["makespan_s"], k[1]) for k in tasks if k[0] == m]
        if g:
            w = max(g)
            print(f"     {m:11s}{NUM_HBM.get(m, 0):8d}{len(g):7d}"
                  f"{100*st.median([x for x, _ in g]):11.1f}%"
                  f"{100*w[0]:9.1f}% {w[1]}")
    # PIM prefill is a win almost everywhere, so the interesting thing is not
    # the average but the exceptions: name every cell where it actually loses.
    lose = sorted(((tasks[k]["A5"]["makespan_s"] - tasks[k]["A4"]["makespan_s"])
                   / tasks[k]["A4"]["makespan_s"], k) for k in tasks
                  if tasks[k]["A5"]["makespan_s"] > tasks[k]["A4"]["makespan_s"])
    print(f"\n  PIM prefill 实际更慢的格子（全部 {len(tasks)} 个任务里）：")
    if not lose:
        print("     无")
    for d, k in sorted(lose, reverse=True):
        js = tasks[k]
        fired_here = any(abs(js["A5"][m] - js["A6"][m]) >= 1e-9
                         for m, _ in METRICS[:3])
        print(f"     {k[0]+'/'+k[1]:28s} {100*d:+6.1f}%   "
              f"A6 {'救回' if fired_here else '未触发'}")
    return same, fired, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--verify", type=int, default=0,
                    help="cross-check this many tasks against dag_*.json")
    a = ap.parse_args()
    root = a.root or open(f"{REPO}/output/_orch2/CURRENT_ROOT").read().strip()
    tasks = load(root)
    print(f"sweep: {root}")
    print(f"完整 9 档的任务: {len(tasks)}\n")
    if not tasks:
        return 1
    if a.verify:
        verify(root, tasks, a.verify)
    q1_pairs(tasks)
    q2_a6_vs_a3(tasks)
    q3_chooser(tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
