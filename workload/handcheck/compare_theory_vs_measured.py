#!/usr/bin/env python3
"""手算 vs 实测：读 A3b/A4/A4b 的报告，逐 channel 对照 (chenyi9 2026-09-03)。

   usage: PYTHONPATH=$PWD python3 workload/handcheck/compare_theory_vs_measured.py [报告目录]
   报告目录默认 ./ ，需含 out_A3b.json / out_A4.json / out_A4b.json
   （用 --workload-report-events full 跑出来的）。
"""
import sys
import json, collections
from types import SimpleNamespace as NS
from src.workload_runner import (_striped_append_channel_extents as EX,
                                 _GEN_BYTES_PER_TOKEN, _GEN_ROW_BYTES)
S = sys.argv[1] if len(sys.argv) > 1 else "."
H = 4
def loc(o, f, k): return NS(owner=o, fingerprint=f, kind=k)
def act(n): return -(-n * _GEN_BYTES_PER_TOKEN // _GEN_ROW_BYTES)
POL = {"A3b": ("slice-append", False), "A4": ("master-diff-slice-append", True),
       "A4b": ("master-diff-table-append", True), "A5": ("master-diff-table-append", True),
       "A6": ("master-diff-table-append", True)}
def common_reads(shadow):                 # 15 个共享块，公共扫描没有 diff
    r = []
    for i in range(15):
        r += [loc("A_owner", "doc%02d" % i, "master")] * (256 if shadow else 248)
    return r
def private_reads(shadow, live=1):        # live 个自有 master 行 + 15 组 x 8 重算行
    r = [loc("B_reuser", "live", "master")] * live
    for i in range(15):
        r += [loc("B_reuser", "doc%02d" % i, "diff")] * 8
    return r
def theory(reads, pol):
    g = {c: p for c, _n, p in EX(reads, policy=pol, heads_per_hbm=H)}
    return {c: (sum(n for _, _, n in p), len(p), sum(act(n) for _, _, n in p))
            for c, p in g.items()}
def measured(tag, batch, steps):
    ev = json.load(open("%s/out_%s.json" % (S, tag)))["events"]
    g = collections.defaultdict(list)
    for e in ev:
        d = str(e.get("device", "")); r = str(e.get("request", ""))
        ok = r.startswith("batch:") if batch else (r == "B_reuser")
        if (d.startswith("PIM:pool") and ok and e.get("transformer_layer") == 0
                and "scan" in str(e.get("name", "")) and "decode" in str(e.get("name", ""))):
            g[r].append(e)
    if not g: return {}
    grp = max(g.values(), key=lambda x: sum(e["rows"] for e in x))
    per = collections.OrderedDict()
    for e in grp:
        c = int(e["device"].split("pool")[1].split("-")[0])
        per.setdefault(c, [0, 0.0]); per[c][0] += e["rows"]
        per[c][1] = max(per[c][1], e["time_s"])
    return per
for label, rd, batch, steps in (("公共扫描：15 个共享块", common_reads, True, 1),
                                ("private 扫描：自有行 + 15 组重算（两个 decode 步之和）", private_reads, False, 1)):
    print("=" * 100); print(label); print("=" * 100)
    print("%-5s %-9s %s" % ("档", "来源", "逐 channel 行数（理论 / 实测）"))
    for tag in ("A3b", "A4", "A4b"):
        pol, sh = POL[tag]
        if batch:
            t = theory(rd(sh), pol)
        else:   # 实测是两个 decode 步之和：第一步带 1 个自有行，第二步 0 个
            t1 = theory(rd(sh, 1), pol); t0 = theory(rd(sh, 0), pol)
            t = {c: tuple(a + b for a, b in zip(t1.get(c, (0, 0, 0)),
                                                t0.get(c, (0, 0, 0))))
                 for c in set(t1) | set(t0)}
        m = measured(tag, batch, steps)
        chans = sorted(set(t) | set(m))
        row = "  ".join("ch%d:%d/%s" % (c, t.get(c, (0,))[0],
                                        m[c][0] if c in m else "-") for c in chans)
        ok = all(t.get(c, (0,))[0] == (m[c][0] if c in m else 0) for c in chans)
        bt = max(t.items(), key=lambda kv: kv[1][2])
        bm = max(m.items(), key=lambda kv: kv[1][1]) if m else (0, [0, 0.0])
        print("%-5s %-9s %s" % (tag, "逐格" + ("一致 ✓" if ok else "不一致 ✗"), row))
        print("      理论最忙 ch%d: %d 行 / %d extent / %d ACT   |   实测最忙 ch%d: %.4f us"
              % (bt[0], bt[1][0], bt[1][1], bt[1][2], bm[0], bm[1][1] * 1e6))
    print()
