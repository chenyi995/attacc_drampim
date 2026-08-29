#!/usr/bin/env python3
"""Extract RESULTS_sweep.md from a completed parametric-sweep run.

Reads the newest output/sweep_<timestamp>/<config>_k<k>/dag_A*.json (14
configs x 7 rungs = 98 runs) and writes output/analysis/RESULTS_sweep.md:
per-config headline (makespan / total energy / average power / prefill-on-PIM
share, per rung A1..A6) plus the OFAT axis views (N / C / D / k) that isolate
each A-rung difference. All values are measured; source fields are labelled.

    python3 output/analysis/make_sweep_tables.py [output/sweep_<ts>]
"""
import json, os, glob, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                      # .../output
RUNGS = ["A1", "A2", "A3", "A3a", "A4", "A5", "A6"]
CACHE = HERE + "/.digest_cache"

# config name -> (topology, N, C, D, k). Matches experiments/run_sweep.sh.
CONFIGS = [
    ("baseline",   "alltoall",       16, 32, 2, 8),
    ("N-lo",       "alltoall",        4, 32, 2, 8),
    ("N-hi",       "alltoall",       64, 32, 2, 8),
    ("C-lo",       "alltoall",       16, 16, 2, 8),
    ("C-hi",       "alltoall",       16, 64, 2, 8),
    ("D-lo",       "alltoall",       16, 32, 1, 8),
    ("D-hi",       "alltoall",       16, 32, 4, 8),
    ("k-lo",       "alltoall",       16, 32, 2, 2),
    ("k-hi",       "alltoall",       16, 32, 2, 32),
    ("broadcast",  "broadcast",      16, 32, 2, 8),
    ("reduce",     "reduce",         16, 32, 2, 8),
    ("supervisor", "supervisor",     16, 32, 4, 8),
    ("pipeline",   "pipeline",       16, 32, 4, 8),
    ("private",    "alltoall(priv)", 16, 32, 2, 8),
]

NEED = ("energy_nj", "makespan_s", "link_bytes", "prefill_attention_sides",
        "energy_breakdown_nj")


def load(d, A):
    p = f"{d}/dag_{A}.json"
    if not os.path.exists(p):
        return None
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.basename(d) + "__" + A + "__" + str(int(os.path.getmtime(p))) + ".json"
    cp = f"{CACHE}/{key}"
    if os.path.exists(cp):
        return json.load(open(cp))
    full = json.load(open(p))
    dig = {k: full.get(k) for k in NEED}
    json.dump(dig, open(cp, "w"))
    return dig


def cfg_dir(root, name, k):
    hits = glob.glob(f"{root}/{name}_k{k}")
    return hits[0] if hits else None


def fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if not root:
        cands = sorted(glob.glob(f"{OUT}/sweep_*"))
        root = cands[-1] if cands else None
    if not root or not os.path.isdir(root):
        sys.exit("no sweep run dir found (output/sweep_<ts>/); pass it as arg")

    rows = collections.OrderedDict()          # name -> {rung -> metrics}
    for name, topo, N, C, D, k in CONFIGS:
        d = cfg_dir(root, name, k)
        if not d:
            print(f"skip {name} (dir missing)"); continue
        m = {}
        for A in RUNGS:
            x = load(d, A)
            if not x:
                continue
            sides = x.get("prefill_attention_sides") or {}
            npim = sum(1 for v in sides.values() if v == "pim")
            be = x.get("energy_breakdown_nj", {}).get("by_event", {})
            dc = sum(e for n, e in be.items() if n.lower().startswith("decode"))
            m[A] = dict(
                mk=x["makespan_s"],
                kj=x["energy_nj"] / 1e12,
                w=x["energy_nj"] / 1e9 / x["makespan_s"],
                pim=(100 * npim / len(sides)) if sides else 0.0,
                link=x.get("link_bytes", 0) / 2**30,
                dec_mj=dc / 1e6, pre_mj=(sum(be.values()) - dc) / 1e6,
            )
        rows[name] = m

    md = ["# 参数化 sweep 结果（LLAMA3-8B, MQ PIM @ 1.30 GHz）\n"]
    md.append(f"由 `output/analysis/make_sweep_tables.py` 从 `{os.path.basename(root)}/"
              "<config>_k<k>/dag_A*.json` 自动生成。全部实测；来源字段见附录。\n")

    md.append("## 配置（14 组）\n")
    md.append("| config | topology | N | C | D | k |")
    md.append("|---|---|---|---|---|---|")
    for name, topo, N, C, D, kk in CONFIGS:
        got = "✅" if rows.get(name) and len(rows[name]) == 7 else \
              (f"⚠️{len(rows.get(name,{}))}/7" if name in rows else "✗")
        md.append(f"| {name} {got} | {topo} | {N} | {C} | {D} | {kk} |")

    def block(title, key, nd):
        md.append(f"\n**{title}**（来源 `makespan_s`/`energy_nj`/`prefill_attention_sides`）\n")
        md.append("| config | " + " | ".join(RUNGS) + " |")
        md.append("|" + "---|" * (len(RUNGS) + 1))
        for name in rows:
            cells = []
            for A in RUNGS:
                v = rows[name].get(A, {}).get(key)
                cells.append(fmt(v, nd) if isinstance(v, (int, float)) else "-")
            md.append(f"| {name} | " + " | ".join(cells) + " |")

    md.append("\n## 总览：每 config × 每档\n")
    block("Makespan (s)", "mk", 2)
    block("总能量 total energy (kJ)", "kj", 2)
    block("平均功率 average power (W)", "w", 0)
    block("prefill attention 落 PIM 的 agent 占比 (%)", "pim", 0)

    # ---- OFAT 判别视图：每个轴看它放大哪对相邻档 ----
    md.append("\n## OFAT 判别视图（每个轴凸显哪对相邻 A 档）\n")
    axes = [("N", ["N-lo", "baseline", "N-hi"], "A1 dense 随共享度爆炸 → A1 vs 其余"),
            ("C", ["C-lo", "baseline", "C-hi"], "A2 link 成本 + A4/A5 prefill → A2 vs A5"),
            ("D", ["D-lo", "baseline", "D-hi"], "A2 每轮过 link + decode 布局档 → A2/A3/A3a/A4"),
            ("k", ["k-lo", "baseline", "k-hi"], "irregular access + 动态选边 → A3 vs A3a、A5 vs A6")]
    for axis, names, note in axes:
        md.append(f"\n**{axis} 轴** — {note}\n")
        md.append("| " + axis + " 点 | 指标 | " + " | ".join(RUNGS) + " |")
        md.append("|" + "---|" * (len(RUNGS) + 2))
        for name in names:
            if name not in rows:
                continue
            for lab, key, nd in [("makespan s", "mk", 2), ("energy kJ", "kj", 2),
                                 ("PIM %", "pim", 0)]:
                cells = [fmt(rows[name].get(A, {}).get(key), nd)
                         if isinstance(rows[name].get(A, {}).get(key), (int, float)) else "-"
                         for A in RUNGS]
                md.append(f"| {name} | {lab} | " + " | ".join(cells) + " |")

    md.append("\n## topology 对照（fan-out / fan-in / all-to-all / chain）\n")
    md.append("| topology config | 指标 | " + " | ".join(RUNGS) + " |")
    md.append("|" + "---|" * (len(RUNGS) + 2))
    for name in ["broadcast", "reduce", "baseline", "supervisor", "pipeline", "private"]:
        if name not in rows:
            continue
        for lab, key, nd in [("makespan s", "mk", 2), ("KV-link GiB", "link", 2),
                             ("PIM %", "pim", 0)]:
            cells = [fmt(rows[name].get(A, {}).get(key), nd)
                     if isinstance(rows[name].get(A, {}).get(key), (int, float)) else "-"
                     for A in RUNGS]
            md.append(f"| {name} | {lab} | " + " | ".join(cells) + " |")

    md.append("\n## 附录：定义与来源\n")
    md.append("- **makespan** = 调度长度（`makespan_s`）；**total energy** = "
              "`energy_nj`/1e12 kJ；**average power** = `energy_nj`/`makespan_s` W；"
              "**PIM %** = `prefill_attention_sides` 里 =='pim' 的占比。\n")
    md.append("- **rung**：A1 dense 无复用 / A2 GPU-only / A3 naive 布局 / "
              "A3a +写掩码 / A4 split-channel / A5 prefill 进 bank / A6 动态选边。\n")
    md.append("- k 是每 block 重算 token 数（run 时 EPIC_K）；N/C/D 是 workload 的 "
              "fan degree / context blocks / tier 数（见 `docs/README_sweep_design.md`）。\n")

    path = f"{HERE}/RESULTS_sweep.md"
    open(path, "w").write("\n".join(md) + "\n")
    done = sum(1 for n in rows if len(rows[n]) == 7)
    print(f"wrote {path}  ({done}/{len(CONFIGS)} configs complete)")


if __name__ == "__main__":
    main()
