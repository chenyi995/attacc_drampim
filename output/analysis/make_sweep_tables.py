#!/usr/bin/env python3
"""Extract RESULTS_sweep.md from a completed parametric-sweep run.

Reads the newest output/sweep_<timestamp>/<config>_k<k>/dag_A*.json (13
configs x 9 rungs = 117 runs per model) and writes output/analysis/RESULTS_sweep.md:
per-config headline (makespan / total energy / average power / prefill-on-PIM
share, per rung A1..A6) plus the OFAT axis views (N / C / D / k) that isolate
each A-rung difference. All values are measured; source fields are labelled.

    python3 output/analysis/make_sweep_tables.py [output/sweep_<ts>]
"""
import json, os, glob, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                      # .../output
RUNGS = ["A1", "A2", "A3", "A3a", "A3b", "A4", "A4b", "A5", "A6"]
CACHE = HERE + "/.digest_cache"

# config name -> (topology, N, C, D, k). Matches experiments/run_sweep.sh.
CONFIGS = [
    ("baseline",   "alltoall",       16, 32, 2, 8),
    ("N-lo",       "alltoall",        4, 32, 2, 8),
    # N-hi (64 agents) dropped 2026-08-31.  It was the heaviest configuration in
    # the sweep -- W=17.33 for the small models, 52 for GPT-175B -- and it never
    # completed on any model: GPT-175B and LLAMA-65B project ~1.3TB and ~1.1TB
    # against a 1008GB node, LLAMA-7B lost six rungs to ENOSPC, and LLAMA3-8B was
    # cancelled by the memory guard eight hours in with eight of nine rungs
    # unfinished.  Partial output stays on disk but is deliberately not listed
    # here, so it cannot enter a results table as a half-filled row.
    # THE N AXIS IS THEREFORE TWO POINTS: N=4 (N-lo) and N=16 (baseline).
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


def collect(model_root):
    """{config name -> {rung -> metrics}} for one model's run directory."""
    rows = collections.OrderedDict()
    for name, topo, N, C, D, k in CONFIGS:
        d = cfg_dir(model_root, name, k)
        if not d:
            continue
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
                mk=x["makespan_s"], kj=x["energy_nj"] / 1e12,
                w=x["energy_nj"] / 1e9 / x["makespan_s"],
                pim=(100 * npim / len(sides)) if sides else 0.0,
                link=x.get("link_bytes", 0) / 2**30,
                dec_mj=dc / 1e6, pre_mj=(sum(be.values()) - dc) / 1e6)
        rows[name] = m
    return rows


def render(rows, model_label):
    """Markdown section (overview + topology view) for one model's rows."""
    md = [f"\n## 模型 {model_label}\n"]
    md.append(f"### 配置完整度（{len(CONFIGS)} 组 × {len(RUNGS)} 档）\n")
    md.append("| config | topology | N | C | D | k | 完整 |")
    md.append("|---|---|---|---|---|---|---|")
    for name, topo, N, C, D, kk in CONFIGS:
        got = "✅" if rows.get(name) and len(rows[name]) == len(RUNGS) else \
              (f"⚠️{len(rows.get(name, {}))}/{len(RUNGS)}" if name in rows else "✗")
        md.append(f"| {name} | {topo} | {N} | {C} | {D} | {kk} | {got} |")

    def block(title, key, nd):
        md.append(f"\n**{title}**\n")
        md.append("| config | " + " | ".join(RUNGS) + " |")
        md.append("|" + "---|" * (len(RUNGS) + 1))
        for name in rows:
            cells = [fmt(rows[name].get(A, {}).get(key), nd)
                     if isinstance(rows[name].get(A, {}).get(key), (int, float)) else "-"
                     for A in RUNGS]
            md.append(f"| {name} | " + " | ".join(cells) + " |")

    md.append("\n### 总览：每 config × 每档\n")
    block("Makespan (s)", "mk", 2)
    block("总能量 total energy (kJ)", "kj", 2)
    block("平均功率 average power (W)", "w", 0)
    block("prefill attention 落 PIM 的 agent 占比 (%)", "pim", 0)

    md.append("\n### topology 对照（fan-out / fan-in / all-to-all / chain）\n")
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
    return md


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if not root:
        cands = sorted(glob.glob(f"{OUT}/sweep_models_*") + glob.glob(f"{OUT}/sweep_*"))
        root = cands[-1] if cands else None
    if not root or not os.path.isdir(root):
        sys.exit("no sweep dir found; pass output/sweep_models_<ts>/ or output/sweep_<ts>/")

    md = ["# 参数化 sweep 结果（多模型 × 9 档, MQ PIM @ 1.30 GHz）\n"]
    md.append(f"由 `output/analysis/make_sweep_tables.py` 从 `{os.path.basename(root)}/"
              "[<model>/]<config>_k<k>/dag_A*.json` 自动生成。全部实测。\n")

    # Multi-model layout writes output/sweep_models_<ts>/<model>/<config>_k<k>;
    # a single-model root has <config>_k<k> directly.
    single = any(glob.glob(f"{root}/{n}_k*") for n, *_ in CONFIGS)
    complete = 0
    if single:
        rows = collect(root)
        md += render(rows, os.path.basename(root))
        complete += sum(1 for n in rows if len(rows[n]) == len(RUNGS))
    else:
        for model_dir in sorted(glob.glob(f"{root}/*")):
            if not os.path.isdir(model_dir):
                continue
            rows = collect(model_dir)
            if not any(rows.values()):
                continue
            md += render(rows, os.path.basename(model_dir))
            complete += sum(1 for n in rows if len(rows[n]) == len(RUNGS))

    md.append("\n## 附录：定义与来源\n")
    md.append("- **rung（9 档）**：A1 dense 无复用 / A2 GPU-only / A3 head→1 channel / "
              "A3a +写掩码 / A3b +head 切片 / A4 +master/diff 分离 / A4b +全局 co-read "
              "placement table / A5 prefill 进 bank+MQ / A6 动态选边。\n")
    md.append("- **模型几何**（层/hidden/heads/KV-heads）出处见 `src/config.py` 注释："
              "LLaMA-1 arXiv:2302.13971 Table 2、Llama-3 arXiv:2407.21783、"
              "GPT-3 arXiv:2005.14165 Table 2.1。\n")
    md.append("- **makespan** = `makespan_s`；**total energy** = `energy_nj`/1e12 kJ；"
              "**average power** = `energy_nj`/`makespan_s` W；**PIM %** = "
              "`prefill_attention_sides` 里 =='pim' 的占比。\n")

    path = f"{HERE}/RESULTS_sweep.md"
    open(path, "w").write("\n".join(md) + "\n")
    print(f"wrote {path}  ({complete} complete config-blocks across models)")


if __name__ == "__main__":
    main()
