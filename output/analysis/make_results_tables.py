#!/usr/bin/env python3
"""生成三个按 k 分的结果 README（RESULTS_k2/k8/k32.md），全部数值取自仿真
原始报告 dag_A*.json（实测，不估算）。正文中文为主，专有名词保留英文，每张表
标注来源的原始数据文件与字段。

layer = agentic workflow 的一层（一轮 / 一个 role stage），即报告里的 tier。
prefill placement 是 per request 决定的，同一 request 的 32 个 transformer
layer 都走同一侧，所以我们看的是 workflow layer（tier）。

energy 只按 phase（decode* vs 其余）、按 unit（GPU/PIM/LINK/DIE）、按整个 run
打标，不带 layer 标签，因此 power 只给 per rung。

    python3 make_results_tables.py   ->  RESULTS_k{2,8,32}.md
"""
import json, os, glob, collections

OUT = "/data2/chenyi9/KV-PIM/attacc_drampim_xinyao/output"
CACHE = OUT + "/analysis/.digest_cache"
RUNGS = ["A1", "A2", "A3", "A3a", "A4", "A5", "A6"]
WL = [("star_repair", "star-repair"), ("pipeline_repair", "pipeline-repair"),
      ("debate", "debate"), ("mapreduce_sum", "map-reduce"),
      ("multisource_rag", "multi-source RAG")]

# --- 每个 workload 的详细编排说明（专有名词保留英文）---
DESC = {
 "star-repair":
  "star topology（星型），仿 AutoGen / MetaGPT / AgentCoder。一个 **main** "
  "agent（300-token system prompt + 200-token task）指挥 **三个 worker**"
  "（各 300-token system prompt），共 **5 轮**。每轮 main 读三个 worker 的 "
  "256-token 回复、发一条 128-token instruction 给全体 worker；共享一个 "
  "47-chunk（12,032-token）codebase，分 5 个 stage 释放。20 个 request，最深 "
  "context 40,692 token。**layer = 编排轮次**（main 层与 3-worker 层交替，"
  "layer 0 是冷启动的 main）。",
 "pipeline-repair":
  "ChatDev / MetaGPT 式 waterfall。一个 **architect**（300-token system "
  "prompt、200-token task、256-token plan）开链，之后 **engineer**（256-token "
  "patch）与 **reviewer**（128-token review）交替 5 个 cycle，各自保留 "
  "history；共享 50-chunk（12,800-token）codebase，沿 12 个链位释放。最后一个 "
  "history-free 的 **tester** 读全部 50 chunk。最深 context 40,668 token。"
  "**layer = 12 个链位**（architect、engineer.c0、reviewer.c0、…、tester），"
  "每层 1 个 agent。",
 "debate":
  "multiagent debate / Mixture-of-Agents。**三个对称 debater** 就一个 "
  "100-token 问题辩论，共享 49-chunk（12,544-token）文档分 5 个 stage 释放；"
  "每轮各 debater 重读自己的 history 与两个对手的 256-token 答案，再答 256 "
  "token。最后一个 history-free 的 **judge** 只读三份终答、出 128-token 裁决。"
  "最深 context 41,616 token。**layer = 辩论轮次**（每轮一个 3-debater 层，"
  "最后一个 judge 层）。",
 "map-reduce":
  "map-reduce 摘要，**低复用对照（low-reuse control）**。8 个并发 **mapper** "
  "各读一段私有、不重叠的 24,576-token 切片 + 共享的 300-token system prompt，"
  "各出 200-token 摘要；一个 **reducer**（1,900-token context）汇成 256-token "
  "终稿。只有 system prompt 和一次性摘要可共享，共享比例很低。**layer = map 层"
  "，然后 reduce 层。**",
 "multi-source RAG":
  "12 个独立单轮 RAG 查询。每个查询用滑窗取 96 个不同的 256-token source "
  "chunk，窗口每次滑 1 个 source，所以相邻查询共享 96 中的 95 个 source"
  "（24,976-token 输入、64-token 答案）。全部共享是内容重叠，无 history、无 "
  "output 复用。**单 layer**，12 个查询。",
}


def newest_complete(stem, k):
    best, bn = None, -1
    for d in glob.glob(f"{OUT}/*{stem}*_{k}"):
        n = len(glob.glob(f"{d}/dag_*.json"))
        if n > bn:
            best, bn = d, n
    return best, bn


def any_dir(stem):
    best, bn = None, -1
    for d in glob.glob(f"{OUT}/*{stem}*_k*"):
        n = len(glob.glob(f"{d}/dag_*.json"))
        if n > bn:
            best, bn = d, n
    return best


def tier_role(name, qs):
    """从实测 agent ID 生成该 tier 的一句话说明（中文，专有名词保留英文）。"""
    ex = qs[0]
    if name == "star-repair":
        rnd = ex.split(".r")[-1]
        if ex.startswith("main"):
            if rnd == "0":
                return "round 0：main 读 200-token task，发 128-token instruction 给 worker"
            return (f"round {rnd}：main 读三个 worker 的 256-token 回复，发下一条 "
                    "128-token instruction")
        return (f"round {rnd}：三个 worker 各读 main 的 instruction，各产出一条 "
                "256-token 回复（在本轮释放的 codebase stage 上）")
    if name == "pipeline-repair":
        if ex == "architect":
            return "开链：把 200-token task 变成 256-token plan"
        if ex == "tester":
            return "history-free：读全部 50 chunk，测试成品代码"
        role, c = ex.split(".c")
        if role == "engineer":
            return f"cycle {c}：读 plan 和上一条 review，写 256-token patch"
        return f"cycle {c}：读 patch，写 128-token review"
    if name == "debate":
        if ex == "judge":
            return "history-free：只读三份终答，出 128-token 裁决"
        rnd = ex.split(".r")[-1]
        if rnd == "0":
            return "round 0：三个 debater 读 100-token 问题与文档，各答 256 token"
        return (f"round {rnd}：三个 debater 重读各自 history 与两个对手的 "
                "256-token 答案，再各答 256 token")
    if name == "map-reduce":
        if ex.startswith("map"):
            return ("map 层：8 个 mapper 各读一段私有、不重叠的 24,576-token 切片 + "
                    "共享 300-token system prompt，各出 200-token 摘要")
        return "reduce 层：reducer 读 8 份摘要，出 256-token 终稿"
    if name == "multi-source RAG":
        return ("12 个独立单轮查询；每个用滑窗取 96 个 256-token source chunk，"
                "窗口每次滑 1 个 source，相邻查询共享 96 中的 95 个")
    return "、".join(qs)


def tkey(t):
    try:
        return (0, float(t))
    except ValueError:
        return (1, str(t))


NEED = ("summary", "energy_breakdown_nj", "energy_nj", "makespan_s",
        "link_bytes", "prefill_attention_sides", "prefill_attention_rows",
        "decode_attn", "kv_mapping", "di_bitmap_bytes")


def load(d, A):
    p = f"{d}/dag_{A}.json"
    if not os.path.exists(p):
        return None
    os.makedirs(CACHE, exist_ok=True)
    key = (os.path.basename(d) + "__" + A + "__" +
           str(int(os.path.getmtime(p))) + ".json")
    cp = f"{CACHE}/{key}"
    if os.path.exists(cp):
        return json.load(open(cp))
    full = json.load(open(p))
    dig = {k: full.get(k) for k in NEED}
    dig["workload_tiers"] = full.get("workload", {}).get("tiers", {})
    json.dump(dig, open(cp, "w"))
    return dig


def per_request_latency(x):
    """rid -> (layer, prefill, decode, e2e=prefill+decode, valid)，单位 ms。
    valid=False 即 A2 GPU-only：它把 phase 合成一个时间戳（prefill_end==end、
    first_token==0），无法拆 prefill/decode。"""
    S = x.get("summary", {})
    reqs, tiers = S.get("requests", {}), S.get("tiers", {})
    out = {}
    for rid, r in reqs.items():
        t = str(r.get("tier"))
        a = tiers.get(t, {}).get("start_s", 0.0)
        pe, ft, en = r["prefill_end_s"], r["first_token_s"], r["end_s"]
        valid = ft > 0 and pe <= ft <= en
        pf = 1000 * max(0.0, pe - a)
        dc = 1000 * max(0.0, en - ft)
        out[rid] = (t, pf, dc, pf + dc, valid)
    return out


def rung_phase_valid(latA):
    v = [r[4] for r in latA.values()]
    return v and sum(v) >= 0.5 * len(v)


def energy_split(x):
    be = x.get("energy_breakdown_nj", {}).get("by_event", {})
    dc = sum(e for n, e in be.items() if n.lower().startswith("decode"))
    tot = sum(be.values())
    cls = x.get("energy_breakdown_nj", {}).get("by_class", {})
    return tot - dc, dc, tot, cls


def fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def build(K):
    kk = K[1:]
    md = []
    md.append(f"# k={kk} 阶梯结果（LLAMA3-8B，MQ PIM @ 1.30 GHz）\n")
    md.append("由 `make_results_tables.py` 自动生成。k 是 reuse policy 对每个 "
              f"shifted chunk 重算的 token 数（此处 {kk}）。每个 workload 取最新"
              "的完整 run；某 workload 在此 k 下不足 7 档则跳过。**所有数值均为"
              "仿真实测**，来源文件与字段在每处标注。\n")

    dirs = {}
    for stem, name in WL:
        d, n = newest_complete(stem, K)
        dirs[stem] = d if n >= 7 else None

    # ---- 编排（分 tier）----
    md.append("## 各 workload 的编排（分 tier 讲）\n")
    for stem, name in WL:
        tag = "" if dirs[stem] else "  *（此 k 下尚未跑完，不进下方数值表）*"
        md.append(f"### {name}{tag}\n")
        md.append(f"{DESC[name]}\n")
        ad = any_dir(stem)
        wt, src = {}, ""
        if ad:
            for A in ("A2", "A6", "A1", "A4"):
                dig = load(ad, A)
                if dig and dig.get("workload_tiers"):
                    wt = dig["workload_tiers"]
                    src = f"{os.path.basename(ad)}/dag_{A}.json"
                    break
        if wt:
            md.append(f"**分 tier**（tier map 取自 `{src}` 的 "
                      "`workload.tiers`；agent ID 为 workload 原样字段）:\n")
            md.append("| tier | agents | 干什么 |")
            md.append("|---|---|---|")
            for t in sorted(wt, key=tkey):
                qs = wt[t]
                ids = "、".join(qs) if len(qs) <= 6 else \
                      "、".join(qs[:5]) + f"、…（共 {len(qs)} 个）"
                md.append(f"| {t} | {ids} | {tier_role(name, qs)} |")
            md.append("")

    # ---- 名词定义 ----
    md.append("## 名词定义\n")
    md.append("- **makespan**（计算机语言）：整个 workload 的 **schedule length"
              "**（调度长度），即从第一个 request 到达、到最后一个 request 完成的"
              "墙钟时间 $\\max_i C_i$（所有 request 完成时间的最大值）。这是"
              "多处理器 scheduling 的经典目标（minimize makespan）；越小=整批 "
              "agent 越早全部跑完。它计入了 overlap 与 queueing，**≠ 各 request "
              "latency 之和**。来源字段 `makespan_s`。\n")
    md.append("- **layer**：agentic workflow 的一层（一轮 / 一个 role stage），"
              "即报告的 `tier`。各 workload 的 layer 见上表。prefill placement 是 "
              "per request 决定的，同一 request 的 32 个 transformer layer 走同一"
              "侧，所以随 workflow layer 变、不随 transformer layer 变。\n")
    md.append("- **latency**（per request，在一个 layer 内对该层 agent 取平均）："
              "**prefill** = 从 request 到达到它 prefill 结束；**decode** = 从它"
              "首个 output token 到最后一个；**e2e = prefill + decode**。到达取该 "
              "layer 的释放时刻（`summary.tiers[tier].start_s`），故不含等待上游 "
              "layer 的时间。来源字段 `summary.requests` 的 `prefill_end_s` / "
              "`first_token_s` / `end_s`。\n")
    md.append("- **power / energy**：轨迹只按 phase（`decode*` 事件 vs 其余一次性 "
              "prefill 事件）、按 unit（GPU/PIM/LINK/DIE）、按整个 run 打标，"
              "**不带 layer 标签**，所以 energy 与 average power（energy / "
              "makespan）**只给 per rung**。来源字段 `energy_breakdown_nj`、"
              "`energy_nj`。\n")
    md.append("- **rung（档）**：**A1** AttAcc dense 无复用（decode 在 bank、"
              "prefill 在 GPU）；**A2** GPU-only 软件复用；**A3** decode 进 bank、"
              "跑 append-order 布局；**A3a** + 陈旧行 write mask；**A4** + "
              "split-channel master-diff 布局；**A5** + 所有 prefill attention "
              "进 bank；**A6 = Fugue**，A5 + 动态 per-request placement rule。\n")

    # ---- 总览 ----
    head = collections.OrderedDict()
    for stem, name in WL:
        d = dirs[stem]
        if not d:
            continue
        rm, re_, rp, rw = [], [], [], []
        for A in RUNGS:
            x = load(d, A)
            rm.append(x["makespan_s"]); re_.append(x["energy_nj"] / 1e12)
            rw.append(x["energy_nj"] / 1e9 / x["makespan_s"])
            s = x.get("prefill_attention_sides") or {}
            npim = sum(1 for v in s.values() if v == "pim")
            rp.append(f"{100*npim/len(s):.0f}%" if s else "0%")
        head[name] = (rm, re_, rp, rw)
    if not head:
        print(f"{K}: 无完整 workload，跳过"); return

    md.append("## 总览：整条阶梯\n")
    md.append("各 workload 的数取自其小节标注的 run 目录下 `dag_A1.json … "
              "dag_A6.json`。\n")
    for title, idx, nd in [("Makespan (s)", 0, 2), ("总能量 total energy (kJ)", 1, 1),
                           ("平均功率 average power (W)", 3, 0),
                           ("prefill attention 落 PIM 的 agent 占比", 2, 0)]:
        md.append(f"\n**{title}**\n")
        md.append("| workload | " + " | ".join(RUNGS) + " |")
        md.append("|" + "---|" * (len(RUNGS) + 1))
        for name, rows in head.items():
            cells = [fmt(v, nd) if isinstance(v, (int, float)) else str(v)
                     for v in rows[idx]]
            md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append("\n**A6（Fugue）相对 A1、A2 的加速**\n")
    md.append("| workload | A6 vs A1 | A6 vs A2 | energy A6 vs A1 |")
    md.append("|---|---|---|---|")
    for name, (m, e, p, w) in head.items():
        md.append(f"| {name} | {m[0]/m[-1]:.1f}x | {m[1]/m[-1]:.1f}x | "
                  f"{e[0]/e[-1]:.0f}x |")

    # ---- 逐 workload ----
    for stem, name in WL:
        d = dirs[stem]
        if not d:
            continue
        base = os.path.basename(d)
        md.append(f"\n---\n\n## {name}\n")
        md.append(f"**原始数据目录** `{base}/`。下面各表的列 **A1…A6 分别取自该"
                  f"目录下的 `dag_A1.json` … `dag_A6.json`**。\n")
        lat = {A: per_request_latency(load(d, A)) for A in RUNGS}
        valid = {A: rung_phase_valid(lat[A]) for A in RUNGS}
        wl = load(d, "A6")["workload_tiers"]
        layers = sorted(wl.keys(), key=tkey)

        for phase, pi in [("prefill", 1), ("decode", 2), ("e2e = prefill+decode", 3)]:
            md.append(f"\n**每层 latency [{phase}]（ms）**  "
                      "来源 `dag_A*.json` 的 `summary.requests` + "
                      "`summary.tiers[tier].start_s`\n")
            md.append("| tier(layer) | " + " | ".join(RUNGS) + " |")
            md.append("|" + "---|" * (len(RUNGS) + 1))
            for t in layers:
                cells = []
                for A in RUNGS:
                    if not valid[A]:
                        cells.append("n/a"); continue
                    vs = [v[pi] for v in lat[A].values() if v[0] == t]
                    cells.append(f"{sum(vs)/len(vs):.1f}" if vs else "-")
                md.append(f"| {t} | " + " | ".join(cells) + " |")
            cells = []
            for A in RUNGS:
                if not valid[A]:
                    cells.append("n/a"); continue
                vs = [v[pi] for v in lat[A].values()]
                cells.append(f"{sum(vs)/len(vs):.1f}" if vs else "-")
            md.append(f"| **全部** | " + " | ".join(cells) + " |")
        md.append("\n> A2 是 GPU-only software baseline，此 sim path 把 prefill "
                  "与 decode 合成一个时间戳（`prefill_end==end`、`first_token=0`），"
                  "所以它的 per-phase latency 记 `n/a`；其 makespan 见总览。\n")

        # 每层 prefill 放置
        md.append("\n**每层 prefill attention 放置（HBM vs PIM），每个 agent 一个"
                  "点**  来源 `dag_A5.json` / `dag_A6.json` 的 "
                  "`prefill_attention_sides`（A1–A4 按构造 prefill 全在 GPU/HBM）\n")
        md.append("| tier(layer) | agents | A5 PIM | A5 HBM | A6 PIM | A6 HBM | A6 PIM% |")
        md.append("|---|---|---|---|---|---|---|")
        q2t = {q: t for t, qs in wl.items() for q in qs}
        s5 = load(d, "A5").get("prefill_attention_sides") or {}
        s6 = load(d, "A6").get("prefill_attention_sides") or {}
        pt = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
        for q, t in q2t.items():
            pt[t][0] += 1
            pt[t][1] += (s5.get(q) == "pim"); pt[t][2] += (s5.get(q, "gpu") != "pim")
            pt[t][3] += (s6.get(q) == "pim"); pt[t][4] += (s6.get(q, "gpu") != "pim")
        tot = [0, 0, 0, 0, 0]
        for t in sorted(pt, key=tkey):
            n, a, b, c, e = pt[t]
            for i, v in enumerate((n, a, b, c, e)):
                tot[i] += v
            md.append(f"| {t} | {n} | {a} | {b} | {c} | {e} | {100*c/n:.0f}% |")
        n, a, b, c, e = tot
        md.append(f"| **全部** | {n} | {a} | {b} | {c} | {e} | {100*c/n:.0f}% |")

        # 每档能量 + 功率
        md.append("\n**每档 energy 与 average power**  来源 `dag_A*.json` 的 "
                  "`energy_breakdown_nj`（by_event / by_class）、`energy_nj`、"
                  "`makespan_s`、`link_bytes`（power = energy / makespan，不分 layer）\n")
        md.append("| rung | prefill E (mJ) | decode E (mJ) | total E (mJ) | "
                  "avg power (W) | GPU | PIM | LINK | KV over link (GiB) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for A in RUNGS:
            x = load(d, A)
            pf, dc, tE, cls = energy_split(x)
            g = cls.get("GPU", 0)/1e6; pm = cls.get("PIM", 0)/1e6
            lk = cls.get("LINK", 0)/1e6
            md.append(f"| {A} | {pf/1e6:.1f} | {dc/1e6:.1f} | {tE/1e6:.1f} | "
                      f"{x['energy_nj']/1e9/x['makespan_s']:.0f} | {g:.1f} | "
                      f"{pm:.1f} | {lk:.2f} | {x.get('link_bytes',0)/2**30:.2f} |")

    path = f"{OUT}/analysis/RESULTS_{K}.md"
    open(path, "w").write("\n".join(md) + "\n")
    print(f"wrote {path}（{len(md)} 行，{len(head)} 个 workload）")


def main():
    for K in ("k2", "k8", "k32"):
        build(K)


if __name__ == "__main__":
    main()
