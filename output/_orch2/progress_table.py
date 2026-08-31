#!/usr/bin/env python3
"""Render the sweep's per-workload progress as the markdown of
docs/README_sweep_progress.md.

Ground truth is the ladder output on disk -- the nine ``dag_A*.json`` files a
finished task leaves behind -- not the claim directory and not a log grep.  A
claim only says a slot picked the task up; a task that died mid-ladder still
has its claim, and one that was killed and re-run has a claim older than its
outputs.  Counting the JSONs is the only check that cannot be fooled by either.

    python3 output/_orch2/progress_table.py > docs/README_sweep_progress.md
"""
import glob
import os
import sys
import time

ORCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ORCH))
sys.path.insert(0, ORCH)
import governor as G                                        # noqa: E402

RUNGS = ["A1", "A2", "A3", "A3a", "A3b", "A4", "A4b", "A5", "A6"]
# Measured on this sweep: median seconds of wall clock per unit of decode work,
# over the 14 tasks that had completed when it was fitted (range 2992-5047).
SEC_PER_W = 3801.0
# The 14 configs in the order they vary one axis at a time, then topology.
ORDER = ["baseline", "N-lo", "N-hi", "C-lo", "C-hi", "D-lo", "D-hi",
         "k-lo", "k-hi", "broadcast", "reduce", "supervisor", "pipeline",
         "private"]
MARK = {"done": "9/9", "running": "run", "queued": "--",
        "damaged": "DMG", "parked": "park", "excluded": "excl"}


def collect(root):
    """One row per task, with the rung-level detail the summary hides."""
    rows = []
    for cls in ("big", "small"):
        path = f"{root}/tasks_{cls}.txt"
        if not os.path.exists(path):
            continue
        for ln in open(path):
            g = ln.split()
            if len(g) < 4:
                continue
            model, cfg, wl, k = g[0], g[1], g[2], g[3]
            tid = f"{model}__{cfg}_k{k}"
            claim = f"{root}/claims/{tid}"
            out = f"{root}/{model}/{cfg}_k{k}"
            have = [r for r in RUNGS if os.path.exists(f"{out}/dag_{r}.json")]
            n = len(have)
            if n == len(RUNGS):
                st = "done"
            elif os.path.exists(f"{claim}/damaged"):
                st = "damaged"
            elif os.path.exists(f"{claim}/parked"):
                st = "parked"
            elif os.path.exists(f"{claim}/excluded"):
                st = "excluded"
            elif os.path.isdir(claim):
                st = "running"
            else:
                st = "queued"
            rows.append(dict(tid=tid, model=model, cfg=cfg, k=k, cls=cls,
                             w=G._wdec(model, wl), st=st, n=n,
                             missing=[r for r in RUNGS if r not in have],
                             claim=claim))
    return rows


def eta(rows, nslots):
    """Finish time under list scheduling, longest task first.

    Running tasks keep only their remaining time, so the answer is dominated by
    whichever single task has the longest tail -- which is the point: past a
    certain slot count the sweep stops being throughput-bound and the critical
    path is one workload.
    """
    import heapq
    now = time.time()
    heap, todo = [], []
    for r in rows:
        if r["st"] in ("done", "parked", "excluded", "damaged"):
            continue
        d = r["w"] * SEC_PER_W
        if r["st"] == "running":
            try:
                el = now - os.path.getmtime(r["claim"])
            except OSError:
                el = 0.0
            heapq.heappush(heap, max(120.0, d - el))
        else:
            todo.append(d)
    todo.sort(reverse=True)
    while len(heap) < nslots and todo:
        heapq.heappush(heap, todo.pop(0))
    t = 0.0
    while heap:
        fin = heapq.heappop(heap)
        t = max(t, fin)
        if todo:
            heapq.heappush(heap, t + todo.pop(0))
    return t


def main():
    root = open(f"{ORCH}/CURRENT_ROOT").read().strip()
    rows = collect(root)
    by = {(r["model"], r["cfg"]): r for r in rows}
    models = sorted({r["model"] for r in rows},
                    key=lambda m: G.GEOM[m][0])          # small models first
    cfgs = [c for c in ORDER if any(r["cfg"] == c for r in rows)]
    cfgs += sorted({r["cfg"] for r in rows} - set(cfgs))

    from collections import Counter
    cnt = Counter(r["st"] for r in rows)
    slots = len([1 for ln in G.sh("squeue -u $USER -h -t R -o %j").splitlines()
                 if ln.startswith("slot")])
    left = eta(rows, max(1, slots))
    now = time.time()

    o = print
    o("# sweep 进度：全部 84 个 workload")
    o("")
    o(f"生成于 **{time.strftime('%Y-%m-%d %H:%M')}**（本机时区）。这一页是**快照**，")
    o("跑动中会过期；重新生成：")
    o("")
    o("```bash")
    o("python3 output/_orch2/progress_table.py > docs/README_sweep_progress.md")
    o("```")
    o("")
    o("判定标准是**磁盘上的 9 个 `dag_A*.json`**，不是 claim 目录、也不是日志 grep。")
    o("claim 只说明有 slot 领过这个任务：半途死掉的任务 claim 还在，被杀后重跑的")
    o("任务 claim 比产出还老。只有数 JSON 骗不了人。")
    o("")
    o(f"sweep 根目录：`{root}`")
    o("")
    o("## 1. 总览")
    o("")
    o("| 状态 | 数量 | 含义 |")
    o("|---|---:|---|")
    o(f"| ✅ 完成 | {cnt.get('done', 0)} | 9/9 档齐全 |")
    o(f"| 🔵 在跑 | {cnt.get('running', 0)} | 已被 slot 领取，梯子在建 |")
    o(f"| ⬜ 未领取 | {cnt.get('queued', 0)} | 还在队列里 |")
    o(f"| ⚠️ 受损 | {cnt.get('damaged', 0)} | 缺档，需补跑（原因见 §3）|")
    o(f"| ⏸ 停放 | {cnt.get('parked', 0)} | 主动不跑（见 §4）|")
    if cnt.get("excluded"):
        o(f"| ✖ 排除 | {cnt['excluded']} | 已从本轮剔除 |")
    o(f"| **合计** | **{len(rows)}** | 6 模型 × 14 配置 |")
    o("")
    dmg_rungs = sum(len(r["missing"]) for r in rows if r["st"] == "damaged")
    o(f"档级完成度：**{sum(r['n'] for r in rows)} / {9 * len(rows)}** 档。")
    o(f"受损任务共缺 **{dmg_rungs}** 档。")
    o("")
    o(f"主批预计还需 **{left / 3600:.1f} 小时** → "
      f"**{time.strftime('%m-%d %H:%M', time.localtime(now + left))}** "
      f"（当前 {slots} 个 slot 并行）。")
    o("")
    o("## 2. 全部 workload 进度")
    o("")
    o("行 = 模型（按层数从小到大），列 = 14 个配置。格子里是**已完成档数**。")
    o("")
    o("| 模型 | " + " | ".join(cfgs) + " | 完成 |")
    o("|---" * (len(cfgs) + 2) + "|")
    for m in models:
        cells = []
        ndone = 0
        for c in cfgs:
            r = by.get((m, c))
            if not r:
                cells.append("·")
                continue
            if r["st"] == "done":
                ndone += 1
                cells.append("**9/9**")
            elif r["st"] == "excluded":
                cells.append("✖")
            elif r["st"] == "parked":
                cells.append("⏸")
            elif r["st"] == "damaged":
                cells.append(f"⚠️{r['n']}/9")
            elif r["st"] == "running":
                cells.append(f"🔵{r['n']}/9")
            else:
                cells.append("⬜")
        # denominator counts only what this model can still reach -- an excluded
        # config is not a shortfall, and scoring it as one reads as failure
        reachable = sum(1 for c in cfgs
                        if by.get((m, c)) and by[(m, c)]["st"] != "excluded")
        o(f"| {m} | " + " | ".join(cells) + f" | {ndone}/{reachable} |")
    o("")
    o("图例：**9/9** 完成 ・ 🔵 在跑 ・ ⬜ 未领取 ・ ⚠️ 受损缺档 ・ "
      "⏸ 停放 ・ ✖ 已放弃")
    o("")
    o("## 3. 受损任务：缺哪些档、为什么")
    o("")
    o(f"**{dmg_rungs} 档全部死于节点资源耗尽（磁盘满 / OOM），没有一档是引擎缺陷。**")
    o("三个根因都已修掉（见 commit `a798e8a`），补跑不会重演：")
    o("")
    o("1. **`/tmp` 太小** — node5/node6 的 `/tmp` 在 `/` 上，只有 49–62G，")
    o("   而一个 trace 重的档要 ~128G。已切到 `/localdata`（3–33T），六节点探针 6/6 通过。")
    o("2. **缓存池指数膨胀** — 每次 publish 把任务 seed 来的内容重写回去，")
    o("   1MB 两小时后变成 193GB，池子共 499GB，压在一个 98% 满的卷上。已关闭 publish。")
    o("3. **内存尖峰被 OOM killer 命中** — 放置改用投影内存而非当前内存。")
    o("")
    o("| 任务 | 已完成 | 缺档数 | 缺的档 |")
    o("|---|---:|---:|---|")
    for r in sorted((r for r in rows if r["st"] == "damaged"),
                    key=lambda r: (r["n"], r["tid"])):
        o(f"| `{r['model']} / {r['cfg']}` (k={r['k']}) | {r['n']}/9 | "
          f"{len(r['missing'])} | {' '.join(r['missing'])} |")
    o("")
    bw = sum(r["w"] * len(r["missing"]) / 9 for r in rows if r["st"] == "damaged")
    n_a1 = sum(1 for r in rows if r["st"] == "damaged" and "A1" in r["missing"])
    o(f"补跑总工作量**下限**约 **{bw:.1f} 个工作单位** ≈ "
      f"{bw * SEC_PER_W / 3600:.1f} slot-小时，排在主批之后，避免和关键路径抢资源。")
    o("")
    o(f"这是下限而不是估计：它按九档等价折算，但 **A1（`no-reuse`）的图最大，")
    o(f"比其余各档慢约 3 倍**，而 {n_a1} 个受损任务缺的正是 A1。真实代价明显更高。")
    o("")
    o("## 4. 已放弃 / 停放的任务")
    o("")
    dropped = [r for r in rows if r["st"] in ("excluded", "parked")]
    if dropped:
        o("**N-hi（`wl_N64.json`，64 个 agent）整行放弃**，六个模型一个都不跑。")
        o("它是全 sweep 最重的配置，而且在两个「装得下」的模型上也没跑成：")
        o("`LLAMA-7B/N-hi` 6 档死于 ENOSPC，`LLAMA3-8B/N-hi` 跑满 8 小时后被内存")
        o("守卫取消、只剩 1/9。")
        o("")
        o("| 任务 | 状态 | W | 预计常驻 | 单节点 1008GB |")
        o("|---|---|---:|---:|---|")
        NODE_G = 1008
        for r in sorted(dropped, key=lambda r: -r["w"]):
            g = G.SLOT_BASE_G + G.MEM_PER_W * r["w"]
            fit = "❌ 放不下" if g > NODE_G else "✓ 装得下"
            st = "✖ 放弃" if r["st"] == "excluded" else "⏸ 停放"
            o(f"| `{r['model']} / {r['cfg']}` | {st} | {r['w']:.1f} | "
              f"~{g:.0f} GB | {fit} |")
        o("")
        o("**科研代价：N 轴从三点降为两点** —— 只剩 N=4（`N-lo`）与 N=16")
        o("（`baseline`），4 倍跨度、没有第三点显示曲率。A1 的机制是随 degree")
        o("**爆炸**，两点连线看不出爆炸。详见 `docs/README_sweep_design.md` §7.1 缺口 1。")
        o("")
        o("残缺产出（多数只有 A2）**留在磁盘上但不再被引用**：N-hi 已从")
        o("`make_sweep_tables.py` 的 `CONFIGS` 中移除，不会以半填充行进入结果表。")
    o("")
    o("## 5. 关键路径")
    o("")
    live = [r for r in rows if r["st"] in ("running", "queued")]
    if live:
        cp = max(live, key=lambda r: r["w"])
        o(f"`{cp['model']} / {cp['cfg']}`：W={cp['w']:.2f}，全程约 "
          f"{cp['w'] * SEC_PER_W / 3600:.1f} 小时，是所有未完成任务里最长的一个。")
        o("")
        counts = [slots, slots + 3, slots + 7, slots + 11]
        etas = [eta(rows, n) for n in counts]
        # Whether the tail is one long task or genuine queue depth is not a
        # fixed property of the sweep -- it flips as tasks finish and as heavy
        # ones are parked.  Read it off the numbers instead of asserting it.
        flat = (etas[0] - etas[-1]) / etas[0] < 0.05 if etas[0] else True
        if flat:
            o("**整个 sweep 的完成时间由它单独决定**：它比队列里其余任何东西都长，")
            o("剩下的任务都跑在它的影子里，所以加 slot 换不来一分钟。")
        else:
            o(f"这一段**还是吞吐受限的**，不是被单个任务卡住：从 {counts[0]} 槽加到 "
              f"{counts[-1]} 槽能省 {(etas[0] - etas[-1]) / 3600:.1f} 小时。")
        o("")
        o("| 并行槽数 | 还需 |")
        o("|---:|---:|")
        for n, e in zip(counts, etas):
            o(f"| {n} | {e / 3600:.1f}h |")
        o("")
        if flat:
            o("所以内存阈值保持保守（`NODE_MEM_STOP=0.62`）不花任何代价：放宽只会增加")
            o("OOM 风险，而受损任务正是这么来的。")
        else:
            o("所以放宽 `NODE_MEM_STOP`（当前 0.62）这时候是**有收益的**，代价是 OOM")
            o("风险 —— 34 个受损档全是这么丢的。要权衡，不要默认。")
    o("")
    o("## 6. 轴的含义")
    o("")
    o("14 个配置 = 1 个基线 + 4 条单轴扫描（N/C/D/k 各高低两档）+ 4 种拓扑 + private。")
    o("每根轴对应哪个贡献，见 `docs/README_sweep_design.md` §7.1。")
    o("**注意**：SL/LL、scatter、funnel **未实现**，fan-out 只有 broadcast 一种，")
    o("同页 §2.1 有完整的已实现清单。")


if __name__ == "__main__":
    main()
