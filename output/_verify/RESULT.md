# Option-A 验证结论 (2026-08-30)

基线 HEAD: 25dcc80 (clean)   补丁: token_offset O(L)->O(1) + common_keys hoist

## 身份验证:通过

| 关卡 | 结果 |
|---|---|
| rows 不变量(真实数据 352 block / 1,165,312 行) | 升序 0 例外、重复 0 例外、bisect!=index 0 例外;352/352 连续 |
| 单元测试 | Ran 61 tests in 131.694s -- OK |
| 35 对报告逐字节(含 8 个 25GB+ 完整事件流) | byte-identical 35 / differing 0 / missing 0 |
| 第 36 对 | 排除 ramulator_signature_cache 后 0 差异;该块的 4 次 hit/miss 归属差异源于 --ramulator-workers 4 vs 2(已用 workers=2 重跑复核) |

排除项依据:docs/sessions/2026-08-27.md 夜 10 的既有先例(缓存统计属运行状态,非仿真结果)。

## 加速比:小

建图耗时(single build):

| run | before | after | 收益 |
|---|---|---|---|
| LLAMA-7B_wl_D1_A1 | 3884s | 3667s | 5.6% |
| GPT-13B_wl_N4_A1 | 2567s | 2492s | 2.9% |
| LLAMA3-8B_pipeline_A1 (evfull) | 1131s | 1044s | 7.7% |
| LLAMA3-8B_pipeline_A1 (evnone) | 1037s | 977s | 5.8% |
| 其余 28 档 | -- | -- | -3%~+4%(噪声 +-3%) |

绝对节省 ~217s/run,与 O(L^2) 项的估算(~288s)吻合;但建图总时长是 3884s,所以占比只有 5.6%。
收益不随 L 上升 -- "L^2 项在大配置占比更高"的推断被实测否定。

## 方法学缺陷(已记录,避免重犯)

1. 用 900s 前缀采样推断 2000s+ 的建图 -> TLB 绑定集中在早期,占比被系统性高估(预测 15%,实测 5.6%)。
2. harness 的 `rc=$?` 前有 $(date) 命令替换,`$?` 被重置,失败一律显示 rc=0(一个 timeout 被误记为成功)。已修。
3. 在报告仍在写入时启动比对,产生假 SIZE DIFF。已改为矩阵完成后再比。
4. 补齐基线的作业 --ramulator-workers 与矩阵不一致(4 vs 2),引入唯一的星号。已重跑消歧。

## 结论

补丁值得保留(净收益、逐字节等价、消除一个真实 O(L^2)),但它不解决跑批时长:
建图时间的 94% 在别处,而当前没有可信的热点图。下一步应先做全程分段采样。
