# AUDIT:workload 有效性体检——哪些还能支撑重算类 claim(2026-08-25,只记录不动手)

起因(用户发现):Mooncake trace 的块哈希是**前缀链式**的(每个 id 唯一
绑定其全部前文与位置;实测 46,560 个 id 中前驱不唯一、位置不唯一者均为
0)。因此该 trace **构造上只能表达严格前缀共享**,任何"同一 chunk 出现在
不同位置/不同前文"的复用都表达不出来——而后者才是本项目重算机制
(EPIC 边界前缀 / CacheBlend 偏差重算 / diff 池 / Q 旋转 / D_i 位图)的
触发条件。

## 判据

一个 workload 能支撑"位置无关复用 + 选择性重算"类 claim,当且仅当它产生
**shifted 复用决策**(`build_reuse_plan` 中:段被复用且位置有偏移)。
进一步区分 shifted 的两种来源:
- **内容位移**:同一 chunk(doc/user 段)被不同请求放在不同 offset——
  真实的位置无关复用;
- **parent_out 规则**:`workload.py` 把 `parent_out` 段**一律**判为
  shifted。这对 relay 语义(父输出被注入到子请求的不同位置)是对的;对
  会话续写(上轮回复本就原位驻留、转换器也标 delta=0)是否应算 shifted
  **存疑**(见发现 W2)。

## 体检结果(EPIC k=8 口径,`build_reuse_plan` 实测)

| case | 复用决策 | shifted | 其中内容位移 | 重算 token | 判定 |
|---|---:|---:|---:|---:|---|
| multihop | 48 | 13 | **13(doc)** | 104 | **可用**——真实文档位移复用的主力(也是当前唯一主力) |
| relay | 8 | 4 | 0(4 条均为 parent_out,relay 语义下成立) | 32 | **可用**(DAG 位移复用,量小) |
| sharegpt | 43 | 43 | **1(user)** | 334 | **半可用**——42/43 靠 parent_out 规则;真实内容位移仅 1 段 |
| mooncakemt | 42 | 35 | **0** | 280 | **半可用**——35 条全是 parent_out 规则;跨会话共享块(17.5k tok)全为前缀共享 |
| mooncake | 167 | **0** | 0 | **0** | **不可用于重算类 claim**——纯前缀共享,重算机制零触发 |

不依赖 shifted 的结论(A2 的 decode 灾难、布局对比、容量/压缩率、前缀
共享下的放置)在五个 case 上仍然有效;失效的是"这些 case 检验了我们的
重算机制"这一层含义。

## 发现清单(均待裁决,未动手)

- **W1 Mooncake 前缀哈希局限**:trace 构造上无位移复用;矩阵里 mooncake
  行标注 "EPIC k=8" 实为**零重算**运行——恰是已被排除出矩阵的
  promptcache 形态,标签误导。处置方向:改标为"严格前缀共享/零重算
  对照象限",重算类 claim 的证据引用中剔除该行。
- **W2 parent_out ⇒ shifted 的一刀切规则**:会话续写场景(sharegpt/
  mooncakemt)中上轮回复原位驻留、无位移,规则仍强制 k=8 边界重算——
  保守方向(多算),但让"shifted 计数"虚高,也让这两个 case 的重算
  触发是**规则产物而非数据产物**。处置方向三选一:(a) 维持并在文档
  注明保守口径;(b) 规则细化为"parent_out 且 delta≠0 或 offset 不同";
  (c) 会话续写的 parent_out 归入 history 建模。
- **W3 覆盖面缺口**:真实内容位移复用目前集中在 multihop 的 13 段
  (104 个重算 token)——支撑本项目核心机制的数据面偏薄。处置方向:
  自建 `agentmix` 生成器(多 agent 共享同一批 chunk、组装顺序各异 →
  天然位移;内容用真实语料,输出长度可采样 mooncake 分布保真),和/或
  扩大 multihop 规模。
- **W4 Mooncake 场景若要保留其"agent 味道"**:必须自己跑出真实输入
  输出(脱敏哈希救不回来),即 W3 的生成器方案;trace 本身只保留作
  前缀共享对照。

## 关联

- 上一层台账:`README_manual_audit_findings.md`(本页结论待并入时另行
  裁决);claim 核对:`../experiments/paper_ladder/CLAIMS_CHECK.md`
  (mooncake 行的引用面待按 W1 修订);workload 说明:
  `README_workloads.md`(身份标注待按 W1 修订)。
- 本页仅记录;**未修改任何代码、未改任何实验、未重跑任何数据**。
