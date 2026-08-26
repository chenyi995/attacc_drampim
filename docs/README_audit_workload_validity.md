# AUDIT:workload 能否支撑重算类 claim(2026-08-25;只记录,未动代码/实验)

**判据**:能支撑,当且仅当数据本身产生**内容位移复用**(同一 chunk 在
不同请求中处于不同位置,从而真实触发 EPIC/CacheBlend 重算)。规则强制
产生的重算(`workload.py` 将 `parent_out` 一律判 shifted)不算数据证据。
判定只有两档:**可用 / 不可用**。

## 判定(EPIC k=8 实测)

| case | 内容位移 / 决策数 | 判定 |
|---|---|---|
| multihop | 13 / 48(doc 段,真实位移) | **可用** |
| relay | 0 / 8(4 条 parent_out,relay 语义位移;合成对照) | **可用**(合成) |
| sharegpt | 1 / 43(其余全为 parent_out 规则产物) | **不可用** |
| mooncake | 0 / 167(trace 哈希为前缀链式,构造上无位移) | **不可用** |
| mooncakemt | 0 / 42(自行从 conversation trace 抽取拼接;重算全为规则产物) | **不可用** |

三个 Mooncake trace(toolagent/synthetic/conversation)已全部实测:
所有块 id 的前驱与位置**均唯一** → 只能表达严格前缀共享;且这是脱敏
方式所致——底层是否存在位移复用,trace 已不可恢复。

不依赖位移的结论(decode 灾难、布局、容量/压缩率、前缀共享放置)在
五个 case 上仍有效;失效的是"该 case 检验了重算机制"这层含义。

## 待裁决事项

1. **W1** mooncake/mooncakemt 在矩阵中标注 "EPIC k=8" 实为零重算,标签
   需改为"前缀共享对照",重算类 claim 引用中剔除;
2. **W2** `parent_out ⇒ shifted` 一刀切规则在会话续写下强制多算
   (保守方向),处置:维持+注明 / 细化条件 / 归入 history;
3. **W3** 内容位移证据只剩 multihop(13 段/104 token),偏薄——候选
   补强:自建多 agent 生成器(真实内容、组装顺序各异 → 天然位移),
   或扩大 multihop 规模。
