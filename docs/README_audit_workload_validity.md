# AUDIT:workload 能否支撑重算类 claim(2026-08-25;只记录,未动代码/实验)

**判据(chenyi9 2026-08-25 收紧)**:能支撑,当且仅当**真实 workload**
的数据自身产生内容位移复用(同一 chunk 在不同请求中处于不同位置)。
两类都不算:规则强制的重算(`parent_out` 一刀切)与**合成设计品**
(位移是手写参数,不是负载事实)。判定只有两档:**可用 / 不可用**。

## 判定(EPIC k=8 实测)

| case | 内容位移 / 决策数 | 判定 |
|---|---|---|
| multihop | 13 / 48(doc 段,真实位移) | **可用(当前唯一;chenyi9 2026-08-25 确认:以 gold evidence 充当检索结果,在软件模拟器层面是合理用法)** |
| relay | 0 / 8(位移 Δ−150 为手写设计值;合成品) | **不可用** |
| sharegpt | 1 / 43(唯一一条为 3-token 重复短句,EPIC 整段重算,有效复用=0;其余全为规则产物) | **不可用** |
| mooncake | 0 / 167(trace 哈希为前缀链式,构造上无位移) | **不可用** |
| mooncakemt | 0 / 42(自行从 conversation trace 抽取拼接;重算全为规则产物) | **不可用** |

三个 Mooncake trace(toolagent/synthetic/conversation)已全部实测:
所有块 id 的前驱与位置**均唯一** → 只能表达严格前缀共享;且这是脱敏
方式所致——底层是否存在位移复用,trace 已不可恢复。

不依赖位移的结论(decode 灾难、布局、容量/压缩率、前缀共享放置)在
五个 case 上仍有效;失效的是"该 case 检验了重算机制"这层含义。

## 口径记录(非缺陷,报数须注明)

- **Token 化口径**(chenyi9 2026-08-25 记录):文本类 workload
  (multihop/sharegpt)的 token 计数用 tiktoken `cl100k_base`(GPT-4 系
  BPE),不是被仿真模型(LLAMA/GPT)各自的 tokenizer——同一文本计数
  一般差 ~10–15%。五个 case、六个档同一把尺,**比值类结论不受影响**;
  绝对 token 数与由此折算的绝对秒数带此近似,论文报绝对值需注明,或
  改用各模型 tokenizer 重计。Mooncake 系 trace 自带 token 计数,不经此
  口径。

## 待裁决事项

1. **W1** mooncake/mooncakemt 在矩阵中标注 "EPIC k=8" 实为零重算,标签
   需改为"前缀共享对照",重算类 claim 引用中剔除;
2. **W2** `parent_out ⇒ shifted` 一刀切规则在会话续写下强制多算
   (保守方向),处置:维持+注明 / 细化条件 / 归入 history;
3. **W3(chenyi9 2026-08-25 定方向)**:证据必须来自**真实 workload
   过真实复用软件**——找真实负载(语料/编排),放 CacheBlend(或同族
   软件)实际跑一遍,由软件产出复用/重算结果,再转成仿真输入。合成
   生成器只能当机制对照,不再作为证据补强路线。当前可用证据仅
   multihop(13 段/104 token),扩大其规模是唯一的既有补强。
