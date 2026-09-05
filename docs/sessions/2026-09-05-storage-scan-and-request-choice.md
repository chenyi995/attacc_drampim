# Session：存储与扫描对应性、A6 逐 request 口径

日期：2026-09-05。审查对象仍为 `cdd89db04a85edae029fd3151165f1a488d6139c`，AttAcc 基点 `c600051`。这是 [前轮公平性复审](2026-09-05-cdd89db-fairness-reaudit.md) 的继续审计；起始已有前轮文档改动和 `experiments/paper_ladder/`，均未撤销。未提交 commit、push 或修改实现。

## 1. 用户要求与本次裁决

用户补充：

> “再检查，每一档的存储和扫描是否对得上？扫描是不是按照实际存储扫描的？还是不是？此外，需要注意的就是，文中的A5- 6 选边，就是简单的逐个request 哪边快选哪个，文中公式可能不准确。”

继续遵守之前“不要修改代码，只能 audit”的约束。允许简单 workload 验证，并沿用用户已要求的独立 agent 审查；本轮重新请同一个独立审计 agent 专项复核。

判定上的具体修订：A5 固定 PIM prefill，A6 每 request 比较 GPU/PIM 预计 attention 耗时，哪边估计快就选哪边。**不再要求构造、试排两套候选 DAG，不要求队列状态或全局最优。** 原先依据论文 DAG/completion wording 作出的机制违规判断撤回。两侧成本与实际请求、布局和操作的对应性仍在审查范围内。

A1/A2 仍为可接受的独立自建 baseline；A3b 起按已确认 claim 逐级变化；DIE/TLB 额外费用继续排除，query 旋转仍归 GPU。没有因本次专项增加无来源 latency/energy。

## 2. 为什么修改文档

前轮已经发现 store 使用旧 TLB pool、scan 使用新 placement，但用户进一步要求明确每档是否真在扫描先前的存储。因此需要把“地址不持久”的概述补成逐档数据流和稳定反例，并排除两个误判来源：

- TLB 的 head-vector byte stride 与 trace 的 bank 展开 stride 不同，不能只比地址整数就判错。
- Ramulator 计时输入是真实 command trace，不等于其输入由同一份持久 KV 写入账本产生；报告中的 `dram_addresses` 也不等于 wrapper 计时的 extent。

另外，用户的新口径优先于旧论文措辞。当前 A6 的简单逐 request 规则本身应接受；继续要求双 DAG 会重复错误的审稿标准。相关主报告、建议 README 和独立报告必须同步，旧 session 则保留原过程并添加后续修订提示。

## 3. 实际审查与新证据

主审沿 `reserve/finalize → KVLocation → store/readback → placement → wrapper → trace generator` 逐段核查，另检查事件报告地址和 validator、prefill/decode 的生产依赖及 history 预置。查阅论文 `04-design.tex` 的持久 channel/row 和写入时 table 定义，以及 `05-execution.tex` 的旧选边文字；没有修改论文。

主审探针只写在 `/tmp`，导入现有真实 helper 和 DAG，使用设备桩，不启动 Ramulator：

1. 五档逐一调用 allocator/store/placement：A3b 单 head c1 写 ch2、扫 ch1；A4c/A4e/A5/A6 两 head diff 的 store 仅登记 ch15，scan 则使用 ch7/ch15。旧 master pool 与 table 最终通道也没有同一份字节分配。
2. 五档同一 TLB、同一 c16：单扫 K=0，共扫 c0+c16 时 K=1024。比较的是同单位 trace 地址，期间没有迁移。
3. A1 的两个 16-token private block 分别 K=0/8192，allocator scan_runs 保留它们；默认 decode 使用的 placement helper 对二者都生成 K=0、256-token padded scan。记录为 dense 抽象和物理等价性缺证，没有将 padding 直接判为不公平或 speedup。
4. A2 的 16-token prompt、2 decode step 真实 DAG：先写远端 16，读回 17，生成并写 1，再读回 18。计算宽度可含当前 token，远端回读却多算当前 token；本模型每步多 4096 B。设备桩不提供真实性能损失。
5. A6 选择函数使用人为透明的价格：最多 token 的 lane 只有 1 extent，另一 lane 更少 token 但 2 extents。现有函数只询价第一条 lane；同一设备价格覆盖全部 lane 后会改变选择。该例说明估价覆盖不同，不是实测误判率，也不恢复候选 DAG 要求。

独立 agent `independent_fairness_audit` 自行重新构造并执行五档通道/row 反例，未使用主审 JSON；更新自己的报告顶部、I05、逐档表、签字条件和新增 S01/S02。随后另请其只读复核新主报告是否夸大单位差、padding、设备桩价格及 A6 要求。

最终交叉检查通过：agent 未发现必须纠正的夸大，确认通道证据与单位差分开、A1 padding 未冒充性能惩罚、A2 多读 token 有事件支持、A6 已充分撤回双 DAG 要求。采纳其可选建议，将 A6 的 lane 表述限定为“本反例构造的全部 scan lanes”，并补上探针文本复制到 `/tmp` 的明确命令。

## 4. 逐文件变更与原因

| 文件 | 本轮修改及原因 |
|---|---|
| `audit/2026-09-05/STORAGE_SCAN_CONSISTENCY.md` | 新增逐档结论、三种地址视图、通道/row 反例、A1/A2 边界、生命周期、A6 新口径、修改建议和验证边界 |
| `audit/2026-09-05/storage_scan_probe.txt` | 归档 `/tmp` 探针为文本，保留复现方法，不接入生产代码或测试 |
| `audit/2026-09-05/storage_scan_evidence.json` | 归档 helper/DAG/选择函数的实际返回值；设备桩价格显式标记 |
| `audit/2026-09-05/storage_scan_manifest.json` | 新证据与更新文档的指纹、源码未变核验；前轮 manifest 仍保留为当时快照 |
| `audit/2026-09-05/REAUDIT_cdd89db.md` | 增加存储专项入口；修订结论、逐档表、R07、修复顺序，撤回双 DAG 要求 |
| `audit/2026-09-05/INDEPENDENT_REAUDIT.md` | 独立 agent 新增 S01/S02 并同步 A6 判定；独立证据保留其措辞 |
| `audit/2026-09-05/MODEL_PROVENANCE_REAUDIT.md` | 将 A6 表格的“同一个候选 DAG”改为请求/成本覆盖一致，并链接新口径 |
| `audit/2026-09-05/REPORT.md` | 只更新历史报告顶部提示，说明旧 F05 双 DAG 要求已经撤回；旧反例正文和 JSON 不覆盖 |
| `docs/README_audit_fixes.md` | 更新当前状态、A6 定义和验收；重写 §3.8 为简单逐 request 选边，保留成本对齐建议 |
| `docs/README_design_ladder.md` | 补充存储审计状态；区分当前“最多 token lane”估价与实际最慢 lane；撤销 A6 全局恒优断言 |
| `docs/README_run_guide.md` | 指向新专项；S5 的分流目标不再要求同时胜过两种固定策略 |
| `docs/README.md` | 最新审计入口改为存储专项与本 session，保持前轮链接 |
| `docs/sessions/README.md` | 新增本 session 的索引项 |
| `docs/sessions/2026-09-05-cdd89db-fairness-reaudit.md` | 顶部添加后续裁决，保留前轮实际过程 |
| 本文件 | 记录为什么修改、修改内容、探针性质和未完成项 |

这些变更仅影响审计结论、文档与复现证据。执行行为、数值结果、CLI 接口和模型参数均未改变。

## 5. 实际验证与证据边界

执行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 KVPIM_CPPCORE=0 KVPIM_PREFILL_SIDE_LOG= python3 /tmp/fugue_storage_scan_audit.py > /tmp/fugue_storage_scan_evidence.json
git diff --check
```

探针退出码为 0。主审检查 JSON 中五档 channel/row、A1 private block、A2 remote bytes 和 A6 询价列表；独立报告得到相同五档布局反例。为了隔离 A6 的两个 channel，临时探针加入明确的 co-read 集合后重新执行；第一次没有共读关系时 table 将所有对象放在 ch0，不能构成该估价反例，未把那个试探结果作为结论证据。

最终核验在 [manifest](../../audit/2026-09-05/archive/storage_scan_manifest.json) 记录：文档 diff 空白检查、更新 Markdown 的本地链接、HEAD、相对于前轮 source snapshot 的 92 个代码/测试/脚本/数据文件指纹，以及 tracked 非文档改动列表。前轮 manifest 的文档 SHA 是前轮快照；本轮更新过的报告以新 manifest 的指纹为准，未覆写历史证据。

未跑全套测试、性能矩阵、真实 Ramulator、GPU benchmark、RTL 或综合，未安装依赖，未修改 implementation/tests/workloads/现有实验结果。探针中的 µs 仅来自设备桩，不能引用为真实性能。A1 的 256-token padded scan 也没有被描述成 16 倍 E2E 差距。

## 6. 尚未实施的工作

当前不能保证“每档扫描的都是此前实际写入的同一份物理存储”。建议先统一持久的对象/版本/head 地址账本，让 store、readback、scan、资源争用与容量统计使用同一分配；再验证 A3b→A4c 只有 diff 变化、A4c→A4e 才改变 master table、A5/A6 继承同布局。A1 需说明 dense/padding 等价范围；A2 修正 remote read 的当前 token 数量。

A6 保留简单逐 request 选择即可，清理两侧成本覆盖和日志说明；不要求升级调度器。论文文字/公式建议按用户实际机制同步，但本轮没有修改论文。所有实现修复与新性能结果都尚未进行。
