> 历史记录：保留当时的技术证据和评价，当前待审事项统一看 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md)。

# 独立复审 8c51672：持久扫描几何已有实质修复，完整存储一致性仍不能确认

审查对象：`8c51672a3ef8b936340354b3211963cde8945c49`。独立执行者：`independent_fairness_audit`。账号：`chenyi9`（通过 `whoami` 获取）。已阅读仓库 [agent.md](../../../agent.md) 和 [session §12–13](../../../docs/sessions/2026-09-05-ladder-fixes-f01-f02-f04.md)。本报告重新读当前实现、运行独立布局 helper 与真实 reuse-plan 构造，不覆盖旧版本审计结论。

**结论：不能说当前已经全部正确；也不能继续照搬旧“扫描每次重排 row”和“master 读写额外带宽差率”的结论。** 新 `PhysicalLedger` 确实修复了扫描对象地址随读集改变、零 diff 子区间在 A3b/A4c 几何不同、同一次相邻修正不能合并等旧问题。但 STORE 事件仍不消费账本，预约集合会把本应被 own KV 隔开的重复 fingerprint 修正错误合为一 burst，且分配顺序尚不等同实际构图顺序。影响包含反向低估 Fugue 收益的情况，不能推定作者有意削弱 baseline。

本轮接受 chenyi9 的最新裁决：普通 DRAM store/read 不另收费，STORE 只保依赖；所有复用档共用随机修正集合；A1 使用精确 L；A6 简单逐 request 比较哪边快即可；decode 当前 token 在 GPU 本地计算是接受的共同模型。此处不重新提出已撤回的候选 DAG 要求。主审的执行/计量核查见 [当前主审](REAUDIT_8c51672.md)。

> **后续裁决更新：** 以下是原独立审计的技术事实与当时评价；新口径接受共同模型限制，是否算问题由 chenyi9 逐项裁决。当前状态及上游对应关系见 [相对 AttAcc 审阅清单](ATTACC_RELATIVE_FAIRNESS_REVIEW.md)，不自动要求按下列优先级修改。

## 已独立核对的修复

- **扫描 row 现在持久。** `PhysicalLedger.build` 固定 object 的 start，`extent_groups` 只查询 object ordinal，再以 start 加偏移生成 extent（`src/workload_runner.py:907`、`992`）。在五档同一 finalized TLB 上分别扫描 c16 与 c0+c16，同一 c16 extent 均保留；旧读集重新从零排地址的反例已关闭。
- **零 diff 的 master 几何已经统一。** 使用实际 `NaiveKVLayout` 与 `LocalDiffKVLayout`，同一大 fingerprint 的跨 block 子区间产生相同 extent。新代码按 `owner_row // _STRIPE_UNIT_ROWS` 划 block（`939–950`），不再让旧 Naive allocator 分页决定 scan 子区间碎片。
- **同一相邻 burst 的不同 fingerprint 修正可在 A3b 合并。** 连续 reserve 同 owner 的两个 diff 组，`954–974` 形成共同 burst，原来逐 fingerprint 各占一 row 的反例已修。其更一般的重复 fingerprint 情况仍有 N02 问题。
- **STORE 的额外时间和能量确为零。** `2447–2470` 无额外带宽分母、无 AttAcc mem 项；与裁决一致。旧 store 带宽惩罚和 pool 争用不能继续用于解释当前结果。
- **未读取的间隙现在保留。** `extent_groups:1013–1036` 只取被选 object/ordinal，不再把所有本次可见 diff 长度直接压成一份新布局。这是几何层面的修复，不等于 command generator 已经正确解释所有 extent；后者由并行独立核查覆盖。

## N01 / P1：STORE 仍未接入 PhysicalLedger；零收费已修，物理映射证据尚未贯通

`_append_channel_kv_stores` 仍按旧 `KVLocation.channel_base/channel_count` 分组，直接把旧 `key_address/value_address` 放进 STORE event（`2447–2470`）。它没有接收 TLB、policy 或 heads，没有调用 `physical_ledger`。实际 ledger 由 scan 第一次调用时懒构造（`1614–1622`、`1065–1070`）。decode 的 STORE 也直接使用 `output_location` 地址（`3434–3441`、`3869–3877`）。

独立控制：按 c0 master → c0 diff → c1 master 预约。调用 STORE helper 后每档的 `_ledgers` 都仍为空；随后调用 scan 才建立 ledger。A3b STORE 的 c1 地址属于旧 channel，ledger scan 把同一对象放到另一 channel。这里比较 channel，避免把 TLB vector stride 与 trace token stride 的单位差误作错误。附录从实际输出自动抽取此对照。

**影响边界：** STORE 现在没有延迟、能量和硬件资源占用，所以该不一致本身不再产生旧带宽惩罚；不能据此声称当前多计了任何具体时间。若 STORE 地址只是逻辑句柄，而 ledger 是唯一物理展开，这种抽象可以定义清楚，但当前报告仍把旧地址称为 DRAM 地址，且没有 STORE→object→head→ledger extent 的一致性验证。因此“实际 store 与 scan 已共同使用账本”的断言不能签字；这也不是实际硬件读错数据的实验。

## N02 / P1：重复 fingerprint 的修正跨 own KV 被错误合并，A3b 获得定义之外的 packing

根因发生在进入 ledger 之前：`CacheBlendTLB.reserve:1612` 用 `(layer,owner,fingerprint,kind)` 做 `setdefault`，后续同 key 的行全部加入第一次创建的集合。`PhysicalLedger.build:931–960` 按这个字典的一次 key 遍历检测 burst，无法看到该 key 两次写入之间曾插入别的对象。

真实 planner 的最小输入：owner 提供 doc；consumer 输入为 `prefix | doc | middle-own | doc`。两次 doc 使用同 fingerprint，前后均发生修正，中间有该 consumer 自己的 fresh KV。`build_reuse_plan(..., policy='epic')` 与 `_prepare_cacheblend_tlb` 生成有效 reservations；ledger 却把两组修正映射到同一个 burst object，输出一段紧凑 extent。具体段长度、修正位置、object id 和 extent 由脚本自动附在文末。

这与该 baseline 声明的“修正被 own KV 隔开时保持分离”不同。它**增强 A3b**，倾向缩小 A4c 聚集的可见收益，并非弱化 baseline；未测量端到端影响。`workload/probe/gen_sweep.py` 的 `_retrieved` 按 corpus 取模，长轮次的 interleaved 输入存在再次引用同 fingerprint 的可能，因此不是接口完全不支持的输入；本轮未统计整个 sweep 的触发率。

此处不以“本次 prefill 全部 QKV 一起计算”推翻用户接受的 interleaved 模型。审查的是实现能否遵循它自己用于定义 A3b 的 own-KV 分隔规则。

## N03 / P2：分配顺序仍来自原 JSON 预约顺序，不是当前 DAG 的写入顺序

`_prepare_cacheblend_tlb:3173–3181` 按 `workload.requests` 的原顺序预约，`PhysicalLedger.build:917–950` 再把该顺序视作 block 的 write order。但当前 DAG 为解决 owner 依赖已经在 `4679–4682` 按 request id 排序构图，planner 也按 `(tier,id)` 选 owner（`src/workload.py:456`）。

独立构造两个内容相同、仅 JSON request 数组顺序相反的 workload。它们的 owner 都是 a，DAG 构图顺序都为 a、b；但同一 shared master 的 ledger channel 不同。自动结果见文末。这不是同一 TLB 跨 scan 改地址，因而不能混同旧 row 持久性 bug；它说明预约时序与实际构图时序不是同一概念。

如果声明在执行前按输入顺序预分配全部对象，这是可以明确采用的模型。它尚不能被称为根据真实运行时写入时序构造的朴素 append；尤其 A4e 对照的碰撞模式会受此顺序影响。净性能方向未测，不判定哪一种顺序一定更有利。

## N04 / P2：A4c 的 diff cursor 跨 owner 共用，per-agent 跨轮连续性有适用边界

`PhysicalLedger.build:927` 只建立一个 `diff_cursor`；所有 layer/owner 的 diff 对象都在 `975–981` 推进这一个游标。独立 helper 依次预约 A 的 diff、B 的较长 diff、A 的另一 diff，仅扫描 A 时，B 所占的空隙正确保留。这关闭了旧“扫描时免费重打包”的问题，同时说明当前不是每 agent 一条独立增长的 diff 区。

**不能过度外推：** 当前 `_prepare_cacheblend_tlb` 通常按 request 连续预约同层的全部 diff，所以默认静态构图并不自然产生同一 owner 的 A→B→A 交错。这个控制证明的是面对真实跨 agent 交错追加时，per-agent 跨轮紧凑保证尚未建立；不能把它报告成默认 workload 已测到额外 ACT 或收益损失。

## N05 / 口径澄清：A3b 不是所有对象严格共享一个轮转计数器

master slot 来自过滤 diff 的 `_block_slot_table`（`917–925`、`940`）；修正 slot 来自包含 master 和先前 burst 的 `naive_index`（`944`、`965–966`）。因此 master → diff → master 可以产生相邻对象使用同一 slot，而不是严格按所有写入对象连续轮转。

这保留了 chenyi9 接受的 A3b/A4c master slot 相同约束，不能仅因它不是最字面的单流轮转就判为不公平。需要把 baseline 规则写成它实际执行的两个计数规则，避免同时声称“master 完全不受 diff 影响”与“所有对象都服从同一个写序轮转”。是否改变该规则属于实现选择，本轮只记录差异。

## 逐档可确认范围

| 比较 | 当前独立结论 |
|---|---|
| A3b → A4c | 零 diff master 几何通过；差异主体确为 diff 位置/packing。重复 fingerprint burst 合并和尚未贯通的 STORE 证据仍阻止全范围确认。 |
| A4c → A4e | `mode` 从 append 变 table（`913`），其余 ledger 分配路径相同；共读表现在按 block 粒度给 scan 固定 slot/row。不能由此声称 STORE 地址已经改成该位置。 |
| A4e → A5 | 复用同一 `TableLocalDiffKVLayout` 与 ledger policy；prefill PIM+MQ 是接受的机制包。存储侧没有发现额外换 allocator 的新开关。 |
| A5 → A6 | 共用上述 allocator；只对 request 增加选边符合裁决，不要求两候选 DAG。估价覆盖与实际操作由主审核查，本文不把旧结论直接标成当前未修。 |

## 复现与证据

本次独立脚本：`/tmp/independent_8c51672_storage_probe.py`；原始结果：`/tmp/independent_8c51672_storage_evidence.json`；终端输出归档：`/tmp/independent_8c51672_storage_probe.log`。由主审统一归档这些文件并记录指纹。脚本只构造 TLB、reuse plan、ledger 和零成本 STORE event；使用真实函数，不执行 Ramulator。执行方式：

```bash
PYTHONDONTWRITEBYTECODE=1 KVPIM_CPPCORE=0 python3 /tmp/independent_8c51672_storage_probe.py
```

以下表格和 JSON 由脚本读结果文件后插入，没有手工转录探针数值。结论以模型结构和几何为限。

| 档 | STORE 后账本数 | STORE channel | scan channel | STORE 时间/能量 | c16 持久测试 |
|---|---:|---:|---:|---|---|
| A3b | 0 | 2 | 1 | 0.0 s / 0.0 nJ | True |
| A4c | 0 | 0 | 1 | 0.0 s / 0.0 nJ | True |
| A4e | 0 | 0 | 1 | 0.0 s / 0.0 nJ | True |
| A5 | 0 | 0 | 1 | 0.0 s / 0.0 nJ | True |
| A6 | 0 | 0 | 1 | 0.0 s / 0.0 nJ | True |

```json
{
  "fixed_controls": {
    "zero_diff_subrange_actual_classes_equal": true,
    "distinct_fingerprint_same_burst_scan": [
      [
        2,
        1,
        [
          [
            2147483648,
            2155872256,
            16
          ]
        ]
      ]
    ]
  },
  "repeated_fingerprint_real_plan": {
    "consumer_segments": [
      [
        "prefix",
        8
      ],
      [
        "doc",
        256
      ],
      [
        "middle",
        256
      ],
      [
        "doc",
        256
      ]
    ],
    "diff_positions": [
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      520,
      521,
      522,
      523,
      524,
      525,
      526,
      527
    ],
    "diff_object_ids": [
      [
        0,
        "b",
        "burst",
        3
      ],
      [
        0,
        "b",
        "burst",
        3
      ]
    ],
    "diff_scan": [
      [
        2,
        1,
        [
          [
            2147483648,
            2155872256,
            16
          ]
        ]
      ]
    ]
  },
  "request_order_vs_allocation": {
    "a,b": {
      "reservation_key_order": [
        [
          0,
          "a",
          "shared",
          "master"
        ],
        [
          0,
          "b",
          "private",
          "master"
        ],
        [
          0,
          "b",
          "shared",
          "diff"
        ],
        [
          0,
          "a",
          "a::output",
          "master"
        ],
        [
          0,
          "b",
          "b::output",
          "master"
        ]
      ],
      "DAG_request_order_from_code": [
        "a",
        "b"
      ],
      "shared_scan": [
        [
          0,
          1,
          [
            [
              0,
              8388608,
              256
            ]
          ]
        ]
      ]
    },
    "b,a": {
      "reservation_key_order": [
        [
          0,
          "b",
          "private",
          "master"
        ],
        [
          0,
          "a",
          "shared",
          "master"
        ],
        [
          0,
          "b",
          "shared",
          "diff"
        ],
        [
          0,
          "b",
          "b::output",
          "master"
        ],
        [
          0,
          "a",
          "a::output",
          "master"
        ]
      ],
      "DAG_request_order_from_code": [
        "a",
        "b"
      ],
      "shared_scan": [
        [
          1,
          1,
          [
            [
              1073741824,
              1082130432,
              256
            ]
          ]
        ]
      ]
    }
  },
  "global_diff_cursor": {
    "A_only_scan": [
      [
        15,
        1,
        [
          [
            16642998272,
            16651386880,
            8
          ],
          [
            16642999328,
            16651387936,
            8
          ]
        ]
      ]
    ],
    "all_scan": [
      [
        15,
        1,
        [
          [
            16642998272,
            16651386880,
            8
          ],
          [
            16642998304,
            16651386912,
            256
          ],
          [
            16642999328,
            16651387936,
            8
          ]
        ]
      ]
    ]
  }
}
```
