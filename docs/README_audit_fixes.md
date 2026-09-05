# Audit 修改建议入口

**当前问题、具体 case 和修改方向统一放在 [CURRENT_ISSUES.md](../audit/2026-09-05/CURRENT_ISSUES.md)。** 这份入口不再维护另一套 P0/P1 清单，避免已修事项和后续接受的共同近似被反复当成必须修改。

每个 case 均给出论文对应段落、AttAcc 是否建模、当前实现差异、影响哪些档、证据边界和需要用户决定的内容。FlashAttention 必须开启；C2/C3/C6 已裁决修改，C5 按实际项估算并忽略 Q，C7 同轮/跨轮边界已定；C4 按共同近似记录，C8 按旧 master/diff 原址继续引用执行；本轮建模口径已定，实现仍待落实和验收。执行交接见 [当前裁决清单](../audit/2026-09-05/CURRENT_ISSUES.md#decisions)；本次审计没有实施修复。

A1/A2 的独立 baseline、A3b 起逐级机制、A5 的 prefill+MQ 机制包、A6 的简单逐 request 比价，以及共同 AttAcc energy/调度近似，继续按用户已确认口径理解。

原来的长篇建议已归档为 [历史快照](../audit/2026-09-05/archive/README_audit_fixes_history.md)。需要追溯原始证据时看 [archive 索引](../audit/2026-09-05/archive/README.md)。
