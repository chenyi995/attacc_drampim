# 2026-09-05：decode scan、GPU 速度与 TBT 传递审查

chenyi9 要求检查：GPU 更快是否暴露布局 scan 收益；单 scan 节省多少、多少体现到 TBT；decode 是否存在可并行却被串行或 pipeline 不充分的地方。沿用此前约束：只 audit；每个问题先说明 AttAcc 是否已有；共同上游近似不因收益低而重新裁定。

审核源码 revision：`167fe08e608b89e32f57402953314ced0b1194c6`。原始对照 `c600051`。已有 C1v2 TINY 五档运行记录是 `4b74849` dirty，源码审核时点与运行时点分开记录。旧 C1v1 A6 全事件仅用于证明具体等待结构，不与 C1v2 配对。

## 为什么更新

此前 Fig. 3b 只测 K 扫描，用户现在追问完整 decode。若直接把图中约 75% 与 C1 TBT 拼成保留比例，会混用 workload 和度量。另，已有 overlap checker 只复核预约规则，无法证明所有 GPU 空窗都充分利用。因此补写独立的扫描到 TBT 审查，保留百分比与绝对时间口径的区别。

## 实际工作和结论

- 读取现有五档 JSON，以运行记录的 workload SHA256 检查当前输入一致；核对修正计划 SHA 相同、Flash/pipeline 开启、批量和几何相同。
- 由 Python 脚本提取原始 scan/请求首末 token 时间，重算加权 TBT，全部数字从证据生成。
- A3b→A4e 的私有 scan 减少 29.256%，每步扫描跨度减少 13.646%，TBT 减少 1.752%。约 6% / 13% 只是相对降幅比，不能称为节省微秒的传递率。
- 原始大事件文件只读抽取标量/依赖，在已有时间戳中找到已就绪 GPU 操作能填入实际空窗的反例，未重排或重新定价生成性能数字。
- 独立 agent 复核尾部预约、Stage A/B、跨层构造与原 AttAcc head pipeline 的差异；排除常规路径 STORE fence 为延迟来源。
- 原 AttAcc 的 Norm/activation 固定项是共同成本，按用户口径只解释，不列修复建议。没有把 decode 小流量重新加固定启动费。

## 文件变更

- `audit/2026-09-05/DECODE_SCAN_TBT_PIPELINE.md`：本轮主报告，数值、条件公式、每项上游依据和已排除候选。
- `audit/2026-09-05/archive/decode_scan_tbt/`：提取/分析脚本、JSON 证据、独立报告；大中间文件放 `/tmp`。
- `docs/analysis/README.md`、`docs/audit/README.md`、日期审计入口、session 索引：加入该报告的用途入口，不覆盖此前历史结果。

本轮没有修改模拟器实现、论文或已有实验数据，没有启动新的 Ramulator/GPU 性能任务，没有 commit/push。报告中的额外串行是供 chenyi9 判断的候选；并未宣称修好后 TBT 或布局倍率必然增加。
