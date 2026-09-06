# 2026-09-05：四项延迟指标与每 KV head 八通道

chenyi9 确定两点：报告 decode scan latency 减少、TBT、TTFT、E2E；布局分析使用每 KV head 8 个 channel，考虑共读集中在少数通道的情况。沿用仅 audit 和文档、修改需记录原因的范围。

## 为什么更新

上轮结构收益和 lane 时间之和不能替代用户要求的 scan latency；原小跑只有两个通道/head，贡献 README 的四通道又只是例子。因此新增一份明确的指标与配置口径，避免把理论八通道、旧两通道结果和真正执行延迟混在一起。

## 实际变更

- 新增 `audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md`：四项指标定义、已有 TTFT、scan 数据缺失原因、八通道的现有配置、受控集中访问的条件上限。
- 新增 `audit/2026-09-05/archive/metrics_eight_channels/`：保存只读提取与几何计算的 Python、JSON、摘要清单。
- 在 audit README、原布局上限报告及 session 索引增加新口径入口；原配置的结果保持其历史含义。
- 本 session 记录用户决定、分析依据、未冒充实测的条件。

## 分析与验证

主审读取 `src/config.py`、`_heads_per_hbm`、软件表、事件生成与运行入口，保存并执行 `/tmp/metrics_eight_channels_geometry.py`。仅调用布局 helper：验证 LLAMA3-8B/TINY、1/2/4 HBM 对应每 head 2/4/8 通道；等成本八块在 1/2/4/8 通道的条件加速为 8/4/2/1 倍。加入完整 corpus 共读关系后，当前软件表无法得到相同改善；这是限制收益的新增表逻辑条件，未代用户决定改动算法。

独立 agent `attacc_model_provenance` 保存并执行 `/tmp/metrics_eight_channels_existing_run.py`，只读上轮归档 small/out 结果，核验首 token 的完成定义和 TTFT 起点。该五请求 case 可按共同 release=0 算 TTFT；多轮 case 不能把全局 first_token 时间直接当逐请求 TTFT。现有 events=null，缺 scan 持续时间，所以 scan latency 列保持缺失，未把 lane-sum 降幅补进去。

`/tmp/metrics_eight_channels_document.py` 读取证据生成数表、复制 Python/JSON 并记录 SHA256，同时检查读取的实现源码摘要未变。文档链接按本地路径检查。没有修改实现、运行入口、collector、输入或已有结果，没有启动 Ramulator 或性能 sweep。

本轮完成了用户指定的报表与八通道假设的审计定义；八通道下四项指标的新实测值尚不存在，不以旧小跑替代。完整解释见 [指标与八通道专项](../../audit/2026-09-05/METRICS_AND_EIGHT_CHANNELS.md)。
