# 链路裁决：沿用原 NVLink，两级方案不采用

**状态：chenyi9 已裁决，讨论关闭。** 用户原话：“那就不要做两级了，就是NV Link 链接GPU 和之前一样”。

GPU 与 AttAcc/PIM 继续使用原 NVLink 连接，沿用现有 `--pim-link nvlink3`、链路计价和各档一致的硬件配置。此前提出的容量阈值、NVLink/PCIe 两级方案以及降带宽实验不作为当前实验要求。

这项决定只需同步文档：代码本来没有按容量自动分层实现，现有实验命令已经使用 NVLink。decode 小流量固定开销继续按此前裁决忽略。

原审计查明：AttAcc 有固定链路带宽配置，没有 context 阈值自动分层；当前仓库的 --pim-link 同样是整次运行固定选项。原始依据和推导保留为历史资料，不代表待实现事项：

- [裁决前链路审计原文](archive/link_tier/audit__2026-09-05__LINK_TIER_ASSUMPTIONS.md.before_nvlink_ruling.txt)。
- [裁决前性能分析原文](archive/link_tier/docs__analysis__README.md.before_nvlink_ruling.txt)。
- [上游 config](archive/link_tier/attacc_c600051_src_config.py.txt)、[上游 main](archive/link_tier/attacc_c600051_main.py.txt)、[上游 devices](archive/link_tier/attacc_c600051_src_devices.py.txt)。

当前执行口径见 [实验指导](../../docs/experiments/README.md#7-已确定的-nvlink-配置)，变更原因见 [session](../../docs/sessions/2026-09-05-docs-roles-k4-scan-max.md#最终链路裁决沿用原-nvlink)。
