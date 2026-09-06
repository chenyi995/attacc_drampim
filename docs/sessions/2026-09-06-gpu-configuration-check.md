# Session：GPU 配置核对记录与提交

chenyi9 要求检查新增 H200/B200 的硬件参数是否传入各 A 档，随后明确 GPU–PIM 链路继续使用 NVLink3，并要求停止 audit、提交和推送已完成部分。本记录保存停止前已经完成的配置核对，不代表整仓库审计完成。

## 已完成部分与依据

[配置核对脚本](../../audit/2026-09-06/gpu_configuration_check.py) 通过真实命令行解析器、设备构造器和 ablation 参数入口捕获各 GPU、各 A 档的实际配置。它在进入 DAG 和 warm 之前拦截执行，PIM 对象不初始化 Ramulator；链路示例仅调用 GPU 解析计价。因此输出属于配置证据，不是 workload 性能结果。

[原始结果](../../audit/2026-09-06/GPU_CONFIGURATION_CHECK.json) 保留源码哈希与逐项捕获值；[配置表](../../audit/2026-09-06/GPU_CONFIGURATION_TABLES.md) 由同一脚本生成。保存这些文件是为了让另一台机器能够复核参数传递，而不需要启动完整实验。

已经完成的检查确认：同一 GPU 型号下，各 A 档接收相同的 GPU 算力、近端 HBM 带宽和 GPU 间互联带宽；GPU–PIM 两端接收相同的独立链路配置。chenyi9 裁决：GPU–PIM 固定使用 NVLink3，不能把它没有随 GPU 型号升级列为问题。结果中的显式其他链路控制项只验证参数传递，不属于实验协议。

配置表同时原样保留入口显存容量覆盖型号默认值的行为。此行为沿用原 AttAcc 的容量参数入口；这份配置捕获没有测量其性能影响，也没有修改它。

## 已执行的验证

停止 audit 前已成功执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 audit/2026-09-06/gpu_configuration_check.py
```

脚本检查同型号各档的 GPU 参数一致，并确认捕获前后相关源码哈希不变。此次提交将已完成的脚本、静态结果和本记录保存到当前分支，模拟器代码没有新增改动。
