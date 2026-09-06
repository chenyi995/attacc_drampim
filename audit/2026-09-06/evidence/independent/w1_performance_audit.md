# W1 独立性能证据复核

只读检查当前源码、既有产物与纯 workload 交集；没有启动 Ramulator、GPU 定价或性能任务。
源码 HEAD：`074cbeb40e10b02fcb5a81815d6d03126cd86921`；证据时间：2026-09-06T07:50:34.480735+00:00。

## 当前 W1 没有完整性能结果

输入为 `/data2/chenyi9/KV-PIM/attacc_drampim_822/workload/probe/sweep/W1_turns.json`，SHA256 `6a19600760c71a46428644f2a8d1bf917159c2b7d361b4e87fda4bd1a15a9093`。
当前配置：{"format": "v2-dag", "kind": "main agent summarizing its workers (gen_main_workers.py, 2026-09-06)", "block_tokens": 256, "rounds": 24, "workers": 2, "sessions": 2, "worker_lout": 128, "main_lout": 128, "doc_tokens": 256, "note_tokens": 16}；145 个请求。
`scratch_0905/w1.log` 记录旧 W1 于 00:38:05 被停止并删除；`proto_w1_CACHEBLEND-TINY` 当前不存在。
新目录 `scratch_0905/proto_w1new_CACHEBLEND-TINY` 于 00:38:10 只启动 A3b/A4e；本次检查完整 dag JSON 数量为 0。没有当前 W1 A5/A6 结果。
session §26:577 所述旧 W1 七档仍在跑、作为改前对照已经过时。新跑早于提交753044b，但启动脚本指明旋转diff实现；最终还需对照run_config，不能仅凭目录名钉住源码。

## W1 跨会话共读没有进入当前整批 MQ 路径

`src/workload_runner.py:3979-3985` 将整个active集合按Q-ready排序后每batch8个切组；本W1每轮最多7个（含只生成1 token的owner），其余6个，始终只有一个整组。
`src/workload_runner.py:4037-4046` 取整组所有请求的物理master交集；4052仅非空时建shared scan，没有按共读子集再拆组。
纯输入核查：24 个tier、3072 个tier/output-step的整个active组，原始fingerprint交集全部为空。新生成的输出按request私有，不会增加整组共读。这里是静态证明，不是实跑的scan计数；假定不存在非预期地址别名。
同号worker跨两个会话确实共读：首轮每对256行，末轮每对6144行，但两个main与另号worker不共享这些集合。
TINY为GQA1：private scan只有1 query，而`src/ramulator_wrapper.py:571`要求shared_queries>1才启用MQ命令。因此“两个会话给decode MQ材料”的预期与当前整批交集实现不符。
LLAMA3的private scan仍有GQA4个Q heads共享同一KV head（runner:4118-4123），可以使用head内部MQ；这不是跨会话MQ。Prefill会把同一request的多个计算位置组成sweep，仍可用MQ，不能由decode交集为空推断A5毫无收益。

## 留存的 M 小跑是旧输入、旧 diff 布局

`/data2/chenyi9/KV-PIM/scratch_0905/proto_m_CACHEBLEND-TINY/M_main_workers_r16_w4_s1`；r16/w4/s1，81请求。报告的`run_config`为958dd24、git_dirty=true，不足以复原全部未提交源码。它不是r24/w2/s2的W1，也不是新旋转/按agent分组diff性能。

| 档位 | E2E s | TTFT mean ms | TBT weighted us | scan private us | scan shared us | scan step elapsed us | prefill PIM/GPU rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| A3b | 0.354053063 | 4.456150384 | 145.664947115 | 1.926335926 | 1.709025900 | 3.263293155 | 0/75840 |
| A4c | 0.360503681 | 4.461307669 | 150.280722914 | 2.020583541 | 1.709025900 | 3.927674102 | 0/75840 |
| A4e | 0.358280327 | 4.460411160 | 149.160542849 | 1.661473066 | 1.342725066 | 2.760974329 | 0/75840 |
| A5 | 0.367204980 | 5.371044699 | 149.160542849 | 1.661473066 | 1.342725066 | 2.760974329 | 75840/0 |
| A6 | 0.337992730 | 3.207174358 | 149.160542849 | 1.661473066 | 1.342725066 | 2.760974329 | 9152/66688 |

该M输入的A4e→A5 E2E降幅为 -2.490969%（负数表示变慢）；A5→A6为 7.955298%。
A4e/A5/A6的decode service与decode scan能量相同；报告虽然有shared_service，但全部4096个shared sweep只有1个成员，是worker输出结束后main单独decode，被命名为shared，不是multi-query收益证据。

## A5/A6 的局部价格与合理性

M的side log共81个决策，79个PIM、2个GPU；脚本逐条验证side与t_bank<=t_xpu一致。

| 请求 | m | R | GPU price us | PIM price us | side |
|---|---:|---:|---:|---:|---|
| a0_corpus | 16640 | 0 | 5719.505455 | 12848.635893 | gpu |
| g00_m_t000 | 32 | 0 | 1.512369 | 2.654645 | gpu |
| g00_w0_t000 | 40 | 248 | 11.033052 | 4.779407 | pim |
| g00_m_t001 | 16 | 160 | 5.910784 | 1.570327 | pim |
| g00_m_t015 | 48 | 5728 | 197.295882 | 31.315304 | pim |

旧M的A6把大语料导入与首个main的fresh prefill放GPU，其余放PIM；A5全部PIM。因此二者改善来自prefill选边，不能归因decode MQ。局部服务价格不包含全部排队/重叠，不能将其差的简单求和当作E2E保证；旧M的选边更不能外推成当前W1或LLAMA3必然同侧。

## Flash、流水与最慢通道

旧M每档run_config显式记录gpu_model=flash、pipeopt=true、powerlimit=true、TINY、A100a、ngpu1、num_hbm5、batch8、NVLink3；overlap_validation通过。
当前Flash分支见`src/devices.py:57-59`；decode head切分见runner:3065/3891，回填调度见runner:2601以后。ff6f225的回填/head流水已在958dd24前。
实际PIM执行保留每channel真实Ramulator服务时间并发到独立PIM:pool；A6每个sweep的估价在runner:4399取所有lane实测time最大值，不是按token最多lane或求和。summary.decode_scans区分max-lane service与elapsed。
新W1 sweep.log也明确flash、TINY、ngpu1、hbm5、k8、batch8；尚无最终报告可检查验证结果与源码指纹。

## 手工形状估计不是执行模型

`output/analysis/b1_levers.py:139-143`仍用ceil(m/8)、4.05us×lane/1536、100us×m*n/(816*2864)、旧6.06us/335GBps回读拟合。它忽略GQA的query容量、实际extent/最慢lane和当前GPU分块/occupancy，不是实际Ramulator选边。
脚本自己在18-25行声明校准来源；协议/README引用它时只能当结构与粗估线索，不能用它保证当前W1的A5/A6性能或真实PIM请求数量。
实际执行 `_resolve_prefill_side` 使用GPU模型与真实Ramulator每lane返回值，不调用这个probe；不能因此把实际PIM timing说成手工拟合。

可复核提取脚本：`/tmp/w1_model_provenance_c78dc76.py`；数据与源文件hash：`/tmp/w1_model_provenance_c78dc76.json`。
