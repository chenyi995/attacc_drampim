# A6:Fugue(我们的方法)= A5 + 逐请求动态选边 (dynamic placement)

一句话:硬件与 A5 完全相同(master/diff 分池、MQ 批命令、bank-whole
能力),但 prefill 注意力**不再强制**进 PIM:**每个请求在运行时把两条路
各算一遍时间,谁便宜走谁,平局归 bank**;decode 恒在 PIM;冷请求(无可
复用/全重算类)归 GPU。A6 就是论文里的 Fugue 系统点,微架构侧对应
C 系列的 C3(MQ 命令 + 流式 P)。

## 1. 配置(`src/ablation.py::PRESETS["A6"]`)

| 旋钮 | 值 |
|---|---|
| prefill_attn | **dynamic** |
| decode_attn | pim |
| kv_mapping | master-diff |
| pim_batch_command | mq |

## 2. 动态规则(两条路各是什么)

对一个携带复用的层类/请求:

- **bank 路 (t_bank)**:Q 链路下行 + TLB 规划 + 每波共享扫描(≤ 容量条
  查询,MQ 间隔)+ PIM softmax + ctx 链路上行——即 A5 的那套;
- **xPU 路 (t_xpu)**:驻留复用/历史行经链路回读 (`kv_pim_to_gpu`) +
  GPU 全上下文注意力块——即 A4 的那套。

判定:`t_bank ≤ t_xpu → PIM,否则 GPU`(平局归 bank)。当前实现用
**仿真器自己的代价模型**现算两边(oracle 口径);论文 Eq.(placement) 的
闭式估计式作为决策输入的替换项留有 TODO(见代码注释),等审阅裁决。

## 3. 模型的每一步在哪里做

**解析路径**(`--ablation A6`):

| 步 | 做什么 | 代码位置 |
|---|---|---|
| 1–5 | 同 A5(解析/计划/组批/历史/投影) | 同 A5 |
| 6 | **逐层类决策块**:估 t_bank(`_pim_scan` 估价 + sfm + q/ctx 链路)与 t_xpu(回读 + GPU score/softmax/context 全上下文),取小者入账;GPU 侧条目以 `gpu_dynamic_*`/`link_kv_pim_to_gpu` 出现在 breakdown | `src/ablation.py::_prefill_batch` 的 `config.prefill_attn == "dynamic"` 块 |
| 7 | 全重算类(冷)固定走"GPU 重建 ∥ PIM 扫历史"、两支取 max | `_prefill_batch` 的 `carries_reuse == False` 分支(`prefill_overlap_saving`) |
| 8 | decode/报告 | 同 A5 |

**物理事件路径**(`--pim-prefill-mode dynamic`,CLI 默认值):

| 步 | 做什么 | 代码位置 |
|---|---|---|
| 1 | 每请求首个携带复用的层上,用同一事件代价模型估 t_bank(共享扫描估价×波数 + TLB 规划 + q/ctx 链路)与 t_xpu(回读 + GPU 块) | `src/workload_runner.py::_run_cacheblend_prefill` 的 dynamic 估计块 |
| 2 | 决策缓存于 `dynamic_prefill_sides[request_id]`(同请求各层一致),并写入报告字段 `pim_prefill_sides` 供审计 | 同上;报告组装处 |
| 3 | 按决策走 pim 分支(= A5 的 bank-whole 事件集)或 gpu 分支(= A4 的回读+GPU 块事件集) | 同文件两个分支 |

## 4. 怎么跑

```bash
# 解析路径
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json \
  --reuse epic --epic-prefix-recompute-tokens 8 \
  --ablation A6 --history-len 3 --pipeopt --workload-report /tmp/a6.json
# 物理路径(dynamic 是默认)
python3 main.py ... --history-len 3 --workload-report /tmp/a6_dag.json
python3 -c "import json;print(json.load(open('/tmp/a6_dag.json'))['pim_prefill_sides'])"
```

自检:`pim_prefill_sides` 里每请求的选边应与该请求的事件集一致
(pim → 有 `die_score_assembly`;gpu → 有 `gpu_prefill_score`);单测
`test_prefill_placement_menu_emits_matching_events` 锁住此契约。

## 5. 性质与保证

- **A6 永不劣于 A5/A4(解析口径)**:逐类取两侧代价的 min(单测
  `test_dynamic_prefill_places_each_class_on_the_cheaper_side`);
- 决策输入 = 模型实价,属 oracle 上界口径;换成论文闭式估计后会有
  次优判例,差距是论文可报的鲁棒性数据点(TODO,待裁决);
- 与 C3 的关系:A6 的 bank 路吃满 C3 微架构(MQ 命令、容量拆波、流式
  P);C 系列在固定放置下测微架构本身,A6 在系统层面把它用起来。
