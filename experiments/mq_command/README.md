# MQ-MAC 批命令研究(2026-08-21)

设计与理由:`PLAN_mq_command.md`。本目录 = 该计划的 O5(微基准校验)与 O6(三方案对比)。

**MQ-MAC(multi-query MAC)语义**:`ACT_AB` 不变(整行进 row buffer);**一条 `MAC_AB`
读一次 col(32 B/bank),bank PE 把它与 GEMV buffer 里驻留的全部 n 条 Q 各乘加一次**
(全流水,每 PE 周期一次 op)。命令间隔 = max(功耗拉伸, PE 吞吐, DRAM 数据通路下限),
见 `src/ramulator_wrapper.py::mq_interval_cycles`。n 的驻留上限 = GEMV buffer 容量
(64 B/条;AttAcc 原配 512 B → 8 条),超界拆成连续 sweep。

三方案(每个数据点 = 一次 patched-Ramulator2 周期级运行,单头/单 HBM 视角,PC 档):

| 方案 | 含义 |
|---|---|
| dense | 每 agent 一份私有 KV,各自单独扫(AttAcc 参照形状),时间为各次扫描之和 |
| replicate | 共享一份 KV,一条 `MAC_AB` 服务一个 (col, Q) 组合——旧命令方案,列被重读 n 次 |
| mq | 共享一份 KV,MQ-MAC 命令——列只读一次 |

运行:`python3 experiments/mq_command/run_mq_study.py --workers 48`
(96 个数据点,签名缓存后 77 次 Ramulator 调用;结果 `results_mq_study.json`)。

## Part A:MQ 间隔模型 vs 实测(L=4096,单 sweep)

模型间隔(cycles)与实测总时间逐点吻合(总时间 ≈ 间隔 × 1024 条 MAC + 固定搬运段;
MAC 命令数与 ACT 数在所有 n 下不变:1024 条 / 16 次):

| PC | f_PE | n=1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| 是 | 0.666 GHz | 6 / 6220 ns | 7 / 7229 | 7 / 7530 | 8 / 8611 | 12 / 13266 | 16 / 17226 |
| 是 | 1.3 GHz | 6 / 6220 | 7 / 7229 | 7 / 7530 | 7 / 7514 | 8 / 9489 | 9 / 11526 |
| 否 | 1.3 GHz | 4 / 4694 | 4 / 4941 | 4 / 5234 | 5 / 6293 | 7 / 8697 | 9 / 11526 |

(单元格 = 模型间隔 cycles / 实测 ns。666 MHz 下 n≥4 转为 PE-bound;1.3 GHz 下
功耗拉伸主导,n=8 只需 9 cycles。)

## Part B:decode,N_ag 个 agent 共享一段 chunk(PC)

| L | N_ag | dense | replicate | **mq** | mq/dense | ACT(d/共享) | MAC 命令(d 或 r / mq) |
|---|---|---|---|---|---|---|---|
| 4096 | 2 | 12.4 µs | 11.5 µs | **7.2 µs** | 0.58 | 32 / 16 | 2048 / 1024 |
| 4096 | 4 | 24.9 µs | 22.6 µs | **8.6 µs** | 0.35 | 64 / 16 | 4096 / 1024 |
| 4096 | 8 | 49.8 µs | 45.0 µs | **17.2 µs** | 0.35 | 128 / 16 | 8192 / 1024 |
| 4096 | 16 | 99.5 µs | 89.9 µs | **34.5 µs**(2 个 sweep) | 0.35 | 256 / 32 | 16384 / 2048 |

(f_PE=0.666 GHz。1.3 GHz 时 mq/dense 进一步到 0.23,n=8 为 11.5 µs。L=1024 同形状。)

- **ACT 次数**:共享方案 = dense ÷ N_ag(sweep 拆分时 ×趟数)——Fugue Eq.(actcost) 的口径。
- **MAC 命令数(∝ DRAM 读能耗)**:mq = dense/replicate ÷ n,列只读一次是 mq 独有的节省。
- replicate 只比 dense 略快(省 ACT):旧命令方案的收益上限;mq 的差距来自去掉冗余列读。

## Part C:prefill,n_r 条 Q 扫同一段复用 KV(L=4096,PC)

与 Part B 同构:n_r=32 拆成 4 个 sweep,mq = 0.35×dense(666 MHz)/ 0.23×(1.3 GHz),
MAC 命令 ÷ n_r,ACT ÷ (n_r/趟数)。

## PE 利用率(mq,全 sweep 口径)

666 MHz:PE-bound 平台约 **0.71**;1.3 GHz:约 0.55。MAC 流内部的 PE 占空比在
PE-bound 平台上 ≈ 0.98(如 n=4 间隔 8 cycles、PE 需 7.81);其余为 WRGB/MVSB/SFM/
barrier 与换行时间——"100% util"成立于 MAC 流之内,sweep 级还含数据搬运段。

## 复现

```bash
# 单元测试(含 MQ trace 形状 / 间隔公式 / sweep 拆分):
python3 -m unittest tests.test_workload            # 30 tests
# 研究本体:
python3 experiments/mq_command/run_mq_study.py --workers 48
# 端到端(main.py 默认已是 --pim-batch-command mq):
python3 main.py --system dgx-attacc --model CACHEBLEND-TINY \
  --workload workload/workload_relay_s400w4t1.json --reuse epic \
  --epic-prefix-recompute-tokens 1 --cacheblend-batch-size 4 \
  --ramulator-workers 24 --pipeopt --workload-report /tmp/e2e_mq.json
```

## 扩展口径(2026-08-21 晚,宸逸裁决)

**context 侧才是最终倍数的墙。** 每 bank 驻留:score 相 1 条 Q = 64 B(与 L 无关);
context 相 1 条 P = L/8 B(L=4096 → 512 B)。同一块 buffer 按相位复用:
S=512 B → (n_q, n_c) = (8, 1);S=1 KB → (16, 2);S=2 KB → (32, 4)。
非对称 sweep:score 相 1 个运行(n_q 条 Q 驻留)+ ⌈n_q/n_c⌉ 个 context 相运行
(各 n_c 条 P 驻留),trace 生成器 `--phase score|context` 切片实现
(单 head 流,`test_phase_slices_partition_the_full_trace` 校验分片完备)。
K 读 ÷n_q、V 读 ÷n_c。**P 流式已否决(宸逸 2026-08-21):context 相 P 被
n_idx = d_head/(n_bank·n_mac) = 2 组输出复用,流式要么重发(上下行 TSV/GBUS
×2,且与流水中下一 head 的 MVSB 抢总线)要么行乒乓。**

**PE 频率**(score 相 n_q 免费所需):n=8 → 1.30 GHz;16 → 2.08 GHz;32 → 3.20 GHz。
但在非对称点上 context 重扫占 ~85%,PE 666 MHz→2.08 GHz 仅省 ~20%——杠杆在 n_c。

**面积(AttAcc §7.7)**:die 121 mm²,AttAcc 用 13.12 mm²(10.84%);GEMV unit
0.094 mm²(算术 63%/buffer 14%/控制 23%),accumulator 0.036 mm²,每 die ≈128+32 个。
buffer ×2 → die 12.2%;×4 → 15.0%(≈论文 FP32 变体的 14.59%);×8 → 20.6%。
均为外加 die 面积,不动存储阵列。

**两个 baseline(宸逸定)**:B1 compact = 共享一份、无 MQ 设计、单 Q 串行轮扫
(latency = N×t1,t1 = 6.22 µs @L=4096);B2 多 channel = chunk 复制 k 份并发
(latency = ⌈N/k⌉×t1,容量 ×k;其并发前提是存在空闲 channel——96 头对 16 通道
时通道已满,B2 退化为 B1 的时间)。

### 非对称点实测(run_asym_points.py,L=4096,PC;PE 假设全流水、频率可加)

| 点 | PE | interval(score/ctx) | score | context ×趟 | 合计 | 每 agent | vs B1 | 列读(÷) | ACT(÷) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| (16,2) S=1KB | 0.666 | 32/7 | 14.6 µs | 3.6 µs ×8 | 43.5 µs | 2.72 µs | 2.29× | ÷3.6 | ÷3.6 |
| (16,2) | 1.3 | 17/7 | 8.5 | 3.6 ×8 | 37.4 | 2.34 | **2.66×** | ÷3.6 | ÷3.6 |
| (16,2) | 2.08 | 11/7 | 6.5 | 3.6 ×8 | 35.4 | 2.21 | 2.81× | ÷3.6 | ÷3.6 |
| (32,4) S=2KB | 0.666 | 63/8 | 29.6 | 4.9 ×8 | 68.6 | 2.14 | 2.90× | ÷7.1 | ÷7.1 |
| (32,4) | 1.3 | 33/7 | 19.0 | 4.5 ×8 | 54.8 | 1.71 | **3.63×** | ÷7.1 | ÷7.1 |
| (32,4) | 2.08 | 21/7 | 14.8 | 4.5 ×8 | 50.6 | 1.58 | 3.93× | ÷7.1 | ÷7.1 |
| (32,4) | 3.2 | 14/7 | 12.3 | 4.5 ×8 | 48.1 | 1.50 | 4.13× | ÷7.1 | ÷7.1 |

(B1 = n_q×t1:(16,2) 对 99.5 µs,(32,4) 对 199 µs。实测比解析估计慢 ~25-40%:
WRGB/MVSB/SFM ×n_q 与每趟 MVGB 装载是真实开销,解析式只含 MAC 项。
B2 等延迟拷贝数:(16,2)@1.3 需 k=3,(32,4)@1.3 需 k=4、@3.2 需 k≈5——
且 B2 不降读能耗,MQ 列读/ACT 另降 3.6–7.1 倍。)
