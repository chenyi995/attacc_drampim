#!/usr/bin/env python3
"""Render the audit from saved calculations; copy independent evidence verbatim."""
import hashlib, json, math, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from w1_handcheck import markdown, maximum, total

def main():
    out=HERE/'evidence'; independent=out/'independent';independent.mkdir(exist_ok=True)
    copies={}
    for pattern in ('w1_address_audit_c78dc76*','w1_decode127_c78dc76*','w1_piece_boundary_c78dc76*',
                    'w1_model_provenance_c78dc76*','w1_performance_audit.md','w1_design_fairness_audit*',
                    'w1_independent_static_c78dc76*'):
        for p in sorted(Path('/tmp').glob(pattern)):
            if not p.is_file():continue
            raw=p.read_bytes();(independent/p.name).write_bytes(raw)
            copies[str(p)]=hashlib.sha256(raw).hexdigest()
    (out/'independent_manifest.json').write_text(json.dumps(copies,indent=2)+'\n')
    d=json.loads((out/'W1_turns_layout.json').read_text())
    checks=json.loads((out/'checks.json').read_text())
    rows=[];last=[];derived={}
    for rung,values in d['variants'].items():
        for rid in ('g00_m_t023','g01_m_t023'):
            v=values[rid]
            rows.append([rid.split('_')[0],rung,total(v['diff'],'physical_rows'),
                         total(v['diff'],'qk_mac_requests'),total(v['full'],'qk_mac_requests'),
                         maximum(v['full'],'physical_rows'),maximum(v['full'],'qk_mac_requests')])
        final={rid:v for rid,v in values.items() if v['tier']==23}
        load=[sum(v['full'].get(str(c),{}).get('qk_mac_requests',0) for v in final.values()) for c in range(8)]
        derived[rung]={'last_tier_six_query_QK_load_per_channel':load,'max':max(load),'sum':sum(load)}
    for rid in sorted(d['variants']['A3b']):
        if d['variants']['A3b'][rid]['tier']!=23:continue
        last.append([rid]+[maximum(d['variants'][r][rid]['full'],'qk_mac_requests') for r in ('A3b','A4c','A4e')])
    a,c,e=[d['variants'][r]['g00_m_t023'] for r in ('A3b','A4c','A4e')]
    min_diff=math.ceil(a['diff_tokens']/256)
    row_gain=100*(1-total(c['diff'],'physical_rows')/total(a['diff'],'physical_rows'))
    possible_gain=100*(1-min_diff/total(a['diff'],'physical_rows'))
    realized=100*(total(a['diff'],'physical_rows')-total(c['diff'],'physical_rows'))/(total(a['diff'],'physical_rows')-min_diff)
    cq=maximum(c['full'],'qk_mac_requests');eq=maximum(e['full'],'qk_mac_requests')
    qmin=math.ceil(total(c['full'],'qk_mac_requests')/8)
    qgain=100*(1-eq/cq);qrealized=100*(cq-eq)/(cq-qmin)
    derived['bounds']={'diff_K_row_min':min_diff,'A3b_A4c_diff_row_reduction_pct':row_gain,
                      'A3b_ideal_diff_row_reduction_pct':possible_gain,'A4c_fraction_of_diff_row_headroom_pct':realized,
                      'A4c_A4e_busiest_QK_reduction_pct':qgain,'ideal_busiest_QK_for_fixed_commands':qmin,
                      'fraction_of_QK_balance_headroom_pct':qrealized}
    batch_gain=100*(1-derived['A4e']['max']/derived['A4c']['max'])
    derived['bounds']['six_query_busiest_total_QK_reduction_pct']=batch_gain
    # The GPU model is analytic, matching the runtime's common flash setup.
    import handcheck as h
    cfg=h.make_xpu_config(h.GPUType.A100a,num_gpu=1,gpu_model='flash')['GPU']
    gpu=h.xPU(h.DeviceType.GPU,cfg,h.SCALING_FACTOR)
    model=h.Transformer(h.make_model_config('LLAMA3-8B',h.DataType.W16A16),1)
    model.build(6,1,2,attn_on_hetero=True)
    layers=model.gen_decoder[0] if isinstance(model.gen_decoder[0],list) else model.gen_decoder
    G=sum(gpu.get_time_and_energy(x)[0] for x in layers if x.type in (h.LayerType.FC,h.LayerType.ACT,h.LayerType.NORM))*1e6
    derived['illustrative_TBT_budget']={'batch':6,'GPU_work_us_per_layer':G,
        'scan_gain_15pct_TBT_gain_5pct_required_S_us':G*.05/(.15-.05),
        'meaning':'Serial T=G+S illustration; G is all GPU work, not measured exposed critical path.'}
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance=json.loads((out/'provenance.json').read_text())
    impl=['src/workload_runner.py','src/workload.py','src/devices.py','src/config.py','src/model.py',
          'src/ramulator_wrapper.py','pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py','workload/probe/gen_main_workers.py']
    derived['closing_revision']={'git':current,'implementation_still_matches_snapshot':all(
        hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==provenance['sources'][p] for p in impl)}
    assert derived['closing_revision']['implementation_still_matches_snapshot']
    for rel in ('docs/README_run_protocol.md','workload/probe/README.md'):
        p=out/'closing_source'/(rel+'.txt');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    (out/'derived.json').write_text(json.dumps(derived,indent=2)+'\n')
    text=f'''# W1 与旋转 diff 布局复审：结论、计算和待核事项

**chenyi9 指正后的更正：[连续 diff 扫描段合并](COALESCING_CLARIFICATION.md)。A4c 同轮、跨轮实际连续的 diff 都能合并；下文“额外列请求”是当前实现遗漏，不是布局设计的固有代价。已定性为低估方法收益的实现问题，未修改模拟器。**

**W1 可以作为“两个固定团队持续阅读并汇总”的合成机制实验；修正确实跨轮继承，A4e 有明显的通道均衡空间。但目前不能确认每一级都能得到预期的性能差异：A4c 的最忙通道没有稳定改善，W1 的两两共读没有进入当前整批 decode MQ。**

本轮只审计并计算，未修改模拟器、论文或 workload，也未启动性能仿真。主审与三个独立 agent 分别核对设计、物理地址和性能证据。下文的行数、列请求数是**静态结果**；GPU 时间是**当前设备模型的解析价格**；没有把它们当作新的 Ramulator/TBT 实测。

## 1. 查了哪个版本，哪些结论可以确认

计算时实现为 `{provenance['git'][:7]}`，结束时 HEAD 为 `{current[:7]}`；中间提交只更新文档/报表标题，本轮涉及的实现与输入哈希相同。设计改动来自 `753044b`，论文为 `{provenance['paper_git'][:7]}`。源码、输入和版本指纹在 [provenance](evidence/provenance.json)，结束核对在 [derived](evidence/derived.json)。

- 论文第 4、7 节明确：A4c 把同一 head 的 diff 紧凑存放、按行轮换通道；A4e 再按 agent 分组并用表选通道。当前实现包含这两步，新增分组已经是 claim 的一部分。
- 全部 13 份 W1 输入的选定请求中，A3b/A4c/A4e 的逻辑 KV、重算量、新旧 diff 和物理 token 读量一致；A3b/A4c 的 master 放置一致。没有在这次修改中发现人为削弱 A3b 的额外计算项。
- main 同轮两份 diff 在 A3b 正常合并，符合 chenyi9 的要求。跨轮旧 diff 仍引用原 writer，没有重新造 master 占位。
- {checks['actual_QK_generator_lists_verified']} 组列地址与真实 trace generator 逐条一致。此项证明审计数字忠实于生成器；它不代表所有列请求都严格落在 token 的存储边界内，见第 6 节。
- 运行入口仍指定 FlashAttention、pipeopt、NVLink、统一功耗限制。PIM 执行仍由各通道真实 Ramulator 结果计价；A6 对每个 sweep 取 **max(channel time)**。没有发现本次改成平均通道计时。

原 AttAcc 的共同模型近似、用户指定的链路价和小 decode 传输固定开销处理继续沿用；没有据此另立整改要求。

## 2. W1 为什么合理，又代表哪种情形

每个团队有两个 worker 和一个持续存在的 main，共两个团队。worker 每轮读一篇新文档，并保留完整上下文；main 每轮把先前 worker 的回答接到自己的上下文末尾。每次加入新回答才产生新修正，下一轮继续读取原回答的 master 与此前的 diff。这与持续编写报告、汇总多轮检索的结构相符，不能因为输入是构造的就判为不合理。

这份输入是**固定编排、完整上下文保留、没有裁剪、两个团队读取同一语料**的机制实验，不能直接声称代表一般线上 agent 的分布。已知整张 DAG 的共读表是之前接受的模型假设；随机在线工具结果未知时，不能直接套用同样的信息条件。

还有一个需要写清的时间语义：main 在第 r 轮读的是 worker 第 r−2 轮的回答，不是当前轮结果。这样构造是为了让 planner 先通过 worker 的 parent_out 确认回答来源，避免 main 被误认成回答的写者。它可以解释为延迟两轮的流水汇总，但前两轮 main 只有指令和自身历史，最后两轮 worker 回答也没有被 main 收尾消费。若文章想描述“最终报告覆盖所有 worker 输出”，当前输入还不符合这个说法。

W1 第 23 轮的手算（轮次从 0 开始）：

```text
main context = 16 + 24×16 + 23×128 + 22×2×128 = {a['prompt_tokens']} token
main diff    = 22×2×8 = {a['diff_tokens']} token
其中旧 diff = {a['inherited_diff_tokens']}，本轮新 diff = {a['new_diff_tokens']}
main 本轮重算 = 16 token 指令 + 2×8 修正 = {a['compute_tokens']}
worker context = 16 + 24×(256+16) + 23×128 = 9488 token
worker 本轮重算 = 16+8 = 24
```

main 的修正对应不同轮次的不同回答，共 44 个回答块；不是把同一个 chunk 的失效修正反复叠加。两位 main 分别保留自己的修正链。

## 3. 实际布局拿到了什么

下表为 W1 最后一轮、开始 decode 前、一个 head 的八个通道。K 物理行数是存储占行；QK 列请求是实际生成器发出的工作量，二者都不是延迟。full 含全部 master、被覆盖而仍扫描的 shadow 和有效 diff。

'''+markdown(['main','档位','diff K行总数','diff QK列请求总数','full QK列请求总数','full最忙通道K行','full最忙通道QK列请求'],rows)+f'''
**A4c：当前实现没有充分兑现连续布局的收益。** diff 从 22 行降到 6 行，减少 {row_gain:.2f}%。A3b 把同轮两份 8-token diff 连成 16-token burst；A4c/A4e 虽然也已将它们连续存放，扫描端却按两个对象分别取整，使 main 的 diff 列请求从 44 增到 88，full 从 1166 增到 1210。这是本仓库扫描描述符未合并的实现遗漏，不能解释成 A4c 本来就需要更多列命令。合并同一扫描所需的物理连续段后，原来的额外取整可以消除；跨轮同样成立。下表计数仍保留为修复前的真实代码行为，不能据此否定布局设计。完整scan/TBT需在修复后重新计价。

**A4e：当前最明确的是通道均衡。** g00 main 的最忙通道从 {cq} 个 QK 列请求降到 {eq}，下降 {qgain:.2f}%；g01 也约减半。独立核查加入本轮已生成的 127 个 token 后，这两个最忙通道计数保持不变，因此结论不只是 decode 第一步的现象。

不能把两个 main 代替整个 batch。最后一轮六个请求分别如下，worker 1 原本较均衡，A4e 在它身上反而增加了最忙通道负载：

'''+markdown(['请求','A3b最忙QK列请求','A4c最忙QK列请求','A4e最忙QK列请求'],last)+'''
各请求的通道服务还会共享资源。为了便于检查，这里也保存同一层六个查询的 QK 工作量逐通道相加后的结果；它只反映共享通道工作量，没有 QK/PV、softmax、链路和 GPU 的完整时序：

'''+markdown(['档','六请求QK列请求按ch0…ch7相加','max'],[[r,','.join(map(str,x['last_tier_six_query_QK_load_per_channel'])),x['max']] for r,x in derived.items() if r in ('A3b','A4c','A4e')])+f'''
整个batch最忙通道的累计QK工作量，A4c→A4e只降低 {batch_gain:.2f}%。这进一步说明：两个汇总者各自约减半的列负载，不能直接代表整批scan，更不能代表TBT。
### 仅对布局计数成立的上限

- 352 个 diff token 最少占 `ceil(352/256)={min_diff}` 个 K 行。相对 A3b 的 22 行，最多减少 {possible_gain:.2f}%；A4c 的 6 行拿到了可省行数的 {realized:.2f}%，A4e 的 2 行达到这个**占行数下界**。
- 固定 A4c/A4e 的 1210 个 QK 列请求，八通道任意均分的乐观下界为 `ceil(1210/8)={qmin}` 个/通道。A4e 的 {eq} 个相当于拿到从 {cq} 到 {qmin} 这段均衡空间的 {qrealized:.2f}%。这只是忽略不可拆对象/其他请求的工作量下界。
- 以上都不能换算成 scan 加速倍数，更不能换算成 TBT。真实时延还受 PV、命令时序和共享资源等待影响。

## 4. 为什么仍不能承诺 TBT，怎样读 sweep

假设仅作解释的串行模型 `T=G+S`，scan 降幅为 g，则 TBT 降幅为 `g×S/(G+S)`。scan 降 15% 而想得到 5% 的 TBT，scan 必须占原 TBT 的三分之一。当前 LLAMA3-8B、实际 batch=6 的 GPU 线性层/norm/activation 解析工作量约 {G:.3f} µs/层；用全部 GPU 工作作量级参考，需要约 {G*.5:.3f} µs/层的 scan 才满足这个例子。head 流水会重叠一部分 GPU 工作，所以这个 G 不是实测关键路径，也不是 W1 的 TBT 预测。

当前 base 每轮只有六个持续 decode 的请求，单把 BATCH 上限从 8 改到 32 不会凭空增加并发。文档的新 BATCH=32 例子仍必须结合 S3 会话数理解。

全部 sweep 的 [详细计算](CALCULATIONS.md) 保留有利和不利的点：

- S1 多轮：diff 和上下文都增长，可以增加布局工作在长上下文中的权重；但 A4c 的最忙列负载依然不保证降低。不能把多累计几轮等同于所有档单调变快。
- S2 一个 worker：去掉 main 每轮两份 diff 的双重取整差别，但 master 热点仍在，单靠这一点也不能保证 A4c 更快。
- S3 多会话：增加实际就绪请求；它也改变写入相位、通道争用和组批，必须看完整 scan/TBT，不能只加 batch 上限。
- S6 长文档：直接拉长 worker 的驻留 KV；main 消费的是固定长度回答，其 context 和 diff 不随文档长度增加。不能将 worker 的长扫描当成 main diff 集中的收益。
- S5 长 main 输出：增加 decode 的份额，同时也增加自己的历史 master；历史增长可能稀释 diff 效果，并非输出越长布局越强。

因此 W1 能展示 A4e 的结构差异；A4c 的贡献应同时报告“diff占行、整个scan最慢lane、TBT”，保留它改善不足的结果，不能只报 diff 占行比例。当前只跑 A3b/A6 的 sweep 可以评价整体方案，但不能用来证明某个中间档的独立收益。

## 5. A4e→A5→A6 的合理预期

**decode 的跨会话 MQ 当前没有按指南预期发生。** batch8 会把每轮六个请求放入一个组；代码要求全组的 master 交集非空才构造共同扫描。两对 worker 各自共读的文档，main 和另一对 worker 并不读。主审核到末轮每对 worker 共读 {d['mq']['same_worker_cross_session_common_tokens']} token，但六人交集为 {d['mq']['all_member_common_master_tokens']}。独立 agent 还核过全部 tier/output-step，整组交集始终为空。

这会低估当前输入可以提供的跨会话复用，但现有代码确实不能得到这部分收益。TINY/MHA 的 private decode 只有一个 query；LLAMA3 的 GQA 仍能在一个 KV head 内复用多个 Q，这要单独归因。Prefill 的多个 query 也仍能使用 MQ，所以不能据此说 A5 没有收益。

**Prefill 的形状合理：后期是几十个 query 对几千 token 的缓存。** [价格表](CALCULATIONS.md#prefill-收支平衡预算) 使用当前 FlashAttention 与链路函数计算 GPU 侧，并算出 PIM 的收支平衡条件。LLAMA3-8B 的每 KV head 有 GQA4，512 B buffer 使一次最多放两个 request-query；不能像旧探针一样一律除以八。

- 末轮 main：16 次 sweep，平均每次实际 Ramulator 时间低于 {d['prefill_budgets']['g00_m_t023']['pim_average_sweep_must_be_below_us']:.3f} µs，PIM 服务价才优于 GPU。
- 末轮 worker：12 次 sweep，门槛 {d['prefill_budgets']['g00_w0_t023']['pim_average_sweep_must_be_below_us']:.3f} µs。
- corpus 导入含前缀，共 {d['prefill_budgets']['a0_corpus']['compute']} token，需要 {d['prefill_budgets']['a0_corpus']['sweeps']} 次 sweep，门槛仅 {d['prefill_budgets']['a0_corpus']['pim_average_sweep_must_be_below_us']:.3f} µs。它与后期低-query请求形成了合理的选边对照。
- 首轮 main 也是 fresh 小请求，其门槛只有 {d['prefill_budgets']['g00_m_t000']['pim_average_sweep_must_be_below_us']:.3f} µs；不能写成“除了 corpus，其他每轮都一定选 PIM”。

这些是**PIM允许预算，不是PIM拟合值或选边结果**。因此可以预计存在让 A5/A6 分开的输入形状，不能保证 A5 一定优于 A4e、或 A6 的逐请求服务选择一定使 E2E 最优。

当前 W1 尚无完整新结果：独立取证时新任务只有 A3b/A4e，未产出完整报告；旧 W1 任务和目录已删除。旧 M workload 的结果只能作为历史反例，不能当作当前 W1。详见 [独立性能复核](evidence/independent/w1_performance_audit.md)。

## 6. 需要过目的事项：先说 AttAcc，再说本项目影响

| 事项 | AttAcc 本身建模了吗 | 当前影响与审计判断 |
|---|---|---|
| W1 两两共读未被整批 MQ 利用 | 原 AttAcc 没有这里新增的多agent共享/MQ分组机制 | 当前 README 的归因不成立，压低新增方法的 decode 收益。需修正文档预期；是否让实现提取共读子组由 chenyi9 决定，本轮未改。 |
| A4e“最少读行”评分不是当前读集的行数 | 原 AttAcc 没有该放置表 | 实现按所有轮次累计master读取次数，新diff行却只加1；独立W1反例中，所选通道并非当前请求读行最少。它是已知DAG上的一种启发式，未证明偏袒或性能方向。需明确评分定义或使实现与文字一致。 |
| 连续 packed diff 的扫描段未合并 | dense trace 的取整来自 AttAcc；新增多对象路径明确禁止相邻extent合并 | chenyi9 已指出：同轮和跨轮实际连续的diff均应合并。已确认实现低估A4c/e收益，应统一按物理连续性构造扫描段并重算；不是设计固有代价，也不是待决定是否允许合并。本轮只audit。 |
| 新布局中的尾列请求仍越过 extent | AttAcc 原有粒度近似，但其原始dense输入没有这里的分散diff对象 | 独立例：g00 第17轮某8token diff存于ch4,row4096,col31，第二列请求到ch4,row4097,col0；真正下一段在ch1。token绑定无漏无重，但不能宣称列命令与存储边界严格一一对应。这是已讨论部分列近似的具体落点，未把它升级为必须实现半列case。 |
| 论文各节与图未同步 | 项目文档问题，不是AttAcc模型问题 | 第4/7节已更新；第6节仍写A4c单通道/per-agent、A4e只改master，且旧14配置/几何/EPIC说明未对齐当前W1协议。布局图仍有dedicated diff channel的TODO。不能确认“整篇论文与实现一致”。 |
| 结构探针的计数与选边估计被当成性能证据 | 实际AttAcc计时来自Ramulator；该probe是本仓库分析脚本 | `b1_levers.py` 的结构读取可参考，但其PIM/选边时间仍是手工标定，忽略实际extent、GQA容量等。执行模型不调用它。其所有turn占行总和也不是当前decode最慢lane延迟。 |

论文待同步的具体位置：`sections/06-methodology.tex` 的 comparison points、workloads、models、policies 段；`sections/07-evaluation.tex` 的 fourteen configurations / all configurations full ladder；`fig/ver1/fig4channelsplit.pdf` 对应第4节图注附近的 TODO（本轮未重新做PDF视觉检查）。第4节的无条件 j mod s 规则也应明确为 A4c/default rotation，A4e则由表覆盖。第6节写一GPU一HBM，而当前默认是一GPU五HBM；这直接决定每head可用通道数，尤其不能带着旧几何解释本轮八通道数字。

A4e评分的具体落点、当前最少行向量与AttAcc来源对照见 [独立设计复核第3节](evidence/independent/w1_design_fairness_audit.md#3-已复现需限定措辞a4e-分数不是当前扫描的最少物理读行数)。本轮没有把已接受的全DAG信息重新判作不公平，也没有据此替换启发式。W1的parent链严格对应同一agent，按root分组成立；其他输入若把parent用于跨agent生产者依赖，必须另行明确身份语义。

## 7. 复现与证据

在仓库根目录执行 `python3 audit/2026-09-06/w1_handcheck.py`，重建13输入的计划/ledger计算与真实生成器地址核对；不会调用Ramulator或DAG运行。随后 `python3 audit/2026-09-06/finalize_report.py` 从已保存结果生成本报告。重算会更新这个审计目录的证据，比较历史版本时先保留目录副本。

- [计算表](CALCULATIONS.md)、[逐项断言](evidence/checks.json)、[原始结构CSV](evidence/layout_summary.csv)、[派生计算](evidence/derived.json)。
- `evidence/W1*_layout.json` 保存请求身份、新旧diff、各通道的extents/行/列计数；`evidence/source/` 保存计算所用源码。
- [独立地址证据](evidence/independent/w1_address_audit_c78dc76_evidence.json)、[跨行边界](evidence/independent/w1_piece_boundary_c78dc76_evidence.json)、[decode后段](evidence/independent/w1_decode127_c78dc76_evidence.json)。对应脚本原样保存在同目录，部分输出路径为其原始 `/tmp` 路径。
- [独立性能证据](evidence/independent/w1_performance_audit.md) 与 [提取脚本](evidence/independent/w1_model_provenance_c78dc76.py)。
- [独立设计证据](evidence/independent/w1_design_fairness_audit.md) 与 [评分核验脚本](evidence/independent/w1_independent_static_c78dc76.py)。

计算中首次闭式断言发现 worker_lout=256 的 sweep 有一次文档恰与 owner 同offset，不需要修正；已按实际偏移修正审计公式并记录 `worker0_zero_offset_rounds`，没有修改输入来强凑每轮八个修正。
'''
    (HERE/'README.md').write_text(text)
    print(json.dumps(derived,indent=2))

if __name__=='__main__':main()
