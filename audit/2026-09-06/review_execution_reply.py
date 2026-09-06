#!/usr/bin/env python3
"""Read-only acceptance check for the coalescing fix; preserve earlier evidence."""
import hashlib, importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'workload/probe/targeted')]
import handcheck as h
import src.workload_runner as r
from src.workload import load_workload,build_reuse_plan

def merged(groups):
    result=[]
    for ch,n,ext in groups:
        out=[]
        for k,v,c in sorted(ext):
            if out and out[-1][0]+out[-1][2]*4==k and out[-1][1]+out[-1][2]*4==v:
                out[-1][2]+=c
            else:out.append([k,v,c])
        result.append([ch,n,out])
    return result

def main():
    out=HERE/'execution_reply_review';out.mkdir(exist_ok=True)
    before=json.loads((HERE/'evidence/W1_turns_layout.json').read_text())
    wl=load_workload(str(ROOT/'workload/probe/sweep/W1_turns.json'))
    plan=build_reuse_plan(wl,'recompute',epic_prefix_recompute_tokens=8)
    target_ids=set(before['variants']['A3b'])
    targets=[q for q in wl.requests if q.request_id in target_ids]
    source=(ROOT/'src/workload_runner.py').read_bytes()
    evidence={'runner_sha256':hashlib.sha256(source).hexdigest(),'scope':'real W1 selected requests, layer0/head0; no performance simulation',
              'checks':{},'last_tier':{}}
    (out/'workload_runner.py.txt').write_bytes(source)
    genpath=ROOT/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py'
    spec=importlib.util.spec_from_file_location('reply_tracegen',genpath)
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    def qk(ext):
        for name in ('cmd_score_wrgb','cmd_score_mac','cmd_score_mvsb','cmd_sfm','cmd_context_mvgb','cmd_context_mac','cmd_context_mvsb','valid_channels'):
            getattr(gen,name).clear()
        gen.Attention(sum(x[2] for x in ext),ext[0][0],ext[0][1],0,valid_channel=1,extents=ext)
        return [int(s.split()[1],0) for row in gen.cmd_score_mac[0] for s in row]
    for rung in ('A3b','A4c','A4e'):
        current=h.describe(r,wl,plan,rung,targets)
        evidence['checks'][rung]={}
        for rid,v in current.items():
            old=before['variants'][rung][rid]
            actual=json.loads(json.dumps(v['extents']))
            assert actual==merged(old['extents']),(rung,rid)
            for field in ('prompt_tokens','diff_tokens','inherited_diff_tokens','new_diff_tokens','physical_scan_tokens','compute_tokens','resident_tokens'):
                assert v[field]==old[field],(rung,rid,field)
            evidence['checks'][rung][rid]={'exactly_coalesces_old_addresses':True,'same_work':True}
            if v['tier']!=23:continue
            item={'diff_QK':sum(x['qk_mac_requests'] for x in v['diff'].values()),
                  'full_QK':sum(x['qk_mac_requests'] for x in v['full'].values()),
                  'full_max_QK':max(x['qk_mac_requests'] for x in v['full'].values()),'lanes':[]}
            for ch,_n,ext in v['extents']:
                if ch>=8:continue
                commands=qk(ext)
                nominal={row for k,_val,n in ext for row in range(k//1024,(k+n*4-1)//1024+1)}
                emitted={a//1024 for a in commands}
                excess=emitted-nominal
                examples=[]
                for k,val,n in ext:
                    nominal_extent=set(range(k//1024,(k+n*4-1)//1024+1))
                    overflow=[a for a in qk([(k,val,n)]) if a//1024 not in nominal_extent]
                    if overflow:examples.append({'key':k,'value':val,'tokens':n,'extra_QK_addresses':overflow})
                item['lanes'].append({'channel':ch,'nominal_K_rows':len(nominal),'QK_rows':len(emitted),
                                      'extra_rows_count':len(excess),'extra_rows':sorted(excess),'examples':examples})
            evidence['last_tier'].setdefault(rung,{})[rid]=item
        (out/(rung+'_current.json')).write_text(json.dumps(current,indent=2)+'\n')
    # Dense AttAcc-style aligned inputs: the same length rounding alone
    # does not visit a wholly extra row when starting at a row boundary.
    evidence['aligned_dense_controls']=[]
    for n in (3,8,16,248,255,256,257,352):
        emitted={a//1024 for a in qk([(0,1<<24,n)])}
        nominal=set(range((n*4+1023)//1024))
        assert emitted==nominal
        evidence['aligned_dense_controls'].append({'tokens':n,'no_extra_entire_row':True})
    assert source==(ROOT/'src/workload_runner.py').read_bytes()
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    from w1_handcheck import markdown
    table=[]
    for rung,values in evidence['last_tier'].items():
        v=values['g01_w1_t023']
        table.append([rung,v['diff_QK'],v['full_max_QK'],sum(x['extra_rows_count'] for x in v['lanes'])])
    covered=sum(len(v) for v in evidence['checks'].values())
    (HERE/'EXECUTION_REPLY_REVIEW.md').write_text('''# 对执行回复的复核：修复通过，保留近似的理由需说准确

核对 `README.md` 第8节执行回复，以及 `074cbeb` 后的未提交修复。只审计，没有修改实现或论文，没有运行性能仿真。

## 连续段合并：通过本轮验收

当前 `PhysicalLedger.extent_groups` 对所有 policy 使用同一合并条件：同通道内，K 和 V 的下一段地址都紧接上一段末尾。对象索引保留，合并只作用于物理扫描描述符。不同通道、不连续地址仍分别描述。

''' + f'本轮用真实 W1 复用计划重建扫描，对 {covered} 个“档位×请求”检查：输出恰好等于旧地址描述符按共同规则合并，逻辑读集、重算量、新旧diff和物理token读量均不变。结果保存在 [execution_reply_review/evidence.json](execution_reply_review/evidence.json)。末轮两位main在各档的diff QK命令已一致；A4e的worker还利用了跨轮连续段。\n\n' + '''
另执行已有 `tests.test_placement.PhysicalLedgerTest`：全部12个测试通过，包括新增的连续修正合并测试。没有复跑执行agent所称的全量139个测试，也不把其结果称为本轮重新验证。

## “尾列越过 extent 不改”：保留可以，但不是误差已消失

执行回复的事实依据有一部分成立：长度取整确实继承自 AttAcc，chenyi9 也已经接受不实现短输出挤偏后继diff的半列case。可以在既定模拟范围内保留该近似，不需要因此自动增加另一套模型。

但这两个机制需要区分：

- 先前接受省略的case是：普通短输出占用半列，改变下一份diff的起始偏移。
- 当前残留现象是：一个已经存储的短diff处于行末，生成器按长度取整后，发出落到名义读集以外下一行的列命令。当前列起点与dense输入不同。

共用一段AttAcc取整代码，不代表各布局都受到相同影响。修复连续段合并后，W1末轮 `g01_w1_t023`、layer0/head0的实际生成器检查如下：

''' + markdown(['档位','diff QK列请求','完整读集最忙QK列请求','QK额外触及的K行数'],table) + '''
“额外K行”指QK命令覆盖到、但不在本次token物理读集内的整行，按通道求和；不是实测ACT数量或延迟。相同取整规则用于行对齐dense输入的控制没有额外整行。因此这不是可直接宣布对比中抵消的共同误差。

就这个现象而言，它向A4c增加扫描工作，会压低A4c的表现、也可能使A4e对A4c的改善看起来更大；实际时延影响尚未量化，不能推出整体speedup一定高估或低估。原 `PARTIAL_COLUMN_APPEND_AUDIT.md` 第3/4节也明确区分“基础取整有来源”和“新增对象场景已正确覆盖”。

建议将这条回复写成：**“保留AttAcc基础取整作为当前模型近似；它在部分diff布局中仍会产生额外尾列/行访问，尚未量化相邻档的时延影响。本轮不修改，不把这项标成存储与扫描严格一致或误差完全抵消。”** 这保留不改的范围决定，也保留审计事实。

## 其他不改实现的回复

- **A4e累计评分：有道理，可以保留。** 给被多轮反复读取的master更高权重是合理的启发式；当前docs与论文已经写出“master累计读取次数 + 已分配diff行数”。不必为了通过audit改成当前请求最少行，只需不称其为精确的总扫描负载最优。master与diff权重并不相同，这也已能从明示公式看出。
- **MQ不新增子组调度：有道理。** 指南已撤回W1跨会话decode MQ收益解释，明确来自prefill/GQA；这解决了归因问题。提取共读子组是另一个实现决定，不为让数字好看而自动增加。W1现有工作量不会因文案更新就获得这部分收益。
- **不改结构探针：有条件合理。** 它已声明手工标定，且执行不调用它；可以作粗估线索，不能当作实际PIM选边或延迟证据。无需把独立分析脚本改成执行模型。
- **保留r−2汇总：合理。** 已声明它是有延迟、未排空最后两轮的固定窗口实验，就不再把它描述为完整消费所有worker输出的最终汇总。

## 文档仍不是全部同步完成

论文方法/评估中的W1矩阵与主要几何已更新，图的TODO仍明确保留。还有几处表述需要避免被当成完成认证：第6节仍写EPIC k8，而当前W1脚本采用recompute随机k；第4节“same rule as a master chunk”容易混淆master的共读邻居计数与diff的新累计评分；第7节不能把只跑A3b/A6的全部变体都描述成已与A1/A2比较。它们是文本范围问题，本轮未改论文。

本轮复核脚本：`python3 audit/2026-09-06/review_execution_reply.py`。新证据独立保存在 `execution_reply_review/`，没有覆盖此前修复前的原始结构或执行agent的第8节回复。
''')
    for rung,values in evidence['last_tier'].items():
        print(rung,{rid:{'diff_QK':v['diff_QK'],'full_max_QK':v['full_max_QK'],
                         'extra_QK_rows':sum(x['extra_rows_count'] for x in v['lanes'])} for rid,v in values.items()},flush=True)
    print('All selected W1 scans match uniform physical coalescing, with unchanged logical work.')

if __name__=='__main__':main()
