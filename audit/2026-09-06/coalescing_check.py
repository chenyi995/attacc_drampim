#!/usr/bin/env python3
"""Counterfactual descriptor coalescing on saved addresses; no simulator changes."""
import hashlib, importlib.util, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
from w1_handcheck import markdown

def merge(extents):
    result=[]
    for k,v,n in sorted(extents):
        # All extents passed here belong to this query, head, and channel.
        # Require both K and V to be physically contiguous; do not skip gaps.
        if result and result[-1][0]+result[-1][2]*4==k and result[-1][1]+result[-1][2]*4==v:
            result[-1][2]+=n
        else:result.append([k,v,n])
    return result

def addresses(extents):
    return {(k+i*4,v+i*4) for k,v,n in extents for i in range(n)}

def main():
    source=HERE/'evidence/W1_turns_layout.json'
    data=json.loads(source.read_text())
    p=ROOT/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py'
    spec=importlib.util.spec_from_file_location('coalescing_tracegen',p)
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    def count(extents):
        if not extents:return {'qk':0,'pv':0}
        for name in ('cmd_score_wrgb','cmd_score_mac','cmd_score_mvsb','cmd_sfm',
                     'cmd_context_mvgb','cmd_context_mac','cmd_context_mvsb','valid_channels'):
            getattr(gen,name).clear()
        gen.Attention(sum(x[2] for x in extents),extents[0][0],extents[0][1],0,valid_channel=1,extents=extents)
        return {'qk':sum(map(len,gen.cmd_score_mac[0])),
                'pv':sum(map(len,gen.cmd_context_mac[0]))}
    evidence={'scope':'Static what-if: coalesce adjacent physical descriptors before real command generation; no timing or code fix',
              'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'trace_generator_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'requests':{}}
    rows=[]
    for rid in ('g00_m_t023','g01_m_t023'):
        record={}
        for rung in ('A4c','A4e'):
            before=[];after=[];channel_records=[]
            for ch,_n,full in data['variants'][rung][rid]['extents']:
                if ch>=8:continue
                ext=[e for e in full if e[0]-ch*(1<<30)>=1<<22]
                if not ext:continue
                combined=merge(ext)
                assert addresses(ext)==addresses(combined)
                assert sum(x[2] for x in ext)==sum(x[2] for x in combined)
                channel_records.append({'channel':ch,'before':ext,'after':combined,
                                        'before_commands':count(ext),'after_commands':count(combined)})
                before.extend(ext);after.extend(combined)
            v={'tokens':sum(x[2] for x in before),'before_extents':len(before),'after_extents':len(after),
               'before_qk':sum(x['before_commands']['qk'] for x in channel_records),
               'after_qk':sum(x['after_commands']['qk'] for x in channel_records),
               'before_pv':sum(x['before_commands']['pv'] for x in channel_records),
               'after_pv':sum(x['after_commands']['pv'] for x in channel_records),
               'after_run_lengths':[e[2] for e in after],'channels':channel_records}
            record[rung]=v
            rows.append([rid,rung,v['tokens'],v['before_extents'],v['after_extents'],
                         v['before_qk'],v['after_qk'],v['before_pv'],v['after_pv']])
        evidence['requests'][rid]=record
    (HERE/'evidence/coalescing_check.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (HERE/'COALESCING_CLARIFICATION.md').write_text('''# 更正：连续 diff 能合并，当前实现漏掉了这部分收益

chenyi9 指出：A4c 已经把 diff 连续存放，同轮能合并，多轮物理连续的部分也能合并。这一设计判断成立。先前审计中的额外列请求是当前实现的现象，不能表述为 A4c 的固有代价或“仍待决定是否允许合并”。

`PhysicalLedger.extent_groups`（`src/workload_runner.py:1099–1147`）只在对象内部合并，末尾明确保留相邻对象之间的边界。`score_mac` 随后对每个 extent 分别向上取整；PV 同样按 extent 循环。Ramulator 能让同一行的后续访问命中行缓冲，却不会替上游删除已经生成的重复列命令。因此“交给 Ramulator 自动合并”不能替代连续扫描段的合并。

这里保持上轮保存的 W1 物理地址不变，只在同一查询、head、channel 内把 K/V 地址都连续的 diff 描述符连接，再调用真实生成器计数。所有 token 的地址集合与数量逐项相同，不跨越未读空洞，没有运行 Ramulator，也没有修改实现。

'''+markdown(['请求','档位','diff token','原扫描段数','合并后段数','原QK命令','合并后QK命令','原PV命令','合并后PV命令'],rows)+'''
在 W1 的 A4c 中，多 agent 的 diff 仍混在全局流里；同一 main 的同轮修正相邻，而其轮间往往夹着其他 agent 的修正，因此本例不能把这个 main 的所有轮次硬连成一个连续段。若多轮实际相邻且本次都读取，则同样能合并。A4e 按 agent 聚集后，g00 的跨轮修正形成两个分属不同通道的长段，g01 则能连接成同一通道内跨行的一个长段；合并不能受 request/round/fingerprint 的对象标签限制。

执行建议是共同的物理扫描规则：对满足同一扫描语义、K/V 连续的段按真实地址合并；A3b 也适用同一规则。对象身份、逻辑位置、mask、query/旋转选择等元数据仍应保留，不能把合并读命令误做成取消逻辑边界。不同通道或不连续的地址继续分别描述。

本项定性：**本仓库多对象扫描描述符未合并，低估了已声明的 diff 连续布局收益，应交执行 agent 修复并重算**。dense trace 的取整来自原 AttAcc；禁止跨对象合并是新增路径的限制，不能把相邻档受影响的部分一概当成双方共同近似。修复后的时延及 TBT 仍要由新 trace/Ramulator 验证，本静态计数不代替性能结果。

复核命令：`python3 audit/2026-09-06/coalescing_check.py`。输入是已保存结构 JSON；逐通道前后描述符、命令计数和哈希见 [evidence/coalescing_check.json](evidence/coalescing_check.json)。
''')
    print(markdown(['请求','档','tokens','原段','合并段','原QK','新QK','原PV','新PV'],rows))

if __name__=='__main__':main()
