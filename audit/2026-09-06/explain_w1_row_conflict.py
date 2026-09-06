#!/usr/bin/env python3
"""Extract actual co-read worker-answer placements for explaining W1 conflicts."""
import hashlib,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.workload import load_workload,build_reuse_plan
import src.workload_runner as r
from w1_handcheck import markdown

def main():
    wl=load_workload(str(ROOT/'workload/probe/sweep/W1_turns.json'))
    plan=build_reuse_plan(wl,'recompute',epic_prefix_recompute_tokens=8)
    fps=r._parent_output_fingerprints(wl)
    consumer=next(x for x in wl.requests if x.request_id=='g00_m_t023')
    wanted={fps['g00_w0_t%03d'%i]:'g00_w0_t%03d'%i for i in range(6)}
    wanted.update({fps[x]:x for x in ('g00_m_t000','g00_w1_t001')})
    rows=[];result={}
    for rung,cls in [('A3b',r.NaiveKVLayout),('A4c',r.LocalDiffKVLayout),('A4e',r.TableLocalDiffKVLayout)]:
        tlb=cls(256,'slice');r._prepare_cacheblend_tlb(wl,plan,1,tlb,fps)
        ledger=tlb.physical_ledger(tlb.layout_policy,2)
        order={}
        for (layer,owner,fp,kind),reserved in tlb._reserved_rows.items():
            if kind=='diff':continue
            for block in sorted({x//256 for x in reserved}):
                key=(fp,block)
                if key not in order:order[key]=len(order)
        bindings=r._cacheblend_tlb_rows(wl,plan,0,consumer,tlb)
        reads=r._pool_reads(tlb,[x[3] for x in bindings])[0]
        records=[];hist=[0]*8
        load_by_kind={'worker_answer_tokens':[0]*8,'own_history_and_instructions_tokens':[0]*8,'diff_tokens':[0]*8}
        worker_fps={fps['g00_w%d_t%03d'%(w,t)] for w in range(2) for t in range(22)}
        buckets={k:[] for k in load_by_kind}
        for loc in reads:
            bucket=('diff_tokens' if loc.kind=='diff' else 'worker_answer_tokens' if loc.fingerprint in worker_fps
                    else 'own_history_and_instructions_tokens')
            buckets[bucket].append(loc)
        for name,locations in buckets.items():
            for ch,_n,ext in ledger.extent_groups(locations):
                if ch<8:load_by_kind[name][ch]+=sum(n for _k,_v,n in ext)
        for fp in worker_fps:
            selected=[x for x in reads if x.kind=='master' and x.fingerprint==fp]
            assert len(selected)==128
            for ch,_n,ext in ledger.extent_groups(selected):
                if ch>=8:continue
                hist[ch]+=len({row for k,_v,n in ext for row in range(k//1024,(k+n*4-1)//1024+1)})
        for fp,producer in wanted.items():
            selected=[x for x in reads if x.kind=='master' and x.fingerprint==fp]
            groups=ledger.extent_groups(selected)
            assert len(selected)==128
            for ch,_n,ext in groups:
                if ch>=8:continue
                assert len(ext)==1
                k,v,n=ext[0]
                rec={'producer':producer,'master_write_index':order[(fp,0)],'channel':ch,
                     'row':(k-ch*r._HBM_CHANNEL_BYTES)//1024,'column':k%1024//32,'tokens':n,
                     'key':k,'value':v}
                records.append(rec);rows.append([rung,producer,rec['master_write_index'],ch,rec['row'],n])
        result[rung]={'consumer':consumer.request_id,'answers':records,'all_44_answer_K_rows_per_channel':hist,
                      'load_by_kind':load_by_kind}
    data={'scope':'static, W1 final main, layer0 head0 eight channels; no Ramulator timing',
          'runner_sha256':hashlib.sha256((ROOT/'src/workload_runner.py').read_bytes()).hexdigest(),'results':result}
    (HERE/'W1_ROW_CONFLICT.json').write_text(json.dumps(data,indent=2)+'\n')
    tables=markdown(['档','回答来源','master预约序号','channel','K row','token'],rows)
    tables+=markdown(['档','44份回答在ch0…ch7的K行数'],[[a,','.join(map(str,v['all_44_answer_K_rows_per_channel']))] for a,v in result.items()])
    tables+=markdown(['档','读集部分','ch0…ch7的token读量'],[[a,k,','.join(map(str,x))] for a,v in result.items() for k,x in v['load_by_kind'].items()])
    (HERE/'W1_ROW_CONFLICT.md').write_text('''# W1 为什么存在共读行冲突

当前每个KV head使用八个channel的几何下，A3b/A4c的master按全局master预约顺序轮转，`channel = index % 8`。汇总者只读取自己的worker回答和自身历史，是全局写入序列的一个子集；全局轮转均衡不保证这个子集也均衡。

W1稳定轮次每轮新增的master对象排列重复。同一worker相邻回答的master预约序号相差12，对八通道取模后，每次只移动四个channel，因此反复落在两个channel。隔一轮的两份回答落到同一个channel、不同DRAM行；持续汇总者在后续attention中同时需要它们。

以下从真实账本抽取，不是手动指定地址。预约序号不是实际GPU完成时刻；行号为channel内K侧地址。所有列出的回答都在g00 main末轮的同一物理读集中。

'''+tables+'''
这些块横跨同一channel的相同一组banks，bank的行缓冲不能同时打开两个不同行，因此要先后读这些块；Ramulator处理行关闭/打开及其时序。其他channel可能同时无事可做。因此这里包含“同一channel上的行切换”和“共读负载集中造成串行”。不同row本身很正常，损失在于可分散的共读被集中到少数channel。A4e也不能消除所有行切换，只能用共读信息改变固定放置、降低热点。

A4c只改变diff区域，master仍沿用A3b分布，所以这些回答master的通道集中继续存在。A4e放置表才对这类master共读起作用。相同channel、相同column号但不同row的对象仍是不同存储地址，并非地址别名。

实际账本还说明：A4e没有把这44份worker回答单独均分到八个channel；它们依然占四个channel。完整读集还包含main自己的历次输出和指令，朴素布局把这些历史叠到回答已多的channel上。表的收益必须按完整读集解释：把这些负载重新分配，使不同部分更少重叠在相同热点；不能拿本例声称“44份worker回答由四通道变成八通道”。上表保留这个子集反例。

这里没有证明反复在同两行间抖动；实际扫描按物理地址排序。也没有从行数推算固定跳转罚时或TBT倍数，时延由实际trace交给Ramulator计算。

复核：`python3 audit/2026-09-06/explain_w1_row_conflict.py`；原始地址与分布见 `W1_ROW_CONFLICT.json`。
''')
    print(tables)

if __name__=='__main__':main()
