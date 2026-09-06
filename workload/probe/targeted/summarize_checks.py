#!/usr/bin/env python3
"""Verify closed-form workloads, actual QK address generation, and write tables."""
import argparse,csv,importlib.util,json,math,re,shutil,sys
from pathlib import Path
from handcheck import HERE,ROOT,sha,prefill_budget,runner_from_source


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)+'\n'


def maxval(lanes,key):return max((v[key] for v in lanes.values()),default=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checks',type=Path,default=HERE/'checks')
    a=ap.parse_args();out=a.checks
    data=json.loads((out/'all.json').read_text())
    # Recalculate prices after review with the SAME transfer helper as HEAD.
    m=runner_from_source((out/'head_workload_runner.py.txt').read_bytes(),'src.targeted_report_head')
    for name,case in data['inputs'].items():
        assert case['input_sha256']==sha((HERE/'inputs'/(name+'.json')).read_bytes())
        if name.startswith('P_'):
            case['prefill_budgets']={rid:{model:prefill_budget(v['compute_tokens'],v['prompt_tokens'],model,m._link_layer)
                                         for model in ('CACHEBLEND-TINY','LLAMA3-8B')}
                                     for rid,v in case['variants']['HEAD_partners']['A4e'].items()}
            if case['meta']['fresh']:
                case['fresh_budgets']={str(n):prefill_budget(n,n,'LLAMA3-8B',m._link_layer) for n in (2048,4096,8192)}
            (out/(name+'.json')).write_text(json.dumps(case,indent=2)+'\n')
    (out/'all.json').write_text(json.dumps(data,indent=2)+'\n')

    rows=[];checks={}
    for R in (16,32,64,128):
        case=data['inputs']['D_rolling_r%d'%R]['variants']['HEAD_partners']
        rid='g00_s_t%03d'%(R-1);base=case['A3b'][rid];packed=case['A4c'][rid]
        assert base['prompt_tokens']==400*R-112
        assert base['diff_tokens']==8*R and base['inherited_diff_tokens']==8*(R-1)
        assert base['compute_tokens']==24 and base['new_diff_tokens']==8
        assert set(base['diff'])=={'7'} and base['diff']['7']['physical_rows']==R
        assert packed['diff']['7']['physical_rows']==math.ceil(R/32)
        assert packed['diff']['7']['qk_address_rows']==R//32+1
        assert base['master']==packed['master']
        assert base['diff']['7']['qk_mac_requests']==packed['diff']['7']['qk_mac_requests']==2*R
        rows.append([R,base['prompt_tokens'],base['diff_tokens'],R,packed['diff']['7']['physical_rows'],packed['diff']['7']['qk_address_rows'],2*R])
        checks['rolling_formula_R%d'%R]=True
    d_table=table(['轮数R','prompt token','有效diff token','A3b K物理行','A4c K物理行','A4c QK触及行','两档QK列请求'],rows)
    rows=[]
    for p in (1,4,8):
        case=data['inputs']['E_coread_stride%d'%p]['variants']['HEAD_partners']
        rid='s00_t001'
        b=case['A4c'][rid]['document_master'];t=case['A4e'][rid]['document_master']
        assert sum(v['token_rows'] for v in b.values())==64*256
        assert sorted(v['physical_rows'] for v in b.values())==[64//(8//math.gcd(p,8))]*(8//math.gcd(p,8))
        assert [t[str(c)]['physical_rows'] for c in range(8)]==[8]*8
        assert sum(v['qk_mac_requests'] for v in b.values())==sum(v['qk_mac_requests'] for v in t.values())==64*32
        rows.append([p,','.join(b),maxval(b,'physical_rows'),maxval(t,'physical_rows'),maxval(b,'qk_mac_requests'),maxval(t,'qk_mac_requests')])
        checks['coread_formula_stride%d'%p]=True
    e_table=table(['周期','朴素活跃通道','A4c最忙K行','A4e最忙K行','A4c最忙QK列请求','A4e最忙QK列请求'],rows)
    # Check the transcription against the generator itself, without emitting
    # or executing a simulator trace. One final target from each rung/case.
    path=ROOT/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py'
    spec=importlib.util.spec_from_file_location('targeted_address_generator',path)
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    count=0
    for case in data['inputs'].values():
        for values in case['variants']['HEAD_partners'].values():
            rid=max(values,key=lambda r:(values[r]['tier'],r));v=values[rid]
            for channel,_n,extents in v['extents']:
                if channel>=8:continue
                for key in ('cmd_score_wrgb','cmd_score_mac','cmd_score_mvsb','cmd_sfm','cmd_context_mvgb','cmd_context_mac','cmd_context_mvsb','valid_channels'):
                    getattr(gen,key).clear()
                gen.Attention(sum(e[2] for e in extents),extents[0][0],extents[0][1],0,valid_channel=1,extents=extents)
                actual=[int(line.split()[1],0) for group in gen.cmd_score_mac[0] for line in group]
                expected=[k+i*32 for k,_v,n in extents for i in range(2*math.ceil(n/16))]
                assert actual==expected
                count+=1
    checks['real_generator_QK_address_lists_matched']=count
    (out/'formula_checks.json').write_text(json.dumps(checks,indent=2)+'\n')

    source=Path('/data2/chenyi9/KV-PIM/scratch_0905/proto_s11_CACHEBLEND-TINY/protocol.csv')
    saved=out/'existing_S11_protocol.csv'
    if source.is_file():saved.write_bytes(source.read_bytes())
    old={r['combo']:r for r in csv.DictReader(saved.open())}
    rows=[]
    for b,n in [('A3b','A4c'),('A4c','A4e'),('A3b','A4e'),('A4e','A6')]:
        rows.append([b+'→'+n]+[f"{100*(1-float(old[n][c])/float(old[b][c])):.3f}%" for c in ('scan_private_us','scan_shared_us','scan_step_us','tbt_weighted_us','e2e_s')])
    old_table=table(['比较','private scan下降','shared scan下降','scan_step下降','TBT下降','E2E下降'],rows)
    rows=[]
    for name in ('CACHEBLEND-TINY','LLAMA3-8B'):
        for b in (1,8,32):
            v=data['gpu_budgets'][name]['batches'][str(b)]
            rows.append([name,b,f"{v['gpu_work_us_per_layer']:.3f}",f"{v['required_exposed_scan_us_for_5pct_if_scan_gain_15pct']:.3f}",f"{v['required_exposed_scan_us_for_10pct_if_scan_gain_15pct']:.3f}"])
    gpu_table=table(['model','batch形状','GPU工作 µs/层','串行预算S：5% TBT','串行预算S：10% TBT'],rows)
    case=data['inputs']['P_reuse_q4'];prices=case['prefill_budgets'];rows=[]
    for rid in ('s00_t000','s00_t001','s00_t031'):
        v=prices[rid]['LLAMA3-8B']
        rows.append([rid,v['compute'],v['resident'],v['sweeps'],f"{v['gpu_price_us_per_layer']:.3f}",f"{v['context_return_us_per_layer']:.3f}",f"{v['pim_average_sweep_must_be_below_us']:.3f}"])
    p_table=table(['请求','m','R','sweeps','GPU价格 µs/层','context返回 µs/层','PIM每sweep须低于 µs'],rows)
    fresh=data['inputs']['P_mixed_q4_fresh']['fresh_budgets']
    f_table=table(['fresh prompt','m','sweeps','GPU µs/层','PIM每sweep须低于 µs'],[[n,v['compute'],v['sweeps'],f"{v['gpu_price_us_per_layer']:.3f}",f"{v['pim_average_sweep_must_be_below_us']:.3f}"] for n,v in fresh.items()])
    extra=[]
    for name in ('D_rolling_r64_background_plus1','D_rolling_r64_sessions7','D_rolling_r64_sessions8'):
        vals=data['inputs'][name]['variants']['HEAD_partners'];b=vals['A3b'];c=vals['A4c']
        ids=[r for r in b if b[r]['tier']==63]
        extra.append([name,
                      ','.join(str(maxval(b[r]['diff'],'qk_address_rows')) for r in ids),
                      ','.join(str(maxval(c[r]['diff'],'qk_address_rows')) for r in ids),
                      min(min(x['token_rows'] for x in b[r]['master'].values()) for r in ids),
                      max(maxval(b[r]['master'],'token_rows') for r in ids)])
    x_table=table(['构造','各summary A3b最忙diff QK行','各summary A4c最忙diff QK行','master最少lane token','master最忙lane token'],extra)
    (HERE/'HANDCALC.md').write_text('''# 手算、静态核验与时间预算

由 `summarize_checks.py` 从保存的输入/结构/旧报告生成。除第1节外，均为新输入的静态结果或解析预算，不是新性能实测。K侧地址行数不等于K/V合计激活数；无手工PIM拟合。

## 1. 最新commit记录的S11：实际改善的是什么

来源：`checks/existing_S11_protocol.csv`，原始运行配置为 TINY、flash、pipe、5 HBM、k8、batch8。各档记录相同输入hash和corrected_rows_sha，代码标记为 `167fe08` + dirty，不能把今天HEAD冒充该次运行版本。

'''+old_table+'''
S11共享简报的scan在A3b/A4c/A4e间相同，布局改的是较小的private部分。这里全体scan_step的降幅小于private scan降幅，TBT还包含其余工作。因此现有数据不是“整步扫描大幅下降却完全没传到TBT”的证明。A6数据还含MQ；本表没有A5，不能据此独立计算A5→A6。

## 2. D：一个持续汇总者，第R轮开始decode前

每档同一逻辑内容，所有有效旧diff继续引用原writer；普通master分布相同。以下为layer0、head0的K侧。

'''+d_table+'''
闭式公式已断言通过：prompt=400R−112，diff=8R，旧diff=8(R−1)，后续m=24。A3b每轮新增对象周期为8、diff一直在ch7；A4c物理占行ceil(R/32)，但每八token extent的两次QK列请求使末列extent还碰下一行。两档列请求数完全相同。

所有列请求地址另与真实trace generator直接输出的地址列表核对，通过项数见 `checks/formula_checks.json`；没有运行Ramulator。其他层、PV/归约、刷新和行缓冲实际状态仍需完整仿真。

## 3. 并发与写流的反例

下表都看最后一轮，按summary编号列出。

'''+x_table+'''
七条链的普通master较均衡；八条链的普通master出现热点。后者可能掩盖diff省时，尚未测到实际遮蔽。background_plus1使A3b的diff分散到八通道，原集中写流条件改变；不能假定A4c在反例也同样获益。

## 4. E：共同读取64个文档的master部分

每owner八个chunk；每chunk=256token；head0八通道；两档总QK列请求均为2048。

'''+e_table+'''
闭式朴素分布：活跃通道数=8/gcd(8,period)，最忙通道文档数=64/活跃通道数。真实HEAD表在三个输入均将文档均分到八个通道。这里的4倍/8倍是共读master的最大行/列负载比，不是完整scan或TBT加速。

整库owner反例：`E_coread_stride8_monolithic_owner` 的HEAD表仍使最忙通道承担64个文档；工作区reader-load表为8个。版本和实际地址分别保存，禁止混作一个实现的性能。

## 5. scan降15%需要什么量级才能传到TBT

示意串行模型T=G+S中，TBT降幅=g×S/(G+S)。达到5%/10%需要S≥0.5G/2G。下面G取当前GPU模型全部线性层、norm和activation工作量作量级参考；head流水能重叠部分工作，所以该G不是实测关键路径，不据此预测TBT。

'''+gpu_table+'''
这里的batch是实际矩阵行数，不是光改CLI上限。单汇总者的大部分输出步只有一个summary；七条链后段才有七个summary。增加并发的同时必须检查通道排队是否也变快。

## 6. P：当前GPU侧价格与PIM允许预算

LLAMA3-8B、GQA4、batch上限8、buffer512 B，capacity=min(8,8/4)=2；每层计价。数据采用HEAD `_link_layer`，小context返回没有附加固定链路latency。

'''+p_table+f_table+'''
PIM实际每个sweep必须由Ramulator取最慢channel。表中“须低于”是 `(GPU价格−context返回)/sweeps`，不是PIM测量值。大prefill需要的sweep多，可允许的平均sweep时间很低；m4的后续轮宽裕得多。

初次导入、首轮大修正必须计入完整E2E，后续32轮用来增加低-query复用的权重。局部选边价是服务成本而非含排队TTFT；不能把各请求差额相加当作精确E2E。
''')
    provenance={}
    for rel in ('src/workload.py','src/model.py','src/config.py','src/devices.py','src/ramulator_wrapper.py','pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py','workload/probe/targeted/build.py','workload/probe/targeted/handcheck.py','workload/probe/targeted/summarize_checks.py','workload/probe/targeted/cohort_report.py','workload/probe/targeted/validate_artifacts.py'):
        content=(ROOT/rel).read_bytes();provenance[rel]=sha(content)
        dest=out/'source'/(rel+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    (out/'source_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
