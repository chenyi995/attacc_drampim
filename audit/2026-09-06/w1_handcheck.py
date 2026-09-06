#!/usr/bin/env python3
"""Audit calculations only: real plans/ledgers and GPU budgets, no DAG/Ramulator."""
import csv, hashlib, importlib.util, json, math, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'workload/probe/targeted')]
import handcheck as h
from src.workload import load_workload, build_reuse_plan
import src.workload_runner as runner

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def maximum(v, field):
    return max((x[field] for x in v.values()), default=0)

def total(v, field):
    return sum(x[field] for x in v.values())

def markdown(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + ''.join('| ' + ' | '.join(map(str,r)) + ' |\n' for r in rows)

def main():
    out = HERE / 'evidence'
    out.mkdir(parents=True, exist_ok=True)
    paths = ['src/workload_runner.py','src/workload.py','src/model.py','src/devices.py','src/config.py',
             'src/ramulator_wrapper.py','pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py',
             'workload/probe/gen_main_workers.py','workload/probe/targeted/handcheck.py',
             'experiments/run_sweep.sh','docs/README_contributions.md','docs/README_run_protocol.md',
             'workload/probe/README.md','output/analysis/b1_levers.py']
    provenance = {'scope':'static, layer 0 head 0 (8 channels); no new simulated latency',
                  'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'sources':{}}
    for rel in paths:
        raw=(ROOT/rel).read_bytes(); provenance['sources'][rel]=digest(raw)
        dst=out/'source'/(rel+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
    paper=ROOT.parent/'KVPIM-1Fugue-ASPLOS2027'
    provenance['paper_git']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=paper,text=True).strip()
    for p in [paper/'sections'/x for x in ('04-design.tex','05-execution.tex','06-methodology.tex','07-evaluation.tex')]:
        raw=p.read_bytes();provenance['sources'][str(p)]=digest(raw);(out/(p.name+'.txt')).write_bytes(raw)
    results={}; checks={}; summary=[]; workload_rows=[]; prices=[]
    for path in sorted((ROOT/'workload/probe/sweep').glob('W1*.json')):
        raw=path.read_bytes();meta=json.loads(raw)['meta'];wl=load_workload(str(path))
        (out/path.name).write_bytes(raw)
        plan=build_reuse_plan(wl,'recompute',epic_prefix_recompute_tokens=8)
        last=max(r.tier for r in wl.requests)
        targets=[r for r in wl.requests if r.tier==last or r.request_id=='a0_corpus' or
                 (r.request_id.startswith('g00_m_') and r.tier in (0,1,2,11))]
        case={'input_sha256':digest(raw),'meta':meta,'request_count':len(wl.requests),
              'variants':{rung:h.describe(runner,wl,plan,rung,targets) for rung in ('A3b','A4c','A4e')}}
        a=case['variants']['A3b']; c=case['variants']['A4c']; e=case['variants']['A4e']
        for rid,v in a.items():
            for rung in (c,e):
                for field in ('prompt_tokens','diff_tokens','inherited_diff_tokens','new_diff_tokens',
                              'physical_scan_tokens','compute_tokens','resident_tokens'):
                    assert v[field]==rung[rid][field],(path.name,rid,field)
            assert v['master']==c[rid]['master'],(path.name,rid,'A3b/A4c master differs')
        main_id='g00_m_t%03d'%last; worker_id='g00_w0_t%03d'%last
        r=last; w=meta['workers']; expected_diff=max(0,r-1)*w*8
        expected_main=16+(r+1)*16+r*meta['main_lout']+max(0,r-1)*w*meta['worker_lout']
        expected_worker=16+(r+1)*(meta['doc_tokens']+meta['note_tokens'])+r*meta['worker_lout']
        assert a[main_id]['prompt_tokens']==expected_main
        assert a[main_id]['diff_tokens']==expected_diff
        assert a[main_id]['new_diff_tokens']==w*8
        assert a[main_id]['compute_tokens']==16+w*8
        assert a[worker_id]['prompt_tokens']==expected_worker
        # A document at its owner's exact offset needs no repair. In the
        # 256-token worker-output sweep, worker 0 hits that equality once.
        equal_offset_rounds=[j for j in range(r+1) if
                             16+j*(meta['doc_tokens']+meta['note_tokens']+meta['worker_lout'])
                             ==256+j*w*meta['doc_tokens']]
        assert a[worker_id]['diff_tokens']==((r+1)-len(equal_offset_rounds))*8
        assert a[worker_id]['compute_tokens']==meta['note_tokens']+8
        checks[path.stem]={'equal_work_and_A3b_A4c_masters':True,'closed_form_main_worker':True,
                          'worker0_zero_offset_rounds':equal_offset_rounds}
        for rung,values in case['variants'].items():
            v=values[main_id]
            summary.append([path.stem,rung,v['prompt_tokens'],v['diff_tokens'],
                            total(v['diff'],'physical_rows'),maximum(v['diff'],'physical_rows'),
                            total(v['diff'],'qk_mac_requests'),maximum(v['full'],'qk_mac_requests'),
                            maximum(v['full'],'physical_rows'),maximum(v['full'],'token_rows')])
        workload_rows.append([path.stem,len(wl.requests),expected_main,expected_worker,expected_diff,
                              ','.join(str(maximum(case['variants'][rg][main_id]['full'],'qk_mac_requests')) for rg in ('A3b','A4c','A4e'))])
        if path.stem=='W1_turns':
            case['prefill_budgets']={rid:h.prefill_budget(v['compute_tokens'],v['prompt_tokens'],'LLAMA3-8B',runner._link_layer)
                                      for rid,v in a.items()}
            for rid in ('a0_corpus','g00_m_t000','g00_m_t002',main_id,worker_id):
                p=case['prefill_budgets'][rid]
                prices.append([rid,p['compute'],p['logical_context'],p['sweeps'],
                               f"{p['gpu_price_us_per_layer']:.3f}",f"{p['context_return_us_per_layer']:.3f}",
                               f"{p['pim_average_sweep_must_be_below_us']:.3f}"])
            # Actual grouping rule intersects the ENTIRE <=8 member decode batch.
            # With 6 requests per steady tier the identity/order cannot change it.
            tlb=runner.TableLocalDiffKVLayout(256,'slice')
            runner._prepare_cacheblend_tlb(wl,plan,1,tlb,runner._parent_output_fingerprints(wl))
            sets={}
            for req in wl.requests:
                if req.tier!=last:continue
                b=runner._cacheblend_tlb_rows(wl,plan,0,req,tlb)
                reads=runner._pool_reads(tlb,[v[3] for v in b])[0]
                sets[req.request_id]={(v.key_address,v.value_address) for v in reads if v.kind=='master'}
                tlb.entries.clear()
            case['mq']={'last_tier_batch_size':len(sets),'all_member_common_master_tokens':len(set.intersection(*sets.values())),
                        'same_worker_cross_session_common_tokens':len(sets['g00_w0_t023']&sets['g01_w0_t023']),
                        'main_cross_session_common_tokens':len(sets['g00_m_t023']&sets['g01_m_t023'])}
            assert case['mq']['all_member_common_master_tokens']==0
        (out/(path.stem+'_layout.json')).write_text(json.dumps(case,indent=2)+'\n')
        results[path.stem]=case
        print(path.stem, 'main prompt/diff',expected_main,expected_diff,'max QK',workload_rows[-1][-1],flush=True)
    # Independent address formula is checked against the real generator.
    spec=importlib.util.spec_from_file_location('w1_tracegen',ROOT/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py')
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    verified=0
    for values in results['W1_turns']['variants'].values():
        for rid,v in values.items():
            for channel,_n,extents in v['extents']:
                if channel>=8:continue
                for k in ('cmd_score_wrgb','cmd_score_mac','cmd_score_mvsb','cmd_sfm','cmd_context_mvgb','cmd_context_mac','cmd_context_mvsb','valid_channels'):
                    getattr(gen,k).clear()
                gen.Attention(sum(x[2] for x in extents),extents[0][0],extents[0][1],0,valid_channel=1,extents=extents)
                actual=[int(s.split()[1],0) for row in gen.cmd_score_mac[0] for s in row]
                expect=[k+32*i for k,_v,n in extents for i in range(2*math.ceil(n/16))]
                assert actual==expect,(rid,channel)
                verified+=1
    checks['actual_QK_generator_lists_verified']=verified
    checks['source_unchanged_during_calculation']=all(digest((ROOT/p).read_bytes())==provenance['sources'][p] for p in paths)
    assert checks['source_unchanged_during_calculation']
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (out/'gpu_budgets.json').write_text(json.dumps(h.gpu_budgets(),indent=2)+'\n')
    with (out/'layout_summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['workload','rung','main_context','diff_tokens','diff_K_rows_sum','diff_K_rows_max',
                                             'diff_QK_commands_sum','full_QK_commands_max','full_K_rows_max','full_token_load_max']);writer.writerows(summary)
    (HERE/'CALCULATIONS.md').write_text('# W1 当前布局的静态计算\n\n由 `w1_handcheck.py` 生成；输入、源码及hash在 `evidence/`。每项取最后一轮 g00 main、layer 0、head 0 八通道。不是 Ramulator 延迟，不把行/列数当作 TBT。\n\n'+
        markdown(['输入','请求数','main context','worker context','main有效diff token','完整读集最忙通道QK列请求 A3b,A4c,A4e'],workload_rows)+
        '\n详细计数：K物理占行与QK命令分开；full包含master、shadow和diff。\n\n'+
        markdown(['输入','档','context','diff token','diff K行总和','diff最忙K行','diff QK命令总和','full最忙QK命令','full最忙K行','full最忙token'],summary)+
        '\n## Prefill 收支平衡预算\n\nLLAMA3-8B、A100a、FlashAttention、NVLink、GQA4，单层；capacity=2；PIM每sweep时间必须来自Ramulator。最后一列仅为 `(GPU价格−context返回)/sweeps`。\n\n'+
        markdown(['请求','m','N','sweeps','GPU µs','context返回 µs','PIM允许平均sweep µs'],prices)+
        '\n## Decode MQ\n\n```json\n'+json.dumps(results['W1_turns']['mq'],indent=2)+'\n```\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
