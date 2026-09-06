"""Read existing artifacts and W1 input; no device/trace pricing or simulation."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib,json,subprocess
ROOT=Path('/data2/chenyi9/KV-PIM/attacc_drampim_822')
SCR=Path('/data2/chenyi9/KV-PIM/scratch_0905')
OUT=Path('/tmp/w1_model_provenance_c78dc76.json')
sources={}
def read(p):
 raw=p.read_bytes();sources[str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()};return raw.decode()
wlpath=ROOT/'workload/probe/sweep/W1_turns.json'
w=json.loads(read(wlpath));tiers=defaultdict(list)
for a in w['agents']:tiers[a['tier']].append(a)
intersection=[]
for tier,aa in sorted(tiers.items()):
 sets={a['id']:{s['sha'] for s in a['segs']} for a in aa}
 workers=[a for a in aa if '_w' in a['id']]
 pairs=[]
 for wi in range(w['meta']['workers']):
  pair=[a for a in workers if '_w%d_'%wi in a['id']]
  shared=set.intersection(*(sets[a['id']] for a in pair))
  lengths={s['sha']:s['len'] for a in pair for s in a['segs']}
  pairs.append({'worker':wi,'members':[a['id'] for a in pair],
    'common_fingerprints':len(shared),'common_doc_rows':sum(lengths[s] for s in shared)})
 # Sorted-by-readiness order cannot alter membership: every active group fits
 # within batch8. Each generated output is request-private, adding no new
 # fingerprint common to the complete set.
 classes=Counter()
 for step in range(max(a['lout'] for a in aa)):
  active=[a for a in aa if step<a['lout']]
  assert len(active)<=8
  common=set.intersection(*(sets[a['id']] for a in active))
  classes[(len(active),len(common))]+=1
 intersection.append({'tier':tier,'requests':len(aa),'members':[a['id'] for a in aa],
   'full_tier_common_fingerprints':len(set.intersection(*sets.values())),
   'active_group_step_classes':[{'members':n,'common_fingerprints':c,'output_steps':count} for (n,c),count in sorted(classes.items())],
   'pair_only_sharing':pairs})
assert all(g['common_fingerprints']==0 for t in intersection for g in t['active_group_step_classes'])
mbase=SCR/'proto_m_CACHEBLEND-TINY/M_main_workers_r16_w4_s1'
mr={r:json.loads(read(mbase/f'dag_{r}.json')) for r in ['A3b','A4c','A4e','A5','A6']}
# Input was removed with the previous workload tree. Recover output counts
# from retained batch membership/output indices, then cross-check total.
louts={}
for b in mr['A4e']['batches']:
 for rid in b['members']:louts[rid]=max(louts.get(rid,0),b['output_row']+1)
assert sum(louts.values())==mr['A4e']['workload']['total_output_tokens']
mrows={}
for rung,d in mr.items():
 recs=d['summary']['requests']
 eligible=[(rid,r) for rid,r in recs.items() if louts[rid]>1]
 tbt=sum(r['end_s']-r['first_token_s'] for rid,r in eligible)/sum(louts[rid]-1 for rid,r in eligible)*1e6
 ss=d['summary']['decode_scans']
 mrows[rung]={'run_config':d['run_config'],'e2e_s':d['makespan_s'],'tbt_weighted_us':tbt,
   'ttft_mean_ms':sum(r['ttft_s'] for r in recs.values())/len(recs)*1000,
   'scan_private_service_us':ss['private_service']['mean_us'],
   'scan_shared_service_us':ss['shared_service']['mean_us'],
   'scan_step_elapsed_us':ss['per_step_elapsed']['mean_us'],
   'scan_shared_service_count':ss['shared_service']['count'],
   'batch_size_counts':dict(Counter(len(b['members']) for b in d['batches'])),
   'shared_sweep_member_counts':dict(Counter(len(s['members']) for b in d['batches'] for s in b.get('sweeps',[]))),
   'prefill_rows':d['prefill_attention_rows'],
   'side_counts':dict(Counter((d.get('pim_prefill_sides') or {}).values())),
   'decode_scan_energy_nj':{k:v for k,v in d['energy_breakdown_nj']['by_event'].items() if 'decode' in k and 'scan' in k},
   'corrected_rows_sha':d['corrected_rows_sha'],'overlap_validation':d['overlap_validation']}
sides=[json.loads(line) for line in read(mbase/'sides.jsonl').splitlines() if line.strip()]
assert len({r['request'] for r in sides})==len(sides)==len(louts)
assert all(r['side']==('pim' if r['t_bank_s']<=r['t_xpu_s'] else 'gpu') for r in sides)
select=[r for r in sides if r['side']=='gpu']+[r for r in sides if r['request'] in ['g00_w0_t000','g00_m_t001','g00_m_t015']]
for r in select:r['P_minus_G_us']=(r['t_bank_s']-r['t_xpu_s'])*1e6
w1log=read(SCR/'w1.log');new_sweep=read(SCR/'proto_w1new_CACHEBLEND-TINY/sweep.log')
new_ladder=read(SCR/'proto_w1new_CACHEBLEND-TINY/W1_turns.log')
read(SCR/'run_w1_new.sh');read(SCR/'run_w1.sh')
for name in ['src/workload_runner.py','src/ramulator_wrapper.py','src/devices.py','main.py','workload/probe/gen_main_workers.py']:
 read(ROOT/name)
payload={'scope':'Existing-results extraction and pure workload intersection analysis only; no Ramulator, GPU timing calls, or performance runs.',
 'audit_time_utc':datetime.now(timezone.utc).isoformat(),
 'audit_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'current_W1':{'path':str(wlpath),'metadata':w['meta'],'requests':len(w['agents']),
   'lout_histogram':dict(Counter(a['lout'] for a in w['agents'])),
   'old_W1_directory_exists':(SCR/'proto_w1_CACHEBLEND-TINY').exists(),
   'new_W1_reports':[str(p) for p in (SCR/'proto_w1new_CACHEBLEND-TINY').rglob('dag_*.json')],
   'control_log':w1log,'new_sweep_log':new_sweep,'new_ladder_log':new_ladder,
   'source_scope':'Old W1 deleted per control log. New run explicitly A3b/A4e only; started before commit753044b but run script identifies rotating-layout source. No completed report/manifest, so exact source fingerprints and validation outcomes cannot yet be checked.'},
 'W1_MQ_intersection':{'tiers':intersection,
   'all_active_groups_have_zero_full_common_fingerprints':True,
   'tier_output_step_count':sum(g['output_steps'] for t in intersection for g in t['active_group_step_classes']),
   'expected_shared_multi_request_decode_scans':0,
   'evidence_type':'Static input proof, not an executed scan count: absent common fingerprint across all members precludes common master rows, assuming no unintended physical address alias. Batch8 always contains the entire <=7/6 active set, independent of readiness sort.',
   'implementation':'workload_runner.py:3979-3985 groups sorted active by batch_size,4037-4046 intersects physical master keys across ALL members,4052 requires nonempty common. There is no subgroup clustering for pairwise co-reads.',
   'TINY_effect':'MHA gqa1 private scan has one resident query. ramulator_wrapper.py:571 enables MQ command only if shared_queries>1. Thus two-session pairwise sharing cannot accelerate W1 TINY decode via the current shared-group path.',
   'GQA_effect':'LLAMA3 private scan still has gqa4 resident Q heads (workload_runner.py:4118-4123), enabling within-KV-head MQ despite zero cross-request common intersection. This is different from sharing across sessions.',
   'prefill_effect':'Prefill batches multiple compute positions of ONE request into sweeps and may use MQ independently of decode batch intersections. A5 can still change prefill time; that outcome requires actual lane pricing.'},
 'old_M_control':{'path':str(mbase),'requests':len(louts),'lout_recovered_from_batches':louts,
   'rows':mrows,'side_log_counts':dict(Counter(r['side'] for r in sides)),'side_log_all_chooser_inequalities_verified':True,
   'selected_side_prices':select,
   'A5_minus_A6_e2e_ms':(mrows['A5']['e2e_s']-mrows['A6']['e2e_s'])*1000,
   'A4e_to_A5_e2e_reduction_pct':100*(1-mrows['A5']['e2e_s']/mrows['A4e']['e2e_s']),
   'A5_to_A6_e2e_reduction_pct':100*(1-mrows['A6']['e2e_s']/mrows['A5']['e2e_s']),
   'source_limit':'M is r16/w4/s1, 81 requests, old last-channel diff. Reports pin958dd24 with git_dirty=true and cannot certify the exact dirty tree. It predates the W1 r24/w2/s2 and per-agent/rotating diff design.',
   'shared_metric_warning':'All4096 retained shared-sweep records have ONE member, after worker outputs finish. shared_service label alone is not proof of multi-query or multi-request reuse. A4e/A5/A6 decode service and decode scan energy are equal.',
   'attribution':'79/81 requests select PIM; fresh corpus and first main select GPU. A5/A6 differ in these prefill choices; same MQ/layout/decode mechanism. Exact E2E delta is not simply a sum of local price deltas because scheduling overlaps resources.'},
 'configuration_checks':{'old_M_flash':'run_config.gpu_model=flash on each report; devices.py:57-59 selects flash implementation.',
    'old_M_pipe':'run_config.pipeopt=true and saved overlap_validation passed. Current scheduler backfills intervals and head-pipelines decode from ff6f225, already preceding958dd24.',
    'max_channel':'Per-lane Ramulator results become independent PIM:pool events; chooser_sweep_price returns max at workload_runner.py:4399. decode_scans summary explicitly distinguishes max lane service from elapsed.',
    'new_W1':'new_sweep log confirmsTINY/ngpu1/hbm5/flash/k8/batch8; no completed report yet. 8KVheads/5HBM uses busiest2heads =>8channels/head. Actual config/source still needs finalreport.'},
 'sources':sources}
read(ROOT/'output/analysis/b1_levers.py')
payload['analytic_probe_limit']={
 'source':'output/analysis/b1_levers.py:139-143',
 'bank':'ceil(m/8) * 4.05us * ceil((m+R)/STRIPE)/1536',
 'gpu':'100us*(m*(m+R))/(816*2864) + (6.06us + R*4096/335GBps if R>0)',
 'basis':'Manual TINY calibration from an earlier small case, explicitly documented in lines18-25.',
 'limitations':'GQA is absent from m/8 (LLAMA3 capacity is2); lane=ceil(n/stripe) is idealized rather than actual layout max-time; it excludes actual shadow/extents and current zero-fixed-charge small-link rule. Scalar GPU scaling is not current FlashAttention occupancy/tile pricing.',
 'runtime_independence':'Actual runner _resolve_prefill_side calls GPU.get_time_and_energy and real get_time_and_energy_runs (max lane). It does not call this probe; the presence of fitted numbers here does not mean execution PIM timing is fitted.'}
OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
report=[
 '# W1 独立性能证据复核', '',
 '只读检查当前源码、既有产物与纯 workload 交集；没有启动 Ramulator、GPU 定价或性能任务。',
 '源码 HEAD：`'+payload['audit_head']+'`；证据时间：'+payload['audit_time_utc']+'。', '',
 '## 当前 W1 没有完整性能结果', '',
 '输入为 `'+str(wlpath)+'`，SHA256 `'+sources[str(wlpath)]['sha256']+'`。',
 '当前配置：'+json.dumps(w['meta'],ensure_ascii=False)+'；'+str(len(w['agents']))+' 个请求。',
 '`scratch_0905/w1.log` 记录旧 W1 于 00:38:05 被停止并删除；`proto_w1_CACHEBLEND-TINY` 当前不存在。',
 '新目录 `scratch_0905/proto_w1new_CACHEBLEND-TINY` 于 00:38:10 只启动 A3b/A4e；本次检查完整 dag JSON 数量为 '+str(len(payload['current_W1']['new_W1_reports']))+'。没有当前 W1 A5/A6 结果。',
 'session §26:577 所述旧 W1 七档仍在跑、作为改前对照已经过时。新跑早于提交753044b，但启动脚本指明旋转diff实现；最终还需对照run_config，不能仅凭目录名钉住源码。', '',
 '## W1 跨会话共读没有进入当前整批 MQ 路径', '',
 '`src/workload_runner.py:3979-3985` 将整个active集合按Q-ready排序后每batch8个切组；本W1每轮最多7个（含只生成1 token的owner），其余6个，始终只有一个整组。',
 '`src/workload_runner.py:4037-4046` 取整组所有请求的物理master交集；4052仅非空时建shared scan，没有按共读子集再拆组。',
 '纯输入核查：'+str(len(intersection))+' 个tier、'+str(payload['W1_MQ_intersection']['tier_output_step_count'])+' 个tier/output-step的整个active组，原始fingerprint交集全部为空。新生成的输出按request私有，不会增加整组共读。这里是静态证明，不是实跑的scan计数；假定不存在非预期地址别名。',
 '同号worker跨两个会话确实共读：首轮每对'+str(intersection[0]['pair_only_sharing'][0]['common_doc_rows'])+'行，末轮每对'+str(intersection[-1]['pair_only_sharing'][0]['common_doc_rows'])+'行，但两个main与另号worker不共享这些集合。',
 'TINY为GQA1：private scan只有1 query，而`src/ramulator_wrapper.py:571`要求shared_queries>1才启用MQ命令。因此“两个会话给decode MQ材料”的预期与当前整批交集实现不符。',
 'LLAMA3的private scan仍有GQA4个Q heads共享同一KV head（runner:4118-4123），可以使用head内部MQ；这不是跨会话MQ。Prefill会把同一request的多个计算位置组成sweep，仍可用MQ，不能由decode交集为空推断A5毫无收益。', '',
 '## 留存的 M 小跑是旧输入、旧 diff 布局', '',
 '`'+str(mbase)+'`；r16/w4/s1，'+str(len(louts))+'请求。报告的`run_config`为958dd24、git_dirty=true，不足以复原全部未提交源码。它不是r24/w2/s2的W1，也不是新旋转/按agent分组diff性能。', '',
 '| 档位 | E2E s | TTFT mean ms | TBT weighted us | scan private us | scan shared us | scan step elapsed us | prefill PIM/GPU rows |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for rung,r in mrows.items():
 report.append('| '+rung+' | '+' | '.join(f'{r[k]:.9f}' for k in ['e2e_s','ttft_mean_ms','tbt_weighted_us','scan_private_service_us','scan_shared_service_us','scan_step_elapsed_us'])+' | '+str(r['prefill_rows']['pim'])+'/'+str(r['prefill_rows']['gpu'])+' |')
report += ['',
 '该M输入的A4e→A5 E2E降幅为 '+f"{payload['old_M_control']['A4e_to_A5_e2e_reduction_pct']:.6f}"+'%（负数表示变慢）；A5→A6为 '+f"{payload['old_M_control']['A5_to_A6_e2e_reduction_pct']:.6f}"+'%。',
 'A4e/A5/A6的decode service与decode scan能量相同；报告虽然有shared_service，但全部'+str(mrows['A4e']['scan_shared_service_count'])+'个shared sweep只有1个成员，是worker输出结束后main单独decode，被命名为shared，不是multi-query收益证据。', '',
 '## A5/A6 的局部价格与合理性', '',
 'M的side log共'+str(len(sides))+'个决策，'+str(Counter(r['side'] for r in sides)['pim'])+'个PIM、'+str(Counter(r['side'] for r in sides)['gpu'])+'个GPU；脚本逐条验证side与t_bank<=t_xpu一致。', '',
 '| 请求 | m | R | GPU price us | PIM price us | side |',
 '|---|---:|---:|---:|---:|---|']
for r in select:
 report.append('| '+r['request']+' | '+str(r['compute_rows'])+' | '+str(r['readback_rows'])+' | '+f"{r['t_xpu_s']*1e6:.6f}"+' | '+f"{r['t_bank_s']*1e6:.6f}"+' | '+r['side']+' |')
report += ['',
 '旧M的A6把大语料导入与首个main的fresh prefill放GPU，其余放PIM；A5全部PIM。因此二者改善来自prefill选边，不能归因decode MQ。局部服务价格不包含全部排队/重叠，不能将其差的简单求和当作E2E保证；旧M的选边更不能外推成当前W1或LLAMA3必然同侧。', '',
 '## Flash、流水与最慢通道', '',
 '旧M每档run_config显式记录gpu_model=flash、pipeopt=true、powerlimit=true、TINY、A100a、ngpu1、num_hbm5、batch8、NVLink3；overlap_validation通过。',
 '当前Flash分支见`src/devices.py:57-59`；decode head切分见runner:3065/3891，回填调度见runner:2601以后。ff6f225的回填/head流水已在958dd24前。',
 '实际PIM执行保留每channel真实Ramulator服务时间并发到独立PIM:pool；A6每个sweep的估价在runner:4399取所有lane实测time最大值，不是按token最多lane或求和。summary.decode_scans区分max-lane service与elapsed。',
 '新W1 sweep.log也明确flash、TINY、ngpu1、hbm5、k8、batch8；尚无最终报告可检查验证结果与源码指纹。', '',
 '## 手工形状估计不是执行模型', '',
 '`output/analysis/b1_levers.py:139-143`仍用ceil(m/8)、4.05us×lane/1536、100us×m*n/(816*2864)、旧6.06us/335GBps回读拟合。它忽略GQA的query容量、实际extent/最慢lane和当前GPU分块/occupancy，不是实际Ramulator选边。',
 '脚本自己在18-25行声明校准来源；协议/README引用它时只能当结构与粗估线索，不能用它保证当前W1的A5/A6性能或真实PIM请求数量。',
 '实际执行 `_resolve_prefill_side` 使用GPU模型与真实Ramulator每lane返回值，不调用这个probe；不能因此把实际PIM timing说成手工拟合。', '',
 '可复核提取脚本：`/tmp/w1_model_provenance_c78dc76.py`；数据与源文件hash：`/tmp/w1_model_provenance_c78dc76.json`。']
Path('/tmp/w1_performance_audit.md').write_text('\n'.join(report)+'\n')
print('W1 reports',payload['current_W1']['new_W1_reports'])
print('W1 whole-group common empty:',payload['W1_MQ_intersection']['all_active_groups_have_zero_full_common_fingerprints'])
print('W1 tier output steps',payload['W1_MQ_intersection']['tier_output_step_count'])
for r,m in mrows.items():print(r,{k:m[k] for k in ['e2e_s','tbt_weighted_us','ttft_mean_ms','scan_private_service_us','scan_shared_service_us','scan_step_elapsed_us','prefill_rows']})
print('M side selected',json.dumps(select))
print('M e2e reductions',payload['old_M_control']['A4e_to_A5_e2e_reduction_pct'],payload['old_M_control']['A5_to_A6_e2e_reduction_pct'])
