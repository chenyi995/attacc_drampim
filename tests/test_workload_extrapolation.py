import copy
import unittest
from types import SimpleNamespace

from src.ablation import resolve_config
from src.model import Transformer
from src.type import DataType, DeviceType, LayerType
from src.workload import Request, Segment, Workload, WorkloadValidationError, build_reuse_plan
from src.workload_runner import run_reuse_prefill, SplitEvent
from src.workload_extrapolation import sample_workload, extrapolate_report, replay


class WorkloadExtrapolationTests(unittest.TestCase):
    def workload(self, consumers=64):
        requests = [Request('owner',0,None,1,(Segment('doc','doc',16),),16)]
        for i in range(consumers):
            segs=(Segment('sys','sys',2),Segment('doc','doc',16),
                  Segment('parent_out','owner-out',1),Segment('user','u'+str(i),2))
            requests.append(Request('r'+str(i),1,'owner',3,segs,21))
        agents=[]
        for r in requests:
            agents.append(dict(id=r.request_id,tier=r.tier,parent=r.parent_id,lout=r.lout,
                history_len=0,segs=[dict(role=s.role,sha=s.fingerprint,len=s.length) for s in r.segments]))
        mapping={r.request_id:('owner' if r.request_id=='owner' else 'r0') for r in requests}
        raw=dict(agents=agents,meta=dict(simulation_sample=dict(representative_for=mapping)))
        return Workload('supervisor',tuple(requests),raw)

    def system(self):
        class GPU:
            def get_time_and_energy(self, op):
                return (1+op.m/100+op.numOp/1000)*1e-6, [1,0,0,0,0,0]
        class PIM:
            peak_memory_bandwidth=1e12
            softmax_peak_bandwidth=1e12
            energy_table={'mem':1,'sram':1}
            def get_time_and_energy(self,op): return 2e-6,[2,0,0,0,0,0]
            def get_time_and_energy_runs(self,op):
                return [self.get_time_and_energy(op) for _ in getattr(op,'pim_kv_runs',((),))]
        model=Transformer(dict(name='toy',ndec=1,num_heads=4,hdim=16,ff_scale=4,dtype=DataType.W16A16),tensor_parallel=1)
        return SimpleNamespace(model=model,devices={'GPU':GPU(),'Acc':PIM()},hetero_name=DeviceType.PIM)

    def test_large_input_is_retained_but_only_two_requests_are_materialized(self):
        full=self.workload()
        selected,sampling=sample_workload(full)
        self.assertEqual(len(full.requests),65)
        self.assertEqual(len(selected.requests),2)
        self.assertEqual(sampling['request_weights'],{'owner':1,'r0':64})
        self.assertFalse(sampling['full_event_graph_materialized'])

    def test_mapping_cannot_hide_different_shapes_or_dependencies(self):
        full=self.workload()
        bad=copy.deepcopy(full.raw)
        bad['meta']['simulation_sample']['representative_for'].pop('r63')
        with self.assertRaises(WorkloadValidationError):
            sample_workload(Workload(full.kind,full.requests,bad))
        from dataclasses import replace
        requests=list(full.requests);requests[-1]=replace(requests[-1],lout=4)
        with self.assertRaisesRegex(WorkloadValidationError,'shape'):
            sample_workload(Workload(full.kind,tuple(requests),full.raw))
        requests=list(full.requests);requests[-1]=replace(requests[-1],parent_id=None)
        with self.assertRaisesRegex(WorkloadValidationError,'parent'):
            sample_workload(Workload(full.kind,tuple(requests),full.raw))

    def test_every_combo_replays_the_small_graph_and_weights_full_metrics(self):
        full=self.workload(); small,sampling=sample_workload(full)
        for rung in ('A1','A2','A3b','A4c','A4e','A5','A6'):
            with self.subTest(rung=rung):
                system=self.system()
                policy='no-reuse' if rung=='A1' else 'recompute'
                plan=build_reuse_plan(small,policy,epic_prefix_recompute_tokens=2)
                ab=resolve_config(rung,None,None,None,policy=policy)
                report=run_reuse_prefill(system,small,plan,pipe=True,cacheblend_batch_size=8,
                    pim_prefill_mode=ab.prefill_attn,pim_batch_command=ab.pim_batch_command,
                    pim_pe_freq_ghz=ab.pim_pe_freq_ghz,gemv_buffer_bytes=ab.gemv_buffer_bytes,
                    decode_attn=ab.decode_attn,kv_mapping=ab.kv_mapping,
                    channel_placement=ab.channel_placement,warm=False)
                before=[e['time_s'] for e in report['events']]
                result=extrapolate_report(report,small,system.devices['GPU'],sampling)
                self.assertTrue(result['source_schedule_reproduced'])
                self.assertEqual(result['simulated_events'],report['event_count'])
                self.assertEqual(len(result['events']),len(report['events']))
                self.assertEqual(result['full_requests'],65)
                self.assertEqual(before,[e['time_s'] for e in report['events']])
                self.assertEqual(result['logical_decode_batch_max'],64)
                s=result['summary']['requests']
                self.assertAlmostEqual(result['mean_ttft_s'],(s['owner']['ttft_s']+64*s['r0']['ttft_s'])/65)
                self.assertAlmostEqual(result['weighted_tbt_s'],(s['r0']['end_s']-s['r0']['first_token_s'])/2)

    def test_scaling_scan_preserves_parallel_channels_and_critical_path(self):
        def e(i,dev,us,deps):
            return SplitEvent(i,0,0,'r',i,dev,1,us*1e-6,0,depends_on=tuple(deps))
        events=[e('q','LINK',1,()),e('scan0','PIM:pool0-0',10,('q',)),
                e('scan8','PIM:pool8-8',6,('q',)),e('ctx','LINK',1,('scan0','scan8'))]
        out=replay(events,{'r':8},None)
        self.assertEqual(out[1].start_s,out[2].start_s)
        self.assertAlmostEqual(out[-1].end_s,96e-6)
        from dataclasses import replace
        faster=list(events);faster[1]=replace(faster[1],time_s=8e-6)
        improved=replay(faster,{'r':8},None)
        self.assertAlmostEqual(out[-1].end_s-improved[-1].end_s,16e-6)

if __name__=='__main__': unittest.main()
