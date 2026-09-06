"""Validate repeated structures without rerunning the physical PIM model."""
import unittest

from docs.experiments.estimate_decode_batch import extract_step, replay_step


class DecodeBatchEstimateTests(unittest.TestCase):
    @staticmethod
    def structure():
        def event(eid, device, duration, deps, name=None):
            return dict(id=eid, device=device, name=name or eid, time_s=duration*1e-6,
                depends_on=deps, transformer_layer=0, link_bytes=128,
                request='batch', batch_members=['r'+str(i) for i in range(8)],
                query_positions=[20]*8, rows=8, start_s=0.0, end_s=0.0)
        return [event('qkv', 'GPU', 2, [], 'decode_batch_qkv'),
            event('q', 'LINK', 1, ['qkv'], 'decode_q_gpu_to_pim'),
            event('scan0', 'PIM:pool0-0', 10, ['q'], 'decode_pim_kv_scan'),
            event('scan8', 'PIM:pool8-8', 6, ['q'], 'decode_pim_kv_scan'),
            event('new', 'PIM', 0, ['q'], 'decode_pim_new_token_qk_pv'),
            event('merge', 'DIE', 0, ['scan0', 'scan8', 'new'], 'decode_die_lse_merge'),
            event('ctx', 'LINK', 1, ['merge'], 'decode_ctx_pim_to_gpu'),
            event('proj', 'GPU', 3, ['ctx'], 'decode_batch_gpu_proj')]

    def test_batch8_to64_keeps_parallel_heads_and_reprices_gpu(self):
        events = self.structure()
        def price(event, batch):
            self.assertEqual(batch, 64)
            return (4 if event['id'] == 'qkv' else 5)*1e-6
        trace, metrics = replay_step(events, 8, 64, price)
        e = {r['id']: r for r in trace}
        self.assertAlmostEqual(e['scan0']['time_s'], 80e-6)
        self.assertAlmostEqual(e['scan8']['time_s'], 48e-6)
        self.assertEqual(e['scan0']['start_s'], e['scan8']['start_s'])
        self.assertAlmostEqual(e['proj']['start_s'], 100e-6)
        self.assertAlmostEqual(metrics['tbt_s'], 105e-6)
        self.assertEqual(e['new']['time_s'], 0)
        self.assertEqual(e['q']['link_bytes'], 1024)
        self.assertAlmostEqual(events[2]['time_s'], 10e-6)

    def test_factor1_reproduces_chain_and_same_channel_contention(self):
        events = self.structure()
        events[3]['device'] = 'PIM:pool0-0'
        trace, metrics = replay_step(events, 8, 8, lambda e, batch: e['time_s'])
        e = {r['id']: r for r in trace}
        self.assertAlmostEqual(e['scan8']['start_s'], e['scan0']['end_s'])
        self.assertAlmostEqual(metrics['tbt_s'], 23e-6)

    def test_scan_savings_propagate_scaled_to_tbt(self):
        first = self.structure()
        second = self.structure()
        second[2]['time_s'] = 8e-6
        price = lambda e, batch: e['time_s']
        _, a = replay_step(first, 8, 64, price)
        _, b = replay_step(second, 8, 64, price)
        self.assertAlmostEqual(a['tbt_s']-b['tbt_s'], 16e-6)
        with self.assertRaises(ValueError):
            replay_step(first, 3, 64, price)

    def test_extract_preserves_other_members_and_drops_previous_token(self):
        source = self.structure()
        source[0]['depends_on'] = ['previous_token']
        for i, e in enumerate(source):
            e['start_s'], e['end_s'] = i*1e-6, (i+1)*1e-6
        selected, base, _ = extract_step(source, 'r0', 20)
        self.assertEqual(base, 8)
        self.assertEqual(selected[0]['depends_on'], [])
        self.assertEqual({e['id'] for e in selected}, {e['id'] for e in source})


if __name__ == '__main__':
    unittest.main()
