"""Check useful-work accounting for shared decode and multi-query prefill."""
import os
import tempfile
import unittest


class BankThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpl_dir = tempfile.TemporaryDirectory()
        cls.previous_mpl_dir = os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR'] = cls.mpl_dir.name
        from docs.experiments.plot_minimal_layout import bank_measure
        cls.measure = staticmethod(bank_measure)

    @classmethod
    def tearDownClass(cls):
        if cls.previous_mpl_dir is None:
            os.environ.pop('MPLCONFIGDIR', None)
        else:
            os.environ['MPLCONFIGDIR'] = cls.previous_mpl_dir
        cls.mpl_dir.cleanup()

    def event(self, name, positions, duration, *, request='target', members=()):
        return dict(request=request, batch_members=list(members), name=name,
                    query_positions=list(positions), transformer_layer=0, time_s=duration)

    def test_decode_includes_shared_scans_without_other_members_work(self):
        events=[]
        for position in (0, 1):
            for duration in (9e-6, 5e-6):
                events.append(self.event('decode_batch_pim_kv_scan', (position,9), duration,
                    request='batch', members=('target','other')))
            events.append(self.event('decode_pim_kv_scan', (position,), 1e-6))
            events.append(self.event('decode_pim_kv_scan_new_token', (position,), 2e-6))
        throughput,seconds=self.measure(events,'target',True,2,3)
        self.assertAlmostEqual(seconds,12e-6)
        self.assertAlmostEqual(throughput,(4*2*3*2)/12e-6/1e9)

    def test_prefill_keeps_every_query_in_an_unbatched_sweep(self):
        events=[self.event('pim_kv_scan', (0,1,2), 3e-6)]
        throughput,seconds=self.measure(events,'target',False,2,3)
        self.assertAlmostEqual(seconds,3e-6)
        self.assertAlmostEqual(throughput,(4*2*3*(1+2+3))/3e-6/1e9)
