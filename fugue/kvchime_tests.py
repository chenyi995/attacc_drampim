"""Focused semantic tests; no model weights or external repositories."""
import unittest
from fugue.kvchime_traces import TraceBank
from fugue.kvchime_selector import Selector

class Capture(TraceBank):
    def __init__(self):pass
    def base(self,n,h):
        return ['PIM_WR_GB 0x00000000','PIM_MAC_AB 0x00000000','PIM_SFM 0x00000000','PIM_MV_GB 0x00800000','PIM_MAC_AB 0x00800000']
    def run(self,name,lines,residents,metadata=None):
        self.lines,self.residents,self.metadata=lines,residents,metadata
        return {'mac_commands':len(residents),'represented_column_MACs':sum(residents)}

class Tests(unittest.TestCase):
    def view(self,agents):
        objects=[dict(id='shared',tokens=4,private=False)]+[dict(id=f'private-{i}',tokens=2,private=True) for i in range(agents)]
        views=[dict(logical_length=5,references=[dict(object='shared',valid_logical_positions=[0,2,3]),dict(object=f'private-{i}',valid_logical_positions=[1,4])]) for i in range(agents)]
        return objects,views
    def test_replacement_is_unique(self):
        objects,views=self.view(2);views[0]['references'][0]['valid_logical_positions'].append(1)
        with self.assertRaises(AssertionError):Capture().shared('invalid',objects,views,1,True)
    def test_mq_reuses_reads_preserves_macs(self):
        objects,views=self.view(4);a,b=Capture(),Capture()
        ra=a.shared('single',objects,views,1,False);rb=b.shared('mq',objects,views,1,True)
        self.assertEqual(ra['represented_column_MACs'],rb['represented_column_MACs'])
        self.assertLess(rb['mac_commands'],ra['mac_commands'])
        self.assertEqual(sum('PIM_SFM' in s for s in b.lines),4)
    def test_global_softmax_boundary(self):
        objects,views=self.view(2);c=Capture();c.shared('mq',objects,views,1,True)
        sfm=[i for i,s in enumerate(c.lines) if s.startswith('PIM_SFM')]
        for block in c.metadata['blocks']:
            if block['phase']=='QK':self.assertLessEqual(block['command_end'],min(sfm))
            else:self.assertGreater(block['command_start'],max(sfm))
    def test_one_query_degenerates(self):
        objects,views=self.view(1);a,b=Capture(),Capture();a.shared('a',objects,views,1,False);b.shared('b',objects,views,1,True)
        self.assertEqual(a.lines,b.lines);self.assertEqual(a.residents,b.residents)
    def test_tail_never_exceeds_capacity(self):
        objects,views=self.view(9);c=Capture();c.shared('mq',objects,views,1,True)
        self.assertEqual(max(c.residents),8)
        self.assertEqual(sum(c.residents),36)
    def test_selector_uses_only_calibration(self):
        class Profile:
            def __init__(self):self.calls=[]
            def get(self,n,h,q):
                self.calls.append((n,h,q));return {'scan_us':1+n*.001}
        p=Profile();s=Selector(p,7);objects,views=self.view(2)
        s.choose_views(objects,views,10,1,1,1)
        self.assertEqual(p.calls,[(1024,7,8),(4096,7,8)])

if __name__=='__main__':unittest.main()
