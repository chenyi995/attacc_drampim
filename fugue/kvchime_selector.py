"""A two-point scan estimator; no evaluation scan latency enters a decision."""
import math
from fugue.attention import interval

class Selector:
    def __init__(self,profiles,h,group_size=1):
        self.group_size=group_size
        # These two shapes are exclusively calibration inputs for every head count.
        p1=profiles.get(1024,h,8);p2=profiles.get(4096,h,8)
        self.a=(p2['scan_us']-p1['scan_us'])/(4096-1024)
        self.b=p1['scan_us']-1024*self.a
        self.calibration=dict(heads_per_HBM=h,n1=1024,n2=4096,q=8,
            scan1_us=p1['scan_us'],scan2_us=p2['scan_us'],a_us_per_token=self.a,b_us=self.b)

    def choose(self,q,n,gpu_service_us,softmax_us,q_input_us,new_kv_us,overlap_us=0):
        groups=q//8;tail=q%8
        one=max(0,self.a*n+self.b)
        scan=one*(groups+(interval(tail)/8 if tail else 0))
        estimate=scan+softmax_us+2*q_input_us+max(0,new_kv_us-overlap_us)
        return ('PIM' if estimate<gpu_service_us else 'GPU'),estimate

    def choose_views(self,objects,views,gpu_us,softmax_us,qin_us,write_us):
        uses={o['id']:0 for o in objects}
        for v in views:
            for ref in v['references']:uses[ref['object']]+=v.get('query_count',1)*self.group_size
        scan=0.
        for obj in objects:
            q=uses[obj['id']];groups,tail=divmod(q,8)
            scan+=max(0,self.a*obj['tokens']+self.b)*(groups+(interval(tail)/8 if tail else 0))
        estimate=scan+softmax_us+qin_us+write_us
        return ('PIM' if estimate<gpu_us else 'GPU'),estimate
