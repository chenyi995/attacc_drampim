"""Native AttAcc cost operators with explicit GQA shapes and bounded query tiles."""
from copy import deepcopy
from collections import Counter
import math,json,hashlib
from pathlib import Path
from fugue.runtime import csvout
from fugue.kvchime_selector import Selector
from fugue.kvchime_traces import TraceBank
from src.config import make_model_config,make_xpu_config,make_pim_config,SCALING_FACTOR,ENERGY_TABLE
from src.devices import xPU,PIM
from src.model import Layer,Transformer
from src.system import System
from src.type import DataType,DeviceType,GPUType,PIMType,InterfaceType,LayerType

def geometry(name):
    official=json.loads(Path(__file__).with_name('official-gqa-geometry.json').read_text())['models']
    if name in official:
        cfg=dict(official[name]);cfg['dtype']=DataType.W16A16
        return cfg
    cfg=make_model_config(name,DataType.W16A16)
    cfg['num_kv_heads']=cfg['num_heads']//cfg['gqa_size']
    return cfg


class Model:
    def __init__(self,name,tp,bank,prepare_profiles=True):
        self.name=name;self.tp=tp;self.m=geometry(name)
        self.layers=self.m['ndec'];self.heads=self.m['num_heads']//tp
        assert self.m['num_heads']%tp==0
        self.trace_dhead=self.m['dhead'];self.logical_dhead=int(self.m['hdim']/self.m['num_heads']);self.dbyte=2
        self.geometry_consistent=self.m['hdim']==self.m['num_heads']*self.trace_dhead
        self.geometry_scope='Native Transformer and native trace dimensions are retained separately, including inherited inconsistencies.'
        self.gqa=self.m['gqa_size'];self.kvheads=self.heads//self.gqa
        assert self.m['num_kv_heads']%tp==0
        self.h=math.ceil(self.kvheads/5);self.qbytes=self.heads*self.logical_dhead*self.dbyte
        # MHA uses the exact native System KV bytes per token/layer/shard; do
        # not replace hidden width with heads*truncated head dimension.
        self.kvbytes=2*self.m['hdim']*self.dbyte/tp if self.gqa==1 else 2*self.kvheads*self.logical_dhead*self.dbyte
        self.gconf=make_xpu_config(GPUType.A100a,num_gpu=tp)['GPU']
        self.gpu=xPU(DeviceType.GPU,self.gconf,SCALING_FACTOR)
        self.pim=PIM(make_pim_config(PIMType.BA,InterfaceType.NVLINK3,num_attacc=tp,num_hbm=5,power_constraint=True),SCALING_FACTOR,None)
        self.bank=bank
        self.selector=None
        if prepare_profiles:
            self.bank.prepare([(1024,self.h,8),(4096,self.h,8)],dhead=self.trace_dhead,dbyte=self.dbyte)
            self.selector=Selector(self.bank,self.h,group_size=self.gqa,dhead=self.trace_dhead,dbyte=self.dbyte)
        self.ops={};self.shared_cache={}
    def op(self,stage,name,kind,m,n,k,num=1,weight=False,device='GPU'):
        key=(stage,name,kind,m,n,k,num,weight,device)
        if key not in self.ops:
            layer=Layer(stage,name,kind,weight,DataType.W16A16,m,n,k,num)
            d=self.gpu if device=='GPU' else self.pim;t,e=d.get_time_and_energy(layer)
            self.ops[key]=(t*1e6,sum(e)*1e-6)
        return self.ops[key]
    def common(self,qkv,q,n,b,decode=False):
        tr=Transformer(self.m,tensor_parallel=self.tp);tr.build(b,q if not decode else n,2,False)
        ls=tr.gen_decoder[0] if decode else tr.sum_decoder
        result=[]
        for l in ls:
            if l.type in (LayerType.MATMUL,LayerType.SOFTMAX,LayerType.X2G):continue
            if l.name=='qkv':
                l.m=b*qkv
                if self.gqa>1:l.n=(self.heads+2*self.kvheads)*self.logical_dhead
            t,e=self.op(l.stage,l.name,l.type,l.m,l.n,l.k,l.numOp,l.has_weight)
            result.append((l.name,t,e))
        return result
    def gpu_attention(self,q,n,b):
        # Group Q rows for a shared KV head. Arithmetic still uses all Q heads;
        # KV operands are read/stored once per group, not replicated g times.
        v=[self.op('sum','score',LayerType.MATMUL,q*self.gqa,n,self.logical_dhead,self.kvheads*b),
           self.op('sum','softmax',LayerType.SOFTMAX,q,n,1,self.heads*b),
           self.op('sum','context',LayerType.MATMUL,q*self.gqa,self.logical_dhead,n,self.kvheads*b)]
        return tuple(sum(x[i] for x in v) for i in [0,1])
    def softmax(self,q,n,b):
        return self.op('sum','softmax',LayerType.SOFTMAX,q,n,1,self.heads*b,device='PIM')
    def link(self,byte_count):
        return byte_count/300e9*1e6,byte_count*self.tp*ENERGY_TABLE['GPU']['comm']*1e-6
    def dense_scan(self,q,n,b,mq=False):
        h=math.ceil(self.kvheads*b/5);total=q*self.gqa
        rs=([8]*(total//8)+([total%8] if total%8 else [])) if mq else [1]*total
        self.bank.prepare([(n,h,r) for r in set(rs)],dhead=self.trace_dhead,dbyte=self.dbyte)
        return self.combine([self.bank.get(n,h,r,dhead=self.trace_dhead,dbyte=self.dbyte) for r in rs])
    @staticmethod
    def combine(rows):
        return dict(scan_us=sum(x['scan_us'] for x in rows),
            memory_energy_pJ_per_attacc=sum(x['memory_energy_pJ_per_attacc'] for x in rows),
            dram_read_bytes_per_attacc=sum(x['dram_read_bytes_per_attacc'] for x in rows),
            mac_commands=sum(x['mac_commands'] for x in rows),
            represented_column_MACs=sum(x['represented_column_MACs'] for x in rows),
            first_private_k_us=rows[0].get('first_private_k_us',0),
            profile_repeats=json.dumps(dict(Counter(x['profile'] for x in rows)),sort_keys=True),
            profile=rows[0]['profile'] if len(rows)==1 else 'query-tiles',query_tiles=len(rows))
    def shared_scan(self,objects,views,mq):
        vs=deepcopy(views)
        for v in vs:
            v['query_count']=v.get('query_count',1)*self.gqa
            v['query_positions']=[p for p in v.get('query_positions',[]) for _ in range(self.gqa)]
        # One-query MHA has no MQ schedule change.
        if sum(v['query_count'] for v in vs)==1:mq=False
        def one(tile):
            key=hashlib.sha256(json.dumps([self.trace_dhead,self.dbyte,'stable-row-arena-v1',self.h,objects,tile,mq],sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
            if key not in self.bank.shared_cache:
                self.bank.shared_cache[key]=self.bank.shared('shared-'+key,objects,tile,self.h,mq,dhead=self.trace_dhead,dbyte=self.dbyte)
            return self.bank.shared_cache[key]
        if max(v['query_count'] for v in vs)<=8:return one(vs)
        # Native operator profiling: bounded resident-query tiles. Every query
        # is executed, every layer/request accounted; no request extrapolation.
        # Rectangular native attention costs do not depend on query position.
        rows=[]
        for v in vs:
            count=v['query_count'];full,tail=divmod(count,8)
            # The old tiled path clears positions before hashing every tile.
            # Preserve that exact identity without copying discarded positions.
            # Each view keeps its own schedule; only repeated cache lookups go.
            for size,repeats in [(8,full),(tail,1 if tail else 0)]:
                if not repeats:continue
                tile={key:deepcopy(value) for key,value in v.items() if key!='query_positions'}
                tile['query_count']=size;tile['query_positions']=[]
                tile['position_scope']='Timing-equivalent rectangular query tile; original positions retained in workload manifest.'
                rows.extend([one([tile])]*repeats)
        return self.combine(rows)
    def capacity(self,b,n,lout):
        w,kv,temp=System(self.gconf,self.m).get_required_mem_capacity(b,n,lout)
        h=self.m['hdim']
        w-=self.layers*2*h*h*(1-1/self.gqa)*2
        return w,kv/self.gqa,temp
    def scan_energy(self,row,q,n,b):
        # Preserve AttAcc's combined-score arithmetic convention for comparability.
        cal=q*n*self.logical_dhead*self.heads*b*ENERGY_TABLE['PIM'][PIMType.BA]['alu']
        return (row['memory_energy_pJ_per_attacc']+cal)*self.tp*1e-6

