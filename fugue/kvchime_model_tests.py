"""Check model geometry and GQA accounting independently of simulated timings."""
import unittest,math
from fugue.runtime import REPO,load
from fugue.kvchime_model import Model,geometry
from src.model import Transformer,Layer
from src.type import LayerType,DataType
from src.system import System

class Bank:
    def prepare(self,keys):pass
    def get(self,n,h,r):return {'scan_us':1+n*.001}

class Tests(unittest.TestCase):
    def test_native_models_unchanged(self):
        for name,tp in [('LLAMA-7B',1),('GPT-13B',1),('LLAMA-65B',4)]:
            m=Model(name,tp,Bank());t=Transformer(m.m,tensor_parallel=tp);t.build(1,32,2,False)
            raw=[m.gpu.get_time_and_energy(l)[0]*1e6 for l in t.sum_decoder if l.type in [LayerType.MATMUL,LayerType.SOFTMAX]]
            self.assertAlmostEqual(m.gpu_attention(32,32,1)[0],sum(raw),places=8)
            a=m.capacity(1,1024,16);b=System(m.gconf,m.m).get_required_mem_capacity(1,1024,16)
            self.assertEqual(a,b)
    def test_all_workloads_fit_native_GPU_capacity(self):
        cb=load(REPO/'artifact/inputs/cacheblend.json');epic=load(REPO/'artifact/inputs/epic.json')
        shapes=[(1,r['total_prompt_tokens'],r['generated_tokens']) for r in cb['requests']]
        shapes += [(len(r['members']),r['total_prompt_tokens'],r['generated_tokens']) for r in epic['cases']]
        for spec in load(REPO/'artifact/inputs/kvchime.json')['models']:
            m=Model(spec['name'],spec['tensor_parallel'],Bank())
            for b,n,out in shapes:
                w,kv,tmp=m.capacity(b,n,out)
                self.assertLessEqual((w+kv+tmp)/m.tp,m.gconf['MEM_CAPACITY_PER_DEVICE'],(m.name,b,n))
        # The failing configuration must still fail; do not loosen its budget.
        m=Model('LLAMA-65B',2,Bank())
        b,n,out=next(x for x in shapes if x[0]==4)
        self.assertGreater(sum(m.capacity(b,n,out))/2,m.gconf['MEM_CAPACITY_PER_DEVICE'])
    def test_gqa_geometry_and_transfer(self):
        m=Model('LLAMA3.1-8B',1,Bank())
        self.assertEqual((m.heads,m.kvheads,m.gqa),(32,8,4))
        self.assertEqual(m.qbytes,8192);self.assertEqual(m.kvbytes,4096)
        self.assertEqual(m.m['ff_scale']*m.m['hdim'],14336)
    def test_gqa_flops_preserved_kv_reduced(self):
        q,n,d=17,1024,128
        mha=Layer('sum','score',LayerType.MATMUL,False,DataType.W16A16,q,n,d,32)
        gqa=Layer('sum','score',LayerType.MATMUL,False,DataType.W16A16,q*4,n,d,8)
        self.assertEqual(mha.get_flops(),gqa.get_flops())
        a,b=mha.get_size(),gqa.get_size();self.assertEqual(a[0],b[0]);self.assertEqual(a[1],b[1]*4);self.assertEqual(a[2],b[2])
    def test_gqa_qkv_projection(self):
        m=Model('LLAMA3.1-8B',1,Bank());m.common(17,17,1024,1)
        qkv=[k for k in m.ops if k[1]=='qkv'];self.assertEqual(len(qkv),1)
        self.assertEqual(qkv[0][4],(32+2*8)*128)
    def test_gqa_capacity(self):
        m=Model('LLAMA3.1-8B',1,Bank());w,kv,t=m.capacity(2,1024,16)
        nw,nkv,nt=System(m.gconf,m.m).get_required_mem_capacity(2,1024,16)
        self.assertEqual(kv,nkv/4);self.assertEqual(t,nt)
        self.assertEqual(nw-w,32*2*4096*4096*.75*2)

if __name__=='__main__':unittest.main()
