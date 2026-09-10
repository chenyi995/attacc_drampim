class Inspect(impl.Model):
    def op(self,stage,name,kind,m,n,k,num=1,weight=False,device='GPU',kv_group=1):
        result=super().op(stage,name,kind,m,n,k,num,weight,device,kv_group)
        native=impl.native
        layer=native.Layer(stage,name,kind,weight,native.DataType.W16A16,m,n,k,num)
        layer.kv_group_size=kv_group
        C=0.;mem=[0.]*4;overlap=0.;decomposed_compute=0.
        if device=='PIM':
            assert kind==native.LayerType.SOFTMAX
            C=self.pim._compute_time(layer)*1e6
            mem[0]=self.pim._mem_time(layer)*1e6
            close(max(C,*mem),result[0])
            parts=dict(memory=0.,compute=0.,other=result[0])
            rule='Unsplit buffer-die softmax; its SRAM bound is not GPU HBM.'
        elif kind in (native.LayerType.G2G,native.LayerType.X2G):
            parts=dict(memory=0.,compute=0.,other=result[0])
            rule='Native TP or operator communication retained in Other.'
        else:
            C=self.gpu._compute_time(layer)*1e6
            mem=[v*1e6 for v in self.gpu._mem_time(layer)]
            close(max(C,*mem),result[0])
            if kind in (native.LayerType.ACT,native.LayerType.NORM):
                parts=dict(memory=0.,compute=0.,other=result[0])
                rule='Unsplit empirical kernel fit, including its fixed intercept.'
            else:
                assert kind in (native.LayerType.FC,native.LayerType.MATMUL,native.LayerType.SOFTMAX)
                M=max(mem)
                parts=dict(memory=M,compute=max(0.,C-M),other=0.)
                overlap=min(C,M);decomposed_compute=C
                rule='Native memory-first attribution: M plus max(C-M,0).'
        close(sum(parts.values()),result[0])
        self.captured.append(dict(name=name,stage=stage,kind=kind.name,device=device,
            shape=dict(m=m,n=n,k=k,numOp=num,dbyte=layer.dbyte,kv_group=kv_group),
            total_us=result[0],parts_us=parts,attribution_rule=rule,
            memory_only_us=parts['memory']-overlap,memory_compute_overlap_us=overlap,
            decomposed_GPU_compute_bound_us_not_additive=decomposed_compute,
            raw_compute_bound_us_not_additive=C,
            raw_memory_bounds_us_not_additive=dict(zip(('HBM_or_empirical_kernel','L2','L1','RF'),mem))))
        return result
