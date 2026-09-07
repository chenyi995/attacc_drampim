"""Shared-object command replay on unmodified AttAcc mapping and Ramulator.

MQ is represented by one DRAM MAC command with r MACs at interval(r).
The sidecar retains the query fanout, position/version masks and phase order.
Ramulator prices commands, not numeric operands; semantic equivalence is proved
separately. Mixed streams use their maximum MQ interval conservatively.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,math,subprocess,sys
from fugue.attention import TCK,interval,expand
from fugue.runtime import REPO,JOBS,save
from src.ramulator_wrapper import Ramulator
from src.config import ENERGY_TABLE
from src.type import PIMType

class TraceBank:
    def __init__(self, root, runtime, model):
        self.root=Path(root);self.root.mkdir(exist_ok=True)
        self.runtime=Path(runtime);self.model=model;self.bases={};self.profiles={}
        self.native_commands=[];self.shared_cache={}

    def base(self,n,h):
        key=(n,h)
        if key in self.bases:return self.bases[key]
        assert n>0 and h>0
        folder=self.root/f'native-N{n}-H{h}';folder.mkdir()
        path=folder/'input.trace'
        cmd=[sys.executable,str(REPO/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py'),
             '-dh','128','-nh',str(h),'-l',str(n),'-maxl',str(max(4096,n)),'-db','2','-o',str(path)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        save(folder/'generator.json',dict(command=cmd,returncode=result.returncode))
        (folder/'generator.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0
        lines=path.read_text().splitlines();self.bases[key]=lines
        return lines

    def run(self,name,lines,residents,metadata=None):
        folder=self.root/name;folder.mkdir()
        trace=folder/'input.trace';trace.write_text('\n'.join(lines)+'\n')
        rmax=max(residents,default=1);yaml=folder/'input.yaml'
        Ramulator(self.model,str(folder)).make_yaml_file(str(yaml),'input',True)
        text=yaml.read_text().replace('      preset: HBM3_5.2Gbps\n',
            f'      preset: HBM3_5.2Gbps\n      nCCDAB: {interval(rmax)}\n')
        text=text.replace('    plugins:\n',f'    plugins:\n      - ControllerPlugin:\n          impl: HBM3TraceRecorder\n          path: {folder}/commands\n')
        yaml.write_text(text)
        cmd=[str(self.runtime/'ramulator2'),'-f',str(yaml)]
        p=subprocess.run(cmd,capture_output=True,text=True)
        (folder/'stdout.log').write_text(p.stdout);(folder/'stderr.log').write_text(p.stderr)
        counts=Counter(s.split()[0] for s in lines)
        save(folder/'command.json',dict(command=cmd,returncode=p.returncode,counts=counts))
        assert p.returncode==0,(name,p.stderr)
        cycles=[int(s.split()[-1]) for s in p.stdout.splitlines() if 'memory_system_cycles:' in s]
        assert len(cycles)==1 and cycles[0]>0
        emitted_mac=sum(1 for f in folder.glob('commands.ch*') for line in f.open() if ',MACAB,' in line.replace(' ',''))
        assert emitted_mac==counts['PIM_MAC_AB'],(name,emitted_mac,counts)
        assert len(residents)==counts['PIM_MAC_AB']
        wr,sb,gb,mac=[counts[k] for k in ['PIM_WR_GB','PIM_MV_SB','PIM_MV_GB','PIM_MAC_AB']]
        traffic=[wr*32,(wr+sb+gb)*32,(wr+sb+gb)*32,(wr+sb+gb)*32,mac*32*2*2*4*4]
        table=ENERGY_TABLE['PIM'][PIMType.BA]
        memory_pj=(sum(v*e for v,e in zip(traffic,table['io']))+traffic[-1]*table['mem'])*5
        # Native combined-score convention: memory traffic covers QK and PV;
        # arithmetic is charged later from the logical score FLOP count.
        row=dict(profile=name,cycles=cycles[0],scan_us=cycles[0]*TCK/1000,
            nCCDAB=interval(rmax),mac_commands=mac,represented_column_MACs=sum(residents),
            dram_read_bytes_per_attacc=traffic[-1]*5,memory_energy_pJ_per_attacc=memory_pj,
            query_movement_commands=wr+sb+gb,softmax_commands=counts['PIM_SFM'],
            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),path=str(folder))
        save(folder/'timing.json',row)
        if metadata is None:
            events=[]
            for f in folder.glob('commands.ch*'):
                for line in f.open():
                    z=[v.strip() for v in line.split(',')]
                    if z[1]=='MACAB' and int(z[7])<8192:events.append([int(z[7])*32+int(z[8]),int(z[0])])
            row['key_events']=events
        if metadata is not None:save(folder/'logical-view.json',metadata)
        if metadata is not None:
            private={i for i,o in enumerate(metadata['objects']) if o.get('private',False)}
            clocks=[]
            for f in folder.glob('commands.ch*'):
                for line in f.open():
                    fields=[s.strip() for s in line.split(',')]
                    if fields[1]!='MACAB':continue
                    addr=list(map(int,fields[2:]));local=(addr[5]*32+addr[6])*32
                    if local<(1<<23) and local//(1<<17) in private:clocks.append(int(fields[0]))
            row['first_private_k_us']=min(clocks)*TCK/1000 if clocks else 0.
            save(folder/'timing.json',row)
        self.native_commands.append(row)
        return row

    def prepare(self,keys):
        keys=sorted(set(keys)-self.profiles.keys())
        for n,h,r in keys:self.base(n,h)
        def one(key):
            n,h,r=key;lines=expand(self.bases[n,h],r)
            row=self.run(f'scan-N{n}-H{h}-Q{r}',lines,[r]*sum(s.startswith('PIM_MAC_AB') for s in lines))
            return key,row
        with ThreadPoolExecutor(max_workers=JOBS) as pool:
            for key,row in pool.map(one,keys):self.profiles[key]=row

    def get(self,n,h,r=1):
        if (n,h,r) not in self.profiles:self.prepare([(n,h,r)])
        return self.profiles[n,h,r]

    def shared(self,name,objects,views,h,mq):
        """QK all objects -> one softmax per query -> PV all objects.

        Each object occupies the native token layout at a disjoint row base.
        Physical reads of replaced shared tokens remain charged; sidecar masks
        remove their scores. Private packed entries carry their logical indices.
        """
        assert h<=16 and len(objects)<=64
        ids={o['id']:i for i,o in enumerate(objects)}
        refs={oid:[] for oid in ids}
        query_ranges=[];nqueries=0
        for i,v in enumerate(views):
            qs=list(range(nqueries,nqueries+v.get('query_count',1)))
            query_ranges.append(qs);nqueries+=len(qs)
            positions=[]
            for ref in v['references']:
                refs[ref['object']].extend(qs);positions.extend(ref['valid_logical_positions'])
            assert sorted(positions)==list(range(v['logical_length'])),(name,i)
        phases={};sfm=None
        for obj in objects:
            src=self.base(obj['tokens'],h)
            markers=[i for i,s in enumerate(src) if s.startswith('PIM_SFM')]
            assert len(markers)==h and markers==list(range(markers[0],markers[-1]+1))
            phases[obj['id']]={'QK':src[:markers[0]],'PV':src[markers[-1]+1:]}
            sfm=src[markers[0]:markers[-1]+1]
        lines=[];resident=[];blocks=[]
        barrier=[f'PIM_BARRIER 0x{ch*(1<<30):08x}' for ch in range(16)]
        def emit(oid,queries,phase):
            offset=ids[oid]*(1<<17);start=len(lines);r=len(queries)
            for line in phases[oid][phase]:
                op,addr=line.split();address=int(addr,16)
                # Native row/channel/bank mapping retained; only base allocation.
                if op!='PIM_BARRIER':address+=offset
                line=f'{op} 0x{address:08x}'
                if op=='PIM_MAC_AB':lines.append(line);resident.append(r)
                elif op=='PIM_BARRIER':lines.append(line)
                else:lines.extend([line]*r)
            blocks.append(dict(phase=phase,object=oid,queries=queries,resident_queries=r,
                command_start=start,command_end=len(lines),row_base_bytes=offset))
        for phase in ['QK','PV']:
            if mq:
                for obj in objects:
                    qs=refs[obj['id']]
                    for start in range(0,len(qs),8):emit(obj['id'],qs[start:start+8],phase)
            else:
                for qs,v in zip(query_ranges,views):
                    for q in qs:
                        for ref in v['references']:emit(ref['object'],[q],phase)
            lines.extend(barrier)
            if phase=='QK':
                for q in range(nqueries):lines.extend(sfm)
                lines.extend(barrier)
        metadata=dict(objects=objects,views=views,query_ranges=query_ranges,blocks=blocks,
            semantics='All QK before query-global softmax, then all PV. Invalid old scores suppressed; private positions replace, never append twice.',
            timing='Maximum resident-query command interval applied to entire mixed stream, including private blocks; conservative.',
            arithmetic='Timing trace and version/position descriptors; Ramulator does not simulate numeric Q/K/V.')
        row=self.run(name,lines,resident,metadata)
        row['logical_tokens']=sum(v['logical_length']*v.get('query_count',1) for v in views)
        row['object_tokens']=sum(o['tokens'] for o in objects)
        row['scan_kind']='shared_MQ' if mq else 'shared_single_query'
        return row
