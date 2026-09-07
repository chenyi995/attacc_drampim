"""Native profiles; the MQ command expansion used in the final published sweeps."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json,math,subprocess,sys
from src.config import ENERGY_TABLE
from src.model import Layer
from src.type import *
from src.ramulator_wrapper import Ramulator
TCK=.769;CAP=8
E_COL=ENERGY_TABLE['PIM'][PIMType.BA]['mem']*32
E_OP=16*ENERGY_TABLE['PIM'][PIMType.BA]['alu']+32*ENERGY_TABLE['PIM'][PIMType.BA]['sram']
def interval(r):
    # Exact one-MAC-per-tCK design point avoids ceil(8/(1.3*0.769))=9
    # caused solely by rounding the stated 1.3-GHz clock.
    assert 1 <= r <= CAP
    return max(6, r, math.ceil(6*(E_COL+r*E_OP)/(E_COL+E_OP)))

def expand(lines, r):
    return [line for line in lines for _ in range(1 if line.startswith(('PIM_BARRIER', 'PIM_MAC_AB')) else r)]

def profiler(repo,root,raw,runtime,model,batch=1,cached_lengths=()):
    REPO,ROOT,RAW,RUNTIME=map(Path,(repo,root,raw,runtime))
    cfg={"cached_lengths":list(cached_lengths)}
    base_lines={}
    def make_base(n):
        folder = RAW/f'Fugue-asplos-N{n:05d}-Q1'
        folder.mkdir()
        path = folder/'Fugue-asplos-input.trace'
        command = [sys.executable,str(REPO/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py'),
                   '-dh','128','-nh',str(math.ceil(32*batch/5)),'-l',str(n),'-maxl',str(max(4096,n)),'-db','2','-o',str(path)]
        result = subprocess.run(command,capture_output=True,text=True)
        (folder/'Fugue-asplos-generator.log').write_text(result.stdout+result.stderr)
        (folder/'Fugue-asplos-generator.json').write_text(json.dumps(dict(command=command,returncode=result.returncode),indent=2)+'\n')
        assert result.returncode == 0
        lines = path.read_text().splitlines()
        assert lines and all(line for line in lines)
        return n,lines

    def simulate(n,r):
        folder = RAW/f'Fugue-asplos-N{n:05d}-Q{r}'
        folder.mkdir(exist_ok=True)
        trace = folder/'Fugue-asplos-input.trace'
        lines = expand(base_lines[n],r)
        if r==1:
            assert lines == base_lines[n] and trace.read_text() == '\n'.join(lines)+'\n'
        else:
            trace.write_text('\n'.join(lines)+'\n')
        counts, basecounts = Counter(line.split()[0] for line in lines), Counter(line.split()[0] for line in base_lines[n])
        for op,count in basecounts.items():
            assert counts[op] == count*(1 if op in ('PIM_MAC_AB','PIM_BARRIER') else r)
        yaml = folder/'Fugue-asplos-input.yaml'
        Ramulator(model,str(folder)).make_yaml_file(str(yaml),'Fugue-asplos-input',True)
        text = yaml.read_text()
        if r>1:
            text = text.replace('      preset: HBM3_5.2Gbps\n',f'      preset: HBM3_5.2Gbps\n      nCCDAB: {interval(r)}\n')
        # Native recorder adds observation only; no simulator source changes.
        text = text.replace('    plugins:\n',f'    plugins:\n      - ControllerPlugin:\n          impl: HBM3TraceRecorder\n          path: {folder}/Fugue-asplos-commands\n')
        yaml.write_text(text)
        command = [str(RUNTIME/'ramulator2'),'-f',str(yaml)]
        result = subprocess.run(command,capture_output=True,text=True)
        (folder/'Fugue-asplos-stdout.log').write_text(result.stdout)
        (folder/'Fugue-asplos-stderr.log').write_text(result.stderr)
        (folder/'Fugue-asplos-command.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,command_counts=dict(counts)),indent=2)+'\n')
        assert result.returncode == 0, folder
        cycles = [int(line.split()[-1]) for line in result.stdout.splitlines() if 'memory_system_cycles:' in line]
        assert len(cycles)==1 and cycles[0]>0
        first_new = {}
        key_events = []
        mac_events = 0
        for path in sorted(folder.glob('Fugue-asplos-commands.ch*')):
            for line in path.read_text().splitlines():
                fields = [s.strip() for s in line.split(',')]
                if fields[1] != 'MACAB':
                    continue
                mac_events += 1
                clock = int(fields[0])
                addr = list(map(int,fields[2:]))
                assert len(addr)==7
                if addr[5] < 8192:  # Native V base = 2^23 bytes; K occupies earlier rows.
                    key_events.append((addr[5]*32+addr[6],clock))
        assert mac_events == counts['PIM_MAC_AB']
        for c in cfg['cached_lengths']:
            if c>=n or c==0:
                continue
            assert c%16==0
            column = 2*(c//16)
            clocks = [clock for index,clock in key_events if index>=column]
            assert clocks, (n,r,c)
            # From scan launch until the first MAC that consumes new K. This is
            # the overlap window of the first Q tile, not the sum over all tiles.
            first_new[c] = min(clocks)*TCK/1000
            assert 0 < first_new[c] < cycles[0]*TCK/1000
        record = dict(n=n,resident_queries=r,nCCDAB=interval(r),cycles=cycles[0],scan_us=cycles[0]*TCK/1000,
                      trace_commands=len(lines), mac_commands=counts['PIM_MAC_AB'],
                      represented_mac_operations=counts['PIM_MAC_AB']*r,
                      key_event_count=len(key_events),softmax_commands=counts['PIM_SFM'],
                      path=str(folder.relative_to(ROOT)))
        (folder/'Fugue-asplos-timing.json').write_text(json.dumps(dict(**record,first_new_k_us=first_new),indent=2)+'\n')
        return (n,r),(record,first_new)

    return SimpleNamespace(make_base=make_base,simulate=simulate,base_lines=base_lines)
