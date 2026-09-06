#!/usr/bin/env python3
"""Read an existing report only; strip address arrays for a bounded timeline audit."""
import hashlib,json,mmap,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=Path('/data2/chenyi9/KV-PIM/scratch_0905/diag_A6_events/dag_A6.json')
out=Path('/tmp/fugue-decode-scan-tbt/diag_A6_events_slim.jsonl')
out.parent.mkdir(parents=True,exist_ok=True)
# json.dump(indent=2) event objects have an exact four-space boundary.
with source.open('rb') as handle,mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as mm:
    start=mm.find(b'\n  "events": [\n');assert start>=0
    start=mm.find(b'    {\n',start)
    end=mm.find(b'\n  ],\n  "gemv_buffer_bytes":',start);assert end>start
    count=0
    with out.open('w') as dest:
        while start<end:
            stop=mm.find(b'\n    }',start)+len(b'\n    }');assert stop>start and stop<=end
            block=mm[start:stop]
            block=re.sub(rb'      "dram_addresses": \[.*?\],\n',b'',block,flags=re.S)
            event=json.loads(block)
            event.pop('energy_nj',None)
            event.pop('dram_addresses',None)
            # Large prefill query arrays are immaterial to this decode audit.
            if not event['name'].startswith('decode_'):event['query_positions']=[]
            dest.write(json.dumps(event,separators=(',',':'))+'\n');count+=1
            start=mm.find(b'    {\n',stop,end)
            if start<0:break
    # Everything except events/TLB/batches is small; retain the summary and run provenance.
    tail=json.loads(b'{'+mm[end+len(b'\n  ],'):mm.find(b'\n  "tlb":',end)].rstrip().rstrip(b',')+b'}')
    head=mm[:mm.find(b'\n  "batches":')].rstrip().rstrip(b',')+b'}'
    tail.update(json.loads(head))
    tail['source_report']=str(source);tail['source_bytes']=source.stat().st_size
    tail['source_mtime_ns']=source.stat().st_mtime_ns
    tail['extracted_events']=count
    tail['slim_path']=str(out)
    assert count==int(re.search(rb'  "event_count": (\d+)',mm[:mm.find(b'\n  "events": [\n')]).group(1))
    tail['slim_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    (HERE/'diag_A6_metadata.json').write_text(json.dumps(tail,indent=2)+'\n')
print(json.dumps({'events':count,'slim_bytes':out.stat().st_size,'config':tail.get('run_config'),'output':str(out)},indent=2))
