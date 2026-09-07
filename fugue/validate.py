"""Input integrity and fresh-results comparison; never imported by cost models."""
from pathlib import Path
import csv,hashlib,json,math,os
from fugue.runtime import REPO,RUN,load,save,sha,check_sources

def inputs():
    for line in (REPO/'artifact/Fugue-asplos-inputs-SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha(REPO/name)!=digest:raise AssertionError('Input modified: '+name)
    sizes={}
    for name in ['cacheblend','epic']:
        data=load(REPO/'artifact/inputs'/f'{name}.json');catalog={c['chunk_id']:c for c in data['cache_catalog']}
        for key,c in catalog.items():
            assert c['tokens']==len(c['token_ids'])
            assert hashlib.sha256(json.dumps(c['token_ids'],separators=(',',':')).encode()).hexdigest()==key
        if name=='cacheblend':
            for r in data['requests']:
                c=sum(catalog[k]['tokens'] for k in r['chunk_ids'])
                assert c==r['cached_input_tokens']
                assert r['total_prompt_tokens']==c+r['new_suffix_tokens']
                assert r['selected_query_tokens']==int(c*.16)+r['new_suffix_tokens']
            sizes[name]=len(data['requests'])
        else:
            for case in data['cases']:
                assert len(case['members'])==case['batch']
                for member in case['members']:
                    lengths=[catalog[k]['tokens'] for k in member['chunk_ids']];offset=[0]
                    for length in lengths:offset.append(offset[-1]+length)
                    selected=[]
                    for i in range(1,len(offset)-1):
                        selected+=list(range(offset[i],offset[i+1] if i==len(offset)-2 else min(offset[i]+16,offset[-1])))
                    assert sorted(set(selected))==member['recomputed_indices']
                    assert len(set(selected))==case['selected_query_tokens'] and offset[-1]==case['total_prompt_tokens']
                unique={k for m in case['members'] for k in m['chunk_ids']}
                assert sum(catalog[k]['tokens'] for k in unique)==case['immutable_pool_tokens']
            sizes[name]=len(data['cases'])
    return dict(input_hashes_pass=True,token_counts_and_recompute_indices_pass=True,workloads=sizes)

def compare(actual,expected,keys):
    a=list(csv.DictReader(actual.open()));e=list(csv.DictReader(expected.open()))
    by={tuple(r[k] for k in keys):r for r in a};assert len(by)==len(a)
    assert len(a)==len(e),(actual,len(a),len(e))
    checked=0;largest=0.
    for r in e:
        t=by[tuple(r[k] for k in keys)]
        for k,value in r.items():
            try:x=float(value)
            except (ValueError,TypeError):
                if k.endswith('_winner'):assert t[k]==value
                continue
            assert k in t,(actual,k)
            y=float(t[k]);assert math.isfinite(y)
            assert math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-9),(actual,tuple(r[k] for k in keys),k,x,y)
            largest=max(largest,abs(x-y)/max(abs(x),1e-12));checked+=1
    return dict(actual=str(actual.relative_to(RUN)),rows=len(a),numeric_fields_compared=checked,max_relative_difference=largest)

def main():
    details=inputs();check_sources();wanted={int(v) for v in os.environ.get('FUGUE_EXPERIMENTS','1,2,3,4,5').split(',')}
    comparisons=[]
    for i in [1,2]:
        if i in wanted:comparisons.append(compare(RUN/f'experiment12/Fugue-asplos-results/Fugue-asplos-experiment{i}.csv',REPO/f'artifact/reference/experiment{i}.csv',['q','cached']))
    if 3 in wanted:comparisons.append(compare(RUN/'experiment3/Fugue-asplos-results/Fugue-asplos-summary.csv',REPO/'artifact/reference/experiment3.csv',['request_id','variant']))
    if wanted&{4,5}:comparisons.append(compare(RUN/'experiment45/Fugue-asplos-results/Fugue-asplos-summary.csv',REPO/'artifact/reference/experiment45.csv',['case_id','variant']))
    profiles=0
    for folder in ['experiment12','experiment3','experiment45']:
        if not (RUN/folder).exists():continue
        for path in (RUN/folder).rglob('Fugue-asplos-command.json'):
            record=load(path);assert record['returncode']==0
            assert Path(record['command'][0]).resolve().is_relative_to(RUN)
            profiles+=1
    report=dict(**details,comparisons=comparisons,fresh_profile_records_verified=profiles,all_native_sources_match=True,
        tolerance='Every numeric field: rel_tol=1e-8, abs_tol=1e-9. Winners exact. No timing data imported by simulation.')
    save(RUN/'Fugue-asplos-verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
