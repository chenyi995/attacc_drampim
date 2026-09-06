#!/usr/bin/env python3
"""Read-only checks plus saving the evidence used by this design session."""
import ast,csv,hashlib,json,re,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    report=HERE/'checks/validation.json'
    result={}
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    result['python_sources_parse']=True
    for p in (HERE/'README.md',HERE/'HANDCALC.md'):
        for command in re.findall(r'```bash\n(.*?)```',p.read_text(),re.S):
            subprocess.run(['bash','-n'],input=command.encode(),check=True)
    result['documented_bash_syntax']=True
    manifest=json.loads((HERE/'inputs/manifest.json').read_text())
    data=json.loads((HERE/'checks/all.json').read_text())
    assert set(manifest)==set(data['inputs'])
    for name,meta in manifest.items():
        digest=hashlib.sha256((HERE/'inputs'/(name+'.json')).read_bytes()).hexdigest()
        assert digest==meta['sha256']==data['inputs'][name]['input_sha256']
    result['workload_manifest_and_check_hashes']=len(manifest)
    # Exercise the new cohort arithmetic on existing real reports, rather
    # than a mock event stream or a newly executed simulation.
    base=Path('/data2/chenyi9/KV-PIM/scratch_0905/proto_s11_CACHEBLEND-TINY')
    wl=ROOT/'workload/probe/sweep/C1_S11_brief_64_turns.json'
    output=HERE/'checks/existing_S11_cohorts.csv'
    subprocess.run([sys.executable,str(HERE/'cohort_report.py'),str(base/'C1_S11_brief_64_turns'),str(wl),
                    '--summary-pattern',r'^w\d+_t','--output',str(output)],check=True)
    existing={r['combo']:r for r in csv.DictReader((HERE/'checks/existing_S11_protocol.csv').open())}
    rows=list(csv.DictReader(output.open()))
    for r in rows:
        if r['cohort']=='all_requests':
            assert abs(float(r['tbt_weighted_us'])-float(existing[r['rung']]['tbt_weighted_us']))<1e-8
    result['existing_report_weighted_TBT_matches_protocol']=True
    configs={}
    for rung in existing:
        p=base/'C1_S11_brief_64_turns'/('dag_'+rung+'.json')
        content=p.read_bytes();record=json.loads(content)
        configs[rung]={'report_path':str(p),'sha256':hashlib.sha256(content).hexdigest(),
                       'run_config':record['run_config'],'corrected_rows_sha':record['corrected_rows_sha']}
    assert len({json.dumps(r['run_config'],sort_keys=True) for r in configs.values()})==1
    assert len({r['corrected_rows_sha'] for r in configs.values()})==1
    (HERE/'checks/existing_S11_provenance.json').write_text(json.dumps(configs,indent=2)+'\n')
    result['existing_S11_recorded_configs_equal']=True
    result['performance_simulation_started']=False
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
