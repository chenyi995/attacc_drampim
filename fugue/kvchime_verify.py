"""Verify six-panel Experiment 1 and independently redraw every figure package."""
import csv,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from fugue.runtime import REPO,RUN,JOBS,load,save,sha,check_sources
from fugue.validate import inputs,compare

def compare_references():
    paper=REPO/'Fugue-paper'
    specs=[('kvchime/Fugue-asplos-results/summary.csv','KVChime-experiment-3-software-reuse-F0-F4/data/summary.csv',['case_id','variant']),
        ('model-sweep/Fugue-asplos-results/sweep.csv','KVChime-experiment-1-prefill-boundary/data/raw-sweep.csv',['model','q','cached','link_GBps']),
        ('kvchime/Fugue-asplos-results/selection.csv','KVChime-experiment-5-device-selection/data/selection.csv',['model','q','cached']),
        ('kvchime/Fugue-asplos-results/shared-mq.csv','KVChime-experiment-4-shared-query-MQ/data/shared-mq.csv',['model','agents','nominal_shared_fraction','mode'])]
    if not all((paper/e).exists() for a,e,k in specs):return []
    return [compare(RUN/a,paper/e,k) for a,e,k in specs]

def verify():
    check_sources();input_report=inputs();paper=RUN/'paper'
    index=load(paper/'figure-index.json')
    assert Counter(r['experiment'] for r in index)=={1:4,2:2,3:5,4:2,5:1}
    indexed={r['path'] for r in index}
    assert {str(p.parent.relative_to(paper)) for p in paper.glob('KVChime-experiment-*/paper/*/plot-config.json')}==indexed
    assert not any('energy' in r['response_metric'].lower() or 'area' in r['response_metric'].lower() for r in index)
    def one(record):
        d=paper/record['path'];cfg=load(d/'plot-config.json');prov=load(d/'provenance.json')
        expected_axes=6 if record['experiment']==1 else 1
        if record['experiment']==1:
            assert cfg['kind']=='crossover_six_panel'
            assert [(p['panel'],p['mode'],p['cached'],p['link_GBps']) for p in cfg['panels']]==[
                ('a','plain',0,300),('b','plain',1024,300),('c','mq',1024,450),
                ('d','mq',0,300),('e','mq',1024,300),('f','mq',1024,32)]
        else:assert cfg['kind']=='grouped'
        for name in ['raw-data.csv','raw-profiles.csv','plot.py','models.json','figure.pdf','figure.png','SHA256SUMS.txt']:assert (d/name).is_file(),(d,name)
        for line in (d/'SHA256SUMS.txt').read_text().splitlines():
            digest,name=line.split('  ',1);assert sha(d/name)==digest,(d,name)
        src=RUN/prov['source_table'];assert sha(src)==prov['source_sha256']
        with src.open() as f:source={json.dumps(r,sort_keys=True) for r in csv.DictReader(f)}
        with (d/'raw-data.csv').open() as f:assert all(json.dumps(r,sort_keys=True) in source for r in csv.DictReader(f))
        with tempfile.TemporaryDirectory(prefix='KVChime-independent-figure-') as tmp:
            t=Path(tmp)
            for name in ['raw-data.csv','plot-config.json','plot.py']:shutil.copy2(d/name,t/name)
            env=dict(os.environ,MPLCONFIGDIR=str(t/'mpl'),PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            p=subprocess.run([sys.executable,str(t/'plot.py')],cwd=t,env=env,capture_output=True,text=True)
            assert p.returncode==0,p.stderr
            assert sha(t/'figure.png')==sha(d/'figure.png'),record
            rendered=load(t/'render-check.json')
            assert rendered['axes']==expected_axes
            assert rendered==load(d/'render-check.json')
        return dict(path=record['path'],png_sha256=sha(d/'figure.png'),raw_rows_match_simulation=True,
            independent_PNG_identical=True,axes=expected_axes,single_response_axis=expected_axes==1,
            experiment1_six_panel_layout=record['experiment']==1)
    with ThreadPoolExecutor(max_workers=min(JOBS,4)) as pool:figures=list(pool.map(one,index))
    report=dict(published_reference_comparisons=compare_references(),inputs=input_report,native_sources_unchanged=True,figures=len(figures),figure_checks=figures,
        experiment1_figures=4,experiment1_panels_per_model=6,later_single_axis_figures=10,
        final_figures_only_performance_and_capacity=True,
        scope='Every figure independently redrawn in an external temporary directory with only CSV, config, and plot.py; numeric operands are not evaluated by this timing simulator.')
    save(RUN/'KVChime-verification.json',report);print(json.dumps({k:v for k,v in report.items() if k!='figure_checks'},indent=2))
if __name__=='__main__':verify()
