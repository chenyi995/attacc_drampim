"""Single reproducibility entry point; run from any checkout name/location."""
import argparse,datetime,importlib.metadata,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from fugue.runtime import REPO,check_sources,sha,load,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['doctor','build','run','plot','verify','all','check-inputs','kvchime','all-models','all-models-plot','all-models-verify'])
    p.add_argument('--output',type=Path,default=REPO/'output/Fugue-asplos-reproduce')
    p.add_argument('--jobs',type=int,default=8,help='Concurrent processes, 1..24 (8 default); <= 500 GB aggregate address-space limit')
    p.add_argument('--experiments',default='all',help='all, or comma-separated 1..5; 1/2 and 4/5 share runs')
    p.add_argument('--from-paper',action='store_true',help='Plot current checked-in tables without re-running simulation')
    p.add_argument('--resume',action='store_true',help='Skip completed stages after verifying code/input identity')
    args=p.parse_args()
    if not 1<=args.jobs<=24:p.error('--jobs must be 1..24')
    requested={1,2,3,4,5} if args.experiments=='all' else {int(x) for x in args.experiments.split(',')}
    if not requested or not requested<={1,2,3,4,5}:p.error('invalid --experiments')
    output=args.output.resolve();os.environ['FUGUE_OUTPUT']=str(output);os.environ['FUGUE_JOBS']=str(args.jobs)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    source_identity={str(q.relative_to(REPO)):sha(q) for folder in ['fugue','src','pim_ramulator_src','artifact/inputs','vendor'] for q in (REPO/folder).rglob('*') if q.is_file() and '__pycache__' not in str(q)}
    def execute(module,stage):
        output.mkdir(parents=True,exist_ok=True);logs=output/'logs';logs.mkdir(exist_ok=True)
        statepath=output/(stage+'-stage.json')
        settings=({'compiler':os.environ.get('CXX','g++')} if stage=='build' else {'experiments':sorted(requested),'from_paper':args.from_paper} if stage in ['plot','verify'] else {})
        if statepath.exists():
            old=load(statepath)
            if args.resume and old.get('returncode')==0 and old['source_identity']==source_identity and old.get('settings')==settings:
                print('RESUME',stage,flush=True);return
            if stage not in ['plot','verify']:raise RuntimeError(f'{stage} already attempted. Use --resume for identical completed stages, or a new --output directory. Failed runs are retained.')
        command=[sys.executable,'-m',module]
        state=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,cwd=str(REPO),source_identity=source_identity,settings=settings)
        save(statepath,state)
        print('RUN',stage,'log:',logs/(stage+'.log'),flush=True)
        with (logs/(stage+'.log')).open('a') as log:
            result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        state.update(returncode=result.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(statepath,state)
        if result.returncode:
            print((logs/(stage+'.log')).read_text()[-5000:],file=sys.stderr);raise SystemExit(result.returncode)
        print('PASS',stage,flush=True)
    if args.command in ['doctor','check-inputs']:
        check_sources()
        from fugue.validate import inputs
        detail=inputs()
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','matplotlib']}
        compiler=os.environ.get('CXX','g++')
        print(json.dumps(dict(repo=str(REPO),python=sys.version,dependencies=versions,compiler=compiler,inputs=detail,
            cmake=shutil.which('cmake'),patch=shutil.which('patch'),runtime='CPU simulation; no CUDA/GPU/LLM weights needed'),indent=2))
        if args.command=='doctor':
            if sys.version_info<(3,10):raise SystemExit('Python >=3.10 required')
            for executable in ['cmake','patch',compiler]:
                if shutil.which(executable) is None:raise SystemExit('Missing tool: '+executable)
            version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0];print(version)
            with tempfile.TemporaryDirectory(prefix='Fugue-asplos-cxx-check-') as tmp:
                probe=Path(tmp)/'probe.cpp';probe.write_text('#include <concepts>\nstatic_assert(std::integral<int>); int main() { return 0; }\n')
                result=subprocess.run([compiler,'-std=c++20',str(probe),'-o',str(Path(tmp)/'probe')],capture_output=True,text=True)
                if result.returncode:raise SystemExit('Selected compiler lacks required C++20 support. Set CXX to a suitable compiler.\n'+result.stderr)
            print('C++20 compiler probe passed')
        return
    if args.command in ['all-models','kvchime']:
        execute('fugue.build','build')
        execute('fugue.kvchime_model_tests','model-tests')
        execute('fugue.kvchime_tests','shared-tests')
        execute('fugue.kvchime_sweep','model-sweep')
        execute('fugue.kvchime','model-workloads')
        execute('fugue.kvchime_paper','model-paper')
        execute('fugue.kvchime_verify','verify')
        return
    if args.command=='all-models-verify':
        execute('fugue.kvchime_verify','verify')
        return
    if args.command=='all-models-plot':
        execute('fugue.kvchime_paper','plot')
        return
    if args.command in ['build','all']:execute('fugue.build','build')
    if args.command in ['run','all']:
        if not (output/'build/Fugue-asplos-build.json').exists():raise SystemExit('Build first: python3 -m fugue build --output '+str(output))
        if requested&{1,2}:execute('fugue.sweep','experiment12')
        if 3 in requested:execute('fugue.cacheblend','experiment3')
        if requested&{4,5}:execute('fugue.epic','experiment45')
    if args.command in ['plot','all']:
        os.environ['FUGUE_PLOT_FROM_PAPER']='1' if args.from_paper else '0'
        os.environ['FUGUE_EXPERIMENTS']=','.join(map(str,sorted(requested)))
        execute('fugue.plot','plot')
    if args.command in ['verify','all']:
        os.environ['FUGUE_EXPERIMENTS']=','.join(map(str,sorted(requested)))
        execute('fugue.validate','verify')
if __name__=='__main__':main()
