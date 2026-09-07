from pathlib import Path
import csv,hashlib,json,os,resource,shutil,sys
REPO=Path(__file__).resolve().parents[1]
RUN=Path(os.environ.get('FUGUE_OUTPUT',str(REPO/'output/Fugue-asplos-reproduce'))).resolve()
JOBS=int(os.environ.get('FUGUE_JOBS','8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def load(p):return json.loads(Path(p).read_text())
def save(p,data):Path(p).write_text(json.dumps(data,indent=2,default=str)+'\n')
def check_sources():
    lock=load(REPO/'artifact/native-source-lock.json')
    for name,digest in lock.items():
        if sha(REPO/name)!=digest:raise RuntimeError(f'Native model differs from publication: {name}. Update the lock only for an explicitly new model revision.')
    return lock
def limit():
    if not 1<=JOBS<=24:raise ValueError('jobs must be in 1..24')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:JOBS])
    resource.setrlimit(resource.RLIMIT_AS,(20_000_000_000,20_000_000_000));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
def setup(root):
    limit();root.mkdir(parents=True)
    folders=[root/s for s in ['Fugue-asplos-raw','Fugue-asplos-results','Fugue-asplos-audit','Fugue-asplos-runtime']]
    for p in folders:p.mkdir()
    build=load(RUN/'build/Fugue-asplos-build.json')
    for source,name,key in [(build['binary'],'ramulator2','binary_sha256'),(build['library'],'libramulator.so','library_sha256')]:
        if sha(source)!=build[key]:raise RuntimeError('Build artifact changed: '+source)
        shutil.copy2(source,folders[3]/name)
    os.environ['LD_LIBRARY_PATH']=str(folders[3])+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    return folders
