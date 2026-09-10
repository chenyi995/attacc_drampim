#!/usr/bin/env python3
"""One command to build, simulate, check and plot the final KVChime experiments."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command, cwd, env):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def verify():
    manifest = ROOT/'artifact/files.sha256.json'
    locked = json.loads(manifest.read_text())
    for name, digest in locked.items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Artifact file has changed: ' + name)
    return sha(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New directory; existing directories are never overwritten.')
    parser.add_argument('--jobs', type=int, default=8, help='Build/trace workers, 1 through 24 (default 8).')
    parser.add_argument('--cxx', default=os.environ.get('CXX'), help='C++20 compiler; defaults to CXX or g++.')
    parser.add_argument('--check-only', action='store_true', help='Verify source and input identities without building or simulating.')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        parser.error('--jobs must be between 1 and 24')
    identity = verify()
    if args.check_only:
        print(json.dumps(dict(artifact_sha256=identity, files_verified=True, simulation_run=False)))
        return
    start = time.monotonic()
    output = (args.output or ROOT/'output'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(FUGUE_OUTPUT=str(output), FUGUE_JOBS=str(args.jobs), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               MPLBACKEND='Agg', MPLCONFIGDIR=str(output/'matplotlib'))
    compiler = args.cxx or shutil.which('g++')
    # RHEL distributions may ship a C++20 toolset alongside the older system compiler.
    if args.cxx is None:
        toolset = Path('/opt/rh/gcc-toolset-14/root/usr/bin/g++')
        if toolset.is_file():
            compiler = str(toolset)
    if not compiler:
        raise RuntimeError('A C++20 compiler is required; set CXX or --cxx.')
    env['CXX'] = compiler
    for name in ['cmake', 'make', 'patch']:
        if not shutil.which(name):
            raise RuntimeError('Missing build tool: '+name)
    python = sys.executable
    state = dict(complete=False, artifact_sha256=identity, output=str(output),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version, platform=platform.platform(), compiler=compiler, jobs=args.jobs,
        scope='Five models x five workloads, F0/F1/F2/F4; final crossover, frequency and area analysis; Readers 4 prefill decomposition.')
    def save():
        (output/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        state['phase']='build'; save()
        execute([python,'-m','fugue.build'], ROOT/'simulator',env)
        build=json.loads((output/'build/Fugue-asplos-build.json').read_text())
        runtime=output/'build/runtime';runtime.mkdir()
        for field,name in [('binary','ramulator2'),('library','libramulator.so')]:
            assert sha(build[field]) == build[field+'_sha256']
            shutil.copy2(build[field],runtime/name)
        state['phase']='requests';save()
        execute([python,'-m','ae.request','--simulator',ROOT/'simulator','--inputs',ROOT/'artifact/inputs',
            '--output',output/'requests','--runtime-built-dir',runtime,'--jobs',args.jobs,
            '--fig1a-output',output/'figure1a'],ROOT,env)
        state['phase']='request figures';save()
        execute([python,'-m','ae.results','--requests',output/'requests','--output',output/'request-figures'],ROOT,env)
        for experiment in ['crossover','frequency','area']:
            state['phase']=experiment;save()
            command=[python,ROOT/'microbench/run.py',experiment,'--output',output/experiment]
            if experiment in ['crossover','frequency']:
                command.extend(['--simulator',ROOT/'simulator'])
            if experiment=='crossover':
                command.extend(['--runtime',runtime,'--jobs',args.jobs])
            execute(command,ROOT,env)
        state.update(complete=True,phase='complete',wall_seconds=time.monotonic()-start)
        save()
    except BaseException as exc:
        state.update(error=repr(exc),wall_seconds=time.monotonic()-start)
        save()
        raise
    print('Reproduction passed. Fresh raw evidence and figures: '+str(output),flush=True)


if __name__=='__main__':
    main()
