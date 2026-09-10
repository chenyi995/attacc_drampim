#!/usr/bin/env python3
"""Freshly build the pinned original AttAcc model, using local Git object caches."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import tarfile

from fugue.runtime import REPO,RUN,JOBS,limit,check_sources
ROOT=RUN/'build';ROOT.mkdir(parents=True)
RAM=ROOT/'Fugue-asplos-source/ramulator2'
RAM_REV='b7c70275f04126c647edb989270cc429776955d1'
limit()
lock=json.loads((REPO/'vendor/Fugue-asplos-dependencies.json').read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(name,dest):
    entry=lock[name];path=REPO/entry['archive']
    assert sha(path)==entry['sha256']
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path,'r:gz') as archive:
        for member in archive.getmembers():
            assert (dest/member.name).resolve().is_relative_to(dest.resolve())
            assert not member.issym() and not member.islnk()
        archive.extractall(dest)
    return entry['tar_sha256']
revision='c60005143a6b492d7ef83231723386478b59a506'
source_names=list(check_sources())+['set_pim_ramulator.sh']
state = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             original_commit=revision, repo=str(REPO), ramulator_commit=RAM_REV,
             bundled_source_manifest=lock, phase='prepare', commands=[],
             source_sha256={name: sha(REPO / name) for name in source_names},
             cpu_affinity=sorted(os.sched_getaffinity(0)), build_parallel_jobs=JOBS,
             per_process_address_limit_bytes=20_000_000_000,
             source_policy='Git archive of exact upstream commits; no local working-tree edits or old binaries copied.')
record = ROOT / 'Fugue-asplos-build.json'


def save():
    record.write_text(json.dumps(state, indent=2) + '\n')


save()
state['ramulator_archive_sha256'] = export('ramulator2',RAM)
# Reproduce the copy/patch steps of this clone's original installer without a reset.
installer = (REPO / 'set_pim_ramulator.sh').read_text()
assert RAM_REV in installer
for line in installer.splitlines():
    if not line.startswith('cp '):
        continue
    parts = shlex.split(line)
    recursive = '-r' in parts
    source, dest = [part for part in parts[1:] if part != '-r']
    target = RAM / Path(dest).relative_to('ramulator2') / Path(source).name
    if recursive:
        shutil.copytree(REPO / source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(REPO / source, target)
for patch in sorted((RAM / 'patches').glob('*.patch')):
    command = ['patch', '--batch', '-p1', '-i', str(patch)]
    result = subprocess.run(command, cwd=RAM, text=True, capture_output=True)
    state['commands'].append(dict(command=command, returncode=result.returncode,
                                  stdout=result.stdout, stderr=result.stderr))
    save()
    assert result.returncode == 0, result
state['dependencies'] = {}
for name, tag in [('yaml-cpp', 'yaml-cpp-0.7.0'), ('spdlog', 'v1.11.0'), ('argparse', 'v2.9')]:
    commit=lock[name]['commit']
    state['dependencies'][name] = dict(tag=tag, commit=commit,
        archive_sha256=export(name,RAM/'ext'/name))
state['ramulator_source_sha256'] = {str(p.relative_to(RAM)): sha(p)
    for p in sorted((RAM / 'src').rglob('*')) if p.is_file()}
state['compiler_compatibility'] = 'C++ compiler compatibility: -include cstdint supplies the uint64_t declaration missing from upstream transitive includes. No model source edits.'
build = ROOT / 'Fugue-asplos-build'
cmake = ['cmake', '-S', str(RAM), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
         '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DCMAKE_CXX_FLAGS=-include cstdint',
         '-DCMAKE_CXX_COMPILER='+os.environ.get('CXX','g++')]
for phase, command in [('configure', cmake), ('compile', ['cmake', '--build', str(build), '-j',str(JOBS)])]:
    state['phase'] = phase
    state['commands'].append(dict(command=command, phase=phase))
    save()
    with (ROOT / ('Fugue-asplos-' + phase + '.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    state['commands'][-1]['returncode'] = result.returncode
    save()
    assert result.returncode == 0, phase
for name, digest in state['source_sha256'].items():
    assert sha(REPO / name) == digest
for name, digest in state['ramulator_source_sha256'].items():
    assert sha(RAM / name) == digest
binary, library = build / 'ramulator2', RAM / 'libramulator.so'
state.update(phase='complete', binary=str(binary), binary_sha256=sha(binary),
             library=str(library), library_sha256=sha(library),
             source_unchanged_after_build=True,
             finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
print('Fresh original build complete:', binary, flush=True)
