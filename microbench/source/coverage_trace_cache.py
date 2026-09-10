from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
"""Exact, process-shared cache for completed native Ramulator command replay.

Only byte-identical traces, resident lists, effective YAML and implementation
identities share a result. Logical views are separate: their private-K timing
is recomputed from the recorded commands on every call. No token-length bins,
interpolation, or synthetic timing results are used.
"""


import argparse


from collections import Counter, OrderedDict


from concurrent.futures import ThreadPoolExecutor


from contextlib import contextmanager


import fcntl


import hashlib


import json


import os


from pathlib import Path


import shutil


import subprocess


import sys


import tempfile


import threading


import uuid


import yaml


sys.dont_write_bytecode = True


from fugue import kvchime_traces as native


from fugue.attention import TCK, expand, interval


from fugue.runtime import JOBS, REPO


from src.ramulator_wrapper import Ramulator


SCHEMA = 'exact-command-cache-v1'


class CacheIntegrityError(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    os.replace(temporary, path)


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def normalized_yaml(text):
    """Exclude only known input/output locations, never arbitrary parameters."""
    config = yaml.safe_load(text)
    config['Frontend']['path'] = '<IDENTICAL_INPUT_TRACE_BYTES>'
    recorders = []
    for item in config['MemorySystem']['Controller']['plugins']:
        plugin = item['ControllerPlugin']
        if plugin['impl'] == 'HBM3TraceRecorder':
            recorders.append(plugin)
            plugin['path'] = '<RECORDED_COMMAND_OUTPUT>'
    if len(recorders) != 1:
        raise CacheIntegrityError('Expected one native command recorder')
    def reject_unhandled_paths(value):
        if isinstance(value, dict):
            for item in value.values(): reject_unhandled_paths(item)
        elif isinstance(value, list):
            for item in value: reject_unhandled_paths(item)
        elif isinstance(value, str) and Path(value).is_absolute():
            raise CacheIntegrityError('Unrecognized absolute YAML parameter: '+value)
    reject_unhandled_paths(config)
    return config


def command_events(folder):
    """Stream the exact recorder output, without retaining key-event arrays."""
    for path in sorted(Path(folder).glob('commands.ch*')):
        with path.open() as stream:
            for line in stream:
                fields = [value.strip() for value in line.split(',')]
                if len(fields) != 9:
                    raise CacheIntegrityError('Malformed native command event: '+str(path))
                yield fields


def first_private_k(folder, metadata):
    private = [metadata['physical_layout'][obj['id']]
               for obj in metadata['objects'] if obj.get('private', False)]
    first = None
    for fields in command_events(folder):
        if fields[1] != 'MACAB': continue
        # Exact address reconstruction used by the current source TraceBank.
        local = (int(fields[7])*32+int(fields[8]))*32
        if any(obj['K_base'] <= local < obj['K_base']+obj['span_bytes'] for obj in private):
            clock = int(fields[0])
            first = clock if first is None else min(first, clock)
    return 0. if first is None else first*TCK/1000


class ExactTraceBank(native.TraceBank):
    """Drop-in TraceBank with verified disk cache and bounded base-trace LRU.

    ``root`` remains the caller's logical profile directory. Physical command
    recordings live in ``cache_root`` and are named explicitly in every local
    cache-reference.json and timing.json. Native key events can be streamed
    with iter_key_events(row); the active Model does not consume that field.
    """
    def __init__(self, root, runtime, model, cache_root=None, base_cache_bytes=64*1024*1024):
        Path(root).mkdir(parents=True, exist_ok=True)
        super().__init__(root, runtime, model)
        self.cache_root = Path(cache_root or os.environ.get('KVCHIME_EXACT_TRACE_CACHE',
                              ROOT/'archive/coverage-expansion/exact-trace-cache')).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.base_cache_bytes = base_cache_bytes
        self.bases = OrderedDict()
        self._base_sizes = {}; self._base_lock = threading.RLock()
        self._append_lock = threading.Lock()
        self.implementation = self._implementation_identity()
        self.cache_stats = Counter()

    def _implementation_identity(self):
        stage = Path(REPO).resolve()
        lock_path = stage/'artifact/native-source-lock.json'
        lock = json.loads(lock_path.read_text())
        for name, expected in lock.items():
            if digest(stage/name) != expected:
                raise CacheIntegrityError('Locked native source changed: '+name)
        native_sources = {}
        for directory in ('src', 'pim_ramulator_src', 'vendor'):
            for path in sorted((stage/directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    native_sources[str(path.relative_to(stage))] = digest(path)
        wrappers = {str(path.relative_to(stage)): digest(path)
                    for path in sorted((stage/'fugue').glob('*.py'))}
        binary = self.runtime/'ramulator2'; library = self.runtime/'libramulator.so'
        loader = subprocess.run(['ldd', str(binary)], capture_output=True, text=True, check=True)
        loaded = [line.split('=>', 1)[1].split()[0] for line in loader.stdout.splitlines()
                  if 'libramulator.so =>' in line]
        if len(loaded) != 1 or Path(loaded[0]).resolve() != library.resolve():
            raise CacheIntegrityError('Runtime does not load the fingerprinted libramulator.so')
        return dict(schema=SCHEMA, cache_script_sha256=digest(__file__),
                    native_lock_sha256=digest(lock_path), native_sources=native_sources,
                    stage_wrappers=wrappers, binary_sha256=digest(binary),
                    library_sha256=digest(library), python_version=sys.version,
                    python_binary_sha256=digest(Path(sys.executable).resolve()))

    def _yaml(self, folder, residents):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'input.yaml'
        Ramulator(self.model, str(folder)).make_yaml_file(str(path), 'input', True)
        text = path.read_text().replace('      preset: HBM3_5.2Gbps\n',
            f'      preset: HBM3_5.2Gbps\n      nCCDAB: {interval(max(residents, default=1))}\n')
        text = text.replace('    plugins:\n',
            f'    plugins:\n      - ControllerPlugin:\n          impl: HBM3TraceRecorder\n          path: {folder}/commands\n')
        path.write_text(text)
        return normalized_yaml(text)

    def _manifest(self, folder, identity, kind):
        files = {str(path.relative_to(folder)): dict(sha256=digest(path), bytes=path.stat().st_size)
                 for path in sorted(folder.rglob('*')) if path.is_file() and path.name != 'complete.json'}
        record = dict(schema=SCHEMA, complete=True, kind=kind, identity=identity, files=files)
        save(folder/'complete.json', record)
        return record

    def _verify(self, folder, identity, kind):
        marker = folder/'complete.json'
        if not marker.exists(): return None
        record = json.loads(marker.read_text())
        if record.get('complete') is not True or record.get('identity') != identity or record.get('kind') != kind:
            raise CacheIntegrityError('Cache completion/identity mismatch: '+str(folder))
        actual = {str(path.relative_to(folder)) for path in folder.rglob('*')
                  if path.is_file() and path.name != 'complete.json'}
        if actual != set(record['files']):
            raise CacheIntegrityError('Cache artifact set changed: '+str(folder))
        for name, expected in record['files'].items():
            path = folder/name
            if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                raise CacheIntegrityError('Cache artifact hash mismatch: '+str(path))
        return record

    def _fresh_folder(self, folder):
        # Incomplete attempts are retained for diagnosis, never treated as hits.
        if folder.exists():
            quarantine = self.cache_root/'incomplete'
            quarantine.mkdir(exist_ok=True)
            folder.rename(quarantine/(folder.name+'-'+uuid.uuid4().hex))
        folder.mkdir(parents=True)

    def base(self, n, h, dhead=128, dbyte=2):
        key = (n, h, dhead, dbyte)
        with self._base_lock:
            if key in self.bases:
                self.bases.move_to_end(key)
                return self.bases[key]
        generator = Path(REPO)/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py'
        arguments = ['-dh', str(dhead), '-nh', str(h), '-l', str(n),
                     '-maxl', str(max(4096, n)), '-db', str(dbyte)]
        identity = dict(implementation=self.implementation, generator_sha256=digest(generator),
                        generator_arguments=arguments)
        identifier = hashlib.sha256(encoded(identity)).hexdigest()
        folder = self.cache_root/'bases'/identifier
        name = f'native-D{dhead}-B{dbyte}-N{n}-H{h}'
        with file_lock(self.cache_root/'locks'/('base-'+identifier+'.lock')):
            prior = self._verify(folder, identity, 'native-generator')
            hit = prior is not None
            if prior is None:
                self._fresh_folder(folder)
                builder = native.TraceBank(folder, self.runtime, self.model)
                native.TraceBank.base(builder, n, h, dhead, dbyte)
                generator_record = json.loads((folder/name/'generator.json').read_text())
                expected = [sys.executable, str(generator), *arguments, '-o', str(folder/name/'input.trace')]
                if generator_record['returncode'] != 0 or generator_record['command'] != expected:
                    raise CacheIntegrityError('Unexpected native generator command')
                self._manifest(folder, identity, 'native-generator')
            path = folder/name/'input.trace'
            lines = path.read_text().splitlines()
        save(self.root/'base-references'/(name+'.json'), dict(cache_hit=hit, canonical_path=str(path),
             trace_sha256=digest(path), generator_identity=identity))
        size = sum(sys.getsizeof(line) for line in lines)+sys.getsizeof(lines)
        with self._base_lock:
            if size <= self.base_cache_bytes:
                self.bases[key] = lines; self._base_sizes[key] = size
                self.bases.move_to_end(key)
                while sum(self._base_sizes.values()) > self.base_cache_bytes:
                    oldest, _ = self.bases.popitem(last=False); self._base_sizes.pop(oldest)
        return lines

    def prepare(self, keys, dhead=128, dbyte=2):
        # Unlike the inherited method, workers do not require every base trace
        # to remain in self.bases simultaneously, so eviction is safe.
        keys = sorted(set((n, h, r, dhead, dbyte) for n, h, r in keys)-self.profiles.keys())
        def one(key):
            n, h, r, d, b = key
            lines = expand(self.base(n, h, d, b), r)
            row = self.run(f'scan-D{d}-B{b}-N{n}-H{h}-Q{r}', lines,
                           [r]*sum(line.startswith('PIM_MAC_AB') for line in lines))
            row.update(dhead=d, dbyte=b, layout_revision=native.LAYOUT_REVISION)
            return key, row
        with ThreadPoolExecutor(max_workers=JOBS) as pool:
            for key, row in pool.map(one, keys): self.profiles[key] = row

    def run(self, name, lines, residents, metadata=None):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Profile name must be one path component')
        local = self.root/name
        trace_bytes = ('\n'.join(lines)+'\n').encode()
        counts = Counter(line.split()[0] for line in lines)
        if len(residents) != counts['PIM_MAC_AB'] or any(r < 1 for r in residents):
            raise CacheIntegrityError('Resident list does not match MAC commands')
        config = self._yaml(local, residents)
        identity = dict(implementation=self.implementation,
                        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
                        trace_bytes=len(trace_bytes), residents=list(residents), effective_yaml=config)
        identifier = hashlib.sha256(encoded(identity)).hexdigest()
        canonical = self.cache_root/'profiles'/identifier
        with file_lock(self.cache_root/'locks'/('profile-'+identifier+'.lock')):
            prior = self._verify(canonical, identity, 'ramulator')
            hit = prior is not None
            if prior is None:
                self._fresh_folder(canonical)
                builder = native.TraceBank(canonical.parent, self.runtime, self.model)
                # Neutral metadata suppresses the unused in-memory key_events
                # list in the unchanged native run; no command or timing changes.
                neutral = dict(objects=[], physical_layout={},
                               scope='Canonical physical replay; logical sidecars are caller-specific.')
                physical = native.TraceBank.run(builder, identifier, lines, residents, neutral)
                physical.pop('first_private_k_us', None)
                physical.pop('key_events', None)
                builder.native_commands.clear()
                if normalized_yaml((canonical/'input.yaml').read_text()) != config:
                    raise CacheIntegrityError('Native run YAML differs from cache identity')
                if (canonical/'input.trace').read_bytes() != trace_bytes:
                    raise CacheIntegrityError('Native run trace differs from cache identity')
                save(canonical/'physical-timing.json', physical)
                save(canonical/'residents.json', residents)
                prior = self._manifest(canonical, identity, 'ramulator')
            physical = json.loads((canonical/'physical-timing.json').read_text())
            command = json.loads((canonical/'command.json').read_text())
            emitted = sum(fields[1] == 'MACAB' for fields in command_events(canonical))
            cycles = [int(line.split()[-1]) for line in (canonical/'stdout.log').read_text().splitlines()
                      if 'memory_system_cycles:' in line]
            if (command['returncode'] != 0 or command['counts'] != dict(counts) or
                emitted != counts['PIM_MAC_AB'] or cycles != [physical['cycles']] or physical['cycles'] <= 0 or
                physical['trace_sha256'] != identity['trace_sha256']):
                raise CacheIntegrityError('Cached native statistics do not match input/recording')
        row = dict(physical, profile=name, path=str(local), cache_hit=hit,
                   cache_key=identifier, canonical_profile=identifier, canonical_path=str(canonical),
                   event_stream_path=str(canonical), key_events_in_memory=False)
        metadata_hash = hashlib.sha256(encoded(metadata)).hexdigest()
        if metadata is not None:
            row['first_private_k_us'] = first_private_k(canonical, metadata)
            save(local/'logical-view.json', metadata)
        row['logical_metadata_sha256'] = metadata_hash
        reference = dict(schema=SCHEMA, requested_profile=name, cache_hit=hit,
                         canonical_profile=identifier, canonical_path=str(canonical),
                         complete_manifest_sha256=digest(canonical/'complete.json'),
                         trace_sha256=identity['trace_sha256'], logical_metadata_sha256=metadata_hash,
                         physical_statistics=dict(cycles=physical['cycles'], mac_commands=emitted,
                             represented_column_MACs=sum(residents)),
                         checks=dict(completed_native_run=True, artifact_hashes_verified=True,
                             exact_trace_and_residents=True, effective_yaml_equal=True,
                             private_k_recomputed_from_recorded_commands=metadata is not None))
        save(local/'cache-reference.json', reference)
        save(local/'timing.json', row)
        with self._append_lock:
            self.native_commands.append(row)
            self.cache_stats['hit' if hit else 'miss'] += 1
        return row

    @staticmethod
    def iter_key_events(row):
        for fields in command_events(row['canonical_path']):
            if fields[1] == 'MACAB' and int(fields[7]) < 8192:
                yield [int(fields[7])*32+int(fields[8]), int(fields[0])]

