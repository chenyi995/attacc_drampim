#!/usr/bin/env python3
"""Verify or provision the paper's frozen, token-free simulator inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'input-provenance.json').read_text())
    if manifest.get('schema') != 'kvchime-ae-input-provenance-v1':
        raise ValueError('Unsupported input manifest schema')
    for record in manifest['packaged_files']:
        path = directory / record['path']
        data = path.read_bytes()
        if len(data) != record['size_bytes'] or sha(data) != record['sha256']:
            raise ValueError('Frozen input file differs: ' + record['path'])
    selected = json.loads((directory / 'selected.json').read_text())
    catalog_rows = [json.loads(line) for line in (directory / 'catalog.jsonl').read_text().splitlines()]
    catalog = {row['chunk_id']: row for row in catalog_rows}
    if len(catalog) != len(catalog_rows):
        raise ValueError('Duplicate catalog object identity')
    if any(set(row) != {'chunk_id', 'tokens'} or type(row['tokens']) is not int or row['tokens'] <= 0
           for row in catalog_rows):
        raise ValueError('Invalid token-free catalog row')
    scope = json.loads((directory / 'scope.json').read_text())
    if [item['case']['id'] for item in selected] != scope['workload_order']:
        raise ValueError('Selected workload order differs from the paper scope')
    receipts = {row['workload_id']: row for row in manifest['cases']}
    required = set()
    for item in selected:
        case = item['case']
        receipt = receipts[case['id']]
        if sha(canonical(case)) != receipt['packaged_case_canonical_sha256']:
            raise ValueError('Frozen case differs: ' + case['id'])
        execution = {key: value for key, value in case.items() if key != 'provenance'}
        if sha(canonical(execution)) != receipt['execution_fields_canonical_sha256']:
            raise ValueError('Execution input differs from its source case: ' + case['id'])
        if case['batch'] != len(case['members']) or case['output'] < 2:
            raise ValueError('Invalid reader count or output horizon')
        for index, member in enumerate(case['members']):
            positions = member['recomputed_indices']
            required.update(member['chunk_ids'])
            if (positions != sorted(set(positions)) or len(positions) != case['q']
                    or any(type(p) is not int or not 0 <= p < case['n'] for p in positions)):
                raise ValueError('Invalid selected positions: ' + case['id'])
            if sha(canonical(positions)) != receipt['selected_positions_sha256'][index]:
                raise ValueError('Selected positions differ from the source: ' + case['id'])
            if sum(catalog[key]['tokens'] for key in member['chunk_ids']) != case['n']:
                raise ValueError('Prompt object lengths do not sum to N: ' + case['id'])
        required.update(case['pool_ids'])
        if sum(catalog[key]['tokens'] for key in set(case['pool_ids'])) != case['pool_tokens']:
            raise ValueError('Warm-pool token count differs: ' + case['id'])
    if required != set(catalog):
        raise ValueError('Catalog does not contain exactly the required object identities')
    return {'workloads': len(selected), 'models': len(scope['model_order']),
            'catalog_objects': len(catalog), 'neural_inference': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path,
                        help='Copy verified inputs here; existing different files are never overwritten')
    args = parser.parse_args()
    result = verify(args.source)
    if args.output is not None:
        names = ['selected.json', 'catalog.jsonl', 'scope.json', 'source-downloads.json', 'input-provenance.json']
        for name in names:
            target = args.output / name
            if target.exists() and target.read_bytes() != (args.source / name).read_bytes():
                raise FileExistsError('Refusing to replace different input file: ' + str(target))
        args.output.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = args.output / name
            if not target.exists():
                shutil.copyfile(args.source / name, target)
        result = verify(args.output)
        result['input_directory'] = str(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
