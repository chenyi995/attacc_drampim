#!/usr/bin/env python3
"""Optionally fetch the exact upstream source files used to derive input shapes.

This fetches source evidence, never installs or executes source repositories,
tokenizes text, downloads model weights, or runs inference or simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def check(path, record):
    return (path.is_file() and path.stat().st_size == record['size_bytes']
            and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().with_name('source-downloads.json'))
    parser.add_argument('--output', type=Path, help='Local source-evidence directory; keep outside Git')
    parser.add_argument('--list', action='store_true', help='Print pinned files without downloading')
    parser.add_argument('--check-only', action='store_true', help='Verify an existing source directory offline')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.list:
        for record in manifest['files']:
            print(f"{record['size_bytes']:>10}  {record['path']}")
        print(f"Total: {manifest['total_size_bytes']} bytes; no model/tokenizer download")
        return
    if args.output is None:
        parser.error('--output is required unless --list is used')
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source-manifest destination')
        target = args.output / relative
        if target.exists():
            if not check(target, record):
                raise ValueError('Existing source differs; refusing overwrite: ' + str(target))
            continue
        if args.check_only:
            raise FileNotFoundError('Source file missing: ' + str(target))
        request = urllib.request.Request(record['url'], headers={'User-Agent': 'KVChime-AE-source-fetch/1'})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(record['size_bytes'] + 1)
        if len(data) != record['size_bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Downloaded source differs from the frozen bytes: ' + record['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='source-', delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        temporary_path.replace(target)
        print('Verified ' + record['path'])
    print(f"Verified {len(manifest['files'])} upstream files ({manifest['total_size_bytes']} bytes)")


if __name__ == '__main__':
    main()
