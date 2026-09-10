#!/usr/bin/env bash
set -euo pipefail
artifact_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$artifact_root"
if [[ ! -x .venv/bin/python3 ]]; then
    python3 -m venv .venv
fi
requirement_hash="$(sha256sum requirements.txt | cut -d ' ' -f 1)"
if [[ ! -f .venv/requirements.sha256 ]] || [[ "$(cat .venv/requirements.sha256)" != "$requirement_hash" ]]; then
    .venv/bin/python3 -m pip install -r requirements.txt
    printf '%s\n' "$requirement_hash" > .venv/requirements.sha256
fi
exec .venv/bin/python3 reproduce.py "$@"
