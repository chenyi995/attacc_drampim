#!/usr/bin/env python3
"""Render the three KVChime flowcharts with Graphviz; no experiments."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
for name in ('tlb_flow', 'di_map_flow', 'tlb_di_overview'):
    for extension in ('svg', 'pdf'):
        output = ROOT / f'{name}.{extension}'
        subprocess.run(['dot', f'-T{extension}', str(ROOT / f'{name}.dot'),
                        '-o', str(output)], check=True)
        print(output)
