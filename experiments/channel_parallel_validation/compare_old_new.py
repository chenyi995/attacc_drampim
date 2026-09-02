#!/usr/bin/env python3
"""Print matched old/new makespans for the channel-parallel validation."""
import json
from pathlib import Path

OLD = Path('/home/cw636/chenyi/attacc_drampim/output/sweep_models_20260830-163226/LLAMA-7B')
NEW = Path('/home/xw338/attacc/attacc_drampim/output/channel_parallel_validation_20260902/LLAMA-7B')
CASES = ('baseline', 'broadcast', 'reduce')
RUNGS = ('A3', 'A4', 'A5', 'A6')

print('case,rung,old_s,new_s,delta_s,delta_pct')
for case in CASES:
    for rung in RUNGS:
        old_path = OLD / f'{case}_k8' / f'dag_{rung}.json'
        new_path = NEW / f'{case}_k8' / f'dag_{rung}.json'
        old = json.loads(old_path.read_text())['makespan_s']
        new = json.loads(new_path.read_text())['makespan_s']
        delta = new - old
        print(f'{case},{rung},{old:.6f},{new:.6f},{delta:.6f},{delta / old * 100:.2f}')
