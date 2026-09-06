#!/usr/bin/env python3
"""Validate this audit's artifacts and unchanged implementation provenance."""
import ast, hashlib, json, re, subprocess
from pathlib import Path

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[1]
missing=[]
for p in (AUDIT/'README.md',AUDIT/'CALCULATIONS.md',ROOT/'docs/sessions/2026-09-06-w1-rotating-diff-audit.md'):
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        target=target.split('#',1)[0]
        if not target or '://' in target:continue
        q=Path(target) if target.startswith('/') else p.parent/target
        if not q.exists():missing.append([str(p.relative_to(ROOT)),target])
for p in (AUDIT/'w1_handcheck.py',AUDIT/'finalize_report.py',AUDIT/'validate.py'):
    ast.parse(p.read_text())
checks=json.loads((AUDIT/'evidence/checks.json').read_text())
prov=json.loads((AUDIT/'evidence/provenance.json').read_text())
unchanged={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in prov['sources'].items()
           if p.startswith(('src/','pim_ramulator_src/','workload/probe/gen_'))}
diff=subprocess.run(['git','diff','--check'],cwd=ROOT,text=True,capture_output=True)
data={'missing_local_links':missing,'audit_scripts_parse':True,'implementation_matches_calculation':unchanged,
      'static_checks':checks,'git_diff_check_exit':diff.returncode,'git_diff_check':diff.stdout+diff.stderr}
(AUDIT/'validation.json').write_text(json.dumps(data,indent=2)+'\n')
assert not missing and all(unchanged.values()) and diff.returncode==0
print(json.dumps({'missing_local_links':missing,'source_matches':all(unchanged.values()),
                  'QK_address_lists':checks['actual_QK_generator_lists_verified'],
                  'w1_cases':len([k for k in checks if k.startswith('W1')])}))
