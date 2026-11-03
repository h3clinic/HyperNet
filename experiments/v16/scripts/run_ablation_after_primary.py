"""Sequence the already approved equal-budget ablation after primary completion.

This orchestration helper does not change training objectives or source files.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
primary=ROOT/'training/primary_4096'
deadline=time.monotonic()+7200
print('Waiting for fixed4096-step primary completion before matched ablation.',flush=True)
while not (primary/'completion.json').exists():
    if (primary/'failure.json').exists():raise RuntimeError('Primary failed; ablation not launched')
    if time.monotonic()>deadline:raise TimeoutError('Primary did not finish within orchestration limit')
    time.sleep(5)
complete=json.loads((primary/'completion.json').read_text())
if complete['optimizer_steps']!=4096 or complete['config']['variant']!='primary' or complete['config']['smoke']:
    raise RuntimeError('Primary completion does not match admitted fixed budget')
for relative,expected in complete['sources'].items():
    if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
        raise RuntimeError(f'Frozen source changed: {relative}')
print('Primary completed; starting fixed4096-step no-relation-aux ablation.',flush=True)
subprocess.run([sys.executable,str(ROOT/'scripts/train_relations.py'),
                '--run','training/no_relation_aux_4096','--steps','4096',
                '--variant','no_relation_aux'],check=True)
