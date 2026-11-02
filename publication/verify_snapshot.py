"""Verify exported model code against the published SHA-256 snapshot."""
from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
snapshot = json.loads((root/'publication/source_snapshot.json').read_text(encoding='utf-8'))
expected = snapshot['core_files']
actual = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
          for p in (root/'hypernet').rglob('*.py')}
if actual != expected:
    changed = sorted(k for k in set(actual)|set(expected) if actual.get(k) != expected.get(k))
    raise SystemExit('Snapshot mismatch: '+', '.join(changed))
print(f'Verified {len(actual)} package files against the published snapshot.')
