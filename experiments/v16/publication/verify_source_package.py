"""Check byte-exact public source copies without private data or model inference."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    count = 0
    for relative, row in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError('Manifest path escapes package')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row['public_sha256'] or actual != row['original_sha256']:
            raise ValueError('Exact source hash mismatch: ' + relative)
        if path.stat().st_size != row['bytes'] or row['transformation'] != 'none':
            raise ValueError('Unexpected source size/transformation: ' + relative)
        count += 1
    artifact_count = 0
    artifact_path = ROOT / 'artifact_manifest.json'
    if artifact_path.exists():
        artifacts = json.loads(artifact_path.read_text(encoding='utf-8'))
        for relative, row in artifacts['files'].items():
            path = (ROOT / relative).resolve()
            if ROOT.resolve() not in path.parents:
                raise ValueError('Artifact path escapes package')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != row['public_sha256'] or path.stat().st_size != row['bytes']:
                raise ValueError('Artifact hash/size mismatch: ' + relative)
            if row['transformation'] == 'none' and actual != row['original_sha256']:
                raise ValueError('Untransformed artifact changed: ' + relative)
            artifact_count += 1
    print(json.dumps({'valid': True, 'exact_source_copies': count,
                      'completed_artifacts': artifact_count,
                      'status': manifest['status'], 'model_calls': 0,
                      'real_image_reads': 0}))


if __name__ == '__main__':
    main()
