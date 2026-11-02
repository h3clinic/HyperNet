"""Recompute published v10 ranking metrics from derived held-out features."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('surface_audit', ROOT/'scripts/audit_surface_encoder.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path, nargs='?', default=ROOT/'benchmarks/v0.10')
    args = parser.parse_args()
    root = args.benchmark
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    result = read('aggregate_metrics.json')
    records = [r for r in read('frame_manifest.json')['records'] if r['split'] == 'heldout']
    provenance = read('export_provenance.json')
    checked = 0
    for arm in ('native128', 'lowpass32'):
        for prefix, phase in (('heldout_', 'trained'), ('heldout_random_', 'random_initialization')):
            name = prefix+arm+'.npz'
            expected = provenance['original_artifact_sha256']['results/'+name]
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Derived features changed: '+name)
            with np.load(root/name, allow_pickle=False) as archive:
                if set(archive.files) != {'shape', 'detail', 'embedding', 'family_logits', 'spectrum'}:
                    raise ValueError('Unexpected feature arrays')
                for head in ('embedding', 'shape', 'detail'):
                    values = torch.from_numpy(archive[head].copy())
                    features = (values/torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-12)).numpy()
                    actual = audit.representation_metrics(features, records)
                    audit.compare(actual, result['arms'][arm][phase][head], arm+'/'+phase+'/'+head)
                    checked += 1
                actual = audit.fractional_accuracy(archive['family_logits'], [r['family_index'] for r in records])
                audit.compare(actual, result['arms'][arm][phase]['trained_family_classifier_accuracy'], arm+'/'+phase+'/classifier')
    print(json.dumps({'passed': True, 'feature_archives': 4,
          'representations_checked': checked, 'classifier_checks': 4, 'heldout_frames_per_archive':len(records),
          'scope':'Ranking and classifier arithmetic from derived features; inference, alternate-input replay and spectrum targets require regenerated data/checkpoints.'}, indent=2))


if __name__ == '__main__':
    main()
