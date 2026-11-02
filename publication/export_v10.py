"""Export the audited surface pilot without weights, generated pixels or local paths."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

PUBLIC = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sanitize(value, root):
    if isinstance(value, dict):
        return {sanitize(k, root): sanitize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, root) for v in value]
    if isinstance(value, str) and (re.match(r'^[A-Za-z]:[/\\]', value) or value.startswith('\\\\')):
        return Path(value).resolve().relative_to(root).as_posix()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--audit', default='experiments/surface_encoder_audit.json')
    args = parser.parse_args()
    root = args.root.resolve()
    audit = read(root/args.audit)
    if audit.get('valid') is not True:
        raise ValueError('Independent inference/metric audit must pass before publication')
    freeze_path = root/'experiments/surface_training_freeze.json'
    freeze = read(freeze_path)
    results = read(root/'results/surface_results.json')
    if audit['results_sha256'] != sha(root/'results/surface_results.json') or audit['freeze_sha256'] != sha(freeze_path):
        raise ValueError('Audit does not bind these final results and frozen sources')
    if not results['completed'] or not results['sources_unchanged']:
        raise ValueError('Incomplete or changed-source experiment')
    for name, expected in freeze['sources'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    for name, expected in audit['bound_features_and_checkpoints'].items():
        if sha(name) != expected:
            raise ValueError('Audited output changed')
    data_audit = read(root/'experiments/surface_data_audit.json')
    if data_audit['valid'] is not True or data_audit['manifest_sha256'] != sha(root/'data/surfaces_v10/manifest.json'):
        raise ValueError('Data audit no longer binds the manifest')
    previous = read(PUBLIC/'publication/source_snapshot.json')
    if previous['core_file_count'] != 67:
        raise ValueError('Expected unmodified v09 public snapshot')
    for name, expected in previous['core_files'].items():
        if sha(PUBLIC/name) != expected:
            raise ValueError('Previously published core file changed: '+name)
    dest = PUBLIC/'benchmarks/v0.10'
    if dest.exists():
        raise FileExistsError('Preserve previously published surface results')
    dest.mkdir(parents=True)
    shutil.copyfile(PUBLIC/'publication/source_snapshot.json', PUBLIC/'publication/source_snapshot_v0_9.json')
    shutil.copyfile(PUBLIC/'publication/validation.json', PUBLIC/'publication/validation_v0_9.json')
    files = ['hypernet/mathematical_surfaces.py', 'hypernet/surface_encoder.py',
             'scripts/prepare_surface_data.py', 'scripts/train_surface_encoder.py',
             'scripts/audit_surface_encoder.py', 'scripts/serve_surface_gallery.py',
             'tests/test_mathematical_surfaces.py', 'tests/test_surface_encoder.py',
             'tests/test_surface_dataset.py', 'tests/test_surface_audit.py', 'surface_gallery.html',
             'research/surface_primary_sources.json', 'research/surface_math.html',
             'research/mathematical_surface_notes.json']
    exported = {}
    for name in files:
        output = PUBLIC/name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, output)
        exported[name] = sha(output)
        if exported[name] != sha(root/name):
            raise ValueError('Export changed source bytes')
    # The local report embeds generated images. The public synthesis links instead
    # to the pixel-free public benchmark; its mathematical text is unchanged.
    synthesis = PUBLIC/'research/surface_math.html'
    synthesis.write_text(synthesis.read_text(encoding='utf-8').replace('../index.html', '../benchmarks/v0.10/README.md'), encoding='utf-8')
    exported['research/surface_math.html'] = sha(synthesis)
    gallery = PUBLIC/'surface_gallery.html'
    text = gallery.read_text(encoding='utf-8')
    text = re.sub(r'<details id="overview">.*?</details>', '', text)
    text = text.replace('/experiments/mathematical_surface_gallery.json', '/benchmarks/v0.10/data_audit.json')
    gallery.write_text(text, encoding='utf-8')
    exported['surface_gallery.html'] = sha(gallery)
    inputs = {'results/surface_results.json': 'aggregate_metrics.json',
              'experiments/surface_training_freeze.json': 'training_protocol.json',
              'experiments/surface_batch_plan.json': 'batch_plan.json',
              'data/surfaces_v10/plan.json': 'dataset_plan.json',
              'data/surfaces_v10/manifest.json': 'frame_manifest.json',
              'experiments/surface_data_audit.json': 'data_audit.json',
              args.audit: 'inference_metric_audit.json'}
    original_hashes = {}
    for source, target in inputs.items():
        write(dest/target, sanitize(read(root/source), root))
        original_hashes[source] = sha(root/source)
    for source in ('heldout_native128.npz', 'heldout_lowpass32.npz',
                   'heldout_random_native128.npz', 'heldout_random_lowpass32.npz'):
        shutil.copyfile(root/'results'/source, dest/source)
        original_hashes['results/'+source] = sha(root/'results'/source)
    core = {p.relative_to(PUBLIC).as_posix():sha(p) for p in sorted((PUBLIC/'hypernet').rglob('*.py'))}
    write(PUBLIC/'publication/source_snapshot.json', {
        'snapshot':'HyperNet v0.10.0 mathematical surface pilot', 'created_date':'2026-11-02',
        'core_file_count':len(core), 'core_files':core,
        'prior_snapshot':'publication/source_snapshot_v0_9.json', 'unchanged_prior_core_files':67,
        'experimental_freeze_sha256':sha(freeze_path), 'new_frozen_behavior_files':freeze['sources'],
        'additional_exported_files':exported,
        'new_renderer_and_encoder_byte_identical_to_frozen_experiment':True,
        'evaluation_status':'Known supervised synthetic families: native128 93.23%, matched lowpass32 93.45%; no natural-transfer claim.',
        'notes':['No generated pixels or checkpoint weights are distributed.',
                 'Held-out NPZ files contain derived features, logits and spectrum predictions only.',
                 'Public synthesis navigation is adapted; frozen behavior code is unchanged.']})
    write(dest/'export_provenance.json', {'original_artifact_sha256':original_hashes,
         'public_artifact_sha256':{p.name:sha(p) for p in sorted(dest.iterdir())},
         'sanitization':'Absolute experiment paths are replaced by logical paths relative to the experiment root; original hashes retained.',
         'pixels_exported':False, 'learned_parameters_exported':False})
    pyproject = PUBLIC/'pyproject.toml'
    pyproject.write_text(pyproject.read_text().replace('version = "0.9.0"', 'version = "0.10.0"'), encoding='utf-8')
    print(json.dumps({'core_files':len(core), 'unchanged_previous_files':67,
                      'new_source_files':len(files), 'benchmark':dest.relative_to(PUBLIC).as_posix()}, indent=2))


if __name__ == '__main__':
    main()
