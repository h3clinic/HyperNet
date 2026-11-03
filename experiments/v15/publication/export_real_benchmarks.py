"""Extend only v15 public staging with exact code and sanitized Grid evidence."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs
import hashlib, json, re, shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT.parents[1] / 'public/HyperNet/experiments/v15'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    manifest_path = PUBLIC / 'release_manifest.json'
    manifest = read(manifest_path)
    for record in manifest['files']:
        if sha(PUBLIC / record['path']) != record['sha256']:
            raise ValueError('Preserve externally changed staging file: ' + record['path'])
    if (PUBLIC / 'benchmarks/public_metadata_manifest.json').exists():
        raise FileExistsError('Preserve the earlier real-benchmark supplement')
    grid = read(ROOT / 'experiments/grid_protocol_freeze.json')
    gravity = read(ROOT / 'experiments/gravity_spy_protocol_freeze.json')
    for protocol in (grid, gravity):
        for group in ('behavior_files', 'checkpoint_files'):
            for name, expected in protocol[group].items():
                path = Path(name) if Path(name).is_absolute() else ROOT / name
                if sha(path) != expected:
                    raise ValueError('Frozen source/artifact changed: ' + name)
    result_path = ROOT / 'results/grid_final_test/grid_results.json'
    result = read(result_path)
    audit = read(ROOT / 'experiments/grid_results_independent_audit.json')
    if not result['completed'] or not audit['valid'] or audit['test_results_sha256'] != sha(result_path):
        raise ValueError('Require audited completed Grid result')

    records = {r['path']: r for r in manifest['files']}
    snapshots = []

    def register(target, source=None, preserved=False, transform=None):
        path = PUBLIC / target
        record = {'path': target, 'bytes': path.stat().st_size, 'sha256': sha(path),
                  'source_relative_path': source, 'source_bytes_preserved': preserved}
        if source is not None:
            record['original_source_sha256'] = sha(ROOT / source)
        if transform:
            record['transformation'] = transform
        records[target] = record

    def exact(source, target=None):
        target = target or source
        path = PUBLIC / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, path)
        if sha(path) != sha(ROOT / source):
            raise RuntimeError('Copy mismatch')
        register(target, source, True)

    modules = ['detail_encoder', 'detail_features', 'pattern_matcher', 'rare_event_metrics',
               'grid_anomaly', 'mvtec_grid', 'gravity_spy', 'gravity_spy_images',
               'gravity_spy_admission', 'gravity_spy_quality']
    scripts = ['prepare_mvtec_grid', 'evaluate_grid_v15', 'audit_grid_results_independent',
               'prepare_gravity_spy', 'audit_gravity_spy_quality', 'evaluate_gravity_spy_v15',
               'freeze_gravity_spy_v15', 'audit_gravity_scores', 'audit_gravity_calibration_independent',
               'profile_pattern_matcher', 'one_reference_patterns', 'recompute_public_grid_metrics']
    tests = ['test_gravity_spy', 'test_gravity_spy_images', 'test_gravity_spy_quality',
             'test_gravity_spy_admission', 'test_gravity_evaluator', 'test_grid_anomaly',
             'test_grid_evaluator', 'test_mvtec_grid_data', 'test_rare_event_metrics',
             'test_detail_features', 'test_pattern_matcher', 'pattern_matcher_scalar_reference']
    code = [f'hypernet/{name}.py' for name in modules] + [f'scripts/{name}.py' for name in scripts]
    code += [f'tests/{name}.py' for name in tests]
    code += ['experiments/audit_gravity_spy_prior_exposure.py', 'experiments/review_gravity_spy_metadata.py']
    for source in code:
        exact(source)
        if not source.startswith('tests/'):
            target = 'benchmarks/source_snapshot/' + source
            exact(source, target)
            snapshots.append({'source': source, 'snapshot': target, 'sha256': sha(ROOT / source),
                              'role': 'byte-identical source snapshot; original relative imports/paths retained'})

    metadata_sources = [
        'experiments/grid_protocol_freeze.json', 'experiments/grid_method_plan.json',
        'experiments/grid_scorer_independent_review.json', 'experiments/grid_results_independent_audit.json',
        'experiments/mvtec_grid_data_audit.json', 'experiments/grid_evidence_bundle.json',
        'data/mvtec_grid_v15/acquisition_plan.json', 'data/mvtec_grid_v15/archive_inventory.json',
        'data/mvtec_grid_v15/roster_freeze.json', 'data/mvtec_grid_v15/manifest.json',
        'data/mvtec_grid_v15/development_manifest.json', 'data/mvtec_grid_v15/test_admission.json',
        'results/grid_final_dev/grid_results.json', 'results/grid_final_test/grid_results.json']
    metadata_sources += [f'results/{stage}/{alias}_scores.jsonl'
                         for stage in ('grid_final_dev', 'grid_final_test')
                         for alias in ('learned_final', 'random', 'analytic_detail')]
    metadata = []
    prefixes = (str(ROOT.resolve()), ROOT.resolve().as_posix())

    def clean_string(value, changes):
        original = value
        for prefix in prefixes:
            value = value.replace(prefix, '${V15_ROOT}')
        if re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]', value):
            raise ValueError('Unmapped absolute local path in public metadata')
        if value.startswith(('https://', 'http://')):
            query = {key.lower() for key in parse_qs(urlsplit(value).query)}
            if query.intersection({'token', 'access_token', 'api_key', 'apikey', 'signature', 'sig', 'x-amz-signature'}):
                raise ValueError('Credential-like query string must not be published')
        if value != original:
            changes.append({'original_string_sha256': hashlib.sha256(original.encode()).hexdigest(),
                            'public_string': value})
        return value

    def sanitize(value, changes):
        if isinstance(value, str):
            return clean_string(value, changes)
        if isinstance(value, list):
            return [sanitize(item, changes) for item in value]
        if isinstance(value, dict):
            return {clean_string(key, changes): sanitize(item, changes) for key, item in value.items()}
        return value

    for source in metadata_sources:
        path = ROOT / source
        target = 'benchmarks/grid/provenance/' + source
        out = PUBLIC / target
        out.parent.mkdir(parents=True, exist_ok=True)
        changes = []
        if path.suffix == '.jsonl':
            values = [sanitize(json.loads(line), changes) for line in path.read_text().splitlines() if line]
            if changes:
                out.write_text(''.join(json.dumps(value, allow_nan=False) + '\n' for value in values), encoding='utf-8')
            else:
                shutil.copyfile(path, out)
        else:
            value = sanitize(read(path), changes)
            if changes:
                write(out, value)
            else:
                shutil.copyfile(path, out)
        transformation = 'recursive local-root path/key sanitization; embedded original hashes remain original' if changes else 'verbatim'
        register(target, source, not changes, transformation)
        metadata.append({'original_artifact': source, 'original_sha256': sha(path),
                         'public_artifact': target, 'public_sha256': sha(out),
                         'byte_identical': not changes, 'transformation': transformation,
                         'path_rewrites': changes})

    aliases = result['method_aliases']
    test_rows = {}
    dev_rows = {}
    for alias in aliases:
        test_rows[alias] = [json.loads(line) for line in (ROOT / f'results/grid_final_test/{alias}_scores.jsonl').read_text().splitlines()]
        dev_rows[alias] = [json.loads(line) for line in (ROOT / f'results/grid_final_dev/{alias}_scores.jsonl').read_text().splitlines()]
    first = test_rows[aliases[0]]
    ids = [row['id'] for row in first]
    dev_ids = [row['id'] for row in dev_rows[aliases[0]]]
    if any([row['id'] for row in test_rows[alias]] != ids or [row['id'] for row in dev_rows[alias]] != dev_ids for alias in aliases):
        raise ValueError('Compact score roster mismatch')
    compact = PUBLIC / 'benchmarks/grid/scores.npz'
    np.savez_compressed(compact, method_names=np.asarray(aliases), image_ids=np.asarray(ids),
                        is_anomaly=np.asarray([row['is_anomaly'] for row in first], dtype=bool),
                        defect_types=np.asarray([row['defect_type'] for row in first]),
                        test_scores=np.asarray([[test_rows[alias][i]['score'] for alias in aliases] for i in range(len(ids))], dtype=np.float64),
                        calibration_ids=np.asarray(dev_ids),
                        calibration_scores=np.asarray([[dev_rows[alias][i]['score'] for alias in aliases] for i in range(len(dev_ids))], dtype=np.float64),
                        thresholds=np.asarray([result['thresholds'][alias] for alias in aliases], dtype=np.float64))
    register('benchmarks/grid/scores.npz', transform='Derived compact image-score table; no pixels/features/masks/maps')
    write(PUBLIC / 'benchmarks/source_snapshot_manifest.json', {
        'format': 'hypernet.v15.benchmark_source_snapshots', 'files': snapshots,
        'grid_original_protocol_sha256': sha(ROOT / 'experiments/grid_protocol_freeze.json'),
        'gravity_original_protocol_sha256': sha(ROOT / 'experiments/gravity_spy_protocol_freeze.json'),
        'sources_byte_identical': True, 'test_image_data_included': False})
    register('benchmarks/source_snapshot_manifest.json', transform='Generated source snapshot index')
    write(PUBLIC / 'benchmarks/public_metadata_manifest.json', {
        'format': 'hypernet.v15.sanitized_benchmark_metadata', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'artifacts': metadata, 'original_files_modified': False,
        'prefix_meaning': '${V15_ROOT} denotes the original experiment root; it is not an executable path binding.',
        'hash_semantics': 'Hashes inside transformed metadata remain references to original bytes. Use original/public SHA mappings here; cleaned protocols are not original gate admissions.',
        'compact_scores_sha256': sha(compact), 'real_rgb_or_masks_or_maps_bundled': False,
        'grid_test_results_original_sha256': sha(result_path), 'gravity_test_results_included': False})
    register('benchmarks/public_metadata_manifest.json', transform='Generated original/public metadata hash mapping')
    exact('publication/README.md', 'README.md')
    exact('publication/GRID_README.md', 'benchmarks/grid/README.md')
    exact('publication/export_real_benchmarks.py')
    requirements = PUBLIC / 'requirements.txt'
    text = requirements.read_text()
    if 'scikit-learn==' not in text:
        requirements.write_text(text + 'scikit-learn==1.9.1\n', encoding='utf-8')
    register('requirements.txt', transform='Recorded synthetic and real benchmark runtime/test dependencies')
    manifest['files'] = sorted(records.values(), key=lambda item: item['path'])
    manifest['real_benchmark_results_bundled'] = True
    manifest['real_benchmark_scope'] = 'Audited Grid only, compact scores and metadata; GravitySpy results pending separate audited supplement.'
    manifest['real_images_or_user_uploads_bundled'] = False
    manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['benchmark_exporter_sha256'] = sha(__file__)
    write(manifest_path, manifest)
    print(json.dumps({'manifest_sha256': sha(manifest_path), 'files': len(manifest['files']),
                      'sanitized_metadata_files': sum(not row['byte_identical'] for row in metadata),
                      'compact_scores_sha256': sha(compact), 'source_snapshots': len(snapshots)}))


if __name__ == '__main__':
    main()
