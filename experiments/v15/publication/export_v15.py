"""Export reviewed mathematical v15 artifacts without touching the Git index."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil, zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PUBLIC = WORKSPACE / 'public/HyperNet/experiments/v15'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=PUBLIC)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination != PUBLIC.resolve():
        raise ValueError('This exporter is restricted to public/HyperNet/experiments/v15')
    if destination.exists():
        raise FileExistsError('Preserve the existing staging directory; reconcile explicitly')
    run = ROOT / 'checkpoints/details_run01'
    completed = json.loads((run / 'completed.json').read_text())
    admission = json.loads((run / 'admission.json').read_text())
    if not completed['completed'] or completed['counts']['unique_training_rgb_sha256'] != 2097152:
        raise ValueError('Completed full audited training is required')
    for name, digest in completed['source_sha256'].items():
        if sha(ROOT / name) != digest or sha(run / 'source_snapshot' / name) != digest:
            raise ValueError('Frozen source/snapshot mismatch: ' + name)

    module_names = ['detail_encoder', 'math_curriculum', 'math_curriculum_torch',
                    'detail_features', 'pattern_matcher', 'grid_anomaly', 'rare_event_metrics']
    script_names = ['train_details', 'evaluate_detail_correspondence', 'probe_detail_retention',
                    'audit_math_validation_overlap', 'audit_training_run',
                    'summarize_silhouette_overlap', 'one_reference_patterns', 'render_math_previews']
    test_names = ['test_detail_encoder', 'test_training_ledger', 'test_math_curriculum',
                  'test_detail_features', 'test_pattern_matcher', 'test_grid_anomaly',
                  'test_rare_event_metrics', 'pattern_matcher_scalar_reference']
    files = [f'hypernet/{name}.py' for name in module_names]
    files += [f'scripts/{name}.py' for name in script_names]
    files += [f'tests/{name}.py' for name in test_names]
    files += ['publication/export_v15.py']
    files += [f'checkpoints/details_run01/{name}' for name in [
        'admission.json', 'completed.json', 'history.jsonl', 'random_initial.pt',
        'families_0065536.pt', 'families_0262144.pt', 'families_1048576.pt']]
    files += ['checkpoints/details_run01/source_snapshot/' + name for name in completed['source_sha256']]
    files += [f'experiments/{name}' for name in [
        'training_independent_audit.json', 'math_validation_overlap.json',
        'detail_retention_final.json', 'dense_silhouette_overlap_posthoc.json',
        'math_generator_cuda_qa.json', 'audit_math_generator.py',
        'dense_learning_visual_audit.json', 'one_reference_cli_identity_smoke.json']]
    files += [f'experiments/{name}/results.json' for name in [
        'dense_fullscene_random', 'dense_fullscene_065536', 'dense_fullscene_262144', 'dense_fullscene_final']]
    files += ['assets/training_latest.png', 'assets/dense_learning_065536.png',
              'assets/math_preview/manifest.json', 'assets/math_preview/current_overview.png']
    preview = json.loads((ROOT / 'assets/math_preview/manifest.json').read_text())
    if preview['source_sha256'] != completed['source_sha256']['hypernet/math_curriculum.py']:
        raise ValueError('Preview uses a different mathematical renderer')
    for pair in preview['pairs']:
        for key in ('view_a', 'view_b', 'mask_a', 'mask_b', 'detail_a', 'detail_b'):
            files.append(pair[key])
    files.append('assets/math_preview/' + preview['source_sha256'][:12] + '/renderer_snapshot.py')
    files = sorted(set(files))
    for name in files:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)

    # A deterministic archive contains only the consumed-sample ledger. It lives
    # outside the public repository until an explicit release-asset upload.
    archive = ROOT / 'release_assets/hypernet_v15_training_ledger.zip'
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        raise FileExistsError('Preserve the previous release ledger archive')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for shard in completed['counts']['shards']:
            path = run / 'ledger' / shard['path']
            if sha(path) != shard['sha256']:
                raise ValueError('Consumed-source ledger changed')
            member = 'checkpoints/details_run01/ledger/' + shard['path']
            info = zipfile.ZipInfo(member, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(info, path.read_bytes(), compresslevel=1)

    destination.mkdir(parents=True)
    manifest = []
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != sha(ROOT / name):
            raise RuntimeError('Export copy differs: ' + name)
        manifest.append({'path': name, 'bytes': target.stat().st_size,
                         'sha256': sha(target), 'source_relative_path': name,
                         'source_bytes_preserved': True})
    shutil.copyfile(ROOT / 'publication/README.md', destination / 'README.md')
    (destination / 'requirements.txt').write_text(
        '# Install a compatible PyTorch build separately; recorded torch: 2.14.1+cu130\n'
        'numpy==2.5.3\nPillow==12.3.0\nopencv-python==5.0.0.93\nscipy==1.18.1\npytest==9.2.0\n', encoding='utf-8')
    (destination / 'pytest.ini').write_text('[pytest]\npythonpath = .\ntestpaths = tests\n', encoding='utf-8')
    (destination / 'hypernet/__init__.py').write_text('', encoding='utf-8')
    (destination / '.gitignore').write_text('__pycache__/\n.pytest_cache/\n.venv/\ncheckpoints/reproduction*/\nexperiments/reproduction*/\n', encoding='utf-8')
    counts = {key: value for key, value in completed['counts'].items() if key != 'shards'}
    ledger = {'format': 'hypernet.v15.ledger_release_manifest',
              'archive_name': archive.name, 'archive_sha256': sha(archive),
              'archive_bytes': archive.stat().st_size, 'uploaded': False,
              'contains': '128 exact per-sample digest shards, with package-relative extraction paths',
              'counts': counts, 'shards': completed['counts']['shards']}
    write(destination / 'ledger_release_manifest.json', ledger)
    for name in ('README.md', 'requirements.txt', 'pytest.ini', '.gitignore', 'hypernet/__init__.py', 'ledger_release_manifest.json'):
        manifest.append({'path': name, 'bytes': (destination / name).stat().st_size,
                         'sha256': sha(destination / name), 'source_bytes_preserved': name == 'README.md',
                         'source_relative_path': 'publication/README.md' if name == 'README.md' else None})
    write(destination / 'release_manifest.json', {
        'format': 'hypernet.v15.public_staging', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'staged_only_not_committed_or_pushed', 'files': manifest,
        'training_source_sha256': completed['source_sha256'],
        'training_admission_sha256': sha(run / 'admission.json'),
        'training_completion_sha256': sha(run / 'completed.json'),
        'final_checkpoint_sha256': sha(run / 'families_1048576.pt'),
        'final_state_sha256': completed['final_state_sha256'],
        'borrowed_pretrained_weights': False, 'natural_training_images': 0,
        'real_images_or_user_uploads_bundled': False, 'real_benchmark_results_bundled': False,
        'ledger_archive_in_git': False, 'exporter_sha256': sha(__file__)})
    print(json.dumps({'destination': str(destination), 'files': len(manifest),
                      'archive': str(archive), 'archive_sha256': sha(archive),
                      'manifest_sha256': sha(destination / 'release_manifest.json')}))


if __name__ == '__main__':
    main()
