"""Copy a locally frozen v09 source tree; publish no photos, weights or local paths."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = Path(__file__).resolve().parents[1]
    freeze_path = source / 'experiments/v09_freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    expected = {Path(path).resolve(): value for path, value in freeze['behavior_files'].items()}
    core = sorted((source / 'hypernet').rglob('*.py'))
    for path in core:
        if expected.get(path.resolve()) != digest(path):
            raise ValueError('Package file is not byte-identical to frozen source: ' + path.name)
    snapshot_path = destination / 'publication/source_snapshot.json'
    archive = destination / 'publication/source_snapshot_v0_8_1.json'
    if not archive.exists():
        shutil.copyfile(snapshot_path, archive)
    previous = json.loads(archive.read_text(encoding='utf-8'))
    exported = {}
    for path in core:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        exported[relative.as_posix()] = digest(target)
    for path in (source / 'tests').glob('test_*.py'):
        # This test imports a machine-specific audit driver, not package code.
        if path.name != 'test_audit_v09.py':
            shutil.copyfile(path, destination / 'tests' / path.name)
    for name in ('procedural_primary_sources.json', 'procedural_research.html'):
        shutil.copyfile(source / 'research' / name, destination / 'research' / name)
    pyproject = destination / 'pyproject.toml'
    pyproject.write_text(pyproject.read_text(encoding='utf-8').replace('version = "0.8.1"',
                        'version = "0.9.0"'), encoding='utf-8')
    snapshot = {'snapshot': 'HyperNet v0.9.0 procedural control experiment',
        'created_date': '2026-11-02', 'core_file_count': len(exported), 'core_files': exported,
        'experimental_freeze_sha256': digest(freeze_path),
        'all_core_files_byte_identical_to_frozen_experiment': True,
        'prior_snapshot': 'publication/source_snapshot_v0_8_1.json',
        'unchanged_prior_core_files': sum(exported.get(k) == v for k, v in previous['core_files'].items()),
        'evaluation_status': 'Fresh-photo evaluation pending independent final audits; no v09 superiority claim.',
        'notes': ['No photographs, checkpoints or private paths are exported.',
                  'Packaging, examples and publication documents are outside frozen model code.',
                  'This primitive-shape curriculum is a control; richer named-shape work is separate.']}
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'core_files': len(exported), 'unchanged_prior_core_files': snapshot['unchanged_prior_core_files'],
                      'freeze_sha256': snapshot['experimental_freeze_sha256']}))


if __name__ == '__main__':
    main()
