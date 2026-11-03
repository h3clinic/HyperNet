"""Stage six audited GravitySpy arms without pixels or Git mutations."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs
import hashlib, json, re, shutil, zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PUBLIC = WORKSPACE / 'public/HyperNet/experiments/v15'
POINTS = ['budget_10_per10000', 'zero_calibration_false_alarms']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def score_arrays(path, methods, targets, split):
    roster = []
    values = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            need(row['split'] == split, 'Wrong score-log split')
            roster.append((row['event_id'], row['group_id'], row['label'], row['pixel_sha256']))
            values.append([[row['scores'][method][target]['score'] for target in targets] for method in methods])
    return roster, np.asarray(values, dtype=np.float64).transpose(1, 0, 2)


def main():
    protocol_path = ROOT / 'experiments/gravity_spy_protocol_freeze.json'
    protocol = read(protocol_path)
    arms = list(protocol['reported_arms'])
    targets = protocol['target_labels']
    need(len(arms) == 6 and len(targets) == 19, 'Expected frozen six-arm roster')
    needed = ['experiments/gravity_spy_paired_analysis.json', 'experiments/gravity_spy_inference_replay.json']
    for model in protocol['models']:
        needed += [f'experiments/gravity_test_{model}/{name}' for name in ('results.json', 'event_scores.jsonl', 'independent_score_audit.json')]
    missing = [name for name in needed if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError('No export until all evidence is ready: ' + ', '.join(missing))
    if (PUBLIC / 'benchmarks/gravity_spy').exists():
        raise FileExistsError('Preserve an earlier GravitySpy public supplement')
    manifest_path = PUBLIC / 'release_manifest.json'
    manifest = read(manifest_path)
    for row in manifest['files']:
        need(sha(PUBLIC / row['path']) == row['sha256'], 'Existing staging file changed: ' + row['path'])
    for group in ('behavior_files', 'checkpoint_files'):
        for name, digest in protocol[group].items():
            need(sha(ROOT / name) == digest, 'Frozen source/calibration binding changed: ' + name)
    analysis = read(ROOT / 'experiments/gravity_spy_paired_analysis.json')
    plan = read(ROOT / 'experiments/gravity_spy_bootstrap_plan.json')
    replay = read(ROOT / 'experiments/gravity_spy_inference_replay.json')
    need(analysis['protocol_sha256'] == sha(protocol_path) and analysis['analysis_plan_sha256'] == sha(ROOT / 'experiments/gravity_spy_bootstrap_plan.json'), 'Bootstrap binding differs')
    need(analysis['analysis_source_sha256'] == sha(ROOT / 'scripts/analyze_gravity_spy_v15.py'), 'Bootstrap source differs')
    need(analysis['valid_draws'] == 5000 and analysis['effective_test_events'] == 4044 and analysis['time_groups'] == 3954, 'Incomplete paired analysis')
    need(replay['valid'] and replay['maximum_absolute_error'] == 0 and replay['original_sources_models_data_results_unchanged'], 'Exact replay incomplete or failed')
    need(plan['selected_arms'] == protocol['reported_arms'], 'Analysis arm selection differs')

    # Read only selected scores from the large immutable logs, without retaining
    # all per-anchor evidence dictionaries or loading a model/image.
    test_sets = {}
    calibration_sets = {}
    results = {}
    inputs = {}
    audit_snapshot_sources = set()
    test_roster = calibration_roster = None
    metadata_sources = []
    for model, spec in protocol['models'].items():
        folder = ROOT / f'experiments/gravity_test_{model}'
        result = read(folder / 'results.json')
        audited = read(folder / 'independent_score_audit.json')
        need(result['completed'] and result['mode'] == 'test' and result['model_state_unchanged'], 'Incomplete TEST')
        need(audited['valid'] and audited['result_sha256'] == sha(folder / 'results.json') and audited['score_log_sha256'] == sha(folder / 'event_scores.jsonl'), 'TEST arithmetic audit binding differs')
        if audited['audit_source_sha256'] != sha(ROOT / 'scripts/audit_gravity_scores.py'):
            archived = 'experiments/audit_source_snapshots/' + audited['audit_source_sha256'] + '.py'
            need((ROOT / archived).is_file() and sha(ROOT / archived) == audited['audit_source_sha256'], 'TEST audit source lacks an exact archived snapshot')
            audit_snapshot_sources.add(archived)
        need(result['scores_sha256'] == audited['score_log_sha256'], 'Result/log binding differs')
        need(result['checkpoint_provenance']['checkpoint_sha256'] == protocol['checkpoint_files'][spec['checkpoint_path']], 'Checkpoint differs')
        for name, digest in result['source_sha256'].items():
            need(sha(ROOT / name) == digest, 'Frozen inference source differs')
        selected = [name for name, arm in protocol['reported_arms'].items() if arm['model'] == model]
        methods = [protocol['reported_arms'][name]['method'] for name in selected]
        roster, values = score_arrays(folder / 'event_scores.jsonl', methods, targets, 'test')
        if test_roster is None:
            test_roster = roster
        else:
            need(roster == test_roster, 'TEST identity/pixel roster differs')
        for i, name in enumerate(selected):
            test_sets[name] = values[i]
        cal_path = ROOT / spec['score_path']
        cal_result = read(ROOT / spec['result_path'])
        need(sha(cal_path) == spec['scores_sha256'] == cal_result['scores_sha256'], 'CALIB score binding differs')
        roster, values = score_arrays(cal_path, methods, targets, 'calibration')
        if calibration_roster is None:
            calibration_roster = roster
        else:
            need(roster == calibration_roster, 'CALIB identity/pixel roster differs')
        for i, name in enumerate(selected):
            calibration_sets[name] = values[i]
        results[model] = result
        for path in (folder / 'results.json', folder / 'event_scores.jsonl', ROOT / spec['result_path'], cal_path):
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        metadata_sources += [f'experiments/gravity_test_{model}/{name}' for name in ('results.json', 'independent_score_audit.json')]
        metadata_sources += [spec['result_path'], str(Path(spec['result_path']).parent / 'independent_score_audit.json').replace('\\', '/')]
    need(len(test_roster) == 4044 and len({row[1] for row in test_roster}) == 3954, 'Fresh TEST dimensions differ')
    need(len(calibration_roster) == 1579, 'CALIB count differs')
    for path, digest in analysis['inputs'].items():
        need(inputs[path] == digest, 'Bootstrap score inputs differ')
    thresholds = np.asarray([[[protocol['reported_arms'][arm]['thresholds'][target][point] for target in targets] for point in POINTS] for arm in arms], dtype=np.float64)
    for a, arm in enumerate(arms):
        spec = protocol['reported_arms'][arm]
        for t, target in enumerate(targets):
            for p, point in enumerate(POINTS):
                actual = results[spec['model']]['methods'][spec['method']]['per_target'][target][point]['cutoff']['threshold']
                need(thresholds[a, p, t] == actual, 'Frozen arm threshold differs')

    # Full diagnostic logs are too large for Git. Archive their original bytes
    # separately; only the archive/digest manifest enters public staging.
    log_names = sorted(name for name in inputs if name.endswith('/event_scores.jsonl'))
    need(len(log_names) == 8, 'Eight CALIB/TEST logs are required')
    log_manifest = {'format': 'hypernet.v15.gravity_full_score_logs',
                    'files': [{'path': name, 'sha256': inputs[name], 'bytes': (ROOT / name).stat().st_size} for name in log_names],
                    'images_or_feature_vectors_included': False, 'absolute_paths_in_manifest': False,
                    'scope': 'Eight exact CALIB/TEST score logs, all frozen diagnostic recipes and targets; no images.'}
    asset = ROOT / 'release_assets/hypernet_v15_gravity_score_logs.zip'
    asset.parent.mkdir(exist_ok=True)
    need(not asset.exists(), 'Preserve earlier score-log archive')
    with zipfile.ZipFile(asset, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for record in log_manifest['files']:
            need(sha(ROOT / record['path']) == record['sha256'], 'Score log changed before archiving')
            info = zipfile.ZipInfo(record['path'], date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT / record['path']).read_bytes(), compresslevel=1)
        info = zipfile.ZipInfo('archive_manifest.json', date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(log_manifest, indent=2).encode() + b'\n', compresslevel=1)
    with zipfile.ZipFile(asset) as archive:
        need(set(archive.namelist()) == {*log_names, 'archive_manifest.json'}, 'Unexpected release archive member')
        for record in log_manifest['files']:
            value = hashlib.sha256()
            with archive.open(record['path']) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(chunk)
            need(value.hexdigest() == record['sha256'], 'Release archive member differs')
        need(json.loads(archive.read('archive_manifest.json')) == log_manifest, 'Archive manifest differs')
    log_release = {**log_manifest, 'format': 'hypernet.v15.gravity_score_logs_release', 'archive_name': asset.name,
                   'archive_sha256': sha(asset), 'archive_bytes': asset.stat().st_size,
                   'uploaded': False, 'uploaded_field_scope': 'Historical state at archive preparation only; later publication is outside this exporter.',
                   'archive_entries_verified_against_originals': True}
    write(ROOT / 'release_assets/hypernet_v15_gravity_score_logs_manifest.json', log_release)

    metadata_sources += [
        'experiments/gravity_spy_protocol_freeze.json', 'experiments/gravity_spy_test_admission.json',
        'experiments/gravity_spy_bootstrap_plan.json', 'experiments/gravity_spy_paired_analysis.json',
        'experiments/gravity_spy_replay_plan.json', 'experiments/gravity_spy_exact_replay_roster.json',
        'experiments/gravity_spy_inference_replay.json', 'experiments/gravity_spy_final_data_independent_audit.json',
        'experiments/gravity_spy_prior_exposure_audit.json', 'experiments/gravity_spy_independent_metadata_review.json',
        'experiments/gravity_spy_development_data_audit.json', 'experiments/gravity_spy_probe_exclusion.json',
        'experiments/gravity_calibration_independent_audit.json', 'experiments/gravity_matcher_independent_review.json',
        'experiments/gravity_audit_group_optimization.json',
        'experiments/v15_objective_scope_review.json', 'data/gravity_spy_v15_final/roster_freeze.json',
        'data/gravity_spy_v15_final/quality_admission.json', 'data/gravity_spy_v15_final/quality_test_audit.json',
        'data/gravity_spy_v15_final/prior_exposure_admission.json',
        'data/gravity_spy_v15_final/acquisition_reference_20.json',
        'data/gravity_spy_v15_final/acquisition_calibration_1608.json',
        'data/gravity_spy_v15_final/acquisition_test_4119.json']
    need(all((ROOT / name).is_file() for name in metadata_sources), 'Metadata evidence is incomplete')
    records = {row['path']: row for row in manifest['files']}
    metadata_manifest = read(PUBLIC / 'benchmarks/public_metadata_manifest.json')
    snapshot_manifest = read(PUBLIC / 'benchmarks/source_snapshot_manifest.json')
    snapshot_records = {row['source']: row for row in snapshot_manifest['files']}

    def register(target, source=None, preserved=False, transformation=None):
        path = PUBLIC / target
        records[target] = {'path': target, 'bytes': path.stat().st_size, 'sha256': sha(path),
                           'source_relative_path': source, 'source_bytes_preserved': preserved,
                           'transformation': transformation}
        if source:
            records[target]['original_source_sha256'] = sha(ROOT / source)

    def exact(source, target=None):
        target = target or source
        path = PUBLIC / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, path)
        need(sha(path) == sha(ROOT / source), 'Source copy mismatch')
        register(target, source, True)

    code = ['scripts/analyze_gravity_spy_v15.py', 'tests/test_gravity_bootstrap.py',
            'scripts/recompute_public_gravity_metrics.py', 'scripts/audit_gravity_scores.py',
            'experiments/audit_gravity_final_data.py', 'experiments/replay_gravity_final.py']
    code += sorted(audit_snapshot_sources)
    for source in code:
        exact(source)
        target = 'benchmarks/source_snapshot/' + source
        exact(source, target)
        snapshot_records[source] = {'source': source, 'snapshot': target, 'sha256': sha(ROOT / source),
                                    'role': 'Byte-identical source; historical audits may cite earlier versions.'}
    prefixes = [(str(ROOT.resolve()), '${V15_ROOT}'), (ROOT.resolve().as_posix(), '${V15_ROOT}'),
                (str(PUBLIC.parents[1].resolve()), '${PUBLIC_REPO}'), (PUBLIC.parents[1].resolve().as_posix(), '${PUBLIC_REPO}'),
                (str(WORKSPACE.resolve()), '${WORKSPACE_ROOT}'), (WORKSPACE.resolve().as_posix(), '${WORKSPACE_ROOT}')]
    prefixes.sort(key=lambda item: len(item[0]), reverse=True)

    def clean(value, rewrites):
        if isinstance(value, str):
            original = value
            for prefix, replacement in prefixes:
                value = value.replace(prefix, replacement)
            need(not re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]', value), 'Unmapped local absolute path')
            if value.startswith(('http://', 'https://')):
                query = {key.lower() for key in parse_qs(urlsplit(value).query)}
                need(not query.intersection({'token', 'access_token', 'api_key', 'apikey', 'signature', 'sig', 'x-amz-signature'}), 'Credential-like URL is not publishable')
            if value != original:
                rewrites.append({'original_string_sha256': hashlib.sha256(original.encode()).hexdigest(), 'public_string': value})
            return value
        if isinstance(value, list):
            return [clean(item, rewrites) for item in value]
        if isinstance(value, dict):
            return {clean(key, rewrites): clean(item, rewrites) for key, item in value.items()}
        return value

    for source in sorted(set(metadata_sources)):
        rewrites = []
        public_value = clean(read(ROOT / source), rewrites)
        target = 'benchmarks/gravity_spy/provenance/' + source
        if rewrites:
            write(PUBLIC / target, public_value)
        else:
            exact(source, target)
        transformation = 'Recursive local-path/key sanitization; original embedded hashes retained' if rewrites else 'verbatim'
        register(target, source, not rewrites, transformation)
        metadata_manifest['artifacts'].append({'original_artifact': source, 'original_sha256': sha(ROOT / source),
            'public_artifact': target, 'public_sha256': sha(PUBLIC / target), 'byte_identical': not rewrites,
            'transformation': transformation, 'path_rewrites': rewrites})

    folder = PUBLIC / 'benchmarks/gravity_spy'
    cohort_plan = read(ROOT / 'data/gravity_spy_v15_final/plan.json')
    np.savez_compressed(folder / 'scores.npz', arm_names=np.asarray(arms),
        arm_models=np.asarray([protocol['reported_arms'][arm]['model'] for arm in arms]),
        arm_methods=np.asarray([protocol['reported_arms'][arm]['method'] for arm in arms]),
        target_labels=np.asarray(targets), operating_points=np.asarray(POINTS), thresholds=thresholds,
        test_scores=np.stack([test_sets[arm] for arm in arms]),
        calibration_scores=np.stack([calibration_sets[arm] for arm in arms]),
        test_event_ids=np.asarray([row[0] for row in test_roster]), test_group_ids=np.asarray([row[1] for row in test_roster]),
        test_labels=np.asarray([row[2] for row in test_roster]), test_pixel_sha256=np.asarray([row[3] for row in test_roster]),
        calibration_event_ids=np.asarray([row[0] for row in calibration_roster]),
        calibration_group_ids=np.asarray([row[1] for row in calibration_roster]),
        calibration_labels=np.asarray([row[2] for row in calibration_roster]),
        calibration_pixel_sha256=np.asarray([row[3] for row in calibration_roster]),
        reference_event_ids=np.asarray([cohort_plan['reference_by_label'][target] for target in targets]))
    register('benchmarks/gravity_spy/scores.npz', transformation='Compact selected-arm image-score tensors and identities; no pixels/features/objects')
    write(folder / 'full_score_logs_release_manifest.json', log_release)
    register('benchmarks/gravity_spy/full_score_logs_release_manifest.json', transformation='Verified external eight-log archive metadata; archive itself is outside Git')
    metrics = {'format': 'hypernet.v15.public_gravity_metrics', 'primary': 'primary_final',
               'test_events': 4044, 'time_groups': 3954, 'references': 19, 'calibration_events': 1579,
               'natural_weight_training_images': 0, 'prior_exposed_references': 3,
               'prior_exposed_calibration_events': 567, 'constant_test_exclusions': 75,
               'arms': {}, 'original_input_sha256': inputs,
               'original_protocol_sha256': sha(protocol_path), 'score_archive_sha256': sha(folder / 'scores.npz'),
               'scope': 'Six CALIB-selected frozen reported arms; macro rates, target rates and joint event rates are distinct.'}
    for arm, spec in protocol['reported_arms'].items():
        model_result = results[spec['model']]
        metrics['arms'][arm] = {'model': spec['model'], 'method': spec['method'],
            'reported_metrics': model_result['methods'][spec['method']],
            'checkpoint_provenance': model_result['checkpoint_provenance'],
            'result_original_sha256': inputs[f"experiments/gravity_test_{spec['model']}/results.json"],
            'score_log_original_sha256': inputs[f"experiments/gravity_test_{spec['model']}/event_scores.jsonl"]}
    write(folder / 'metrics.json', metrics)
    register('benchmarks/gravity_spy/metrics.json', transformation='Selected audited metrics with original result/log/checkpoint bindings')
    table = ['| Arm | CALIB-selected method | Recall | Mean FP/10k | Mean AUROC |',
             '|---|---|---:|---:|---:|']
    for arm in arms:
        info = metrics['arms'][arm]
        values = info['reported_metrics']['macro']
        table.append(f"| {arm} | {info['method']} | {values['recall'] * 100:.2f}% | {values['false_alarms_per10000']:.2f} | {values['auroc']:.4f} |")
    primary = analysis['methods']['primary_final'][POINTS[0]]
    recall_ci = primary['macro_recall']['ci95']
    fp_ci = primary['macro_false_alarms_per10000_negatives']['ci95']
    intervals = (f"Final primary paired-group bootstrap 95% intervals: macro recall **{recall_ci[0]*100:.2f}%–{recall_ci[1]*100:.2f}%**; "
                 f"mean per-target FP/10k **{fp_ci[0]:.2f}–{fp_ci[1]:.2f}**. These are conditional benchmark intervals, not deployment guarantees.")
    random_delta = analysis['contrasts']['primary_final_minus_random'][POINTS[0]]['macro_recall']
    raw_delta = analysis['contrasts']['primary_final_minus_raw_pixels48'][POINTS[0]]['macro_recall']
    intervals += (f"\n\nThe paired final-minus-random recall interval is **{random_delta['ci95'][0]*100:.2f} to {random_delta['ci95'][1]*100:.2f} percentage points**. "
                  "It includes zero, so this benchmark does not demonstrate a recall gain over random initialization. "
                  f"Final-minus-raw48 recall is **{raw_delta['difference']*100:.2f} percentage points**, with a 95% interval of "
                  f"**{raw_delta['ci95'][0]*100:.2f} to {raw_delta['ci95'][1]*100:.2f} points**.")
    rendered = (ROOT / 'publication/GRAVITY_README.md').read_text().replace('{{RESULTS_TABLE}}', '\n'.join(table)).replace('{{PRIMARY_INTERVALS}}', intervals)
    (folder / 'README.md').write_text(rendered, encoding='utf-8')
    register('benchmarks/gravity_spy/README.md', 'publication/GRAVITY_README.md', False, 'Template filled with frozen six-arm metrics and predeclared paired intervals')
    exact('publication/README.md', 'README.md')
    exact('publication/GRID_README.md', 'benchmarks/grid/README.md')
    exact('publication/export_gravity_results.py')
    metadata_manifest['gravity_test_results_included'] = True
    metadata_manifest['symbolic_path_prefixes'] = ['${V15_ROOT}', '${PUBLIC_REPO}', '${WORKSPACE_ROOT}']
    metadata_manifest['gravity_compact_scores_sha256'] = sha(folder / 'scores.npz')
    metadata_manifest['original_hashes_inside_sanitized_files_are_not_rewritten'] = True
    write(PUBLIC / 'benchmarks/public_metadata_manifest.json', metadata_manifest)
    register('benchmarks/public_metadata_manifest.json', transformation='Original/public metadata SHA map; Grid plus six-arm GravitySpy')
    snapshot_manifest['files'] = sorted(snapshot_records.values(), key=lambda item: item['source'])
    snapshot_manifest['historical_audit_source_note'] = 'Primary/random TEST audits used an archived exact audit source; later TEST audits use the equivalent group-count optimization, whose validation record is included. Earlier CALIB audits may cite older source hashes not represented by the current version. Original hashes are retained.'
    write(PUBLIC / 'benchmarks/source_snapshot_manifest.json', snapshot_manifest)
    register('benchmarks/source_snapshot_manifest.json', transformation='Exact original source snapshot index with historical audit-version note')
    ledger_release = read(PUBLIC / 'ledger_release_manifest.json')
    ledger_release['uploaded_field_scope'] = 'Historical state at initial staging only; subsequent release upload is outside this artifact.'
    write(PUBLIC / 'ledger_release_manifest.json', ledger_release)
    register('ledger_release_manifest.json', transformation='Original ledger archive hashes/counts with explicit historical upload-state scope')
    manifest['files'] = sorted(records.values(), key=lambda item: item['path'])
    manifest['real_benchmark_scope'] = 'Audited Grid and six frozen GravitySpy arms; compact score arrays/metadata only.'
    manifest['gravity_exporter_sha256'] = sha(__file__)
    manifest['status'] = 'Historical staging record at export time; subsequent Git/release publication is not tracked by this artifact.'
    manifest['git_actions_performed_by_exporters'] = False
    manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write(manifest_path, manifest)
    print(json.dumps({'manifest_sha256': sha(manifest_path), 'files': len(manifest['files']),
                      'compact_scores_sha256': sha(folder / 'scores.npz'), 'compact_score_bytes': (folder / 'scores.npz').stat().st_size,
                      'gravity_metadata_files': len(set(metadata_sources)), 'original_files_modified': False,
                      'full_log_archive_sha256': sha(asset), 'full_log_archive_bytes': asset.stat().st_size}))


if __name__ == '__main__':
    main()
