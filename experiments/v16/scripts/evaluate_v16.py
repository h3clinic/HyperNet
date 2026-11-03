"""Calibrate one-reference structural matching, then score a sealed fresh cohort.

Natural images never enter optimization. The one reference per target is reused
across all arms. CALIB chooses one v16 recipe, shared by the matched ablation and
random control. The previous v15 recipe and raw48 control remain fixed.
"""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, argparse, json, hashlib, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'hypernet_v15'
sys.path.insert(0, str(ROOT))
from hypernet.gravity_spy_images import GravitySpyCohort
from hypernet.gravity_spy_quality import is_constant_plot
from hypernet.relational_matcher import FrozenRelations, score_vectors, RECIPES, INFERENCE_BATCH_SIZE
from hypernet.detail_features import FrozenDetails
from hypernet.pattern_matcher import PatternMatcher, raw_pixel_vector
from hypernet.evaluation_metrics import aggregate

ALIASES = ('v16', 'v16_no_relations', 'random', 'v15', 'raw48')
CNN_ALIASES = ALIASES[:3]
SOURCES = ('hypernet/relational_encoder.py', 'hypernet/relational_matcher.py',
           'hypernet/detail_encoder.py', 'hypernet/detail_features.py',
           'hypernet/pattern_matcher.py', 'hypernet/evaluation_metrics.py',
           'hypernet/gravity_spy_images.py', 'hypernet/gravity_spy_quality.py',
           'hypernet/gravity_spy_admission.py', 'hypernet/rare_event_metrics.py',
           'scripts/evaluate_v16.py', 'scripts/prepare_fresh_gravity.py', 'scripts/admit_fresh_test.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def progress(stage, scored, total, seconds, complete=False):
    record = {'stage': stage, 'scored': scored, 'total': total, 'seconds': seconds, 'complete': complete,
              'updated_utc': datetime.now(timezone.utc).isoformat()}
    path = ROOT / 'benchmark_progress.json'; tmp = path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8'); tmp.replace(path)
    print(json.dumps(record), flush=True)


def source_hashes():
    return {str((ROOT / p).resolve()): sha(ROOT / p) for p in SOURCES}


def verify_files(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Frozen artifact changed: ' + path)


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_models(path):
    models = read(path)
    if set(models) != set(CNN_ALIASES):
        raise ValueError('Exactly primary, matched ablation and initial random models required')
    return {alias: str(Path(p).resolve()) for alias, p in models.items()}


def old_data():
    return GravitySpyCohort(OLD / 'data/gravity_spy_v15_final')


def old_calibration_scores(events, target_indices):
    """Reuse the exact prior v15 CALIB scores; binding and pixels are checked."""
    folder = OLD / 'experiments/gravity_calibration_final'
    result = read(folder / 'results.json')
    if result['scores_sha256'] != sha(folder / 'event_scores.jsonl'):
        raise ValueError('Old calibration score binding differs')
    if result['checkpoint_provenance']['checkpoint_sha256'] != sha(OLD / 'checkpoints/details_run01/families_1048576.pt'):
        raise ValueError('The actual v15 control checkpoint differs from its calibration')
    for name in ('hypernet/detail_encoder.py', 'hypernet/detail_features.py', 'hypernet/pattern_matcher.py'):
        if sha(ROOT / name) != result['source_sha256'][name]:
            raise ValueError('The reused v15 inference source differs from its calibration')
    rows = {}
    with (folder / 'event_scores.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); rows[r['event_id']] = r
    targets = result['target_labels']
    if len(rows) != len(events) or set(rows) != {r['event_id'] for r in events}:
        raise ValueError('Old calibration event roster differs')
    return rows, targets, {str((folder / name).resolve()): sha(folder / name)
                          for name in ('results.json', 'event_scores.jsonl')}


def save_scores(path, scores, methods, rows, targets, pixel_hashes, thresholds):
    np.savez_compressed(path, scores=np.asarray(scores, dtype=np.float64), methods=np.asarray(methods),
                        event_ids=np.asarray([r['event_id'] for r in rows]), labels=np.asarray([r['label'] for r in rows]),
                        group_ids=np.asarray([r['group_id'] for r in rows]), targets=np.asarray(targets),
                        pixel_sha256=np.asarray(pixel_hashes), thresholds=np.asarray(thresholds, dtype=np.float64))


def calibration(args):
    if args.output.exists(): raise FileExistsError('Keep completed and partial runs; choose a new output')
    plan = read(args.plan); targets = list(plan['target_labels'])
    cohort = old_data(); all_targets = list(cohort.target_labels)
    indices = [all_targets.index(t) for t in targets]
    if len(targets) < 2 or len(set(targets)) != len(targets): raise ValueError('Invalid fixed target set')
    events = list(cohort.query_events('calibration'))
    models = resolve_models(args.models)
    extractors = {k: FrozenRelations(v, args.device) for k, v in models.items()}
    reference_images = [cohort.reference(t) for t in all_targets]
    banks, references = {}, {}
    for alias, extractor in extractors.items():
        banks[alias], references[alias] = extractor.enroll_many(reference_images)
    raw_bank = np.stack([raw_pixel_vector(rgb) for rgb in reference_images])
    previous, previous_targets, old_bindings = old_calibration_scores(events, indices)
    if previous_targets != all_targets: raise ValueError('Old target order differs')
    if any(references[k] != references['v16'] for k in CNN_ALIASES): raise ValueError('Reference pixels differ')
    sources = source_hashes()
    methods = [a + ':' + r for a in CNN_ALIASES for r in RECIPES] + ['v15', 'raw48']
    values = np.zeros((len(methods), len(events), len(targets)), np.float64)
    pixels = []; started = time.perf_counter(); args.output.mkdir(parents=True)
    for i, event in enumerate(events):
        if i % INFERENCE_BATCH_SIZE == 0:
            query_images = [cohort.load_rgb(r['event_id']) for r in events[i:i + INFERENCE_BATCH_SIZE]]
            encoded_queries = {a: e.encode_many(query_images) for a, e in extractors.items()}
        position = i % INFERENCE_BATCH_SIZE
        rgb = query_images[position]; pixel = hashlib.sha256(rgb.tobytes()).hexdigest(); pixels.append(pixel)
        prior = previous[event['event_id']]
        if any(prior[k] != event[k] for k in ('event_id', 'label', 'group_id')) or prior['pixel_sha256'] != pixel:
            raise ValueError('Prior calibration event/pixel mismatch')
        for a, extractor in extractors.items():
            matched = score_vectors(banks[a], encoded_queries[a][position])
            for recipe in RECIPES: values[methods.index(a + ':' + recipe), i] = matched[recipe][indices]
        values[methods.index('v15'), i] = [prior['scores']['learned:translation_006'][t]['score'] for t in targets]
        values[methods.index('raw48'), i] = (raw_bank @ raw_pixel_vector(rgb))[indices]
        if (i + 1) % 32 == 0:
            progress('calibration', i + 1, len(events), time.perf_counter() - started)
    labels = [r['label'] for r in events]; groups = [r['group_id'] for r in events]
    candidate_metrics = {r: aggregate(values[methods.index('v16:' + r)], labels, targets, groups) for r in RECIPES}
    if any(candidate_metrics[r]['macro']['recall'] is None for r in RECIPES): raise ValueError('Incomplete calibration target outcomes')
    recipe = max(RECIPES, key=lambda r: (candidate_metrics[r]['macro']['recall'], candidate_metrics[r]['macro']['auroc'], -RECIPES.index(r)))
    chosen = [a + ':' + recipe for a in CNN_ALIASES] + ['v15', 'raw48']
    metrics = {a: aggregate(values[methods.index(m)], labels, targets, groups) for a, m in zip(ALIASES, chosen)}
    thresholds = np.array([metrics[a]['thresholds'] for a in ALIASES])
    save_scores(args.output / 'candidate_scores.npz', values, methods, events, targets, pixels, np.full((len(methods), len(targets)), np.nan))
    save_scores(args.output / 'scores.npz', np.stack([values[methods.index(m)] for m in chosen]), ALIASES, events, targets, pixels, thresholds)
    for extractor in extractors.values(): extractor.assert_unchanged()
    verify_files(sources); verify_files(old_bindings)
    bindings = {str(Path(v).resolve()): sha(v) for v in models.values()}
    for name in ('plan.json', 'roster_freeze.json', 'quality_admission.json', 'prior_exposure_admission.json'):
        p = OLD / 'data/gravity_spy_v15_final' / name; bindings[str(p)] = sha(p)
    result = {'completed': True, 'stage': 'calibration', 'target_labels': targets, 'events': len(events),
              'models': models, 'source_sha256': sources, 'input_bindings': {**bindings, **old_bindings},
              'fresh_plan_sha256': sha(args.plan), 'plan_path': str(args.plan.resolve()),
              'reference_pixel_hashes_all19': references['v16'], 'all_reference_target_order': all_targets,
              'selected_recipe': recipe, 'recipe_order': list(RECIPES),
              'recipe_selection': 'Primary v16 macro recall at empirical10/10000 CALIB budget; tie AUROC then fixed order. Same recipe for all new-CNN arms.',
              'candidate_metrics_primary': candidate_metrics, 'metrics': metrics,
              'scores_sha256': sha(args.output / 'scores.npz'), 'candidate_scores_sha256': sha(args.output / 'candidate_scores.npz'),
              'model_provenance': {a: e.provenance() for a, e in extractors.items()},
              'natural_weight_training_images': 0, 'additional_real_calibration_events': len(events),
              'runtime': {'device': args.device, 'torch': str(torch.__version__), 'numpy': str(np.__version__)},
              'seconds': time.perf_counter() - started}
    write_new(args.output / 'results.json', result)
    progress('calibration', len(events), len(events), time.perf_counter() - started, complete=True)
    print(json.dumps({'complete': True, 'recipe': recipe, 'calibration_events': len(events)}), flush=True)


def freeze(args):
    calibration_path = args.calibration / 'results.json'; c = read(calibration_path)
    if not c['completed'] or c['stage'] != 'calibration': raise ValueError('Completed calibration required')
    if sha(args.plan) != c['fresh_plan_sha256']: raise ValueError('Target cohort changed after calibration')
    verify_files(c['source_sha256']); verify_files(c['input_bindings'])
    # Recompute the complete selection from saved, hash-bound calibration
    # scores instead of treating a mutable summary as a new source of truth.
    if sha(args.calibration / 'scores.npz') != c['scores_sha256'] or sha(args.calibration / 'candidate_scores.npz') != c['candidate_scores_sha256']:
        raise ValueError('Completed calibration score archive changed')
    with np.load(args.calibration / 'scores.npz', allow_pickle=False) as z:
        chosen = {k: z[k].copy() for k in z.files}
    with np.load(args.calibration / 'candidate_scores.npz', allow_pickle=False) as z:
        candidates = {k: z[k].copy() for k in z.files}
    if chosen['methods'].tolist() != list(ALIASES) or chosen['targets'].tolist() != c['target_labels']:
        raise ValueError('Calibration target/method order differs')
    expected_methods = [a + ':' + r for a in CNN_ALIASES for r in RECIPES] + ['v15', 'raw48']
    if candidates['methods'].tolist() != expected_methods:
        raise ValueError('Calibration candidate order differs')
    for key in ('event_ids', 'labels', 'group_ids', 'targets', 'pixel_sha256'):
        if not np.array_equal(chosen[key], candidates[key]): raise ValueError('Candidate/selected metadata differs: ' + key)
    expected_events = list(old_data().query_events('calibration'))
    for field, key in (('event_ids', 'event_id'), ('labels', 'label'), ('group_ids', 'group_id')):
        if chosen[field].tolist() != [r[key] for r in expected_events]:
            raise ValueError('Calibration archive differs from the admitted event roster')
    labels = chosen['labels'].tolist(); targets = chosen['targets'].tolist(); groups = chosen['group_ids'].tolist()
    checks = {r: aggregate(candidates['scores'][expected_methods.index('v16:' + r)], labels, targets, groups) for r in RECIPES}
    if checks != c['candidate_metrics_primary']: raise ValueError('Primary calibration statistics do not reproduce')
    recipe = max(RECIPES, key=lambda r: (checks[r]['macro']['recall'], checks[r]['macro']['auroc'], -RECIPES.index(r)))
    if recipe != c['selected_recipe']: raise ValueError('Calibration recipe selection does not reproduce')
    selected_names = [a + ':' + recipe for a in CNN_ALIASES] + ['v15', 'raw48']
    for i, (alias, name) in enumerate(zip(ALIASES, selected_names)):
        if not np.array_equal(chosen['scores'][i], candidates['scores'][expected_methods.index(name)]):
            raise ValueError('Selected score row is not the declared candidate')
        metrics = aggregate(chosen['scores'][i], labels, targets, groups)
        if metrics != c['metrics'][alias] or not np.array_equal(metrics['thresholds'], chosen['thresholds'][i]):
            raise ValueError('Frozen calibration metrics/cutoffs do not reproduce')
    extra = {str(calibration_path.resolve()): sha(calibration_path),
             str((args.calibration / 'scores.npz').resolve()): sha(args.calibration / 'scores.npz'),
             str((args.calibration / 'candidate_scores.npz').resolve()): sha(args.calibration / 'candidate_scores.npz'),
             str(args.plan.resolve()): sha(args.plan),
             str((OLD / 'checkpoints/details_run01/families_1048576.pt').resolve()): sha(OLD / 'checkpoints/details_run01/families_1048576.pt')}
    for p in args.bind or []: extra[str(p.resolve())] = sha(p)
    protocol = {'format': 'hypernet.v16.frozen_comparison', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'test_scores_computed': False, 'methods': list(ALIASES), 'primary': 'v16',
                'model_selection_complete': True, 'test_scores_computed_before_freeze': False,
                'fresh_plan_sha256': sha(args.plan),
                'behavior_files': c['source_sha256'],
                'checkpoint_files': {str(Path(p).resolve()): sha(p) for p in [*c['models'].values(), OLD / 'checkpoints/details_run01/families_1048576.pt']},
                'models': c['models'], 'target_labels': c['target_labels'], 'selected_recipe': c['selected_recipe'],
                'reference_pixel_hashes_all19': c['reference_pixel_hashes_all19'],
                'all_reference_target_order': c['all_reference_target_order'],
                'runtime': c['runtime'],
                'calibration_path': str(args.calibration.resolve()), 'plan_path': str(args.plan.resolve()),
                'bindings': {**c['source_sha256'], **c['input_bindings'], **extra},
                'thresholds': {a: c['metrics'][a]['thresholds'] for a in ALIASES},
                'quality_policy': 'Previously frozen plan/source rule; no replacements or score-based exclusions.',
                'reference_images_per_target': 1, 'additional_calibration_events': c['events'],
                'test_admission_requires': 'This exact protocol hash, final acquired inventory hash and blind quality audit hash.'}
    write_new(args.output, protocol)
    print(json.dumps({'protocol': str(args.output), 'sha256': sha(args.output)}))


def fresh_rgb(root, event):
    record = read(root / 'image_records' / (event['event_id'] + '.json'))
    path = (root / event['image_path']).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Image path escapes data root')
    if sha(path) != record['crop_png_sha256']: raise ValueError('Acquired image changed')
    with Image.open(path) as im: rgb = np.asarray(im.convert('RGB')).copy()
    if hashlib.sha256(rgb.tobytes()).hexdigest() != record['crop_rgb_sha256']: raise ValueError('Decoded pixels differ')
    if is_constant_plot(rgb): raise ValueError('Blindly excluded constant plot reached scorer')
    return rgb


def test(args):
    if args.output.exists(): raise FileExistsError('Keep all scored runs; choose a new output')
    admission = read(args.admission); protocol_path = Path(admission['protocol_path'])
    if admission.get('test_scoring_authorized') is not True:
        raise ValueError('TEST scoring admission is not authorized')
    original_admission_sha = sha(args.admission); original_protocol_sha = sha(protocol_path)
    if sha(protocol_path) != admission['protocol_sha256']: raise ValueError('Protocol/admission binding mismatch')
    protocol = read(protocol_path); verify_files(protocol['bindings']); verify_files(admission['bindings'])
    if protocol['runtime'] != {'device': args.device, 'torch': str(torch.__version__), 'numpy': str(np.__version__)}:
        raise ValueError('Actual runtime differs from frozen calibration')
    if str(args.plan.resolve()) != protocol['plan_path']: raise ValueError('Wrong actual TEST plan')
    plan = read(args.plan); root = args.plan.parent; targets = protocol['target_labels']
    if admission['fresh_plan_sha256'] != sha(args.plan) or protocol['fresh_plan_sha256'] != sha(args.plan):
        raise ValueError('Actual TEST plan differs from admission')
    quality_path = root / 'blind_quality_audit.json'
    if sha(quality_path) != admission['blind_quality_audit_sha256']:
        raise ValueError('Blind quality audit differs from admission')
    quality = read(quality_path)
    if admission['effective_event_ids'] != quality['eligible_event_ids'] or admission['image_record_sha256'] != quality['image_record_sha256']:
        raise ValueError('Scoring admission changed the blindly eligible roster or records')
    for event_id, expected in admission['image_record_sha256'].items():
        p = (root / 'image_records' / (event_id + '.json')).resolve()
        if admission['bindings'].get(str(p)) != expected or sha(p) != expected:
            raise ValueError('Actual image record is not frozen in scoring admission')
    if targets != plan['target_labels']: raise ValueError('Target set differs')
    eligible = set(admission['effective_event_ids'])
    events = [r for r in plan['events'] if r['event_id'] in eligible]
    if len(events) != len(eligible): raise ValueError('Effective TEST roster mismatch')
    cohort = old_data(); all_targets = list(cohort.target_labels); indices = [all_targets.index(t) for t in targets]
    refs = [cohort.reference(t) for t in all_targets]
    if all_targets != protocol['all_reference_target_order'] or [hashlib.sha256(x.tobytes()).hexdigest() for x in refs] != protocol['reference_pixel_hashes_all19']:
        raise ValueError('Actual enrolled reference pixels differ from calibration')
    extractors = {a: FrozenRelations(protocol['models'][a], args.device) for a in CNN_ALIASES}
    banks = {a: e.enroll_many(refs)[0] for a, e in extractors.items()}
    raw_bank = np.stack([raw_pixel_vector(x) for x in refs])
    old_model = FrozenDetails(OLD / 'checkpoints/details_run01/families_1048576.pt', device=args.device, batch_size=64)
    old_provenance = old_model.provenance(); old_matcher = PatternMatcher(old_model, recipe='translation_006')
    old_banks = [old_matcher.enroll(x) for x in refs]
    recipe = protocol['selected_recipe']; thresholds = np.array([protocol['thresholds'][a] for a in ALIASES])
    values = np.empty((len(ALIASES), len(events), len(targets)), np.float64); pixels = []; started = time.perf_counter()
    args.output.mkdir(parents=True)
    with (args.output / 'event_scores.jsonl').open('x', encoding='utf-8') as log:
        for i, event in enumerate(events):
            if i % INFERENCE_BATCH_SIZE == 0:
                query_images = [fresh_rgb(root, r) for r in events[i:i + INFERENCE_BATCH_SIZE]]
                encoded_queries = {a: e.encode_many(query_images) for a, e in extractors.items()}
            position = i % INFERENCE_BATCH_SIZE
            rgb = query_images[position]; pixels.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            for j, a in enumerate(CNN_ALIASES):
                values[j, i] = score_vectors(banks[a], encoded_queries[a][position])[recipe][indices]
            old_scored = old_matcher.score_many(old_banks, old_model.encode(rgb, sampling_stride=4), recipes=('translation_006',))
            values[3, i] = [old_scored[k]['translation_006']['score'] for k in indices]
            values[4, i] = (raw_bank @ raw_pixel_vector(rgb))[indices]
            log.write(json.dumps({'event_id': event['event_id'], 'pixel_sha256': pixels[-1],
                                  'scores': {a: values[j, i].tolist() for j, a in enumerate(ALIASES)}}, separators=(',', ':')) + '\n')
            if (i + 1) % 32 == 0:
                log.flush(); progress('fresh test', i + 1, len(events), time.perf_counter() - started)
    labels = [r['label'] for r in events]; groups = [r['group_id'] for r in events]
    metrics = {a: aggregate(values[j], labels, targets, groups, thresholds[j]) for j, a in enumerate(ALIASES)}
    save_scores(args.output / 'scores.npz', values, ALIASES, events, targets, pixels, thresholds)
    for e in extractors.values(): e.assert_unchanged()
    if old_model.provenance() != old_provenance: raise RuntimeError('Old model changed')
    verify_files(protocol['bindings']); verify_files(admission['bindings'])
    if sha(args.admission) != original_admission_sha or sha(protocol_path) != original_protocol_sha:
        raise RuntimeError('The protocol or admission document changed during TEST')
    result = {'completed': True, 'stage': 'test', 'target_labels': targets, 'events': len(events),
              'time_groups': len(set(groups)), 'metrics': metrics, 'selected_recipe': recipe,
              'protocol_sha256': original_protocol_sha, 'admission_sha256': original_admission_sha,
              'scores_sha256': sha(args.output / 'scores.npz'), 'score_log_sha256': sha(args.output / 'event_scores.jsonl'),
              'model_state_unchanged': True, 'natural_weight_training_images': 0,
              'additional_real_calibration_events': protocol['additional_calibration_events'],
              'seconds': time.perf_counter() - started,
              'label_scope': 'Predeclared independent volunteer consensus, not machine classifier predictions or expert-confirmed physical truth.'}
    write_new(args.output / 'results.json', result)
    progress('fresh test', len(events), len(events), time.perf_counter() - started, complete=True)
    print(json.dumps({'complete': True, 'events': len(events), 'macro': {a: metrics[a]['macro'] for a in ALIASES}}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('calibration', 'freeze', 'test'), required=True)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', type=Path); p.add_argument('--calibration', type=Path)
    p.add_argument('--bind', type=Path, action='append'); p.add_argument('--admission', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = p.parse_args(); configure()
    if args.stage == 'calibration' and not args.models: p.error('--models required')
    if args.stage == 'freeze' and not args.calibration: p.error('--calibration required')
    if args.stage == 'test' and not args.admission: p.error('--admission required')
    {'calibration': calibration, 'freeze': freeze, 'test': test}[args.stage](args)


if __name__ == '__main__': main()
