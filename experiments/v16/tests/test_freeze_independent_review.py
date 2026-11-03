"""Independent artificial-score regression checks; no real images or models."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluate_independent', ROOT / 'scripts/evaluate_v16.py')
evaluation = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)


@pytest.fixture
def case(tmp_path, monkeypatch):
    old = tmp_path / 'old'; (old / 'checkpoints/details_run01').mkdir(parents=True)
    (old / 'checkpoints/details_run01/families_1048576.pt').write_bytes(b'artificial-only')
    monkeypatch.setattr(evaluation, 'OLD', old)
    rows = [{'event_id': 'e' + str(i), 'label': ('a', 'b', 'other')[i % 3], 'group_id': 'g' + str(i // 2)} for i in range(12)]
    monkeypatch.setattr(evaluation, 'old_data', lambda: SimpleNamespace(query_events=lambda split: rows))
    target = ['a', 'b']; methods = [a + ':' + r for a in evaluation.CNN_ALIASES for r in evaluation.RECIPES] + ['v15', 'raw48']
    rng = np.random.default_rng(16); scores = rng.normal(size=(len(methods), len(rows), 2))
    labels = [r['label'] for r in rows]; groups = [r['group_id'] for r in rows]
    candidates = {r: evaluation.aggregate(scores[methods.index('v16:' + r)], labels, target, groups) for r in evaluation.RECIPES}
    recipe = max(evaluation.RECIPES, key=lambda r: (candidates[r]['macro']['recall'], candidates[r]['macro']['auroc'], -evaluation.RECIPES.index(r)))
    chosen = [a + ':' + recipe for a in evaluation.CNN_ALIASES] + ['v15', 'raw48']
    chosen_scores = np.stack([scores[methods.index(m)] for m in chosen])
    metrics = {a: evaluation.aggregate(chosen_scores[i], labels, target, groups) for i, a in enumerate(evaluation.ALIASES)}
    cal = tmp_path / 'calibration'; cal.mkdir(); plan = tmp_path / 'plan.json'; plan.write_text(json.dumps({'target_labels': target}))
    pixel = [str(i).zfill(64) for i in range(len(rows))]
    evaluation.save_scores(cal / 'scores.npz', chosen_scores, evaluation.ALIASES, rows, target, pixel, [metrics[a]['thresholds'] for a in evaluation.ALIASES])
    evaluation.save_scores(cal / 'candidate_scores.npz', scores, methods, rows, target, pixel, np.full((len(methods), 2), np.nan))
    result = {'completed': True, 'stage': 'calibration', 'fresh_plan_sha256': evaluation.sha(plan),
              'source_sha256': {}, 'input_bindings': {}, 'scores_sha256': evaluation.sha(cal / 'scores.npz'),
              'candidate_scores_sha256': evaluation.sha(cal / 'candidate_scores.npz'),
              'target_labels': target, 'candidate_metrics_primary': candidates, 'selected_recipe': recipe,
              'metrics': metrics, 'models': {}, 'reference_pixel_hashes_all19': pixel[:2],
              'all_reference_target_order': target, 'runtime': {}, 'events': len(rows)}
    resultpath = cal / 'results.json'; resultpath.write_text(json.dumps(result))
    return SimpleNamespace(calibration=cal, plan=plan, bind=[], output=tmp_path / 'protocol.json'), result, resultpath


def replace_npz(path, mutate):
    with np.load(path, allow_pickle=False) as archive: data = {key: archive[key].copy() for key in archive.files}
    mutate(data); np.savez_compressed(path, **data)


def test_valid_completed_calibration_freezes(case):
    args, _, _ = case; evaluation.freeze(args)
    protocol = evaluation.read(args.output)
    assert protocol['primary'] == 'v16'
    assert str((args.calibration / 'candidate_scores.npz').resolve()) in protocol['bindings']


def test_changed_archive_is_rejected_even_with_same_shape(case):
    args, _, _ = case
    replace_npz(args.calibration / 'scores.npz', lambda x: x['scores'].__setitem__((0, 0, 0), 10))
    with pytest.raises(ValueError, match='archive changed'): evaluation.freeze(args)


def test_incorrect_recipe_cannot_be_frozen(case):
    args, result, path = case
    result['selected_recipe'] = next(r for r in evaluation.RECIPES if r != result['selected_recipe'])
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='recipe selection'): evaluation.freeze(args)


def test_changed_cutoff_cannot_be_reauthorized_by_refreshing_archive_hash(case):
    args, result, path = case
    replace_npz(args.calibration / 'scores.npz', lambda x: x['thresholds'].__setitem__((0, 0), 10))
    result['scores_sha256'] = evaluation.sha(args.calibration / 'scores.npz'); path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='metrics/cutoffs'): evaluation.freeze(args)


def test_selected_scores_must_match_actual_recipe_candidate(case):
    args, result, path = case
    replace_npz(args.calibration / 'scores.npz', lambda x: x['scores'].__setitem__((0, 0, 0), 10))
    result['scores_sha256'] = evaluation.sha(args.calibration / 'scores.npz'); path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='declared candidate'): evaluation.freeze(args)


def test_changed_metrics_cannot_be_reauthorized(case):
    args, result, path = case
    result['metrics']['v16']['per_target']['a']['true_positive'] += 1; path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='metrics/cutoffs'): evaluation.freeze(args)
