import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison_analysis', ROOT / 'scripts/analyze_comparison.py')
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


def plan():
    value = analysis.read_json(ROOT / 'experiments/comparison_analysis_plan.json')
    value['valid_bootstrap_draws'] = 120; value['bootstrap_batch_size'] = 32
    return value


def data(prefix='test'):
    p = plan(); rng = np.random.default_rng(41)
    scores = rng.uniform(-1, 1, (5, 10, 2))
    scores[1:] = scores[0]  # Every paired contrast must be exactly zero.
    return {'scores': scores, 'event_ids': np.array([prefix + str(i) for i in range(10)]),
            'labels': np.array(['a', 'a', 'b', 'b', 'other', 'other', 'a', 'b', 'other', 'other']),
            'group_ids': np.array([prefix + str(i) for i in [0, 0, 1, 1, 2, 2, 3, 4, 5, 6]]),
            'targets': np.array(['a', 'b']), 'methods': np.array(p['methods']),
            'thresholds': np.zeros((5, 2)),
            'pixel_sha256': np.array([analysis.digest([prefix, i]) for i in range(10)])}


def test_paired_identical_methods_zero_intervals_and_deterministic_rejections():
    d = data(); result = analysis.paired_bootstrap(d, plan())
    assert result == analysis.paired_bootstrap(d, plan())
    assert result['valid_draws'] == 120 and result['rejected_draws'] > 0
    assert result['attempted_draws'] == 120 + result['rejected_draws']
    for contrast in result['contrasts'].values():
        for row in contrast.values():
            assert row['difference'] == 0 and row['ci95'] == [0, 0]


def test_bincount_weighting_matches_explicit_event_replication_with_mixed_groups():
    d = data(); groups, counts, positives, tp, fp = analysis.group_counts(d)
    weights = np.array([[2, 1, 0, 2, 1, 3, 1]], dtype=float)
    valid, recall, false_alarms = analysis.weighted_metrics(weights, counts, positives, tp, fp)
    assert valid[0]
    index = np.concatenate([np.tile(np.flatnonzero(d['group_ids'] == group), int(w)) for group, w in zip(groups, weights[0])])
    for i in range(5):
        row = analysis.aggregate(d['scores'][i, index], d['labels'][index], d['targets'], d['group_ids'][index], d['thresholds'][i])
        assert recall[0, i] == pytest.approx(row['macro']['recall'])
        assert false_alarms[0, i] == pytest.approx(row['macro']['false_alarms_per10000'])


def test_missing_target_is_unavailable_without_target_removal():
    d = data(); d['labels'][d['labels'] == 'b'] = 'other'
    result = analysis.paired_bootstrap(d, plan())
    assert result['available'] is False and result['missing_positive_targets'] == ['b']


def test_binding_and_end_to_end_saved_score_analysis(tmp_path):
    p = plan(); planpath = tmp_path / 'plan.json'; planpath.write_text(json.dumps(p))
    cal = data('cal'); test = data('test')
    for i in range(5):
        cal['thresholds'][i] = analysis.aggregate(cal['scores'][i], cal['labels'], cal['targets'], cal['group_ids'])['thresholds']
    test['thresholds'] = cal['thresholds'].copy()
    calpath = tmp_path / 'cal.npz'; testpath = tmp_path / 'test.npz'
    np.savez(calpath, **cal)
    binding = analysis.bind_calibration(calpath, testpath, planpath)
    bindingpath = tmp_path / 'binding.json'; analysis.save_new(bindingpath, binding)
    protocolpath = tmp_path / 'protocol.json'; protocolpath.write_text('{"frozen": true}')
    protocol = analysis.bind_protocol(protocolpath, planpath, bindingpath)
    protocolbinding = tmp_path / 'protocol_binding.json'; analysis.save_new(protocolbinding, protocol)
    np.savez(testpath, **test)
    result = analysis.analyze(calpath, testpath, planpath, bindingpath, protocolbinding)
    assert result['bootstrap']['available'] and result['no_model_or_image_calls']
    with pytest.raises(ValueError, match='already exists'):
        analysis.bind_calibration(calpath, testpath, planpath)
    with pytest.raises(ValueError, match='already exists'):
        analysis.bind_protocol(protocolpath, planpath, bindingpath)
    test['thresholds'][0, 0] += .01; np.savez(testpath, **test)
    with pytest.raises(ValueError, match='CALIB/TEST differ'):
        analysis.analyze(calpath, testpath, planpath, bindingpath, protocolbinding)
    protocolpath.write_text('{"frozen": false}')
    with pytest.raises(ValueError, match='protocol changed'):
        analysis.analyze(calpath, testpath, planpath, bindingpath, protocolbinding)


def test_calibration_binding_rejects_nonconforming_cutoffs_and_bad_npz(tmp_path):
    p = tmp_path / 'plan.json'; p.write_text(json.dumps(plan()))
    c = tmp_path / 'cal.npz'; t = tmp_path / 'test.npz'; d = data('cal')
    np.savez(c, **d)
    with pytest.raises(ValueError, match='cutoffs differ'):
        analysis.bind_calibration(c, t, p)
    d['event_ids'][1] = d['event_ids'][0]; np.savez(c, **d)
    with pytest.raises(ValueError, match='Duplicate event_ids'):
        analysis.load_scores(c)
    d = data(); d['scores'][0, 0, 0] = np.nan; np.savez(c, **d)
    with pytest.raises(ValueError, match='finite'):
        analysis.load_scores(c)


def test_analysis_never_overwrites_evidence(tmp_path):
    out = tmp_path / 'evidence.json'; analysis.save_new(out, {'a': 1})
    with pytest.raises(FileExistsError): analysis.save_new(out, {'a': 2})
    assert json.loads(out.read_text()) == {'a': 1}
