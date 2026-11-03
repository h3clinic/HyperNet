import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from hypernet.evaluation_metrics import aggregate, calibration_cutoff, ranking_metrics
from hypernet.rare_event_metrics import calibration_cutoff as previous_cutoff
from hypernet.rare_event_metrics import summary as previous_summary


def test_conservative_cutoff_matches_previous_rule_and_excludes_boundary_ties():
    scores = np.array([.9, .9, .8, .7])
    result = calibration_cutoff(scores, budget=2500)
    assert result['threshold'] == previous_cutoff(scores, 2500)['threshold']
    assert result['threshold'] > .9
    assert result['observed_false_alarms'] == 0
    assert calibration_cutoff(np.zeros(1000), 10)['threshold'] == np.nextafter(0., np.inf)


def test_tie_aware_ranking_matches_sklearn_for_reversed_and_tied_scores():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        labels = rng.random(47) > .62
        scores = rng.integers(-3, 4, 47).astype(float)
        auc, ap = ranking_metrics(scores, labels)
        assert auc == pytest.approx(roc_auc_score(labels, scores), abs=1e-14)
        assert ap == pytest.approx(average_precision_score(labels, scores), abs=1e-14)


def test_group_bincount_and_union_count_each_event_once_even_with_correct_alarm():
    scores = np.array([[.9, .8], [.7, .9], [.8, .1], [.2, .8], [.1, .1]])
    labels = ['a', 'b', 'a', 'other', 'other']
    groups = ['mixed', 'mixed', 'only_a', 'noise', 'noise']
    result = aggregate(scores, labels, ['a', 'b'], groups, thresholds=[.5, .5])
    assert result['per_target']['a']['true_positive'] == 2
    assert result['per_target']['a']['false_positive'] == 1
    assert result['per_target']['a']['negative_time_groups']['groups'] == 1
    assert result['per_target']['a']['negative_time_groups']['mixed_label_groups_excluded_from_group_rate'] == 1
    assert result['per_target']['b']['negative_time_groups']['groups_with_any_false_alarm'] == 1
    assert result['union']['wrong_target_alarm_events'] == 3
    assert result['union']['target_recovered_events'] == 3
    assert result['union']['time_groups_with_any_wrong_alarm'] == 2
    assert result['union']['background_events_with_any_alarm'] == 1


def test_each_target_matches_previous_metric_implementation_without_quadratic_group_scan():
    scores = np.array([[.5, .1], [.2, .9], [.5, .8], [.8, .7], [.1, .2], [.3, .8]])
    labels = np.array(['a', 'b', 'a', 'b', 'other', 'other'])
    groups = ['g1', 'g1', 'g2', 'g3', 'g4', 'g4']
    result = aggregate(scores, labels, ['a', 'b'], groups, thresholds=[.5, .7])
    for index, target in enumerate(['a', 'b']):
        old = previous_summary(scores[:, index], labels == target, result['thresholds'][index], groups)
        new = result['per_target'][target]
        for field in ['true_positive', 'false_positive', 'recall', 'precision', 'auroc', 'average_precision', 'false_alarms_per_10000_negatives']:
            assert new[field] == pytest.approx(old[field], abs=1e-14)
        assert new['independent_event_binomial_reference']['recall_ci95'] == old['independent_event_binomial_reference']['recall_ci95']
        assert new['negative_time_groups']['groups'] == old['negative_time_groups']['groups']
        assert new['negative_time_groups']['groups_with_any_false_alarm'] == old['negative_time_groups']['groups_with_any_false_alarm']


def test_unsupported_target_is_explicit_and_never_dropped_from_macro():
    result = aggregate(np.array([[.2, .8], [.8, .2], [.1, .3]]),
                       ['a', 'other', 'other'], ['a', 'missing'], ['g1', 'g2', 'g3'], [.5, .5])
    assert result['per_target']['missing']['recall'] is None
    assert result['per_target']['missing']['auroc'] is None
    assert result['per_target']['missing']['false_positive'] == 1
    assert result['macro']['recall'] is None
    assert result['macro']['auroc'] is None
    assert result['macro']['missing_positive_targets'] == ['missing']
    assert result['macro']['fixed_target_count'] == 2
    assert result['macro']['false_alarms_per10000'] is not None


def test_frozen_thresholds_are_not_refit_using_test_labels():
    scores = np.array([[.1, .8], [.2, .9], [.3, .7], [.4, .6]])
    first = aggregate(scores, ['a', 'a', 'b', 'b'], ['a', 'b'], ['1', '2', '3', '4'], [.25, .75])
    changed = aggregate(scores, ['b', 'b', 'a', 'a'], ['a', 'b'], ['1', '2', '3', '4'], [.25, .75])
    assert first['thresholds'] == changed['thresholds'] == [.25, .75]
    assert first['thresholds_fitted_here'] is False


def test_validation_rejects_nonfinite_scores_duplicate_targets_and_missing_calibration_negatives():
    with pytest.raises(ValueError):
        aggregate([[np.nan]], ['a'], ['a'], ['g'])
    with pytest.raises(ValueError):
        aggregate([[0., 1.]], ['a'], ['a', 'a'], ['g'], [0., 0.])
    with pytest.raises(ValueError):
        aggregate([[0.], [1.]], ['a', 'a'], ['a'], ['g', 'h'])
