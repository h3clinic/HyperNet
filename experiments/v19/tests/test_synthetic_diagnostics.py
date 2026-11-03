import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_synthetic import summarize, diagnostic_summary, result_summary, global_cutoff, wilson


def record(positive, full, sampled, *, index=0, point_count=10, point_accuracy=.8):
    return {'positive': positive, 'family': index%2,
        'full_score': full, 'full_top1_iou': .7 if positive else 0.,
        'sampled_score': sampled, 'sampled_top1_iou': .8 if positive else 0.,
        'oracle_score': sampled, 'oracle_top1_iou': 1. if positive else 0.,
        'oracle_target_logit': 4. if positive else None, 'oracle_near_logit': -1. if positive else None,
        'oracle_top_index': 0 if positive else 1, 'generic_proposal_best_iou': .9 if positive else 0.,
        'top100_best_iou': .7 if positive else 0., 'generic_proposal_count': 6487,
        'point_candidate_count': point_count if positive else 0, 'point_candidate_accuracy': point_accuracy,
        'point_loss': 1.2 if positive else 0., 'sampled_roi_logits': [4., -1.] if positive else [-1., -2.],
        'sampled_roi_quality_targets': [1., 0.] if positive else [0., 0.]}


def test_tied_dev_maxima_are_conservative_and_float64_threshold_survives():
    cutoff = global_cutoff([1.]*64, .05)
    result = summarize([record(False, 1., 0.) for _ in range(64)], 'full_score', 'full_top1_iou', cutoff)
    assert cutoff > 1. and result['false_alarms'] == 0
    assert result['positive_episodes'] == 0 and result['localized_recall'] is None


def test_each_protocol_calibrates_only_on_its_own_dev_scene_maxima():
    dev = [record(True, 20., 5.), record(False, 10., 1.), record(True, 21., 6., index=1), record(False, 11., 2., index=1)]
    test = [record(True, 22., 7.), record(False, 12., 1.)]
    result = result_summary({'dev': dev, 'test': test, 'compositional': test})
    full = result['protocols']['full_search']; sampled = result['protocols']['training_style_sampled64_only']
    assert full['cutoff'] > 11 and full['cutoff'] < 12
    assert sampled['cutoff'] > 2 and sampled['cutoff'] < 3
    assert full['splits']['dev']['false_alarms'] == 0
    assert full['splits']['test']['false_alarms'] == 1
    assert sampled['splits']['test']['false_alarms'] == 0
    # Applying a sampled-region cutoff to thousands of full-search candidates is
    # explicitly isolated as a diagnostic, not silently used for the benchmark.
    assert result['diagnostics']['dev']['full_search_at_sampled64_dev_cutoff']['false_alarms'] == 2


def test_correspondence_accuracy_is_count_weighted_and_oracle_near_separate():
    rows = [record(True, 5., 4., point_count=10, point_accuracy=.8),
            record(True, 6., 4., point_count=20, point_accuracy=.5), record(False, 0., 0.)]
    result = diagnostic_summary(rows)
    assert result['point_candidate_count'] == 30
    assert abs(result['point_candidate_accuracy']-.6) < 1e-12
    assert result['correct_above_near_fraction'] == 1.
    assert result['correct_vs_near_auroc'] == 1.
    assert result['generic_proposal_coverage_iou50'] == 1.


def test_small_negative_sample_uncertainty_not_zero_when_no_false_alarms():
    interval = wilson(0, 64)
    assert interval[0] == 0 and interval[1] > .05
