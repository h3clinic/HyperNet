import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/audit_2d_detector.py'
spec=importlib.util.spec_from_file_location('audit2d',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def row(episode,present,score,iou=1.):
    return dict(episode_id=episode,present=present,score=score,iou=iou if present else 0.,proposal_best_iou=1. if present else 0.)


def test_dev_ties_cannot_exceed_false_positive_budget():
    rows=[row(str(i),False,.5) for i in range(256)]
    cutoff=audit.dev_cutoff(rows)
    assert cutoff>.5 and not any(r['score']>=cutoff for r in rows)
    rows=[row(str(i),False,float(i)) for i in range(256)]
    cutoff=audit.dev_cutoff(rows)
    assert sum(r['score']>=cutoff for r in rows)==12


def test_counterbalanced_scene_only_predictor_stays_at_chance():
    rows=[row('a',True,.9),row('a',False,.2),row('b',True,.2),row('b',False,.9)]
    metrics=audit.independent_metrics(rows,.5)
    assert metrics['auroc']==.5 and metrics['balanced_accuracy']==.5
    assert metrics['paired_scene_ranking_accuracy']==.5


def test_localization_failure_does_not_count_as_successful_detection():
    rows=[row('a',True,.9,iou=.1),row('a',False,.2)]
    metrics=audit.independent_metrics(rows,.5)
    assert metrics['recall']==1 and metrics['localized_recall_at_iou50']==0
    assert metrics['single_box_ap50']==0 and metrics['proposal_recall_at_iou50']==1


def test_tied_score_ap_is_order_invariant_and_auroc_credits_half_ties():
    rows=[row('a',True,.8),row('a',False,.8),row('b',True,.8,iou=.1),row('b',False,.1)]
    first=audit.independent_metrics(rows,.5)
    second=audit.independent_metrics(list(reversed(rows)),.5)
    assert first==second
    assert first['auroc']==.75 and first['single_box_ap50']==pytest.approx(1/6)


def test_independent_iou_handles_empty_and_partial_overlap():
    assert audit.overlap(None,[0,0,5,5])==0
    assert audit.overlap([0,0,10,10],[5,0,15,10])==pytest.approx(1/3)
    assert audit.overlap([0,0,10,10],[10,0,20,10])==0


def test_duplicate_or_missing_counterbalanced_labels_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        audit.independent_metrics([row('a',True,.9),row('a',True,.8),row('a',False,.2)],.5)
    with pytest.raises(ValueError,match='Incomplete'):
        audit.independent_metrics([row('a',True,.9),row('b',False,.2)],.5)
