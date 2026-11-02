import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('train2d', ROOT/'scripts/train_2d_detector.py')
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def test_threshold_uses_dev_negatives_with_conservative_ties():
    rows = [dict(present=False, score=float(x)) for x in range(20)]
    threshold = driver.cutoff_from_dev(rows)
    assert sum(r['score'] >= threshold for r in rows) == 1
    rows = [dict(present=False, score=.5)]*20
    assert driver.cutoff_from_dev(rows) > .5


def test_counterbalanced_query_only_scores_cannot_solve_reference_task():
    rows = [dict(episode_id=str(reference), present=(reference==scene), score=[.2,.8][scene],
        iou=1. if reference==scene else 0., proposal_best_iou=1.)
        for reference in (0,1) for scene in (0,1)]
    scores = driver.summarize(rows, .5)
    assert scores['auroc'] == .5
    assert scores['paired_scene_ranking_accuracy'] == .5
    assert scores['balanced_accuracy'] == .5


def test_localization_failure_is_not_counted_as_detected_target():
    rows = [dict(episode_id='a', present=True, score=.9, iou=.1, proposal_best_iou=.8),
        dict(episode_id='a', present=False, score=.1, iou=0, proposal_best_iou=0)]
    metrics = driver.summarize(rows, .5)
    assert metrics['recall'] == 1
    assert metrics['localized_recall_at_iou50'] == 0
    assert metrics['single_box_ap50'] == 0
    assert metrics['proposal_recall_at_iou50'] == 1


def test_plan_visits_every_training_episode_and_never_heldout():
    episodes = [dict(split='train') for _ in range(16)]+[dict(split='dev'),dict(split='test')]
    a, b = driver.training_plan(episodes, 2), driver.training_plan(episodes, 2)
    assert a == b
    assert sorted(i for batch in a for i in batch) == sorted(list(range(16))*2)


def test_coarse_ablation_only_ignores_paired_micro_negative():
    row = dict(present=False, all_boxes=[
        dict(role='candidate', bbox_xyxy=[20,20,80,80]),
        dict(role='distractor', bbox_xyxy=[120,120,180,180])])
    boxes, labels, include, index = driver.proposal_targets(row, torch.zeros(256,256), np.random.default_rng(1))
    assert index == 0 and include[0] is False and all(include[1:])
    assert labels == [0.]*len(boxes)
    row['present'] = True
    _, labels, include, _ = driver.proposal_targets(row, torch.zeros(256,256), np.random.default_rng(1))
    assert labels[0] == 1 and all(include)
