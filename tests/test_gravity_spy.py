import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from hypernet import gravity_spy as gs


def test_average_precision_matches_sklearn():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=200); positive = rng.random(200) < .1
    assert gs.average_precision(scores, positive) == pytest.approx(average_precision_score(positive, scores))


def test_auroc_matches_sklearn_with_ties():
    scores = np.array([.1, .4, .4, .4, .9, .2, .9]); positive = np.array([0, 1, 0, 1, 1, 0, 0], bool)
    assert gs.auroc(scores, positive) == pytest.approx(roc_auc_score(positive, scores))


def test_threshold_accepts_at_most_allowed_negatives():
    negatives = np.arange(200, dtype=float)
    t = gs.threshold_at_false_alarm_rate(negatives, .01)
    assert (negatives >= t).sum() == 2
    assert (negatives >= gs.threshold_at_false_alarm_rate(negatives, 0)).sum() == 0


def test_operating_point_counts():
    out = gs.operating_point([.9, .8, .1, .7], [True, False, True, False], .75)
    assert out['found'] == 1 and out['false_alarms'] == 1 and out['false_alarms_per_10k'] == 5000


def test_references_come_only_from_train_and_are_reproducible():
    labels = np.array(['a'] * 6 + ['b'] * 6); splits = np.array(['train', 'test'] * 6)
    first = gs.choose_references(labels, splits, ['a', 'b'], 2, seed=0)
    assert first == gs.choose_references(labels, splits, ['a', 'b'], 2, seed=0)
    assert all(splits[i] == 'train' for rows in first.values() for i in rows)
    with pytest.raises(ValueError):
        gs.choose_references(labels, splits, ['a'], 4, seed=0)


def test_target_classes_exclude_catch_all_labels():
    classes, counts = gs.target_classes(['Blip', 'No_Glitch', 'None_of_the_Above', 'Chirp', 'Blip'])
    assert classes == ['Blip', 'Chirp'] and counts['Blip'] == 2


def test_one_shot_accuracy():
    rows = {'a': [.9, .1, .5], 'b': [.2, .8, .6]}
    assert gs.one_shot_accuracy(rows, ['a', 'b', 'a'], ['a', 'b']) == (pytest.approx(2 / 3), 3)


def test_bootstrap_interval_contains_mean():
    mean, (low, high) = gs.paired_class_bootstrap([.5, .6, .7, .2], [.1, .2, .3, .4])
    assert low <= mean <= high


def test_evaluate_method_perfect_and_cutoff_from_dev_only():
    labels = np.array(['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'])
    dev = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    perfect = {('a', 0): (labels == 'a').astype(float), ('b', 0): (labels == 'b').astype(float)}
    summary, records = gs.evaluate_method(perfect, labels, dev, ~dev, ['a', 'b'], ['a'], 1, .01)
    assert summary['mean_ap'] == 1 and summary['one_shot_accuracy_mean'] == 1
    assert all(r['test_at_dev_cutoff']['false_alarms'] == 0 for r in records)
    # A TEST negative scoring above every DEV negative still counts as a false alarm.
    leaky = {k: v.copy() for k, v in perfect.items()}; leaky[('a', 0)][5] = .5
    _, records = gs.evaluate_method(leaky, labels, dev, ~dev, ['a', 'b'], ['a'], 1, .01)
    assert records[0]['test_at_dev_cutoff']['false_alarms'] == 1


def test_dev_mean_ap_ignores_test_items():
    labels = np.array(['a', 'b', 'a', 'b']); dev = np.array([1, 1, 0, 0], bool)
    scores = {('a', 0): np.array([1., 0, 0, 1]), ('b', 0): np.array([0., 1, 1, 0])}
    assert gs.dev_mean_ap(scores, labels, dev, ['a', 'b'], 1) == 1


def test_to_uint8_rgb():
    rgb = gs.to_uint8_rgb(np.full((1, 4, 5), .5))
    assert rgb.shape == (4, 5, 3) and rgb.dtype == np.uint8 and rgb[0, 0, 0] == 128


def test_o3_png_conversion_matches_training_arrays():
    """PNG -> array must reproduce the v1.1 HDF5 arrays; uses locally cached official PNGs if present."""
    import os
    from pathlib import Path
    from hypernet import gravity_spy_o3 as o3
    raw = Path('C:/Users/brita/Documents/Codex/2026-11-01/wan/outputs/hypernet_v15/data/gravity_spy_v15_final/raw')
    data = Path('C:/Users/brita/Documents/Codex/2026-11-01/wan/gravity_spy_data')
    if not raw.exists() or not (data / 'trainingsetv1d1.h5').exists():
        pytest.skip('local Gravity Spy files not present')
    import h5py, pandas as pd
    meta = pd.read_csv(data / 'trainingset_v1d1_metadata.csv').set_index('gravityspy_id')
    checked = 0
    with h5py.File(data / 'trainingsetv1d1.h5', 'r') as f:
        for name in sorted(os.listdir(raw))[:60]:
            gid, dur = name.split('_')[0], name.split('_')[-1][:-4]
            if gid not in meta.index:
                continue
            a = f[meta.loc[gid, 'label']][meta.loc[gid, 'sample_type']][gid][f'{dur}.png'][()][0]
            if a.max() - a.min() < 1e-6:
                continue
            b = o3.png_to_array(raw / name)
            assert b.shape == (140, 170) and np.abs(b - a).max() < 2e-3
            checked += 1
            if checked == 5:
                break
    assert checked == 5


def test_o3_time_groups_chain_same_detector_only():
    import pandas as pd
    from hypernet import gravity_spy_o3 as o3
    t = pd.DataFrame({'ifo': ['H1', 'H1', 'H1', 'L1', 'H1'], 'event_time': [100.0, 103.0, 106.5, 103.0, 200.0]})
    g = o3.time_groups(t, seconds=4.0)
    assert g[0] == g[1] == g[2] and g[3] != g[0] and g[4] != g[0]
