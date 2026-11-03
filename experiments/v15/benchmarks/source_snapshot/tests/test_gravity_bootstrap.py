import importlib.util
from pathlib import Path
import numpy as np

SPEC = importlib.util.spec_from_file_location('gravity_analysis', Path(__file__).resolve().parents[1]/'scripts/analyze_gravity_spy_v15.py')
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def test_group_resampling_preserves_identical_paired_arms_and_rejects_absent_targets():
    positive = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 0.]])
    events = np.array([1., 1., 1., 1.])
    tp = np.repeat(positive[:, None, None, :], 2, axis=1)
    fp = np.zeros_like(tp)
    samples, counts = MODULE.resample(positive, events, tp, fp, draws=211, seed=1515)
    assert samples.shape == (211, 2, 1, 2)
    assert np.array_equal(samples[:, 0], samples[:, 1])
    assert np.all(samples[..., 0] == 1.)
    assert np.all(samples[..., 1] == 0.)
    assert counts['rejected_absent_positive_or_negative_draws'] > 0
    assert counts['attempted_draws'] == 211 + counts['rejected_absent_positive_or_negative_draws']
    assert counts['generated_draws'] == counts['attempted_draws'] + counts['unused_generated_draws_in_final_batch']
    repeated, repeated_counts = MODULE.resample(positive, events, tp, fp, draws=211, seed=1515)
    assert np.array_equal(repeated, samples) and repeated_counts == counts


def test_group_event_multiplicity_and_rate_units_are_preserved():
    positive = np.array([[2., 1.], [1., 2.]])
    events = np.array([3., 3.])
    tp = positive[:, None, None, :]
    fp = (events[:, None] - positive)[:, None, None, :]
    samples, counts = MODULE.resample(positive, events, tp, fp, draws=41, seed=1515)
    assert np.all(samples[..., 0] == 1.)
    assert np.all(samples[..., 1] == 10000.)
    assert counts['rejected_absent_positive_or_negative_draws'] == 0
