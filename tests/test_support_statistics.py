import inspect
import numpy as np
import pytest
import torch

from hypernet.models.region_shape import RegionShapeMatcher
from hypernet.support_statistics import foreground_part_weights, support_statistics


def _reference():
    y, x = np.mgrid[:80, :100]
    return np.stack(((3*x + y) % 256, (x + 4*y) % 256,
                     ((x//9 + y//7) % 2)*180), axis=-1).astype(np.uint8)


def test_support_statistics_deterministic_finite_and_nonmutating():
    torch.set_num_threads(1)
    image = _reference()
    before = image.copy()
    matcher = RegionShapeMatcher()
    first = support_statistics(image, matcher, samples=2)
    second = support_statistics(image, matcher, samples=2)
    np.testing.assert_array_equal(image, before)
    for name, shape in [('mu', (2,14,64)), ('precision', (2,14,64)),
                        ('original_features', (2,14,64)), ('part_weights', (2,14)),
                        ('mask96', (96,96))]:
        assert first[name].shape == shape
        assert np.isfinite(first[name]).all()
        np.testing.assert_array_equal(first[name], second[name])
    assert np.all((first['precision'] >= .25) & (first['precision'] <= 4))
    assert np.max(first['part_weights'], axis=1).max() / np.min(first['part_weights'], axis=1).min() <= 4.00001
    np.testing.assert_allclose(np.linalg.norm(first['mu'], axis=-1), 1, atol=1e-5)
    assert first['evidence']['variant_mode'] == 'shape_pattern'
    assert first['evidence']['learner_updates'] == 0
    assert first['evidence']['query_input'] is False
    assert first['evidence']['annotation_input'] is False


def test_uniform_reference_abstains_with_bounded_statistics():
    result = support_statistics(np.full((40,40,3), 120, np.uint8), RegionShapeMatcher(), samples=1)
    assert result['abstain']
    assert result['evidence']['foreground_fallback'] == 'spatially_uniform_reference'
    np.testing.assert_array_equal(result['mask96'], np.ones((96,96)))
    np.testing.assert_allclose(result['precision'], 1, atol=1e-6)
    np.testing.assert_allclose(result['part_weights'], 1, atol=1e-6)
    assert np.isfinite(result['mu']).all()


def test_foreground_fallback_and_fixed_control_share_exact_weights(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('controlled failure')
    monkeypatch.setattr('hypernet.support_statistics.cv2.grabCut', fail)
    prior = foreground_part_weights(_reference())
    full = support_statistics(_reference(), RegionShapeMatcher(), samples=1)
    assert prior['evidence']['foreground_fallback'] == 'grabcut_exception:RuntimeError'
    np.testing.assert_array_equal(prior['mask96'], np.ones((96,96)))
    np.testing.assert_array_equal(prior['part_weights'], full['part_weights'])


def test_no_query_input_or_learner_update():
    assert tuple(inspect.signature(support_statistics).parameters) == (
        'referenceRGB', 'region_matcher', 'samples', 'seed')
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                    torch.nn.Linear(3,64)).eval().requires_grad_(False)
    matcher = RegionShapeMatcher(descriptor=model)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    support_statistics(_reference(), matcher, samples=1)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_training_mode_descriptor_rejected_before_buffer_updates():
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.AdaptiveAvgPool2d(1),
                                torch.nn.Flatten(), torch.nn.Linear(3,64))
    before = model[0].running_mean.clone()
    with pytest.raises(ValueError, match='evaluation-mode'):
        support_statistics(_reference(), RegionShapeMatcher(descriptor=model), samples=1)
    torch.testing.assert_close(model[0].running_mean, before, rtol=0, atol=0)


@pytest.mark.parametrize('samples', [0, -1, True, 1.5])
def test_invalid_sample_count(samples):
    with pytest.raises(ValueError, match='positive integer'):
        support_statistics(_reference(), RegionShapeMatcher(), samples=samples)
