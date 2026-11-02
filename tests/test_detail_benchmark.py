import numpy as np
import pytest
import torch

from hypernet.detail_benchmark import (FAMILIES, IMAGE_KEYS, build_detail_benchmark,
                                     evaluate_descriptor, teacher_collision_case)
from hypernet.training_v3 import analytic_shape_target


@pytest.mark.parametrize('mode', ['native32', 'downsample96'])
def test_generator_is_reproducible_balanced_and_visible(mode):
    first = build_detail_benchmark(seed=31, examples_per_family=6, mode=mode)
    second = build_detail_benchmark(seed=31, examples_per_family=6, mode=mode)
    assert first['provenance']['dataset_sha256'] == second['provenance']['dataset_sha256']
    assert len(first['cases']) == 24
    for key in IMAGE_KEYS:
        assert first[key].shape == (24, 32, 32, 3) and first[key].dtype == np.uint8
        assert np.array_equal(first[key], second[key])
    assert {r['family'] for r in first['cases']} == set(FAMILIES)
    assert all(r['foreground_area_matched'] for r in first['cases'])
    assert all(r['reference_changed_pixels'] > 0 for r in first['cases'])
    assert all(r['query_changed_pixels'] > 0 for r in first['cases'])
    assert all(not np.array_equal(r, p) for r, p in zip(first['reference'], first['positive']))
    assert {r['source_detail_pixels'] for r in first['cases']} == {1, 2, 3}
    assert all(r['nominal_final_detail_pixels'] == r['source_detail_pixels']/(3 if mode == 'downsample96' else 1)
               for r in first['cases'])


def test_nuisance_is_matched_for_positive_and_negative(monkeypatch):
    import hypernet.detail_benchmark as module
    original = module._render
    recorded = []
    def recording(obj, nuisance, size, color):
        recorded.append(nuisance.copy())
        return original(obj, nuisance, size, color)
    monkeypatch.setattr(module, '_render', recording)
    build_detail_benchmark(examples_per_family=1)
    for start in range(0, len(recorded), 4):
        assert recorded[start] == recorded[start+3]   # R/D
        assert recorded[start+1] == recorded[start+2] # P/N
        assert recorded[start] != recorded[start+1]


def test_reference_identity_is_balanced_within_severity_and_nuisance_cells():
    benchmark = build_detail_benchmark(examples_per_family=12)
    for family in FAMILIES:
        for detail in (1, 2, 3):
            for nuisance in ('backdrop_illumination', 'backdrop_illumination_translation'):
                selected = [r for r in benchmark['cases'] if r['family'] == family
                            and r['source_detail_pixels'] == detail and r['nuisance_group'] == nuisance]
                assert sorted(r['reference_identity'] for r in selected) == ['altered', 'base']


def test_benchmark_changes_with_seed_and_does_not_consume_torch_rng():
    state = torch.random.get_rng_state().clone()
    a = build_detail_benchmark(seed=41, examples_per_family=2)
    b = build_detail_benchmark(seed=42, examples_per_family=2)
    assert a['provenance']['dataset_sha256'] != b['provenance']['dataset_sha256']
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize('constant', [0., 1.])
def test_constant_descriptor_gets_chance_and_all_metrics_are_bounded(constant):
    benchmark = build_detail_benchmark(examples_per_family=3)
    result = evaluate_descriptor(lambda x: torch.full((len(x), 7), constant), benchmark, batch_size=5)
    assert result['metrics']['triplet_win'] == .5
    assert result['metrics']['tie_fraction'] == 1
    assert result['normalized_feature_variance_sum'] == pytest.approx(0)
    for key, value in result['metrics'].items():
        if key != 'n':
            assert 0 <= value <= 1


def test_known_feature_oracle_and_batching():
    benchmark = build_detail_benchmark(examples_per_family=1)
    n = len(benchmark['cases'])
    # Distinct image markers select features; expected comparisons are specified
    # independently of the benchmark evaluator's distance implementation.
    for kind, key in enumerate(IMAGE_KEYS):
        for i in range(n):
            benchmark[key][i].fill((kind*n+i+1)*10)
    table = torch.tensor([[1., 0.]]*n + [[1., 0.]]*n + [[0., 1.]]*n + [[0., 1.]]*n)
    def describe(x):
        ids = (x[:, 0, 0, 0]*255/10).round().long()-1
        return table[ids]
    a = evaluate_descriptor(describe, benchmark, batch_size=1)
    b = evaluate_descriptor(describe, benchmark, batch_size=7)
    assert a == b
    assert a['metrics']['triplet_win'] == 1
    assert a['metrics']['nuisance_distance'] == 0
    assert a['metrics']['detail_only_distance'] == .5
    assert a['metrics']['negative_distance'] == .5
    assert a['metrics']['detail_to_nuisance_ratio'] == 1


def test_old_teacher_has_real_within_cell_geometry_collision():
    example = teacher_collision_case()
    assert example['changed_pixels'] == 16
    pair = torch.stack([torch.from_numpy(example[k]).permute(2, 0, 1).float()/255
                        for k in ('reference', 'detail_only')])
    target = analytic_shape_target(pair)
    assert torch.linalg.vector_norm(target[0]-target[1]) < 1e-6
    # The distant anchor is unchanged; this is not a global image translation.
    assert torch.equal(pair[0, :, 20:24, 20:24], pair[1, :, 20:24, 20:24])
    assert not torch.equal(pair[0], pair[1])


def test_descriptor_contract_rejects_nonfinite_and_training_mode():
    benchmark = build_detail_benchmark(examples_per_family=1)
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate_descriptor(lambda x: torch.full((len(x), 2), float('nan')), benchmark)
    with pytest.raises(ValueError, match='NxD'):
        evaluate_descriptor(lambda x: x[:, 0], benchmark)
    with pytest.raises(ValueError, match='evaluation mode'):
        evaluate_descriptor(torch.nn.Dropout(), benchmark)


@pytest.mark.parametrize('kwargs', [dict(examples_per_family=0), dict(examples_per_family=True),
                                   dict(mode='unknown'), dict(image_size=64)])
def test_invalid_generator_configuration(kwargs):
    with pytest.raises(ValueError):
        build_detail_benchmark(**kwargs)
