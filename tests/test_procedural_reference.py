"""Budget, reference isolation and inference immutability for own pretraining."""
from hashlib import sha256

import numpy as np
import pytest
import torch

from hypernet.procedural_pretraining import train_procedural
from hypernet.training_procedural_reference import (
    train_procedural_reference, load_procedural_checkpoint, tensor_state_sha256,
)
from hypernet.models.procedural_reference import (
    FrozenProceduralMatcher, ProceduralReferenceMatcher, ProceduralReferenceRouter,
)
from hypernet.models.cnn import image_digest


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp('procedural_source')
    path = root / 'source.pt'
    train_procedural(path, mode='detail_metric', steps=2, triplets_per_step=2, seed=73)
    return path


@pytest.fixture
def reference():
    y, x = np.mgrid[:48, :64]
    return np.stack(((x * 7 + y * 3) % 256, (x * y) % 256,
                     ((x // 4 + y // 6) % 2) * 255), axis=-1).astype(np.uint8)


def test_adaptation_keeps_v07_recipe_and_source_budget(source, tmp_path, reference):
    from hypernet.training_gaussian import train_reference_gaussian
    before = reference.copy()
    source_bytes = source.read_bytes()
    result = train_procedural_reference(reference, source, tmp_path / 'adapted.pt', steps=2)
    baseline = train_reference_gaussian(reference, tmp_path / 'baseline.pt', steps=2)
    assert result['target_weight_updates'] == result['real_reference_weight_updates'] == 2
    assert result['target_reference_images'] == result['real_training_source_images'] == 1
    assert result['additional_real_training_images'] == result['natural_pretraining_images'] == 0
    assert result['pretrained'] is True and result['externally_provided_pretrained_weights'] is False
    assert result['num_reference_patches'] == baseline['num_reference_patches'] == 297
    assert result['learning_rate'] == .002 and result['reference_derived_patch_views_seen'] == 128
    for key in ('patch_sequence_sha256', 'geometry_rng_state_sha256',
                'corruption_rng_state_sha256', 'patch_training_draw_counts'):
        assert result[key] == baseline[key]
    _, file_hash, tensor_hash = load_procedural_checkpoint(source)
    assert result['source_checkpoint_sha256'] == file_hash
    assert result['source_tensor_sha256'] == result['initial_state_sha256'] == tensor_hash
    assert result['final_tensor_sha256'] != tensor_hash
    assert result['synthetic_rendered_image_exposures'] == 12
    assert result['synthetic_corrupted_view_exposures'] == 24
    assert source.read_bytes() == source_bytes
    np.testing.assert_array_equal(reference, before)
    with pytest.raises(FileExistsError):
        train_procedural_reference(reference, source, tmp_path / 'adapted.pt', steps=2)


@pytest.mark.parametrize('key,value', [
    ('externally_provided_pretrained_weights', True),
    ('externally_provided_pretrained_weights', 0),
    ('externally_provided_pretrained_weights', None),
    ('natural_pretraining_images', 1), ('natural_pretraining_images', False),
    ('real_reference_images_seen', 1), ('learned_teacher', True),
    ('corrupted_view_exposures', 25),
])
def test_invalid_source_budgets_rejected(source, tmp_path, key, value):
    data = torch.load(source, weights_only=True)
    data['metadata'][key] = value
    path = tmp_path / 'invalid.pt'
    torch.save(data, path)
    with pytest.raises(ValueError):
        FrozenProceduralMatcher(path)


def test_frozen_and_random_never_update_or_call_optimizer(source, reference, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not construct an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    frozen = FrozenProceduralMatcher(source)
    random = FrozenProceduralMatcher(None, random_seed=73)
    random_again = FrozenProceduralMatcher(None, random_seed=73)
    assert tensor_state_sha256(random.model.state_dict()) == tensor_state_sha256(random_again.model.state_dict())
    query = np.roll(reference, 4, axis=1)
    before = {key: value.clone() for key, value in frozen.model.state_dict().items()}
    frozen.prepare(reference)
    answer = frozen.predict(reference, query)
    again = frozen.predict(reference, query)
    assert answer.score == again.score and 0 <= answer.score <= 1
    assert answer.bbox_xyxy == again.bbox_xyxy
    assert answer.evidence['real_reference_weight_updates'] == 0
    assert answer.evidence['pretrained'] is True
    assert all(torch.equal(before[key], value) for key, value in frozen.model.state_dict().items())
    assert not frozen.model.training and all(not value.requires_grad for value in frozen.model.parameters())
    assert random.provenance()['pretrained'] is False
    assert random.provenance()['synthetic_rendered_image_exposures'] == 0
    assert frozen.provenance()['target_reference_images'] == 1


def test_reference_router_is_independent_and_bound(source, tmp_path, reference, monkeypatch):
    router = ProceduralReferenceRouter(source, tmp_path / 'adaptations', steps=2)
    other = np.ascontiguousarray(reference[::-1])
    router.prepare(reference)
    router.prepare(other)
    a = router.matchers[image_digest(reference)]
    b = router.matchers[image_digest(other)]
    assert a.model is not b.model
    assert a.metadata['initial_state_sha256'] == b.metadata['initial_state_sha256'] == router.source_tensor_hash
    assert a.metadata['final_tensor_sha256'] != b.metadata['final_tensor_sha256']
    assert a.metadata['inter_reference_weight_transfer'] is False
    with pytest.raises(ValueError, match='different reference'):
        a.predict(other, reference)
    before = tensor_state_sha256(a.model.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError('Prepared inference cannot train')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    answer = router.predict(reference, other)
    assert np.isfinite(answer.score)
    assert tensor_state_sha256(a.model.state_dict()) == before
    assert answer.evidence['real_reference_weight_updates'] == 2
    assert len(router.training_records) == 2
    # A second router must reuse only correctly bound reference/source caches.
    cached = ProceduralReferenceRouter(source, tmp_path / 'adaptations', steps=2)
    cached.prepare(reference)
    assert cached.training_records[0]['training']['cached_checkpoint'] is True


def test_uniform_reference_abstains_and_preserves_source(source, tmp_path):
    reference = np.full((32, 48, 3), 123, dtype=np.uint8)
    path = tmp_path / 'blank.pt'
    data = train_procedural_reference(reference, source, path, steps=2)
    assert data['real_reference_weight_updates'] == 0
    assert data['initial_state_sha256'] == data['final_tensor_sha256']
    matcher = ProceduralReferenceMatcher(path)
    answer = matcher.predict(reference, reference)
    assert answer.score == 0 and answer.evidence['reason'] == 'spatially_uniform_reference'


def test_mutated_source_is_rejected_before_next_reference(source, tmp_path, reference):
    copied = tmp_path / 'copy.pt'
    copied.write_bytes(source.read_bytes())
    router = ProceduralReferenceRouter(copied, tmp_path / 'models', steps=2)
    data = torch.load(copied, weights_only=True)
    data['metadata']['seed'] += 1
    torch.save(data, copied)
    with pytest.raises(ValueError, match='changed'):
        router.prepare(reference)
