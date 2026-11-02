import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from hypernet.surface_encoder import (SurfaceEncoder, input_for_arm,
    surface_spectrum, geometry_targets, surface_loss)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('surface_train', ROOT/'scripts/train_surface_encoder.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
torch.set_num_threads(2)


def test_native_input_preserved_control_removes_subpixel_pattern():
    x = (torch.arange(128) % 2).float()[None, None, None, :].expand(1, 3, 128, 128).clone()
    assert input_for_arm(x, 'native128') is x
    y = input_for_arm(x, 'lowpass32')
    assert y.std() == 0 and x.std() > .4
    with pytest.raises(ValueError):
        SurfaceEncoder()(x[:, :, :32, :32])


def test_spectrum_distinguishes_fine_frequencies_destroyed_by_control():
    grid = torch.arange(128).float()
    a = (.5+.4*torch.cos(2*torch.pi*grid/2))[None, None, None, :].expand(1, 3, 128, 128)
    b = (.5+.4*torch.cos(2*torch.pi*grid/4))[None, None, None, :].expand(1, 3, 128, 128)
    mask = torch.ones(1, 128, 128)
    assert torch.allclose(input_for_arm(a, 'lowpass32'), input_for_arm(b, 'lowpass32'), atol=1e-5)
    target_a, target_b = surface_spectrum(a, mask), surface_spectrum(b, mask)
    assert torch.nn.functional.cosine_similarity(target_a, target_b).item() < .5


def test_geometry_excludes_background_and_loss_backpropagates():
    mask = torch.zeros(4, 128, 128)
    mask[:, 20:100, 20:100] = 1
    normals = torch.zeros(4, 3, 128, 128)
    normals[:, 2] = mask
    depth = mask*3
    targets = geometry_targets(mask, normals, depth)
    assert torch.allclose(targets[1][:, 2][targets[0][:, 0] > .9], torch.ones_like(targets[1][:, 2][targets[0][:, 0] > .9]))
    assert torch.isfinite(surface_spectrum(torch.zeros(4, 3, 128, 128), torch.zeros_like(mask))).all()
    x = torch.rand(4, 3, 128, 128)
    model = SurfaceEncoder()
    output = model(x)
    assert output['geometry'].shape == (4, 5, 64, 64)
    assert torch.allclose(output['embedding'].norm(dim=1), torch.ones(4), atol=1e-5)
    loss, _ = surface_loss(output, torch.tensor([0, 0, 1, 1]), surface_spectrum(x, mask), *targets)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def records_fixture():
    return [dict(split=split, family_index=f, object_id=f'{split}_{f}_{o}', view_index=v,
        surface_strength=s) for split, count in [('train', 32), ('heldout', 8)]
        for f in range(8) for o in range(count) for v in (0, 1) for s in (0., .03, .07)]


def test_training_plan_only_train_objects_consecutive_cross_view_positives():
    records = records_fixture()
    plan = runner.batch_plan(records, 20, 731)
    assert plan == runner.batch_plan(records, 20, 731)
    for batch in plan:
        assert len(batch) == 16
        for r, p in zip(batch[::2], batch[1::2]):
            assert records[r]['split'] == records[p]['split'] == 'train'
            assert records[r]['object_id'] == records[p]['object_id']
            assert records[r]['view_index'] != records[p]['view_index']
        assert len({records[i]['object_id'] for i in batch}) == 8
        assert len({records[i]['family_index'] for i in batch}) == 4


def test_family_protocol_has_one_support_and_excludes_same_objects():
    records = [r for r in records_fixture() if r['split'] == 'heldout']
    labels = np.array([r['family_index'] for r in records])
    results = runner.representation_metrics(np.eye(8)[labels], records)
    assert results['one_support_per_family_accuracy'] == 1
    assert results['balanced_cross_view_surface_match'] == .5
    # Identity-only descriptor should succeed at instance retrieval while carrying
    # no type information for different objects in one-support family recognition.
    ids = {name: i for i, name in enumerate(sorted({r['object_id'] for r in records}))}
    identity = np.eye(64)[[ids[r['object_id']] for r in records]]
    results = runner.representation_metrics(identity, records)
    assert results['cross_view_instance_retrieval'] == 1
    assert results['one_support_per_family_accuracy'] == .125


def test_constant_descriptors_receive_chance_not_index_order_wins():
    records = [r for r in records_fixture() if r['split'] == 'heldout']
    results = runner.representation_metrics(np.ones((len(records), 4)), records)
    assert results['one_support_per_family_accuracy'] == .125
    assert results['cross_view_instance_retrieval'] == 1/64
    assert results['balanced_cross_view_surface_match'] == .5
