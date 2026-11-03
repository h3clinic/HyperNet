import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from hypernet_v19.model import (DenseMatchNet, bilinear_sample, predicted_extent,
    pyramid_indices, resize_by_two, state_hash, FORMAT)
from hypernet_v19.training import collate, objective, correspondence_loss
from hypernet_v19.proposals import grid_boxes, box_iou, propose
from hypernet_v19.matcher import FrozenDenseMatcher

torch.set_num_threads(2)


def test_explicit_sampler_matches_grid_sample_forward_and_gradient():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 9, 7, dtype=torch.float64, requires_grad=True)
    points = torch.rand(2, 3, 11, 2, dtype=torch.float64)*20-2
    out = bilinear_sample(x, points, stride=2., origin=.5)
    xy = (points-.5)/2
    normalized = torch.stack([2*xy[..., 0]/6-1, 2*xy[..., 1]/8-1], -1)
    reference = F.grid_sample(x, normalized, padding_mode='border', align_corners=True)
    assert torch.allclose(out, reference, atol=1e-12)
    a = torch.autograd.grad(out.square().sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(reference.square().sum(), x)[0]
    assert torch.allclose(a, b, atol=1e-10)


def test_double_resize_matches_bilinear_and_no_batch_statistics():
    torch.manual_seed(2); x = torch.rand(2, 3, 32, 40)
    assert torch.allclose(resize_by_two(x, 2), F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False), atol=1e-6)
    model = DenseMatchNet().eval()
    with torch.no_grad():
        combined = model.encode(x)['dense']; separate = torch.cat([model.encode(row[None])['dense'] for row in x])
    assert torch.allclose(combined, separate, atol=2e-6)


def test_predicted_extent_empty_and_tiny_are_valid():
    logits = torch.full((2, 1, 16, 20), -10.)
    logits[1, 0, 7, 10] = 10
    boxes = predicted_extent(logits, 32, 40)[:, 0]
    assert torch.equal(boxes[0], torch.tensor([0., 0., 40., 32.]))
    assert torch.equal(boxes[1, 2:]-boxes[1, :2], torch.tensor([8., 8.]))
    assert torch.equal(pyramid_indices(torch.tensor([[[0., 0., 40., 40.], [0, 0, 96, 96], [0, 0, 200, 200]]])), torch.tensor([[2, 1, 0]]))


def test_reference_changes_score_and_geometry_has_finite_gradient():
    torch.manual_seed(3); model = DenseMatchNet()
    source = torch.randn(1, 1, 64, 64, requires_grad=True)
    region = torch.randn(1, 3, 64, 64, requires_grad=True)
    def tokens(x):
        return {'descriptors': F.normalize(x, dim=2), 'foreground': torch.full(x.shape[:2]+(64,), .7),
                'norms': torch.ones(x.shape[:2]+(64,)), 'boxes': torch.zeros(*x.shape[:2], 4)}
    score, diag = model.score_regions(tokens(source), tokens(region), diagnostics=True)
    changed = model.score_regions(tokens(source.flip(-1)), tokens(region))
    assert not torch.allclose(score, changed, atol=1e-7)
    gradient = torch.autograd.grad(diag['geometry_residual'].mean(), (source, region))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradient)


def test_batched_reference_scoring_matches_independent_references():
    torch.manual_seed(33); model = DenseMatchNet().eval()
    def tokens(batch, count):
        return {'descriptors': F.normalize(torch.rand(batch, count, 64, 64), dim=2),
                'foreground': torch.rand(batch, count, 64), 'norms': torch.rand(batch, count, 64),
                'boxes': torch.zeros(batch, count, 4)}
    support = tokens(3, 1); query = tokens(1, 5)
    expanded = {key: value.expand(3, *value.shape[1:]) for key, value in query.items()}
    with torch.no_grad():
        combined = model.score_regions(support, expanded)
        individual = torch.cat([model.score_regions({key: value[i:i+1] for key, value in support.items()}, query)
                                for i in range(3)], 0)
    assert torch.allclose(combined, individual, atol=2e-6)


def test_region_sampling_retains_within_old_cell_detail():
    # A mean-preserving change inside each old4x4 bank cell changes these point
    # samples. This tests a descriptor bottleneck, not arbitrary RGB invariance.
    model = DenseMatchNet(); raw = torch.ones(1, 64, 32, 32)
    changed = raw.clone(); changed[:, 0, 1::4, 1::4] += 2; changed[:, 0, 3::4, 3::4] -= 2
    assert torch.allclose(raw.reshape(1, 64, 4, 8, 4, 8).mean((3, 5)), changed.reshape(1, 64, 4, 8, 4, 8).mean((3, 5)))
    encoded = {'raw': raw, 'foreground_logits': torch.zeros(1, 1, 32, 32)}
    other = {'raw': changed, 'foreground_logits': encoded['foreground_logits']}
    boxes = torch.tensor([[[0., 0., 64., 64.]]])
    assert not torch.allclose(model.sample_regions(encoded, boxes)['descriptors'], model.sample_regions(other, boxes)['descriptors'])


def test_constant_images_have_finite_model_gradients():
    torch.manual_seed(4); model = DenseMatchNet()
    output = model(torch.full((1, 3, 32, 32), .5), torch.full((1, 3, 32, 32), .5),
                   torch.tensor([[[0., 0., 32., 32.], [2., 3., 20., 22.]]]))
    output['scores'].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_local_descriptor_receptive_field_is_bounded():
    torch.manual_seed(44); model = DenseMatchNet()
    rgb = torch.rand(1, 3, 64, 64, requires_grad=True)
    local_value = model.encode(rgb)['raw'][0, 0, 16, 16]
    gradient = torch.autograd.grad(local_value, rgb)[0].abs().sum((0, 1))
    y, x = torch.where(gradient > 1e-12)
    assert len(x) and int(x.max()-x.min()+1) <= 16 and int(y.max()-y.min()+1) <= 16
    assert torch.count_nonzero(gradient[:20]) == 0 and torch.count_nonzero(gradient[45:]) == 0


def test_absent_type_does_not_imply_every_local_part_unmatched():
    model = DenseMatchNet(); scores = torch.randn(2, 64, requires_grad=True)
    batch = {'boxes': torch.tensor([[[0., 0., 96., 96.]], [[0., 0., 48., 48.]]]),
             'positive': torch.tensor([False, False])}
    loss, accuracy = correspondence_loss(model, {'scores': scores}, batch)
    assert loss == 0 and accuracy == 0
    loss.backward(); assert torch.count_nonzero(scores.grad) == 0


def test_generic_grid_covers_varied_boxes_and_is_reference_independent():
    boxes = np.asarray([[11, 24, 37, 62], [70, 55, 160, 144], [20, 12, 210, 100], [90, 0, 144, 256]], np.float32)
    regions = grid_boxes(256, 256)
    assert (box_iou(regions, boxes).max(0) >= .5).all()
    assert np.array_equal(propose(np.zeros((64, 96, 3), np.uint8)), propose(np.zeros((64, 96, 3), np.uint8)))


def test_actual_episode_objective_and_descriptor_gradients():
    from hypernet_v19.curriculum import episode
    rows = [episode(i, split='dev', seed=1901, difficulty=.25) for i in (0, 1)]
    batch = collate(rows); again = collate(rows)
    assert torch.equal(batch['rois'], again['rois'])
    assert batch['quality_targets'][0].max() == 1 and batch['quality_targets'][1].max() == 0
    torch.manual_seed(5); model = DenseMatchNet()
    loss, stats = objective(model, batch, batch['rois']); loss.backward()
    assert np.isfinite(stats['loss']) and stats['positive_rois'] > 0
    assert model.local[0].weight.grad.abs().sum() > 0
    assert model.context[0].weight.grad.abs().sum() > 0
    assert model.score_head[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_frozen_checkpoint_guards_and_single_reference(tmp_path):
    model = DenseMatchNet(); path = tmp_path/'random.pt'
    torch.save({'format': FORMAT, 'state_dict': model.state_dict(), 'state_sha256': state_hash(model),
                'natural_training_images': 0, 'borrowed_pretrained_weights': False}, path)
    matcher = FrozenDenseMatcher(path, region_batch=32)
    bank = matcher.enroll(np.full((31, 57, 3), 120, np.uint8))
    assert bank['reference_images'] == 1 and bank['real_weight_updates'] == 0
    assert matcher.provenance()['natural_training_images'] == 0
    matcher.assert_unchanged()
    with torch.no_grad(): next(matcher.model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match='changed'): matcher.assert_unchanged()
