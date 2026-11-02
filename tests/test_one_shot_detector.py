import inspect
import numpy as np
import pytest
import torch
from hypernet.one_shot_detector import (OneReferenceDetector, spatial_rois,
    foreground_loss, balanced_match_loss, proposals_from_foreground, box_iou,
    expand_box, detect, reference_views, quarter_turn_box)

torch.set_num_threads(2)


def test_region_sampler_preserves_coordinates_from_shared_scene_features():
    # A spatial ramp has a known value at every requested image coordinate.
    x = .5+2*torch.arange(32).float()
    feature = x[None, None, None, :].expand(1, 1, 32, 32)
    regions = spatial_rois(feature, [[8, 8, 56, 56]], (64, 64), cells=8)
    expected = 8+(torch.arange(8)+.5)*6
    assert torch.allclose(regions[0, 0], expected[None].expand(8, 8), atol=1e-5)
    assert spatial_rois(feature, [], (64, 64)).shape == (0, 1, 24, 24)
    with pytest.raises(ValueError):
        spatial_rois(feature, [[5, 5, 4, 8]], (64, 64))


def test_proposals_are_prediction_only_and_translate_with_foreground():
    p = np.zeros((128, 128), np.float32)
    p[20:40, 30:60] = 1
    a = proposals_from_foreground(p, (256, 256))
    b = proposals_from_foreground(np.roll(np.roll(p, 4, 0), 8, 1), (256, 256))
    assert len(a) == len(b) == 1
    assert np.array_equal(np.asarray(b[0])-a[0], [16, 8, 16, 8])
    assert proposals_from_foreground(np.zeros_like(p), (256, 256)) == []
    assert list(inspect.signature(detect).parameters) == ['model', 'reference', 'query', 'reference_box']


def test_detector_training_has_finite_gradients_in_encoder_and_both_heads():
    torch.manual_seed(111)
    model = OneReferenceDetector()
    references, scenes = torch.rand(2, 3, 96, 96), torch.rand(2, 3, 256, 256)
    ref, query = model.encode_scene(references), model.encode_scene(scenes)
    descriptors = [model.describe(ref, i, [[8, 8, 88, 88]])[0] for i in range(2)]
    loss = foreground_loss(query['foreground_logits'], torch.rand(2, 256, 256))
    for i in range(2):
        candidates = model.describe(query, i, [[12, 20, 76, 84], [130, 140, 194, 204]])
        assert torch.allclose(candidates.norm(dim=1), torch.ones(2), atol=1e-5)
        loss = loss+balanced_match_loss(model.match_logits(descriptors[i], candidates), [1, 0], [True, True])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_hard_negative_ablation_excludes_only_declared_examples():
    logits = torch.tensor([1., -1., 2.], requires_grad=True)
    coarse = balanced_match_loss(logits, [1, 0, 0], [True, True, False])
    coarse.backward(retain_graph=True)
    assert logits.grad[2] == 0
    logits.grad.zero_()
    full = balanced_match_loss(logits, [1, 0, 0], [True, True, True])
    full.backward()
    assert logits.grad[2] > 0 and full > coarse


def test_detection_is_read_only_without_query_annotation_access():
    torch.manual_seed(111)
    model = OneReferenceDetector()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    output = detect(model, torch.rand(1, 3, 96, 96), torch.rand(1, 3, 256, 256), [8, 8, 88, 88])
    assert -1 <= output['score'] <= 1.000001
    assert len(output['scores']) == len(output['proposals'])
    assert all(torch.equal(before[k], v) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_box_boundaries_and_iou():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1/3)
    assert box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0
    assert expand_box([0, 0, 10, 10], 1.2, 10, 10) == [0, 0, 10., 10.]


def test_reference_bank_preserves_pixels_and_transforms_annotation_exactly():
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 4:12, 8:20] = 1
    views, boxes = reference_views(image, [[8,4,20,12]])
    assert views.shape == (4,3,32,32)
    for view, box in zip(views, boxes):
        y, x = torch.where(view[0] > 0)
        assert box == [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]
        assert torch.equal(view.sort().values.flatten().sort().values, image[0].flatten().sort().values)
    assert quarter_turn_box([8,4,20,12], 4, 32, 32) == [8,4,20,12]
