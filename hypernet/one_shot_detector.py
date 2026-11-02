"""A 2D reference-conditioned CNN detector, initialized from scratch.

The query encoder sees the full scene. Its category-independent foreground map
proposes regions; a shared spatial descriptor compares each proposal to the one
enrolled reference. Query annotations are only training/evaluation targets.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import cv2
from .training_v3 import shape_channels

FORMAT = 'hypernet.one_reference_2d_detector.v11'
ROI_CELLS = 24


def expand_box(box, factor, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1)*factor, (y2-y1)*factor
    return [max(0., cx-w/2), max(0., cy-h/2), min(float(width), cx+w/2), min(float(height), cy+h/2)]


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0., right-left)*max(0., bottom-top)
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return overlap/max(area_a+area_b-overlap, 1e-12)


def quarter_turn_box(box, turns, height, width):
    x1, y1, x2, y2 = box
    for _ in range(turns % 4):
        x1, y1, x2, y2 = y1, width-x2, y2, width-x1
        height, width = width, height
    return [x1, y1, x2, y2]


def reference_views(references, boxes):
    """Four exact pixel permutations from each enrolled image; no interpolation."""
    height, width = references.shape[-2:]
    if height != width:
        raise ValueError('Reference view bank expects a square enrollment image')
    images, transformed_boxes = [], []
    for i, box in enumerate(boxes):
        for k in range(4):
            images.append(torch.rot90(references[i], k, (-2, -1)))
            transformed_boxes.append(quarter_turn_box(box, k, height, width))
    return torch.stack(images), transformed_boxes


def spatial_rois(feature, boxes, image_shape, cells=ROI_CELLS):
    """Sample full-scene CNN features at region-relative spatial coordinates.

    Conv3/pad1/stride2 keeps receptive-field centers at image pixel0.5. This
    mapping accounts for that origin, including at non-square image sizes.
    Sampling does not require or accept class labels or query ground-truth boxes.
    """
    if feature.shape[0] != 1:
        raise ValueError('Pass one feature map and its list of candidate boxes')
    boxes = torch.as_tensor(boxes, dtype=feature.dtype, device=feature.device).reshape(-1, 4)
    if len(boxes) == 0:
        return feature.new_empty((0, feature.shape[1], cells, cells))
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError('Region boxes must be finite and have positive area')
    height, width = image_shape
    fy, fx = feature.shape[-2:]
    fraction = (torch.arange(cells, device=feature.device, dtype=feature.dtype)+.5)/cells
    x = boxes[:, 0, None]+fraction*(boxes[:, 2]-boxes[:, 0])[:, None]
    y = boxes[:, 1, None]+fraction*(boxes[:, 3]-boxes[:, 1])[:, None]
    x = 2*((x-.5)/(width/fx)+.5)/fx-1
    y = 2*((y-.5)/(height/fy)+.5)/fy-1
    grid = torch.stack((x[:, None, :].expand(-1, cells, -1),
        y[:, :, None].expand(-1, -1, cells)), -1).reshape(1, -1, cells, 2)
    values = F.grid_sample(feature, grid, align_corners=False, padding_mode='border')
    return values[0].reshape(feature.shape[1], len(boxes), cells, cells).permute(1, 0, 2, 3)


class OneReferenceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.GELU())
        self.fine = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU())
        self.coarse = nn.Sequential(nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.GroupNorm(4, 24), nn.GELU(), nn.Conv2d(24, 24, 3, padding=1),
            nn.GroupNorm(4, 24), nn.GELU())
        # A native-resolution branch bypasses all strided convolutions. The24x24
        # region grid samples roughly every2–3pixels on the declared target sizes,
        # instead of collapsing the entire pattern into a few coarse cells.
        self.region_head = nn.Sequential(nn.Conv2d(48, 12, 1), nn.GELU(),
            nn.Flatten(), nn.Linear(12*ROI_CELLS*ROI_CELLS, 128))
        self.foreground_head = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1),
            nn.GELU(), nn.Conv2d(8, 1, 1))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.)))
        self.score_bias = nn.Parameter(torch.tensor(-5.))

    def encode_scene(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 32 or any(v % 4 for v in images.shape[-2:]):
            raise ValueError('Expected RGB NCHW images, dimensions at least32 and divisible by4')
        native = self.native(shape_channels(images))
        fine = self.fine(native)
        return dict(native=native, fine=fine, coarse=self.coarse(fine),
            foreground_logits=self.foreground_head(fine), image_shape=tuple(images.shape[-2:]))

    def describe(self, encoded, image_index, boxes):
        samples = [spatial_rois(encoded[key][image_index:image_index+1], boxes, encoded['image_shape'])
            for key in ('native', 'fine', 'coarse')]
        if len(samples[0]) == 0:
            return encoded['fine'].new_empty((0, 128))
        return F.normalize(self.region_head(torch.cat(samples, 1)), dim=1)

    def match_logits(self, reference, candidates):
        return self.similarities(reference, candidates)*self.log_scale.exp().clamp(max=30.)+self.score_bias

    def similarities(self, reference, candidates):
        if reference.ndim == 1:
            return candidates@reference
        return (candidates@reference.T).max(dim=1).values


def foreground_loss(logits, masks):
    target = F.interpolate(masks[:, None].float(), size=logits.shape[-2:], mode='area')
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(3.))
    probability = logits.sigmoid()
    dice = 1-(2*(probability*target).sum((1, 2, 3))+1)/(probability.sum((1, 2, 3))+target.sum((1, 2, 3))+1)
    return bce+.5*dice.mean()


def balanced_match_loss(logits, labels, include):
    labels = torch.as_tensor(labels, dtype=logits.dtype, device=logits.device)
    include = torch.as_tensor(include, dtype=torch.bool, device=logits.device)
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    groups = [losses[include & (labels == value)] for value in (0, 1)]
    groups = [group.mean() for group in groups if len(group)]
    return sum(groups)/len(groups) if groups else logits.sum()*0


def proposals_from_foreground(probability, scene_shape, threshold=.5, maximum=32):
    """Connected components of predicted foreground only; no query annotation.

    This proposal method is intended for the declared separated-object2D pilot.
    Occlusion/merged objects can reduce proposal recall and are reported limits.
    """
    height, width = scene_shape
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError('Expected a finite two-dimensional probability map')
    resized = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (resized >= threshold).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions = []
    for x, y, w, h, area in stats[1:]:
        if area >= 20 and w >= 8 and h >= 8 and w < .8*width and h < .8*height:
            regions.append((int(area), [int(x), int(y), int(x+w), int(y+h)]))
    return [box for area, box in sorted(regions, key=lambda pair: (-pair[0], pair[1]))[:maximum]]


@torch.inference_mode()
def detect(model, reference, query, reference_box):
    """One enrolled reference, full2D query, predicted boxes and scores only.

    Input tensors are single images 1x3xHxW. Reference localization is supplied
    with enrollment; no query box, mask, class label or target position is used.
    """
    if len(reference) != 1 or len(query) != 1:
        raise ValueError('Detection enrolls one reference and handles one scene')
    model.eval()
    views, ref_boxes = reference_views(reference, [reference_box])
    ref = model.encode_scene(views)
    search = model.encode_scene(query)
    rh, rw = reference.shape[-2:]
    qh, qw = query.shape[-2:]
    descriptor = torch.stack([model.describe(ref, i, [expand_box(box, 1.12, rw, rh)])[0]
        for i, box in enumerate(ref_boxes)])
    probability = search['foreground_logits'][0, 0].sigmoid().cpu().numpy()
    proposals = proposals_from_foreground(probability, (qh, qw))
    if not proposals:
        return dict(score=-1., bbox_xyxy=None, proposals=[], scores=[], foreground_probability=probability)
    features = model.describe(search, 0, [expand_box(box, 1.12, qw, qh) for box in proposals])
    scores = model.similarities(descriptor, features).cpu().numpy()
    winner = int(scores.argmax())
    return dict(score=float(scores[winner]), bbox_xyxy=proposals[winner], proposals=proposals,
        scores=scores.tolist(), foreground_probability=probability)
