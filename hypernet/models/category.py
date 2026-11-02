"""Image-conditioned category-presence adapters using frozen semantic features.

The encoder is the published ImageNet ResNet18, not a model learned from one
image in total. Only one support image is supplied per prediction. No labels,
text prompts, support/query annotations, or query-set statistics are inputs.

Source architecture/weights:
https://arxiv.org/abs/1512.03385
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

The crop search and partial-cell fusion below are experimental engineering
baselines, not reproductions of a published one-shot detector. Winning crops
are proposals; neither correspondence geometry nor object boundaries are verified.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from ..types import MatchResult, rgb_uint8
from .cnn import FrozenCNNMatcher, _tensor, image_digest


@dataclass(frozen=True)
class CategoryFeatures:
    """Frozen descriptors for each crop; parts are a 3 by 3 feature-cell grid."""
    descriptors: np.ndarray
    parts: np.ndarray
    strengths: np.ndarray
    boxes: tuple[tuple[int, int, int, int], ...]


def query_regions(shape) -> tuple[tuple[int, int, int, int], ...]:
    """Full image, four 2/3 corner crops, and nine overlapping 1/2 crops.

    Coordinates are integer pixel boundaries [x0,y0,x1,y1). Exact duplicate
    boxes caused by tiny images are omitted; no content or annotations are read.
    """
    h, w = int(shape[0]), int(shape[1])
    if h < 8 or w < 8:
        raise ValueError("Region images must be at least 8 pixels per axis")
    boxes = [(0, 0, w, h)]
    for fraction, count in [(2 / 3, 2), (.5, 3)]:
        cw, ch = max(8, round(w * fraction)), max(8, round(h * fraction))
        for y in np.linspace(0, h - ch, count).round().astype(int):
            for x in np.linspace(0, w - cw, count).round().astype(int):
                box = (int(x), int(y), int(x + cw), int(y + ch))
                if box not in boxes:
                    boxes.append(box)
    return tuple(boxes)


def _validated_boxes(image, boxes):
    h, w = image.shape[:2]
    boxes = ((0, 0, w, h),) if boxes is None else tuple(tuple(b) for b in boxes)
    if not boxes:
        raise ValueError("At least one crop is required")
    checked = []
    for b in boxes:
        if len(b) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in b):
            raise ValueError("Crop coordinates must be four integers")
        x0, y0, x1, y1 = map(int, b)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or min(x1-x0, y1-y0) < 8:
            raise ValueError("Crop must remain within image and have at least 8 pixels per axis")
        checked.append((x0, y0, x1, y1))
    return tuple(checked)


def _immutable(array):
    result = np.asarray(array, dtype=np.float32)
    result.setflags(write=False)
    return result


class ResNetCategoryFeatures:
    """One shared frozen encoder and bounded per-image/crop feature cache."""
    def __init__(self, weights_path=None, allow_download=False, device="cpu",
                 image_size=224, batch_size=14, cache_size=512):
        if batch_size < 1 or cache_size < 1:
            raise ValueError("batch_size and cache_size must be positive")
        self.base = FrozenCNNMatcher(mode="global", weights_path=weights_path,
                                     allow_download=allow_download, device=device,
                                     image_size=image_size, cache_size=1)
        self.batch_size = int(batch_size)
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.forward_crops = 0

    def extract(self, image, boxes=None):
        image = rgb_uint8(image)
        boxes = _validated_boxes(image, boxes)
        key = image_digest(image)
        if key not in self.cache:
            self.cache[key] = {}
        self.cache.move_to_end(key)
        entry = self.cache[key]
        missing = list(dict.fromkeys(b for b in boxes if b not in entry))
        torch = self.base.torch
        for start in range(0, len(missing), self.batch_size):
            batch_boxes = missing[start:start + self.batch_size]
            batch = torch.cat([_tensor(image[y0:y1, x0:x1], self.base.image_size, self.base.device)
                               for x0, y0, x1, y1 in batch_boxes])
            with torch.inference_mode():
                features = self.base.encoder(batch)
                global_vectors = torch.nn.functional.normalize(features.mean((2, 3)), dim=1)
                cells = torch.nn.functional.adaptive_avg_pool2d(features, (3, 3)).flatten(2).transpose(1, 2)
                strengths = torch.linalg.vector_norm(cells, dim=2)
                parts = torch.nn.functional.normalize(cells, dim=2)
                global_arrays = global_vectors.cpu().numpy()
                part_arrays = parts.cpu().numpy()
                strength_arrays = strengths.cpu().numpy()
            for i, b in enumerate(batch_boxes):
                entry[b] = (_immutable(global_arrays[i].copy()), _immutable(part_arrays[i].copy()),
                            _immutable(strength_arrays[i].copy()))
            self.forward_crops += len(batch_boxes)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return CategoryFeatures(
            _immutable(np.stack([entry[b][0] for b in boxes])),
            _immutable(np.stack([entry[b][1] for b in boxes])),
            _immutable(np.stack([entry[b][2] for b in boxes])), boxes)

    def provenance(self):
        return {**self.base.provenance(),
                "method": "shared_frozen_resnet18_category_features",
                "preprocessing": "Each full frame or proposed crop is resized to square; ImageNet mean/std. This differs from the published center-crop classification preprocessing.",
                "feature_grid": [3, 3],
                "target_weight_updates": 0,
                "query_weight_updates": 0,
                "query_feature_fitting": False,
                "category_text_input": False,
                "support_annotation_input": False,
                "query_annotation_input": False}


def _cell_centers(box):
    x0, y0, x1, y1 = box
    # Center of equal-width feature cells in original pixel-center coordinates.
    return np.array([[x0 + (x + .5) * (x1 - x0) / 3 - .5,
                      y0 + (y + .5) * (y1 - y0) / 3 - .5]
                     for y in range(3) for x in range(3)], dtype=float)


def _partial_support(reference_parts, query_parts, keep_fraction):
    """Partial reciprocal cosine score with a match-count support factor."""
    similarity = np.clip(reference_parts @ query_parts.T, -1., 1.)
    forward = similarity.argmax(axis=1)
    backward = similarity.argmax(axis=0)
    refs = np.flatnonzero(backward[forward] == np.arange(len(reference_parts)))
    queries = forward[refs]
    values = similarity[refs, queries]
    keep = max(1, math.ceil(len(reference_parts) * keep_fraction))
    order = np.argsort(-values, kind="stable")[:keep]
    score = (float(np.maximum(values[order], 0.).mean()) *
             math.sqrt(min(1., len(refs) / keep))) if len(refs) else 0.
    return score, refs[order], queries[order]


class CategoryMatcher:
    """Select global, automatic-region, or experimental semantic-cell evidence.

    The parts mode selects the strongest reference activation cells, without
    query fitting, then combines global and partial-cell cosine support within
    each crop. Activation strength is not a foreground mask.
    """
    MODES = ("global", "regions", "parts")

    def __init__(self, mode="global", *, extractor=None, weights_path=None,
                 allow_download=False, device="cpu", global_weight=.7,
                 support_parts=5, keep_fraction=.6):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if not 0 <= global_weight <= 1 or not 1 <= support_parts <= 9 or not 0 < keep_fraction <= 1:
            raise ValueError("Invalid fusion weight, support part count or keep fraction")
        self.mode = mode
        self.name = "category_" + mode
        self.extractor = extractor if extractor is not None else ResNetCategoryFeatures(
            weights_path=weights_path, allow_download=allow_download, device=device)
        self.global_weight = float(global_weight)
        self.support_parts = int(support_parts)
        self.keep_fraction = float(keep_fraction)
        self._references: OrderedDict[str, tuple] = OrderedDict()

    def prepare(self, reference):
        reference = rgb_uint8(reference)
        key = image_digest(reference)
        if key in self._references:
            self._references.move_to_end(key)
            return
        features = self.extractor.extract(reference)
        selected = np.argsort(-features.strengths[0], kind="stable")[:self.support_parts]
        selected.setflags(write=False)
        self._references[key] = (features, selected)
        while len(self._references) > 256:
            self._references.popitem(last=False)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        self.prepare(reference)
        ref, selected = self._references[image_digest(reference)]
        boxes = None if self.mode == "global" else query_regions(query.shape)
        q = self.extractor.extract(query, boxes)
        global_scores = np.clip(q.descriptors @ ref.descriptors[0], -1., 1.)
        scores = global_scores.copy()
        part_results = []
        if self.mode == "parts":
            for parts in q.parts:
                part_results.append(_partial_support(ref.parts[0, selected], parts, self.keep_fraction))
            partial_scores = np.array([r[0] for r in part_results])
            scores = self.global_weight * global_scores + (1 - self.global_weight) * partial_scores
        winner = int(np.argmax(scores))
        evidence: dict[str, Any] = {
            "mode": self.mode, "reference_images": 1, "target_weight_updates": 0,
            "query_weight_updates": 0, "category_text_input": False,
            "support_annotation_input": False, "query_annotation_input": False,
            "query_regions": len(q.boxes), "winning_region_index": winner,
            "global_cosine": float(global_scores[winner]),
            "score_is_probability": False, "geometry_verified": False,
            "localization_verified": False,
            "proposal_kind": "whole image" if self.mode == "global" else "fixed multiscale image crop",
        }
        reference_points, query_points = [], []
        if self.mode == "parts":
            part_score, ri, qi = part_results[winner]
            evidence.update(part_score=float(part_score), global_weight=self.global_weight,
                            selected_reference_cells=selected.tolist(), retained_part_matches=len(ri),
                            part_selection="Reference activation magnitude; not semantic foreground annotation",
                            correspondence_kind="Semantic feature-cell proposals; not verified physical points")
            reference_points = _cell_centers(ref.boxes[0])[selected[ri]].tolist()
            query_points = _cell_centers(q.boxes[winner])[qi].tolist()
        return MatchResult(float(scores[winner]), evidence,
                           points_reference=reference_points, points_query=query_points,
                           bbox_xyxy=None if self.mode == "global" else list(map(float, q.boxes[winner])))

    def provenance(self):
        base = self.extractor.provenance()
        return {
            "method": self.name,
            "representation": base,
            "external_training_images": base.get("external_training_images"),
            "external_pretraining": base.get("external_pretraining"),
            "target_reference_images": 1, "target_weight_updates": 0, "query_weight_updates": 0,
            "category_text_input": False, "support_annotation_input": False, "query_annotation_input": False,
            "query_feature_fitting": False, "score_is_probability": False,
            "region_policy": "Full image only" if self.mode == "global" else
                "Full + four 2/3 corner crops + nine half-size crops; up to14 after deduplication",
            "parameters": {"global_weight": self.global_weight if self.mode == "parts" else None,
                           "support_parts": self.support_parts if self.mode == "parts" else None,
                           "part_keep_fraction": self.keep_fraction if self.mode == "parts" else None},
            "score": {"global": "full-frame embedding cosine", "regions": "maximum crop embedding cosine",
                      "parts": "maximum fixed fusion of crop cosine and reference-selected partial reciprocal cells"}[self.mode],
            "limitations": [
                "One support image per category, not one image of total training.",
                "ImageNet pretraining may include evaluation categories; unseen-in-all-training is not established.",
                "Full reference can include background. Activation strength is not a semantic subject mask.",
                "Maximum region search creates extra false-acceptance opportunities; calibrate each variant on development categories.",
                "Winning crop is a proposal, not a verified object box.",
                "Fixed cell fusion is experimental and can suppress useful category variation; no accuracy gain is assumed."
            ],
        }

