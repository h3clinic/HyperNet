"""Reference-only compressed shape/texture proposals and a scratch-CNN hybrid.

The HOG-inspired orientation histograms, uniform LBP histograms and compressed
intensity layout are fixed analytical cues, NOT learned CNN representations.
No labels, external images, learned codebooks, pretrained weights, query fitting,
or homography are used here. CNN training belongs to the wrapped reference model.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from ..types import MatchResult, rgb_uint8
from .cnn import image_digest


def region_boxes(shape, *, reference: bool = False) -> tuple[tuple[int, int, int, int], ...]:
    """Deterministic image-only proposals, with half-open pixel-boundary boxes."""
    h, w = map(int, shape[:2])
    if min(h, w) < 8:
        raise ValueError("Region images must be at least 8 pixels per axis")
    boxes = [(0, 0, w, h)]
    schedules = [(.8, 1), (.7, 2)] if reference else [(.75, 2), (.5, 3), (1 / 3, 4)]
    for fraction, count in schedules:
        cw, ch = max(8, round(w * fraction)), max(8, round(h * fraction))
        xs = [(w - cw) // 2] if count == 1 else np.rint(np.linspace(0, w - cw, count)).astype(int)
        ys = [(h - ch) // 2] if count == 1 else np.rint(np.linspace(0, h - ch, count)).astype(int)
        boxes.extend((int(x), int(y), int(x + cw), int(y + ch)) for y in ys for x in xs)
    return tuple(dict.fromkeys(boxes))


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.zeros_like(vector)


def _frozen(vector):
    result = np.array(vector, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RegionDescriptor:
    orientations: np.ndarray
    texture: np.ndarray
    compressed_shape: np.ndarray
    informative: bool


def describe_region(image: np.ndarray) -> RegionDescriptor:
    """Compress a crop and retain local edges, microtexture and coarse shape.

    The spatial orientation pyramid uses fixed isotropic centering, so equal
    orientation mass in every bin does not look like a distinctive shape.
    The texture code is rotation-invariant uniform LBP with ten bins.
    """
    image = rgb_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(gray, (3, 3), .65)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi) * (9. / np.pi)
    orientation = np.minimum(orientation, np.nextafter(np.float32(9), np.float32(0)))
    bins = np.floor(orientation).astype(np.int32)
    frac = orientation - bins
    orientations = []
    for grid in (1, 2, 4):
        cells = []
        step = 64 // grid
        for y in range(grid):
            for x in range(grid):
                cell = (slice(y * step, (y + 1) * step), slice(x * step, (x + 1) * step))
                b, f, m = bins[cell].ravel(), frac[cell].ravel(), magnitude[cell].ravel()
                hist = (np.bincount(b, weights=m * (1 - f), minlength=9)
                        + np.bincount((b + 1) % 9, weights=m * f, minlength=9)).astype(np.float32)
                if hist.sum() > 1e-8:
                    hist /= hist.sum()
                    hist -= 1 / 9
                cells.append(_unit(hist))
        orientations.append(np.concatenate(cells) / grid)
    orientations = _unit(np.concatenate(orientations))

    center = gray[1:-1, 1:-1]
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    bits = np.stack([gray[1 + dy:63 + dy, 1 + dx:63 + dx] >= center for dy, dx in offsets])
    transitions = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    codes = np.where(transitions <= 2, bits.sum(axis=0), 9)
    textures = []
    for grid in (1, 2):
        rows, cols = np.array_split(np.arange(62), grid), np.array_split(np.arange(62), grid)
        for ys in rows:
            for xs in cols:
                hist = np.bincount(codes[np.ix_(ys, xs)].ravel(), minlength=10).astype(np.float32)
                textures.append(np.sqrt(hist / max(1., float(hist.sum()))) / grid)
    texture = _unit(np.concatenate(textures))

    low = cv2.resize(smooth, (16, 16), interpolation=cv2.INTER_AREA)
    low = low - low.mean()
    low /= max(float(low.std()), 1e-6)
    dct = cv2.dct(low)[:4, :4].ravel()[1:]
    edge_layout = cv2.resize(magnitude, (4, 4), interpolation=cv2.INTER_AREA).ravel()
    edge_layout -= edge_layout.mean()
    compressed = _unit(np.concatenate([_unit(dct), _unit(edge_layout)]))
    informative = bool(gray.std() >= .015 and magnitude.mean() >= .015)
    return RegionDescriptor(_frozen(orientations), _frozen(texture), _frozen(compressed), informative)


def _component_matrices(reference, query):
    # Shape is query regions x reference regions throughout this module.
    def stack(name, descriptors):
        return np.stack([getattr(item, name) for item in descriptors])
    def signed(name):
        a, b = stack(name, query), stack(name, reference)
        valid = (np.linalg.norm(a, axis=1) > 1e-8)[:, None] & (np.linalg.norm(b, axis=1) > 1e-8)[None, :]
        return np.where(valid, (np.clip(a @ b.T, -1, 1) + 1) * .5, 0)
    texture = np.clip(stack("texture", query) @ stack("texture", reference).T, 0, 1)
    return {"orientations": signed("orientations"), "texture": texture,
            "compressed_shape": signed("compressed_shape")}


class ShapeTextureMatcher:
    """Fixed, interpretable single-reference cue bank, with no learned weights."""
    name = "shape_texture"

    def __init__(self, *, cache_size: int = 512, slice_penalty: float = .92):
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if not math.isfinite(slice_penalty) or not 0 < slice_penalty <= 1:
            raise ValueError("slice_penalty must be finite and in (0,1]")
        self.cache_size = cache_size
        self.slice_penalty = float(slice_penalty)
        self.cache = OrderedDict()

    def _bank(self, image, reference):
        image = rgb_uint8(image)
        key = (image_digest(image), bool(reference))
        if key not in self.cache:
            boxes = region_boxes(image.shape, reference=reference)
            descriptors = tuple(describe_region(image[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes)
            self.cache[key] = (boxes, descriptors)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def prepare(self, reference):
        self._bank(reference, True)

    def _region_scores(self, reference, query):
        ref_boxes, ref_bank = self._bank(reference, True)
        query_boxes, query_bank = self._bank(query, False)
        components = _component_matrices(ref_bank, query_bank)
        penalty = np.full(len(ref_bank), self.slice_penalty, np.float32)
        penalty[0] = 1.
        informative = np.array([d.informative for d in query_bank])[:, None] & np.array([d.informative for d in ref_bank])[None, :]
        scores = (.5 * components["orientations"] + .3 * components["texture"]
                  + .2 * components["compressed_shape"]) * penalty[None, :]
        scores = np.where(informative, scores, 0.)
        return ref_boxes, query_boxes, scores, components, informative, penalty

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        rb, qb, scores, components, informative, penalty = self._region_scores(reference, query)
        qi, ri = np.unravel_index(int(scores.argmax()), scores.shape)
        evidence = {
            "method": self.name, "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "target_weight_updates": 0,
            "handcrafted_cues_not_learned": True, "score_is_probability": False,
            "geometry_verified": False, "box_is_search_proposal": True,
            "reference_slices": len(rb), "query_windows": len(qb),
            "reference_slice_xyxy": list(rb[ri]), "reference_slice_penalty": float(penalty[ri]),
            "components": {key: float(value[qi, ri]) for key, value in components.items()},
            "component_weights": {"orientations": .5, "texture": .3, "compressed_shape": .2},
        }
        if not informative.any():
            evidence["reason"] = "insufficient shape or texture in reference/query regions"
        return MatchResult(float(scores[qi, ri]), evidence,
                           bbox_xyxy=list(map(float, qb[qi])) if informative.any() else None)

    def provenance(self):
        return {
            "method": self.name, "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "target_weight_updates": 0,
            "handcrafted_cues_not_learned": True, "score_is_probability": False,
            "input": "RGB pixels only; no labels, category strings or bounding boxes",
            "query_updates": 0, "normalization": "fixed per-crop formulas; no query fitting",
            "compressed_crop_size": 64, "gradient_bins": 9, "orientation_grids": [1, 2, 4],
            "texture": "rotation-invariant uniform 8-neighbor LBP, ten bins, grids 1 and 2",
            "compressed_shape": "4x4 low-frequency DCT without DC and 4x4 edge layout",
            "query_windows_max": 30, "reference_slices_max": 6,
            "component_weights": {"orientations": .5, "texture": .3, "compressed_shape": .2},
            "slice_penalty": self.slice_penalty,
            "sources": ["https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf",
                        "https://doi.org/10.1109/TPAMI.2002.1017623"],
            "limitation": "Appearance similarity, not learned category semantics; matching textures and backgrounds can fool it. Search boxes are not verified object detections.",
        }


def _indices_in_box(points, box):
    x0, y0, x1, y1 = box
    return np.flatnonzero((points[:, 0] >= x0) & (points[:, 0] < x1)
                         & (points[:, 1] >= y0) & (points[:, 1] < y1))


def _normalized_rows(values):
    values = np.asarray(values, np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("CNN features must be a finite N x D matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _cnn_region_evidence(ref_features, query_features, ref_points, query_points, ref_boxes, query_boxes):
    a, b = _normalized_rows(ref_features), _normalized_rows(query_features)
    pa, pb = np.asarray(ref_points, float), np.asarray(query_points, float)
    if a.shape[1] != b.shape[1] or pa.shape != (len(a), 2) or pb.shape != (len(b), 2):
        raise ValueError("CNN feature dimensions and point coordinates do not agree")
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        raise ValueError("CNN point coordinates must be finite")
    similarity = np.clip(a @ b.T, -1, 1)
    reference_indices = [_indices_in_box(pa, box) for box in ref_boxes]
    query_indices = [_indices_in_box(pb, box) for box in query_boxes]
    scores = np.zeros((len(query_boxes), len(ref_boxes)), np.float32)
    details = {}
    for qi, query_ids in enumerate(query_indices):
        for ri, ref_ids in enumerate(reference_indices):
            if len(ref_ids) < 2 or len(query_ids) < 2:
                continue
            local = similarity[np.ix_(ref_ids, query_ids)]
            forward, backward = local.argmax(axis=1), local.argmax(axis=0)
            mutual = np.flatnonzero(backward[forward] == np.arange(len(ref_ids)))
            values = local[mutual, forward[mutual]]
            # Support scales with the smaller visible region, not the full source.
            required = max(2, math.ceil(.35 * min(len(ref_ids), len(query_ids))))
            order = np.argsort(-values, kind="stable")[:required]
            partial = float(np.maximum(values[order], 0).mean() * math.sqrt(min(1, len(mutual) / required))) if len(order) else 0.
            pooled = float(np.clip(_unit(a[ref_ids].mean(axis=0)) @ _unit(b[query_ids].mean(axis=0)), 0., 1.))
            scores[qi, ri] = .5 * pooled + .5 * partial
            details[(qi, ri)] = {
                "pooled_cosine": pooled, "partial_reciprocal": partial,
                "retained_matches": len(order),
                "reference_ids": ref_ids[mutual[order]],
                "query_ids": query_ids[forward[mutual[order]]],
            }
    return scores, details


class ShapeTextureCNNMatcher:
    """Combine fixed shape/texture cues with an explicitly scratch-trained CNN.

    The supplied matcher can be bound to one reference or be a router exposing
    prepare(), _key() and matchers. Search-region pooling reads each image's CNN
    feature map once; it does not train on query images or forward every crop.
    cnn_weight=1 is the CNN-region-only ablation, with generic contrast gating.
    """
    name = "shape_texture_cnn"

    def __init__(self, base_matcher, *, shape_matcher=None, cnn_weight: float = .5):
        if not math.isfinite(cnn_weight) or not 0 <= cnn_weight <= 1:
            raise ValueError("cnn_weight must be finite and in [0,1]")
        self.base_matcher = base_matcher
        self.shape_matcher = shape_matcher if shape_matcher is not None else ShapeTextureMatcher()
        self.cnn_weight = float(cnn_weight)
        self._assert_no_pretraining(base_matcher)

    @staticmethod
    def _assert_no_pretraining(matcher):
        provenance = matcher.provenance()
        if provenance.get("external_training_images") != 0 or provenance.get("external_pretraining", False) not in (False, None):
            raise ValueError("Hybrid requires a verified zero-external-training-image CNN")

    def prepare(self, reference):
        reference = rgb_uint8(reference)
        if hasattr(self.base_matcher, "prepare"):
            self.base_matcher.prepare(reference)
        matcher = self._bound_matcher(reference)
        self._assert_no_pretraining(matcher)
        if matcher.metadata.get("reference_rgb_sha256") != image_digest(reference):
            raise ValueError("Checkpoint was trained on a different reference image")
        self.shape_matcher.prepare(reference)

    def _bound_matcher(self, reference):
        if hasattr(self.base_matcher, "_features"):
            return self.base_matcher
        if hasattr(self.base_matcher, "matchers") and hasattr(self.base_matcher, "_key"):
            return self.base_matcher.matchers[self.base_matcher._key(reference)]
        raise TypeError("CNN must expose reference-bound patch features or be a reference router")

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        self.prepare(reference)
        base = self._bound_matcher(reference)
        evidence = {
            "method": self.name, "cnn_method": base.name,
            "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "query_weight_updates": 0,
            "cnn_weight": self.cnn_weight, "shape_texture_weight": 1 - self.cnn_weight,
            "handcrafted_cues_not_learned": True, "score_is_probability": False,
            "geometry_verified": False, "box_is_search_proposal": True,
        }
        if base.metadata.get("insufficient_informative_patches", False):
            return MatchResult(0., {**evidence, "reason": "insufficient informative reference patches; scratch CNN abstained"})
        rb, qb, shape_scores, components, informative, penalty = self.shape_matcher._region_scores(reference, query)
        if not informative.any():
            return MatchResult(0., {**evidence, "reason": "insufficient shape or texture in reference/query regions"})
        a, pa = base._features(reference)
        b, pb = base._features(query)
        cnn_scores, details = _cnn_region_evidence(a, b, pa, pb, rb, qb)
        cnn_scores *= penalty[None, :]
        scores = np.where(informative, self.cnn_weight * cnn_scores + (1 - self.cnn_weight) * shape_scores, 0.)
        qi, ri = np.unravel_index(int(scores.argmax()), scores.shape)
        detail = details.get((qi, ri), {})
        evidence.update({
            "cnn_region_score": float(cnn_scores[qi, ri]), "shape_texture_score": float(shape_scores[qi, ri]),
            "components": {key: float(value[qi, ri]) for key, value in components.items()},
            "cnn_components": {key: value for key, value in detail.items() if not key.endswith("_ids")},
            "reference_slices": len(rb), "query_windows": len(qb),
            "reference_slice_xyxy": list(rb[ri]), "reference_slice_penalty": float(penalty[ri]),
            "feature_passes_requested": 2, "all_regions_share_full_image_cnn_features": True,
        })
        ref_ids, query_ids = detail.get("reference_ids", []), detail.get("query_ids", [])
        return MatchResult(float(scores[qi, ri]), evidence,
                           points_reference=np.asarray(pa)[ref_ids].astype(float).tolist(),
                           points_query=np.asarray(pb)[query_ids].astype(float).tolist(),
                           bbox_xyxy=list(map(float, qb[qi])))

    def provenance(self):
        return {
            "method": self.name, "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "query_weight_updates": 0,
            "cnn_weight": self.cnn_weight, "shape_texture_weight": 1 - self.cnn_weight,
            "handcrafted_cues_not_learned": True, "score_is_probability": False,
            "cnn": self.base_matcher.provenance(), "fixed_shape_texture": self.shape_matcher.provenance(),
            "cnn_region_score": "half mean-descriptor cosine and half partial reciprocal patch support",
            "inference": "one full-image CNN feature extraction per input, then pool existing patch descriptors in fixed regions",
            "limitation": "A hybrid of reference-trained CNN descriptors and hand-designed cues; neither proves semantic category learning from one image.",
        }

