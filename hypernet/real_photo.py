"""Real-photo input adapter for the exact project-trained v12 model.

Enrollment extracts descriptors from one boxed reference; no optimizer runs.
The four reported methods differ only in matcher and proposal source. The fixed
grid is a separately named diagnostic, never a silent fallback or a learned model.
"""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from .detail_pipeline import DetailPipeline, tensor, state_hash
from .one_shot_detector import expand_box, proposals_from_foreground
from .gradient_detail import describe_gradient_region, compare as compare_gradient

ALIASES = ('old_cnn', 'native_patch', 'old_cnn_grid', 'native_patch_grid')
PRIMARY = 'native_patch'
TRANSFERRED_CUTOFFS = {'old_cnn': .9806613922119142, 'native_patch': .41308703128766816,
                      'old_cnn_grid': .9806613922119142, 'native_patch_grid': .41308703128766816}
REFERENCE_SIDE = 96
QUERY_CNN_LONG_EDGE = 512
QUERY_DETAIL_LONG_EDGE = 1024
MAX_INPUT_PIXELS = 16_000_000


def validate_rgb(rgb):
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 HxWx3 RGB')
    h, w = image.shape[:2]
    if min(h, w) < 8 or h*w > MAX_INPUT_PIXELS:
        raise ValueError('Photo must be at least8pixels per side and at most16million pixels')
    return image


def validate_box(box, width, height):
    values = np.array(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Reference box must contain four finite coordinates')
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or min(x2-x1, y2-y1) < 2:
        raise ValueError('Reference box must lie inside the image and span at least2pixels')
    return values.tolist()


def resize_long_edge(rgb, maximum):
    h, w = rgb.shape[:2]
    ratio = min(1., maximum/max(h, w))
    nw, nh = max(1, round(w*ratio)), max(1, round(h*ratio))
    image = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA) if (nw, nh) != (w, h) else rgb.copy()
    return image, (nw/w, nh/h)


def map_boxes(boxes, sx, sy, width, height):
    """Map using actual rounded resize scales, clipping padding to real pixels."""
    mapped = []
    for x1, y1, x2, y2 in boxes:
        box = [float(np.clip(x1/sx, 0, width)), float(np.clip(y1/sy, 0, height)),
               float(np.clip(x2/sx, 0, width)), float(np.clip(y2/sy, 0, height))]
        if box[2] > box[0] and box[3] > box[1]:
            mapped.append(box)
    return mapped


def reference_image(rgb, box):
    h, w = rgb.shape[:2]
    box = validate_box(box, w, h)
    expanded = expand_box(box, 1.25, w, h)
    x1, y1 = map(math.floor, expanded[:2]); x2, y2 = map(math.ceil, expanded[2:])
    crop = rgb[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    ratio = REFERENCE_SIDE/max(cw, ch)
    nw, nh = max(1, round(cw*ratio)), max(1, round(ch*ratio))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0)
    fill = np.median(border, axis=0).astype(np.uint8)
    canvas = np.empty((REFERENCE_SIDE, REFERENCE_SIDE, 3), np.uint8); canvas[:] = fill
    ox, oy = (REFERENCE_SIDE-nw)//2, (REFERENCE_SIDE-nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = small
    sx, sy = nw/cw, nh/ch
    transformed = [(box[0]-x1)*sx+ox, (box[1]-y1)*sy+oy,
                   (box[2]-x1)*sx+ox, (box[3]-y1)*sy+oy]
    metadata = dict(original_size=[w, h], supplied_box=box, expanded_crop=[x1, y1, x2, y2],
        model_size=[REFERENCE_SIDE, REFERENCE_SIDE], model_box=transformed,
        actual_resize_scales=[sx, sy], padding_xy=[ox, oy], reference_images=1)
    return canvas, transformed, metadata


def grid_proposals(width, height):
    """51 predeclared overlapping windows; uses image dimensions only."""
    recipes = [(1., 1., 1, 1), (.75, .75, 2, 2), (.5, .5, 3, 3),
               (.25, .25, 5, 5), (.5, .75, 3, 2), (.75, .5, 2, 3)]
    boxes = []
    for fw, fh, nx, ny in recipes:
        bw, bh = width*fw, height*fh
        for y in np.linspace(0, height-bh, ny):
            for x in np.linspace(0, width-bw, nx):
                boxes.append([float(x), float(y), float(x+bw), float(y+bh)])
    assert len(boxes) == 51
    return boxes


class RealPhotoDetector:
    def __init__(self, checkpoint):
        self.base = DetailPipeline(Path(checkpoint))
        self.model = self.base.model

    @torch.inference_mode()
    def enroll(self, rgb, box=None):
        rgb = validate_rgb(rgb)
        h, w = rgb.shape[:2]
        normalized, transformed, metadata = reference_image(rgb, box if box is not None else [0, 0, w, h])
        enrolled = self.base.prepare_reference(normalized, transformed)
        gradient = enrolled['gradient']
        metadata['resolved_reference_foreground'] = gradient is not None
        metadata['reference_patch_count'] = len(gradient['patches']) if gradient is not None else 0
        enrolled['metadata'] = metadata
        return enrolled

    @torch.inference_mode()
    def predict(self, enrolled, rgb):
        rgb = validate_rgb(rgb)
        original_h, original_w = rgb.shape[:2]
        detail_rgb, (detail_sx, detail_sy) = resize_long_edge(rgb, QUERY_DETAIL_LONG_EDGE)
        dh, dw = detail_rgb.shape[:2]
        cnn_rgb, (cnn_sx, cnn_sy) = resize_long_edge(detail_rgb, QUERY_CNN_LONG_EDGE)
        ch, cw = cnn_rgb.shape[:2]
        # Padding is at most3pixels per side, except unusually narrow images
        # need at least32pixels for the inherited CNN's input contract.
        ph, pw = max(32, math.ceil(ch/4)*4), max(32, math.ceil(cw/4)*4)
        padded = np.pad(cnn_rgb, ((0, ph-ch), (0, pw-cw), (0, 0)), mode='edge')
        encoded = self.model.encode_scene(tensor(padded))
        probability = encoded['foreground_logits'][0, 0].sigmoid().cpu().numpy()
        predicted = proposals_from_foreground(probability, (ph, pw))
        cnn_boxes = []
        for b in predicted:
            clipped = [float(np.clip(b[0], 0, cw)), float(np.clip(b[1], 0, ch)),
                       float(np.clip(b[2], 0, cw)), float(np.clip(b[3], 0, ch))]
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                cnn_boxes.append(clipped)
        cnn_detail_boxes = map_boxes(cnn_boxes, cnn_sx, cnn_sy, dw, dh)
        groups = {'cnn': cnn_detail_boxes, 'grid': grid_proposals(dw, dh)}
        diagnostics = dict(original_query_size=[original_w, original_h], detail_size=[dw, dh],
            cnn_unpadded_size=[cw, ch], cnn_padded_size=[pw, ph],
            detail_actual_resize_scales=[detail_sx, detail_sy], cnn_actual_resize_scales=[cnn_sx, cnn_sy],
            cnn_proposal_count=len(cnn_detail_boxes), fixed_grid_proposal_count=51,
            predicted_foreground_fraction=float((cv2.resize(probability, (pw, ph))[:ch, :cw] >= .5).mean()),
            reference=dict(enrolled['metadata']), learned_weight_updates=0,
            reference_enrollment_only=True, borrowed_pretrained_weights=False)
        output = {}
        for source, boxes in groups.items():
            feature_boxes = [[b[0]*cnn_sx, b[1]*cnn_sy, b[2]*cnn_sx, b[3]*cnn_sy] for b in boxes]
            descriptors = self.model.describe(encoded, 0, [expand_box(b, 1.12, pw, ph) for b in feature_boxes])
            cnn_scores = self.model.similarities(enrolled['descriptors'], descriptors).cpu().numpy().tolist()
            descriptions = [describe_gradient_region(detail_rgb, b) for b in boxes]
            local = [compare_gradient(enrolled['gradient'], item, detail_rgb, 1) for item in descriptions]
            original_boxes = map_boxes(boxes, detail_sx, detail_sy, original_w, original_h)
            assert len(original_boxes) == len(boxes)
            for matcher, evidence in [('old_cnn', [dict(score=float(s)) for s in cnn_scores]), ('native_patch', local)]:
                alias = matcher+('_grid' if source == 'grid' else '')
                scores = [float(e['score']) for e in evidence]
                winner = int(np.argmax(scores)) if scores else None
                output[alias] = dict(score=scores[winner] if winner is not None else (-1. if matcher == 'old_cnn' else 0.),
                    bbox_xyxy=original_boxes[winner] if winner is not None else None,
                    proposals=original_boxes, proposal_scores=scores,
                    evidence=evidence[winner] if winner is not None else {}, all_evidence=evidence,
                    diagnostics=dict(diagnostics, proposal_source=source,
                        unresolved_region_count=sum(x is None for x in descriptions)))
        return {alias: output[alias] for alias in ALIASES}

    def detect(self, reference_rgb, reference_box, query_rgb):
        return self.predict(self.enroll(reference_rgb, reference_box), query_rgb)

    def provenance(self):
        return dict(reference_images_per_enrollment=1, learned_weight_updates=0,
            shared_checkpoint='own synthetic-only v11 CNN, reused by v12',
            synthetic_pretraining_unique_images=8192, borrowed_pretrained_weights=False,
            natural_pretraining_images=0, model_state_sha256=state_hash(self.model),
            reference_processing='1.25x boxed crop, aspect-preserving96px letterbox',
            query_processing='detail image maxedge1024; CNN maxedge512; boxes mapped to original pixels',
            diagnostic_grid='51 fixed windows, separate from learned CNN proposals')
