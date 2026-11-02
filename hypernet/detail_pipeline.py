"""Frozen CNN proposals plus enrollment-conditioned native 2D evidence.

No inference API accepts query labels, target coordinates, masks, or family IDs.
The learned proposal network is our v11 synthetic-only CNN. All new matching
branches are analytic: this is a hybrid detector, not new learned CNN weights.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from .one_shot_detector import (OneReferenceDetector, FORMAT, reference_views,
    expand_box, proposals_from_foreground)
from .native_detail import describe_native, compare_native
from .gradient_detail import make_reference, compare as compare_gradient

METHODS = ('old_cnn', 'outline_only', 'centroid_only', 'local_moments',
           'lowres_patch', 'native_patch')
PRIMARY = 'native_patch'
CHECKPOINT_SHA256 = '15c64b81747f3ebec224ef19621793c2d24ed9dd6f93de25c6ad5f7c5c04d49a'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def tensor(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 HxWx3 RGB')
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()/255


class DetailPipeline:
    def __init__(self, checkpoint):
        if sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Requires the frozen, own synthetic-only v11 checkpoint')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (payload['format'] != FORMAT or payload['natural_training_images'] != 0
                or payload['externally_pretrained_weights'] is not False):
            raise ValueError('Checkpoint provenance does not meet this experiment')
        self.model = OneReferenceDetector()
        self.model.load_state_dict(payload['state_dict'])
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def prepare_reference(self, rgb, box):
        image = tensor(rgb)
        views, boxes = reference_views(image, [box])
        encoded = self.model.encode_scene(views)
        h, w = rgb.shape[:2]
        descriptors = torch.stack([self.model.describe(encoded, i,
            [expand_box(b, 1.12, w, h)])[0] for i, b in enumerate(boxes)])
        return dict(descriptors=descriptors, native=describe_native(rgb, box),
                    gradient=make_reference(rgb, box))

    @torch.inference_mode()
    def prepare_query(self, rgb):
        encoded = self.model.encode_scene(tensor(rgb))
        h, w = rgb.shape[:2]
        probability = encoded['foreground_logits'][0, 0].sigmoid().cpu().numpy()
        boxes = proposals_from_foreground(probability, (h, w))
        descriptors = self.model.describe(encoded, 0,
            [expand_box(b, 1.12, w, h) for b in boxes])
        return dict(rgb=np.asarray(rgb).copy(), proposals=boxes,
            descriptors=descriptors, native=[describe_native(rgb, b) for b in boxes])

    @torch.inference_mode()
    def score(self, reference, query):
        boxes = query['proposals']
        cnn = self.model.similarities(reference['descriptors'], query['descriptors']).cpu().numpy().tolist()
        evidence = {'old_cnn': [dict(score=v) for v in cnn]}
        evidence['outline_only'] = [compare_native(reference['native'], q, False) for q in query['native']]
        evidence['centroid_only'] = [compare_native(reference['native'], q, True, False) for q in query['native']]
        evidence['local_moments'] = [compare_native(reference['native'], q, True, True) for q in query['native']]
        for alias, stride in [('lowres_patch', 4), ('native_patch', 1)]:
            evidence[alias] = [compare_gradient(reference['gradient'], q, query['rgb'], stride)
                               for q in query['native']]
        results = {}
        for alias in METHODS:
            values = [float(e['score']) for e in evidence[alias]]
            best = int(np.argmax(values)) if values else None
            results[alias] = dict(score=values[best] if best is not None else (-1. if alias == 'old_cnn' else 0.),
                bbox_xyxy=boxes[best] if best is not None else None, proposals=boxes,
                proposal_scores=values, evidence=evidence[alias][best] if best is not None else {},
                all_evidence=evidence[alias])
        return results

    def detect(self, reference_rgb, reference_box, query_rgb):
        return self.score(self.prepare_reference(reference_rgb, reference_box), self.prepare_query(query_rgb))
