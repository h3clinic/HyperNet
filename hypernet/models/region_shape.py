"""Reference-conditioned region search with explicit spatial shape matching.

No category-specific proposal training, query boxes, text, or pretrained weights.
The fixed descriptor is a mathematical prior. Learned descriptors may be supplied
by a reference-bound scratch CNN; both use exactly the same tiles and scorer.
"""
from __future__ import annotations
from collections import OrderedDict
import math
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ..types import MatchResult, rgb_uint8
from ..training_v3 import analytic_shape_target
from .cnn import image_digest, _cached


def canonical_region_tiles(image, device='cpu'):
    """Whole region, 2x2 cells, then 3x3 cells; 14 RGB 32px tiles."""
    image = rgb_uint8(image)
    base = Image.fromarray(image).resize((96, 96), Image.Resampling.BILINEAR)
    tiles = []
    for n in (1, 2, 3):
        side = 96 // n
        for y in range(n):
            for x in range(n):
                tile = base.crop((x*side, y*side, (x+1)*side, (y+1)*side))
                tiles.append(np.asarray(tile.resize((32, 32), Image.Resampling.BILINEAR)).copy())
    return torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).to(device, dtype=torch.float32)/255.


def candidate_windows(height, width):
    """Deterministic multi-scale windows, independent of image content/labels."""
    if min(height, width) < 8:
        raise ValueError('Images must have at least 8 pixels per dimension')
    boxes = {(0, 0, width, height)}
    for scale in (1., .65, .4, .25):
        for aspect in (.6, 1., 1/.6):
            fw = min(1., scale*math.sqrt(aspect))
            fh = min(1., scale/math.sqrt(aspect))
            w, h = max(8, round(width*fw)), max(8, round(height*fh))
            w, h = min(w, width), min(h, height)
            nx = min(5, math.ceil((width-w)/max(1, .75*w))+1)
            ny = min(5, math.ceil((height-h)/max(1, .75*h))+1)
            for y in np.linspace(0, height-h, ny).round().astype(int):
                for x in np.linspace(0, width-w, nx).round().astype(int):
                    boxes.add((int(x), int(y), int(x+w), int(y+h)))
    # Full frame always first; tie breaks and descriptor caches are reproducible.
    return [(0, 0, width, height)] + sorted(boxes-{(0, 0, width, height)})


def _grid_score(reference, query, n, structured):
    """Symmetric part similarity; structure limits how far a part may move."""
    similarity = np.einsum('id,bjd->bij', reference, query, optimize=True)
    if structured:
        y, x = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        xy = np.column_stack([x.ravel(), y.ravel()])
        distance = np.abs(xy[:, None]-xy[None]).sum(-1)
        # One adjacent cell may move, penalized. Distant disconnected parts cannot
        # explain the support. No geometric transform is estimated from labels.
        penalty = np.where(distance <= 1, .12*distance, 2.).astype(np.float32)
        similarity = similarity-penalty[None]
    return .5*(similarity.max(2).mean(1)+similarity.max(1).mean(1))


def spatial_region_scores(reference_features, query_features, mode='structured'):
    """Return one score per candidate; [14,64] reference, [N,14,64] regions."""
    if mode not in {'structured', 'unordered', 'full'}:
        raise ValueError('Unknown region scoring mode')
    a, b = np.asarray(reference_features), np.asarray(query_features)
    if a.shape != (14,64) or b.ndim != 3 or b.shape[1:] != (14,64):
        raise ValueError('Expected 14 spatial tiles with 64 descriptor dimensions')
    whole = b[:,0] @ a[0]
    coarse = _grid_score(a[1:5], b[:,1:5], 2, mode != 'unordered')
    fine = _grid_score(a[5:], b[:,5:], 3, mode != 'unordered')
    return np.clip(.25*whole+.25*coarse+.5*fine, 0., 1.)


class RegionShapeMatcher:
    name = 'region_shape'

    def __init__(self, descriptor=None, provenance=None, device='cpu', mode='structured',
                 cache_size=96, batch_size=128):
        if mode not in {'structured','unordered','full'}:
            raise ValueError('Unknown region scoring mode')
        if cache_size < 1 or batch_size < 1:
            raise ValueError('Positive cache and batch size required')
        self.descriptor = descriptor or analytic_shape_target
        self.descriptor_provenance = provenance or {
            'external_training_images': 0, 'external_pretraining': False,
            'target_reference_images': 1, 'learned_parameters': 0,
            'target_weight_updates': 0, 'descriptor': 'fixed 64-D HOG/gray/edge prior'}
        if self.descriptor_provenance.get('external_training_images') != 0:
            raise ValueError('Region experiment forbids external training')
        self.device, self.mode = str(device), mode
        self.cache_size, self.batch_size = int(cache_size), int(batch_size)
        self.reference_cache, self.query_cache = OrderedDict(), OrderedDict()

    def provenance(self):
        return {**self.descriptor_provenance, 'method':self.name, 'region_mode':self.mode,
                'query_weight_updates':0, 'learned_teacher':False,
                'proposal_prior':'fixed sliding windows; scales 1,.65,.4,.25; 3 aspects; <=5x5 positions',
                'tiles':'whole + 2x2 + 3x3; horizontal support flip allowed',
                'spatial_prior':'symmetric adjacent-cell matching; .12 displacement penalty',
                'score':'max region of .25 whole + .25 coarse + .5 fine cosine',
                'score_is_probability':False, 'query_annotation_input':False}

    def with_mode(self, mode):
        other = RegionShapeMatcher(self.descriptor, self.descriptor_provenance, self.device,
                                   mode, self.cache_size, self.batch_size)
        other.reference_cache, other.query_cache = self.reference_cache, self.query_cache
        return other

    def _encode(self, tiles):
        features = []
        with torch.inference_mode():
            for start in range(0, len(tiles), self.batch_size):
                batch = self.descriptor(tiles[start:start+self.batch_size].to(self.device))
                if batch.ndim != 2 or batch.shape[1] != 64:
                    raise ValueError('Descriptor must return Nx64 tensor')
                features.append(F.normalize(batch, dim=1).cpu().numpy())
        result = np.concatenate(features)
        if not np.isfinite(result).all():
            raise ValueError('Descriptor returned nonfinite features')
        return result

    def _reference(self, reference):
        key = image_digest(reference)
        result = _cached(self.reference_cache, key)
        if result is None:
            tiles = torch.cat([canonical_region_tiles(reference),
                               canonical_region_tiles(reference[:,::-1].copy())])
            result = self._encode(tiles).reshape(2,14,64)
            _cached(self.reference_cache, key, result, self.cache_size)
        return result

    def _query(self, query):
        key = image_digest(query)
        result = _cached(self.query_cache, key)
        if result is None:
            h,w = query.shape[:2]
            boxes = candidate_windows(h,w)
            arrays = []
            # Bound temporary patch memory rather than accumulating all RGB tiles.
            for start in range(0,len(boxes),16):
                tiles = torch.cat([canonical_region_tiles(query[y0:y1,x0:x1])
                                   for x0,y0,x1,y1 in boxes[start:start+16]])
                arrays.append(self._encode(tiles).reshape(-1,14,64))
            result = (np.concatenate(arrays), boxes)
            _cached(self.query_cache, key, result, self.cache_size)
        return result

    def prepare(self, reference):
        self._reference(rgb_uint8(reference))

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        if not np.any(reference != reference[0,0]):
            return MatchResult(0., {'reason':'spatially_uniform_reference', 'external_pretraining':False})
        ref = self._reference(reference)
        features, boxes = self._query(query)
        if self.mode == 'full':
            features, boxes = features[:1], boxes[:1]
        scores = np.stack([spatial_region_scores(a,features,self.mode) for a in ref])
        orientation, best = np.unravel_index(np.argmax(scores), scores.shape)
        return MatchResult(float(scores[orientation,best]),
            {**self.provenance(), 'candidate_regions':len(boxes),
             'reference_flipped':bool(orientation), 'selected_region_index':int(best),
             'geometry_verified':False}, bbox_xyxy=[float(v) for v in boxes[best]])
