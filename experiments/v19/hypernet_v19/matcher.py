"""Frozen one-reference dense matching with generic region coverage."""
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from .model import DenseMatchNet, FORMAT, MODEL_CONFIG, state_hash, file_sha, parameter_summary
from .proposals import propose, nms, PROPOSAL_CONFIG

MATCHER_FORMAT = 'hypernet.dense_matcher.v19.1'


def checked_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError('Expected a nonempty RGB uint8 HWC image')
    return np.ascontiguousarray(image)


def prepare_image(image, support=False, max_side=384):
    rgb = checked_rgb(image); h, w = rgb.shape[:2]
    if max_side < 32: raise ValueError('max_side must be >=32')
    scale = 128/max(h, w) if support else min(1., max_side/max(h, w))
    rh, rw = max(1, round(h*scale)), max(1, round(w*scale))
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if support: oh = ow = 128; top = (128-rh)//2; left = (128-rw)//2
    else: oh = max(32, math.ceil(rh/4)*4); ow = max(32, math.ceil(rw/4)*4); top = left = 0
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    fill = np.median(border, axis=0).round().astype(np.uint8)
    canvas = np.broadcast_to(fill, (oh, ow, 3)).copy(); canvas[top:top+rh, left:left+rw] = resized
    return canvas, {'original_height': h, 'original_width': w, 'resized_height': rh, 'resized_width': rw,
                    'scale_x': rw/w, 'scale_y': rh/h, 'top': top, 'left': left,
                    'canvas_height': oh, 'canvas_width': ow}


class FrozenDenseMatcher:
    def __init__(self, checkpoint, device='cpu', max_side=384, proposal_limit=100, nms_iou=.5, region_batch=128):
        self.path = Path(checkpoint).resolve(); self.checkpoint_sha256 = file_sha(self.path)
        self.device = str(device); self.max_side = int(max_side); self.proposal_limit = int(proposal_limit)
        self.nms_iou = float(nms_iou); self.region_batch = int(region_batch)
        if self.max_side < 32 or self.proposal_limit < 1 or self.region_batch < 1 or not 0 <= self.nms_iou <= 1:
            raise ValueError('Invalid inference settings')
        self.saved = torch.load(self.path, map_location='cpu', weights_only=True)
        if self.saved.get('format') != FORMAT or self.saved.get('natural_training_images') != 0 or self.saved.get('borrowed_pretrained_weights') is not False:
            raise ValueError('Expected a v19 own synthetic-only checkpoint')
        self.model = DenseMatchNet().to(device)
        self.model.load_state_dict(self.saved['state_dict'], strict=True); self.model.eval().requires_grad_(False)
        self.initial_state = state_hash(self.model)
        if self.initial_state != self.saved['state_sha256']: raise ValueError('Checkpoint state hash mismatch')

    def _tensor(self, rgb):
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(self.device, dtype=torch.float32)/255

    @torch.inference_mode()
    def enroll(self, reference):
        rgb = checked_rgb(reference); image, meta = prepare_image(rgb, support=True, max_side=self.max_side)
        encoded = self.model.encode(self._tensor(image))
        return {'format': MATCHER_FORMAT, 'state_sha256': self.initial_state,
                'bank': self.model.support_bank(encoded), 'preprocessing': meta,
                'reference_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                'reference_images': 1, 'real_weight_updates': 0}

    @torch.inference_mode()
    def encode_query(self, query):
        rgb = checked_rgb(query); image, meta = prepare_image(rgb, max_side=self.max_side)
        rh, rw = meta['resized_height'], meta['resized_width']
        boxes = propose(image[:rh, :rw])
        pyramid = self.model.encode_query(self._tensor(image))
        groups = []
        for start in range(0, len(boxes), self.region_batch):
            region = torch.from_numpy(boxes[start:start+self.region_batch]).to(self.device)[None]
            groups.append(self.model.sample_query_regions(pyramid, region))
        # The shared query descriptors and region samples are independent of all
        # references and cached once. The large image feature pyramid can expire.
        regions = {key: torch.cat([group[key] for group in groups], 1) for key in groups[0]}
        original_boxes = boxes.copy()
        original_boxes[:, (0, 2)] /= meta['scale_x']; original_boxes[:, (1, 3)] /= meta['scale_y']
        original_boxes[:, (0, 2)] = original_boxes[:, (0, 2)].clip(0, meta['original_width'])
        original_boxes[:, (1, 3)] = original_boxes[:, (1, 3)].clip(0, meta['original_height'])
        return {'format': MATCHER_FORMAT, 'state_sha256': self.initial_state, 'regions': regions,
                'boxes': original_boxes, 'preprocessing': meta, 'proposal_count': len(boxes),
                'query_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}

    @torch.inference_mode()
    def predict(self, bank, query):
        return self.predict_many([bank], query)[0]

    @torch.inference_mode()
    def predict_many(self, banks, query):
        encoded = query if isinstance(query, dict) else self.encode_query(query)
        if not banks: return []
        for value in [*banks, encoded]:
            if value.get('format') != MATCHER_FORMAT or value.get('state_sha256') != self.initial_state:
                raise ValueError('Reference/query belongs to a different model')
        support = {key: torch.cat([bank['bank'][key] for bank in banks], 0) for key in banks[0]['bank']}
        scores = []
        for start in range(0, len(encoded['boxes']), self.region_batch):
            regions = {key: value[:, start:start+self.region_batch].expand(len(banks), -1, *value.shape[2:])
                       for key, value in encoded['regions'].items()}
            scores.append(self.model.score_regions(support, regions).float().cpu().numpy())
        scores = np.concatenate(scores, 1)
        if not np.isfinite(scores).all(): raise RuntimeError('Nonfinite region score')
        results = []
        for row in scores:
            selected = nms(encoded['boxes'], row, self.proposal_limit, self.nms_iou)
            results.append([{'box': encoded['boxes'][i].astype(float).tolist(), 'score': float(row[i])} for i in selected])
        return results

    def provenance(self):
        return {'format': MATCHER_FORMAT, 'model_format': FORMAT,
                'checkpoint_sha256': self.checkpoint_sha256, 'state_sha256': self.initial_state,
                'model_config': dict(MODEL_CONFIG), 'parameters': parameter_summary(self.model),
                'max_side': self.max_side, 'proposal_limit': self.proposal_limit, 'nms_iou': self.nms_iou,
                'region_batch': self.region_batch, 'proposals': dict(PROPOSAL_CONFIG),
                'reference_batching': 'All supplied independent reference banks batched over each region chunk',
                'support_preprocessing': '128px aspect-preserving resize and median-border letterbox',
                'query_preprocessing': 'Downscale only to max_side; aspect preserved; pad to multiple4',
                'score': 'Conditional IoU-quality logit; no confidence prefilter',
                'cache': 'Query pyramid and all sampled ROIs independent of reference; one encode per image',
                'training': self.saved.get('training', {}), 'natural_training_images': 0,
                'borrowed_pretrained_weights': False, 'real_weight_updates': 0, 'reference_images': 1}

    def assert_unchanged(self):
        if file_sha(self.path) != self.checkpoint_sha256 or state_hash(self.model) != self.initial_state:
            raise RuntimeError('Frozen checkpoint/model changed')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Frozen model became trainable')
