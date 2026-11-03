"""One-reference, fixed-axis comparison of independently learned shape types.

Native whole-image features and a reduced whole-image context view are kept
separate. The three recipes are fixed before calibration. No query labels,
test-time weight updates, borrowed features, or raw-pixel fusion are used.
"""
from pathlib import Path
import hashlib
import numpy as np
import cv2
import torch
from torch.nn import functional as F

RECIPES = ('mean', 'native', 'context128')
CONTEXT_SIDE = 128
INFERENCE_BATCH_SIZE = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rgb(rgb):
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('Expected uint8 HWC RGB')
    if min(a.shape[:2]) < 32 or a.shape[0] * a.shape[1] > 4_000_000:
        raise ValueError('Image dimensions outside the declared inference range')
    return a


def score_vectors(reference, query):
    """Cosines for each scale, and their fixed equal-weight mean."""
    result = {}
    for name in ('native', 'context128'):
        a = np.asarray(reference[name], dtype=np.float32)
        b = np.asarray(query[name], dtype=np.float32)
        if a.ndim != 2 or b.shape != (a.shape[1],) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Finite aligned reference and query vectors required')
        result[name] = np.clip(a @ b, -1, 1)
    result['mean'] = (result['native'] + result['context128']) * .5
    return {name: result[name] for name in RECIPES}


class FrozenRelations:
    """Frozen type embedding, with native input retained as one explicit arm."""
    def __init__(self, checkpoint, device='cuda'):
        from .relational_encoder import RelationalEncoder, FORMAT, state_hash
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.state_hash = state_hash
        saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if saved.get('format') != FORMAT:
            raise ValueError('Unexpected relational checkpoint format')
        if saved.get('natural_training_images') != 0 or saved.get('borrowed_pretrained_weights') is not False:
            raise ValueError('Expected mathematical-only training provenance')
        self.model = RelationalEncoder()
        self.model.load_state_dict(saved['state_dict'], strict=True)
        if state_hash(self.model) != saved['state_sha256']:
            raise ValueError('Checkpoint tensor hash differs')
        self.model.to(device).eval().requires_grad_(False)
        self.checkpoint_sha256 = sha(self.checkpoint)
        self.initial_state = state_hash(self.model)
        self.steps = saved['optimizer_steps']
        self.saved = {k: saved[k] for k in ('consumed_instances', 'consumed_types', 'training_views') if k in saved}

    @torch.inference_mode()
    def _embedding_batch(self, images):
        count = len(images)
        if not 1 <= count <= INFERENCE_BATCH_SIZE:
            raise ValueError('Expected a bounded inference batch')
        padded = list(images) + [images[-1]] * (INFERENCE_BATCH_SIZE - count)
        x = torch.from_numpy(np.stack(padded)).to(self.device).permute(0, 3, 1, 2).float() / 255
        # The model pads internally and crops descriptors back before pooling.
        # Padding here would incorrectly make border padding part of the scene.
        encoded = self.model(x)
        vector = F.normalize(encoded['embedding'], dim=1, eps=1e-8)[:count].cpu().numpy().copy()
        if vector.ndim != 2 or not np.isfinite(vector).all():
            raise ValueError('Invalid model embedding')
        return vector

    def encode(self, rgb):
        return self.encode_many([rgb])[0]

    def encode_many(self, images):
        images = [validate_rgb(x) for x in images]
        records = []
        for start in range(0, len(images), INFERENCE_BATCH_SIZE):
            group = images[start:start + INFERENCE_BATCH_SIZE]
            if len({x.shape for x in group}) != 1:
                # General callers can supply different dimensions; no rescaling
                # is introduced to make native inputs fit a shared batch.
                records.extend(self.encode(x) for x in group)
                continue
            small = [cv2.resize(x, (CONTEXT_SIDE, CONTEXT_SIDE), interpolation=cv2.INTER_AREA) for x in group]
            native = self._embedding_batch(group); context = self._embedding_batch(small)
            records.extend({'native': a, 'context128': b, 'rgb_sha256': hashlib.sha256(x.tobytes()).hexdigest()}
                           for x, a, b in zip(group, native, context))
        return records

    def enroll_many(self, references):
        records = self.encode_many(references)
        return {name: np.stack([r[name] for r in records]) for name in ('native', 'context128')}, [r['rgb_sha256'] for r in records]

    def provenance(self):
        return {'checkpoint_sha256': sha(self.checkpoint), 'state_sha256': self.state_hash(self.model),
                'optimizer_steps': self.steps, **self.saved, 'natural_weight_training_images': 0,
                'borrowed_pretrained_weights': False, 'reference_images_per_target': 1,
                'enrollment_weight_updates': 0, 'query_weight_updates': 0,
                'preprocessing': 'Native RGB plus independently encoded 128x128 INTER_AREA context; model-internal reflect padding and crop before pooling; RGB/255.',
                'recipes': list(RECIPES), 'mean_recipe': 'Equal mean of native and context embedding cosines; no raw-pixel fusion.',
                'inference_batch_size': INFERENCE_BATCH_SIZE,
                'tail_batch': 'Repeat the last input to fill the fixed batch; discard padded outputs. No normalization or matching across queries.',
                'axis_policy': 'No rotations, reflections, translation search or per-query alignment.'}

    def assert_unchanged(self):
        if self.state_hash(self.model) != self.initial_state or sha(self.checkpoint) != self.checkpoint_sha256:
            raise RuntimeError('Frozen model changed during evaluation')
