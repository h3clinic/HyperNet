"""Our synthetic CNN and independent reference adaptations in the old scorer."""
from pathlib import Path
from hashlib import sha256
import time

import torch

from ..registry import ReferenceCNNRouter
from ..training_v3 import TinyShapeCNN
from ..types import rgb_uint8
from ..training_procedural_reference import (
    FORMAT, load_procedural_checkpoint, source_provenance, tensor_state_sha256,
)
from .cnn import image_digest
from .spatial_reference import SpatialReferenceMatcher
from .region_shape import RegionShapeMatcher


class FrozenProceduralMatcher(SpatialReferenceMatcher):
    name = 'frozen_own_procedural_cnn_v09'

    def __init__(self, checkpoint_path, device='cpu', random_seed=None):
        self.device = str(device)
        self.enrolled_references = set()
        self.reference_bound = False
        with torch.random.fork_rng(devices=[]):
            if random_seed is not None:
                if isinstance(random_seed, bool) or int(random_seed) != random_seed:
                    raise ValueError('random_seed must be an integer')
                torch.manual_seed(int(random_seed))
            self.model = TinyShapeCNN(64)
        if random_seed is None:
            data, file_hash, tensor_hash = load_procedural_checkpoint(checkpoint_path)
            self.model.load_state_dict(data['state_dict'], strict=True)
            self.metadata = source_provenance(data, file_hash, tensor_hash)
            self.checkpoint_sha256 = file_hash
            self.checkpoint_path = str(Path(checkpoint_path).resolve())
        else:
            self.metadata = {
                'pretrained': False, 'pretraining_origin': 'random initialization; no pretraining',
                'random_seed': int(random_seed), 'natural_pretraining_images': 0,
                'externally_provided_pretrained_weights': False,
                'external_pretraining': False, 'external_training_images': 0,
                'learned_teacher': False, 'synthetic_rendered_image_exposures': 0,
                'synthetic_corrupted_view_exposures': 0,
                'initial_state_sha256': tensor_state_sha256(self.model.state_dict()),
            }
            self.checkpoint_sha256 = None
            self.checkpoint_path = None
        self.metadata.update(target_reference_images=1, real_training_source_images=0,
            reference_images_for_enrollment=1, additional_real_training_images=0,
            real_reference_weight_updates=0, target_weight_updates=0, query_weight_updates=0,
            inter_reference_weight_transfer=False, reference_specific_weight_adaptation=False)
        self._configure()

    def _configure(self):
        self.model.eval().to(self.device).requires_grad_(False)
        # Keep per-query evidence compact; full source/training records remain
        # available from matcher.provenance(), not copied into every score row.
        fields = ('pretrained', 'pretraining_origin', 'natural_pretraining_images',
                  'externally_provided_pretrained_weights', 'external_pretraining',
                  'external_training_images', 'target_reference_images', 'learned_teacher',
                  'source_checkpoint_sha256', 'source_tensor_sha256',
                  'synthetic_rendered_image_exposures', 'synthetic_corrupted_view_exposures',
                  'real_reference_weight_updates', 'query_weight_updates')
        self.region_matcher = RegionShapeMatcher(self._describe,
            {**{key: self.metadata[key] for key in fields if key in self.metadata},
             'descriptor': '64-D TinyShapeCNN; own procedural or explicit random control',
             'learned_descriptor_only': True}, device=self.device, cache_size=24)

    def _check_reference(self, reference):
        reference = rgb_uint8(reference)
        key = image_digest(reference)
        if self.reference_bound and key != self.metadata['reference_rgb_sha256']:
            raise ValueError('Checkpoint was adapted on a different reference image')
        return reference, key

    def prepare(self, reference):
        reference, key = self._check_reference(reference)
        self.region_matcher.prepare(reference)
        self.enrolled_references.add(key)

    def provenance(self):
        return {**super().provenance(), 'method': self.name,
                'distinct_enrolled_references': len(self.enrolled_references),
                'reference_specific_weight_adaptation': self.reference_bound,
                'one_reference_scope': 'One real source per prediction episode; shared own synthetic checkpoint'}

    def predict(self, reference, query):
        reference, key = self._check_reference(reference)
        query = rgb_uint8(query)
        self.enrolled_references.add(key)
        result = self.region_matcher.predict(reference, query)
        result.evidence.update(method=self.name, learned_descriptor_only=True,
            checkpoint_sha256=self.checkpoint_sha256, query_weight_updates=0,
            pretrained=self.metadata['pretrained'], natural_pretraining_images=0,
            externally_provided_pretrained_weights=False, external_pretraining=False,
            external_pretraining_field_meaning=self.metadata.get('external_pretraining_field_meaning',
                                                                 'No pretraining for random control'),
            real_reference_weight_updates=self.metadata['real_reference_weight_updates'],
            reference_rgb_sha256=key, target_reference_images=1)
        return result


class ProceduralReferenceMatcher(FrozenProceduralMatcher):
    name = 'adapted_own_procedural_cnn_v09'

    def __init__(self, checkpoint_path, device='cpu'):
        path = Path(checkpoint_path)
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('format') != FORMAT:
            raise ValueError('Invalid procedural reference checkpoint format')
        self.metadata = data['metadata']
        for key in ('externally_provided_pretrained_weights', 'external_pretraining',
                    'learned_teacher', 'inter_reference_weight_transfer'):
            if self.metadata.get(key) is not False:
                raise ValueError('Invalid reference provenance: ' + key)
        for key, value in {'natural_pretraining_images': 0, 'additional_real_training_images': 0,
                           'query_weight_updates': 0, 'real_training_source_images': 1,
                           'target_reference_images': 1}.items():
            if type(self.metadata.get(key)) is not int or self.metadata[key] != value:
                raise ValueError('Invalid reference budget: ' + key)
        if self.metadata.get('pretrained') is not True:
            raise ValueError('Adaptation requires own synthetic pretraining')
        if self.metadata.get('mode') != 'shape_pattern':
            raise ValueError('Unknown per-reference adaptation recipe')
        requested = self.metadata.get('requested_steps')
        actual = self.metadata.get('real_reference_weight_updates')
        if (type(requested) is not int or requested < 1 or type(actual) is not int or
                not 0 <= actual <= requested or actual != self.metadata.get('target_weight_updates') or
                actual != len(self.metadata.get('training_loss', []))):
            raise ValueError('Inconsistent reference adaptation update count')
        if self.metadata.get('initial_state_sha256') != self.metadata.get('source_tensor_sha256'):
            raise ValueError('Adaptation source tensor binding failed')
        if tensor_state_sha256(data['state_dict']) != self.metadata.get('final_tensor_sha256'):
            raise ValueError('Adapted tensor hash mismatch')
        self.device = str(device)
        with torch.random.fork_rng(devices=[]):
            self.model = TinyShapeCNN(64)
        self.model.load_state_dict(data['state_dict'], strict=True)
        self.checkpoint_sha256 = sha256(path.read_bytes()).hexdigest()
        self.checkpoint_path = str(path.resolve())
        self.enrolled_references = set()
        self.reference_bound = True
        self._configure()


class ProceduralReferenceRouter(ReferenceCNNRouter):
    name = 'own_procedural_reference_router_v09'

    def __init__(self, pretrained_checkpoint, checkpoint_dir, steps=120, seed=73, device='cpu'):
        if isinstance(steps, bool) or int(steps) != steps or steps < 1:
            raise ValueError('steps must be positive integer')
        super().__init__(checkpoint_dir, steps, seed, device)
        self.pretrained_checkpoint = Path(pretrained_checkpoint)
        data, self.source_file_hash, self.source_tensor_hash = load_procedural_checkpoint(self.pretrained_checkpoint)
        self.source_metadata = source_provenance(data, self.source_file_hash, self.source_tensor_hash)

    def prepare(self, reference):
        from .. import training_procedural_reference, training_v3, training_v4, training_v5
        from .. import gaussian_variation, training, types, procedural_pretraining
        from . import procedural_reference, region_shape, spatial_reference, cnn
        reference = rgb_uint8(reference)
        key = image_digest(reference)
        if key in self.matchers:
            return
        if sha256(self.pretrained_checkpoint.read_bytes()).hexdigest() != self.source_file_hash:
            raise ValueError('Procedural source checkpoint changed after router creation')
        modules = (training_procedural_reference, training_v3, training_v4, training_v5,
                   gaussian_variation, training, types, procedural_pretraining,
                   procedural_reference, region_shape, spatial_reference, cnn)
        fingerprint = sha256(b''.join(Path(module.__file__).read_bytes() for module in modules)).hexdigest()[:16]
        path = self.checkpoint_dir / (
            f'v09_{fingerprint}_{self.source_file_hash[:16]}_{key[:20]}_steps{self.steps}_seed{self.seed}.pt')
        started = time.perf_counter()
        record = ({'cached_checkpoint': True, 'checkpoint': str(path)} if path.exists() else
            training_procedural_reference.train_procedural_reference(reference, self.pretrained_checkpoint,
                path, steps=self.steps, seed=self.seed, device=self.device))
        matcher = ProceduralReferenceMatcher(path, self.device)
        for name, value in {'reference_rgb_sha256': key, 'seed': self.seed,
                            'requested_steps': self.steps, 'source_checkpoint_sha256': self.source_file_hash,
                            'source_tensor_sha256': self.source_tensor_hash, 'mode': 'shape_pattern'}.items():
            if matcher.metadata.get(name) != value:
                raise ValueError('Cached procedural model mismatch: ' + name)
        matcher.prepare(reference)
        self.matchers[key] = matcher
        self.training_records.append({'reference_key': key, 'training': record,
            'model': matcher.provenance(), 'elapsed_seconds': time.perf_counter() - started})

    def provenance(self):
        return {**super().provenance(), **self.source_metadata,
                'method': self.name, 'real_reference_weight_updates_requested': self.steps,
                'additional_real_training_images': 0, 'real_training_source_images_per_model': 1,
                'inter_reference_weight_transfer': False,
                'scope': 'Fresh copy of the same own synthetic checkpoint for each real reference; unchanged v07 adaptation recipe'}
