"""Adapt a fresh copy of our procedural CNN using exactly one real reference.

The per-reference recipe is v07 shape_pattern, including its RNG streams,
balanced 297-patch bank, restoration loss, optimizer and learning rate.
"""
from pathlib import Path
from hashlib import sha256
import json
import time

import numpy as np
import torch

from .types import rgb_uint8
from .training_v3 import TinyShapeCNN, analytic_shape_target
from .training_v4 import shared_geometry, corrupt_same_geometry, restoration_loss
from .training_v5 import reference_patch_bank, BalancedPatchCycle
from .procedural_pretraining import FORMAT as PRETRAINING_FORMAT, MODES

FORMAT = 'hypernet.procedural_reference_cnn.v09'


def tensor_state_sha256(state):
    """Hash tensor names, shapes, dtypes and CPU bytes, independent of file layout."""
    digest = sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_procedural_checkpoint(path):
    """Only the project's own zero-natural-image format is accepted."""
    path = Path(path)
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format') != PRETRAINING_FORMAT:
        raise ValueError('Expected our procedural pretraining checkpoint format')
    metadata = data.get('metadata', {})
    for key in ('natural_pretraining_images', 'natural_training_images',
                'query_images_seen', 'real_reference_images_seen'):
        if type(metadata.get(key)) is not int or metadata[key] != 0:
            raise ValueError('Procedural source must have zero ' + key)
    if metadata.get('externally_provided_pretrained_weights') is not False:
        raise ValueError('Externally provided pretrained weights are forbidden')
    if metadata.get('pretrained') is not True or metadata.get('mode') not in MODES:
        raise ValueError('Invalid own-pretraining mode or status')
    if metadata.get('pretraining_origin') != 'our procedural generator from random initialization':
        raise ValueError('Unknown pretraining origin')
    for key in ('learned_teacher', 'learned_external_teacher', 'external_teacher'):
        if key in metadata and metadata[key] is not False:
            raise ValueError('Learned external teacher is forbidden')
    expected_objective = ('Old restoration' if metadata['mode'] == 'restoration' else
        'Restoration+.25foreground mask/detail prediction+.2matched-nuisance triplet')
    if metadata.get('objective') != expected_objective:
        raise ValueError('Unknown procedural objective')
    for key in ('steps', 'triplets_per_step', 'rendered_image_exposures', 'corrupted_view_exposures'):
        if type(metadata.get(key)) is not int or metadata[key] < 1:
            raise ValueError('Missing positive procedural training budget: ' + key)
    steps, triplets = metadata['steps'], metadata['triplets_per_step']
    if (metadata['rendered_image_exposures'] != steps * triplets * 3 or
            metadata['corrupted_view_exposures'] != steps * triplets * 6):
        raise ValueError('Inconsistent procedural exposure budget')
    state = data['state_dict']
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite procedural model state')
    return data, sha256(path.read_bytes()).hexdigest(), tensor_state_sha256(state)


def source_provenance(data, file_hash, tensor_hash):
    metadata = data['metadata']
    return {
        'pretrained': True,
        'pretraining_origin': metadata['pretraining_origin'],
        'source_checkpoint_sha256': file_hash,
        'source_tensor_sha256': tensor_hash,
        'procedural_pretraining_mode': metadata['mode'],
        'procedural_pretraining': {key: value for key, value in metadata.items()
                                  if key not in ('loss_history', 'elapsed_seconds')},
        'synthetic_rendered_image_exposures': metadata['rendered_image_exposures'],
        'synthetic_corrupted_view_exposures': metadata['corrupted_view_exposures'],
        'natural_pretraining_images': 0,
        'externally_provided_pretrained_weights': False,
        'external_pretraining': False,
        'external_pretraining_field_meaning': 'No externally provided weights; our synthetic pretraining is present',
        'external_training_images': 0,
        'external_training_images_field_meaning': 'No additional natural training images; synthetic exposures counted separately',
        'learned_teacher': False,
        'teacher_validation_basis': 'Known own procedural format and objective; analytic HOG/mask/detail targets, no learned teacher',
        'query_weight_updates': 0,
    }


def train_procedural_reference(reference, pretrained_checkpoint, checkpoint_path,
                               steps=120, seed=73, device='cpu'):
    from .models.cnn import image_digest
    from .gaussian_variation import gaussian_variation, gaussian_variation_recipe
    if isinstance(steps, bool) or int(steps) != steps or steps < 1:
        raise ValueError('steps must be positive integer')
    reference = rgb_uint8(reference)
    path = Path(checkpoint_path)
    sidecar = path.with_suffix('.pt.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Preserve existing checkpoint/provenance')
    source, file_hash, tensor_hash = load_procedural_checkpoint(pretrained_checkpoint)
    started = time.perf_counter()
    dev = torch.device(device)
    patches, groups, bank = reference_patch_bank(reference, device=dev)
    sampler_rng = torch.Generator(device=dev).manual_seed(seed + 1)
    geometry_rng = torch.Generator(device=dev).manual_seed(seed + 2)
    corruption_rng = torch.Generator(device=dev).manual_seed(seed + 3)
    sampler = BalancedPatchCycle(groups, sampler_rng)
    counts = torch.zeros(len(patches), dtype=torch.long, device=dev)
    patch_sequence = sha256()
    losses = []
    insufficient = not np.any(reference != reference[0, 0])
    rng_devices = [] if dev.type == 'cpu' else [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        model = TinyShapeCNN(64).to(dev)
        model.load_state_dict(source['state_dict'], strict=True)
        initial = tensor_state_sha256(model.state_dict())
        if initial != tensor_hash:
            raise RuntimeError('Source initialization was not preserved')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
        model.train()
        if not insufficient:
            for step in range(int(steps)):
                ids = sampler.take(32)
                counts += torch.bincount(ids, minlength=len(patches))
                patch_sequence.update(ids.cpu().numpy().tobytes())
                clean = shared_geometry(patches[ids], geometry_rng)
                augmentation_rng = torch.Generator(device=dev).manual_seed(seed + 10000 + step)
                clean = gaussian_variation(clean, augmentation_rng, kind='shape_pattern', strength=1.)
                first = corrupt_same_geometry(clean, corruption_rng)
                second = corrupt_same_geometry(clean, corruption_rng)
                with torch.no_grad():
                    target = analytic_shape_target(clean)
                z0, z1 = model(torch.cat([first, second])).chunk(2)
                loss, _ = restoration_loss(z0, z1, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            predicted = torch.cat([model(patches[i:i + 128]) for i in range(0, len(patches), 128)])
            target = analytic_shape_target(patches)
            fidelity = float((predicted * target).sum(1).mean())
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        metadata = {
            **source_provenance(source, file_hash, tensor_hash),
            'method': 'own_procedural_cnn_one_reference_adaptation', 'training_revision': 9,
            'mode': 'shape_pattern', 'target_reference_images': 1,
            'real_training_source_images': 1, 'additional_real_training_images': 0,
            'reference_rgb_sha256': image_digest(reference), 'reference_shape': list(reference.shape),
            'target_weight_updates': len(losses), 'real_reference_weight_updates': len(losses),
            'requested_steps': int(steps), 'seed': seed,
            'batch_size': 32, 'learning_rate': .002, 'image_size': 256, 'patch_size': 32,
            'parameters': sum(p.numel() for p in model.parameters()), 'embedding_dim': 64,
            'num_reference_patches': len(patches), **bank,
            'patch_training_draw_counts': counts.cpu().tolist(),
            'all_reference_patches_trained': bool((counts > 0).all()),
            'insufficient_informative_patches': bool(insufficient),
            'initial_state_sha256': initial, 'final_tensor_sha256': tensor_state_sha256(state),
            'patch_sequence_sha256': patch_sequence.hexdigest(),
            'geometry_rng_state_sha256': sha256(geometry_rng.get_state().cpu().numpy().tobytes()).hexdigest(),
            'corruption_rng_state_sha256': sha256(corruption_rng.get_state().cpu().numpy().tobytes()).hexdigest(),
            'rng_policy': 'Same sampler/affine/corruption/per-step Gaussian streams as v07 shape_pattern',
            'objective': 'Unchanged v07 shape_pattern reference restoration; only initialization differs',
            'variation_strength': 1., 'gaussian_variation_recipe': gaussian_variation_recipe(),
            'training_loss': losses, 'clean_reference_target_cosine': fidelity,
            'synthetic_patch_views_seen': len(losses) * 32 * 2,
            'reference_derived_patch_views_seen': len(losses) * 32 * 2,
            'elapsed_seconds': time.perf_counter() - started,
            'torch_version': str(torch.__version__), 'device': str(dev),
            'inter_reference_weight_transfer': False,
            'limitation': 'Procedural pretraining plus one real reference is not one training image total.',
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format': FORMAT, 'metadata': metadata, 'state_dict': state}, path)
    sidecar.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata
