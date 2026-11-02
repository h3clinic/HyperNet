"""Independent procedural diagnostics for nuisance tolerance versus fine detail.

This is an identity/detail-retention test, NOT a natural category benchmark.
Images never come from the training generator, external datasets or pretrained
models. Hard negatives and positives share exactly the same query nuisance.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Callable

import numpy as np
from PIL import Image
import torch

FAMILIES = ('gap', 'notch', 'corner', 'texture_frequency')
MODES = ('native32', 'downsample96')
IMAGE_KEYS = ('reference', 'positive', 'negative', 'detail_only')


def _object_pair(family, size, detail, rng):
    scale = size // 32
    width, height = int(rng.integers(21, 25))*scale, int(rng.integers(19, 24))*scale
    left, top = (size-width)//2, (size-height)//2
    right, bottom = left+width, top+height
    base = np.zeros((size, size), bool)
    base[top:bottom, left:right] = True
    a, b = base.copy(), base.copy()
    meta = {'body_box': [left, top, right, bottom], 'source_detail_pixels': detail,
            'nominal_final_detail_pixels': detail/scale}
    if family == 'gap':
        thickness = 3*scale
        a[top+thickness:bottom-thickness, left+thickness:right-thickness] = False
        b[:] = a
        center = (left+right)//2
        # Equal removed area: shallow closed-ring notch versus full-width gap.
        notch_width = thickness*detail
        start = center-notch_width//2
        a[top, start:start+notch_width] = False
        start = center-detail//2
        b[top:top+thickness, start:start+detail] = False
        meta['identity_difference'] = 'closed ring with area-matched shallow notch versus open ring'
    elif family == 'notch':
        a[top:top+detail, left+width//3:left+width//3+detail] = False
        b[top:top+detail, left+2*width//3:left+2*width//3+detail] = False
        meta['identity_difference'] = 'same-size notch at a different location on the top edge'
    elif family == 'corner':
        for dy in range(detail):
            a[top+dy, left:left+detail-dy] = False
            b[top+dy, right-detail+dy:right] = False
        # Common asymmetric interior anchor rules out a simple mirror match.
        a[bottom-5*scale:bottom-3*scale, left+3*scale:left+5*scale] = False
        b[bottom-5*scale:bottom-3*scale, left+3*scale:left+5*scale] = False
        meta['identity_difference'] = 'small corner chamfer at a different corner; asymmetric anchor fixed'
    elif family == 'texture_frequency':
        meta['identity_difference'] = 'different stripe frequency with matched source mean and RMS before illumination'
    else:
        raise ValueError('Unknown detail family')
    fields = [np.full((size, size), .70, dtype=np.float64) for _ in range(2)]
    if family == 'texture_frequency':
        # In the downsample stress, source frequencies retain pixel-scale detail
        # and may alias at the final 32px input. Those are reported separately.
        cycles = int(rng.integers(3, 6))*scale
        delta = .5*detail
        phase = float(rng.uniform(-np.pi, np.pi))
        xx = (np.arange(size)[None, :]-left)/width
        for i, frequency in enumerate((cycles, cycles+delta)):
            pattern = np.broadcast_to(np.cos(2*np.pi*frequency*xx+phase), (size, size)).copy()
            pattern -= pattern[a].mean()
            pattern /= max(float(pattern[a].std()), 1e-8)
            fields[i] += .075*pattern
        meta.update(source_pattern_cycles=cycles, frequency_difference_cycles=delta)
    meta.update(reference_foreground_pixels=int(a.sum()), negative_foreground_pixels=int(b.sum()),
                foreground_area_matched=bool(a.sum() == b.sum()))
    return (a, fields[0]), (b, fields[1]), meta


def _nuisance(rng, size, translated):
    scale = size//32
    dx = int(rng.choice([-2, -1, 1, 2]))*scale if translated else 0
    dy = int(rng.choice([-2, -1, 1, 2]))*scale if translated else 0
    return {'background_seed': int(rng.integers(0, 2**31-1)), 'translation': [dx, dy],
            'gain': float(rng.uniform(.86, 1.12)), 'offset': float(rng.uniform(-.025, .025)),
            'gamma': float(rng.uniform(.88, 1.12)), 'color_gain': rng.uniform(.95, 1.05, 3).tolist()}


def _render(obj, nuisance, size, color):
    mask, field = obj
    dx, dy = nuisance['translation']
    mask = np.roll(np.roll(mask, dy, 0), dx, 1)
    field = np.roll(np.roll(field, dy, 0), dx, 1)
    rng = np.random.default_rng(nuisance['background_seed'])
    yy, xx = np.mgrid[0:size, 0:size]/size
    wave = np.sin(2*np.pi*(rng.uniform(.5, 2)*xx+rng.uniform(.5, 2)*yy)+rng.uniform(-np.pi, np.pi))
    background = (rng.uniform(.17, .37, 3)[None, None, :] + .035*wave[..., None]
                  + rng.normal(0, .008, (size, size, 3)))
    foreground = field[..., None]*color[None, None, :]
    image = np.where(mask[..., None], foreground, background)
    image = np.clip((image*nuisance['gain']+nuisance['offset'])*np.asarray(nuisance['color_gain']), 0, 1)
    image = np.clip(image**nuisance['gamma'], 0, 1)
    image = np.rint(image*255).astype(np.uint8)
    if size != 32:
        image = np.asarray(Image.fromarray(image).resize((32, 32), Image.Resampling.BILINEAR)).copy()
    return image


def build_detail_benchmark(seed=29073, examples_per_family=24, image_size=32, mode='native32'):
    """Return four uint8 NHWC image arrays and per-case metadata.

    R=reference, P=same object/new nuisance, N=detail-modified object with P's
    nuisance, D=detail-modified object with R's nuisance. Widths of 1/2/3 pixels
    refer to source pixels: native32 is the primary diagnostic; downsample96
    intentionally stresses subpixel loss before the CNN. This renderer is
    independent of procedural_shapes.py and must not be used as training data.
    """
    if isinstance(examples_per_family, bool) or not isinstance(examples_per_family, int) or examples_per_family < 1:
        raise ValueError('examples_per_family must be a positive integer')
    if image_size != 32 or mode not in MODES:
        raise ValueError('Only 32px descriptor input and declared render modes are supported')
    size = 32 if mode == 'native32' else 96
    images = {key: [] for key in IMAGE_KEYS}
    cases = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(examples_per_family):
            # Independent streams prevent mode/family iteration order affecting cases.
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9137, family_index, index]))
            detail = 1+index % 3
            original, altered, geometry = _object_pair(family, size, detail, rng)
            # Reverse roles over complete severity x nuisance blocks so neither
            # open/closed topology nor higher frequency always means negative.
            reverse_identity = bool((index//6) % 2)
            if reverse_identity:
                original, altered = altered, original
            color = rng.uniform(.91, 1.06, 3)
            reference_view = _nuisance(rng, size, False)
            query_view = _nuisance(rng, size, bool(index % 2))
            rendered = {'reference': _render(original, reference_view, size, color),
                        'positive': _render(original, query_view, size, color),
                        'negative': _render(altered, query_view, size, color),
                        'detail_only': _render(altered, reference_view, size, color)}
            for key in IMAGE_KEYS:
                images[key].append(rendered[key])
            query_delta = np.abs(rendered['positive'].astype(float)-rendered['negative'])/255
            reference_delta = np.abs(rendered['reference'].astype(float)-rendered['detail_only'])/255
            cases.append({'id': f'{mode}_{family}_{index:04d}', 'family': family, 'index': index,
                'seed': int(seed), 'mode': mode, 'source_resolution': size, 'input_resolution': 32,
                'reference_identity': 'altered' if reverse_identity else 'base',
                'nuisance_group': 'backdrop_illumination_translation' if index % 2 else 'backdrop_illumination',
                'reference_nuisance': reference_view, 'positive_and_negative_nuisance': query_view,
                **geometry, 'query_detail_visible': bool(np.any(query_delta)),
                'query_changed_pixels': int(np.any(query_delta > 0, axis=-1).sum()),
                'reference_changed_pixels': int(np.any(reference_delta > 0, axis=-1).sum()),
                'query_detail_mean_absolute_pixel_delta': float(query_delta.mean()),
                'reference_detail_mean_absolute_pixel_delta': float(reference_delta.mean())})
    arrays = {key: np.stack(value) for key, value in images.items()}
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True).encode())
    for key in IMAGE_KEYS:
        digest.update(arrays[key].tobytes())
    unique = {hashlib.sha256(image.tobytes()).hexdigest() for key in IMAGE_KEYS for image in arrays[key]}
    return {**arrays, 'cases': cases, 'provenance': {
        'generator': 'independent_detail_benchmark_v09_v2_identity_balanced', 'seed': int(seed), 'mode': mode,
        'cases': len(cases), 'generated_images': len(cases)*4, 'unique_generated_images': len(unique),
        'dataset_sha256': digest.hexdigest(), 'external_images': 0, 'pretrained_generator': False,
        'training_use': False, 'nuisance_is_matched_between_positive_and_negative': True,
        'identity_roles_balanced_over_12_case_blocks': True,
        'complete_balanced_blocks_per_family': examples_per_family//12,
        'tail_cases_per_family_outside_complete_balance_blocks': examples_per_family % 12,
        'primary_scope': 'Identity/detail retention under declared nuisances, not semantic category accuracy',
        'downsample_scope': 'Subpixel/aliasing stress is separate from native32 model detail retention'}}


def teacher_collision_case():
    """Two visible relative arrangements that the old pooled teacher collapses."""
    reference = np.full((32, 32, 3), 51, dtype=np.uint8)
    reference[20:24, 20:24] = 153  # fixed distant anchor
    changed = reference.copy()
    reference[2:6, 2:4] = 204
    changed[2:6, 4:6] = 204
    return {'reference': reference, 'detail_only': changed,
        'description': 'A2x4bar shifts2pixels relative to a fixed square, inside one8x8teacher cell',
        'changed_pixels': int(np.any(reference != changed, axis=-1).sum())}


def _summary(rows):
    fields = ('triplet_win', 'nuisance_distance', 'detail_only_distance', 'negative_distance',
              'bounded_margin', 'detail_to_nuisance_ratio')
    return {'n': len(rows), **{field: float(np.mean([r[field] for r in rows])) for field in fields},
            'tie_fraction': float(np.mean([r['tie'] for r in rows])),
            'visible_query_detail_fraction': float(np.mean([r['query_detail_visible'] for r in rows]))}


def evaluate_descriptor(describe: Callable, benchmark, batch_size=128, device='cpu', tie_tolerance=1e-7):
    """Evaluate an NCHW float-tensor -> NxD tensor callable without fitting it.

    Cosine distance is (1-cosine)/2 in [0,1]. Primary triplet_win is 1 when
    d(R,P)<d(R,N), 0 for a loss and .5 for ties. A collapsed constant descriptor
    therefore cannot win by merely being invariant. No threshold is fitted.
    """
    if batch_size < 1 or not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError('Invalid batch size or tie tolerance')
    n = len(benchmark['cases'])
    if not n or any(np.asarray(benchmark[k]).shape != (n, 32, 32, 3)
                    or np.asarray(benchmark[k]).dtype != np.uint8 for k in IMAGE_KEYS):
        raise ValueError('Expected aligned nonempty NHWC RGB arrays')
    if isinstance(describe, torch.nn.Module) and any(m.training for m in describe.modules()):
        raise ValueError('Descriptor must already be in evaluation mode')
    stacked = np.concatenate([benchmark[key] for key in IMAGE_KEYS])
    features = []
    with torch.inference_mode():
        for start in range(0, len(stacked), batch_size):
            tensor = torch.from_numpy(stacked[start:start+batch_size].copy()).permute(0, 3, 1, 2).to(device, dtype=torch.float32)/255.
            result = describe(tensor)
            if not isinstance(result, torch.Tensor) or result.ndim != 2 or result.shape[0] != len(tensor):
                raise ValueError('Descriptor must return an NxD torch tensor')
            features.append(result.detach().cpu().numpy().astype(np.float64))
    features = np.concatenate(features)
    if features.shape[1] < 1 or not np.isfinite(features).all():
        raise ValueError('Nonfinite or empty descriptor')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features /= np.where(norms > 1e-12, norms, 1.)
    ref, pos, neg, detail = features.reshape(4, n, -1)
    distance = lambda a, b: np.clip((1-np.einsum('ij,ij->i', a, b))/2, 0, 1)
    dpos, dneg, ddetail = distance(ref, pos), distance(ref, neg), distance(ref, detail)
    margin = dneg-dpos
    ties = np.abs(margin) <= tie_tolerance
    wins = np.where(ties, .5, (margin > 0).astype(float))
    denominator = ddetail+dpos
    ratio = np.divide(ddetail, denominator, out=np.full_like(ddetail, .5), where=denominator > tie_tolerance)
    rows = [{**case, 'triplet_win': float(wins[i]), 'tie': bool(ties[i]),
             'nuisance_distance': float(dpos[i]), 'detail_only_distance': float(ddetail[i]),
             'negative_distance': float(dneg[i]), 'bounded_margin': float((margin[i]+1)/2),
             'detail_to_nuisance_ratio': float(ratio[i])} for i, case in enumerate(benchmark['cases'])]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['family'], row['source_detail_pixels'], row['nuisance_group'])].append(row)
    return {'provenance': benchmark['provenance'], 'metrics': _summary(rows),
        'per_family': {family: _summary([r for r in rows if r['family'] == family])
                       for family in sorted({r['family'] for r in rows})},
        'per_condition': [{'family': family, 'source_detail_pixels': detail, 'nuisance_group': nuisance,
                           **_summary(group)} for (family, detail, nuisance), group in sorted(groups.items())],
        'descriptor_dimensions': features.shape[1], 'zero_norm_fraction': float(np.mean(norms <= 1e-12)),
        'normalized_feature_variance_sum': float(np.clip(features.var(0).sum(), 0, 1)),
        'tie_tolerance': tie_tolerance, 'rows': rows,
        'interpretation': 'Triplet wins compare matched-nuisance alternatives; detail retention does not imply semantic recognition.'}
