"""Support-only nuisance statistics and an explicit, imperfect foreground prior.

No query, annotation or pretrained model is accepted. Synthetic variance measures
our chosen warp tolerance, not category uncertainty or a calibrated likelihood.
"""
from __future__ import annotations

import threading
import cv2
import numpy as np
from PIL import Image
import torch

from .types import rgb_uint8
from .gaussian_variation import gaussian_variation
from .models.region_shape import canonical_region_tiles

_GRABCUT_LOCK = threading.Lock()


def _tile_values(array, function):
    values = []
    for n in (1, 2, 3):
        side = 96 // n
        for y in range(n):
            for x in range(n):
                values.append(function(array[y*side:(y+1)*side, x*side:(x+1)*side]))
    return np.asarray(values, dtype=np.float32)


def foreground_part_weights(referenceRGB, seed=607):
    """Heuristic GrabCut mask and contrast-weighted parts, requiring no encoder.

The central foreground seed is an assumption, not an object annotation. Invalid
area estimates fall back to an all-foreground mask. Uniform supports abstain.
"""
    reference = rgb_uint8(referenceRGB)
    base = np.array(Image.fromarray(reference).resize((96, 96), Image.Resampling.BILINEAR))
    blank = not np.any(reference != reference[0, 0])
    fallback = None
    mask = np.ones((96, 96), np.float32)
    if blank:
        fallback = 'spatially_uniform_reference'
    else:
        labels = np.full((96, 96), cv2.GC_PR_FGD, np.uint8)
        labels[:6] = labels[-6:] = cv2.GC_PR_BGD
        labels[:, :6] = labels[:, -6:] = cv2.GC_PR_BGD
        for ys in (slice(0, 2), slice(-2, None)):
            for xs in (slice(0, 2), slice(-2, None)):
                labels[ys, xs] = cv2.GC_BGD
        labels[43:53, 43:53] = cv2.GC_FGD
        try:
            # OpenCV's RNG is global; serialize seeding and this operation.
            with _GRABCUT_LOCK:
                cv2.setRNGSeed(int(seed) % (2**31 - 1))
                cv2.grabCut(base.copy(), labels, None, np.zeros((1, 65)),
                            np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
            proposed = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
            fraction = float(proposed.mean())
            if .15 <= fraction <= .95:
                mask = proposed
            else:
                fallback = 'foreground_area_out_of_bounds'
        except Exception as error:
            fallback = 'grabcut_exception:' + type(error).__name__
    gray = (base.astype(np.float32) @ np.array([.299, .587, .114], np.float32)) / 255.
    weights, fractions, information = [], [], []
    for flip in (False, True):
        m = mask[:, ::-1] if flip else mask
        g = gray[:, ::-1] if flip else gray
        fg = _tile_values(m, np.mean)
        # A standard deviation of .15 saturates this deliberately bounded prior.
        info = np.clip(_tile_values(g, np.std) / .15, .1, 1.)
        weight = np.maximum(.1, fg * info)
        # Floors prevent empty cells from dominating precision; ratio <= 4.
        weight = np.maximum(weight, weight.max() / 4.)
        weight /= weight.mean()
        weights.append(weight)
        fractions.append(fg)
        information.append(info)
    evidence = {
        'source': 'one_reference_rgb_only', 'foreground_mode': 'grabcut_support_heuristic',
        'foreground_is_ground_truth': False, 'foreground_area': float(mask.mean()),
        'foreground_fallback': fallback, 'foreground_seed': int(seed),
        'foreground_prior': '2px corner background; 6px probable background border; central 10px foreground',
        'part_weight_prior': 'foreground fraction times bounded grayscale contrast; ratio <= 4',
        'part_foreground_fraction': np.stack(fractions).tolist(),
        'part_contrast_information': np.stack(information).tolist(),
        'spatially_uniform_reference': bool(blank), 'abstain': bool(blank),
        'query_input': False, 'annotation_input': False, 'pretrained_weights': False,
    }
    return {'part_weights': np.stack(weights), 'mask96': mask,
            'abstain': bool(blank), 'evidence': evidence}


def support_statistics(referenceRGB, region_matcher, samples=16, seed=607):
    """Encode original/flipped support tiles plus independent synthetic variants.

Returns normalized mean/original features and bounded diagonal precision, each
with shape [2,14,64]. The caller must honor ``abstain`` for uniform references.
The supplied region matcher must already use a frozen evaluation descriptor.
"""
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError('samples must be a positive integer')
    descriptor = getattr(region_matcher, 'descriptor', None)
    owner = getattr(descriptor, '__self__', None)
    module = descriptor if isinstance(descriptor, torch.nn.Module) else getattr(owner, 'model', None)
    if isinstance(module, torch.nn.Module) and any(layer.training for layer in module.modules()):
        raise ValueError('Support statistics require a frozen evaluation-mode descriptor')
    reference = rgb_uint8(referenceRGB)
    prior = foreground_part_weights(reference, seed)
    tiles = torch.cat((canonical_region_tiles(reference),
                       canonical_region_tiles(reference[:, ::-1].copy())))
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    with torch.inference_mode():
        original = region_matcher._encode(tiles).reshape(2, 14, 64)
        varied = gaussian_variation(tiles.repeat(int(samples), 1, 1, 1), generator,
                                   kind='shape_pattern')
        features = region_matcher._encode(varied).reshape(int(samples), 2, 14, 64)
    observations = np.concatenate((original[None], features), axis=0)
    mean = observations.mean(0)
    mu = mean / np.maximum(np.linalg.norm(mean, axis=-1, keepdims=True), 1e-8)
    variance = observations.var(0, ddof=0)
    shrunk = .5 * variance + .5 * variance.mean(-1, keepdims=True) + .01
    precision = 1. / shrunk
    precision /= precision.mean(-1, keepdims=True)
    precision = np.clip(precision, .25, 4.)
    evidence = {**prior['evidence'], 'statistics_mode': 'support_gaussian_diagonal_shrinkage',
                'synthetic_variants_per_orientation': int(samples), 'statistics_seed': int(seed),
                'variant_mode': 'shape_pattern', 'observations_include_original': True,
                'precision_rule': '1/(.5*variance+.5*mean_variance+.01), mean normalized, clipped [.25,4]',
                'uncertainty_interpretation': 'local synthetic nuisance tolerance; not category covariance',
                'score_is_probability': False, 'learner_updates': 0,
                'mean_synthetic_feature_variance': float(variance.mean())}
    return {**prior, 'mu': mu.astype(np.float32), 'precision': precision.astype(np.float32),
            'original_features': original.astype(np.float32), 'evidence': evidence}
