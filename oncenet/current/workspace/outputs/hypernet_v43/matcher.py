"""Frozen one-positive/negative-memory readouts for the v43 baseline comparison.

The background is the same v42 label-balanced 512 FIT rows for every encoder.
No CAL/DEV/test feature participates in fitting. This is not PatchCore: it is
image-descriptor matching, with two explicit distance/density alternatives.
"""
import hashlib
import numpy as np
from scipy.special import logsumexp

METHODS = ('exemplar_lda', 'normalized_whitened_cosine',
           'negative_cosine_margin', 'negative_distance_margin', 'kernel_density_ratio')


def fit_indices(rows, target):
    """Exact v42 selection, duplicated to avoid global common-module collisions."""
    pools = {}
    for i, row in enumerate(rows):
        if row['split'] == 'fit' and row['label'] != target:
            pools.setdefault(row['label'], []).append(i)
    for label, inds in pools.items():
        inds.sort(key=lambda i: hashlib.sha256(
            ('v42-fit|' + rows[i]['group'] + '|' + rows[i]['event_id']).encode()).hexdigest())
    order = sorted(pools, key=lambda x: hashlib.sha256(('v42-label|' + x).encode()).hexdigest())
    chosen, rank = [], 0
    while len(chosen) < 512:
        addition = [pools[k][rank] for k in order if len(pools[k]) > rank]
        if not addition:
            break
        chosen.extend(addition[:512 - len(chosen)])
        rank += 1
    return np.asarray(chosen, dtype=np.int64)


def fit(background, reference):
    """Return an array-only state; save via np.savez to bind a fitted model."""
    bg = np.asarray(background, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if bg.ndim != 2 or r.shape != (bg.shape[1],) or len(bg) < 2:
        raise ValueError('Expected >=2 background vectors and exactly one reference vector')
    if not np.isfinite(bg).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite fitting feature')
    mu = bg.mean(0)
    x = bg - mu
    n, d = x.shape
    beta = .5 / (n - 1)
    ridge = max(.5 * np.square(x).sum() / ((n - 1) * d), 1e-12)
    gram = x @ x.T
    values, basis = np.linalg.eigh(gram)
    inverse = (basis / (ridge + beta * np.maximum(values, 0))) @ basis.T
    bg_cross = gram @ inverse
    bg_norm = np.maximum(np.diag(bg_cross), 0)
    # Bandwidth comes only from FIT-to-FIT geometry, excluding the identity.
    distances = np.maximum(bg_norm[:, None] + bg_norm[None, :] - 2 * bg_cross, 0)
    np.fill_diagonal(distances, np.inf)
    kth = min(5, n - 1)
    bandwidth2 = max(float(np.median(np.partition(distances, kth - 1, axis=1)[:, kth - 1])), 1e-12)
    centered_reference = r - mu
    rx = x @ centered_reference
    rn = max(float(centered_reference @ centered_reference / ridge
                   - beta / ridge * (rx @ inverse @ rx)), 0)
    return dict(mu=mu, x=x, inverse=inverse, bg_norm=bg_norm,
                beta=np.array(beta), ridge=np.array(ridge), r=centered_reference,
                rx=rx, rn=np.array(rn), bandwidth2=np.array(bandwidth2))


def predict(features, state, batch_size=256):
    """Each returned vector increases with target evidence; no threshold fitting."""
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != len(state['mu']) or not np.isfinite(f).all():
        raise ValueError('Invalid prediction feature array')
    mu, x, inverse = state['mu'], state['x'], state['inverse']
    beta, ridge = float(state['beta']), float(state['ridge'])
    rn, bandwidth2 = float(state['rn']), float(state['bandwidth2'])
    result = {method: [] for method in METHODS}
    for start in range(0, len(f), batch_size):
        q = f[start:start + batch_size] - mu
        qx = q @ x.T
        cross = qx @ inverse
        qn = np.maximum(np.square(q).sum(1) / ridge
                        - beta / ridge * np.einsum('ij,ij->i', cross, qx), 0)
        dot = q @ state['r'] / ridge - beta / ridge * (cross @ state['rx'])
        denr = np.sqrt(qn * rn)
        cosine = np.divide(dot, denr, out=np.zeros_like(dot), where=denr > 1e-12)
        cosine = np.clip(cosine, -1, 1)
        denb = np.sqrt(qn[:, None] * state['bg_norm'][None, :])
        negative_cos = np.divide(cross, denb, out=np.zeros_like(cross), where=denb > 1e-12)
        # Half the difference of nearest-background and reference squared
        # Mahalanobis distances; query norm cancels exactly.
        neg_center = cross - .5 * state['bg_norm'][None, :]
        pos_center = dot - .5 * rn
        parts = dict(
            exemplar_lda=dot,
            normalized_whitened_cosine=cosine,
            negative_cosine_margin=cosine - np.clip(negative_cos, -1, 1).max(1),
            negative_distance_margin=pos_center - neg_center.max(1),
            kernel_density_ratio=pos_center / bandwidth2
                - logsumexp(neg_center / bandwidth2, axis=1) + np.log(len(x)),
        )
        for method, value in parts.items():
            assert np.isfinite(value).all(), method
            result[method].append(value)
    return {key: np.concatenate(value) if value else np.zeros(0) for key, value in result.items()}


def scores(features, rows, target, reference_indices):
    """Drop-in v42 scoring API; reference variants are scored independently."""
    f = np.asarray(features, dtype=np.float64)
    fi = fit_indices(rows, target)
    assert len(f) == len(rows)
    assert all(rows[i]['split'] == 'fit' and rows[i]['label'] != target for i in fi)
    assert all(rows[i]['label'] == target and rows[i]['split'] in ('reference', 'reference_check')
               for i in reference_indices)
    result, meta = {}, None
    for ri in reference_indices:
        state = fit(f[fi], f[ri])
        result[ri] = predict(f, state)
        details = dict(fit_indices=fi.tolist(), ridge=float(state['ridge']),
                       dimensions=f.shape[1], fit_count=len(fi),
                       bandwidth2=float(state['bandwidth2']), bandwidth_rule='FIT median fifth-neighbor squared Mahalanobis distance',
                       negative_fit_label_counts={k: sum(rows[i]['label'] == k for i in fi)
                                                  for k in sorted({rows[i]['label'] for i in fi})})
        if meta is not None:
            assert meta == details
        meta = details
    return result, meta


def self_test():
    rng = np.random.default_rng(430071)
    bg, r, q = rng.normal(size=(31, 9)), rng.normal(size=9), rng.normal(size=(19, 9))
    state = fit(bg, r)
    got = predict(q, state)
    mu = bg.mean(0)
    inv = np.linalg.inv(.5 * np.cov(bg, rowvar=False) + float(state['ridge']) * np.eye(9))
    cq, cr, cb = q - mu, r - mu, bg - mu
    qn = np.einsum('ni,ij,nj->n', cq, inv, cq)
    rn = cr @ inv @ cr
    bn = np.einsum('ni,ij,nj->n', cb, inv, cb)
    dot = cq @ inv @ cr
    cross = cq @ inv @ cb.T
    dpos = np.einsum('ni,ij,nj->n', q - r, inv, q - r)
    dneg = np.stack([np.einsum('ni,ij,nj->n', q - b, inv, q - b) for b in bg], axis=1)
    sigma2 = float(state['bandwidth2'])
    expected = dict(exemplar_lda=dot,
                    normalized_whitened_cosine=dot / np.sqrt(qn * rn),
                    negative_cosine_margin=dot / np.sqrt(qn * rn) - (cross / np.sqrt(qn[:, None] * bn)).max(1),
                    negative_distance_margin=.5 * (dneg.min(1) - dpos),
                    kernel_density_ratio=-dpos / (2 * sigma2) - logsumexp(-dneg / (2 * sigma2), axis=1) + np.log(len(bg)))
    for method in METHODS:
        np.testing.assert_allclose(got[method], expected[method], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(got[method], predict(q, state, batch_size=3)[method], rtol=1e-10, atol=1e-10)
    zero = predict(np.zeros((2, 9)), fit(np.zeros((7, 9)), np.zeros(9)))
    assert all(np.isfinite(v).all() for v in zero.values())
    return dict(passed=True, direct_inverse_and_kernel_parity=True, batch_invariance=True,
                constant_features_finite=True,
                max_absolute_error={k: float(np.max(np.abs(got[k] - expected[k]))) for k in METHODS})
