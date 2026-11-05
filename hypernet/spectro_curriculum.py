"""Synthetic time-frequency curriculum: glitch-like families as parameter distributions.

Every image is a scalar energy field, 140 rows (frequency, top = high) by 170
columns (time), the same layout as the Gravity Spy HDF5 spectrograms. It is a
1/f^alpha noise background plus one to three additive elements drawn from a
small vocabulary (blobs, chirps, bands, bursts, repeated blips, arches,
wandering lines, ripples, mirrored pairs).

A FAMILY is not one picture. Its seed fixes a recipe: which element types,
and a centre and spread for every parameter. Each MEMBER is an independent
sample from those distributions, so two members of one family differ in
position, size, brightness, count and shape, the way two real glitches of one
type differ. Nothing here is fitted to real spectrograms; the ranges below were
chosen by eye from the published Gravity Spy class descriptions.
"""
from __future__ import annotations
import numpy as np

HEIGHT, WIDTH = 140, 170
ELEMENTS = ('blob', 'chirp', 'band', 'burst', 'blips', 'arches', 'wander', 'ripples', 'pair')
FAMILY_COUNT = 4096
HELDOUT_FAMILIES = 512
FORMAT = 'hypernet.spectro_curriculum.v0.4'

_F = (1 - (np.arange(HEIGHT) + .5) / HEIGHT)[:, None]  # 0 at bottom (low frequency), 1 at top
_T = ((np.arange(WIDTH) + .5) / WIDTH)[None, :]


def _pink(rng, alpha):
    """2D 1/f^alpha noise with unit standard deviation."""
    white = rng.standard_normal((HEIGHT, WIDTH))
    fy = np.fft.fftfreq(HEIGHT)[:, None]; fx = np.fft.fftfreq(WIDTH)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2); radius[0, 0] = 1
    field = np.real(np.fft.ifft2(np.fft.fft2(white) / radius ** (alpha / 2)))
    return (field - field.mean()) / (field.std() + 1e-9)


def background(rng):
    """Noise floor: pink noise, more energy at low frequency, a few faint steady lines."""
    base = .10 + .035 * rng.random()
    field = base + .014 * _pink(rng, .8 + .9 * rng.random()) + .012 * rng.standard_normal((HEIGHT, WIDTH)) \
        + .05 * rng.random() * (1 - _F) ** (1 + 2 * rng.random())
    # Faint time-localized streaks, like the vertical texture of real q-transform noise.
    streaks = np.maximum(0, rng.standard_normal(WIDTH) - 1.2)[None, :] * .04 * rng.random()
    field = field + streaks * (1 - _F) ** (.5 + rng.random())
    for _ in range(rng.integers(0, 3)):
        f0 = rng.random(); width = .004 + .006 * rng.random()
        field = field + .03 * rng.random() * np.exp(-.5 * ((_F - f0) / width) ** 2)
    return field


def _gauss(x, centre, sigma):
    return np.exp(-.5 * ((x - centre) / np.maximum(sigma, 1e-3)) ** 2)


def _curve(f_of_t, t_mask, thickness, amp_t):
    """Energy along a frequency curve f(t) with Gaussian thickness in frequency."""
    return _gauss(_F, f_of_t[None, :], thickness) * t_mask[None, :] * amp_t[None, :]


def render_element(kind, p, rng):
    """One element from sampled parameters p. Returns a non-negative field with peak about 1."""
    t0, f0 = p['t0'], p['f0']
    if kind == 'blob':
        rise, decay = p['size_t'] * p['asym'], p['size_t'] / p['asym']
        sig_t = np.where(_T < t0, rise, decay)
        # Teardrop: the time width grows toward low frequency when flare > 0, like a Blip.
        below = np.clip(f0 - _F, 0, 1) / max(f0, 1e-3)
        return _gauss(_T, t0, sig_t * (1 + p['flare'] * below)) * _gauss(_F, f0 + p['tilt'] * (_T - t0), p['size_f'])
    if kind == 'chirp':
        t = np.clip(_T[0], t0, t0 + p['length']); u = (t - t0) / max(p['length'], 1e-3)
        f = f0 + p['rise'] * u ** p['power']
        mask = ((_T[0] >= t0) & (_T[0] <= t0 + p['length'])).astype(float)
        amp = 1 - p['fade'] * u
        return _curve(f, mask, p['thick'], amp)
    if kind == 'band':
        field = np.zeros((HEIGHT, WIDTH))
        window = _gauss(_T, t0, p['length']) if p['windowed'] > .5 else np.ones_like(_T)
        modulation = 1 + p['mod'] * np.sin(2 * np.pi * p['mod_rate'] * _T + p['phase'])
        for k in range(1, int(p['harmonics']) + 1):
            field = field + p['decay'] ** (k - 1) * _gauss(_F, f0 * k, p['thick'])
        return field * window * modulation
    if kind == 'burst':
        envelope = (1 - _F) ** p['slope'] * _gauss(_F, p['centre_f'], p['band']) ** p['banded']
        return _gauss(_T, t0, p['width_t']) * envelope
    if kind == 'blips':
        field = np.zeros((HEIGHT, WIDTH))
        for k in range(int(p['count'])):
            tk = t0 + k * p['period'] + p['jitter'] * rng.standard_normal()
            field = field + _gauss(_T, tk, p['size_t']) * _gauss(_F, f0 + p['drift'] * k, p['size_f'])
        return field
    if kind == 'arches':
        phase = (_T[0] - t0) / max(p['period'], 1e-3)
        f = p['base_f'] + p['height'] * np.abs(np.sin(np.pi * phase))
        mask = ((_T[0] >= t0) & (_T[0] <= t0 + p['period'] * p['count'])).astype(float)
        return _curve(f, mask, p['thick'], np.ones(WIDTH))
    if kind == 'wander':
        steps = rng.standard_normal(WIDTH) * p['step']
        kernel = np.exp(-.5 * (np.arange(-8, 9) / p['smooth']) ** 2); kernel /= kernel.sum()
        f = f0 + np.cumsum(np.convolve(steps, kernel, mode='same'))
        mask = ((_T[0] >= t0) & (_T[0] <= t0 + p['length'])).astype(float)
        return _curve(np.clip(f, 0, 1), mask, p['thick'], np.ones(WIDTH))
    if kind == 'ripples':
        region = _gauss(_T, t0, p['size_t']) * _gauss(_F, f0, p['size_f'])
        dots = (.5 + .5 * np.cos(2 * np.pi * _T / p['pitch_t'])) ** p['sharp'] * (.5 + .5 * np.cos(2 * np.pi * _F / p['pitch_f'])) ** p['sharp']
        return region * dots
    if kind == 'pair':
        half = _gauss(_F, f0, p['size_f']) * _gauss(_T, t0 - p['gap'] / 2, p['size_t'])
        other = _gauss(_F, f0 + p['offset_f'], p['size_f']) * _gauss(_T, t0 + p['gap'] / 2, p['size_t'])
        return half + other
    raise ValueError(kind)


# Parameter ranges: (low, high) for the family CENTRE; the family also draws a
# relative spread, and members sample centre + spread * N(0, 1), clipped to range.
RANGES = {
    'blob': {'t0': (.2, .8), 'f0': (.05, .9), 'size_t': (.01, .08), 'size_f': (.02, .25), 'asym': (.5, 2.5), 'tilt': (-.3, .3), 'flare': (0, 3)},
    'chirp': {'t0': (.05, .7), 'f0': (.02, .6), 'length': (.08, .6), 'rise': (-.6, .9), 'power': (.5, 3), 'thick': (.008, .05), 'fade': (0, 1)},
    'band': {'t0': (.2, .8), 'f0': (.03, .45), 'harmonics': (1, 4), 'decay': (.3, .9), 'thick': (.004, .03), 'windowed': (0, 1), 'length': (.05, .3), 'mod': (0, .9), 'mod_rate': (1, 8), 'phase': (0, 6.28)},
    'burst': {'t0': (.3, .7), 'f0': (0, 1), 'width_t': (.004, .04), 'slope': (0, 4), 'centre_f': (.05, .9), 'band': (.05, .5), 'banded': (0, 1)},
    'blips': {'t0': (.05, .5), 'f0': (.05, .6), 'count': (2, 7), 'period': (.05, .2), 'jitter': (0, .02), 'size_t': (.006, .03), 'size_f': (.02, .15), 'drift': (-.03, .03)},
    'arches': {'t0': (0, .3), 'f0': (0, 1), 'base_f': (.02, .3), 'height': (.05, .5), 'period': (.08, .4), 'count': (1, 5), 'thick': (.006, .04)},
    'wander': {'t0': (0, .4), 'f0': (.1, .8), 'length': (.3, 1), 'step': (.002, .02), 'smooth': (1, 6), 'thick': (.005, .03)},
    'ripples': {'t0': (.3, .7), 'f0': (.1, .9), 'size_t': (.05, .3), 'size_f': (.03, .3), 'pitch_t': (.02, .1), 'pitch_f': (.02, .1), 'sharp': (1, 6)},
    'pair': {'t0': (.3, .7), 'f0': (.05, .8), 'size_t': (.01, .05), 'size_f': (.02, .15), 'gap': (.05, .3), 'offset_f': (-.2, .2)},
}
INTEGER = {'harmonics', 'count'}


def family_recipe(family):
    """Deterministic recipe for one family id: element kinds and per-parameter (centre, spread)."""
    rng = np.random.default_rng(40_000_000 + int(family))
    count = rng.choice([1, 2, 3], p=[.5, .35, .15])
    kinds = list(rng.choice(ELEMENTS, size=count, replace=count > len(ELEMENTS)))
    elements = []
    for kind in kinds:
        params = {}
        for name, (low, high) in RANGES[kind].items():
            centre = low + (high - low) * rng.random()
            spread = (high - low) * (.03 + .17 * rng.random())
            params[name] = (centre, spread, low, high)
        elements.append((kind, params))
    return {'family': int(family), 'elements': elements,
            'snr': (np.log(.25 + 2.0 * rng.random()), .25 + .4 * rng.random()),
            'weights': rng.dirichlet(np.ones(count) * 2)}


def render_member(recipe, seed):
    """One independent member of a family. Returns (image in [0,1], foreground mask in [0,1])."""
    rng = np.random.default_rng(seed)
    image = background(rng); signal = np.zeros((HEIGHT, WIDTH))
    for weight, (kind, params) in zip(recipe['weights'], recipe['elements']):
        p = {}
        for name, (centre, spread, low, high) in params.items():
            value = float(np.clip(centre + spread * rng.standard_normal(), low, high))
            p[name] = int(round(value)) if name in INTEGER else value
        element = render_element(kind, p, rng)
        signal = signal + weight * element / (element.max() + 1e-9)
    gain = float(np.exp(recipe['snr'][0] + recipe['snr'][1] * rng.standard_normal()))
    image = np.clip(image + gain * signal * (1 + .15 * _pink(rng, 1.0) * (signal > .05)), 0, 1)
    mask = np.clip(signal / (signal.max() + 1e-9) * 1.5, 0, 1) * (signal > .05 * signal.max())
    return image.astype(np.float32), mask.astype(np.float32)


def to_rgb_uint8(image):
    """Same conversion as the Gravity Spy HDF5 loader: scalar field to repeated uint8 channels."""
    gray = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def augment(image, rng, max_shift=.08, time_mask=.15, freq_gain=.25, gain=.3):
    """Physics-safe view: time shift, time masking, frequency-band gain, overall gain. No flips or rotations."""
    out = image.copy()
    shift = int(round(max_shift * WIDTH * (2 * rng.random() - 1)))
    if shift:
        floor = np.median(out[:, :3]) if shift > 0 else np.median(out[:, -3:])
        out = np.roll(out, shift, axis=1)
        if shift > 0: out[:, :shift] = floor
        else: out[:, shift:] = floor
    if rng.random() < .5:
        width = int(time_mask * WIDTH * rng.random()); start = rng.integers(0, WIDTH - width + 1)
        out[:, start:start + width] = np.median(out)
    if rng.random() < .5:
        f0 = rng.random(); band = .05 + .25 * rng.random()
        out = out * (1 + freq_gain * (2 * rng.random() - 1) * _gauss(_F, f0, band))
    out = out * (1 + gain * (2 * rng.random() - 1) * .5)
    return np.clip(out, 0, 1).astype(np.float32)
