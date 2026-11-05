"""Whale audio -> HyperNet-G spectrogram windows.

Every recording is brought to 1 kHz (the rate Scripps analysts used for the
Bering Sea right whale search). Each window is 4 s long, 140 frequency rows by
160 time columns, so it tiles into 14 x 16 patches of 10 x 10 like the Gravity
Spy model input.

Rows: FFT bins 6..145 of a 320-point FFT at 1 kHz, i.e. about 19-456 Hz at
3.125 Hz per row. Row 0 is the HIGHEST frequency (image convention, as in
Gravity Spy). Columns: 25 ms hop. Right whale upcalls (90-250 Hz, ~1 s) and
gunshots (broadband impulses) both fit.
"""
import numpy as np
import soundfile as sf
from scipy import signal

SR = 1000
NFFT = 320
HOP = 25
ROW_LO, ROWS = 6, 140
WIN_COLS = 160            # 4.0 s
WIN_SECONDS = WIN_COLS * HOP / SR


def load_mono(path, start_s=None, dur_s=None):
    info = sf.info(path)
    start = 0 if start_s is None else int(start_s * info.samplerate)
    frames = -1 if dur_s is None else int(dur_s * info.samplerate)
    x, sr = sf.read(path, start=start, frames=frames, dtype='float32', always_2d=True)
    x = x[:, 0]
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def log_spectrogram(x):
    """(140, T) float32 log power, row 0 = highest frequency."""
    f, t, z = signal.stft(x, SR, window='hann', nperseg=NFFT, noverlap=NFFT - HOP,
                          boundary=None, padded=False)
    p = np.log10(np.abs(z[ROW_LO:ROW_LO + ROWS]) ** 2 + 1e-12).astype(np.float32)
    return p[::-1].copy(), t  # t: centre time of each column in seconds


def whiten(p, minutes=2.0):
    """Remove the slowly varying background per frequency row (running median
    over `minutes`), as analysts do with spectrogram equalization. Keeps
    transient calls, removes ship hum and band-limited ambient noise."""
    k = int(minutes * 60 * SR / HOP) | 1
    step = max(1, k // 15)
    med = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(p, ((0, 0), (k // 2, k // 2)), mode='reflect')[:, ::step], k // step, axis=1), axis=2)
    med = np.repeat(med, step, axis=1)[:, :p.shape[1]]
    return p - med


def to_uint8(p, lo=0.3, hi=2.5):
    """Fixed scale (log10 power above background) so brightness means the same everywhere."""
    return np.clip((p - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def windows(p, stride_cols):
    """Yield (start_col, window) over a whitened spectrogram."""
    for s in range(0, p.shape[1] - WIN_COLS + 1, stride_cols):
        yield s, p[:, s:s + WIN_COLS]
