import numpy as np, soundfile as sf, os
from scipy.signal import resample_poly
import spectro as S
SR=1000
CHUNK_MIN,MARGIN_MIN=30,1.5

def convert(src, dst):
    info = sf.info(src); sr0 = int(info.samplerate); g = np.gcd(sr0, SR)
    block = sr0 * 1800; out = []
    with sf.SoundFile(src) as f:
        while True:
            x = f.read(block, dtype='float32', always_2d=True)
            if len(x) == 0:
                break
            out.append(resample_poly(x[:, 0], SR // g, sr0 // g).astype(np.float32))
    y = np.concatenate(out)
    peak = float(np.abs(y).max()) or 1.
    sf.write(dst + '.part.flac', np.clip(y / peak * .9, -1, 1), SR, subtype='PCM_16')
    os.replace(dst + '.part.flac', dst)
    return len(y) / SR, peak, info.duration

def spectro_chunks(x):
    """Whitened uint8 spectrogram of a long signal, in 30 min chunks with margins."""
    n, c, m = len(x), CHUNK_MIN * 60 * S.SR, int(MARGIN_MIN * 60 * S.SR)
    parts = []
    for a in range(0, n, c):
        lo, hi = max(0, a - m), min(n, a + c + m)
        p, t = S.log_spectrogram(x[lo:hi]); w = S.to_uint8(S.whiten(p))
        t0 = (a - lo) / S.SR; t1 = t0 + min(c, n - a) / S.SR
        keep = (t >= t0) & (t < t1)
        parts.append(w[:, keep])
    return np.concatenate(parts, 1)
