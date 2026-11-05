"""One-shot search of the cached 1 kHz Bering archive from a single example.

  python scan_oneshot.py TARGET DEPLOYMENT [DEPLOYMENT ...]

TARGET names ../examples/TARGET.npz with keys window (140x160 uint8), box
(col_a, col_b, row_top, row_bottom) and speed (time compression factor K).
With K > 1 the audio is played K times faster before the spectrogram, so a
window covers 4*K real seconds and rows cover (19..456)/K real Hz (Bloop: K=8).
Windows step by 2 s of spectrogram time. For every file the scores of every
window are saved for each method (v3, pixels, xcorr) in
../bering/oneshot/TARGET/DEPLOYMENT/FILE.npz. Resumable.
"""
import os, sys, time, json, numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
import spectro as S, oneshot as O

W = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STRIDE = 80  # columns = 2 s
CHUNK_MIN, MARGIN_MIN = 30, 1.5


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


def main():
    target, deps = sys.argv[1], sys.argv[2:]
    ex = np.load(f'{W}/examples/{target}.npz')
    K = int(ex['speed'])
    os_ = O.OneShot(ex['window'], tuple(int(v) for v in ex['box']))
    for dep in deps:
        src = f'{W}/bering/audio1k/{dep}'; out = f'{W}/bering/oneshot/{target}/{dep}'; os.makedirs(out, exist_ok=True)
        files = sorted(f for f in os.listdir(src) if f.endswith('.flac') and not f.endswith('.part.flac'))
        t0 = time.time(); n = 0
        for f in files:
            dst = os.path.join(out, f.replace('.flac', '.npz'))
            if os.path.exists(dst):
                continue
            x, sr = sf.read(os.path.join(src, f), dtype='float32')
            if K > 1:  # play K times faster: keep every K-th sample after low-pass, then call it 1 kHz
                x = resample_poly(x, 1, K).astype(np.float32)
            w = spectro_chunks(x)
            starts = np.arange(0, w.shape[1] - S.WIN_COLS + 1, STRIDE)
            win = np.lib.stride_tricks.sliding_window_view(w, S.WIN_COLS, axis=1)[:, starts].transpose(1, 0, 2)
            sc = O.score_array(os_, win, bs=512)
            np.savez(dst, start_s=(starts * S.HOP / S.SR * K).astype(np.float32),
                     **{m: v.astype(np.float16) for m, v in sc.items()})
            n += 1
            if n % 10 == 0:
                print(target, dep, n, len(files), f'{time.time() - t0:.0f}s', flush=True)
        print(target, dep, 'done', n, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
