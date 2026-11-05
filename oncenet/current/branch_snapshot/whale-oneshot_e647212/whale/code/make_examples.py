"""Single examples for the one-shot Bering searches -> ../examples/NAME.npz (+ .png).

bloop   NOAA PMEL 'Bloop' (1997), public file sped up 16x (8 kHz). The event is
        ~10 s at 5-35 Hz real (real 45-57 s into the file). Search speed K = 8:
        5-35 Hz real -> 40-280 Hz in the model's rows, 4 s window = 32 s real.
narw    one North Atlantic right whale upcall: the first Level 0 test reference
        (seed 2013), fixed by rule, not picked by eye. K = 1.
sei     one sei whale downsweep from DCLDE 2013 (2 April 19:47:20). K = 1.
"""
import numpy as np, soundfile as sf, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import spectro as S, eval_level0 as E, labels as L, oneshot as O

OUT = '../examples'


def save(name, win, box, speed, note):
    import os; os.makedirs(OUT, exist_ok=True)
    np.savez(f'{OUT}/{name}.npz', window=win, box=np.array(box), speed=speed, note=note)
    a, b, r0, r1 = box
    plt.figure(figsize=(4, 3.5)); plt.imshow(win, cmap='magma', aspect='auto')
    plt.gca().add_patch(plt.Rectangle((a, r0), b - a, r1 - r0, fill=False, ec='cyan')); plt.title(f'{name} (K={speed})')
    plt.tight_layout(); plt.savefig(f'{OUT}/{name}.png', dpi=80); plt.close()
    print(name, box, note)


# Bloop: raw samples are 8 kHz of 16x-sped audio = 500 Hz real. At K = 8 they are 4 kHz.
x, sr = sf.read('../bloop/bloop.wav', dtype='float32', always_2d=True); x = x[:, 0]
x1k = resample_poly(x, 1, 4).astype(np.float32)  # 4 kHz (K=8 time) -> 1 kHz
p, t = S.log_spectrogram(x1k); w = S.to_uint8(S.whiten(p))
K = 8; ev0, ev1 = 45 / K, 57 / K
c0 = int(np.clip(np.searchsorted(t, (ev0 + ev1) / 2) - S.WIN_COLS // 2, 0, w.shape[1] - S.WIN_COLS))
box = (int(np.searchsorted(t, ev0) - c0), int(np.searchsorted(t, ev1) - c0), O.hz_row(35 * K), O.hz_row(5 * K) + 1)
save('bloop', w[:, c0:c0 + S.WIN_COLS], box, K, 'NOAA PMEL bloop.wav, real 45-57 s, 5-35 Hz')

refs, boxes, meta = E.references()
save('narw', refs[0], boxes[0], 1, f'DCLDE 2013 RIWH upcall {meta.start.iloc[0]} (first Level 0 test reference)')

d = L.load('../dclde2013'); r = d[(d.species == 'SEWH')].sort_values('start').iloc[1]
f0 = pd.Timestamp('2009-04-02 19:45:00')
win, box = O.example_from_audio('../dclde2013/audio/NOPP6_EST_20090402_194500_CH10.wav',
                                (r.start - f0).total_seconds(), (r.end - f0).total_seconds(), r.low, r.high)
save('sei', win, box, 1, f'DCLDE 2013 SEWH {r.start}')
