"""Build the DCLDE 2013 one-week benchmark: every 4 s window at 2 s stride.

Outputs in ../dclde2013/:
  windows.npy   uint8 (N, 140, 160) memmap
  index.csv     file, start_s, and per-window labels:
     riwh   1 = a definite upcall lies wholly inside the window (0.25 s slack)
            0 = no right whale annotation of any confidence touches the window
           -1 = ambiguous (partial call, or only 'Possibly_Detected')
     huwh, fiwh, blwh, sewh: 1 if any call of that species touches the window
"""
import os, sys, numpy as np, pandas as pd
import spectro as S, labels as L

R = '../dclde2013'
STRIDE = 80  # columns = 2 s
SLACK = 0.25


def main():
    d = L.load(R)
    d['s'] = d.start.astype('int64') / 1e9; d['e'] = d.end.astype('int64') / 1e9
    files = sorted(f for f in os.listdir(f'{R}/audio') if f.endswith('.wav'))
    per = (900 * S.SR - S.NFFT) // S.HOP + 1
    n_per = (per - S.WIN_COLS) // STRIDE + 1
    N = n_per * len(files)
    mm = np.lib.format.open_memmap(f'{R}/windows.npy', 'w+', np.uint8, (N, S.ROWS, S.WIN_COLS))
    rows = []; k = 0
    for i, f in enumerate(files):
        t0 = L.file_start(f).timestamp() if False else pd.Timestamp(L.file_start(f)).value / 1e9
        x = S.load_mono(f'{R}/audio/{f}')
        p, t = S.log_spectrogram(x); w = S.to_uint8(S.whiten(p))
        sub = d[(d.e > t0 - 10) & (d.s < t0 + 910)]
        for s, win in S.windows(w, STRIDE):
            if k >= N: break
            a = t0 + t[s] - S.NFFT / 2 / S.SR; b = a + S.WIN_SECONDS
            touch = sub[(sub.e > a) & (sub.s < b)]
            rw = touch[touch.species == 'RIWH']
            inside = rw[(rw.s >= a - SLACK) & (rw.e <= b + SLACK) & (rw.confidence == 'Detected')]
            riwh = 1 if len(inside) else (0 if len(rw) == 0 else -1)
            sp = set(touch.species.str[:4])
            rows.append((f, round(t[s] - S.NFFT / 2 / S.SR, 3), riwh, int('HUWH' in sp), int('FIWH' in sp),
                         int('BLWH' in sp), int('SEWH' in sp)))
            mm[k] = win; k += 1
        if i % 50 == 0: print(i, len(files), k, flush=True)
    mm.flush()
    idx = pd.DataFrame(rows, columns=['file', 'start_s', 'riwh', 'huwh', 'fiwh', 'blwh', 'sewh'])
    idx.to_csv(f'{R}/index.csv', index=False)
    print(idx.riwh.value_counts().to_string(), idx[['huwh', 'fiwh']].sum().to_string(), k, N)


if __name__ == '__main__':
    main()
