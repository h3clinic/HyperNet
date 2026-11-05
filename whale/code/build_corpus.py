"""Unlabeled pretraining corpus: random 4 s windows from the downloaded Bering
files, plus DCLDE 2013 windows from 28 March only (the reference/selection day;
29 March to 3 April stay unseen for Level 0). No labels are read."""
import os, numpy as np, soundfile as sf
import spectro as S

W = '..'; PER_FILE = 3000; DCLDE_N = 20000; CHUNK = 900
rng = np.random.default_rng(4)
out = []
for f in sorted(os.listdir(f'{W}/bering/raw')):
    p = f'{W}/bering/raw/{f}'
    try:
        dur = sf.info(p).duration
    except Exception as e:
        print('skip', f, e); continue
    starts = np.arange(0, dur - CHUNK, CHUNK)
    pick = rng.choice(starts, min(8, len(starts)), replace=False)
    got = []
    for s in pick:
        w = S.to_uint8(S.whiten(S.log_spectrogram(S.load_mono(p, s, CHUNK))[0]))
        cols = rng.choice(w.shape[1] - S.WIN_COLS, PER_FILE // len(pick), replace=False)
        got += [w[:, c:c + S.WIN_COLS] for c in cols]
    out.append(np.stack(got)); print(f, len(got), flush=True)
X = np.load(f'{W}/dclde2013/windows.npy', mmap_mode='r')
import pandas as pd
idx = pd.read_csv(f'{W}/dclde2013/index.csv')
day = np.flatnonzero(idx.file.str.contains('20090328').values)
out.append(np.asarray(X[np.sort(rng.choice(day, min(DCLDE_N, len(day)), replace=False))]))
C = np.concatenate(out); np.save(f'{W}/corpus.npy', C); print('corpus', C.shape)
