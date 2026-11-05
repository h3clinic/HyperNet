"""Sei whale, caught once: each of the 3 sei calls in the DCLDE 2013 week is used
alone as the example; the other 2 are the answer key among all 301,056 windows.
Windows within 10 s of the example call are removed (they contain it).
Reports, per held-out call, the rank of its best window (1 = top of the list),
and the AP over all sei-touching windows."""
import json, numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score as aps
import labels as L, oneshot as O

R = '../dclde2013'
d = L.load(R); sei = d[d.species == 'SEWH'].sort_values('start').reset_index(drop=True)
idx = pd.read_csv(f'{R}/index.csv'); X = np.load(f'{R}/windows.npy', mmap_mode='r')
FILE = 'NOPP6_EST_20090402_194500_CH10.wav'; f0 = pd.Timestamp('2009-04-02 19:45:00')
infile = idx.file.values == FILE
out = {}
for j, r in sei.iterrows():
    s, e = (r.start - f0).total_seconds(), (r.end - f0).total_seconds()
    win, box = O.example_from_audio(f'{R}/audio/{FILE}', s, e, r.low, r.high)
    os_ = O.OneShot(win, box)
    near = infile & (idx.start_s.values + 4 > s - 10) & (idx.start_s.values < e + 10)
    ids = np.flatnonzero(~near)
    sc = O.score_array(os_, X, ids)
    res = {}
    for m, v in sc.items():
        order = np.argsort(-v); rank = np.empty(len(v), int); rank[order] = np.arange(1, len(v) + 1)
        held = {}
        for k, r2 in sei.iterrows():
            if k == j:
                continue
            s2, e2 = (r2.start - f0).total_seconds(), (r2.end - f0).total_seconds()
            w = infile[ids] & (idx.start_s.values[ids] <= s2) & (idx.start_s.values[ids] + 4 >= e2)
            held[f'call_{k}'] = int(rank[w].min()) if w.any() else None
        y = (idx.sewh.values[ids] == 1) & ~near[ids]
        res[m] = {'best_rank': held, 'ap': float(aps(y, v)), 'n': int(len(v))}
        print(f'example call {j} ({s:.0f} s)', m, res[m], flush=True)
    out[f'example_{j}'] = res
json.dump(out, open('../sei_oneshot_results.json', 'w'), indent=1)
