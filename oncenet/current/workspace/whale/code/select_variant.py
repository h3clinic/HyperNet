"""Choose the HyperNet scoring variant on 28 March only (never the Level 0 test days).
References: 10 definite upcalls from 28 March, seed 7 (distinct draw from the
Level 0 references). Queries: 28 March windows, excluding any window within 8 s
of a reference call, all positives plus 6,000 negatives. Writes ../variant_selection.json."""
import sys, json, numpy as np, pandas as pd, torch
import eval_level0 as E, labels as L
from sklearn.metrics import average_precision_score as aps

model_dir = sys.argv[1]
refs, boxes, meta = E.references(seed=7, n=10)
idx = pd.read_csv(f'{E.R}/index.csv')
day = idx.file.str.contains('20090328').values & (idx.riwh.values >= 0)
t = pd.to_datetime(idx.file.str.extract(r'_(\d{8}_\d{6})_')[0], format='%Y%m%d_%H%M%S') + pd.to_timedelta(idx.start_s, 's')
near = np.zeros(len(idx), bool)
for s in meta.start:
    near |= (abs((t - s).dt.total_seconds()) < 8).values
rng = np.random.default_rng(7)
pos = np.flatnonzero(day & ~near & (idx.riwh.values == 1)); neg = np.flatnonzero(day & ~near & (idx.riwh.values == 0))
ids = np.sort(np.concatenate([pos, rng.choice(neg, 6000, replace=False)]))
X = np.load(f'{E.R}/windows.npy', mmap_mode='r'); y = idx.riwh.values[ids]
out = {'queries': int(len(ids)), 'positives': int(y.sum())}
mean_ap = lambda sc: float(np.mean([aps(y, sc[:, j]) for j in range(sc.shape[1])]))
out['raw_pixels'] = mean_ap(E.score_raw(refs, X, ids, E.std_flat))
out['xcorr'] = mean_ap(E.score_xcorr(refs, boxes, X, ids))
out['xcorr_box'] = mean_ap(E.score_xcorr(refs, boxes, X, ids, freq_box=True))
for k, v in E.score_model(model_dir, refs, boxes, X, ids).items():
    out[k] = mean_ap(v)
out['chosen'] = max(['token_nn', 'box_token_nn', 'aligned'], key=lambda k: out[k])
json.dump(out, open('../variant_selection.json', 'w'), indent=1); print(json.dumps(out, indent=1))
