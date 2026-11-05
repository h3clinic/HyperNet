"""Selection day only (28 March): score every saved checkpoint of a model at every
depth with the phase_block matcher, and pick the best by mean AP. Same 10
references (seed 7), same exclusion (8 s around each reference) and the same
6,000 negatives as select_phase.py, so v1 and v2 are chosen the same way.
Usage: python select_v2.py <model_dir>. Writes <model_dir>/selection_v2.json and
copies the winning checkpoint to <model_dir>/selected.pt."""
import sys, json, shutil, glob, os, numpy as np, pandas as pd, torch
import eval_level0 as E
from sklearn.metrics import average_precision_score as aps

torch.set_num_threads(16)
d = sys.argv[1]
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
LAYERS = ['local', 'block1', 'block2', 'block3', 'block4', 'final']
out_p = os.path.join(d, 'selection_v2.json')
res = json.load(open(out_p)) if os.path.exists(out_p) else {}
m, _ = E.load_model('../whale_ssl_v1')  # module only (SpectroEncoder class)
for ck in sorted(glob.glob(os.path.join(d, 'ckpt_*.pt'))):
    enc = m.SpectroEncoder(); enc.load_state_dict(torch.load(ck, map_location='cpu')); enc.eval()
    for layer in LAYERS:
        key = f'{os.path.basename(ck)}:{layer}'
        if key in res:
            continue
        sc = E.phase_block(m, enc, refs, boxes, X, ids, layer)
        res[key] = float(np.mean([aps(y, sc[:, j]) for j in range(sc.shape[1])]))
        print(key, round(res[key], 4), flush=True)
        json.dump(res, open(out_p, 'w'), indent=1)
cands = [k for k in res if ':' in k and k.startswith('ckpt_')]
best = max(cands, key=lambda k: res[k])
res['chosen'] = best; res['reference_v1_phase_block_local'] = 0.5434; res['reference_pixels'] = 0.5282
json.dump(res, open(out_p, 'w'), indent=1)
shutil.copy(os.path.join(d, best.split(':')[0]), os.path.join(d, 'selected.pt'))
print('chosen', best, res[best])
