"""Selection day only: phase-aligned score at each depth. Appends to ../variant_selection.json."""
import sys, json, numpy as np, pandas as pd, torch
import eval_level0 as E
from sklearn.metrics import average_precision_score as aps
torch.set_num_threads(10)
model_dir = sys.argv[1]
refs, boxes, meta = E.references(seed=7, n=10)
idx = pd.read_csv(f'{E.R}/index.csv')
day = idx.file.str.contains('20090328').values & (idx.riwh.values >= 0)
t = pd.to_datetime(idx.file.str.extract(r'_(\d{8}_\d{6})_')[0], format='%Y%m%d_%H%M%S') + pd.to_timedelta(idx.start_s, 's')
near = np.zeros(len(idx), bool)
for s in meta.start: near |= (abs((t - s).dt.total_seconds()) < 8).values
rng = np.random.default_rng(7)
pos = np.flatnonzero(day & ~near & (idx.riwh.values == 1)); neg = np.flatnonzero(day & ~near & (idx.riwh.values == 0))
ids = np.sort(np.concatenate([pos, rng.choice(neg, 6000, replace=False)]))
X = np.load(f'{E.R}/windows.npy', mmap_mode='r'); y = idx.riwh.values[ids]
m, enc = E.load_model(model_dir)
NAME = sys.argv[2] if len(sys.argv) > 2 else 'phase_aligned'
FN = {'phase_aligned': E.phase_aligned, 'phase_block': E.phase_block}[NAME]
LAYERS = ['pixels', 'local', 'block1', 'block2', 'block4'] + (['desc'] if NAME == 'phase_aligned' else [])
res = json.load(open('../variant_selection.json'))
for layer in LAYERS:
    sc = FN(m, enc, refs, boxes, X, ids, layer)
    res[f'{NAME}:{layer}'] = float(np.mean([aps(y, sc[:, j]) for j in range(sc.shape[1])]))
    print(layer, round(res[f'{NAME}:{layer}'], 4), flush=True)
cands = [k for k in res if k in ('token_nn', 'box_token_nn', 'aligned') or (k.startswith(('phase_aligned:', 'phase_block:')) and not k.endswith(':pixels'))]
res['chosen'] = max(cands, key=lambda k: res[k])
json.dump(res, open('../variant_selection.json', 'w'), indent=1); print('chosen', res['chosen'])
