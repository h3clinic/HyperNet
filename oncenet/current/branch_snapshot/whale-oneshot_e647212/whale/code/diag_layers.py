"""Selection-day diagnostic: aligned score using tokens from different depths."""
import sys, json, numpy as np, pandas as pd, torch
from torch.nn import functional as F
import eval_level0 as E
from sklearn.metrics import average_precision_score as aps
torch.set_num_threads(4)
model_dir = sys.argv[1]
refs, boxes, meta = E.references(seed=7, n=10)
sel = json.load(open('../variant_selection.json'))
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
masks = [E.box_mask(b) for b in boxes]

@torch.no_grad()
def feats(x, which):
    x = x[:, None]
    if which == 'pixels':
        u = F.unfold(x, 10, stride=10).transpose(1, 2)            # (B, 224, 100), row-major tokens
        u = u - u.mean(-1, keepdim=True); return F.normalize(u, dim=-1)
    z = enc.patch(m.two_channel(x))
    if which == 'patch_embed':
        return F.normalize(z.flatten(2).transpose(1, 2), dim=-1)
    z = z + enc.freq; z = z + enc.local(z)
    if which == 'local':
        return F.normalize(z.flatten(2).transpose(1, 2), dim=-1)
    b, c, h, w = z.shape; tt = z.flatten(2).transpose(1, 2)
    for k, layer in enumerate(enc.blocks.layers):
        tt = layer(tt)
        if which == f'block{k + 1}':
            return F.normalize(tt, dim=-1)
    if which == 'final_desc':
        return enc.describe(x[:, 0:1].squeeze(1)[:, None] if False else x)[0]

out = {}
rt = torch.from_numpy(refs.astype(np.float32) / 255)
for which in ['pixels', 'patch_embed', 'local', 'block1', 'block2', 'block4', 'final_desc']:
    if which == 'final_desc':
        rd = enc.describe(rt[:, None])[0]
    else:
        rd = feats(rt, which)
    sc = []
    for q in E.batches(X, ids, 512):
        qd = enc.describe(q[:, None])[0] if which == 'final_desc' else feats(q, which)
        sc.append(torch.stack([E.aligned_score(qd, rd[j], masks[j]) for j in range(len(rd))], 1))
    sc = torch.cat(sc).numpy()
    out[which] = float(np.mean([aps(y, sc[:, j]) for j in range(sc.shape[1])])); print(which, round(out[which], 4), flush=True)
json.dump(out, open('../diag_layers.json', 'w'), indent=1)
