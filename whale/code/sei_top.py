"""Top-ranked windows from the single sei example (2 April 19:47:20) that are NOT
labeled sei: candidate sei calls the analysts did not mark. Saves a contact
sheet and a CSV with the labels each window carries."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import oneshot as O

R = '../dclde2013'
ex = np.load('../examples/sei.npz')
idx = pd.read_csv(f'{R}/index.csv'); X = np.load(f'{R}/windows.npy', mmap_mode='r')
os_ = O.OneShot(ex['window'], tuple(int(v) for v in ex['box']))
sc = O.score_array(os_, X)
rows = []
for m in ('v3', 'xcorr'):
    v = sc[m].copy()
    # one window per 10 s: suppress neighbours of each pick
    order = np.argsort(-v); taken = []
    for i in order:
        f, s = idx.file.values[i], idx.start_s.values[i]
        if any(f == idx.file.values[j] and abs(s - idx.start_s.values[j]) < 10 for j in taken):
            continue
        taken.append(i)
        if len(taken) == 40:
            break
    for rnk, i in enumerate(taken, 1):
        rows.append(dict(method=m, rank=rnk, i=int(i), file=idx.file.values[i], start_s=float(idx.start_s.values[i]),
                         score=float(v[i]), sei=int(idx.sewh.values[i]), riwh=int(idx.riwh.values[i]),
                         huwh=int(idx.huwh.values[i]), fiwh=int(idx.fiwh.values[i]), blwh=int(idx.blwh.values[i])))
df = pd.DataFrame(rows); df.to_csv('../sei_top.csv', index=False)
print(df.groupby('method')[['sei', 'riwh', 'huwh', 'fiwh', 'blwh']].sum())
cand = df[(df.method == 'v3') & (df.sei == 0)].head(24)
fig, ax = plt.subplots(4, 6, figsize=(18, 10))
for a, (_, r) in zip(ax.flat, cand.iterrows()):
    a.imshow(X[r.i][60:], cmap='magma', aspect='auto', extent=[0, 4, 19, 19 + 80 * 3.125])
    lab = ','.join(k for k in ('riwh', 'huwh', 'fiwh', 'blwh') if r[k] == 1) or 'unlabeled'
    a.set_title(f'#{r["rank"]} {r.file[10:25]} +{r.start_s:.0f}s\n{lab}', fontsize=8); a.tick_params(labelsize=6)
plt.suptitle('Top non-sei windows from ONE sei example (HyperNet v3), 19-270 Hz'); plt.tight_layout()
plt.savefig('../sei_candidates.png', dpi=70)
print(cand[['rank', 'file', 'start_s', 'score', 'riwh', 'huwh', 'fiwh', 'blwh']].to_string())
