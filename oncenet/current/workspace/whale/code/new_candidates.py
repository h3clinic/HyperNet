"""Top sei-example matches across the cached Bering data (excluding the June 2005
fin whale control), each classified by call shape as in fair_measure.py:
sei-like = ends <= 38 Hz and sweeps >= 18 Hz; fin-like = ends >= 42 Hz."""
import glob, os, re, json, numpy as np, soundfile as sf
from fair_measure import measure
rows, bad = [], 0
for f in glob.glob('../bering/oneshot/sei/*/*.npz'):
    if re.search(r'_0506(2[2-9]|30)_', f): continue
    try: z = np.load(f); v = z['v3'].astype(float); s = z['start_s']
    except Exception: bad += 1; continue
    for i in np.argsort(-v)[:5]: rows.append((v[i], f, float(s[i])))
rows.sort(reverse=True); pick = []
for r in rows:
    if any(r[1] == p[1] and abs(r[2] - p[2]) < 20 for p in pick): continue
    pick.append(r)
    if len(pick) == 40: break
out = []
for v, f, s in pick:
    dep = os.path.basename(os.path.dirname(f)); name = os.path.basename(f)[:-4]
    x = sf.read(f'../bering/audio1k/{dep}/{name}.flac', dtype='float32')[0]; m = measure(x, s) or {}
    kind = 'sei-like' if m and m['end'] <= 38 and m['width'] >= 18 else 'fin-like' if m and m['end'] >= 42 else 'unclear'
    out.append(dict(score=round(v, 3), dep=dep, file=name, s=s, kind=kind, **m))
    print(f'{v:.3f} {dep[-7:]} {name[-13:]} +{s:6.0f}  {kind:9s} {m}')
print('unreadable files:', bad)
json.dump(out, open('../new_candidates.json', 'w'), indent=1, default=float)
