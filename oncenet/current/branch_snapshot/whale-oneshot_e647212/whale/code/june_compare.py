"""Compare the one-shot sei search on the June 2005 fin whale peak (control) with
autumn 2004 at the same mooring: matching events per hour above fixed score
thresholds, the best June matches as pictures, and their measured frequencies."""
import glob, os, re, json, numpy as np, soundfile as sf, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import ndimage
import spectro as S
W = '..'

def events(files, thr):
    n, hours, best = 0, 0., []
    for f in files:
        z = np.load(f); s = z['start_s']; v = z['v3'].astype(float); hours += (s[-1] + 4) / 3600 if len(s) else 0
        last = -1e9
        for i in np.flatnonzero(v >= thr):
            if s[i] - last > 4: n += 1
            last = s[i]
        for i in np.argsort(-v)[:3]: best.append((v[i], float(z['xcorr'][i]), os.path.basename(f)[:-4], float(s[i])))
    return n, hours, best

def hz(r): return np.interp(r, [0, 139], [456, 19])
def measure(w):
    m = (w >= 110) & (np.arange(140)[:, None] >= 95)
    lab, n = ndimage.label(ndimage.binary_opening(m, np.ones((2, 2))))
    if not n: return None
    k = np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1; rr, cc = np.nonzero(lab == k); cols = np.unique(cc)
    ridge = [hz(rr[cc == c][np.argmax(w[rr[cc == c], c])]) for c in cols]
    return round(ridge[0], 1), round(ridge[-1], 1), round((cols[-1] - cols[0] + 1) * S.HOP / S.SR, 2)
def window(path, s):
    x, _ = sf.read(path, dtype='float32'); seg = x[max(0, int((s - 30) * 1000)):int((s + 34) * 1000)]
    p, t = S.log_spectrogram(seg); w = S.to_uint8(S.whiten(p)); c = int(np.searchsorted(t, min(30, s))); return w[:, c:c + 160]

oct_ = sorted(glob.glob(f'{W}/bering/oneshot/sei/mbarc_lmr_bering02-m2/*.npz'))
jun = sorted(f for f in glob.glob(f'{W}/bering/oneshot/sei/mbarc_lmr_bering03-m2/*.npz') if re.search(r'_0506(2[2-9]|30)_', f))
res = {}
for thr in (0.65, 0.70, 0.75):
    a = events(oct_, thr); b = events(jun, thr)
    res[thr] = dict(autumn_events=a[0], autumn_hours=round(a[1]), autumn_per_100h=round(100 * a[0] / a[1], 2),
                    june_events=b[0], june_hours=round(b[1]), june_per_100h=round(100 * b[0] / b[1], 2))
    print(thr, res[thr])
_, _, best = events(jun, 1.0); best.sort(reverse=True); pick = []
for r in best:
    if any(r[2] == p[2] and abs(r[3] - p[3]) < 30 for p in pick): continue
    pick.append(r)
    if len(pick) == 8: break
fig, ax = plt.subplots(2, 4, figsize=(16, 7)); rows = []
for a, (v, x, f, s) in zip(ax.flat, pick):
    w = window(f'{W}/bering/audio1k/mbarc_lmr_bering03-m2/{f}.flac', s); m = measure(w)
    a.imshow(w[60:], cmap='magma', aspect='auto', extent=[0, 4, 19, 269]); a.set_title(f'{f[14:]} +{s:.0f}s  v3 {v:.2f}\n{m}', fontsize=9)
    rows.append(dict(file=f, start_s=s, v3=round(v, 3), xcorr=round(x, 3), start_end_dur=m)); print(rows[-1])
plt.suptitle('CONTROL: best sei-example matches, M2 22-30 June 2005 (fin whale 40 Hz peak)'); plt.tight_layout()
plt.savefig(f'{W}/june_control_hits.png', dpi=75)
json.dump(dict(rates=res, june_top=rows), open(f'{W}/june_control_results.json', 'w'), indent=1, default=float)
