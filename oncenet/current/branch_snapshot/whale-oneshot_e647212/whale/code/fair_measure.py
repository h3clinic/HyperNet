"""Measure October candidates and June control hits the same way: an 8 s window
centred on the match, the largest bright blob below ~160 Hz, its ridge start and
end frequency, duration and sweep width. Writes ../fair_measure.json."""
import json, numpy as np, soundfile as sf
from scipy import ndimage
import spectro as S
def hz(r): return float(np.interp(r, [0, 139], [456, 19]))
def measure(x, s):
    a = s - 2; seg = x[max(0, int((a - 30) * 1000)):int((a + 38) * 1000)]
    p, t = S.log_spectrogram(seg); w = S.to_uint8(S.whiten(p)); c = int(np.searchsorted(t, min(30, a))); w = w[:, c:c + 320]
    m = (w >= 110) & (np.arange(140)[:, None] >= 95)
    lab, n = ndimage.label(ndimage.binary_opening(m, np.ones((2, 2))))
    if not n: return None
    k = np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1; rr, cc = np.nonzero(lab == k); cols = np.unique(cc)
    ridge = [hz(rr[cc == c][np.argmax(w[rr[cc == c], c])]) for c in cols]
    d = (cols[-1] - cols[0] + 1) * S.HOP / S.SR; edge = cols[0] == 0 or cols[-1] == w.shape[1] - 1
    return dict(start=round(max(ridge[:3]), 1), end=round(min(ridge[-3:]), 1), dur=round(d, 2), width=round(max(ridge[:3]) - min(ridge[-3:]), 1), edge=bool(edge))
def main():
    oct_ = [('mbarc_lmr_bering02-m2', f, s) for f, s in [('Bering_2_stM2_041025_074456',3366),('Bering_2_stM2_041025_034841',7710),('Bering_2_stM2_041025_074456',3980),
            ('Bering_2_stM2_041025_034841',6092),('Bering_2_stM2_041025_034841',7552),('Bering_2_stM2_041121_012045',5624),('Bering_2_stM2_041025_192804',8460)]]
    jun = [('mbarc_lmr_bering03-m2', r['file'], r['start_s']) for r in json.load(open('../june_control_results.json'))['june_top']]
    res = {}
    for name, L in (('october', oct_), ('june', jun)):
        res[name] = []
        for d, f, s in L:
            x = sf.read(f'../bering/audio1k/{d}/{f}.flac', dtype='float32')[0]; r = measure(x, s); res[name].append(dict(file=f, s=s, **(r or {})))
            print(name, f[-13:], s, r)
        v = [r for r in res[name] if 'end' in r]
        print(name, 'median end', np.median([r['end'] for r in v]), 'median dur', np.median([r['dur'] for r in v]), 'median width', np.median([r['width'] for r in v]))
    json.dump(res, open('../fair_measure.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
