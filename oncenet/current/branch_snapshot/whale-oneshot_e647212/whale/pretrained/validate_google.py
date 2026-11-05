"""Independent check of the one-shot findings with Google's multispecies whale
model (trained, 12 classes, no sei class). Only the found windows are scored:

  candidates  sei-like matches (Oct-Nov 2004)        ../new_candidates.json
  fin         fin-like matches: June 2005 control top 8 + Oct-Nov M4 fin-like
  sei_ref     the 3 labeled DCLDE 2013 sei calls (positive reference)
  background  random windows from the same Bering files (negative reference)

Bering audio exists here only as the 1 kHz cache, so every group (including
sei_ref, scored both ways) is upsampled 1 kHz -> 24 kHz the same way.
Writes ../google_validation.json."""
import json, os, numpy as np, soundfile as sf
from scipy.signal import resample_poly
import audio_models as M

W = os.path.abspath('..'); rng = np.random.default_rng(0)


def from1k(path, starts):
    x = sf.read(path, dtype='float32')[0]
    return M.whale(M.clips(resample_poly(x, 24, 1).astype(np.float32), 24000, starts))[0]


def a1k(dep, f): return f'{W}/bering/audio1k/{dep}/{f}.flac'


cand = json.load(open(f'{W}/new_candidates.json'))
june = json.load(open(f'{W}/june_control_results.json'))['june_top']
groups = {'candidates': [(c['dep'], c['file'], c['s']) for c in cand if c['kind'] == 'sei-like'],
          'fin': [('mbarc_lmr_bering03-m2', r['file'], r['start_s']) for r in june] +
                 [(c['dep'], c['file'], c['s']) for c in cand if c['kind'] == 'fin-like']}
files = sorted({(d, f) for g in groups.values() for d, f, _ in g})
groups['background'] = [(d, f, float(rng.uniform(60, 9000))) for d, f in files for _ in range(4)]

res = {'classes': M.WHALE_CLASSES}
for g, items in groups.items():
    lg = np.concatenate([from1k(a1k(d, f), [s]) for d, f, s in items])
    res[g] = lg.tolist()
# labeled sei: original 2 kHz+ audio, and the same audio through the 1 kHz path
wav = f'{W}/dclde2013/audio/NOPP6_EST_20090402_194500_CH10.wav'; st = [47.5, 139.5, 170.5]
res['sei_ref_full'] = M.whale(M.clips(M.load(wav, 24000), 24000, st))[0].tolist()
x1 = M.load(wav, 1000); res['sei_ref_1k'] = M.whale(M.clips(resample_poly(x1, 24, 1).astype(np.float32), 24000, st))[0].tolist()
json.dump(res, open(f'{W}/google_validation.json', 'w'), indent=1)

C = M.WHALE_CLASSES
print('class        ' + ' '.join(f'{c[:6]:>7s}' for c in C))
for g in ('candidates', 'fin', 'background', 'sei_ref_full', 'sei_ref_1k'):
    a = np.array(res[g]); print(f'{g:12s} n={len(a):2d} ' + ' '.join(f'{v:7.2f}' for v in np.median(a, 0)))
bp = C.index('Bp')
for g in ('candidates', 'fin', 'background', 'sei_ref_1k'):
    a = np.array(res[g])[:, bp]; print(g, 'fin (Bp) logit > 0 in', int((a > 0).sum()), 'of', len(a), ' values', np.round(a, 1).tolist())
