"""Google audio models on whale recordings (run in the TensorFlow venv).

- Multispecies whale model (Kaggle google/multispecies-whale, v2): 24 kHz, 5 s
  context. Classes Oo Mn Eg Be Upcall Bp Call Gunshot Echolocation Bm Whistle Ba.
  It was trained with labels, plausibly on NOAA Northeast recordings that may
  include the DCLDE 2013 week, so on DCLDE it is a supervised reference point.
- Perch 2.0 (Kaggle google/bird-vocalization-classifier, perch_v2): 32 kHz, 5 s,
  1536-d embedding plus species logits.

A 4 s spectrogram window starting at t is matched by the 5 s clip [t - 0.5, t + 4.5].

Usage: python audio_models.py dclde   -> ../audio_feats_dclde.npz aligned to dclde2013/index.csv
"""
import os, sys, time, numpy as np, pandas as pd, soundfile as sf, tensorflow as tf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, '..')
KH = os.path.expanduser('~/.cache/kagglehub/models/google')
WHALE = tf.saved_model.load(f'{KH}/multispecies-whale/tensorFlow2/default/2')
PERCH = tf.saved_model.load(f'{KH}/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')
WHALE_CLASSES = [c.decode() for c in WHALE.signatures['metadata']()['class_names'].numpy()]
LABELS = pd.read_csv(f'{KH}/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1/assets/labels.csv').iloc[:, 0].tolist()
PERCH_KEEP = [i for i, n in enumerate(LABELS) if any(s in str(n) for s in
              ('Eubalaena', 'Balaena', 'Balaenoptera', 'Megaptera', 'Orcinus', 'Physeter', 'Odobenus', 'Erignathus'))]


def load(path, sr):
    x, r = sf.read(path, dtype='float32', always_2d=True)
    x = x[:, 0]
    if r != sr:
        g = np.gcd(int(r), sr); x = resample_poly(x, sr // g, int(r) // g).astype(np.float32)
    return x


def clips(x, sr, starts, dur=5.0, lead=0.5):
    n = int(dur * sr); out = np.zeros((len(starts), n), np.float32)
    for i, t in enumerate(starts):
        a = int(round((t - lead) * sr)); lo, hi = max(a, 0), min(a + n, len(x))
        if hi > lo:
            out[i, lo - a:hi - a] = x[lo:hi]
    return out


def whale(c24):
    """c24: (N, 120000) -> logits (N,12), features (N, D)."""
    spec = WHALE.signatures['front_end'](waveform=tf.constant(c24[..., None]))['output_0']
    spec = spec[:, :128, :128] if spec.shape[1] >= 128 else spec
    lg = WHALE.signatures['logits'](spectrogram=spec)['output_0'].numpy()
    ft = WHALE.signatures['features'](spectrogram=spec)['output_0'].numpy()
    return lg.reshape(len(c24), -1), ft.reshape(len(c24), -1)


def perch(c32, bs=64):
    emb, lab = [], []
    for i in range(0, len(c32), bs):
        o = PERCH.signatures['serving_default'](inputs=tf.constant(c32[i:i + bs]))
        emb.append(o['embedding'].numpy()); lab.append(o['label'].numpy()[:, PERCH_KEEP])
    return np.concatenate(emb), np.concatenate(lab)


def file_feats(path, starts):
    x24, x32 = load(path, 24000), load(path, 32000)
    wl, wf = whale(clips(x24, 24000, starts))
    pe, pl = perch(clips(x32, 32000, starts))
    return wl, wf, pe, pl


def whale_only(path, starts):
    return whale(clips(load(path, 24000), 24000, starts))


def ref_starts(seed, n):
    """Start times (file, s) of 4 s windows centred on the Level 0 reference calls (same draw as eval_level0.references)."""
    sys.path.insert(0, os.path.join(W, 'code'))
    import labels as L
    d = L.load(f'{W}/dclde2013')
    c = d[(d.species == 'RIWH') & (d.confidence == 'Detected') & (d.start.dt.day == 28)].sample(n, random_state=seed)
    out = []
    for _, r in c.iterrows():
        f0 = r.start.floor('15min'); mid = (r.start - f0).total_seconds() + (r.end - r.start).total_seconds() / 2
        out.append((f'NOPP6_EST_{f0:%Y%m%d_%H%M%S}_CH10.wav', float(np.clip(mid - 2, 0, 896))))
    return out


def main():
    mode = sys.argv[1]
    idx = pd.read_csv(f'{W}/dclde2013/index.csv')
    t0 = time.time()
    if mode == 'whale':  # Google whale model on every DCLDE window
        wl, wf = [], []
        for n, f in enumerate(idx.file.unique()):
            a, b = whale_only(f'{W}/dclde2013/audio/{f}', idx.start_s.values[idx.file.values == f])
            wl.append(a.astype(np.float32)); wf.append(b.astype(np.float16))
            if n % 50 == 0:
                print(n, f'{time.time() - t0:.0f}s', flush=True)
        np.savez(f'{W}/whale_model_dclde.npz', logits=np.concatenate(wl), feats=np.concatenate(wf),
                 classes=np.array(WHALE_CLASSES))
    elif mode == 'refs':  # embeddings of the Level 0 reference windows (selection and test references)
        out = {}
        for tag, seed, n in (('sel', 7, 10), ('test', 2013, 20)):
            rs = ref_starts(seed, n); wl, wf, pe = [], [], []
            for f, s in rs:
                a, b = whale(clips(load(f'{W}/dclde2013/audio/{f}', 24000), 24000, [s]))
                e, _ = perch(clips(load(f'{W}/dclde2013/audio/{f}', 32000), 32000, [s]))
                wf.append(b[0]); pe.append(e[0])
            out[f'{tag}_whale'] = np.stack(wf); out[f'{tag}_perch'] = np.stack(pe)
        np.savez(f'{W}/audio_refs.npz', **out)
    elif mode == 'perch_ids':  # Perch embeddings for a list of window ids (npy file)
        ids = np.load(sys.argv[2]); emb = np.zeros((len(ids), 1536), np.float16)
        sub = idx.iloc[ids]
        for n, f in enumerate(sub.file.unique()):
            m = (sub.file.values == f)
            emb[m] = perch(clips(load(f'{W}/dclde2013/audio/{f}', 32000), 32000, sub.start_s.values[m]))[0]
            if n % 10 == 0:
                print(n, f'{time.time() - t0:.0f}s', flush=True)
        np.save(sys.argv[3], emb)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
