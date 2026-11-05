"""DEV-only readout study for Gravity Spy v0.1-v0.3 artifacts. Never reads TEST scores or TEST rows."""
import sys, json, time
import numpy as np, pandas as pd, h5py, cv2, torch, torch.nn.functional as F
ROOT = 'C:/Users/brita/Documents/Codex/2026-11-01/wan'
sys.path.insert(0, ROOT + '/public/HyperNet'); sys.path.insert(0, ROOT + '/public/HyperNet/scripts')
from hypernet import gravity_spy as gs
from hypernet.part_encoder import crop_image
from hypernet.part_detector import PartDetector, tensor
t0 = time.time()
meta = pd.read_csv(ROOT + '/gravity_spy_data/trainingset_v1d1_metadata.csv')
refs_recs = [json.loads(l) for l in open(ROOT + '/public/HyperNet/benchmarks/gravity_spy_v0.1/per_reference.jsonl')]
ref_ids = {}
for r in refs_recs:
    ref_ids[(r['class'], r['reference'])] = r['reference_gravityspy_id']
classes = sorted({k[0] for k in ref_ids})
id2row = {g: i for i, g in enumerate(meta['gravityspy_id'])}
ref_rows = {c: [id2row[ref_ids[(c, r)]] for r in range(5)] for c in classes}
all_ref_rows = set(sum(ref_rows.values(), []))
pool = np.array([i for i in meta.index[meta['sample_type'] == 'train'] if i not in all_ref_rows])
std_rows = np.sort(np.random.default_rng(7).choice(pool, 600, replace=False))
dev_rows = meta.index[meta['sample_type'] == 'validation'].to_numpy()
dev_labels = meta.loc[dev_rows, 'label'].to_numpy()

f = h5py.File(ROOT + '/gravity_spy_data/trainingsetv1d1.h5', 'r')
def raw(row):
    m = meta.loc[row]
    return np.squeeze(f[m['label']][m['sample_type']][m['gravityspy_id']]['1.0.png'][()]).astype(np.float32)
def crop_of(a):
    return crop_image(gs.to_uint8_rgb(a))[0]
dev_raw = [raw(r) for r in dev_rows]; std_raw = [raw(r) for r in std_rows]
ref_order = [(c, r) for c in classes for r in range(5)]
ref_raw = [raw(ref_rows[c][r]) for c, r in ref_order]
dev_crops = [crop_of(a) for a in dev_raw]; std_crops = [crop_of(a) for a in std_raw]; ref_crops = [crop_of(a) for a in ref_raw]
dev_blank = np.array([a.std() == 0 for a in dev_raw]); ref_blank = np.array([a.std() == 0 for a in ref_raw])
print('loaded', time.time() - t0, 'dev blanks', dev_blank.sum(), 'ref blanks', [ref_order[i] for i in np.flatnonzero(ref_blank)])

def unit(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
def gray_px(crop, side):
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
def per_image_z(X):
    X = X - X.mean(1, keepdims=True); return X / (X.std(1, keepdims=True) + 1e-6)

def eval_dev(ref_feats, dev_feats, keep=None, nontarget_blank=False, name='', verbose=False):
    """ref_feats: (100*2, d) rows [ref view0, view1,...]; cosine via unit vectors; returns mean AP and per-class."""
    rf = unit(ref_feats); df = unit(dev_feats)
    keep = np.ones(len(df), bool) if keep is None else keep
    per = {}
    for ci, c in enumerate(classes):
        aps = []
        for r in range(5):
            k = ci * 5 + r
            s = (rf[2 * k:2 * k + 2] @ df.T).max(0)
            pos = dev_labels == c
            if nontarget_blank: pos = pos & ~dev_blank
            aps.append(gs.average_precision(s[keep], pos[keep]))
        per[c] = float(np.mean(aps))
    m = float(np.mean(list(per.values())))
    rare = float(np.mean([per[c] for c in ('1400Ripples', 'Air_Compressor', 'Chirp', 'Paired_Doves', 'Wandering_Line')]))
    focus = ' '.join(f'{c[:5]}={per[c]:.2f}' for c in ('Blip', 'Tomte', 'Chirp', 'Power_Line', 'Air_Compressor', 'Extremely_Loud', 'Scattered_Light', '1080Lines'))
    print(f'{name:48s} DEV mAP {m:.3f} rare {rare:.3f} | {focus}')
    return m, per

def score_fn(ref_feats, dev_feats):
    rf = unit(ref_feats); df = unit(dev_feats); out = {}
    for ci, c in enumerate(classes):
        for r in range(5):
            k = ci * 5 + r; out[(c, r)] = (rf[2 * k:2 * k + 2] @ df.T).max(0)
    return out
def eval_scores(S, name):
    per = {c: float(np.mean([gs.average_precision(S[(c, r)], dev_labels == c) for r in range(5)])) for c in classes}
    m = float(np.mean(list(per.values())))
    focus = ' '.join(f'{c[:5]}={per[c]:.2f}' for c in ('Blip', 'Tomte', 'Chirp', 'Power_Line', 'Air_Compressor', 'Extremely_Loud', 'Scattered_Light', '1080Lines'))
    print(f'{name:48s} DEV mAP {m:.3f} | {focus}'); return m, per

def views(feat_fn, crops):
    out = []
    for c in crops: out += [feat_fn(c), feat_fn(np.ascontiguousarray(c[:, ::-1]))]
    return np.stack(out)

results = {}
# ---------- raw pixel arms ----------
print('\n== Raw pixel normalization arms (48x48 of 96px crop) ==')
px = lambda c: gray_px(c, 48).ravel()
R = views(px, ref_crops); D = np.stack([px(c) for c in dev_crops]); S = np.stack([px(c) for c in std_crops])
results['raw48_perimage_z'] = eval_dev(per_image_z(R), per_image_z(D), name='raw48 per-image z (v0.1 baseline)')
mu, sd = S.mean(0), S.std(0) + 1e-6
results['raw48_feat_std600'] = eval_dev((R - mu) / sd, (D - mu) / sd, name='raw48 per-feature std, 600 imgs (v0.2 baseline)')
results['raw48_feat_mean_only'] = eval_dev(R - mu, D - mu, name='raw48 per-feature mean-subtract only (600)')
g_mu, g_sd = S.mean(), S.std()
results['raw48_global'] = eval_dev((R - g_mu) / g_sd, (D - g_mu) / g_sd, name='raw48 global scalar mean/std (600)')
results['raw48_none'] = eval_dev(R, D, name='raw48 no centering (cosine on positive pixels)')
results['raw48_perimage_then_featstd'] = eval_dev((per_image_z(R) - per_image_z(S).mean(0)) / (per_image_z(S).std(0) + 1e-6), (per_image_z(D) - per_image_z(S).mean(0)) / (per_image_z(S).std(0) + 1e-6), name='raw48 per-image z then per-feature std')
print('   per-feature sd range over 600 imgs: min %.2f median %.2f max %.2f (ratio max/min %.1f)' % (sd.min(), np.median(sd), sd.max(), sd.max() / sd.min()))
# where is the per-feature sd small? rows=frequency (top=high f)
sd_img = sd.reshape(48, 48); print('   row-mean sd (top->bottom, every 6th row):', np.round(sd_img.mean(1)[::6], 1))
# blank handling under per-image z baseline
results['raw48_perimage_z_noblank'] = eval_dev(per_image_z(R), per_image_z(D), keep=~dev_blank, name='raw48 per-image z, DEV blanks excluded')
results['raw48_perimage_z_blank_nontarget'] = eval_dev(per_image_z(R), per_image_z(D), nontarget_blank=True, name='raw48 per-image z, DEV blanks as non-targets')
results['raw48_feat_std600_noblank'] = eval_dev((R - mu) / sd, (D - mu) / sd, keep=~dev_blank, name='raw48 per-feature std, DEV blanks excluded')

# PCA whitening on the 600 (label-free), on per-image-z pixels
print('\n== PCA / whitening arms (fit on 600 unlabeled train) ==')
Z = per_image_z(S); zm = Z.mean(0); Zc = Z - zm
U, s, Vt = np.linalg.svd(Zc, full_matrices=False); eig = s ** 2 / len(Zc)
for k in (32, 64, 128, 256):
    for shrink in (0.0, 0.1, 1.0):
        lam = eig[:k] + shrink * eig.mean()
        W = Vt[:k].T / np.sqrt(lam)
        results[f'pca{k}_shrink{shrink}'] = eval_dev((per_image_z(R) - zm) @ W, (per_image_z(D) - zm) @ W, name=f'raw48 PCA-whiten k={k} shrink={shrink}')
    results[f'pca{k}_noWhiten'] = eval_dev((per_image_z(R) - zm) @ Vt[:k].T, (per_image_z(D) - zm) @ Vt[:k].T, name=f'raw48 PCA k={k} no whitening')

# resolution and pooling geometry
print('\n== Resolution / pooling geometry (per-image z) ==')
for side in (6, 12, 24, 48, 96):
    fn = lambda c, side=side: gray_px(c, side).ravel()
    results[f'raw{side}'] = eval_dev(per_image_z(views(fn, ref_crops)), per_image_z(np.stack([fn(c) for c in dev_crops])), name=f'raw {side}x{side} per-image z')
# native h5 array (no crop/pad), 140x170
nat = lambda a: a.ravel()
Rn = [];
for a in ref_raw: Rn += [nat(a), nat(a[:, ::-1])]
results['native140x170'] = eval_dev(per_image_z(np.stack(Rn)), per_image_z(np.stack([nat(a) for a in dev_raw])), name='native 140x170 per-image z')
# frequency profile (pool over time only) and time profile, from native array (rows = frequency)
for nrow in (140, 48, 24):
    fp = lambda a, nrow=nrow: cv2.resize(a, (1, nrow), interpolation=cv2.INTER_AREA).ravel()
    Rf = []
    for a in ref_raw: Rf += [fp(a), fp(a[:, ::-1])]
    results[f'freqprofile{nrow}'] = eval_dev(per_image_z(np.stack(Rf)), per_image_z(np.stack([fp(a) for a in dev_raw])), name=f'frequency profile {nrow} rows (pool over time)')
tp = lambda a: cv2.resize(a, (48, 1), interpolation=cv2.INTER_AREA).ravel()
Rt = []
for a in ref_raw: Rt += [tp(a), tp(a[:, ::-1])]
results['timeprofile48'] = eval_dev(per_image_z(np.stack(Rt)), per_image_z(np.stack([tp(a) for a in dev_raw])), name='time profile 48 cols (pool over frequency)')
# freq-row-preserving grid: 48 rows x 6 time columns
fg = lambda a: cv2.resize(a, (6, 48), interpolation=cv2.INTER_AREA).ravel()
Rg = []
for a in ref_raw: Rg += [fg(a), fg(a[:, ::-1])]
results['freqgrid48x6'] = eval_dev(per_image_z(np.stack(Rg)), per_image_z(np.stack([fg(a) for a in dev_raw])), name='48 freq rows x 6 time cols per-image z')
# max over time instead of mean (frequency profile of peaks)
fpm = lambda a: a.max(1)
Rm = []
for a in ref_raw: Rm += [fpm(a), fpm(a)]
results['freqprofile_max'] = eval_dev(per_image_z(np.stack(Rm)), per_image_z(np.stack([fpm(a) for a in dev_raw])), name='frequency profile 140 rows, MAX over time')

# ---------- CNN arms (random weights) ----------
print('\n== Random-weight CNN readouts, normalization arms ==')
model = PartDetector(ROOT + '/outputs/hypernet_v14/public_release/parts_random_v14.pt').model
@torch.inference_mode()
def feats(crops, batch=64):
    out = {}
    for s in range(0, len(crops), batch):
        x = tensor(crops[s:s + batch]); m = {}
        m['input'] = model.inputs(x); m['stem'] = model.stem(m['input']); m['fine'] = model.fine(m['stem']); m['middle'] = model.middle(m['fine']); m['coarse'] = model.coarse(m['middle'])
        rows = {}
        for n in ('stem', 'fine', 'middle', 'coarse'):
            t = m[n]; H = t.shape[2]
            rows[f'{n}_grid6'] = F.adaptive_avg_pool2d(t, 6).flatten(1)
            rows[f'{n}_grid12'] = F.adaptive_avg_pool2d(t, 12).flatten(1)
            rows[f'{n}_freqrows'] = F.adaptive_avg_pool2d(t, (min(H, 24), 1)).flatten(1)   # pool over time only
            rows[f'{n}_freqrows_t3'] = F.adaptive_avg_pool2d(t, (min(H, 24), 3)).flatten(1)
            rows[f'{n}_gap'] = F.adaptive_avg_pool2d(t, 1).flatten(1)
            rows[f'{n}_shape'] = torch.tensor([list(t.shape[1:])] * len(x))
        for k, v in rows.items(): out.setdefault(k, []).append(v.double().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
ref_crop_views = []
for c in ref_crops: ref_crop_views += [c, np.ascontiguousarray(c[:, ::-1])]
FR, FD, FS = feats(ref_crop_views), feats(dev_crops), feats(std_crops)
print('   feature map shapes:', {n: FS[f'{n}_shape'][0].tolist() for n in ('stem', 'fine', 'middle', 'coarse')})
def norm_arms(key, label):
    r, d, s = FR[key], FD[key], FS[key]
    mu, sd = s.mean(0), s.std(0) + 1e-6
    results[f'{key}_featstd'] = eval_dev((r - mu) / sd, (d - mu) / sd, name=f'{label} per-feature std (v0.2 rule)')
    results[f'{key}_meanonly'] = eval_dev(r - mu, d - mu, name=f'{label} mean-subtract only')
    results[f'{key}_global'] = eval_dev((r - s.mean()) / s.std(), (d - s.mean()) / s.std(), name=f'{label} global scalar')
    results[f'{key}_none'] = eval_dev(r, d, name=f'{label} none')
    results[f'{key}_perimage'] = eval_dev(per_image_z(r), per_image_z(d), name=f'{label} per-image z of feature vector')
    results[f'{key}_perimage_featstd'] = eval_dev((per_image_z(r) - per_image_z(s).mean(0)) / (per_image_z(s).std(0) + 1e-6), (per_image_z(d) - per_image_z(s).mean(0)) / (per_image_z(s).std(0) + 1e-6), name=f'{label} per-image z then per-feature std')
    print('   per-feature sd ratio max/min over 600: %.1f' % (sd.max() / sd.min()))
norm_arms('middle_grid6', 'random middle 6x6')
def concat_std(keys, R_, D_, S_):
    parts_r, parts_d = [], []
    for k in keys:
        mu, sd = S_[k].mean(0), S_[k].std(0) + 1e-6
        parts_r.append(unit((R_[k] - mu) / sd)); parts_d.append(unit((D_[k] - mu) / sd))
    return np.concatenate(parts_r, 1), np.concatenate(parts_d, 1)
er, ed = concat_std(['stem_grid6', 'fine_grid6', 'middle_grid6'], FR, FD, FS)
results['early_concat_featstd'] = eval_dev(er, ed, name='random early concat (v0.2 winner rule)')
print('\n== Finer grids and frequency-row pooling on random CNN (per-feature std) ==')
for key in ('middle_grid12', 'fine_grid12', 'stem_grid12', 'middle_freqrows', 'fine_freqrows', 'stem_freqrows', 'middle_freqrows_t3', 'fine_freqrows_t3', 'middle_gap'):
    mu, sd = FS[key].mean(0), FS[key].std(0) + 1e-6
    results[f'{key}_featstd'] = eval_dev((FR[key] - mu) / sd, (FD[key] - mu) / sd, name=f'random {key}')
er12, ed12 = concat_std(['stem_grid12', 'fine_grid12', 'middle_grid12'], FR, FD, FS)
results['early_concat12'] = eval_dev(er12, ed12, name='random early concat 12x12 grids')
erf, edf = concat_std(['stem_grid6', 'fine_grid6', 'middle_grid6', 'stem_freqrows', 'fine_freqrows', 'middle_freqrows'], FR, FD, FS)
results['early_concat_plus_freqrows'] = eval_dev(erf, edf, name='random early concat 6x6 + freq-row profiles')

# ---------- fusion with raw pixels ----------
print('\n== Fusion of raw pixels (per-image z 48x48) with random early concat ==')
rawR, rawD = unit(per_image_z(R)), unit(per_image_z(D))
for w in (0.5, 1.0, 2.0):
    results[f'concat_raw_w{w}'] = eval_dev(np.concatenate([w * rawR, er], 1), np.concatenate([w * rawD, ed], 1), name=f'feature concat raw(w={w}) + early concat')
S_raw = score_fn(rawR, rawD); S_cnn = score_fn(er, ed)
def rrf(S1, S2, k=60):
    out = {}
    for key in S1:
        r1 = np.empty(len(S1[key])); r1[np.argsort(-S1[key], kind='stable')] = np.arange(1, len(r1) + 1)
        r2 = np.empty(len(S2[key])); r2[np.argsort(-S2[key], kind='stable')] = np.arange(1, len(r2) + 1)
        out[key] = 1 / (k + r1) + 1 / (k + r2)
    return out
def zsum(S1, S2, w=1.0):
    return {key: w * (S1[key] - S1[key].mean()) / (S1[key].std() + 1e-9) + (S2[key] - S2[key].mean()) / (S2[key].std() + 1e-9) for key in S1}
for k in (10, 60, 200):
    results[f'rrf_k{k}'] = eval_scores(rrf(S_raw, S_cnn, k), f'RRF k={k} raw + early concat')
for w in (0.5, 1.0, 2.0):
    results[f'zsum_w{w}'] = eval_scores(zsum(S_raw, S_cnn, w), f'z-score sum raw(w={w}) + early concat')
results['maxcos'] = eval_scores({k: np.maximum(S_raw[k], S_cnn[k]) for k in S_raw}, 'max of cosines')
results['raw_only_scores'] = eval_scores(S_raw, 'raw only (check)'); results['cnn_only_scores'] = eval_scores(S_cnn, 'early concat only (check)')
# oracle-free per-reference selection: pick arm whose top-k self-consistency is higher? (skip)
# fusion of raw + best PCA-whitened + cnn
print('\n== Per-class winner table (DEV) ==')
arms = ['raw48_perimage_z', 'raw48_feat_std600', 'early_concat_featstd', 'middle_grid6_featstd', 'middle_grid6_perimage', 'rrf_k60', 'zsum_w1.0', 'concat_raw_w1.0', 'freqprofile48', 'early_concat_plus_freqrows']
print('class'.ljust(20) + ''.join(a[:14].rjust(15) for a in arms))
for c in classes:
    print(c.ljust(20) + ''.join(f'{results[a][1][c]:15.3f}' for a in arms))
json.dump({k: {'mean': v[0], 'per_class': v[1]} for k, v in results.items()}, open(sys.argv[1], 'w'), indent=1)
print('done', time.time() - t0)
