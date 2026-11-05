"""DEV-only: per-class and per-duration AP for raw pixels vs random early/full CNN mix. No TEST rows are read."""
import sys, json, numpy as np, torch, pandas as pd
from pathlib import Path
ROOT = Path('C:/Users/brita/Documents/Codex/2026-11-01/wan/public/HyperNet'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from hypernet import gravity_spy as gs
from hypernet.part_detector import PartDetector
from hypernet.part_encoder import crop_image
from gravity_spy_benchmark import pixel_vector
from gravity_spy_diagnose import layer_features, STANDARDIZE_IMAGES, STANDARDIZE_SEED
from gravity_spy_v02 import unit
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
DATA = Path('C:/Users/brita/Documents/Codex/2026-11-01/wan/gravity_spy_data')
v01 = json.loads((ROOT/'benchmarks/gravity_spy_v0.1/protocol.json').read_text())
meta = pd.read_csv(DATA/'trainingset_v1d1_metadata.csv'); by_id = {g:i for i,g in enumerate(meta['gravityspy_id'])}
refs = {c:[by_id[g] for g in ids] for c,ids in v01['reference_gravityspy_ids'].items()}; ref_rows = {i for r in refs.values() for i in r}
classes, rare, k = v01['target_classes'], v01['rare_classes'], v01['references_per_class']
pool = np.array([i for i in meta.index[meta['sample_type']=='train'] if i not in ref_rows])
std_rows = np.sort(np.random.default_rng(STANDARDIZE_SEED).choice(pool, STANDARDIZE_IMAGES, replace=False))
dev_rows = meta.index[meta['sample_type']=='validation'].to_numpy(); dev_labels = meta.loc[dev_rows,'label'].to_numpy()
model = PartDetector(ROOT.parents[1]/'outputs/hypernet_v14/checkpoints/random_v3_1403_statistics.pt').model
GRID = ('stem_grid6','fine_grid6','middle_grid6','coarse_grid6')
out = {}
for dur in gs.DURATIONS:
    crops = {}
    rows = sorted(set(dev_rows)|ref_rows|set(std_rows))
    for i, im in zip(rows, gs.load_split_images(DATA/'trainingsetv1d1.h5', meta.loc[rows], dur)): crops[i] = crop_image(im)[0]
    std = [crops[i] for i in std_rows]; dev = [crops[i] for i in dev_rows]
    ref_list, views = [], {}
    for c, rr in refs.items():
        for r, i in enumerate(rr): views[(c,r)] = [len(ref_list), len(ref_list)+1]; ref_list += [crops[i], np.ascontiguousarray(crops[i][:, ::-1])]
    fs, fr, fd = layer_features(model, std), layer_features(model, ref_list), layer_features(model, dev)
    sc = lambda key, f: unit((f[key]-fs[key].mean(0))/(fs[key].std(0)+1e-6))
    feats = {'raw_pixels': (np.stack([pixel_vector(c) for c in ref_list]), np.stack([pixel_vector(c) for c in dev])),
             'random_early3': (unit(np.concatenate([sc(g,fr) for g in GRID[:3]],1)), unit(np.concatenate([sc(g,fd) for g in GRID[:3]],1))),
             'random_full4': (unit(np.concatenate([sc(g,fr) for g in GRID],1)), unit(np.concatenate([sc(g,fd) for g in GRID],1)))}
    out[dur] = {}
    for name,(a,b) in feats.items():
        scores = {key: (a[rows_][...]@b.T).max(0) for key, rows_ in views.items()}
        per = {c: float(np.mean([gs.average_precision(scores[(c,r)], dev_labels==c) for r in range(k)])) for c in classes}
        out[dur][name] = per
    # also: no-flip reference view, raw pixels, to test whether the horizontal flip view hurts
    a,b = feats['raw_pixels']; scores = {key: a[rows_[0]]@b.T for key, rows_ in views.items()}
    out[dur]['raw_pixels_noflip'] = {c: float(np.mean([gs.average_precision(scores[(c,r)], dev_labels==c) for r in range(k)])) for c in classes}
    print('done', dur, flush=True)
# multi-duration fusion for raw pixels and random_early3: average of per-duration scores — needs scores, recompute cheaply by averaging APs is wrong; so report per-duration only here.
print(f"{'class':22s}{'n_dev':>6s}" + ''.join(f"{'px'+d:>8s}{'nf'+d:>8s}{'r3'+d:>8s}{'r4'+d:>8s}" for d in gs.DURATIONS))
for c in classes:
    print(f"{c:22s}{int((dev_labels==c).sum()):6d}" + ''.join(f"{out[d]['raw_pixels'][c]:8.3f}{out[d]['raw_pixels_noflip'][c]:8.3f}{out[d]['random_early3'][c]:8.3f}{out[d]['random_full4'][c]:8.3f}" for d in gs.DURATIONS), '*' if c in rare else '')
print('MEAN'.ljust(28) + ''.join(f"{np.mean(list(out[d]['raw_pixels'].values())):8.3f}{np.mean(list(out[d]['raw_pixels_noflip'].values())):8.3f}{np.mean(list(out[d]['random_early3'].values())):8.3f}{np.mean(list(out[d]['random_full4'].values())):8.3f}" for d in gs.DURATIONS))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
