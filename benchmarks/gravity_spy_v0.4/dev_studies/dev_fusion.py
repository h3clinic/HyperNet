"""DEV-only fusion study. Z-normalization stats per reference come from 600 unlabeled train images, never DEV."""
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
raw, z = {}, {}   # raw[(dur,method)][(c,r)] -> dev scores ; z = normalized
for dur in gs.DURATIONS:
    rows = sorted(set(dev_rows)|ref_rows|set(std_rows)); crops = {}
    for i, im in zip(rows, gs.load_split_images(DATA/'trainingsetv1d1.h5', meta.loc[rows], dur)): crops[i] = crop_image(im)[0]
    std = [crops[i] for i in std_rows]; dev = [crops[i] for i in dev_rows]
    ref_list, views = [], {}
    for c, rr in refs.items():
        for r, i in enumerate(rr): views[(c,r)] = [len(ref_list), len(ref_list)+1]; ref_list += [crops[i], np.ascontiguousarray(crops[i][:, ::-1])]
    fs, fr, fd = layer_features(model, std), layer_features(model, ref_list), layer_features(model, dev)
    sc = lambda key, f: unit((f[key]-fs[key].mean(0))/(fs[key].std(0)+1e-6))
    px = lambda cs: np.stack([pixel_vector(c) for c in cs])
    feats = {'px': (px(ref_list), px(dev), px(std)),
             'cnn4': (unit(np.concatenate([sc(g,fr) for g in GRID],1)), unit(np.concatenate([sc(g,fd) for g in GRID],1)), unit(np.concatenate([sc(g,fs) for g in GRID],1)))}
    for m,(a,b,s) in feats.items():
        raw[(dur,m)] = {key: (a[rows_]@b.T).max(0) for key, rows_ in views.items()}
        bg = {key: (a[rows_]@s.T).max(0) for key, rows_ in views.items()}
        z[(dur,m)] = {key: (raw[(dur,m)][key]-bg[key].mean())/(bg[key].std()+1e-6) for key in views}
    print('done', dur, flush=True)
def map_of(scores):
    per = {c: float(np.mean([gs.average_precision(scores[(c,r)], dev_labels==c) for r in range(k)])) for c in classes}
    return float(np.mean(list(per.values()))), float(np.mean([per[c] for c in rare])), per
def fuse(keys, src):
    return {key: np.mean([src[kk][key] for kk in keys], 0) for key in src[keys[0]]}
rows = []
for m in ('px','cnn4'):
    for dur in gs.DURATIONS: rows.append((f'{m} {dur}s', map_of(raw[(dur,m)])))
    rows.append((f'{m} 4-duration z-fusion', map_of(fuse([(d,m) for d in gs.DURATIONS], z))))
    rows.append((f'{m} 0.5+1.0 z-fusion', map_of(fuse([('0.5',m),('1.0',m)], z))))
rows.append(('px+cnn4 0.5s z-fusion', map_of(fuse([('0.5','px'),('0.5','cnn4')], z))))
rows.append(('px+cnn4 1.0s z-fusion', map_of(fuse([('1.0','px'),('1.0','cnn4')], z))))
rows.append(('px+cnn4 all 4 durations z-fusion', map_of(fuse([(d,m) for d in gs.DURATIONS for m in ('px','cnn4')], z))))
rows.append(('px+cnn4 0.5+1.0 z-fusion', map_of(fuse([(d,m) for d in ('0.5','1.0') for m in ('px','cnn4')], z))))
# z-normalization alone (single method, single duration) to see the effect of background normalization on AP
rows.append(('cnn4 1.0s z-normalized only', map_of(z[('1.0','cnn4')])))
rows.append(('px 1.0s z-normalized only', map_of(z[('1.0','px')])))
print(f"{'method':38s}{'meanAP':>8s}{'rareAP':>8s}")
for name,(m_,r_,_) in rows: print(f'{name:38s}{m_:8.3f}{r_:8.3f}')
best = max(rows, key=lambda x: x[1][0]); print('\nbest:', best[0]); print({c: round(v,3) for c,v in best[1][2].items()})
json.dump({n:{'mean_ap':a,'rare_ap':b,'per_class':c} for n,(a,b,c) in rows}, open(sys.argv[1],'w'), indent=1)
