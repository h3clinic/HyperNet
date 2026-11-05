"""Where does synthetic training lose Gravity Spy glitch information? DEV split only.

Uses the references frozen by gravity_spy_benchmark.py and only the DEV
(validation) haystack, so TEST stays untouched for any later model. Three checks:

1. Layer probe: one-reference AP from each stage of the CNN, trained vs its own
   random initialization (same seed, same architecture).
2. Training trajectory: global-embedding and coarse-layer AP at epochs 0/1/4/8/12
   of the three v0.14 development runs.
3. Shift sensitivity: cosine similarity between a spectrogram and the same
   spectrogram moved up (higher frequency) or right (later time) by k pixels.

Per-dimension standardization for every feature uses unlabeled train-split
spectrograms that are not references, never DEV or TEST.

    python scripts/gravity_spy_diagnose.py --data ../../gravity_spy_data \
        --benchmark benchmarks/gravity_spy_v0.1 --runs ../../outputs/hypernet_v14/checkpoints \
        --output benchmarks/gravity_spy_v0.1/diagnosis
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from hypernet import gravity_spy as gs
from hypernet.part_detector import PartDetector, tensor
from hypernet.part_encoder import crop_image
from gravity_spy_benchmark import pixel_vector

STANDARDIZE_IMAGES = 600
STANDARDIZE_SEED = 7
SHIFTS = (2, 4, 8, 16)
RUNS = {'run01': ('parts_run01', 'random1403.pt'), 'run02': ('parts_run02', 'random_v2_1403.pt'),
        'run03': ('parts_run03', 'random_v3_1403.pt')}
EPOCHS = (1, 4, 8, 12)


@torch.inference_mode()
def layer_features(model, crops, batch=64):
    """Each stage pooled two ways: the 1+2+3 pyramid the embedding sees, and a 6x6 grid."""
    stages = {}
    for start in range(0, len(crops), batch):
        x = tensor(crops[start:start + batch])
        maps = {'input': model.inputs(x)}
        maps['stem'] = model.stem(maps['input']); maps['fine'] = model.fine(maps['stem'])
        maps['middle'] = model.middle(maps['fine']); maps['coarse'] = model.coarse(maps['middle'])
        out = model(x)
        rows = {}
        for name, m in maps.items():
            rows[f'{name}_pyramid'] = torch.cat([F.adaptive_avg_pool2d(m, n).flatten(1) for n in (1, 2, 3)], 1)
            rows[f'{name}_grid6'] = F.adaptive_avg_pool2d(m, 6).flatten(1)
        rows['embedding_raw'] = out['raw'] if 'raw' in out else model.embedding(rows['coarse_pyramid'])
        rows['global'] = out['global']
        for k, v in rows.items():
            stages.setdefault(k, []).append(v.double().numpy())
    return {k: np.concatenate(v) for k, v in stages.items()}


def standardize(train, *others):
    mean, std = train.mean(0), train.std(0) + 1e-6
    return [((o - mean) / std) / (np.linalg.norm((o - mean) / std, axis=1, keepdims=True) + 1e-12) for o in others]


def mean_ap(ref_feats, ref_views, dev_feats, dev_labels, refs):
    """ref_feats rows are [ref0 view0, ref0 view1, ref1 view0, ...]; returns mean AP over classes."""
    per_class = {}
    for c, rows in refs.items():
        aps = []
        for r in range(len(rows)):
            k = ref_views[(c, r)]
            s = (ref_feats[k] @ dev_feats.T).max(0)
            aps.append(gs.average_precision(s, dev_labels == c))
        per_class[c] = float(np.mean(aps))
    return float(np.mean(list(per_class.values()))), per_class


def shifted(crop, dy, dx):
    """Move content up by dy and right by dx, filling with the crop's median border value."""
    out = np.empty_like(crop); out[:] = np.median(np.concatenate([crop[0], crop[-1]]), axis=0).astype(np.uint8)
    h, w = crop.shape[:2]
    out[0:h - dy, dx:w] = crop[dy:h, 0:w - dx]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output folder')
    import pandas as pd
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True); started = time.time()
    protocol = json.loads((args.benchmark / 'protocol.json').read_text())
    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv')
    by_id = {g: i for i, g in enumerate(meta['gravityspy_id'])}
    refs = {c: [by_id[g] for g in ids] for c, ids in protocol['reference_gravityspy_ids'].items()}
    ref_rows = {i for rows in refs.values() for i in rows}
    dev_rows = meta.index[meta['sample_type'] == protocol['dev_split']].to_numpy()
    pool = np.array([i for i in meta.index[meta['sample_type'] == 'train'] if i not in ref_rows])
    std_rows = np.sort(np.random.default_rng(STANDARDIZE_SEED).choice(pool, STANDARDIZE_IMAGES, replace=False))
    rows = sorted(set(dev_rows) | ref_rows | set(std_rows))
    images = dict(zip(rows, gs.load_split_images(args.data / 'trainingsetv1d1.h5', meta.loc[rows], protocol['duration_seconds'])))
    crop = {i: crop_image(images[i])[0] for i in rows}
    ref_list, ref_views = [], {}
    for c, rr in refs.items():
        for r, i in enumerate(rr):
            ref_views[(c, r)] = [len(ref_list), len(ref_list) + 1]
            ref_list += [crop[i], np.ascontiguousarray(crop[i][:, ::-1])]
    dev_crops = [crop[i] for i in dev_rows]; std_crops = [crop[i] for i in std_rows]
    dev_labels = meta.loc[dev_rows, 'label'].to_numpy()

    def probe(model):
        f_ref, f_dev, f_std = layer_features(model, ref_list), layer_features(model, dev_crops), layer_features(model, std_crops)
        out = {}
        for k in f_dev:
            a, b = standardize(f_std[k], f_ref[k], f_dev[k])
            out[k] = mean_ap(a, ref_views, b, dev_labels, refs)
        return out

    px = lambda cs: np.stack([pixel_vector(c) for c in cs])
    a, b = standardize(px(std_crops), px(ref_list), px(dev_crops))
    report = {'format': 'hypernet.gravity_spy_diagnosis.v0.1', 'split': 'DEV (validation) only',
              'dev_items': len(dev_rows), 'standardization_images': STANDARDIZE_IMAGES,
              'chance_mean_ap': float(np.mean([(dev_labels == c).mean() for c in refs])),
              'raw_pixels_standardized_mean_ap': mean_ap(a, ref_views, b, dev_labels, refs)[0]}

    # 1. Layer probe on the frozen model and its own random start.
    frozen = PartDetector(args.runs / 'parts_run03' / 'epoch_12_statistics.pt').model
    start = PartDetector(args.runs / RUNS['run03'][1]).model
    layers = {'trained_run03_epoch12': probe(frozen), 'random_run03_start': probe(start)}
    report['layer_probe'] = {name: {k: v[0] for k, v in res.items()} for name, res in layers.items()}
    report['layer_probe_per_class'] = {name: {k: v[1] for k, v in res.items() if k in ('coarse_grid6', 'global')} for name, res in layers.items()}
    print(f'layer probe done {time.time() - started:.0f}s', flush=True)

    # 2. Trajectory over training for each development run.
    trajectory = {}
    for run, (folder, random_file) in RUNS.items():
        points = {0: args.runs / random_file, **{e: args.runs / folder / f'epoch_{e:02d}.pt' for e in EPOCHS}}
        trajectory[run] = {}
        for epoch, path in points.items():
            res = probe(PartDetector(path).model)
            trajectory[run][epoch] = {k: res[k][0] for k in ('input_grid6', 'fine_grid6', 'middle_grid6', 'coarse_grid6', 'coarse_pyramid', 'global')}
        print(f'{run} trajectory done {time.time() - started:.0f}s', flush=True)
    report['trajectory'] = trajectory

    # 3. Shift sensitivity on DEV crops (no labels used).
    sample = dev_crops[::4]
    shift = {}
    for name, model in (('trained_run03_epoch12', frozen), ('random_run03_start', start)):
        base = layer_features(model, sample); shift[name] = {}
        for axis in ('frequency_up', 'time_right'):
            for k in SHIFTS:
                moved = layer_features(model, [shifted(c, k, 0) if axis == 'frequency_up' else shifted(c, 0, k) for c in sample])
                shift[name][f'{axis}_{k}px'] = {key: float(np.mean(np.sum(F.normalize(torch.from_numpy(base[key]), dim=1).numpy()
                                                                         * F.normalize(torch.from_numpy(moved[key]), dim=1).numpy(), 1)))
                                                for key in ('coarse_grid6', 'global')}
    p0 = px(sample)
    shift['raw_pixels'] = {f'{axis}_{k}px': float(np.mean(np.sum(p0 * px([shifted(c, k, 0) if axis == 'frequency_up' else shifted(c, 0, k) for c in sample]), 1)))
                           for axis in ('frequency_up', 'time_right') for k in SHIFTS}
    report['shift_cosine_similarity'] = shift
    report['shift_items'] = len(sample)
    report['runtime_seconds'] = round(time.time() - started, 1)
    args.output.mkdir(parents=True)
    (args.output / 'diagnosis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('chance_mean_ap', 'raw_pixels_standardized_mean_ap', 'layer_probe', 'trajectory', 'shift_cosine_similarity')}, indent=2))


if __name__ == '__main__':
    main()
