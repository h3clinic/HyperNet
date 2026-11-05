"""One-reference glitch retrieval on Gravity Spy with the frozen v0.14 CNN.

For each target glitch class, one train-split spectrogram is the reference.
Every DEV (validation) and TEST spectrogram is scored against it. DEV
negatives set a 1% false-alarm cutoff; TEST measures ranking and recall.
No weights are updated; the shared CNN never saw a spectrogram.

    python scripts/gravity_spy_benchmark.py --data ../../gravity_spy_data \
        --checkpoints ../../outputs/hypernet_v14/public_release --output benchmarks/gravity_spy_v0.1
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys, time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet import gravity_spy as gs
from hypernet.part_detector import PartDetector, tensor, shape_descriptor
from hypernet.part_encoder import crop_image

PROTOCOL = {
    'format': 'hypernet.gravity_spy_oneshot.v0.1',
    'task': 'One-reference retrieval: rank whole spectrograms by similarity to one labeled reference.',
    'dataset': {'source': gs.ZENODO_RECORD, 'license': 'CC BY 4.0', 'md5': gs.MD5},
    'duration_seconds': '1.0',
    'references_per_class': 5,
    'reference_seed': 20261102,
    'reference_split': 'train',
    'dev_split': 'validation',
    'test_split': 'test',
    'non_target_classes_kept_as_distractors': list(gs.NON_TARGET),
    'rare_class_rule': f'fewer than {gs.RARE_LIMIT} labeled examples in total',
    'false_alarm_rate_on_dev': 0.01,
    'primary_method': 'hypernet_v14',
    'primary_metric': 'TEST mean average precision, averaged over references then over target classes',
    'methods': {
        'hypernet_v14': 'Frozen v0.14 synthetic-trained CNN, its selected whitened_global channel',
        'random_v14': 'Same architecture with the published random-weight control checkpoint, same channel',
        'analytic_shape': 'Untrained 9-orientation gradient histogram at 2/4/8 grids (HyperNet shape control)',
        'raw_pixels': 'Z-scored 48x48 grayscale of the same 96px crop, cosine similarity',
    },
    'diagnostic_channels': ['hypernet_v14_global', 'hypernet_v14_parts', 'hypernet_v14_centered_parts'],
    'reference_views': 'crop and its horizontal reflection for every method (HyperNet enrollment rule)',
    'weight_updates': 0,
    'spectrogram_images_used_in_weight_training': 0,
    'input_conversion': 'HDF5 float spectrogram x255 to uint8 gray, copied to 3 channels',
}
CHECKPOINT_SHA = {'parts_trained_v14.pt': 'a64618f1d748acbb239c690abd2e5c2d26c1328162de5ac2b62a14394706013e',
                  'parts_random_v14.pt': 'fc78ae5c4833fc533d12564d6ee1c9dd57e3ee7b27fa5eea21f1ccc515016231'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pixel_vector(crop):
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    return gray / (np.linalg.norm(gray) + 1e-8)


@torch.inference_mode()
def encode_items(detector, crops, batch=64):
    out = {'global': [], 'local': []}
    for start in range(0, len(crops), batch):
        encoded = detector.model(tensor(crops[start:start + batch]))
        out['global'].append(encoded['global']); out['local'].append(encoded['local'])
    return {'global': torch.cat(out['global']), 'local': torch.cat(out['local'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output folder; earlier results are preserved')
    import pandas as pd
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)

    for name, digest in gs.MD5.items():
        h = hashlib.md5()
        with open(args.data / name, 'rb') as f:
            for block in iter(lambda: f.read(1 << 22), b''):
                h.update(block)
        if h.hexdigest() != digest:
            raise ValueError(f'{name} checksum differs from Zenodo')
    for name, digest in CHECKPOINT_SHA.items():
        if sha256(args.checkpoints / name) != digest:
            raise ValueError(f'{name} differs from the published v0.14 release')

    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv')
    classes, counts = gs.target_classes(meta['label'])
    rare = [c for c in classes if counts[c] < gs.RARE_LIMIT]
    refs = gs.choose_references(meta['label'], meta['sample_type'], classes, PROTOCOL['references_per_class'], PROTOCOL['reference_seed'])
    haystack = meta[meta['sample_type'].isin([PROTOCOL['dev_split'], PROTOCOL['test_split']])].index.to_numpy()

    # Freeze every choice before any score exists.
    args.output.mkdir(parents=True)
    frozen = {**PROTOCOL, 'target_classes': classes, 'rare_classes': rare, 'class_counts': counts,
              'reference_gravityspy_ids': {c: meta.loc[rows, 'gravityspy_id'].tolist() for c, rows in refs.items()},
              'checkpoint_sha256': CHECKPOINT_SHA, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (args.output / 'protocol.json').write_text(json.dumps(frozen, indent=2))

    rows = sorted(set(haystack) | {i for r in refs.values() for i in r})
    started = time.time()
    images = dict(zip(rows, gs.load_split_images(args.data / 'trainingsetv1d1.h5', meta.loc[rows], PROTOCOL['duration_seconds'])))
    print(f'loaded {len(images)} spectrograms in {time.time() - started:.0f}s', flush=True)

    trained = PartDetector(args.checkpoints / 'parts_trained_v14.pt')
    random = PartDetector(args.checkpoints / 'parts_random_v14.pt')
    crops = [crop_image(images[i])[0] for i in haystack]
    items = {'trained': encode_items(trained, crops), 'random': encode_items(random, crops)}
    shapes = np.stack([shape_descriptor(c) for c in crops]); pixels = np.stack([pixel_vector(c) for c in crops])
    for key in items:
        items[key]['shape'] = shapes
    print(f'encoded haystack in {time.time() - started:.0f}s', flush=True)

    labels = meta.loc[haystack, 'label'].to_numpy(); split = meta.loc[haystack, 'sample_type'].to_numpy()
    dev, test = split == PROTOCOL['dev_split'], split == PROTOCOL['test_split']
    methods = list(PROTOCOL['methods']) + PROTOCOL['diagnostic_channels']
    scores = {m: {} for m in methods}
    for name, ref_rows in refs.items():
        for r, row in enumerate(ref_rows):
            enrolled_t = trained.enroll(images[row]); enrolled_r = random.enroll(images[row])
            ch_t = trained.match(enrolled_t, items['trained']); ch_r = random.match(enrolled_r, items['random'])
            crop = crop_image(images[row])[0]
            views = np.stack([pixel_vector(crop), pixel_vector(np.ascontiguousarray(crop[:, ::-1]))])
            random_channel = 'whitened_global' if 'whitened_global' in ch_r else 'global'
            row_scores = {'hypernet_v14': ch_t['whitened_global'], 'random_v14': ch_r[random_channel],
                          'analytic_shape': ch_t['shape'], 'raw_pixels': (views @ pixels.T).max(0),
                          'hypernet_v14_global': ch_t['global'], 'hypernet_v14_parts': ch_t['parts'],
                          'hypernet_v14_centered_parts': ch_t['centered_parts']}
            for m in methods:
                scores[m][(name, r)] = np.asarray(row_scores[m], np.float64)
        print(f'scored {name} at {time.time() - started:.0f}s', flush=True)

    rate = PROTOCOL['false_alarm_rate_on_dev']; per_reference = []; summary = {}
    ids = meta.loc[haystack, 'gravityspy_id'].to_numpy()
    for m in methods:
        summary[m], records = gs.evaluate_method(scores[m], labels, dev, test, classes, rare,
                                                 PROTOCOL['references_per_class'], rate, item_ids=ids)
        for rec in records:
            per_reference.append({'method': m, 'reference_gravityspy_id': frozen['reference_gravityspy_ids'][rec['class']][rec['reference']], **rec})
    primary = PROTOCOL['primary_method']
    contrasts = {}
    for m in methods:
        if m == primary:
            continue
        diff, ci = gs.paired_class_bootstrap([summary[primary]['per_class'][c]['ap'] for c in classes],
                                             [summary[m]['per_class'][c]['ap'] for c in classes])
        contrasts[f'{primary}_minus_{m}_mean_ap'] = {'difference': diff, 'class_bootstrap_95ci': ci}
    chance = float(np.mean([summary[primary]['per_class'][c]['base_rate'] for c in classes]))
    result = {'format': PROTOCOL['format'], 'protocol_sha256': sha256(args.output / 'protocol.json'),
              'test_items': int(test.sum()), 'dev_items': int(dev.sum()), 'chance_mean_ap': chance,
              'one_shot_chance': 1 / len(classes), 'methods': summary, 'contrasts': contrasts,
              'runtime_seconds': round(time.time() - started, 1), 'torch': torch.__version__}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    with (args.output / 'per_reference.jsonl').open('w') as f:
        for rec in per_reference:
            f.write(json.dumps(rec) + '\n')
    np.savez_compressed(args.output / 'scores.npz', haystack_gravityspy_ids=meta.loc[haystack, 'gravityspy_id'].to_numpy(),
                        **{f'{m}|{c}|{r}': v.astype(np.float32) for m in methods for (c, r), v in scores[m].items()})
    print(json.dumps({m: {k: round(v, 4) for k, v in summary[m].items() if isinstance(v, float)} for m in methods}, indent=2))
    print(json.dumps(contrasts, indent=2))


if __name__ == '__main__':
    main()
