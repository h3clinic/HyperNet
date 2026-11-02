"""Fixed-budget native128 versus32px information-loss pilot. No natural photos."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from hypernet.surface_encoder import (SurfaceEncoder, FORMAT, input_for_arm,
    geometry_targets, surface_spectrum, surface_loss)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def load_data(path):
    manifest = json.loads((path/'manifest.json').read_text())
    records = manifest['records']
    images, masks, normals, depths = [], [], [], []
    for row in records:
        for path_key, hash_key in (('rgb_path', 'rgb_sha256'), ('target_path', 'target_sha256'), ('metadata_path', 'metadata_sha256')):
            if digest(path/row[path_key]) != row[hash_key]:
                raise ValueError('Dataset file hash mismatch: '+row[path_key])
        rgb = np.array(Image.open(path/row['rgb_path']).convert('RGB'))
        if rgb.shape != (128, 128, 3):
            raise ValueError('Dataset is not native128')
        images.append(rgb)
        with np.load(path/row['target_path'], allow_pickle=False) as t:
            masks.append(t['mask'])
            normals.append(t['normal'])
            depths.append(t['depth'])
    rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()/255
    mask = torch.from_numpy(np.stack(masks)).float()
    normal = torch.from_numpy(np.stack(normals)).permute(0, 3, 1, 2).float()
    depth = torch.from_numpy(np.stack(depths)).float()
    targets = [[] for _ in range(4)]
    for start in range(0, len(rgb), 32):
        s = slice(start, start+32)
        geometry = geometry_targets(mask[s], normal[s], depth[s])
        for dest, value in zip(targets, (surface_spectrum(rgb[s], mask[s]), *geometry)):
            dest.append(value)
    return records, rgb, tuple(torch.cat(values) for values in targets)


def batch_plan(records, steps, seed):
    rng = np.random.default_rng(seed)
    objects = {}
    for i, row in enumerate(records):
        if row['split'] == 'train':
            objects.setdefault((row['family_index'], row['object_id']), {}).setdefault(row['view_index'], []).append(i)
    by_family = {f: [key for key in objects if key[0] == f] for f in range(8)}
    plan = []
    for _ in range(steps):
        indices = []
        # Four families x two objects gives genuine within-family negatives.
        for family in rng.choice(8, 4, replace=False):
            choices = rng.choice(len(by_family[int(family)]), 2, replace=False)
            for choice in choices:
                views = objects[by_family[int(family)][choice]]
                indices += [int(rng.choice(views[0])), int(rng.choice(views[1]))]
        plan.append(indices)
    return plan


def accuracy(scores, correct):
    # Fractional tie credit rather than arbitrary index-order wins.
    maxima = scores.max(1, keepdims=True)
    tied = np.isclose(scores, maxima, atol=1e-7, rtol=0)
    return float((tied[np.arange(len(scores)), correct]/tied.sum(1)).mean())


def representation_metrics(features, records):
    objects = {}
    for i, row in enumerate(records):
        objects.setdefault((row['family_index'], row['object_id']), {})[(row['view_index'], round(row['surface_strength'], 2))] = i
    keys = sorted(objects)
    support = [objects[key][(0, 0.)] for key in keys]
    queries = [objects[key][(1, strength)] for key in keys for strength in (0., .03, .07)]
    instance = accuracy(features[queries]@features[support].T, np.repeat(np.arange(len(keys)), 3))
    by_family = {f: [key for key in keys if key[0] == f] for f in range(8)}
    episodes = []
    for slot in range(8):
        enrollment = [objects[by_family[f][slot]][(0, 0.)] for f in range(8)]
        q = [objects[key][(1, strength)] for f in range(8)
            for key in by_family[f] if key != by_family[f][slot] for strength in (0., .03, .07)]
        labels = np.array([records[i]['family_index'] for i in q])
        episodes.append(accuracy(features[q]@features[enrollment].T, labels))
    # Balance both reference identities; only changing negative difficulty cannot
    # produce a directional clean-versus-rough preference win.
    detail = []
    for key in keys:
        for strength, opposite in ((0., .07), (.07, 0.)):
            r, p, n = [objects[key][pair] for pair in ((0, strength), (1, strength), (1, opposite))]
            margin = float(features[r]@features[p]-features[r]@features[n])
            detail.append(.5 if abs(margin) <= 1e-7 else float(margin > 0))
    return dict(one_support_per_family_accuracy=float(np.mean(episodes)),
        family_episode_accuracies=episodes, family_decisions=8*8*7*3,
        cross_view_instance_retrieval=instance, instance_decisions=len(queries),
        balanced_cross_view_surface_match=float(np.mean(detail)), surface_triplets=len(detail))


@torch.inference_mode()
def evaluate(model, arm, records, rgb, targets):
    indices = [i for i, row in enumerate(records) if row['split'] == 'heldout']
    held_records = [records[i] for i in indices]
    collected = {name: [] for name in ('shape', 'detail', 'embedding', 'family_logits', 'spectrum')}
    for start in range(0, len(indices), 16):
        ids = indices[start:start+16]
        out = model(input_for_arm(rgb[ids], arm))
        for name in collected:
            collected[name].append(out[name].cpu())
    values = {name: torch.cat(v) for name, v in collected.items()}
    labels = np.array([row['family_index'] for row in held_records])
    results = {name: representation_metrics(F.normalize(values[name], dim=1).numpy(), held_records)
        for name in ('embedding', 'shape', 'detail')}
    results['trained_family_classifier_accuracy'] = accuracy(values['family_logits'].numpy(), labels)
    results['spectrum_cosine'] = float(F.cosine_similarity(values['spectrum'], targets[0][indices], dim=1).mean())
    return results, {name: value.numpy() for name, value in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/surfaces_v10')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.steps < 1:
        raise ValueError('steps must be positive')
    freeze_path = ROOT/'experiments/surface_training_freeze.json'
    if freeze_path.exists():
        raise FileExistsError('Preserve the existing frozen experiment; use a new version')
    (ROOT/'results').mkdir(parents=True, exist_ok=True)
    (ROOT/'checkpoints').mkdir(parents=True, exist_ok=True)
    sources = [ROOT/'hypernet/surface_encoder.py', ROOT/'hypernet/training_v3.py',
        ROOT/'hypernet/mathematical_surfaces.py', Path(__file__)]
    data_manifest = json.loads((args.data/'manifest.json').read_text())
    if data_manifest.get('completed') is not True:
        raise ValueError('Dataset generation must finish before training')
    records = data_manifest['records']
    train_ids = {r['object_id'] for r in records if r['split'] == 'train'}
    held_ids = {r['object_id'] for r in records if r['split'] == 'heldout'}
    if train_ids & held_ids or len(train_ids) != 256 or len(held_ids) != 64 or len(records) != 1920:
        raise ValueError('Unexpected or overlapping data split')
    plan = batch_plan(records, args.steps, 731)
    plan_path = ROOT/'experiments/surface_batch_plan.json'
    write_json(plan_path, plan)
    torch.manual_seed(731)
    prototype = SurfaceEncoder()
    initial = {key: value.clone() for key, value in prototype.state_dict().items()}
    freeze = dict(format=FORMAT, sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        manifest_sha256=digest(args.data/'manifest.json'), batch_plan_sha256=digest(plan_path),
        initial_state_sha256=state_digest(prototype), parameters=sum(p.numel() for p in prototype.parameters()),
        native_resolution=128, arms=['native128', 'lowpass32'], steps_per_arm=args.steps,
        batch_images=16, image_exposures_per_arm=16*args.steps, unique_training_frames=1536,
        training_object_programs=256, heldout_object_programs=64, families=8,
        natural_training_images=0, external_pretrained_weights=False, seed=731,
        runtime={'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'threads': args.threads, 'device': 'cpu'},
        optimizer={'name': 'AdamW', 'lr': .001, 'weight_decay': .0001},
        loss_weights={'family': 1., 'contrastive': .3, 'spectrum': .5, 'mask': .2, 'normal': .2, 'depth': .2},
        primary_metric='embedding.one_support_per_family_accuracy',
        evaluation='8-way family nearest support: 8 enrollment rotations; different heldout objects in queries',
        surface_metric='balanced cross-view clean-versus-.07 matching; 128 triples',
        selection='fixed final step; no heldout-based model selection or early stopping',
        cross_resolution_diagnostic='Also evaluate each final checkpoint with the other input arm; no retraining',
        limitation='Synthetic heldout objects from known generated families; no natural-category generalization claim')
    write_json(freeze_path, freeze)
    print('Frozen. Loading native images and mathematical targets.', flush=True)
    records, rgb, targets = load_data(args.data)
    labels = torch.tensor([r['family_index'] for r in records], dtype=torch.long)
    results = dict(freeze_sha256=digest(freeze_path), arms={}, completed=False)
    for arm in freeze['arms']:
        model = SurfaceEncoder()
        model.load_state_dict(initial)
        model.eval()
        before, before_arrays = evaluate(model, arm, records, rgb, targets)
        np.savez_compressed(ROOT/'results'/f'heldout_random_{arm}.npz', **before_arrays)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        model.train()
        history, start = [], time.perf_counter()
        for step, ids in enumerate(plan):
            optimizer.zero_grad(set_to_none=True)
            out = model(input_for_arm(rgb[ids], arm))
            loss, terms = surface_loss(out, labels[ids], *(target[ids] for target in targets))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step == 0 or (step+1) % 25 == 0:
                row = dict(step=step+1, total=float(loss.detach()),
                    **{name: float(value.detach()) for name, value in terms.items()})
                history.append(row)
                print(arm, json.dumps(row), flush=True)
        training_seconds = time.perf_counter()-start
        model.eval()
        for p in model.parameters():
            p.grad = None
        before_hash = state_digest(model)
        after, arrays = evaluate(model, arm, records, rgb, targets)
        alternate_arm = 'lowpass32' if arm == 'native128' else 'native128'
        cross_resolution, _ = evaluate(model, alternate_arm, records, rgb, targets)
        assert before_hash == state_digest(model) and all(p.grad is None for p in model.parameters())
        checkpoint = ROOT/'checkpoints'/f'surface_{arm}.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(format=FORMAT, state_dict=model.state_dict(), arm=arm,
            freeze_sha256=results['freeze_sha256'], natural_training_images=0,
            external_pretrained_weights=False), checkpoint)
        np.savez_compressed(ROOT/'results'/f'heldout_{arm}.npz', **arrays)
        results['arms'][arm] = dict(random_initialization=before, trained=after,
            alternate_input_arm=alternate_arm, cross_resolution_evaluation=cross_resolution,
            history=history, training_seconds=training_seconds, checkpoint_sha256=digest(checkpoint),
            final_state_sha256=before_hash, evaluation_read_only=True,
            actual_distinct_frames_seen=len({i for ids in plan for i in ids}))
        write_json(ROOT/'results/surface_results.json', results)
        print(arm, 'heldout', json.dumps(after), flush=True)
    results['completed'] = True
    results['sources_unchanged'] = all(digest(ROOT/p) == value for p, value in freeze['sources'].items())
    if not results['sources_unchanged']:
        raise RuntimeError('Frozen source changed')
    write_json(ROOT/'results/surface_results.json', results)
    print('Surface pilot complete.', flush=True)


if __name__ == '__main__':
    main()
