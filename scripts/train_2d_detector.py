"""Train and evaluate an exemplar-conditioned2D detector on disjoint grammars."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from hypernet.one_shot_detector import (OneReferenceDetector, FORMAT, detect,
    expand_box, box_iou, foreground_loss, balanced_match_loss, reference_views)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class FrameStore:
    """Cache uint8 training pixels, never a global statistic fitted to heldout data."""
    def __init__(self, data):
        self.data = Path(data)
        self.images = {}
        self.unions = {}

    def rgb(self, record):
        path = record['rgb_path']
        if path not in self.images:
            if sha(self.data/path) != record['rgb_sha256']:
                raise ValueError('RGB hash changed: '+path)
            array = np.array(Image.open(self.data/path).convert('RGB'))
            self.images[path] = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return self.images[path]

    def union(self, record):
        # Counterbalanced enrollment deliberately reuses each query scene.
        key = record['rgb_path']
        if key not in self.unions:
            if sha(self.data/record['target_path']) != record['target_sha256']:
                raise ValueError('Target hash changed')
            with np.load(self.data/record['target_path'], allow_pickle=False) as f:
                self.unions[key] = torch.from_numpy(f['union_mask'].copy())
        return self.unions[key]


def training_plan(episodes, epochs=1):
    train = [i for i, row in enumerate(episodes) if row['split'] == 'train']
    rng = np.random.default_rng(111)
    return [list(map(int, ids)) for _ in range(epochs)
        for ids in np.array_split(rng.permutation(train), len(train)//4)]


def proposal_targets(row, mask, rng):
    boxes, labels, include_coarse, candidate_index = [], [], [], None
    for item in row['all_boxes']:
        box = item['bbox_xyxy']
        candidate = item['role'] == 'candidate'
        if candidate:
            if candidate_index is not None:
                raise ValueError('A scene must have exactly one controlled candidate')
            candidate_index = len(boxes)
        boxes.append(expand_box(box, 1.12, 256, 256))
        labels.append(float(candidate and row['present']))
        include_coarse.append(not (candidate and not row['present']))
    if candidate_index is None:
        raise ValueError('Controlled candidate absent from training annotations')
    # These background training examples use annotations; detect() cannot access
    # this sampler or any query annotation at inference.
    for _ in range(4):
        for attempt in range(80):
            w, h = int(rng.integers(24, 81)), int(rng.integers(24, 81))
            x, y = int(rng.integers(0, 257-w)), int(rng.integers(0, 257-h))
            if float(mask[y:y+h, x:x+w].float().mean()) < .02:
                boxes.append([x, y, x+w, y+h]); labels.append(0.); include_coarse.append(True)
                break
    return boxes, labels, include_coarse, candidate_index


def batch_loss(model, episodes, store, rng, micro_aware, proposal_digest):
    references = torch.stack([store.rgb(e['support']) for e in episodes]).float()/255
    rows = [q for e in episodes for q in e['queries']]
    if any([bool(q['present']) for q in e['queries']] != [True, False] for e in episodes):
        raise ValueError('Training expects positive then paired-negative scene')
    scenes = torch.stack([store.rgb(q) for q in rows]).float()/255
    masks = torch.stack([store.union(q) for q in rows])
    views, view_boxes = reference_views(references, [e['support']['bbox_xyxy'] for e in episodes])
    r = model.encode_scene(views)
    q = model.encode_scene(scenes)
    view_descriptors = [model.describe(r, i, [expand_box(box, 1.12, 96, 96)])[0]
        for i, box in enumerate(view_boxes)]
    references = torch.stack(view_descriptors).reshape(len(episodes), 4, -1)
    logits, labels, includes, candidate_scores = [], [], [], []
    for i, row in enumerate(rows):
        boxes, y, coarse_include, candidate_index = proposal_targets(row, masks[i], rng)
        proposal_digest.update(json.dumps([row['id'], boxes, y, coarse_include], separators=(',', ':')).encode())
        descriptors = model.describe(q, i, boxes)
        scores = model.similarities(references[i//2], descriptors)
        candidate_scores.append(scores[candidate_index])
        logits.append(model.match_logits(references[i//2], descriptors))
        labels.extend(y)
        includes.extend([True]*len(y) if micro_aware else coarse_include)
    match = balanced_match_loss(torch.cat(logits), labels, includes)
    objectness = foreground_loss(q['foreground_logits'], masks)
    scores = torch.stack(candidate_scores).reshape(-1, 2)
    margin = (.08+scores[:, 1]-scores[:, 0]).clamp_min(0).mean()
    total = match+.4*objectness+(0.5*margin if micro_aware else 0.)
    return total, dict(match=match, objectness=objectness, paired_micro_margin=margin)


def cutoff_from_dev(rows, max_fpr=.05):
    negatives = sorted([r['score'] for r in rows if not r['present']], reverse=True)
    if not negatives:
        raise ValueError('Calibration requires DEV negatives')
    allowed = math.floor(max_fpr*len(negatives))
    return float(np.nextafter(float(negatives[allowed]), math.inf))


def summarize(rows, cutoff):
    positives = [r for r in rows if r['present']]
    negatives = [r for r in rows if not r['present']]
    tp = sum(r['score'] >= cutoff for r in positives)
    fp = sum(r['score'] >= cutoff for r in negatives)
    localized = sum(r['score'] >= cutoff and r['iou'] >= .5 for r in positives)
    proposal_recall = sum(r['proposal_best_iou'] >= .5 for r in positives)/len(positives)
    auc = float(np.mean([[float(p['score'] > n['score'])+.5*float(p['score'] == n['score'])
        for n in negatives] for p in positives]))
    # Group tied scores, preventing arbitrary row-order gains. One predicted box
    # per scene; this is not the full COCO multiple-detection/multiple-IoU metric.
    ranked = sorted(rows, key=lambda r:r['score'], reverse=True)
    cumulative_tp, cumulative_predictions, ap, cursor = 0, 0, 0., 0
    while cursor < len(ranked):
        end = cursor+1
        while end < len(ranked) and ranked[end]['score'] == ranked[cursor]['score']:
            end += 1
        group_tp = sum(r['present'] and r['iou'] >= .5 for r in ranked[cursor:end])
        cumulative_tp += group_tp; cumulative_predictions += end-cursor
        ap += cumulative_tp/cumulative_predictions*group_tp/len(positives)
        cursor = end
    paired = {}
    for row in rows:
        paired.setdefault(row['episode_id'], {})[bool(row['present'])] = row['score']
    pair_accuracy = np.mean([float(v[True] > v[False])+.5*float(v[True] == v[False]) for v in paired.values()])
    return dict(positive_scenes=len(positives), negative_scenes=len(negatives), threshold=cutoff,
        accepted_positive=tp, false_positive=fp, localized_true_positive=localized,
        recall=tp/len(positives), false_positive_rate=fp/len(negatives),
        localized_recall_at_iou50=localized/len(positives), proposal_recall_at_iou50=proposal_recall,
        auroc=auc, single_box_ap50=float(ap), paired_scene_ranking_accuracy=float(pair_accuracy),
        balanced_accuracy=.5*(tp/len(positives)+1-fp/len(negatives)))


@torch.inference_mode()
def evaluate(model, episodes, store):
    rows = []
    before = state_hash(model)
    for episode in episodes:
        if episode['split'] not in ('dev', 'test'):
            continue
        ref = store.rgb(episode['support'])[None].float()/255
        for row in episode['queries']:
            result = detect(model, ref, store.rgb(row)[None].float()/255, episode['support']['bbox_xyxy'])
            # Only now, after inference, read query annotations for scoring.
            target = row['target_bbox_xyxy'] if row['present'] else None
            iou = box_iou(result['bbox_xyxy'], target) if target and result['bbox_xyxy'] else 0.
            proposal_iou = max([box_iou(box, target) for box in result['proposals']], default=0.) if target else 0.
            rows.append(dict(id=row['id'], episode_id=episode['episode_id'], split=episode['split'],
                family_id=episode['family_id'], enrolled_variant=episode['enrolled_variant'],
                present=bool(row['present']), score=result['score'], bbox_xyxy=result['bbox_xyxy'],
                proposals=result['proposals'], proposal_scores=result['scores'], iou=iou,
                proposal_best_iou=proposal_iou, support_path=episode['support']['rgb_path'], query_path=row['rgb_path']))
    if state_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Inference changed weights or created gradients')
    dev = [r for r in rows if r['split'] == 'dev']
    cutoff = cutoff_from_dev(dev)
    return rows, {split:summarize([r for r in rows if r['split'] == split], cutoff) for split in ('dev', 'test')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/shapes2d_v11')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    freeze_path = ROOT/'experiments/detector_training_freeze.json'
    if freeze_path.exists():
        raise FileExistsError('Preserve the frozen run; make a new version for changes')
    manifest_path = args.data/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('completed') is not True:
        raise ValueError('Require a completed audited dataset')
    data_audit_path = ROOT/'experiments/shapes2d_data_audit.json'
    data_freeze_path = ROOT/'experiments/shapes2d_data_freeze.json'
    data_audit = json.loads(data_audit_path.read_text())
    if data_audit.get('valid') is not True or data_audit.get('manifest_sha256') != sha(manifest_path):
        raise ValueError('Require the passing audit of this exact dataset manifest')
    episodes = manifest['episodes']
    split_families = {s:{e['family_id'] for e in episodes if e['split']==s} for s in ('train', 'dev', 'test')}
    if [len(split_families[s]) for s in ('train', 'dev', 'test')] != [2048, 128, 128]:
        raise ValueError('Unexpected family counts')
    if any(split_families[a]&split_families[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):
        raise ValueError('Whole-family leakage')
    (ROOT/'results').mkdir(parents=True, exist_ok=True)
    (ROOT/'checkpoints').mkdir(parents=True, exist_ok=True)
    plan = training_plan(episodes, args.epochs)
    plan_path = ROOT/'experiments/detector_batch_plan.json'
    write(plan_path, plan)
    torch.manual_seed(111)
    prototype = OneReferenceDetector()
    initial = {k:v.clone() for k,v in prototype.state_dict().items()}
    sources = [ROOT/'hypernet/one_shot_detector.py', ROOT/'hypernet/training_v3.py', Path(__file__)]
    freeze = dict(format=FORMAT, sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        manifest_sha256=sha(manifest_path), batch_plan_sha256=sha(plan_path), initial_state_sha256=state_hash(prototype),
        data_audit_sha256=sha(data_audit_path), data_freeze_sha256=sha(data_freeze_path),
        replay_plan_sha256=sha(ROOT/'experiments/detector_replay_plan.json'),
        parameter_count=sum(p.numel() for p in prototype.parameters()), seed=111,
        arms=['coarse_match', 'micro_aware'], steps_per_arm=len(plan), episodes_per_step=4,
        query_rows_per_step=8, reference_exposures_per_arm=4*len(plan), query_exposures_per_arm=8*len(plan),
        reference_cnn_views_per_arm=16*len(plan), reference_augmentation='four exact quarter-turn pixel permutations per enrolled image',
        training_families=2048, dev_families=128, test_families=128,
        natural_training_images=0, externally_pretrained_weights=False,
        optimizer={'name':'AdamW', 'lr':.001, 'weight_decay':.0001},
        loss={'match':'balanced binary cross entropy', 'foreground_weight':.4,
            'paired_micro_margin':.08, 'paired_micro_weight':.5,
            'ablation':'coarse_match omits candidate hard-negative BCE and paired margin; all other inputs/labels/steps identical'},
        primary_metric='TEST localized recall IoU>=0.5 using threshold fixed at <=5% DEV negative-scene FPR',
        detection='learned scene foreground -> connected components -> shared spatial CNN region descriptor -> reference cosine',
        inference_inputs=['one support RGB','support bounding box','full query RGB'],
        query_annotations_at_inference=False, checkpoint_selection='fixed final step, no DEV/TEST early stopping',
        calibration='Separate threshold per arm, DEV only; strict tie handling',
        runtime={'torch':torch.__version__, 'numpy':np.__version__, 'python':platform.python_version(), 'threads':args.threads},
        scope='2D synthetic composed shapes in separated clutter; heldout compositions share grammar building blocks, not natural semantic categories')
    write(freeze_path, freeze)
    store = FrameStore(args.data)
    results = dict(completed=False, freeze_sha256=sha(freeze_path), arms={})
    for arm in freeze['arms']:
        model = OneReferenceDetector(); model.load_state_dict(initial); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(4441)
        proposal_digest = hashlib.sha256()
        history, start = [], time.perf_counter()
        for step, indices in enumerate(plan):
            batch = [episodes[i] for i in indices]
            if any(e['split'] != 'train' for e in batch):
                raise RuntimeError('Heldout episode in a gradient batch')
            optimizer.zero_grad(set_to_none=True)
            loss, terms = batch_loss(model, batch, store, rng, arm=='micro_aware', proposal_digest)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            if step == 0 or (step+1)%50 == 0 or step+1 == len(plan):
                row = dict(step=step+1, total=float(loss.detach()), **{k:float(v.detach()) for k,v in terms.items()})
                history.append(row); print(arm, json.dumps(row), flush=True)
        elapsed = time.perf_counter()-start
        model.eval()
        for parameter in model.parameters(): parameter.grad = None
        checkpoint = ROOT/'checkpoints'/f'detector_{arm}.pt'
        torch.save(dict(format=FORMAT, state_dict=model.state_dict(), arm=arm,
            freeze_sha256=results['freeze_sha256'], natural_training_images=0, externally_pretrained_weights=False), checkpoint)
        predictions, metrics = evaluate(model, episodes, store)
        prediction_path = ROOT/'results'/f'{arm}_predictions.jsonl'
        prediction_path.write_text(''.join(json.dumps(r)+'\n' for r in predictions), encoding='utf-8')
        results['arms'][arm] = dict(metrics=metrics, history=history, training_seconds=elapsed,
            checkpoint_sha256=sha(checkpoint), final_state_sha256=state_hash(model),
            proposal_training_roster_sha256=proposal_digest.hexdigest(),
            predictions_sha256=sha(prediction_path), evaluation_read_only=True)
        write(ROOT/'results/detector_results.json', results)
        print(arm, 'evaluation', json.dumps(metrics), flush=True)
    if len({v['proposal_training_roster_sha256'] for v in results['arms'].values()}) != 1:
        raise RuntimeError('Paired training proposals diverged')
    results['completed'] = True
    results['sources_unchanged'] = all(sha(ROOT/p)==h for p,h in freeze['sources'].items())
    if not results['sources_unchanged']:
        raise RuntimeError('Frozen behavior source changed')
    write(ROOT/'results/detector_results.json', results)
    print('2D detection experiment complete.', flush=True)


if __name__ == '__main__':
    main()
