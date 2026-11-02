"""Fixed classical NCC diagnostic on frozen CNN proposals; no model fitting.

The scorer receives only support RGB, its enrollment box, query RGB, and CNN
proposal boxes. No query target boxes, masks, family labels or presence labels.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, math, os, platform, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZE = 96
PAD = 1.35
ANGLES = tuple(range(0, 360, 15))
SCALES = (.85, 1., 1.15)
SHIFTS = ((0., 0.), (-2., 0.), (2., 0.), (0., -2.), (0., 2.))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def gray(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image')
    return rgb.astype(np.float32) @ np.array([.299, .587, .114], np.float32)

def crop(image, box, angle=0., scale=1., shift=(0., 0.)):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Expected finite, nonempty half-open box')
    center = (box[:2]+box[2:])/2-.5
    span = max(box[2:]-box[:2])*PAD
    a = math.radians(angle)
    matrix = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]], np.float64)
    matrix *= span/(SIZE*scale)
    offset = center+np.asarray(shift)*span/SIZE-matrix@np.array([(SIZE-1)/2]*2)
    affine = np.column_stack([matrix, offset])
    perimeter = np.concatenate([image[0], image[-1], image[1:-1, 0], image[1:-1, -1]])
    # Image-derived boundary value only; no foreground or target mask is used.
    return cv2.warpAffine(image, affine, (SIZE, SIZE),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float(np.median(perimeter)))

def normalized(rows):
    rows = np.asarray(rows, np.float32).reshape(len(rows), -1)
    rows = rows-rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    # A nominally constant 8-bit image can acquire ~1e-5 floating point warp
    # residuals; do not amplify those into a unit-length artificial pattern.
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > .001*SIZE)

def reference_bank(rgb, support_box):
    image = gray(rgb)
    return normalized([crop(image, support_box, angle, scale)
        for angle in ANGLES for scale in SCALES])

def correlate(bank, query_rgb, proposals):
    image = gray(query_rgb)
    scores, transforms = [], []
    for box in proposals:
        queries = normalized([crop(image, box, shift=shift) for shift in SHIFTS])
        correlations = bank@queries.T
        flat = int(np.argmax(correlations))
        bank_index, shift_index = np.unravel_index(flat, correlations.shape)
        scores.append(float(np.clip(correlations[bank_index, shift_index], -1., 1.)))
        transforms.append(dict(angle_degrees=ANGLES[bank_index//len(SCALES)],
            reference_scale=SCALES[bank_index % len(SCALES)], query_shift=list(SHIFTS[shift_index])))
    best = int(np.argmax(scores)) if scores else None
    return dict(score=scores[best] if best is not None else -1.,
        bbox_xyxy=list(proposals[best]) if best is not None else None,
        proposals=[list(box) for box in proposals], proposal_scores=scores,
        winning_transform=transforms[best] if best is not None else None)

def iou(a, b):
    if a is None or b is None:
        return 0.
    w, h = max(0., min(a[2], b[2])-max(a[0], b[0])), max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = w*h
    return area/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area, 1e-12)

def cutoff(rows):
    scores = sorted([r['score'] for r in rows if not r['present']], reverse=True)
    if not scores:
        raise ValueError('DEV negatives are required')
    return float(np.nextafter(scores[math.floor(.05*len(scores))], math.inf))

def metrics(rows, threshold):
    positives = [r for r in rows if r['present']]
    negatives = [r for r in rows if not r['present']]
    tp = sum(r['score'] >= threshold for r in positives)
    fp = sum(r['score'] >= threshold for r in negatives)
    local = sum(r['score'] >= threshold and r['iou'] >= .5 for r in positives)
    auc = float(np.mean([float(p['score'] > n['score'])+.5*float(p['score'] == n['score'])
        for p in positives for n in negatives]))
    pairs = {}
    for row in rows:
        pairs.setdefault(row['episode_id'], {})[row['present']] = row['score']
    paired = float(np.mean([float(v[True] > v[False])+.5*float(v[True] == v[False]) for v in pairs.values()]))
    ranked = sorted(rows, key=lambda r:r['score'], reverse=True)
    seen, found, ap, cursor = 0, 0, 0., 0
    while cursor < len(ranked):
        end = cursor+1
        while end < len(ranked) and ranked[end]['score'] == ranked[cursor]['score']:
            end += 1
        hits = sum(r['present'] and r['iou'] >= .5 for r in ranked[cursor:end])
        found += hits; seen += end-cursor
        ap += found/seen*hits/len(positives)
        cursor = end
    return dict(positive_scenes=len(positives), negative_scenes=len(negatives), threshold=threshold,
        accepted_positive=tp, false_positive=fp, localized_true_positive=local,
        recall=tp/len(positives), false_positive_rate=fp/len(negatives),
        localized_recall_at_iou50=local/len(positives),
        proposal_recall_at_iou50=sum(r['proposal_best_iou'] >= .5 for r in positives)/len(positives),
        auroc=auc, single_box_ap50=ap, paired_scene_ranking_accuracy=paired,
        balanced_accuracy=.5*(tp/len(positives)+1-fp/len(negatives)))

def preview_checks():
    # Reserved gallery only. Preview boxes are allowed for timing/testing this
    # crop routine; they are never used for the heldout diagnostic evaluation.
    data = ROOT/'assets/grammar_preview'
    examples = json.loads((data/'manifest.json').read_text())['episodes'][:2]
    times, cases = [], []
    for episode in examples:
        support = np.array(Image.open(data/episode['support']['rgb_path']).convert('RGB'))
        start = time.perf_counter()
        bank = reference_bank(support, episode['support']['bbox_xyxy'])
        for query in episode['queries']:
            rgb = np.array(Image.open(data/query['rgb_path']).convert('RGB'))
            boxes = [o['bbox_xyxy'] for o in query['all_boxes']]
            result = correlate(bank, rgb, boxes)
            assert len(result['proposal_scores']) == len(boxes) and -1 <= result['score'] <= 1
            cases.append({'id':query['id'], 'box_count':len(boxes), 'finite':True})
        times.append(time.perf_counter()-start)
    # Pixel/identity tests do not inspect actual DEV/TEST scores or annotations.
    constant = np.full((96, 96, 3), 123, np.uint8)
    z = reference_bank(constant, [12, 12, 84, 84])
    assert not z.any()
    assert correlate(z, constant, [])['score'] == -1
    assert correlate(z, constant, [[12, 12, 84, 84]])['score'] == 0
    image = np.full((96, 96, 3), 30, np.uint8)
    image[20:77, 27:69] = 220; image[35:40, 37:45] = 30
    exact = correlate(reference_bank(image, [27, 20, 69, 77]), image, [[27, 20, 69, 77]])
    assert exact['score'] > .99999
    for scores in ([.5]*256, np.linspace(-1, 1, 256).tolist()):
        c = cutoff([{'score':s, 'present':False} for s in scores])
        assert sum(s >= c for s in scores) <= 12
    return dict(passed=True, scope='Reserved preview timing and synthetic unit cases; no DEV/TEST evaluation',
        episode_seconds=times, mean_seconds_per_query_including_half_enrollment=float(np.mean(times)/2),
        preview_cases=cases, exact_identity_score=exact['score'], constant_score=0., empty_score=-1.)

def freeze():
    path = ROOT/'experiments/analytic_detail_bypass_plan.json'
    if path.exists():
        raise FileExistsError('Preserve the diagnostic predeclaration')
    # No trained scores are read here. Record whether scoring files already
    # existed to make the timing of the diagnostic declaration auditable.
    prediction_files = {name:(ROOT/'results'/f'{name}_predictions.jsonl').exists()
        for name in ('coarse_match', 'micro_aware')}
    checks = preview_checks()
    checkpoint_paths = [ROOT/'checkpoints'/f'detector_{a}.pt' for a in ('coarse_match','micro_aware')]
    plan = dict(format='hypernet.analytic_detail_bypass.v11',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__), training_freeze_sha256=sha(ROOT/'experiments/detector_training_freeze.json'),
        manifest_sha256=sha(ROOT/'data/shapes2d_v11/manifest.json'),
        full_scored_prediction_files_exist_at_declaration=prediction_files,
        trained_checkpoints_exist_at_declaration={p.name:p.exists() for p in checkpoint_paths},
        interpretation='Classical NCC descriptor bypass using the micro_aware CNN predicted proposals; diagnostic hybrid, not a novel algorithm.',
        scorer_inputs=['support RGB','support enrollment bbox','query RGB','saved micro_aware predicted proposal boxes'],
        forbidden_scoring_inputs=['query ground-truth boxes','query masks','presence labels','family identity','candidate pose'],
        configuration=dict(patch_size=SIZE, square_box_padding=PAD, angles_degrees=ANGLES,
            relative_reference_scales=SCALES, query_translations_patch_pixels=SHIFTS,
            grayscale_weights=[.299,.587,.114], constant_rms_floor_luma_units=.001,
            score='maximum zero-mean unit-L2 normalized cross correlation',
            sampling='single bilinear affine resampling from native RGB-derived grayscale; no Gaussian blur',
            border='median original-image perimeter', transforms_per_proposal=len(ANGLES)*len(SCALES)*len(SHIFTS),
            proposal_source='exact saved micro_aware proposal lists, same order; no added/refined proposals'),
        calibration='DEV negatives only: strict nextafter above floor(.05*N)-indexed descending score; unchanged threshold on TEST',
        primary='TEST localized recall IoU>=.5 at <=5% DEV negative-scene FPR',
        training_updates=0, external_pretrained_weights=False, unique_enrollment_images=1,
        limitations=['Uses the trained CNN proposal stage, so it is not a wholly training-free detector.',
            'Fixed affine search interpolates pixels and can miss rotations/scales between bank samples.',
            'Full-patch NCC can favor macro silhouettes and background over microdetail.',
            'Any gain does not isolate information loss from rotation/alignment search differences.',
            'Synthetic separated-object scope; not a natural category result.'],
        preview_checks=checks, runtime=dict(numpy=np.__version__, opencv=cv2.__version__, python=platform.python_version()))
    write(path, plan)
    print(json.dumps({'frozen':str(path), 'sha256':sha(path), 'preview':checks, 'prediction_file_existence':prediction_files}, indent=2))

def evaluate():
    plan_path = ROOT/'experiments/analytic_detail_bypass_plan.json'
    plan = json.loads(plan_path.read_text())
    if sha(__file__) != plan['source_sha256']:
        raise ValueError('Diagnostic source changed after declaration')
    data = ROOT/'data/shapes2d_v11'
    if sha(data/'manifest.json') != plan['manifest_sha256']:
        raise ValueError('Dataset changed')
    results_path = ROOT/'results/detector_results.json'
    trained = json.loads(results_path.read_text())
    if not trained['completed']:
        raise ValueError('Wait for both fixed final model evaluations')
    freeze_path = ROOT/'experiments/detector_training_freeze.json'
    if sha(freeze_path) != plan['training_freeze_sha256']:
        raise ValueError('Training freeze changed')
    frozen = json.loads(freeze_path.read_text())
    tracked = {ROOT/p:h for p,h in frozen['sources'].items()}
    for arm in ('coarse_match','micro_aware'):
        tracked[ROOT/'checkpoints'/f'detector_{arm}.pt'] = trained['arms'][arm]['checkpoint_sha256']
    prediction_path = ROOT/'results/micro_aware_predictions.jsonl'
    tracked[prediction_path] = trained['arms']['micro_aware']['predictions_sha256']
    tracked[results_path] = sha(results_path)
    if any(sha(p) != h for p,h in tracked.items()):
        raise ValueError('Frozen source, checkpoint, or predictions changed')
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    by_id = {row['id']:row for row in predictions}
    episodes = [e for e in json.loads((data/'manifest.json').read_text())['episodes'] if e['split'] in ('dev','test')]
    assert len(by_id) == len(predictions) == sum(len(e['queries']) for e in episodes)
    output = ROOT/'results/analytic_detail_bypass_predictions.jsonl'
    if output.exists():
        raise FileExistsError('Preserve prior diagnostic results')
    rows, start, images = [], time.perf_counter(), {}
    def load_rgb(record):
        path = record['rgb_path']
        if path not in images:
            if sha(data/path) != record['rgb_sha256']:
                raise ValueError('Image hash mismatch')
            images[path] = np.array(Image.open(data/path).convert('RGB'))
        return images[path]
    with output.open('w', encoding='utf-8') as stream:
        for index, episode in enumerate(episodes):
            bank = reference_bank(load_rgb(episode['support']), episode['support']['bbox_xyxy'])
            for row in episode['queries']:
                previous = by_id[row['id']]
                if previous['support_path'] != episode['support']['rgb_path'] or previous['query_path'] != row['rgb_path']:
                    raise ValueError('CNN proposal/source binding mismatch')
                result = correlate(bank, load_rgb(row), previous['proposals'])
                # Query annotations are used only after the image/box scorer returns.
                target = row['target_bbox_xyxy'] if row['present'] else None
                result.update(id=row['id'],episode_id=episode['episode_id'],split=episode['split'],
                    family_id=episode['family_id'],enrolled_variant=episode['enrolled_variant'],present=bool(row['present']),
                    iou=iou(result['bbox_xyxy'],target),
                    proposal_best_iou=max([iou(box,target) for box in result['proposals']],default=0.),
                    support_path=episode['support']['rgb_path'],query_path=row['rgb_path'])
                assert result['proposals'] == previous['proposals']
                assert result['proposal_best_iou'] == previous['proposal_best_iou']
                rows.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n')
            if (index+1)%64 == 0:
                print(f'Completed {index+1}/{len(episodes)} enrollments',flush=True)
    threshold = cutoff([r for r in rows if r['split']=='dev'])
    if any(sha(p) != h for p,h in tracked.items()):
        raise RuntimeError('Read-only artifact hashes changed')
    summary = dict(completed=True,plan_sha256=sha(plan_path),source_sha256=sha(__file__),
        predictions_sha256=sha(output),frozen_input_hashes={str(p.relative_to(ROOT)):h for p,h in tracked.items()},
        weights_and_frozen_sources_unchanged=True,training_updates=0,model_loaded=False,
        exact_same_micro_aware_proposals=True,score_rows=len(rows),seconds=time.perf_counter()-start,
        metrics={split:metrics([r for r in rows if r['split']==split],threshold) for split in ('dev','test')},
        interpretation=plan['interpretation'],limitations=plan['limitations'])
    write(ROOT/'results/analytic_detail_bypass_results.json',summary)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','evaluate'])
    args = parser.parse_args();cv2.setNumThreads(1)
    freeze() if args.mode == 'freeze' else evaluate()
