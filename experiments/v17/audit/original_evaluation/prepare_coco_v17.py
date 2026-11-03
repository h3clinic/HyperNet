"""Metadata-first COCO cohort; old v14 photos are DEV, new TEST stays gated."""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, hashlib, json, time, urllib.request
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
OLD = WORKSPACE / 'outputs/hypernet_v14'
DATA = ROOT / 'data/coco_v17'
PLAN = DATA / 'plan.json'
SALT = 'HyperNet-v17-fresh-natural-scenes-v1:'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(data, f, indent=2, allow_nan=False); f.write('\n')
def pixel_sha(rgb):
    a = np.ascontiguousarray(rgb)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()
def rgb(path):
    with Image.open(path) as im:
        if im.getexif().get(274, 1) != 1: raise ValueError('Unexpected EXIF orientation')
        return np.asarray(im.convert('RGB')).copy()
def verify(bindings):
    for p, h in bindings.items():
        if sha(p) != h: raise ValueError('Bound input changed: ' + str(p))
def boxes(row):
    x, y, w, h = row['bbox']; return [float(x), float(y), float(x + w), float(y + h)]
def source_path(row): return WORKSPACE / row['workspace_path']

def annotations_for(records, targets):
    box_map = {}; valid = {}
    for target in targets:
        cid = int(target['category_id']); key = str(cid)
        selected = [a for a in records if a['category_id'] == cid]
        box_map[key] = [boxes(a) for a in selected if not a['iscrowd']]
        valid[key] = not any(a['iscrowd'] for a in selected)
    return {'boxes_by_target': box_map, 'valid_by_target': valid}

def rank(role, image_id): return hashlib.sha256((SALT + role + ':' + str(image_id)).encode()).hexdigest()

def choose_fresh(images, annotations, targets, excluded, positives=16, negatives=32):
    """Fixed annotation-only strata; all photos unique and free of target crowds."""
    target_ids = {t['category_id'] for t in targets}
    pool = {iid: r for iid, r in images.items() if iid not in excluded and
            not any(a['iscrowd'] and a['category_id'] in target_ids for a in annotations[iid])}
    used = set(); chosen = []
    for target in targets:
        cid = target['category_id']
        for present, count in ((True, positives), (False, negatives)):
            role = f'{cid:03d}:' + ('positive' if present else 'negative')
            candidates = []
            for iid, im in pool.items():
                if iid in used: continue
                selected = [a for a in annotations[iid] if a['category_id'] == cid]
                if present:
                    eligible = any(not a['iscrowd'] and min(a['bbox'][2:]) >= 16 and
                                   a['bbox'][2] * a['bbox'][3] >= .01 * im['width'] * im['height'] for a in selected)
                else: eligible = not selected
                if eligible: candidates.append(iid)
            candidates.sort(key=lambda iid: rank(role, iid))
            if len(candidates) < count: raise ValueError('Insufficient fixed stratum: ' + role)
            for iid in candidates[:count]:
                used.add(iid); chosen.append((iid, cid, present))
    return chosen, len(pool)

def plan_stage():
    if PLAN.exists(): raise FileExistsError('Preserve existing metadata plan')
    old_data = OLD / 'data/coco_real_v14'
    manifest_path = old_data / 'manifest.json'; prior_path = old_data / 'prior_image_inventory.json'
    index_path = old_data / 'sealed/source_annotation_index.json'
    audit_path = OLD / 'experiments/real_photo_data_audit.json'
    old_freeze_path = OLD / 'experiments/real_photo_data_plan_freeze.json'
    manifest = read(manifest_path); prior = read(prior_path); index = read(index_path)
    audit = read(audit_path); freeze = read(old_freeze_path)
    if not audit['valid'] or audit['manifest_sha256'] != sha(manifest_path): raise ValueError('Old manifest audit differs')
    if audit['prior_inventory_sha256'] != sha(prior_path) or freeze['source_index_sha256'] != sha(index_path):
        raise ValueError('Old source/index binding differs')
    targets = sorted(manifest['categories'], key=lambda x: x['category_id'])
    excluded = set(prior['excluded_coco_image_ids']) | {r['image_id'] for r in manifest['references'] + manifest['queries']}
    if len(excluded) != 3072: raise ValueError('Expected 3,072 prior COCO sources')
    by_image = defaultdict(list)
    for a in index['annotations']: by_image[a['image_id']].append(a)
    images = {r['id']: r for r in index['images']}
    licenses = {r['id']: r for r in index['licenses']}
    refs = []; dev = []
    for row in manifest['references']:
        refs.append({**row, 'workspace_path': str((old_data / row['rgb_path']).relative_to(WORKSPACE)).replace('\\', '/')})
    for row in sorted(manifest['queries'], key=lambda x: x['image_id']):
        dev.append({**row, 'split': 'dev', 'original_v14_split': row['split'],
                    'workspace_path': str((old_data / row['rgb_path']).relative_to(WORKSPACE)).replace('\\', '/'),
                    **annotations_for(by_image[row['image_id']], targets)})
    selected, pool_count = choose_fresh(images, by_image, targets, excluded)
    test = []
    for iid, assigned, present in sorted(selected):
        im = images[iid]; lic = licenses[im['license']]
        test.append({'id': f'coco_v17_{iid:012d}', 'image_id': iid, 'split': 'test',
                     'width': im['width'], 'height': im['height'], 'rgb_path': f'images/{iid:012d}.jpg',
                     'selection_category_id': assigned, 'selection_present': present,
                     'source': {'dataset': 'COCO2017', 'official_source_split': 'train2017',
                                'coco_url': im['coco_url'], 'file_name': im['file_name'],
                                'flickr_url': im.get('flickr_url'), 'date_captured': im.get('date_captured'),
                                'license_id': im['license'], 'license_name': lic['name'], 'license_url': lic['url']},
                     **annotations_for(by_image[iid], targets)})
    if len(dev) != 288 or len(test) != 576 or len({r['image_id'] for r in dev + test + refs}) != 876:
        raise ValueError('Unexpected unique cohort dimensions')
    bindings = {str(p.resolve()): sha(p) for p in (manifest_path, prior_path, index_path, audit_path, old_freeze_path, Path(__file__))}
    counts = {}
    for split, rows in [('dev', dev), ('test', test)]:
        counts[split] = {str(t['category_id']): {'positive': sum(bool(r['boxes_by_target'][str(t['category_id'])]) and r['valid_by_target'][str(t['category_id'])] for r in rows),
                                              'negative': sum(not r['boxes_by_target'][str(t['category_id'])] and r['valid_by_target'][str(t['category_id'])] for r in rows),
                                              'ignored_crowd': sum(not r['valid_by_target'][str(t['category_id'])] for r in rows)} for t in targets}
    plan = {'format': 'hypernet.v17.coco.metadata_plan.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'targets': targets, 'references': refs, 'dev': dev, 'test': test, 'input_bindings': bindings,
            'prior_excluded_image_ids': sorted(excluded), 'per_target_counts': counts,
            'selection': {'salt': SALT, 'positive_sources_per_assigned_category': 16, 'negative_sources_per_assigned_category': 32,
                          'eligible_pool': pool_count, 'query_photos_globally_unique': True,
                          'test_pixels_acquired': False, 'model_scores_used': False,
                          'test_crowd_policy': 'Exclude candidate photo if any of the 12 target categories has a crowd annotation.',
                          'dev_crowd_policy': 'Ignore only target/photo pairs with a target crowd annotation; retain the photo for other targets.',
                          'positive_stratum_eligibility': 'At least one target box with min edge16pixels and area>=1% of image. All noncrowd boxes, including smaller incidental targets, define evaluation truth.',
                          'reference_crop': 'Original v14 single boxed support; floor left/top, ceil right/bottom, clip to image; no query annotation is a model input.'},
            'protocol': {'all_targets_on_all_photos': True, 'primary': 'Macro localized recall: accepted highest-score box overlaps any target GT box at IoU>=0.5.',
                         'primary_calibration_negative_fpr': .05, 'secondary_calibration_negative_fpr': .001,
                         'cutoff': 'One global cutoff per method from all valid DEV target-absent pairs; no TEST tuning.',
                         'methods': ['v17', 'v17_no_geometry', 'random', 'owlvit_base_patch32', 'raw_template'],
                         'confidence_intervals': 'Paired resampling of query photos, carrying all 12 target outcomes together; references/calibration fixed.',
                         'real_weight_training_images': 0, 'borrowed_hypernet_weights': False, 'reference_images_per_type': 1,
                         'additional_dev_photos': 288, 'fresh_test_photos': 576, 'nominal_test_pairs': 6912},
            'limitations': ['Fresh means absent from prior HyperNet photo iterations, not unseen by the pretrained OWL checkpoint.',
                            'COCO is named in OWL training; no absence of pretraining overlap or novel category status is established.',
                            'One boxed support per category is additional annotation; 288 labeled DEV photos are supervision beyond the support.',
                            'COCO absence means no supplied annotation; missing labels and physical-instance duplicates may remain.',
                            'Selected source strata are not natural prevalence. False alarms per10000 are normalized rates, not certification of rare-event operation.',
                            'Top1 localized recall is a custom metric, not the published COCO AP metric.']}
    write_new(PLAN, plan)
    print(json.dumps({'plan_sha256': sha(PLAN), 'dev_photos': len(dev), 'test_photos': len(test), 'test_pairs': 6912,
                      'prior_excluded_sources': len(excluded), 'per_target_counts': counts}))

def verify_plan():
    plan = read(PLAN); verify(plan['input_bindings']); return plan

def load_dev_reference(row):
    path = source_path(row)
    if sha(path) != row['sha256']: raise ValueError('Old source photo changed')
    a = rgb(path)
    if pixel_sha(a) != row['pixel_sha256']: raise ValueError('Old photo pixels changed')
    return a

def load_reference(row):
    a = load_dev_reference(row); x1, y1, x2, y2 = row['bbox_xyxy']
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(a.shape[1], int(np.ceil(x2))), min(a.shape[0], int(np.ceil(y2)))
    if right <= left or bottom <= top: raise ValueError('Empty support crop')
    return a[top:bottom, left:right].copy()

def fetch_stage(protocol_path, expected, workers):
    plan = verify_plan(); protocol_path = Path(protocol_path).resolve()
    if sha(protocol_path) != expected: raise ValueError('Expected model protocol hash differs')
    protocol = read(protocol_path)
    if not protocol.get('model_selection_complete') or protocol.get('test_scores_computed_before_freeze') is not False:
        raise ValueError('Model selection must be frozen before TEST acquisition')
    if protocol['data_plan_sha256'] != sha(PLAN): raise ValueError('Wrong data plan')
    verify(protocol['behavior_files']); verify(protocol['checkpoint_files'])
    gate_path = DATA / 'acquisition_admission.json'
    if gate_path.exists():
        gate = read(gate_path)
        if gate['protocol_sha256'] != expected or gate['data_plan_sha256'] != sha(PLAN): raise ValueError('Different existing acquisition gate')
    else:
        if (DATA / 'images').exists() and any((DATA / 'images').iterdir()): raise ValueError('TEST pixels existed before gate')
        write_new(gate_path, {'protocol_path': str(protocol_path), 'protocol_sha256': expected, 'data_plan_sha256': sha(PLAN),
                              'created_utc': datetime.now(timezone.utc).isoformat(), 'before_first_test_image_write': True})
    (DATA / 'images').mkdir(exist_ok=True); (DATA / 'image_records').mkdir(exist_ok=True)
    def fetch(row):
        path = DATA / row['rgb_path']; record_path = DATA / 'image_records' / f"{row['image_id']:012d}.json"
        if record_path.exists():
            record = read(record_path)
            if sha(path) != record['sha256'] or pixel_sha(rgb(path)) != record['pixel_sha256']: raise ValueError('Existing download changed')
            return record
        if path.exists(): raise ValueError('Uncommitted partial image; do not silently replace')
        url = 'https://s3.amazonaws.com/images.cocodataset.org/train2017/' + row['source']['file_name']
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=45) as response: payload = response.read()
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(attempt + 1)
        with path.open('xb') as f: f.write(payload)
        a = rgb(path)
        if a.shape != (row['height'], row['width'], 3): raise ValueError('Source dimensions differ')
        record = {'image_id': row['image_id'], 'sha256': sha(path), 'pixel_sha256': pixel_sha(a), 'transport_url': url}
        write_new(record_path, record); return record
    successes = {}; failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, row): row for row in plan['test']}
        for future in as_completed(futures):
            row = futures[future]
            try: successes[row['image_id']] = future.result()
            except Exception as e: failures.append({'image_id': row['image_id'], 'error': str(e)})
            if (len(successes) + len(failures)) % 64 == 0: print(json.dumps({'acquired': len(successes), 'failures': len(failures), 'total': len(plan['test'])}), flush=True)
    if failures: raise RuntimeError(json.dumps({'incomplete_download': failures}))
    prior = read(OLD / 'data/coco_real_v14/prior_image_inventory.json')
    old = read(OLD / 'data/coco_real_v14/manifest.json')
    prior_raw = {r['sha256'] for r in prior['files'] + old['references'] + old['queries']}
    prior_pixels = {r['pixel_sha256'] for r in prior['files'] + old['references'] + old['queries']}
    raw_groups = defaultdict(list); pixel_groups = defaultdict(list); excluded = []
    for iid, record in successes.items():
        raw_groups[record['sha256']].append(iid); pixel_groups[record['pixel_sha256']].append(iid)
    for row in plan['test']:
        record = successes[row['image_id']]; reasons = []
        if record['sha256'] in prior_raw or record['pixel_sha256'] in prior_pixels: reasons.append('exact_prior_pixels')
        if len(raw_groups[record['sha256']]) > 1 or len(pixel_groups[record['pixel_sha256']]) > 1: reasons.append('within_test_exact_duplicate')
        if reasons: excluded.append({'image_id': row['image_id'], 'reasons': reasons})
    removed = {r['image_id'] for r in excluded}
    result = {'completed': True, 'data_plan_sha256': sha(PLAN), 'protocol_sha256': expected, 'protocol_path': str(protocol_path),
              'acquisition_admission_sha256': sha(gate_path), 'requested': len(plan['test']), 'downloaded': len(successes),
              'eligible_image_ids': [r['image_id'] for r in plan['test'] if r['image_id'] not in removed], 'excluded': excluded,
              'records': {str(iid): row for iid, row in successes.items()}, 'failures': failures,
              'model_scores_used_for_quality': False, 'replacement_sampling': False}
    write_new(DATA / 'acquisition.json', result)
    print(json.dumps({'completed': True, 'eligible': len(result['eligible_image_ids']), 'excluded': len(excluded), 'sha256': sha(DATA / 'acquisition.json')}))

def load_test(row, acquisition):
    record = acquisition['records'][str(row['image_id'])]; path = DATA / row['rgb_path']
    if sha(path) != record['sha256']: raise ValueError('TEST source changed')
    a = rgb(path)
    if pixel_sha(a) != record['pixel_sha256']: raise ValueError('TEST pixels changed')
    return a

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('plan', 'fetch-test'))
    p.add_argument('--protocol', type=Path); p.add_argument('--expected-protocol-sha'); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.stage == 'plan': plan_stage()
    else:
        if not a.protocol or not a.expected_protocol_sha or not 1 <= a.workers <= 8: p.error('Frozen protocol/hash and workers1..8 required')
        fetch_stage(a.protocol, a.expected_protocol_sha, a.workers)
