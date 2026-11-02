"""Annotation-only selection and exact provenance audit for the v13 real-photo test.

Stages are write-once: plan (including prior-pixel inventory), fetch, audit.
No model code, weights, predictions, or score files are imported or read.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
DATA = ROOT / 'data/coco_real_v13'
EXPERIMENTS = ROOT / 'experiments'
ANNOTATIONS = WORKSPACE / 'work/data/coco2017/annotations/instances_val2017.json'
CACHE = WORKSPACE / 'work/data/coco2017/val2017'
SALT = 'HyperNet-v13-real-category-1:'
PHOTO_SALT = 'HyperNet-v13-real-photo-1:'
LICENSE_IDS = [4, 5, 7, 8]
PRIOR_RELS = [
    'outputs/hypernet_starter/data/coco24_category_v03/manifest.json',
    'outputs/hypernet_starter/data/coco24_scratch_v04/manifest.json',
    'outputs/hypernet_v05/data/coco_scratch_v05/manifest.json',
    'outputs/hypernet_v06/data/coco_fresh_v06/manifest.json',
    'outputs/hypernet_v08/data/coco_fresh_v08/manifest.json',
    'outputs/hypernet_v09/data/coco_fresh_v09/manifest.json',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'), dtype=np.uint8)


def pixels(array):
    array = np.ascontiguousarray(array)
    return sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def legacy_pixels(array):
    return sha256(str(array.shape).encode() + np.ascontiguousarray(array).tobytes()).hexdigest()


def rank(label, value):
    return sha256((label + str(value)).encode()).hexdigest()


def xyxy(annotation):
    x, y, w, h = annotation['bbox']
    return [float(x), float(y), float(x+w), float(y+h)]


def eligible(annotation, image, *, reference=False):
    w, h = annotation['bbox'][2:]
    return (not annotation.get('iscrowd', 0) and min(w, h) >= (32 if reference else 16)
            and w*h >= image['width']*image['height']*(.02 if reference else .01))


def match_unique(candidates):
    """Deterministic bipartite matching; each photo can occupy only one slot."""
    ownership = {}

    def augment(slot, seen):
        for iid in candidates[slot]:
            if iid in seen:
                continue
            seen.add(iid)
            if iid not in ownership or augment(ownership[iid], seen):
                ownership[iid] = slot
                return True
        return False

    for slot in sorted(candidates, key=lambda s: (len(candidates[s]), s)):
        require(augment(slot, set()), f'Insufficient globally unique photos for {slot}')
    return {slot: iid for iid, slot in ownership.items()}


def prior_inventory():
    manifests = []
    expected = {}
    old_ids = set()
    source_paths = set()
    stored_paths = set()
    for relative in PRIOR_RELS:
        path = WORKSPACE / relative
        manifest = read(path)
        manifests.append({'path': relative, 'sha256': digest(path)})
        for row in manifest['images']:
            old_ids.add(row['coco_image_id'])
            p = CACHE / row['source_relative_path']
            expected[p] = (row['source_sha256'], row['source_pixel_sha256'])
            source_paths.add(p)
            stored_paths.add(path.parent / row['path'])
    duplicate = WORKSPACE / 'outputs/hypernet_v08_1/data/coco_fresh_v08/manifest.json'
    require(digest(duplicate) == digest(WORKSPACE / PRIOR_RELS[-2]), 'v08_1 is not a duplicate cohort')
    # Caltech category cohort and its full-size sources.
    caltech = WORKSPACE / 'outputs/hypernet_starter/data/caltech24_category_v03/manifest.json'
    manifests.append({'path': str(caltech.relative_to(WORKSPACE)), 'sha256': digest(caltech)})
    for row in read(caltech)['images']:
        p = WORKSPACE / 'work/data/caltech101/101_ObjectCategories' / row['source_relative_path']
        source_paths.add(p)
        expected[p] = (row['source_sha256'], row['source_pixel_sha256'])
        stored_paths.add(caltech.parent / row['path'])
    # All cached legacy ants/bees and independent Oxford views are conservatively
    # excluded, including old unmanifested first live-test inputs and copies.
    for relative in ['work/data/hymenoptera_data', 'work/data/oxford_affine',
                     'work/fresh_hypernet_sources', 'work/hypernet_v02_sources']:
        source_paths.update(p for p in (WORKSPACE / relative).rglob('*')
                            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.ppm', '.bmp'})
    for relative in ['synthetic24', 'synthetic32_fresh', 'synthetic32_workarounds', 'oxford8']:
        folder = WORKSPACE / 'outputs/hypernet_starter/data' / relative
        path = folder / 'manifest.json'
        manifests.append({'path': str(path.relative_to(WORKSPACE)), 'sha256': digest(path)})
        stored_paths.update(p for p in folder.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    all_paths = sorted(source_paths | stored_paths)

    def inspect(path):
        a = rgb(path)
        raw, oldpixel = digest(path), legacy_pixels(a)
        if path in expected:
            require((raw, oldpixel) == expected[path], f'Prior source changed: {path}')
        return {'path': str(path.relative_to(WORKSPACE)), 'sha256': raw, 'pixel_sha256': pixels(a),
                'width': int(a.shape[1]), 'height': int(a.shape[0]),
                'role': 'source' if path in source_paths else 'prior_stored_input'}

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(inspect, all_paths))
    return {'schema': 'hypernet-v13-prior-image-inventory-v1', 'prior_manifests': manifests,
            'duplicate_v08_1_sha256': digest(duplicate), 'excluded_coco_image_ids': sorted(old_ids),
            'source_files': len(source_paths), 'stored_input_files': len(stored_paths),
            'files': records, 'pixel_hash_convention': 'SHA256(str(dtype)+str(shape)+contiguous RGB bytes); EXIF transposed',
            'scope': 'All manifested prior COCO/Caltech sources and stored inputs; conservative full legacy ants/bees and Oxford source caches, including earlier same-source transformations.'}


def select_plan(source, inventory):
    images = {r['id']: r for r in source['images']}
    category_names = {r['id']: r['name'] for r in source['categories']}
    old_ids = set(inventory['excluded_coco_image_ids'])
    allowed = {i for i, row in images.items() if i not in old_ids and row['license'] in LICENSE_IDS}
    anns = defaultdict(list)
    labels = defaultdict(set)
    positives = defaultdict(set)
    supports = defaultdict(set)
    for a in source['annotations']:
        iid, category = a['image_id'], a['category_id']
        anns[iid].append(a)
        labels[iid].add(category)  # includes crowd: crowd-containing photos cannot be negatives
        if iid in allowed and eligible(a, images[iid]):
            positives[category].add(iid)
        if iid in allowed and eligible(a, images[iid], reference=True):
            supports[category].add(iid)
    eligible_categories = sorted((c for c in category_names if len(positives[c]) >= 13 and supports[c]),
                                 key=lambda c: rank(SALT, c))
    require(len(eligible_categories) >= 12, 'Fewer than 12 annotation-eligible categories')
    selected = eligible_categories[:12]
    candidates = {}
    slot_info = {}
    for category in selected:
        support_key = f'{category:03d}:reference'
        candidates[support_key] = sorted(supports[category], key=lambda i: rank(PHOTO_SALT+support_key+':', i))
        slot_info[support_key] = (category, 'reference', True)
        for split, count in [('dev', 4), ('test', 8)]:
            for n in range(count):
                key = f'{category:03d}:{split}:positive:{n:02d}'
                candidates[key] = sorted(positives[category], key=lambda i: rank(PHOTO_SALT+key+':', i))
                slot_info[key] = (category, split, True)
    assignment = match_unique(candidates)
    reserved = set(assignment.values())
    negative_candidates = {}
    for category in selected:
        for split, count in [('dev', 4), ('test', 8)]:
            for n in range(count):
                key = f'{category:03d}:{split}:negative:{n:02d}'
                pool = allowed - reserved
                negative_candidates[key] = sorted((i for i in pool if category not in labels[i]),
                                                  key=lambda i: rank(PHOTO_SALT+key+':', i))
                slot_info[key] = (category, split, False)
    assignment.update(match_unique(negative_candidates))
    require(len(assignment) == len(set(assignment.values())) == 300, 'Expected 300 globally unique photographs')
    licenses = {r['id']: r for r in source['licenses']}
    refs, queries = [], []
    for key, iid in sorted(assignment.items()):
        c, split, present = slot_info[key]
        image = images[iid]
        row = {'id': 'coco_v13_'+key.replace(':', '_'), 'category_id': c, 'name': category_names[c],
               'image_id': iid, 'rgb_path': f'images/{iid:012d}.jpg',
               'width': image['width'], 'height': image['height'],
               'source': {'dataset': 'COCO2017 val2017', 'file_name': image['file_name'],
                          'coco_url': image['coco_url'], 'flickr_url': image.get('flickr_url'),
                          'date_captured': image.get('date_captured'),
                          'license_id': image['license'], 'license_name': licenses[image['license']]['name'],
                          'license_url': licenses[image['license']]['url']}}
        if split == 'reference':
            eligible_annotations = [a for a in anns[iid] if a['category_id'] == c and eligible(a, image, reference=True)]
            chosen = sorted(eligible_annotations, key=lambda a: (-a['bbox'][2]*a['bbox'][3], a['id']))[0]
            refs.append({**row, 'bbox_xyxy': xyxy(chosen), 'annotation_id': chosen['id']})
        else:
            target_annotations = [a for a in anns[iid] if a['category_id'] == c and not a.get('iscrowd', 0)]
            queries.append({**row, 'split': split, 'present': present,
                            'gt_boxes_xyxy': [xyxy(a) for a in target_annotations],
                            'gt_annotation_ids': [a['id'] for a in target_annotations],
                            'target_crowd_annotations': [a['id'] for a in anns[iid] if a['category_id'] == c and a.get('iscrowd', 0)]})
    return {'schema_version': 'hypernet-real-photo-v13-plan-v1',
            'task': 'Any instance of the enrolled natural COCO category; not exact-object identity',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'categories': [{'category_id': c, 'name': category_names[c]} for c in selected],
            'references': refs, 'queries': queries,
            'selection': {'category_hash_salt': SALT, 'photo_hash_salt': PHOTO_SALT,
                          'eligible_categories_hash_order': eligible_categories,
                          'availability': {category_names[c]: {'positive_sources': len(positives[c]), 'reference_sources': len(supports[c])} for c in eligible_categories},
                          'remaining_licensed_pool': len(allowed), 'excluded_prior_coco_sources': len(old_ids),
                          'license_ids': LICENSE_IDS, 'category_scores_used': False,
                          'positive_eligibility': 'At least one noncrowd box with minimum edge16px and box area >=1% image area',
                          'support_eligibility': 'Largest eligible noncrowd box; minimum edge32px, box area >=2% image area',
                          'negative_eligibility': 'No target category annotation, including crowd',
                          'assignment': 'First12 SHA-ranked annotation-eligible categories; deterministic maximum bipartite matching of refs/positives, then negatives; all300photos unique'},
            'counts': {'categories': 12, 'references': 12, 'dev_queries': 96, 'test_queries': 192,
                       'positive_queries': 144, 'negative_queries': 144, 'unique_source_photos': 300},
            'budget': {'real_reference_photos_per_category': 1, 'real_reference_boxes_per_category': 1,
                       'dev_calibration_photos': 96, 'test_photos': 192, 'natural_photo_weight_training': 0,
                       'query_boxes_or_labels_model_inputs': 0},
            'limits': ['COCO category types were used in earlier project studies; photographs are new to these studies.',
                       'Exact original IDs/raw/decoded hashes cannot prove different physical individuals, capture sessions, or absence of near-duplicates.',
                       'Annotation-absent negatives can include missed COCO annotations.',
                       'This tests synthetic-to-natural category transfer, a different task from synthetic micro-identity discrimination.']}


def plan_stage():
    require(not (DATA/'plan.json').exists(), 'Selection plan already exists')
    inventory = prior_inventory()
    write_new(DATA/'prior_image_inventory.json', inventory)
    plan = select_plan(read(ANNOTATIONS), inventory)
    write_new(DATA/'plan.json', plan)
    write_new(EXPERIMENTS/'real_photo_data_freeze.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest(DATA/'plan.json'), 'prior_inventory_sha256': digest(DATA/'prior_image_inventory.json'),
        'annotation_sha256': digest(ANNOTATIONS), 'builder_sha256': digest(__file__),
        'photos_fetched_for_v13': False, 'model_scores_read': False,
        'prior_manifests': inventory['prior_manifests'], 'counts': plan['counts']})
    print(json.dumps({'status':'frozen', 'categories':plan['categories'], 'counts':plan['counts'],
                      'prior_files':len(inventory['files']), 'freeze_sha256':digest(EXPERIMENTS/'real_photo_data_freeze.json')}), flush=True)


def verify_freeze():
    freeze = read(EXPERIMENTS/'real_photo_data_freeze.json')
    for key, path in [('plan_sha256', DATA/'plan.json'), ('prior_inventory_sha256', DATA/'prior_image_inventory.json'),
                      ('annotation_sha256', ANNOTATIONS), ('builder_sha256', Path(__file__))]:
        require(freeze[key] == digest(path), f'Frozen file changed: {path}')
    for row in freeze['prior_manifests']:
        require(digest(WORKSPACE/row['path']) == row['sha256'], 'Prior manifest changed')
    return freeze


def fetch_stage():
    verify_freeze()
    require(not (DATA/'manifest.json').exists(), 'Manifest already exists')
    write_new(EXPERIMENTS/'real_photo_fetch_admission.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_freeze_sha256': digest(EXPERIMENTS/'real_photo_data_freeze.json'),
        'plan_sha256': digest(DATA/'plan.json'), 'before_first_v13_image_write': True,
        'inference_performed': False})
    plan = read(DATA/'plan.json')
    (DATA/'images').mkdir(parents=True, exist_ok=True)
    provenance = []

    def fetch(row):
        dest = DATA/row['rgb_path']
        require(not dest.exists(), f'Output photo already exists: {dest}')
        cached = CACHE/row['source']['file_name']
        if cached.is_file():
            shutil.copyfile(cached, dest)
            method = 'existing_unselected_coco_cache'
        else:
            url = row['source']['coco_url'].replace('http://', 'https://')
            failure = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=45) as response:
                        payload = response.read()
                    with dest.open('xb') as stream:
                        stream.write(payload)
                    failure = None
                    break
                except Exception as error:
                    failure = error
                    time.sleep(attempt+1)
            if failure:
                raise failure
            method = 'download_official_coco_url'
        with Image.open(dest) as opened:
            orientation = opened.getexif().get(274, 1)
        require(orientation == 1, 'Nontrivial EXIF orientation requires an explicitly frozen box transform')
        image = rgb(dest)
        require(image.shape == (row['height'], row['width'], 3), 'Original dimensions differ after EXIF transpose')
        enriched = {**row, 'sha256': digest(dest), 'pixel_sha256': pixels(image), 'exif_orientation': orientation}
        return enriched, {'image_id': row['image_id'], 'method': method,
                          'url': row['source']['coco_url'], 'sha256': enriched['sha256']}

    with ThreadPoolExecutor(max_workers=6) as pool:
        loaded = list(pool.map(fetch, plan['references']+plan['queries']))
    rows = [r for r, _ in loaded]
    provenance = [p for _, p in loaded]
    manifest = {**plan, 'schema_version': 'hypernet-real-photo-v13-v1',
                'references': rows[:12], 'queries': rows[12:],
                'pixel_hash_convention': 'SHA256(str(dtype)+str(shape)+contiguous RGB bytes); dtype uint8; EXIF transposed',
                'plan_sha256': digest(DATA/'plan.json'),
                'data_freeze_sha256': digest(EXPERIMENTS/'real_photo_data_freeze.json'),
                'annotation_sha256': digest(ANNOTATIONS)}
    write_new(DATA/'manifest.json', manifest)
    write_new(DATA/'download_provenance.json', {'manifest_sha256': digest(DATA/'manifest.json'), 'images': provenance})
    write_new(DATA/'source_licenses.json', {'dataset': 'COCO2017 val2017',
        'annotation_license': 'CC BY4.0; this does not supersede individual photograph licenses',
        'annotation_terms': 'https://cocodataset.org/#termsofuse',
        'photo_licenses': read(ANNOTATIONS)['licenses'],
        'images': [{'image_id': row['image_id'], **row['source'], 'sha256': row['sha256'], 'pixel_sha256': row['pixel_sha256']} for row in rows]})
    print(json.dumps({'status':'fetched', 'methods':dict(Counter(p['method'] for p in provenance)),
                      'manifest_sha256':digest(DATA/'manifest.json')}), flush=True)


def audit_stage():
    verify_freeze()
    manifest = read(DATA/'manifest.json')
    plan = read(DATA/'plan.json')
    prior = read(DATA/'prior_image_inventory.json')
    source = read(ANNOTATIONS)
    anns = {r['id']: r for r in source['annotations']}
    by_image = defaultdict(list)
    for a in source['annotations']:
        by_image[a['image_id']].append(a)
    rows = manifest['references']+manifest['queries']
    planned_rows = plan['references']+plan['queries']
    require(len(rows)==len(planned_rows), 'Frozen roster length changed')
    for row, planned in zip(rows, planned_rows):
        require({key:row[key] for key in planned} == planned, 'Manifest changed the frozen selection/annotations')
    require(len(rows) == len({r['image_id'] for r in rows}) == 300, 'Source ID repetition')
    require(not set(r['image_id'] for r in rows) & set(prior['excluded_coco_image_ids']), 'Old source ID reused')
    old_raw = {r['sha256'] for r in prior['files']}
    old_pixels = {r['pixel_sha256'] for r in prior['files']}
    seen_raw, seen_pixels = set(), set()
    for row in rows:
        path = (DATA/row['rgb_path']).resolve()
        require(path.is_relative_to(DATA.resolve()), 'Unsafe image path')
        a = rgb(path)
        with Image.open(path) as opened:
            require(opened.getexif().get(274,1)==row['exif_orientation']==1,'EXIF annotation coordinate mismatch')
        raw, pixel = digest(path), pixels(a)
        require((raw, pixel) == (row['sha256'], row['pixel_sha256']), 'Photo hash mismatch')
        require(raw not in old_raw|seen_raw and pixel not in old_pixels|seen_pixels, 'Duplicate prior/new photo')
        seen_raw.add(raw); seen_pixels.add(pixel)
        require(a.shape == (row['height'],row['width'],3), 'Dimension mismatch')
        require(row['source']['license_id'] in LICENSE_IDS, 'Unsupported source license')
        if 'split' not in row:
            chosen = anns[row['annotation_id']]
            require(chosen['image_id']==row['image_id'] and chosen['category_id']==row['category_id'], 'Reference binding')
            require(xyxy(chosen)==row['bbox_xyxy'] and eligible(chosen,row,reference=True), 'Invalid enrollment box')
        else:
            target = [a for a in by_image[row['image_id']] if a['category_id']==row['category_id']]
            noncrowd = [a for a in target if not a.get('iscrowd',0)]
            require(row['gt_annotation_ids']==[a['id'] for a in noncrowd], 'Incomplete target annotations')
            require(row['gt_boxes_xyxy']==[xyxy(a) for a in noncrowd], 'Target box mismatch')
            require(any(eligible(a,row) for a in noncrowd) if row['present'] else not target, 'Presence mismatch')
    counts=Counter((r['category_id'],r['split'],r['present']) for r in manifest['queries'])
    for c in manifest['categories']:
        for split, n in [('dev',4),('test',8)]:
            for present in [True,False]:
                require(counts[c['category_id'],split,present]==n,'Unbalanced category/split/presence')
    # Verify the complete old inventory remains stable, not only stored claims.
    for row in prior['files']:
        require(digest(WORKSPACE/row['path'])==row['sha256'], 'Prior stored/source image changed')
    admission=read(EXPERIMENTS/'real_photo_fetch_admission.json')
    require(admission['data_freeze_sha256']==digest(EXPERIMENTS/'real_photo_data_freeze.json'), 'Admission binding mismatch')
    audit={'valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),
           'manifest_sha256':digest(DATA/'manifest.json'),'plan_sha256':digest(DATA/'plan.json'),
           'data_freeze_sha256':digest(EXPERIMENTS/'real_photo_data_freeze.json'),
           'fetch_admission_sha256':digest(EXPERIMENTS/'real_photo_fetch_admission.json'),
           'prior_inventory_sha256':digest(DATA/'prior_image_inventory.json'),
           'source_licenses_sha256':digest(DATA/'source_licenses.json'),
           'download_provenance_sha256':digest(DATA/'download_provenance.json'),
           'new_photos_verified':300,'prior_files_verified':len(prior['files']),
           'excluded_prior_coco_ids':len(prior['excluded_coco_image_ids']),
           'source_id_overlap':[],'raw_hash_overlap':[],'decoded_pixel_overlap':[],
           'unique_photos_across_references_categories_dev_test':True,
           'counts':manifest['counts'],'budget':manifest['budget'],
           'all_matching_noncrowd_boxes_preserved':True,
           'all_exif_orientations_missing_or_one':True,
           'query_annotations_for_evaluation_only':True,'no_model_runs':True,
           'photo_license_counts':dict(Counter(r['source']['license_id'] for r in rows)),
           'limitations':manifest['limits']}
    write_new(EXPERIMENTS/'real_photo_data_audit.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','fetch','audit'],required=True)
    args=parser.parse_args()
    {'plan':plan_stage,'fetch':fetch_stage,'audit':audit_stage}[args.stage]()
