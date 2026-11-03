"""Freeze metadata and acquire official one-duration images without test leakage."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import sys
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet.gravity_spy import write_plan, sha256, OFFICIAL_PLOT_CROP_XYXY


def digest(path):
    return sha256(Path(path).read_bytes())


def validate_test_admission(plan_path, admission_path):
    if not admission_path:
        raise ValueError('TEST pixels are sealed; provide the frozen artifact admission')
    from hypernet.gravity_spy_admission import verify_test_admission
    return verify_test_admission(Path(plan_path).parent, admission_path)


def fetch_one(event, destination):
    from PIL import Image
    url = event['image_url']
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'panoptes-uploads.zooniverse.org':
        raise ValueError('Only the official metadata image host is accepted')
    raw_path = destination/'raw'/Path(event['image_path']).name
    crop_path = destination/event['image_path']
    sidecar = destination/'image_records'/f'{event["event_id"]}.json'
    for path in (raw_path, crop_path, sidecar):
        path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists():
        prior = json.loads(sidecar.read_text(encoding='utf-8'))
        if prior['image_url'] != url or digest(raw_path) != prior['raw_sha256'] or digest(crop_path) != prior['crop_png_sha256']:
            raise ValueError('Existing image provenance no longer matches')
        return prior
    data = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                data = response.read(4_000_001)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.25 * (attempt+1))
    if len(data) > 4_000_000:
        raise ValueError('Unexpected image size')
    with Image.open(io.BytesIO(data)) as source:
        if source.size != (800, 600):
            raise ValueError(f'Unexpected plot dimensions {source.size}; no automatic adaptive crop')
        rgb = source.convert('RGB')
        cropped = rgb.crop(OFFICIAL_PLOT_CROP_XYXY)
    raw_path.write_bytes(data)
    cropped.save(crop_path)
    record = {**event, 'raw_path': raw_path.relative_to(destination).as_posix(),
        'raw_sha256': sha256(data), 'raw_bytes': len(data), 'raw_dimensions': [800, 600],
        'crop_png_sha256': digest(crop_path), 'crop_rgb_sha256': sha256(cropped.tobytes()),
        'crop_dimensions': list(cropped.size), 'crop_xyxy': list(OFFICIAL_PLOT_CROP_XYXY),
        'crop_note': 'Publisher fixed plot-only crop removes axes, title, ticks and color bar; no resizing.',
        'license': 'CC-BY-4.0', 'label_source': 'curated Gravity Spy v1.1 metadata, not 2021 ML predictions'}
    sidecar.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'fetch'))
    parser.add_argument('--metadata', type=Path, default=ROOT/'data/gravity_spy_official/trainingset_v1d1_metadata.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'data/gravity_spy_v15')
    parser.add_argument('--split', choices=('reference', 'calibration', 'test'), default='reference')
    parser.add_argument('--limit', type=int, default=0, help='Zero means all frozen events in this split')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test-admission', type=Path)
    parser.add_argument('--exclude-exposed-event', action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/'plan.json'
    if args.action == 'plan':
        result = write_plan(args.metadata, plan_path, pre_exposed_event_ids=args.exclude_exposed_event)
        freeze = {'format': 'hypernet.gravity_spy.roster_freeze.v15',
            'roster_sha256': digest(plan_path), 'metadata_sha256': result['metadata_sha256'],
            'adapter_sha256': result['adapter_sha256'], 'acquisition_script_sha256': digest(__file__),
            'created_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'model_selection_started': False, 'test_pixels_downloaded': False,
            'crop_fixed_before_test': list(OFFICIAL_PLOT_CROP_XYXY)}
        (args.output/'roster_freeze.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
        print(json.dumps(result['counts'], indent=2))
        print('ROSTER_SHA256', freeze['roster_sha256'])
        return
    if args.workers not in range(1, 9) or args.limit < 0:
        raise ValueError('Bound workers to 1..8 and nonnegative limit')
    freeze = json.loads((args.output/'roster_freeze.json').read_text(encoding='utf-8'))
    if digest(plan_path) != freeze['roster_sha256']:
        raise ValueError('Frozen roster changed')
    if args.split == 'test':
        validate_test_admission(plan_path, args.test_admission)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    events = [row for row in plan['events'] if row['split'] == args.split]
    if args.split=='test':
        prior=json.loads((args.output/'prior_exposure_admission.json').read_text(encoding='utf-8'))
        primary=set(prior['primary_test_event_ids'])
        events=[row for row in events if row['event_id']in primary]
    if args.limit:
        events = events[:args.limit]
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch_one, row, args.output): row for row in events}
        for job in as_completed(jobs):
            row = jobs[job]
            try:
                records.append(job.result())
            except Exception as error:
                failures.append({'event_id': row['event_id'], 'error': str(error)})
            if (len(records)+len(failures)) % 100 == 0:
                print(f'{args.split}: {len(records)} acquired, {len(failures)} failures / {len(events)}', flush=True)
    report = {'roster_sha256': freeze['roster_sha256'], 'split': args.split, 'requested': len(events),
        'downloaded_and_verified': len(records), 'failures': failures, 'no_replacement_sampling': True,
        'event_ids': sorted(row['event_id'] for row in records),
        'raw_bytes': sum(row['raw_bytes'] for row in records),
        'test_admission_sha256': digest(args.test_admission) if args.test_admission else None}
    report_path = args.output/f'acquisition_{args.split}_{len(events)}.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'event_ids'}, indent=2))
    if failures:
        raise RuntimeError('Some frozen events were unavailable; no replacement or resampling performed')


if __name__ == '__main__':
    main()
