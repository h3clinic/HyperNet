"""Metadata-first, frozen-cohort Gravity Spy acquisition for the factorial study.

plan never opens images. acquire requires an explicit frozen protocol digest.
The authorizing orchestrator must bind this file, intent, labels and data_plan
in protocol.source_hashes before authorizing acquisition. No model code runs.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent
W = R.parents[1]
V16 = W / 'outputs/hypernet_v16/data/gravity_fresh_v16'
F = W / 'outputs/hypernet_v41/data_feasibility'
GS = W / 'outputs/cloud_plan_20261105/gs_fp32/kaggle_pilot'
DUR = dict(d05='url1', d1='url2', d2='url3', d4='url4')
TARGETS = ['Low_Frequency_Burst', 'Low_Frequency_Lines', 'Scattered_Light', 'Blip']
SALT = 'oncenet-factorial-20261105'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_new(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def status(stage, **kw):
    p = R / 'data_status.json'
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(stage=stage, updated_utc=now(), **kw), indent=2))
    t.replace(p)


def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def historical_sources():
    a = read(W / 'outputs/hypernet_v45/reserve_metadata_audit.json')
    paths = [Path(p) for p in a['source_bindings']
             if 'official_metadata' not in p and 'data_feasibility' not in p]
    paths += [W / p for p in read(F / 'summary.json')['excluded_plans']]
    paths += [W / p for p in [
        'outputs/hypernet_v42/used_records.json',
        'outputs/hypernet_v43/used_records.json',
        'outputs/hypernet_v44/used_records.json',
        'outputs/hypernet_v46/data/records.json',
        'outputs/oncenet_confirmation_20261105/records_unlabelled.json',
        'outputs/cloud_plan_20261105/gravity_archive/records.json',
    ]]
    for version in ['hypernet_v15', 'hypernet_v16']:
        paths += list((W / 'outputs' / version).rglob('image_records/*.json'))
    return sorted(set(p.resolve() for p in paths if p.exists()))


def history(paths):
    ids, events = set(), {}
    for p in paths:
        if p.suffix == '.csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                records = list(csv.DictReader(f))
        else:
            records = walk(read(p))
            if p.parent.name == 'image_records':
                ids.add(p.stem)
        for row in records:
            key = row.get('gravityspy_id') or row.get('event_id') or row.get('subject_id')
            if isinstance(key, str) and re.fullmatch('[A-Za-z0-9]{10}', key):
                ids.add(key)
                if row.get('ifo') and row.get('event_time') is not None:
                    try:
                        events[key] = (row['ifo'], float(row['event_time']))
                    except (TypeError, ValueError):
                        pass
    return ids, events


def hour(row):
    return (row['ifo'], math.floor(float(row['event_time']) / 3600))


def order(row):
    return hashlib.sha256((SALT + '|' + row['event_id']).encode()).hexdigest()


def vote_audit(selected):
    """Recount selected metadata from raw registered ballots, read-only SQL."""
    dbpath = V16 / 'private_volunteer_votes.sqlite'
    db = sqlite3.connect(dbpath.resolve().as_uri() + '?mode=ro', uri=True)
    sources = collections.defaultdict(list)
    for row in selected:
        sources[row['consensus_source']].append(row)
    audited = []
    for source, rows in sources.items():
        name = source.removeprefix('candidates_').removesuffix('.json')
        workflow = int(name.split('_')[0])
        version = None if name.endswith('_all_versions') else name.split('_v')[1].replace('p', '.')
        # Index workflow,subject,user makes bounded subject queries efficient.
        for start in range(0, len(rows), 150):
            batch = rows[start:start + 150]
            ids = [r['event_id'] for r in batch]
            query = ('SELECT subject,user,label,version FROM votes WHERE workflow=? '
                     'AND subject IN (' + ','.join('?' for _ in ids) + ')')
            params = [workflow] + ids
            if version is not None:
                query += ' AND version=?'
                params.append(version)
            ballots = collections.defaultdict(lambda: collections.defaultdict(set))
            versions = collections.defaultdict(set)
            for eid, user, label, ver in db.execute(query, params):
                ballots[eid][user].add(label)
                versions[eid].add(ver)
            for row in batch:
                counts = collections.Counter(next(iter(labels)) for labels in ballots[row['event_id']].values()
                                             if len(labels) == 1)
                total = sum(counts.values())
                winning = counts[row['label']]
                assert total == row['registered_users'] and winning == row['winning_votes'], row['event_id']
                assert total >= 5 and winning / total >= .8
                audited.append(dict(event_id=row['event_id'], workflow=workflow,
                                    version=version, observed_versions=sorted(versions[row['event_id']]),
                                    consistent_registered_users=total, winning_votes=winning))
    db.close()
    return audited


def plan(allow_within_workflow_versions=False):
    for name in ['data_plan.json', 'intent_unlabelled.json', 'labels.json']:
        if (R / name).exists():
            raise FileExistsError('Preserve existing plan: ' + name)
    paths = historical_sources()
    ids, events = history(paths)
    candidates = collections.defaultdict(list)
    candidate_paths = []
    for p in sorted(F.glob('candidates_*.json')):
        if '_all_versions' in p.name and not allow_within_workflow_versions:
            continue
        candidate_paths.append(p)
        for row in read(p):
            candidates[row['event_id']].append(dict(row, consensus_source=p.name))
    official_paths = sorted((V16 / 'official_metadata').glob('[HL]1_O*.csv'))
    meta, ambiguous = {}, set()
    for p in official_paths:
        with p.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                eid = row['gravityspy_id']
                if eid in ids:
                    events[eid] = (row['ifo'], float(row['event_time']))
                if eid in candidates:
                    item = dict(event_id=eid, ifo=row['ifo'], event_time=float(row['event_time']),
                                source_csv=p.name, urls={d: row[field] for d, field in DUR.items()})
                    if eid in meta and meta[eid] != item:
                        ambiguous.add(eid)
                    meta[eid] = item
    if ids - set(events):
        raise RuntimeError('Historical event timestamps missing: ' + str(len(ids - set(events))))
    # Also prohibit a next-door hour if a previous event lies within8s of its boundary.
    prior_hours = {(ifo, math.floor((t + delta) / 3600))
                   for ifo, t in events.values() for delta in [-8, 0, 8]}
    remaining, removed = [], collections.Counter()
    for eid, choices in candidates.items():
        if len({row['label'] for row in choices}) != 1:
            removed['conflicting_consensus_labels'] += 1
            continue
        if eid in ids:
            removed['previous_event_id'] += 1
            continue
        if eid not in meta or eid in ambiguous:
            removed['missing_or_ambiguous_official_metadata'] += 1
            continue
        m = meta[eid]
        if '_O3a.csv' not in m['source_csv']:
            removed['not_O3a'] += 1
            continue
        if hour(m) in prior_hours:
            removed['previous_detector_hour_or_boundary_buffer'] += 1
            continue
        if len(set(m['urls'].values())) != 4 or not all(
            u.startswith('https://panoptes-uploads.zooniverse.org/') and u.endswith('.png')
            for u in m['urls'].values()
        ):
            removed['four_authentic_urls_unavailable'] += 1
            continue
        # Choose source by evidence strength, not by any model performance.
        row = min(choices, key=lambda r: (-r['agreement'], -r['registered_users'], r['consensus_source']))
        if row['ifo'] != m['ifo'] or abs(float(row['event_time']) - m['event_time']) > 1e-5:
            raise RuntimeError('Consensus/catalog event identity mismatch')
        remaining.append(dict(row, **m, detector_hour=hour(m)[1]))
    selected, taken = [], set()
    for target in TARGETS:
        count = 0
        for row in sorted((r for r in remaining if r['label'] == target), key=order):
            if hour(row) in taken:
                continue
            selected.append(row)
            taken.add(hour(row))
            count += 1
            if count == 100:
                break
    positive_count = len(selected)
    for row in sorted((r for r in remaining if r['label'] not in TARGETS), key=order):
        if hour(row) in taken:
            continue
        selected.append(row)
        taken.add(hour(row))
        if len(selected) - positive_count == 2000:
            break
    status('metadata_vote_audit', selected=len(selected))
    ballots = vote_audit(selected)
    audit_by_id = {r['event_id']: r for r in ballots}
    intents = [{k: row[k] for k in ['event_id', 'ifo', 'event_time', 'source_csv', 'urls', 'detector_hour']}
               for row in selected]
    labels = [dict(event_id=row['event_id'], label=row['label'], consensus_source=row['consensus_source'],
                   agreement=row['agreement'], **{k: v for k, v in audit_by_id[row['event_id']].items() if k != 'event_id'})
              for row in selected]
    pixel_history = [p for p in paths if p.suffix == '.json']
    pixel_history += [W / p for p in ['gravity_spy_o3/discovery/images.json',
                                    'gravity_spy_o3/cohort/calib_images.json', 'gravity_spy_o3/cohort/test_images.json']]
    sources = sorted(set(paths + candidate_paths + official_paths + pixel_history + [
        V16 / 'private_volunteer_votes.sqlite', V16 / 'official_metadata/classifications.csv.bz2.provenance.json',
        GS / 'preprocess.py', Path(__file__).resolve()]))
    source_hashes = {str(p): sha(p) for p in sources}
    report = dict(version='oncenet_factorial_data_v1', created_utc=now(), metadata_only=True,
                  new_images_opened=False, new_images_downloaded=False, scores_computed=False,
                  selection_salt=SALT, target_order=TARGETS, cap_per_target=100, negative_cap=2000,
                  label_rule='At least5 distinct consistent registered voters, at least80% exact named label; no anonymous/machine votes; no cross-workflow pooling.',
                  within_workflow_versions_allowed=allow_within_workflow_versions,
                  raw_selected_vote_recount_passed=True,
                  rule='O3a only; exclude all known prior IDs and detector-hours, with8s boundary buffer; targets first in declared order then hashed outside-target pool; one event per detector-hour across entire cohort; all4 authentic official URLs required.',
                  historical_event_ids=len(ids), historical_detector_hours=len(prior_hours),
                  eligible_events=len(remaining), eligible_hours=len({hour(r) for r in remaining}),
                  eligible_label_counts=dict(collections.Counter(r['label'] for r in remaining)),
                  removed=dict(removed), intent_count=len(selected),
                  selected_counts=dict(collections.Counter(r['label'] for r in selected)),
                  selected_by_detector={ifo: dict(collections.Counter(r['label'] for r in selected if r['ifo'] == ifo)) for ifo in ['H1','L1']},
                  selected_hour_count=len(taken), positives=positive_count, outside_target_negatives=len(selected)-positive_count,
                  source_urls={'human_ballots':'https://zenodo.org/records/13904422', 'official_event_catalog':'https://zenodo.org/records/5649212'},
                  source_pixel_history_files=[str(p) for p in sorted(set(pixel_history))], source_hashes=source_hashes,
                  limitations=['Volunteer consensus is not expert adjudication; previous ML routing affected volunteer exposure.',
                               'Versions within a workflow can expose different vocabularies; exact named label, user consistency and source version counts are retained.',
                               'Class-stratified sampling changes prevalence; false alarms are per non-target event, not deployment precision.',
                               'Detector-hour separation does not prove physical independence across hours; detector imbalance remains.',
                               'Historical exact pixel hashes are incomplete; identity/time exclusion is primary.'])
    save_new(R / 'intent_unlabelled.json', intents)
    save_new(R / 'labels.json', labels)
    report['output_hashes'] = {str(R / name): sha(R / name) for name in ['intent_unlabelled.json','labels.json']}
    save_new(R / 'data_plan.json', report)
    status('metadata_plan_complete', intent_count=len(selected), counts=report['selected_counts'], acquisition_started=False)
    print(json.dumps({k: report[k] for k in ['intent_count','selected_counts','selected_by_detector','eligible_events','eligible_hours','historical_event_ids','outside_target_negatives']}, indent=2))


def frozen_protocol(digest):
    p = R / 'protocol.json'
    if not digest or sha(p) != digest:
        raise RuntimeError('Explicit expected frozen protocol SHA256 required')
    protocol = read(p)
    bindings = protocol.get('source_hashes', {})
    for file in [Path(__file__).resolve(), R/'intent_unlabelled.json', R/'labels.json', R/'data_plan.json', GS/'preprocess.py']:
        if bindings.get(str(file)) != sha(file):
            raise RuntimeError('Protocol must bind exact input/code: ' + str(file))
    for name, expected in read(R/'data_plan.json')['source_hashes'].items():
        if sha(name) != expected:
            raise RuntimeError('Metadata/source changed since plan: ' + name)
    return protocol


def acquire(digest, workers=10):
    frozen_protocol(digest)
    if (R/'acquisition.json').exists():
        raise FileExistsError('Completed acquisition exists; do not rerun')
    import numpy as np
    from PIL import Image
    spec = importlib.util.spec_from_file_location('factorial_frozen_preprocessing', GS/'preprocess.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    intent = read(R/'intent_unlabelled.json')
    def fetch(job):
        row, duration = job
        dest = R/'raw'/duration/(row['event_id']+'.png')
        dest.parent.mkdir(parents=True, exist_ok=True)
        errors=[]
        for attempt in range(3):
            try:
                if not dest.exists():
                    request = urllib.request.Request(row['urls'][duration], headers={'User-Agent':'OnceNetResearch/1.0'})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        data=response.read(3_000_001)
                    if len(data)>3_000_000:
                        raise ValueError('Native PNG exceeded3MB bound')
                    with dest.open('xb') as f:
                        f.write(data)
                with Image.open(dest) as im:
                    if im.format!='PNG' or im.size!=(800,600):
                        raise ValueError('Expected native800x600 PNG')
                    rgb=np.asarray(im.convert('RGB'))
                renderer=prep.validate_renderer(rgb)
                return dict(event_id=row['event_id'], duration=duration, path=str(dest), url=row['urls'][duration],
                            sha256=sha(dest), pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            crop_pixel_sha256=hashlib.sha256(prep.native_rgb(rgb,'full').tobytes()).hexdigest(),
                            renderer=renderer)
            except Exception as exc:
                errors.append(type(exc).__name__+': '+str(exc))
                # A returned invalid native image is reproducible exclusion, never alter its pixels.
                if dest.exists():
                    break
                if attempt<2:
                    time.sleep(1+attempt)
        return dict(event_id=row['event_id'],duration=duration,url=row['urls'][duration],error=errors)
    jobs=[(r,d) for r in intent for d in DUR]
    bindings=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,j) for j in jobs]
        for future in as_completed(futures):
            bindings.append(future.result())
            if len(bindings)%50==0:
                status('download',done=len(bindings),total=len(jobs),failed=sum('error'in b for b in bindings))
    bindings.sort(key=lambda b:(b['event_id'],b['duration']))
    save_new(R/'source_bindings.json',bindings)
    seen={k:set() for k in ['sha256','pixel_sha256','crop_pixel_sha256']}
    coverage=[]
    for filename in read(R/'data_plan.json')['source_pixel_history_files']:
        counts=collections.Counter()
        for row in walk(read(filename)):
            # Legacy raw_sha256 is the same native-file hash as sha256.
            # Legacy crop_rgb hashes are not assumed to share this crop geometry.
            raw_alias = row.get('raw_sha256')
            if isinstance(raw_alias, str) and re.fullmatch('[0-9a-f]{64}', raw_alias):
                seen['sha256'].add(raw_alias)
                counts['raw_sha256_alias'] += 1
            for key in seen:
                value=row.get(key)
                if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value):
                    seen[key].add(value);counts[key]+=1
                elif key=='sha256' and isinstance(value,dict):
                    for val in value.values():
                        if isinstance(val,str) and re.fullmatch('[0-9a-f]{64}',val):
                            seen[key].add(val);counts[key]+=1
        coverage.append(dict(path=filename,hash_counts=dict(counts)))
    prior_counts={key:len(v) for key,v in seen.items()}
    byid=collections.defaultdict(dict)
    for b in bindings:
        byid[b['event_id']][b['duration']]=b
    kept=[];excluded=[]
    for row in intent:
        windows=byid[row['event_id']]
        if any(d not in windows or 'error'in windows[d] for d in DUR):
            excluded.append(dict(event_id=row['event_id'],reason='native_download_or_renderer_failure'));continue
        collisions=[dict(duration=d,key=key) for d in DUR for key in seen if windows[d][key]in seen[key]]
        if collisions:
            excluded.append(dict(event_id=row['event_id'],reason='historical_or_prior_selected_exact_pixel_collision',collisions=collisions));continue
        kept.append(dict(row,windows=windows))
        for d in DUR:
            for key in seen:
                seen[key].add(windows[d][key])
    if not kept:
        raise RuntimeError('All intent events excluded; no replacements allowed')
    arrays={}
    (R/'inputs').mkdir(exist_ok=True)
    for d in DUR:
        for view in ['full','low']:
            key=view+'__'+d
            path=R/'inputs'/(key+'.npy')
            if path.exists():
                raise FileExistsError('Preserve existing input array: '+str(path))
            arrays[key]=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(kept),128,128))
    for i,row in enumerate(kept):
        for d in DUR:
            with Image.open(row['windows'][d]['path'])as im:
                rgb=np.asarray(im.convert('RGB'))
            for view in ['full','low']:
                arrays[view+'__'+d][i]=prep.preprocess(rgb,'energy_'+view,128)[...,0]
        if (i+1)%25==0:
            status('preprocess',done=i+1,total=len(kept))
    for a in arrays.values():
        a.flush()
    del arrays
    save_new(R/'records_unlabelled.json',kept)
    files=[R/'records_unlabelled.json',R/'source_bindings.json']+sorted((R/'inputs').glob('*.npy'))
    save_new(R/'acquisition.json',dict(complete=True,completed_utc=now(),intent_count=len(intent),eligible_count=len(kept),
             excluded=excluded,source_pixel_history_counts=prior_counts,source_pixel_history_coverage=coverage,
             all_models_same_cohort=True,no_model_inference=True,protocol_sha256=digest,
             files={str(p):sha(p)for p in files},
             pixel_audit_limitation='Only recorded exact native/crop hashes; all historical IDs and detector-hours excluded; no universal perceptual-duplicate claim.'))
    status('acquisition_complete',intent_count=len(intent),eligible_count=len(kept),excluded_count=len(excluded),model_inference=False)
    print(json.dumps(dict(eligible_count=len(kept),excluded_count=len(excluded))))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['plan','acquire'])
    ap.add_argument('--allow-within-workflow-versions',action='store_true')
    ap.add_argument('--protocol-sha256')
    ap.add_argument('--workers',type=int,choices=range(8,13),default=10)
    args=ap.parse_args()
    if args.action=='plan':
        plan(args.allow_within_workflow_versions)
    else:
        acquire(args.protocol_sha256,args.workers)
