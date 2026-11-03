"""Metadata-only one-reference Gravity Spy protocol; no model or weight fitting.

The upstream v1.1 table has four duration views per event. This adapter uses one
fixed duration and groups temporally overlapping events before any split. Its
custom evaluation split is not the publisher's supervised-learning benchmark.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

DATASET_URL = 'https://zenodo.org/records/1486046'
METADATA_MD5 = 'c16d135f5ad3f029a3ef32114e280e79'
DURATIONS = {'0.5': 'url1', '1.0': 'url2', '2.0': 'url3', '4.0': 'url4'}
BACKGROUND_LABELS = frozenset(('No_Glitch', 'None_of_the_Above'))
OFFICIAL_PLOT_CROP_XYXY = (105, 66, 671, 532)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank(key: str) -> str:
    return sha256(key.encode('utf-8'))


def load_metadata(path: str | Path, verify_official: bool = True) -> list[dict]:
    path = Path(path)
    if verify_official and hashlib.md5(path.read_bytes()).hexdigest() != METADATA_MD5:
        raise ValueError('Official v1.1 metadata checksum mismatch')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty metadata')
    required = {'gravityspy_id', 'label', 'ifo', 'event_time', 'sample_type', *DURATIONS.values()}
    if not required.issubset(rows[0]):
        raise ValueError('Expected v1.1 event table, not machine-predicted catalog labels')
    ids = [row['gravityspy_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated event identifier')
    for row in rows:
        if not row['gravityspy_id'].isalnum() or not row['label'] or row['ifo'] not in ('H1', 'L1'):
            raise ValueError('Invalid event identity or detector')
        if not Decimal(row['event_time']).is_finite():
            raise ValueError('Invalid event time')
        for column in DURATIONS.values():
            url = urlsplit(row[column])
            if url.scheme != 'https' or url.hostname != 'panoptes-uploads.zooniverse.org':
                raise ValueError('Unexpected image host in official metadata')
    return rows


def time_ns(row: dict) -> int:
    if row.get('peak_time') and row.get('peak_time_ns'):
        return int(row['peak_time']) * 1_000_000_000 + int(row['peak_time_ns'])
    return int(Decimal(row['event_time']) * 1_000_000_000)


def group_events(rows: list[dict], overlap_seconds: float = 4.0) -> tuple[dict, dict]:
    """Same-detector connected clusters of overlapping centered maximum views.

This prevents duration siblings and nearby overlapping scans crossing splits.
It does not prove independence of distant triggers or simultaneous detectors.
"""
    if not math.isfinite(overlap_seconds) or overlap_seconds < 4:
        raise ValueError('Grouping must cover all four-second views')
    width_ns = int(Decimal(str(overlap_seconds)) * 1_000_000_000)
    by_detector = defaultdict(list)
    for row in rows:
        by_detector[row['ifo']].append(row)
    mapping, groups = {}, {}
    for detector, events in sorted(by_detector.items()):
        current = []
        previous = None
        def save():
            if not current:
                return
            members = sorted(row['gravityspy_id'] for row in current)
            key = detector + '_' + rank('|'.join(members))[:20]
            groups[key] = members
            for event_id in members:
                mapping[event_id] = key
        for row in sorted(events, key=lambda item: (time_ns(item), item['gravityspy_id'])):
            now = time_ns(row)
            if previous is not None and now - previous > width_ns:
                save()
                current = []
            current.append(row)
            previous = now
        save()
    return mapping, groups


def build_plan(rows: list[dict], *, seed: int = 1517, duration: str = '2.0',
               calibration_fraction: float = 0.2, overlap_seconds: float = 4.0,
               target_labels: list[str] | None = None,
               pre_exposed_event_ids: list[str] | None = None) -> dict:
    if duration not in DURATIONS or not 0 < calibration_fraction < 1:
        raise ValueError('Invalid duration or calibration fraction')
    if len({r['gravityspy_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated event identifier')
    labels = sorted({row['label'] for row in rows})
    targets = sorted(set(target_labels if target_labels is not None else set(labels) - BACKGROUND_LABELS))
    if not targets or not set(targets).issubset(labels):
        raise ValueError('Unknown or empty targets')
    groups_by_id, groups = group_events(rows, overlap_seconds)
    pre_exposed = sorted(set(pre_exposed_event_ids or []))
    if not set(pre_exposed).issubset(groups_by_id):
        raise ValueError('Unknown exposed event identifier')
    exposed_groups = {groups_by_id[event_id] for event_id in pre_exposed}
    chosen, support_groups = {}, set()
    for label in targets:
        candidates = sorted((r for r in rows if r['label'] == label and r['sample_type'] == 'train'),
            key=lambda r: rank(f'{seed}:reference:{label}:{r["gravityspy_id"]}'))
        candidates = [r for r in candidates if groups_by_id[r['gravityspy_id']] not in support_groups | exposed_groups]
        if not candidates:
            raise ValueError(f'No independent publisher-training reference for {label}')
        selected = candidates[0]
        chosen[label] = selected['gravityspy_id']
        support_groups.add(groups_by_id[selected['gravityspy_id']])
    splits = {}
    for group in groups:
        if group in exposed_groups:
            splits[group] = 'excluded_pre_exposed_group'
        elif group in support_groups:
            splits[group] = 'reference_or_excluded_neighbor'
        else:
            uniform = int(rank(f'{seed}:split:{group}'), 16) / 2**256
            splits[group] = 'calibration' if uniform < calibration_fraction else 'test'
    reference_ids = set(chosen.values())
    events = []
    for row in sorted(rows, key=lambda r: r['gravityspy_id']):
        event_id = row['gravityspy_id']
        group = groups_by_id[event_id]
        split = 'reference' if event_id in reference_ids else splits[group]
        if split == 'reference_or_excluded_neighbor':
            split = 'excluded_reference_neighbor'
        events.append({'event_id': event_id, 'group_id': group, 'label': row['label'],
            'ifo': row['ifo'], 'peak_time_ns': time_ns(row), 'publisher_split': row['sample_type'],
            'split': split, 'duration_seconds': float(duration), 'image_url': row[DURATIONS[duration]],
            'all_duration_urls': {key: row[column] for key, column in DURATIONS.items()},
            'image_path': f'images/{event_id}_{duration}.png', 'source_url': DATASET_URL})
    summary = {}
    for label in targets:
        summary[label] = {}
        for split in ('calibration', 'test'):
            subset = [r for r in events if r['split'] == split]
            positives = [r for r in subset if r['label'] == label]
            negatives = [r for r in subset if r['label'] != label]
            if not positives or not negatives:
                raise ValueError(f'Empty positive or negative {split} for {label}; do not silently change seed')
            summary[label][split] = {'positive_events': len(positives), 'negative_events': len(negatives),
                'positive_time_groups': len({r['group_id'] for r in positives}),
                'negative_time_groups': len({r['group_id'] for r in negatives})}
    return {'format': 'hypernet.gravity_spy.one_reference.v15', 'status': 'metadata_plan_not_scored',
        'dataset_url': DATASET_URL, 'dataset_version': 'v1.1.0', 'license': 'CC-BY-4.0',
        'seed': seed, 'target_labels': targets, 'reference_by_label': chosen,
        'pre_exposed_event_ids': pre_exposed,
        'exposure_exclusion': 'Entire time groups containing an already inspected acquisition probe are excluded.',
        'duration_seconds': float(duration), 'plot_crop_xyxy': list(OFFICIAL_PLOT_CROP_XYXY),
        'group_overlap_seconds': overlap_seconds, 'calibration_fraction': calibration_fraction,
        'one_real_image_per_reference': True, 'real_image_weight_training': False,
        'calibration_is_additional_data': True,
        'publisher_split_reused_for_custom_evaluation': True,
        'counts': {'metadata_events': len(events), 'time_groups': len(groups),
            'split_events': dict(Counter(r['split'] for r in events)),
            'split_groups': {split: len({r['group_id'] for r in events if r['split'] == split})
                for split in sorted({r['split'] for r in events})}},
        'per_target': summary, 'events': events,
        'limitations': ['Curated triggered-glitch benchmark, not a random continuous-noise stream.',
            'Nonoverlapping groups are distinct sampling units, not a proof of physical independence.',
            'All target tasks reuse query events; pooled task-event rows are not independent negatives.',
            'Pattern labels are provided annotations, not verified astrophysical event discoveries.',
            'No claim that four duration views create four independent events.']}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError('Invalid binomial counts')
    p = successes / total
    divisor = 1 + z*z/total
    center = (p + z*z/(2*total))/divisor
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/divisor
    return [max(0., center-radius), min(1., center+radius)]


def rare_event_metrics(present: list[bool], accepted: list[bool]) -> dict:
    """Descriptive counts + binomial intervals; clustered CIs belong upstream."""
    if len(present) != len(accepted) or any(type(v) is not bool for v in present + accepted):
        raise ValueError('Aligned strict boolean decisions required')
    positives = sum(present); negatives = len(present)-positives
    if not positives or not negatives:
        raise ValueError('Both positive and negative events required')
    tp = sum(p and a for p, a in zip(present, accepted))
    fp = sum(not p and a for p, a in zip(present, accepted))
    return {'positive_events': positives, 'negative_events': negatives,
        'true_positives': tp, 'false_positives': fp, 'recall': tp/positives,
        'false_positives_per_10000_negative_events': 10000*fp/negatives,
        'recall_binomial_wilson_ci95': wilson_interval(tp, positives),
        'fp_per_10000_binomial_wilson_ci95': [10000*x for x in wilson_interval(fp, negatives)],
        'zero_fp_one_sided95_upper_per10000': 10000*(-math.expm1(math.log(.05)/negatives)) if fp == 0 else None,
        'interval_scope': 'Binomial descriptive approximation; use event/time-group resampling for dependence.'}


def write_plan(metadata: str | Path, output: str | Path, **kwargs) -> dict:
    metadata, output = Path(metadata), Path(output)
    if output.exists():
        raise FileExistsError('Preserve plans; choose a new output')
    plan = build_plan(load_metadata(metadata), **kwargs)
    plan['metadata_sha256'] = sha256(metadata.read_bytes())
    plan['adapter_sha256'] = sha256(Path(__file__).read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding='utf-8')
    return plan
