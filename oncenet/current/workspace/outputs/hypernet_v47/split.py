"""Freeze metadata-only v47 FIT/TUNE/CAL roles among already consumed events.

No score, feature array, image, model, or reserved pixel is read.  The original
four FIT512 sets, reference events, CAL and DEV remain unchanged.  Only unused
FIT-role groups are assigned to TUNE or additional negative calibration.
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parent
RECORDS = P / 'hypernet_v46/data/records.json'
FITS = P / 'hypernet_v46/evaluation/fits.json'
INPUT_BINDING = P / 'hypernet_v46/data/inputs_binding.json'
RESERVE = P / 'hypernet_v45/reserve_metadata_audit.json'
TARGETS = ['Low_Frequency_Burst', 'Low_Frequency_Lines', 'Scattered_Light', 'Blip']
REFERENCE_IDS = {
    'Low_Frequency_Burst': '632xGf8n9x',
    'Low_Frequency_Lines': 'Vengh4mj2t',
    'Scattered_Light': 'dac0D6ni5R',
    'Blip': 'yxumvoPABn',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_basename(row):
    return row['source_csv'].replace('\\', '/').split('/')[-1]


def counts(rows, indices):
    indices = list(indices)
    def by(field):
        return dict(sorted(collections.Counter(rows[i][field] for i in indices).items()))
    return {
        'n': len(indices),
        'label': by('label'),
        'ifo': by('ifo'),
        'source_csv': dict(sorted(collections.Counter(source_basename(rows[i]) for i in indices).items())),
        'label_ifo_source': [
            dict(label=k[0], ifo=k[1], source_csv=k[2], n=n)
            for k, n in sorted(collections.Counter(
                (rows[i]['label'], rows[i]['ifo'], source_basename(rows[i])) for i in indices
            ).items())
        ],
    }


def build():
    rows, fits, reserve = read(RECORDS), read(FITS), read(RESERVE)
    assert len(rows) == 8250
    assert len({r['event_id'] for r in rows}) == len(rows)
    assert all(r['index'] == i for i, r in enumerate(rows)), 'Use compact v46 row positions'
    reserved = reserve['records']
    assert len(reserved) == reserve['remaining_after_v44_ids_and_eight_seconds'] == 195
    reserve_source = Path(reserve['original_candidate_path'])
    assert sha(reserve_source) == reserve['original_candidate_sha256']
    reserve_ids = {r['event_id'] for r in reserved}
    assert not reserve_ids.intersection(r['event_id'] for r in rows)
    reserve_times = {
        ifo: sorted(float(r['event_time']) for r in reserved if r['ifo'] == ifo)
        for ifo in ['H1', 'L1']
    }
    minimum_reserve_gap = float('inf')
    for r in rows:
        times = reserve_times[r['ifo']]
        at = bisect.bisect_left(times, float(r['event_time']))
        for j in [at - 1, at]:
            if 0 <= j < len(times):
                gap = abs(times[j] - float(r['event_time']))
                minimum_reserve_gap = min(minimum_reserve_gap, gap)
                assert gap > 8, 'Reserved same-detector time neighbor'

    canonical = fits['old_v37']
    per_target = {}
    for target in TARGETS:
        original = canonical['single__' + target]
        fi = original['fit_indices']
        ri = original['reference_index']
        assert len(fi) == len(set(fi)) == 512
        assert all(rows[i]['split'] == 'fit' and rows[i]['label'] != target for i in fi)
        assert rows[ri]['event_id'] == REFERENCE_IDS[target]
        assert rows[ri]['label'] == target
        assert rows[ri]['split'] in ['reference', 'reference_check']
        for model in fits.values():
            for view in ['single', 'multi']:
                v = model[view + '__' + target]
                assert v['fit_indices'] == fi
                assert v['reference_index'] == ri
                assert v['calibration_indices'] == original['calibration_indices']
        per_target[target] = {'fit': list(fi), 'reference': ri}

    union = sorted({i for v in per_target.values() for i in v['fit']})
    union_groups = {rows[i]['group'] for i in union}
    assert len(union) == len(union_groups) == 540
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if row['split'] == 'fit' and row['group'] not in union_groups:
            groups[row['group']].append(i)
    assert sum(map(len, groups.values())) == len(groups) == 1523
    strata = collections.defaultdict(list)
    for group, indices in groups.items():
        keys = {(rows[i]['label'], rows[i]['ifo'], source_basename(rows[i])) for i in indices}
        assert len(keys) == 1, 'A group spans metadata strata'
        strata[next(iter(keys))].append(group)
    assert len(strata) == 92
    tune, cal_add, stratum_rows = [], [], []
    for key, gs in sorted(strata.items()):
        gs.sort(key=lambda g: hashlib.sha256(('v47-unused-fit-split|' + g).encode()).hexdigest())
        n = len(gs) // 3
        ti = [i for g in gs[:n] for i in groups[g]]
        ca = [i for g in gs[n:] for i in groups[g]]
        tune.extend(ti)
        cal_add.extend(ca)
        stratum_rows.append(dict(label=key[0], ifo=key[1], source_csv=key[2],
                                 groups=len(gs), tune=len(ti), cal_add=len(ca)))

    indices = dict(
        original_fit_union=union,
        tune=sorted(tune),
        cal_add=sorted(cal_add),
        original_cal=[i for i, r in enumerate(rows) if r['split'] == 'calibration'],
        development=[i for i, r in enumerate(rows) if r['split'] == 'development'],
        references=[i for i, r in enumerate(rows) if r['split'] in ['reference', 'reference_check']],
    )
    expected = dict(original_fit_union=540, tune=474, cal_add=1049,
                    original_cal=4608, development=1558, references=21)
    assert {k: len(v) for k, v in indices.items()} == expected
    role_by_row = {}
    role_by_group = {}
    for role, ix in indices.items():
        for i in ix:
            assert i not in role_by_row, 'Row shared between roles'
            role_by_row[i] = role
            group = rows[i]['group']
            assert group not in role_by_group or role_by_group[group] == role, 'Group shared between roles'
            role_by_group[group] = role
    assert set(role_by_row) == set(range(len(rows)))
    minimum_cross_role_gap = float('inf')
    for ifo in ['H1', 'L1']:
        ordered = sorted((float(r['event_time']), i) for i, r in enumerate(rows) if r['ifo'] == ifo)
        # Adjacent changes of role contain the closest pair of different roles.
        for (ta, a), (tb, b) in zip(ordered, ordered[1:]):
            if role_by_row[a] != role_by_row[b]:
                minimum_cross_role_gap = min(minimum_cross_role_gap, tb - ta)
                assert tb - ta > 8, 'Same-detector time neighbor spans roles'

    expanded = sorted(indices['original_cal'] + indices['cal_add'])
    target_counts = {}
    for target, value in per_target.items():
        value['tune'] = [i for i in indices['tune'] if rows[i]['label'] != target]
        value['original_cal'] = [i for i in indices['original_cal'] if rows[i]['label'] != target]
        value['expanded_cal'] = [i for i in expanded if rows[i]['label'] != target]
        assert value['original_cal'] == canonical['single__' + target]['calibration_indices']
        assert all(rows[i]['label'] != target for k in ['fit', 'tune', 'original_cal', 'expanded_cal'] for i in value[k])
        target_counts[target] = {k: counts(rows, value[k]) for k in ['fit', 'tune', 'original_cal', 'expanded_cal']}
    assert [len(per_target[t]['expanded_cal']) for t in TARGETS] == [4958, 4992, 4970, 5008]
    files = [Path(__file__).resolve(), RECORDS, FITS, INPUT_BINDING, RESERVE, reserve_source]
    return dict(
        event_count=len(rows),
        metadata_only=True,
        no_model_outcomes_read=True,
        no_images_or_features_read=True,
        no_new_real_holdout=True,
        index_convention='Compact zero-based row position in v46 data/records.json; never source_index.',
        split_policy=dict(
            preserve='All four original target FIT512 sets, original CAL, DEV and all reference-role events unchanged.',
            eligible='FIT-role groups outside the entire original FIT union; exclude whole groups.',
            strata=['label', 'ifo', 'source_csv basename'],
            group_field='group',
            order="SHA256 UTF8('v47-unused-fit-split|' + group), ascending within each stratum",
            allocation='First floor(n_groups/3) TUNE; remaining groups CAL_ADD.',
            target_negative_only='Exclude the current target label from FIT/TUNE/CAL sets for that target.',
            interpretation='One positive reference event per target plus labeled negative FIT/TUNE/CAL and repeatedly observed DEV; not one total real image.',
        ),
        indices=indices,
        targets=per_target,
        reference_ids=REFERENCE_IDS,
        source_hashes={str(path.resolve()): sha(path) for path in files},
        counts={k: counts(rows, v) for k, v in indices.items()},
        target_counts=target_counts,
        strata=stratum_rows,
        audits=dict(
            passed=True,
            disjointness=True,
            complete_partition=True,
            no_shared_groups=True,
            no_cross_role_eight_second_neighbors=True,
            minimum_cross_role_same_detector_gap_seconds=minimum_cross_role_gap,
            no_reserved=True,
            reserved_metadata_events=len(reserved),
            no_reserved_id_overlap=True,
            no_reserved_eight_second_neighbors=True,
            minimum_reserved_same_detector_gap_seconds=minimum_reserve_gap,
            referenceone=True,
            active_reference_count=4,
            preserved_reference_role_events=21,
            all_original_fit512_sets_unchanged=True,
            all_original_calibration_sets_unchanged=True,
            all_eligible_groups_singleton=True,
            counts=expected,
            combined_background_calibration_negatives=sum(rows[i]['label'] not in TARGETS for i in expanded),
        ),
    )


if __name__ == '__main__':
    destination = R / 'splits.json'
    assert not destination.exists(), 'Frozen splits already exist; do not overwrite'
    result = build()
    # Exclusive creation also prevents an accidental concurrent second writer.
    with destination.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(path=str(destination), sha256=sha(destination),
                         counts=result['audits']['counts'], audits_passed=True)))
