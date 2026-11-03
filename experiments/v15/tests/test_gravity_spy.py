import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hypernet.gravity_spy import group_events, build_plan, rare_event_metrics, wilson_interval
import pytest


def sample(event_id, t, label='A', ifo='H1'):
    return {'gravityspy_id': str(event_id), 'event_time': str(t), 'label': label, 'ifo': ifo,
        'sample_type': 'train', **{f'url{i}': f'https://example.invalid/{event_id}_{i}' for i in range(1, 5)}}


def test_overlap_transitive_groups_preserve_detectors():
    rows=[sample('a',0),sample('b',3),sample('c',6),sample('d',11),sample('e',0,ifo='L1')]
    mapping, groups=group_events(rows)
    assert mapping['a']==mapping['b']==mapping['c']
    assert len({mapping['a'],mapping['d'],mapping['e']})==3
    assert len(groups)==3


def test_reference_neighbors_excluded_and_event_splits_deterministic():
    rows=[sample(i,10*i,label='A' if i%2 else'B') for i in range(100)]
    plan=build_plan(rows)
    assert plan==build_plan(list(reversed(rows)))
    assert len(plan['reference_by_label'])==2
    assert all(x['duration_seconds']==2 and x['image_url'].endswith('_3') for x in plan['events'])
    assert all(len(x['all_duration_urls'])==4 for x in plan['events'])
    assert all(x['calibration']['positive_events']>0 for x in plan['per_target'].values())
    groups={}
    for event in plan['events']:
        groups.setdefault(event['group_id'],set()).add(event['split'])
    assert all(len(s)==1 for s in groups.values())
    first=plan['reference_by_label']['A']; row=next(r for r in rows if r['gravityspy_id']==first)
    nearby=sample('neighbor',float(row['event_time'])+.5)
    augmented=build_plan(rows+[nearby])
    chosen=set(augmented['reference_by_label'].values())
    for event in augmented['events']:
        if event['split']=='excluded_reference_neighbor':assert event['event_id']not in chosen


def test_target_background_and_repeated_ids_rejected():
    rows=[sample(i,10*i,label='No_Glitch' if i%3==0 else'A')for i in range(80)]
    assert build_plan(rows)['target_labels']==['A']
    with pytest.raises(ValueError):build_plan(rows+[rows[0]])
    with pytest.raises(ValueError):build_plan(rows,target_labels=['unseen'])
    with pytest.raises(ValueError):group_events(rows,1.)


def test_preexposed_group_cannot_enter_reference_calibration_or_test():
    rows=[sample(i,10*i,label='A'if i%2 else'B')for i in range(100)]
    rows.append(sample('neighbor',1.))
    plan=build_plan(rows,pre_exposed_event_ids=['0'])
    assert all(r['split']=='excluded_pre_exposed_group'for r in plan['events']if r['event_id']in('0','neighbor'))
    assert '0'not in plan['reference_by_label'].values()
    with pytest.raises(ValueError):build_plan(rows,pre_exposed_event_ids=['missing'])


def test_zero_false_alarm_interval_is_not_zero_uncertainty():
    result=rare_event_metrics([True]*10+[False]*100,[True]*5+[False]*105)
    assert result['recall']==.5
    assert result['false_positives_per_10000_negative_events']==0
    assert result['fp_per_10000_binomial_wilson_ci95'][1]>0
    assert 290<result['zero_fp_one_sided95_upper_per10000']<300
    assert wilson_interval(10,10)[0]<1


def test_metrics_reject_silent_count_errors():
    with pytest.raises(ValueError):rare_event_metrics([True,False],[True])
    with pytest.raises(ValueError):rare_event_metrics([True,False],[1,0])
    with pytest.raises(ValueError):rare_event_metrics([False],[False])
