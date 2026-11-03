"""Freeze prior event/time-group exposure using identity metadata only."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import csv,json,pickletools,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1]
DATA=ROOT/'data/gravity_spy_v15_final';PRIOR=WORKSPACE/'public/HyperNet/benchmarks/gravity_spy_v0.1'
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    metadata=ROOT/'data/gravity_spy_official/trainingset_v1d1_metadata.csv'
    with metadata.open(encoding='utf-8-sig',newline='') as stream:source={r['gravityspy_id']:r for r in csv.DictReader(stream)}
    # Inspect pickle opcodes as data, never unpickle/execute an object array.
    with zipfile.ZipFile(PRIOR/'scores.npz') as archive,archive.open('haystack_gravityspy_ids.npy') as stream:
        version=np.lib.format.read_magic(stream)
        shape,fortran,dtype=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(stream)
        ids=[arg for op,arg,pos in pickletools.genops(stream.read()) if op.name in ('SHORT_BINUNICODE','BINUNICODE','UNICODE') and isinstance(arg,str) and arg in source]
    assert shape==(2379,) and dtype.hasobject and len(ids)==len(set(ids))==2379
    old_protocol=read(PRIOR/'protocol.json');old_results=read(PRIOR/'results.json')
    assert old_results['protocol_sha256']==digest(PRIOR/'protocol.json')
    assert Counter(source[i]['sample_type'] for i in ids)=={'validation':1200,'test':1179}
    assert set(ids)=={key for key,row in source.items() if row['sample_type'] in ('validation','test')}
    references={key for values in old_protocol['reference_gravityspy_ids'].values() for key in values}
    assert len(references)==100 and not references.intersection(ids)
    per_reference=[json.loads(line) for line in (PRIOR/'per_reference.jsonl').read_text().splitlines()]
    for row in per_reference:assert row['reference_gravityspy_id']==old_protocol['reference_gravityspy_ids'][row['class']][row['reference']]
    assert {row['reference_gravityspy_id'] for row in per_reference}==references
    assert len(per_reference)==700
    plan=read(DATA/'plan.json');freeze=read(DATA/'roster_freeze.json');quality=read(DATA/'quality_admission.json')
    assert digest(DATA/'plan.json')==freeze['roster_sha256']==quality['roster_sha256']
    metadata_review=read(ROOT/'experiments/gravity_spy_independent_metadata_review.json')
    assert metadata_review['metadata_valid'] and metadata_review['plan_sha256']==digest(DATA/'plan.json')
    events={event['event_id']:event for event in plan['events']};old=set(ids)|references
    assert old.issubset(events)
    groups={events[event_id]['group_id'] for event_id in old}
    affected={event_id for event_id,event in events.items() if event['group_id'] in groups}
    primary=sorted(event_id for event_id,event in events.items() if event['split']=='test' and event_id not in affected)
    exposed_test=sorted(event_id for event_id,event in events.items() if event['split']=='test' and event_id in affected)
    raw_test=sorted(event_id for event_id,event in events.items() if event['split']=='test')
    assert len(primary)==4119 and len(exposed_test)==2212 and set(primary).isdisjoint(exposed_test)
    assert set(primary)|set(exposed_test)==set(raw_test)
    acquired=[read(path) for path in (DATA/'image_records').glob('*.json')]
    assert not any(record['split']=='test' for record in acquired)
    quality_excluded=set(quality['excluded_event_ids'])
    reused_references={label:{'event_id':event_id,'exact_prior_event':event_id in old,'time_group':events[event_id]['group_id']}
        for label,event_id in plan['reference_by_label'].items() if event_id in affected}
    per_target={}
    for label in quality['eligible_target_labels']:
        positive=[key for key in primary if events[key]['label']==label]
        negative=[key for key in primary if events[key]['label']!=label]
        assert positive and negative
        per_target[label]={'positive_events_before_test_quality':len(positive),'negative_events_before_test_quality':len(negative),
            'positive_time_groups':len({events[key]['group_id'] for key in positive}),
            'negative_time_groups':len({events[key]['group_id'] for key in negative})}
    source_artifacts={str((PRIOR/name).resolve()):digest(PRIOR/name) for name in ('protocol.json','scores.npz','per_reference.jsonl','results.json')}
    amendment={'format':'hypernet.gravity_spy.prior_exposure.v15','created_utc':datetime.now(timezone.utc).isoformat(),
        'roster_sha256':digest(DATA/'plan.json'),'quality_admission_sha256':digest(DATA/'quality_admission.json'),
        'prior_source_artifacts':source_artifacts,'audit_script_sha256':digest(__file__),
        'independent_group_review_sha256':digest(ROOT/'experiments/gravity_spy_independent_metadata_review.json'),
        'no_replacement_sampling':True,'original_roster_and_quality_amendment_unchanged':True,
        'test_ids_excluded_before_any_v15_test_inference':True,'v15_test_image_records_present':0,
        'scoring_policy':'fresh_only','secondary_reused_test_download_or_scoring_authorized':False,
        'prior_event_ids':sorted(old),'prior_reference_ids':sorted(references),'prior_query_ids':sorted(ids),
        'excluded_group_ids':sorted(groups),'group_exposed_event_ids':sorted(affected),
        'primary_test_event_ids':primary,'exposed_test_event_ids':exposed_test,
        'counts':{'prior_queries':2379,'prior_references':100,'prior_unique_events':len(old),'prior_time_groups':len(groups),
            'all_v15_group_exposed_events':len(affected),'raw_v15_test_events':len(raw_test),
            'primary_fresh_test_events_before_quality':len(primary),'excluded_exposed_test_events':len(exposed_test),
            'primary_fresh_test_time_groups':len({events[key]['group_id'] for key in primary}),
            'raw_calibration_group_exposed_events':sum(event['split']=='calibration' and key in affected for key,event in events.items()),
            'quality_valid_calibration_group_exposed_events':sum(event['split']=='calibration' and key in affected and key not in quality_excluded for key,event in events.items())},
        'per_target_primary_test_before_quality':per_target,'retained_prior_exposed_references':reused_references,
        'calibration_policy':'Keep all1579 quality-valid calibration events as development, including prior-exposed events; no claim of fresh calibration.',
        'reference_policy':'Keep the fixed one reference per19eligibletarget. Three references have previous event/group exposure; no replacement or freshness claim for references.',
        'test_quality_policy':'Apply only the already frozen exact-constant RGB exclusion to these4119freshTEST events after acquisition. Do not replace excluded events.',
        'reason':'Old public v0.1 scored1secondviews of the same GravitySpy events. New2secondviews are not independent fresh events; expand exclusion to entire same-detector overlapping4second groups.',
        'limitations':['Freshness is relative to the identified prior projectbenchmark and disclosed probe, not proof against all external pretraining exposure.',
            'H1/L1 groups are separate; cross-detector physical dependence may remain.',
            'Prior outcome arrays were not used to choose event exclusions; only complete identity inventories were used.']}
    path=DATA/'prior_exposure_admission.json';write_new(path,amendment)
    report={key:value for key,value in amendment.items() if key not in ('prior_event_ids','prior_reference_ids','prior_query_ids','excluded_group_ids','group_exposed_event_ids','primary_test_event_ids','exposed_test_event_ids')}
    report.update(valid=True,admission_sha256=digest(path),raw_roster_preserved=digest(DATA/'plan.json')==freeze['roster_sha256'])
    out=ROOT/'experiments/gravity_spy_prior_exposure_audit.json';write_new(out,report)
    print(json.dumps({'valid':True,'amendment_sha256':digest(path),'audit_sha256':digest(out),'counts':amendment['counts'],
        'fresh_positive_counts':{key:value['positive_events_before_test_quality'] for key,value in per_target.items()},
        'retained_prior_exposed_references':reused_references}))
if __name__=='__main__':main()
