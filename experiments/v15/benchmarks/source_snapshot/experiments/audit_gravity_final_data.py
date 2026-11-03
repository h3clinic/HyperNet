"""Independent final identity, raw/cropped pixel and exact-constant QC audit.

No model, scores, production loader, or quality-classification helper is used.
Run only after the admitted acquisition and its blind TEST QC are complete.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/gravity_spy_v15_final'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)
def safe(value):
    path=(DATA/value).resolve()
    require(path.is_relative_to(DATA.resolve()),'Data path escaped root')
    return path

def inspect_image(item):
    event,record_path=item;record=read(record_path);event_id=event['event_id']
    for key,value in event.items():require(record.get(key)==value,'Sidecar metadata differs '+event_id+' '+key)
    raw=safe(record['raw_path']);crop=safe(record['image_path'])
    require(digest(raw)==record['raw_sha256'],'Raw hash differs '+event_id)
    require(digest(crop)==record['crop_png_sha256'],'Crop file hash differs '+event_id)
    require(raw.stat().st_size==record['raw_bytes'],'Raw byte count differs '+event_id)
    require(record['crop_xyxy']==[105,66,671,532],'Crop changed '+event_id)
    with Image.open(raw)as image:
        require(image.size==(800,600),'Raw dimensions differ '+event_id)
        expected=np.asarray(image.convert('RGB').crop((105,66,671,532)))
    with Image.open(crop)as image:actual=np.asarray(image.convert('RGB'))
    require(actual.shape==(466,566,3)and np.array_equal(actual,expected),'Crop pixels differ '+event_id)
    pixel_hash=sha256(actual.tobytes()).hexdigest()
    require(pixel_hash==record['crop_rgb_sha256'],'RGB hash differs '+event_id)
    constant=bool(np.all(actual==actual[0,0]))
    return {'event_id':event_id,'group_id':event['group_id'],'split':event['split'],'label':event['label'],
        'constant':constant,'constant_rgb':actual[0,0].tolist()if constant else None,
        'crop_rgb_sha256':pixel_hash,'raw_path':record['raw_path'],'image_path':record['image_path'],
        'sidecar_sha256':digest(record_path)}

def main():
    start=time.perf_counter();output=ROOT/'experiments/gravity_spy_final_data_independent_audit.json'
    require(not output.exists(),'Preserve existing independent audit')
    names=['plan.json','roster_freeze.json','prior_exposure_admission.json','quality_admission.json',
        'quality_test_audit.json','acquisition_reference_20.json','acquisition_calibration_1608.json',
        'acquisition_test_4119.json']
    before={DATA/name:digest(DATA/name)for name in names}
    admission_path=ROOT/'experiments/gravity_spy_test_admission.json';before[admission_path]=digest(admission_path)
    admission=read(admission_path);protocol_path=(ROOT/admission['protocol_path']).resolve()
    require(digest(protocol_path)==admission['protocol_sha256'],'Protocol binding changed')
    protocol=read(protocol_path);before[protocol_path]=admission['protocol_sha256']
    require(protocol['model_selection_complete']and not protocol['test_scores_computed_before_freeze'],'Invalid protocol state')
    for field in ('behavior_files','checkpoint_files'):
        for name,expected in protocol[field].items():
            path=(ROOT/name).resolve();require(digest(path)==expected,'Frozen source/model changed '+name);before[path]=expected
    plan=read(DATA/'plan.json');roster=read(DATA/'roster_freeze.json')
    quality=read(DATA/'quality_admission.json');testqc=read(DATA/'quality_test_audit.json')
    exposure=read(DATA/'prior_exposure_admission.json');acquired=read(DATA/'acquisition_test_4119.json')
    require(roster['roster_sha256']==digest(DATA/'plan.json'),'Roster changed')
    for obj in (admission,protocol,quality,testqc,exposure):require(obj['roster_sha256']==roster['roster_sha256'],'Roster binding differs')
    for obj in (admission,protocol,testqc,exposure):require(obj['quality_admission_sha256']==digest(DATA/'quality_admission.json'),'QC binding differs')
    for obj in (admission,protocol):require(obj['prior_exposure_admission_sha256']==digest(DATA/'prior_exposure_admission.json'),'Prior exposure binding differs')
    require(testqc['test_admission_sha256']==digest(admission_path)==acquired['test_admission_sha256'],'TEST gate binding differs')
    require(testqc['rule']==quality['rule']=='exact_constant_rgb_plot_v1','QC rule changed')
    for obj in (quality,testqc):
        require(obj['raw_roster_unchanged']and not obj['replacement_sampling']and not obj['model_scores_used'],'QC selected by outcome')
        require(obj['quality_source_sha256']==digest(ROOT/'hypernet/gravity_spy_quality.py'),'QC source changed')
    events={row['event_id']:row for row in plan['events']}
    require(len(events)==len(plan['events']),'Duplicate metadata IDs')
    for path,expected in exposure['prior_source_artifacts'].items():
        require(digest(path)==expected,'Previous benchmark identity artifact changed');before[Path(path)]=expected
    prior_ids=set(exposure['prior_event_ids'])
    prior_groups={events[event]['group_id']for event in prior_ids}
    require(prior_groups==set(exposure['excluded_group_ids']),'Prior group inventory differs')
    all_test={key for key,event in events.items()if event['split']=='test'}
    fresh={key for key in all_test if events[key]['group_id']not in prior_groups}
    require(fresh==set(exposure['primary_test_event_ids'])and len(fresh)==4119,'Fresh identity filter differs')
    require(all_test-fresh==set(exposure['exposed_test_event_ids']),'Exposed TEST inventory differs')
    require(exposure['scoring_policy']=='fresh_only'and exposure['secondary_reused_test_download_or_scoring_authorized']is False,'Reused TEST authorized unexpectedly')
    require(not acquired['failures']and acquired['requested']==acquired['downloaded_and_verified']==4119,'Incomplete fresh acquisition')
    require(set(acquired['event_ids'])==fresh and len(acquired['event_ids'])==4119 and acquired['no_replacement_sampling'],'Acquired roster differs')
    development={key for key,event in events.items()if event['split']in('reference','calibration')}
    expected_cache=fresh|development
    sidecars={path.stem:path for path in(DATA/'image_records').glob('*.json')}
    require(set(sidecars)==expected_cache,'Cache has missing or unapproved events')
    require(not(set(sidecars)&(all_test-fresh)),'Prior-exposed TEST pixels cached')
    with ThreadPoolExecutor(max_workers=4)as pool:
        inspected=list(pool.map(inspect_image,((events[key],sidecars[key])for key in sorted(sidecars))))
    indexed={row['event_id']:row for row in inspected}
    require({path.relative_to(DATA).as_posix()for path in(DATA/'raw').glob('*.png')}=={row['raw_path']for row in inspected},'Unexpected raw cache file')
    require({path.relative_to(DATA).as_posix()for path in(DATA/'images').glob('*.png')}=={row['image_path']for row in inspected},'Unexpected cropped cache file')
    for report,expected_ids in ((quality,development),(testqc,fresh)):
        records={row['event_id']:row for row in report['records']}
        require(set(records)==expected_ids and len(records)==len(report['records']),'QC roster differs')
        invalid={key for key in expected_ids if indexed[key]['constant']}
        require(set(report['excluded_event_ids'])==invalid,'Exact-constant QC differs')
        require(report['requested_events']==len(expected_ids)and report['eligible_events']==len(expected_ids)-len(invalid),'QC totals differ')
        for key,row in records.items():
            actual=indexed[key]
            require(row['eligible']==(not actual['constant'])and row['constant_rgb']==actual['constant_rgb'],'QC decision differs '+key)
            require(row['label']==actual['label']and row['split']==actual['split'],'QC metadata differs '+key)
    retained=fresh-set(testqc['excluded_event_ids'])
    calibration={key for key in development if events[key]['split']=='calibration'}-set(quality['excluded_event_ids'])
    refs={key for key in development if events[key]['split']=='reference'}-set(quality['excluded_event_ids'])
    labels=quality['eligible_target_labels'];require(len(labels)==19 and len(refs)==19,'Target count differs')
    for label in labels:require(plan['reference_by_label'][label]in refs,'Reference changed')
    group_sets=[{events[key]['group_id']for key in ids}for ids in (refs,calibration,retained)]
    require(not(group_sets[0]&group_sets[1]or group_sets[0]&group_sets[2]or group_sets[1]&group_sets[2]),'Split time-group leakage')
    require(not(group_sets[2]&prior_groups),'Retained TEST time-group exposure')
    per_target={}
    groups=defaultdict(list)
    for key in retained:groups[events[key]['group_id']].append(key)
    for label in labels:
        positives={key for key in retained if events[key]['label']==label}
        positive_groups={events[key]['group_id']for key in positives}
        mixed=sum(any(events[key]['label']==label for key in members)and any(events[key]['label']!=label for key in members)for members in groups.values())
        per_target[label]={'positive_events':len(positives),'negative_events':len(retained)-len(positives),
            'positive_time_groups':len(positive_groups),'wholly_negative_time_groups':len(groups)-len(positive_groups),'mixed_label_groups':mixed}
    duplicates=defaultdict(list)
    for row in inspected:
        if not row['constant']:duplicates[row['crop_rgb_sha256']].append(row['event_id'])
    duplicated={key:ids for key,ids in duplicates.items()if len(ids)>1}
    cross_split=[ids for ids in duplicated.values()if len({events[key]['split']for key in ids})>1]
    for path,expected in before.items():require(digest(path)==expected,'Bound artifact changed during audit '+str(path))
    result={'format':'hypernet.gravity_spy.final_data_independent_audit.v15','valid':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':digest(__file__),
        'elapsed_seconds':time.perf_counter()-start,'bound_artifacts':{str(path):value for path,value in before.items()},
        'checks':{'all_raw_and_crop_hashes':True,'raw_crop_exact_pixel_equivalence':True,'no_unapproved_test_pixels':True,
            'fresh_whole_time_groups':True,'exact_constant_rule_independently_recomputed':True,'all_unavailable_ids_preserved_no_replacement':True,
            'reference_calibration_test_group_disjoint':True,'source_model_protocol_hashes_unchanged':True},
        'counts':{'acquired_records':len(inspected),'fresh_test_before_quality':len(fresh),'quality_excluded_test':len(fresh-retained),
            'eligible_fresh_test':len(retained),'eligible_fresh_time_groups':len(groups),'eligible_calibration_events':len(calibration),
            'eligible_reference_images':len(refs),'previously_exposed_test_excluded_and_not_cached':len(all_test-fresh)},
        'per_target':per_target,'fresh_test_labels':dict(sorted(Counter(events[key]['label']for key in retained).items())),
        'nonconstant_identical_pixel_groups':duplicated,'nonconstant_identical_pixels_across_splits':cross_split,
        'retained_development_exposure':{'calibration_events_in_prior_groups':sum(events[key]['group_id']in prior_groups for key in calibration),
            'reference_events_in_prior_groups':sorted(key for key in refs if events[key]['group_id']in prior_groups)},
        'scope':'Mechanical identity/pixel/QC audit, no model calls, prediction scores, UI inspection, or new exclusion rules. Nonconstant exact duplicate groups, if any, are disclosed without posthoc removal.'}
    with output.open('x',encoding='utf-8')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'path':str(output),'sha256':digest(output),'counts':result['counts'],'cross_split_duplicate_groups':len(cross_split)}))
if __name__=='__main__':main()
