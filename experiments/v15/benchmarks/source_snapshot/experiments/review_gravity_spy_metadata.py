"""Independent metadata-only recomputation; no image decoder or inference imports."""
from collections import Counter,defaultdict
from decimal import Decimal
from hashlib import sha256,md5
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/gravity_spy_v15_final'
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rank(value):return sha256(value.encode()).hexdigest()
def main():
    source=ROOT/'data/gravity_spy_official/trainingset_v1d1_metadata.csv'
    assert md5(source.read_bytes()).hexdigest()=='c16d135f5ad3f029a3ef32114e280e79'
    with source.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    plan=read(DATA/'plan.json');freeze=read(DATA/'roster_freeze.json');assert digest(DATA/'plan.json')==freeze['roster_sha256']
    assert digest(source)==plan['metadata_sha256']
    by_id={row['gravityspy_id']:row for row in rows};assert len(rows)==len(by_id)
    mapping={};groups={}
    for detector in sorted({row['ifo'] for row in rows}):
        timed=[]
        for row in rows:
            if row['ifo']!=detector:continue
            t=(int(row['peak_time'])*10**9+int(row['peak_time_ns'])) if row.get('peak_time') and row.get('peak_time_ns') else int(Decimal(row['event_time'])*10**9)
            timed.append((t,row['gravityspy_id']))
        timed.sort();components=[]
        for t,event_id in timed:
            if not components or t-components[-1][-1][0]>4*10**9:components.append([])
            components[-1].append((t,event_id))
        for component in components:
            members=sorted(event_id for _,event_id in component);group=detector+'_'+rank('|'.join(members))[:20]
            groups[group]=members
            for event_id in members:mapping[event_id]=group
    exposed={mapping[event_id] for event_id in plan['pre_exposed_event_ids']};support_groups=set();supports={}
    for label in plan['target_labels']:
        choices=[row for row in rows if row['label']==label and row['sample_type']=='train' and mapping[row['gravityspy_id']] not in exposed|support_groups]
        selected=min(choices,key=lambda row:rank(f"{plan['seed']}:reference:{label}:{row['gravityspy_id']}"))
        supports[label]=selected['gravityspy_id'];support_groups.add(mapping[selected['gravityspy_id']])
    assert supports==plan['reference_by_label']
    splitgroups={}
    for group in groups:
        if group in exposed:split='excluded_pre_exposed_group'
        elif group in support_groups:split='excluded_reference_neighbor'
        else:split='calibration' if int(rank(f"{plan['seed']}:split:{group}"),16)/2**256<plan['calibration_fraction'] else 'test'
        splitgroups[group]=split
    for event in plan['events']:
        row=by_id[event['event_id']];expected='reference' if event['event_id'] in supports.values() else splitgroups[mapping[event['event_id']]]
        assert event['group_id']==mapping[event['event_id']] and event['split']==expected
        assert event['label']==row['label'] and event['ifo']==row['ifo'] and event['publisher_split']==row['sample_type']
        assert event['image_url']==row['url3'] and event['duration_seconds']==2.
    counts=Counter(event['split'] for event in plan['events']);assert dict(counts)==plan['counts']['split_events']
    for label,partitions in plan['per_target'].items():
        for split,statistics in partitions.items():
            subset=[e for e in plan['events'] if e['split']==split]
            positive=[e for e in subset if e['label']==label];negative=[e for e in subset if e['label']!=label]
            assert statistics=={'positive_events':len(positive),'negative_events':len(negative),
                'positive_time_groups':len({e['group_id'] for e in positive}),'negative_time_groups':len({e['group_id'] for e in negative})}
    quality=read(DATA/'quality_admission.json');assert quality['roster_sha256']==digest(DATA/'plan.json')
    invalid={r['event_id'] for r in quality['records'] if not r['eligible']}
    assert invalid==set(quality['excluded_event_ids']) and len(invalid)==30
    invalid_refs={label for label,event_id in supports.items() if event_id in invalid}
    assert invalid_refs=={'Extremely_Loud'} and set(quality['eligible_target_labels'])==set(supports)-invalid_refs
    image_records=[read(path) for path in (DATA/'image_records').glob('*.json')]
    assert not any(record['split']=='test' for record in image_records)
    original=read(ROOT/'experiments/gravity_spy_probe_exclusion.json')
    assert original['event_id'] in plan['pre_exposed_event_ids'] and original['final_roster_sha256']==digest(DATA/'plan.json')
    assert all(event['split']=='excluded_pre_exposed_group' for event in plan['events'] if event['group_id'] in exposed)
    report={'metadata_valid':True,'review_scope':'Independent officialCSV/group/reference/hash-split/per-target-count recomputation; no pixel decoding or model calls',
        'plan_sha256':digest(DATA/'plan.json'),'quality_admission_sha256':digest(DATA/'quality_admission.json'),
        'source_metadata_events':len(rows),'time_groups':len(groups),'split_events':dict(counts),
        'raw_targets':len(supports),'eligible_targets':len(quality['eligible_target_labels']),
        'constant_plot_exclusions':len(invalid),'excluded_reference_labels':sorted(invalid_refs),
        'eligible_calibration_events':sum(e['split']=='calibration' and e['event_id'] not in invalid for e in plan['events']),
        'test_image_sidecars_present':0,'exposure_groups_fully_excluded':True,
        'release_gate_review':'Current loader/downloader validate hash syntax, not actual checkpoint/source/calibration paths. Strong outer protocol artifact verification is required beforeTEST acquisition, or strengthen this gate.',
        'quality_review_limit':'This metadata-only review checks the predeclared exact-constant rule and exclusion bookkeeping, not the actual RGB values of30excluded crops.',
        'cross_detector_caveat':'Grouping separates H1/L1; physically coincident cross-detector events may remain dependent, as disclosed.'}
    output=ROOT/'experiments/gravity_spy_independent_metadata_review.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
