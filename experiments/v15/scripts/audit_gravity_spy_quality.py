"""Blind plot validity audit before scoring; retains every excluded event ID."""
from pathlib import Path
import argparse,json,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.gravity_spy_images import GravitySpyCohort,digest
from hypernet.gravity_spy_quality import classify_plot,RULE_ID


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/gravity_spy_v15_final')
    p.add_argument('--split',choices=('development','test'),default='development');p.add_argument('--test-admission',type=Path)
    args=p.parse_args();out=args.data/('quality_admission.json'if args.split=='development'else'quality_test_audit.json')
    if out.exists():raise FileExistsError('Preserve frozen quality evidence; output already exists')
    # Validate active policy/admission first, then read raw pixels without filtering.
    if args.split=='test':
        checked=GravitySpyCohort(args.data,args.test_admission)
        if checked.test_admission is None:raise PermissionError('TEST quality audit requires frozen admission')
        cohort=GravitySpyCohort(args.data,apply_quality=False)
        cohort.test_admission=checked.test_admission
        events=list(checked.query_events('test'))
    else:
        cohort=GravitySpyCohort(args.data,apply_quality=False)
        events=[r for r in cohort.plan['events']if r['split']in('reference','calibration')]
    records=[]
    for row in events:
        quality=classify_plot(cohort.load_rgb(row['event_id']))
        records.append({'event_id':row['event_id'],'split':row['split'],'label':row['label'],**quality})
    excluded=[r['event_id']for r in records if not r['eligible']]
    result={'format':'hypernet.gravity_spy.source_quality.v15','rule':RULE_ID,
        'roster_sha256':cohort.roster_sha256,'created_utc':datetime.now(timezone.utc).isoformat(),
        'quality_source_sha256':digest(ROOT/'hypernet/gravity_spy_quality.py'),
        'raw_roster_unchanged':True,'replacement_sampling':False,'model_scores_used':False,
        'pixel_rule':'Exclude iff every RGB pixel is exactly identical to the top-left RGB pixel; no low-contrast threshold.',
        'requested_events':len(records),'eligible_events':len(records)-len(excluded),
        'excluded_event_ids':excluded,'records':records}
    if args.split=='development':
        unavailable={label:event_id for label,event_id in cohort.plan['reference_by_label'].items()if event_id in excluded}
        result['unavailable_targets']=unavailable
        result['eligible_target_labels']=[label for label in cohort.plan['target_labels']if label not in unavailable]
        result['eligible_calibration_events']=sum(r['eligible']and r['split']=='calibration'for r in records)
        result['test_pixels_seen']=False
    else:
        result['quality_admission_sha256']=checked.quality_sha256
        result['test_admission_sha256']=digest(args.test_admission)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k not in('records','excluded_event_ids')},indent=2))
    print('OUTPUT_SHA256',digest(out))


if __name__=='__main__':main()
