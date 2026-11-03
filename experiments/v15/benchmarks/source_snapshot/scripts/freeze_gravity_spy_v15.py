"""Bind actual calibration artifacts before opening fresh TEST pixel access."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.gravity_spy_admission import digest,mapping_digest,verify_test_admission
from hypernet.gravity_spy_images import GravitySpyCohort
from hypernet.detail_features import FrozenDetails
from hypernet.pattern_matcher import RECIPES

MODELS={
    'random':('random_initial.pt','gravity_calibration_random'),
    '065536':('families_0065536.pt','gravity_calibration_065536'),
    '262144':('families_0262144.pt','gravity_calibration_262144'),
    'final':('families_1048576.pt','gravity_calibration_final')}


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/gravity_spy_v15_final')
    p.add_argument('--output',type=Path,default=ROOT/'experiments/gravity_spy_protocol_freeze.json')
    p.add_argument('--admission',type=Path,default=ROOT/'experiments/gravity_spy_test_admission.json')
    args=p.parse_args()
    if args.output.exists()or args.admission.exists():raise FileExistsError('Preserve previous protocol freezes')
    cohort=GravitySpyCohort(args.data)
    if cohort.prior_exposure is None or len(cohort.target_labels)!=19:raise ValueError('Final quality/exposure amendments required')
    image_records=[json.loads(path.read_text(encoding='utf-8'))for path in(args.data/'image_records').glob('*.json')]
    if any(row['split']=='test'for row in image_records):raise ValueError('TEST image acquisition already occurred before freeze')
    behavior={};checkpoints={};calibrations={};results={};expected_sources=None
    for name,(filename,folder)in MODELS.items():
        checkpoint=ROOT/'checkpoints/details_run01'/filename
        results_path=ROOT/'experiments'/folder/'results.json';score_path=results_path.parent/'event_scores.jsonl'
        result=json.loads(results_path.read_text(encoding='utf-8'))
        results[name]=result
        if result['mode']!='calibration'or result['events']!=len(cohort.query_events('calibration')):
            raise ValueError('A model lacks complete CALIB evaluation: '+name)
        if result['target_labels']!=list(cohort.target_labels)or result['scores_sha256']!=digest(score_path):
            raise ValueError('Calibration roster or score artifact mismatch: '+name)
        for key in('roster_sha256','quality_admission_sha256','prior_exposure_admission_sha256'):
            if result['cohort_provenance'][key]!=cohort.provenance()[key]:raise ValueError('Calibration data policy differs: '+name)
        if expected_sources is None:expected_sources=result['source_sha256']
        if result['source_sha256']!=expected_sources:raise ValueError('Models used different inference source versions')
        for source,value in result['source_sha256'].items():
            if digest(ROOT/source)!=value:raise ValueError('Inference source changed since calibration: '+source)
            behavior[source]=value
        actual=FrozenDetails(checkpoint,device='cpu').provenance()
        if actual!=result['checkpoint_provenance']:raise ValueError('Actual checkpoint differs from calibrated weights: '+name)
        checkpoints[relative(checkpoint)]=digest(checkpoint)
        behavior[relative(results_path)]=digest(results_path);behavior[relative(score_path)]=digest(score_path)
        selection=result['selection'];selected=selection['selected_method']
        thresholds={target:result['methods'][selected]['per_target'][target]['budget_10_per10000']['cutoff']['threshold']for target in cohort.target_labels}
        calibrations[name]={'checkpoint_path':relative(checkpoint),'result_path':relative(results_path),
            'score_path':relative(score_path),'scores_sha256':digest(score_path),
            'selected_recipe':selection['selected_recipe'],'selected_method':selected,'thresholds':thresholds,
            'test_methods':list(result['methods']),
            'selection':'one global recipe per checkpoint selected on CALIB only; checkpoint primary role predeclared'}
    for filename in('plan.json','roster_freeze.json','quality_admission.json','prior_exposure_admission.json'):
        path=args.data/filename;behavior[relative(path)]=digest(path)
    for name in('scripts/prepare_gravity_spy.py','scripts/audit_gravity_spy_quality.py',
                 'scripts/freeze_gravity_spy_v15.py','checkpoints/details_run01/admission.json','checkpoints/details_run01/completed.json'):
        behavior[name]=digest(ROOT/name)
    final=calibrations['final']
    analytic_candidates=['analytic:'+recipe for recipe in RECIPES]
    analytic_selected=max(analytic_candidates,key=lambda key:(results['final']['methods'][key]['macro']['recall'],
        results['final']['methods'][key]['macro']['auroc'],-analytic_candidates.index(key)))
    reported_arms={
        'primary_final':{'model':'final','method':final['selected_method']},
        'random':{'model':'random','method':calibrations['random']['selected_method']},
        'scale_065536':{'model':'065536','method':calibrations['065536']['selected_method']},
        'scale_262144':{'model':'262144','method':calibrations['262144']['selected_method']},
        'analytic':{'model':'final','method':analytic_selected},
        'raw_pixels48':{'model':'final','method':'raw_pixels48'}}
    for arm in reported_arms.values():
        per_target=results[arm['model']]['methods'][arm['method']]['per_target']
        arm['thresholds']={target:{point:per_target[target][point]['cutoff']['threshold']for point in('budget_10_per10000','zero_calibration_false_alarms')}for target in cohort.target_labels}
    protocol={'format':'hypernet.gravity_spy.frozen_pattern_search.v15','created_utc':datetime.now(timezone.utc).isoformat(),
        'model_selection_complete':True,'test_scores_computed_before_freeze':False,'test_pixels_acquired_before_freeze':False,
        'roster_sha256':cohort.roster_sha256,'quality_admission_sha256':cohort.quality_sha256,
        'prior_exposure_admission_sha256':cohort.prior_exposure_sha256,
        'thresholds':final['thresholds'],'model_checkpoint_path':final['checkpoint_path'],
        'calibration_scores_path':final['score_path'],'calibration_scores_sha256':final['scores_sha256'],
        'behavior_files':behavior,'checkpoint_files':checkpoints,'models':calibrations,
        'reported_arms':reported_arms,
        'analytic_selection':'Own one global recipe on the same CALIB macro-recall/AUROC/order criterion; no TEST selection. Raw48 has no recipe search.',
        'primary_checkpoint':'final','primary_method':final['selected_method'],
        'secondary_controls':['random','065536','262144','analytic same geometry recipes','raw_pixels48'],
        'operating_points':['10 false alarms per10000 CALIB negatives','zero CALIB false alarms (secondary)'],
        'test_scoring_policy':'fresh_only; all3 locked recipes retained as diagnostics; no TEST recipe/checkpoint selection',
        'primary_test_events_before_blind_quality':len(cohort.query_events('test')),
        'target_labels':list(cohort.target_labels),'reference_images_per_target':1,
        'real_weight_training_images':0,'additional_calibration_events':len(cohort.query_events('calibration')),
        'rate_scope':'Per-target negative-event false alarms and multi-target wrong-label union events are separate metrics; queries are never19 independent images.'}
    args.output.write_text(json.dumps(protocol,indent=2,allow_nan=False),encoding='utf-8')
    admission={'protocol_path':relative(args.output),'protocol_sha256':digest(args.output),
        'calibration_complete':True,'roster_sha256':cohort.roster_sha256,'quality_admission_sha256':cohort.quality_sha256,
        'prior_exposure_admission_sha256':cohort.prior_exposure_sha256,'thresholds':final['thresholds'],
        'model_checkpoint_sha256':checkpoints[final['checkpoint_path']],
        'inference_source_sha256':mapping_digest(behavior),'calibration_scores_sha256':final['scores_sha256']}
    args.admission.write_text(json.dumps(admission,indent=2,allow_nan=False),encoding='utf-8')
    verified=verify_test_admission(args.data,args.admission)
    print(json.dumps({'protocol_sha256':digest(args.output),'admission_sha256':digest(args.admission),
        'primary_method':protocol['primary_method'],'actual_files_verified':verified['verified_actual_artifacts'],
        'fresh_test_events_before_quality':protocol['primary_test_events_before_blind_quality']},indent=2))


if __name__=='__main__':main()
