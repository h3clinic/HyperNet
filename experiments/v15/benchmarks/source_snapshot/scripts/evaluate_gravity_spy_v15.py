"""One-reference pattern search: calibration first, sealed TEST afterward.

No label enters the encoder/matcher. Multiple recipes share one feature pass.
Source/checkpoint/pixel-bound score caches avoid re-encoding unchanged inputs.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.detail_features import FrozenDetails
from hypernet.pattern_matcher import PatternMatcher,RECIPES,CONFIG,raw_pixel_vector
from hypernet.gravity_spy_images import GravitySpyCohort,digest
from hypernet.rare_event_metrics import calibration_cutoff,summary,binomial_interval,upper_bound
from hypernet.gravity_spy_admission import verify_test_admission,resolve

SOURCE_NAMES=['hypernet/detail_encoder.py','hypernet/detail_features.py','hypernet/pattern_matcher.py',
    'hypernet/gravity_spy.py','hypernet/gravity_spy_images.py','hypernet/gravity_spy_quality.py',
    'hypernet/rare_event_metrics.py','hypernet/gravity_spy_admission.py','scripts/evaluate_gravity_spy_v15.py']


def union_event_summary(rows,targets,method,per_target,point='budget_10_per10000'):
    if not rows or any(not per_target[target].get('available')for target in targets):
        return {'available':False,'reason':'All target cutoffs require full calibration outcomes'}
    wrong=[];recovered=[];present=[];background=[];group_flags={}
    for row in rows:
        accepted={target for target in targets if row['scores'][method][target]['score']>=per_target[target][point]['cutoff']['threshold']}
        has_wrong=bool(accepted-{row['label']});has_target=row['label']in targets
        wrong.append(has_wrong);present.append(has_target);recovered.append(has_target and row['label']in accepted)
        if not has_target:background.append(bool(accepted))
        group_flags[row['group_id']]=group_flags.get(row['group_id'],False)or has_wrong
    total=len(rows);false=sum(wrong);npositive=sum(present);tp=sum(recovered);ngroups=len(group_flags);gfalse=sum(group_flags.values())
    return {'available':True,'screened_events':total,'wrong_target_alarm_events':false,
        'wrong_target_alarm_events_per10000_screened_events':10000*false/total,
        'wrong_target_alarm_event_binomial_ci95_per10000':[10000*x for x in binomial_interval(false,total)],
        'events_containing_an_enrolled_target':npositive,'target_recovered_events':tp,'target_recovery_recall':tp/npositive if npositive else None,
        'target_recovery_binomial_ci95':binomial_interval(tp,npositive)if npositive else None,
        'background_events':len(background),'background_events_with_any_alarm':sum(background),
        'screened_time_groups':ngroups,'time_groups_with_any_wrong_alarm':gfalse,
        'wrong_alarm_time_groups_per10000_groups':10000*gfalse/ngroups,
        'wrong_alarm_time_group_binomial_upper95_per10000':10000*upper_bound(gfalse,ngroups),
        'definition':'An event alarms wrongly if any incorrect target exceeds its own frozen cutoff, even if the correct target also alarms. Count each event once, never19 times.',
        'uncertainty_scope':'Binomial intervals assume independent events/groups; distinct groups do not prove physical independence.'}


def aggregate(rows,targets,methods,calibration=None):
    result={}
    for method in methods:
        per_target={}
        for target in targets:
            scores=[row['scores'][method][target]['score']for row in rows]
            present=[row['label']==target for row in rows]
            groups=[row['group_id']for row in rows]
            negatives=[score for score,y in zip(scores,present)if not y]
            if not negatives or not any(present):
                per_target[target]={'available':False,'reason':'plumbing subset lacks both label outcomes','positive_events':sum(present),'negative_events':len(negatives)}
                continue
            metrics={}
            for point,budget in(('budget_10_per10000',10.),('zero_calibration_false_alarms',0.)):
                cutoff=calibration['methods'][method]['per_target'][target][point]['cutoff']if calibration else calibration_cutoff(negatives,budget)
                metrics[point]={'cutoff':cutoff,'metrics':summary(scores,present,cutoff['threshold'],groups)}
            per_target[target]={'available':True,**metrics}
        valid=[r['budget_10_per10000']['metrics']for r in per_target.values()if r.get('available')]
        result[method]={'per_target':per_target,'union_events':{
            point:union_event_summary(rows,targets,method,per_target,point)for point in('budget_10_per10000','zero_calibration_false_alarms')},'macro':{
            'targets_with_positive_and_negative_events':len(valid),
            'recall':float(np.mean([r['recall']for r in valid]))if valid else None,
            'false_alarms_per10000':float(np.mean([r['false_alarms_per_10000_negatives']for r in valid]))if valid else None,
            'auroc':float(np.mean([r['auroc']for r in valid]))if valid else None,
            'scope':'Macro average across targets; shared negative events are not independent repeated trials.'}}
    return result


def select_recipe(methods):
    candidates=['learned:'+recipe for recipe in RECIPES]
    if any(methods[m]['macro']['recall']is None for m in candidates):raise ValueError('Full calibration required for selection')
    winner=max(candidates,key=lambda m:(methods[m]['macro']['recall'],methods[m]['macro']['auroc'],-candidates.index(m)))
    return {'selected_recipe':winner.split(':',1)[1],
        'criterion':'Macro recall at empirical10/10000 CALIB negative budget; tie macro AUROC, then declared recipe order.',
        'selected_method':winner,'query_labels_used_for_calibration_only':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/gravity_spy_v15_final');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('plumbing','calibration','test'),default='plumbing')
    p.add_argument('--limit-queries',type=int,default=24);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--include-analytic',action='store_true');p.add_argument('--select-primary',action='store_true')
    p.add_argument('--calibration-results',type=Path);p.add_argument('--test-admission',type=Path)
    p.add_argument('--cache',type=Path,default=ROOT/'data/gravity_spy_score_cache')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous run outputs')
    if args.mode!='plumbing'and args.limit_queries!=24:
        raise ValueError('Query limits apply only to plumbing; full calibration/test ignores default limit')
    if args.select_primary and args.mode!='calibration':raise ValueError('Primary selection requires full calibration')
    if args.mode=='test'and(not args.calibration_results or not args.test_admission):raise ValueError('Sealed TEST requires frozen calibration artifacts')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources={name:digest(ROOT/name)for name in SOURCE_NAMES}
    cohort=GravitySpyCohort(args.data,args.test_admission if args.mode=='test'else None)
    if args.mode=='test'and not(args.data/'quality_test_audit.json').exists():raise ValueError('Run blind TEST source-quality audit before scoring')
    extractor=FrozenDetails(args.checkpoint,device=args.device,batch_size=args.batch_size)
    provenance=extractor.provenance()
    if args.select_primary and provenance['consumed_synthetic_families']!=1048576:
        raise ValueError('Only the predeclared million-family final checkpoint may select the primary recipe')
    matcher=PatternMatcher(extractor);targets=cohort.target_labels
    calibration=None
    if args.mode=='test':
        calibration=json.loads(args.calibration_results.read_text())
        protocol=json.loads(Path(cohort.test_admission['verified_protocol_path']).read_text(encoding='utf-8'))
        bound={resolve(name):value for name,value in protocol['behavior_files'].items()}
        if args.calibration_results.resolve()not in bound or digest(args.calibration_results)!=bound[args.calibration_results.resolve()]:
            raise ValueError('Actual calibration result artifact is not protocol-bound')
        checkpoint_bindings={resolve(name):value for name,value in protocol['checkpoint_files'].items()}
        if args.checkpoint.resolve()not in checkpoint_bindings or provenance['checkpoint_sha256']!=checkpoint_bindings[args.checkpoint.resolve()]:
            raise ValueError('Evaluated checkpoint is not protocol-bound')
        if calibration['mode']!='calibration'or calibration['checkpoint_provenance']!=provenance:
            raise ValueError('TEST checkpoint differs from calibrated checkpoint')
        if calibration['source_sha256']!=sources or calibration['cohort_provenance']['roster_sha256']!=cohort.roster_sha256:
            raise ValueError('TEST source or roster differs from calibration')
        if calibration['cohort_provenance']['quality_admission_sha256']!=cohort.quality_sha256:
            raise ValueError('TEST source-quality policy differs from calibration')
        if calibration['cohort_provenance']['prior_exposure_admission_sha256']!=cohort.prior_exposure_sha256:
            raise ValueError('TEST prior-exposure policy differs from calibration')
    features_kinds=('learned','analytic')if args.include_analytic else('learned',)
    methods=[kind+':'+recipe for kind in features_kinds for recipe in RECIPES]+['raw_pixels48']
    if calibration and set(methods)!=set(calibration['methods']):raise ValueError('TEST methods differ from calibration')
    args.output.mkdir(parents=True)
    started=time.perf_counter();reference_images=[cohort.reference(label)for label in targets]
    banks=[matcher.enroll(rgb)for rgb in reference_images]
    raw_reference=np.stack([raw_pixel_vector(rgb)for rgb in reference_images])
    enrollment_seconds=time.perf_counter()-started
    bank_identity=[{'label':label,'rgb_sha256':bank.reference_rgb_sha256,'anchors':len(bank.xy)}for label,bank in zip(targets,banks)]
    cache_identity={'sources':sources,'checkpoint':provenance['checkpoint_sha256'],'banks':bank_identity,
        'recipes':RECIPES,'config':CONFIG,'kinds':features_kinds,'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'device':args.device}}
    cache_id=hashlib.sha256(json.dumps(cache_identity,sort_keys=True).encode()).hexdigest()
    cache=args.cache/cache_id;cache.mkdir(parents=True,exist_ok=True)
    identity_path=cache/'identity.json'
    if not identity_path.exists():identity_path.write_text(json.dumps(cache_identity,indent=2))
    split='test'if args.mode=='test'else'calibration';events=cohort.query_events(split)
    if args.mode=='plumbing':events=events[:args.limit_queries]
    rows=[];timings=[];cache_hits=0;out_scores=args.output/'event_scores.jsonl'
    with out_scores.open('w',encoding='utf-8')as handle:
        for index,event in enumerate(events):
            event_start=time.perf_counter();rgb=cohort.load_rgb(event['event_id'])
            pixel_sha=hashlib.sha256(rgb.tobytes()).hexdigest();cached=cache/(pixel_sha+'.json')
            if cached.exists():
                scored=json.loads(cached.read_text());cache_hits+=1
                if scored['pixel_sha256']!=pixel_sha or scored['cache_identity_sha256']!=cache_id:raise ValueError('Cached prediction binding mismatch')
                predictions=scored['scores']
            else:
                encoded=extractor.encode(rgb,sampling_stride=4);predictions={}
                for kind in features_kinds:
                    matched=matcher.score_many(banks,encoded,recipes=RECIPES,feature_kind=kind)
                    for recipe in RECIPES:
                        predictions[kind+':'+recipe]={}
                        for label,item in zip(targets,matched):
                            value=item[recipe]
                            keep=('anchor_count','contributing_anchors','reference_cells','covered_reference_cells','reference_cell_coverage','translation_xy','abstain')
                            predictions[kind+':'+recipe][label]={'score':value['score'],'evidence':{k:v for k,v in value['evidence'].items()if k in keep}}
                raw_scores=raw_reference@raw_pixel_vector(rgb)
                predictions['raw_pixels48']={label:{'score':float(score),'evidence':{'control':'fixed48x48gray centeredL2 dot; noCNN'}}for label,score in zip(targets,raw_scores)}
                scored={'pixel_sha256':pixel_sha,'cache_identity_sha256':cache_id,'scores':predictions}
                temporary=cached.with_suffix('.tmp');temporary.write_text(json.dumps(scored,separators=(',',':'),allow_nan=False));temporary.replace(cached)
            row={'event_id':event['event_id'],'group_id':event['group_id'],'label':event['label'],'split':split,
                'pixel_sha256':pixel_sha,'scores':predictions}
            handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');handle.flush();rows.append(row)
            timings.append(time.perf_counter()-event_start)
            if(index+1)%16==0:print(json.dumps({'mode':args.mode,'events_scored':index+1,'target_events':len(events),'seconds':time.perf_counter()-started,'cached':cache_hits}),flush=True)
    metrics=aggregate(rows,targets,methods,calibration)
    after=extractor.provenance()
    if after!=provenance or any(digest(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('Model or inference source changed during evaluation')
    if args.mode=='test':verify_test_admission(args.data,args.test_admission)
    selection=select_recipe(metrics)if args.mode=='calibration'else None
    if selection:selection['checkpoint_role']='predeclared final primary'if args.select_primary else'checkpoint control; no TEST model selection'
    result={'completed':True,'mode':args.mode,'scope':'One positive reference per target morphology; no real CNN weight updates. Additional labeled calibration events are counted.',
        'checkpoint_provenance':provenance,'cohort_provenance':cohort.provenance(),'source_sha256':sources,
        'target_labels':list(targets),'events':len(rows),'time_groups':len({r['group_id']for r in rows}),
        'methods':metrics,'selection':selection,'reference_banks':bank_identity,
        'model_state_unchanged':after==provenance,'calibration_results_sha256':digest(args.calibration_results)if args.calibration_results else None,
        'scores_sha256':digest(out_scores),'cache_identity_sha256':cache_id,'cache_hits':cache_hits,
        'runtime':{'total_seconds':time.perf_counter()-started,'enrollment_seconds':enrollment_seconds,
            'median_query_seconds':float(np.median(timings)),'mean_query_seconds':float(np.mean(timings)),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()if args.device=='cuda'else None},
        'plumbing_only_not_model_selection':args.mode=='plumbing'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'mode':args.mode,'events':len(rows),'runtime':result['runtime'],'selection':selection},indent=2),flush=True)


if __name__=='__main__':main()
