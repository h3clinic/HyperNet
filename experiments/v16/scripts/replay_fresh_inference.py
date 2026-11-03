"""Prespecified two-query exact replay using the original frozen inference APIs.

This is an execution/reproducibility check, not an independently implemented
model. Resolution reads metadata only, after blind QC and before TEST scores.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import evaluate_v16 as evaluation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as file:json.dump(value,file,indent=2,allow_nan=False);file.write('\n')


def select_context(plan,quality,replay_plan):
    if replay_plan.get('query_count')!=2 or replay_plan.get('new_cnn_batch_size')!=4 or replay_plan.get('replay_tolerance')!=0:
        raise ValueError('The fixed two-query batch4 zero-tolerance replay is required')
    if replay_plan.get('batch_context_policy')!='first_original_batch' or replay_plan.get('context_queries_scored')is not False:
        raise ValueError('Original batch-context policy must be clarified before TEST')
    if replay_plan.get('methods')!=list(evaluation.ALIASES):raise ValueError('Fixed five replay arms differ')
    ids=[row['event_id']for row in plan['events']];eligible=quality['eligible_event_ids']
    if len(set(ids))!=len(ids)or len(set(eligible))!=len(eligible)or not set(eligible)<=set(ids):
        raise ValueError('Invalid frozen/eligible roster')
    eligible_set=set(eligible)
    effective=[row for row in plan['events']if row['event_id']in eligible_set]
    if len(effective)<2:raise ValueError('At least two blind-QC-valid events required')
    return effective,effective[:2],effective[:4]


def metadata_gate(plan_path,protocol_path,admission_path,replay_plan_path):
    plan_path=Path(plan_path).resolve();protocol_path=Path(protocol_path).resolve()
    admission_path=Path(admission_path).resolve();replay_plan_path=Path(replay_plan_path).resolve()
    plan=read(plan_path);protocol=read(protocol_path);admission=read(admission_path);replay_plan=read(replay_plan_path)
    if admission.get('test_scoring_authorized')is not True or admission['protocol_sha256']!=sha(protocol_path):
        raise ValueError('Scoring admission/protocol mismatch')
    if Path(admission['protocol_path']).resolve()!=protocol_path or Path(protocol['plan_path']).resolve()!=plan_path:
        raise ValueError('Wrong actual protocol/plan path')
    if protocol['fresh_plan_sha256']!=sha(plan_path)or admission['fresh_plan_sha256']!=sha(plan_path):
        raise ValueError('Frozen fresh roster changed')
    evaluation.verify_files(protocol['bindings']);evaluation.verify_files(admission['bindings'])
    evaluation.verify_files(protocol['checkpoint_files']);evaluation.verify_files(protocol['behavior_files'])
    quality_path=plan_path.parent/'blind_quality_audit.json';quality=read(quality_path)
    if sha(quality_path)!=admission['blind_quality_audit_sha256']or quality['protocol_sha256']!=sha(protocol_path)or quality['fresh_plan_sha256']!=sha(plan_path):
        raise ValueError('Blind QC binding mismatch')
    if any(quality.get(key)is not False for key in ('model_scores_used','manual_image_curation','replacement_sampling')):
        raise ValueError('Expected blind fixed quality exclusions')
    if admission['effective_event_ids']!=quality['eligible_event_ids']or admission['image_record_sha256']!=quality['image_record_sha256']:
        raise ValueError('Admission differs from blind QC roster/records')
    if protocol['methods']!=list(evaluation.ALIASES)or protocol['target_labels']!=plan['target_labels']or len(plan['target_labels'])!=14:
        raise ValueError('Expected fixed five arms and fourteen targets')
    if len(protocol['all_reference_target_order'])!=19 or len(protocol['reference_pixel_hashes_all19'])!=19:
        raise ValueError('Expected the same full nineteen-reference enrollment bank')
    if protocol['selected_recipe']not in evaluation.RECIPES:raise ValueError('Unknown frozen recipe')
    effective,queries,context=select_context(plan,quality,replay_plan)
    for row in context:
        record=plan_path.parent/'image_records'/(row['event_id']+'.json');expected=quality['image_record_sha256'][row['event_id']]
        if sha(record)!=expected or admission['bindings'].get(str(record.resolve()))!=expected:
            raise ValueError('Actual replay image sidecar is not admitted')
    return plan,protocol,admission,replay_plan,quality_path,effective,queries,context


def resolve(args):
    test_run=Path(args.test_run).resolve()
    if any((test_run/name).exists()for name in ('results.json','scores.npz','event_scores.jsonl')):
        raise ValueError('Resolve replay IDs before any TEST scores exist')
    parts=metadata_gate(args.plan,args.protocol,args.admission,args.replay_plan)
    plan,protocol,admission,replay_plan,quality_path,effective,queries,context=parts
    bindings={str(Path(path).resolve()):sha(path)for path in
              (args.plan,args.protocol,args.admission,args.replay_plan,quality_path,Path(__file__))}
    bindings.update(evaluation.source_hashes());bindings.update(protocol['checkpoint_files'])
    result={'format':'hypernet.v16.resolved_two_query_replay','created_utc':datetime.now(timezone.utc).isoformat(),
            'resolved_before_test_scores':True,'test_run_path':str(test_run),'replay_plan_path':str(Path(args.replay_plan).resolve()),
            'plan_path':str(Path(args.plan).resolve()),'protocol_path':str(Path(args.protocol).resolve()),
            'admission_path':str(Path(args.admission).resolve()),'bindings':bindings,
            'query_event_ids':[r['event_id']for r in queries],'batch_context_event_ids':[r['event_id']for r in context],
            'extra_context_event_ids':[r['event_id']for r in context[2:]],'batch_context_policy':'first_original_batch',
            'tail_padding_policy':'FrozenRelations repeats last input to batch4, exactly as original TEST',
            'effective_event_count':len(effective),'effective_order_sha256':hashlib.sha256(json.dumps([r['event_id']for r in effective],separators=(',',':')).encode()).hexdigest(),
            'scored_queries':2,'targets':list(protocol['target_labels']),'methods':list(evaluation.ALIASES),
            'numeric_scores_to_compare':140,'selected_recipe':protocol['selected_recipe'],'tolerance':0,
            'implementation_reuse':'Reuses evaluate_v16 configure, image loading, FrozenRelations, score_vectors, FrozenDetails and PatternMatcher. Not an independent model implementation.',
            'test_pixels_read_here':False,'model_calls_here':0,'weights_updated':False}
    write_new(args.output,result);print(json.dumps({'resolved':True,'queries':result['query_event_ids'],'context':result['batch_context_event_ids'],'sha256':sha(args.output)}))


def completed_scores(test_run,result,events,protocol):
    test_run=Path(test_run)
    if result.get('completed')is not True or result.get('stage')!='test':raise ValueError('Completed TEST is required before replay')
    if result['target_labels']!=protocol['target_labels']or result['selected_recipe']!=protocol['selected_recipe']or result['events']!=len(events):
        raise ValueError('Completed TEST metadata differs from frozen scope')
    if result['scores_sha256']!=sha(test_run/'scores.npz')or result['score_log_sha256']!=sha(test_run/'event_scores.jsonl'):
        raise ValueError('Completed TEST score archive/log changed')
    with np.load(test_run/'scores.npz',allow_pickle=False)as data:archive={key:data[key].copy()for key in data.files}
    if archive['methods'].tolist()!=list(evaluation.ALIASES)or archive['targets'].tolist()!=protocol['target_labels']:
        raise ValueError('TEST score method/target order differs')
    for key,field in (('event_ids','event_id'),('labels','label'),('group_ids','group_id')):
        if archive[key].tolist()!=[row[field]for row in events]:raise ValueError('TEST score roster differs: '+key)
    expected_thresholds=np.asarray([protocol['thresholds'][a]for a in evaluation.ALIASES],np.float64)
    if not np.array_equal(archive['thresholds'],expected_thresholds):raise ValueError('TEST thresholds differ from frozen protocol')
    if archive['scores'].shape!=(5,len(events),14)or not np.isfinite(archive['scores']).all():raise ValueError('Invalid complete TEST score tensor')
    with (test_run/'event_scores.jsonl').open(encoding='utf-8')as file:
        for i in range(2):
            row=json.loads(next(file))
            if row['event_id']!=events[i]['event_id']or row['pixel_sha256']!=str(archive['pixel_sha256'][i]):raise ValueError('TEST prefix log metadata differs')
            values=np.asarray([row['scores'][a]for a in evaluation.ALIASES],np.float64)
            if not np.array_equal(values,archive['scores'][:,i]):raise ValueError('TEST prefix log numeric scores differ')
    return archive


def compare_exact(expected,actual):
    expected=np.asarray(expected,np.float64);actual=np.asarray(actual,np.float64)
    if expected.shape!=(5,2,14)or actual.shape!=expected.shape or not np.isfinite(actual).all()or not np.isfinite(expected).all():raise ValueError('Expected140 finite replay scores')
    difference=np.abs(actual-expected);locations=np.argwhere(actual!=expected)
    return {'passed':len(locations)==0,'numeric_values_compared':140,'exact_values':int(np.sum(actual==expected)),
            'maximum_absolute_error':float(difference.max()),'tolerance':0,
            'mismatches':[{'method_index':int(a),'query_index':int(q),'target_index':int(t),
                           'expected':float(expected[a,q,t]),'actual':float(actual[a,q,t])}for a,q,t in locations]}


def replay(args):
    if Path(args.output).exists():raise FileExistsError(args.output)
    resolution=read(args.resolution);resolution_sha=sha(args.resolution)
    evaluation.verify_files(resolution['bindings'])
    if resolution.get('format')!='hypernet.v16.resolved_two_query_replay'or resolution.get('resolved_before_test_scores')is not True:
        raise ValueError('Expected the prescoring resolved replay admission')
    parts=metadata_gate(resolution['plan_path'],resolution['protocol_path'],resolution['admission_path'],resolution['replay_plan_path'])
    plan,protocol,admission,replay_plan,quality_path,events,queries,context=parts
    if resolution['query_event_ids']!=[r['event_id']for r in queries]or resolution['batch_context_event_ids']!=[r['event_id']for r in context]:
        raise ValueError('Resolved replay roster changed')
    if resolution['tolerance']!=0 or resolution['numeric_scores_to_compare']!=140:raise ValueError('Resolution changed replay scope/tolerance')
    order_hash=hashlib.sha256(json.dumps([r['event_id']for r in events],separators=(',',':')).encode()).hexdigest()
    if order_hash!=resolution['effective_order_sha256']or len(events)!=resolution['effective_event_count']:raise ValueError('Effective TEST order changed')
    test_run=Path(resolution['test_run_path']);results_path=test_run/'results.json';result=read(results_path)
    if result['protocol_sha256']!=sha(resolution['protocol_path'])or result['admission_sha256']!=sha(resolution['admission_path']):
        raise ValueError('Completed TEST binding differs from replay resolution')
    archive=completed_scores(test_run,result,events,protocol)
    device=args.device or protocol['runtime']['device']
    if protocol['runtime']!={'device':device,'torch':str(torch.__version__),'numpy':str(np.__version__)}:raise ValueError('Replay runtime differs')
    bound_outputs={str(path.resolve()):sha(path)for path in (results_path,test_run/'scores.npz',test_run/'event_scores.jsonl',Path(args.resolution))}
    # Every gate above is metadata-only. Pixel/model calls begin only here.
    evaluation.configure();started=time.perf_counter();cohort=evaluation.old_data();all_targets=list(cohort.target_labels)
    refs=[cohort.reference(target)for target in all_targets]
    if all_targets!=protocol['all_reference_target_order']or [hashlib.sha256(x.tobytes()).hexdigest()for x in refs]!=protocol['reference_pixel_hashes_all19']:
        raise ValueError('Actual reference enrollment bank differs')
    indices=[all_targets.index(t)for t in protocol['target_labels']]
    extractors={a:evaluation.FrozenRelations(protocol['models'][a],device)for a in evaluation.CNN_ALIASES}
    banks={a:e.enroll_many(refs)[0]for a,e in extractors.items()}
    raw_bank=np.stack([evaluation.raw_pixel_vector(x)for x in refs])
    old_model=evaluation.FrozenDetails(evaluation.OLD/'checkpoints/details_run01/families_1048576.pt',device=device,batch_size=64)
    old_before=old_model.provenance();old_matcher=evaluation.PatternMatcher(old_model,recipe='translation_006')
    old_banks=[old_matcher.enroll(x)for x in refs]
    query_images=[evaluation.fresh_rgb(Path(resolution['plan_path']).parent,row)for row in context]
    pixel_hashes=[hashlib.sha256(x.tobytes()).hexdigest()for x in query_images]
    if pixel_hashes!=archive['pixel_sha256'][:len(context)].tolist():raise ValueError('Actual replay/context pixels differ from original TEST')
    encoded={a:e.encode_many(query_images)for a,e in extractors.items()}
    actual=np.empty((5,2,14),np.float64)
    for i,rgb in enumerate(query_images[:2]):
        for j,a in enumerate(evaluation.CNN_ALIASES):
            actual[j,i]=evaluation.score_vectors(banks[a],encoded[a][i])[protocol['selected_recipe']][indices]
        old_scored=old_matcher.score_many(old_banks,old_model.encode(rgb,sampling_stride=4),recipes=('translation_006',))
        actual[3,i]=[old_scored[k]['translation_006']['score']for k in indices]
        actual[4,i]=(raw_bank@evaluation.raw_pixel_vector(rgb))[indices]
    for extractor in extractors.values():extractor.assert_unchanged()
    if old_model.provenance()!=old_before:raise RuntimeError('v15 state/provenance changed')
    evaluation.verify_files(resolution['bindings']);evaluation.verify_files(protocol['bindings']);evaluation.verify_files(admission['bindings']);evaluation.verify_files(bound_outputs)
    report={'format':'hypernet.v16.exact_two_query_replay','created_utc':datetime.now(timezone.utc).isoformat(),
            **compare_exact(archive['scores'][:,:2],actual),'query_event_ids':resolution['query_event_ids'],
            'batch_context_event_ids':resolution['batch_context_event_ids'],'extra_context_event_ids':resolution['extra_context_event_ids'],
            'batch_context_rgb_sha256':pixel_hashes,'methods':list(evaluation.ALIASES),'target_labels':protocol['target_labels'],
            'actual_scores':actual.tolist(),'expected_scores':archive['scores'][:,:2].tolist(),
            'selected_recipe':protocol['selected_recipe'],'reference_count':19,'scored_target_count':14,
            'weights_unchanged':True,'gradients_or_optimization':False,'new_query_scores':140,
            'implementation_reuse':resolution['implementation_reuse'],'scope':'Two scored queries only, with original fixed-batch context; not a full inference/training replay',
            'resolution_sha256':resolution_sha,'bindings':{**resolution['bindings'],**bound_outputs},
            'runtime':protocol['runtime'],'seconds':time.perf_counter()-started}
    write_new(args.output,report)
    print(json.dumps({key:report[key]for key in ('passed','numeric_values_compared','exact_values','maximum_absolute_error','seconds')}))
    if not report['passed']:raise RuntimeError('Exact inference replay failed; tolerance remains zero')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resolve',action='store_true')
    p.add_argument('--replay-plan',type=Path,default=ROOT/'experiments/fresh_inference_replay_plan.json')
    p.add_argument('--plan',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--admission',type=Path)
    p.add_argument('--test-run',type=Path);p.add_argument('--resolution',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'))
    args=p.parse_args()
    if args.resolve:
        if any(getattr(args,key)is None for key in ('plan','protocol','admission','test_run')):p.error('--resolve requires --plan --protocol --admission --test-run')
        resolve(args)
    else:
        if args.resolution is None:p.error('Replay requires --resolution and a completed hash-bound TEST run')
        replay(args)


if __name__=='__main__':main()
