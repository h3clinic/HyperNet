"""Fixed raw-pixel template comparison on the frozen v17 synthetic diagnosis.

Uses the exact RawTemplate implementation and the same PLAN/episodes as
diagnose_synthetic.py. No fitting, parameter search, natural images or GPU use.
One global cutoff is frozen on development negatives before either holdout.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json,sys,time,traceback
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.scene_curriculum import episode
from hypernet.raw_template import RawTemplate
from hypernet.detection_metrics import global_cutoff,iou,auc,clip_box,COMMON_BOX_RULE

DIAGNOSIS_SCRIPT=ROOT/'scripts/diagnose_synthetic.py'
INPUT_HASH_KEYS=('support','query','support_mask','target_mask','query_foreground_mask')


def read_fixed_plan():
    # Read the declared literal without importing the torch/GPU diagnosis runner.
    module=ast.parse(DIAGNOSIS_SCRIPT.read_text(encoding='utf-8'))
    assignments=[node for node in module.body if isinstance(node,ast.Assign)
                 and any(isinstance(target,ast.Name) and target.id=='PLAN' for target in node.targets)]
    if len(assignments)!=1:raise ValueError('One explicit synthetic diagnosis PLAN is required')
    plan=ast.literal_eval(assignments[0].value)
    if plan['seed']!=1777 or plan['splits']!={'development':256,'test':512,'compositional':512}:
        raise ValueError('The predeclared synthetic episode plan changed')
    if (plan['support_size'],plan['query_size'],plan['nominal_dev_absent_fpr'],plan['iou_threshold'])!=(96,192,.05,.5):
        raise ValueError('The predeclared synthetic metric or dimensions changed')
    return plan


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')


def summarize(rows,threshold):
    positive=[r for r in rows if r['present']];negative=[r for r in rows if not r['present']]
    true=sum(r['score']>=threshold and r['iou']>=.5 for r in positive)
    false=sum(r['score']>=threshold for r in negative)
    return {'episodes':len(rows),'positives':len(positive),'negatives':len(negative),
            'localized_true_positives':true,'false_alarms':false,
            'localized_recall':true/len(positive) if positive else None,
            'false_positive_rate':false/len(negative) if negative else None,
            'false_alarms_per_10000':10000*false/len(negative) if negative else None,
            'unthresholded_localized_recall':float(np.mean([r['iou']>=.5 for r in positive])) if positive else None,
            'positive_top1_mean_iou':float(np.mean([r['iou'] for r in positive])) if positive else None,
            'presence_roc_auc':auc([r['present'] for r in rows],[r['score'] for r in rows])}


def evaluate(model,split,count,plan):
    rows=[];hashes={key:[] for key in INPUT_HASH_KEYS};stream=hashlib.sha256()
    # Match the neural diagnosis's default batch16 digest grouping exactly.
    # RawTemplate itself processes one reference/query pair at a time.
    for start in range(0,count,16):
        group_hashes={key:[] for key in INPUT_HASH_KEYS}
        for index in range(start,min(start+16,count)):
            e=episode(index,split=split,seed=plan['seed'],support_size=plan['support_size'],query_size=plan['query_size'])
            for key in INPUT_HASH_KEYS:
                digest=np.frombuffer(hashlib.sha256(e[key].tobytes()).digest(),np.uint8)
                hashes[key].append(digest);group_hashes[key].append(digest)
            bank=model.enroll(e['support']);query=model.encode_query(e['query'])
            predictions=model.predict(bank,query)
            if not predictions:raise ValueError('Raw baseline must produce a top-one proposal')
            top=max(predictions,key=lambda prediction:prediction['score'])
            raw_box=list(top['box']);box=clip_box(raw_box,e['query'].shape[1],e['query'].shape[0])
            score=float(top['score']);overlap=iou(box,[e['bbox']]) if e['present'] else 0.
            if not np.isfinite([score,*box,overlap]).all():raise ValueError('Nonfinite raw prediction')
            rows.append({'index':index,'split':split,'program_id':e['program_id'],'family':e['family'],
                         'present':bool(e['present']),'score':score,'box':box,'raw_box':raw_box,
                         'truth_box':e['bbox'].tolist(),'iou':overlap})
        for key in sorted(group_hashes):
            stream.update(key.encode());stream.update(np.stack(group_hashes[key]).tobytes())
    model.assert_unchanged()
    arrays={'scores':np.asarray([r['score'] for r in rows],np.float64),
            'overlaps':np.asarray([r['iou'] for r in rows],np.float64),
            'boxes':np.asarray([r['box'] for r in rows],np.float64),
            'truth_boxes':np.asarray([r['truth_box'] for r in rows],np.float32),
            'present':np.asarray([r['present'] for r in rows],bool),
            'indices':np.asarray([r['index'] for r in rows],np.int64),
            'program_ids':np.asarray([r['program_id'] for r in rows]),
            'families':np.asarray([r['family'] for r in rows]),
            **{'input_sha256_'+key:np.stack(value) for key,value in hashes.items()}}
    return rows,arrays,stream.hexdigest()


def main(output):
    cv2.setNumThreads(1);plan=read_fixed_plan();model=RawTemplate()
    destination=ROOT/output
    if destination.exists():raise FileExistsError('Preserve existing/partial raw diagnosis: '+str(destination))
    destination.mkdir(parents=True)
    files=[Path(__file__),DIAGNOSIS_SCRIPT,ROOT/'hypernet/scene_curriculum.py',
           ROOT/'hypernet/raw_template.py',ROOT/'hypernet/detection_metrics.py',ROOT/'scripts/train_local.py']
    sources={str(path.relative_to(ROOT)):sha(path) for path in files}
    admission={'format':'hypernet.raw_synthetic_diagnosis.v17.1','created_utc':datetime.now(timezone.utc).isoformat(),
               'plan':plan,'source_sha256':sources,'provenance':model.provenance(),'opencv_threads':cv2.getNumThreads(),
               'input_digest_batch_size':16,'common_box_rule':COMMON_BOX_RULE,
               'input_policy':'Exact generated support/query RGB; masks/boxes are scoring labels only, never matcher inputs.',
               'training_images_used':0,'weight_updates':0,'gpu_used':False,'parameter_search':False,
               'unsupported_neural_metrics':['target-mask IoU: no mask output','reference logit change: correlation scores are not logits']}
    write_new(destination/'admission.json',admission)
    started=time.perf_counter();result={**admission,'splits':{},'rows':{}};threshold=None
    try:
        for split,count in plan['splits'].items():
            split_start=time.perf_counter();rows,arrays,digest=evaluate(model,split,count,plan)
            if split=='development':
                negative=[r['score'] for r in rows if not r['present']]
                threshold=global_cutoff(negative,plan['nominal_dev_absent_fpr'])
                write_new(destination/'calibration.json',{'threshold':threshold,'comparison':'score >= threshold',
                    'threshold_source':'Only the fixed development absent episodes, before holdout evaluation',
                    'negative_count':len(negative),'allowed_false_alarms':int(np.floor(.05*len(negative))),
                    'actual_false_alarms':sum(score>=threshold for score in negative),
                    'nominal_development_fpr':.05,'source_sha256':sources})
            metrics=summarize(rows,threshold)
            metrics['families']={family:summarize([r for r in rows if r['family']==family],threshold)
                                for family in sorted({r['family'] for r in rows})}
            array_file=destination/(split+'.npz')
            with array_file.open('xb') as handle:np.savez_compressed(handle,**arrays)
            result['splits'][split]={'input_stream_sha256':digest,'threshold':threshold,'metrics':metrics,
                                    'arrays_file':array_file.name,'arrays_sha256':sha(array_file),
                                    'seconds':time.perf_counter()-split_start}
            result['rows'][split]=rows
            print(json.dumps({'split':split,**{key:metrics[key] for key in (
                'episodes','localized_recall','false_positive_rate','unthresholded_localized_recall','presence_roc_auc')}}),flush=True)
        result['seconds']=time.perf_counter()-started
        result['sources_unchanged']=all(sha(path)==sources[str(path.relative_to(ROOT))] for path in files)
        if not result['sources_unchanged']:raise RuntimeError('A frozen behavior source changed during diagnosis')
        write_new(destination/'results.json',result)
        print(json.dumps({'status':'complete','seconds':result['seconds'],'destination':str(destination)}),flush=True)
    except Exception as exc:
        write_new(destination/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),
                  'completed_splits':list(result['splits']),'elapsed_seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments/synthetic_raw_template')
    args=parser.parse_args();main(args.output)
