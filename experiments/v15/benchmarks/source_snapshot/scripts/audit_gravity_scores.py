"""Recompute frozen Gravity Spy score metrics independently with sklearn.

No image inference, model selection, or threshold changes. Verifies event
identities against the admitted cohort and all target-specific confusion counts.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,sys
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.gravity_spy_images import GravitySpyCohort

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def close(actual,expected,name):
    require(math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=1e-12),'Metric mismatch: '+name)

def audit(result_path,output,admission=None):
    result_path=Path(result_path).resolve();output=Path(output).resolve()
    require(not output.exists(),'Preserve prior audit')
    result=read(result_path);require(result['completed'] and result['mode'] in ('test','calibration'),'Complete scored cohort required')
    cohort=GravitySpyCohort(ROOT/'data/gravity_spy_v15_final',admission if result['mode']=='test' else None)
    calibration=None
    if result['mode']=='test':
        protocol=read(cohort.test_admission['verified_protocol_path'])
        candidates=[value for value in protocol['models'].values()
                    if sha(ROOT/value['checkpoint_path'])==result['checkpoint_provenance']['checkpoint_sha256']]
        require(len(candidates)==1,'TEST checkpoint does not identify one frozen model')
        calibration_path=ROOT/candidates[0]['result_path'];calibration=read(calibration_path)
        require(sha(calibration_path)==result['calibration_results_sha256'],'TEST calibration binding differs')
        require(calibration['checkpoint_provenance']==result['checkpoint_provenance'],'Calibrated checkpoint differs')
        require(set(calibration['methods'])==set(result['methods']),'Calibrated method roster differs')
    score_path=result_path.parent/'event_scores.jsonl';require(sha(score_path)==result['scores_sha256'],'Score log hash mismatch')
    rows=[json.loads(line) for line in score_path.read_text(encoding='utf-8').splitlines()]
    expected=cohort.query_events('test' if result['mode']=='test' else 'calibration')
    require([r['event_id'] for r in rows]==[r['event_id'] for r in expected],'Actual query roster/order differs')
    require(len(rows)==result['events'],'Event count differs')
    for actual,original in zip(rows,expected):
        require(actual['label']==original['label'] and actual['group_id']==original['group_id'],'Label/time group differs from metadata')
        require(set(actual['scores'])==set(result['methods']),'Method roster differs')
    targets=cohort.target_labels;require(list(targets)==result['target_labels'],'Target roster differs')
    for key in ('roster_sha256','quality_admission_sha256','prior_exposure_admission_sha256'):
        require(result['cohort_provenance'][key]==cohort.provenance()[key],'Cohort policy changed: '+key)
    for path,value in result['source_sha256'].items():require(sha(ROOT/path)==value,'Inference source changed: '+path)
    groups=np.asarray([r['group_id'] for r in rows]);labels=np.asarray([r['label'] for r in rows])
    group_names,group_index=np.unique(groups,return_inverse=True)
    reports={}
    for method,reported in result['methods'].items():
        values=np.array([[r['scores'][method][t]['score'] for t in targets] for r in rows],np.float64)
        require(np.isfinite(values).all(),'Nonfinite score')
        recalls=[];rates=[];aucs=[];checked=0
        for j,target in enumerate(targets):
            truth=labels==target;pos=int(truth.sum());neg=len(truth)-pos
            require(pos>0 and neg>0 and reported['per_target'][target]['available'],'Missing target outcomes')
            auc=roc_auc_score(truth,values[:,j]);ap=average_precision_score(truth,values[:,j])
            for point,budget in [('budget_10_per10000',10),('zero_calibration_false_alarms',0)]:
                record=reported['per_target'][target][point];threshold=record['cutoff']['threshold'];m=record['metrics']
                if calibration:
                    require(record['cutoff']==calibration['methods'][method]['per_target'][target][point]['cutoff'],
                            'TEST cutoff differs from the frozen CALIB cutoff')
                if result['mode']=='calibration':
                    negatives=np.sort(values[~truth,j])[::-1];allowed=math.floor(neg*budget/10000)
                    cutoff=float(np.nextafter(negatives[allowed],np.inf))
                    require(cutoff==threshold,'CALIB cutoff does not follow its declared rule')
                detected=values[:,j]>=threshold;tp=int((detected&truth).sum());fp=int((detected&~truth).sum())
                for key,value in [('true_positive',tp),('false_positive',fp),('positive_events',pos),('negative_events',neg)]:
                    require(m[key]==value,'Count mismatch: '+method+'/'+target+'/'+key)
                for key,value in [('recall',tp/pos),('false_positive_rate',fp/neg),('false_alarms_per_10000_negatives',10000*fp/neg),('auroc',auc),('average_precision',ap)]:
                    close(m[key],value,method+'/'+target+'/'+key)
                group_positive=np.bincount(group_index,weights=truth,minlength=len(group_names))>0
                group_alarm=np.bincount(group_index,weights=detected,minlength=len(group_names))>0
                negative_group_count=int((~group_positive).sum())
                false_groups=int((group_alarm&~group_positive).sum())
                require(m['negative_time_groups']['groups']==negative_group_count and m['negative_time_groups']['groups_with_any_false_alarm']==false_groups,'Grouped count mismatch')
                if point=='budget_10_per10000':recalls.append(tp/pos);rates.append(10000*fp/neg);aucs.append(auc)
                checked+=1
        for key,value in [('recall',np.mean(recalls)),('false_alarms_per10000',np.mean(rates)),('auroc',np.mean(aucs))]:
            close(reported['macro'][key],value,method+'/macro/'+key)
        for point in ('budget_10_per10000','zero_calibration_false_alarms'):
            cutoffs=np.array([reported['per_target'][t][point]['cutoff']['threshold'] for t in targets])
            accepted=values>=cutoffs;target_truth=labels[:,None]==np.array(targets)[None,:]
            wrong=(accepted&~target_truth).any(1);recovered=(accepted&target_truth).any(1)
            u=reported['union_events'][point]
            require(u['screened_events']==len(rows) and u['wrong_target_alarm_events']==int(wrong.sum()) and u['target_recovered_events']==int(recovered.sum()),'Multi-target union count differs')
        reports[method]={'target_operating_points_checked':checked,'macro_recall':float(np.mean(recalls)),
                         'macro_false_alarms_per10000':float(np.mean(rates)),'macro_auroc':float(np.mean(aucs))}
    if result['selection']:
        ordered=['learned:bag','learned:translation_006','learned:translation_012']
        chosen=max(ordered,key=lambda x:(reports[x]['macro_recall'],reports[x]['macro_auroc'],-ordered.index(x)))
        require(chosen==result['selection']['selected_method'],'Declared CALIB selection differs')
    out={'valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),'mode':result['mode'],
         'events':len(rows),'targets':len(targets),'methods':reports,'result_sha256':sha(result_path),
         'score_log_sha256':sha(score_path),'audit_source_sha256':sha(__file__),
         'checks':['Exact admitted event roster, labels and time groups','All target counts at both operating points',
                   'AUROC/AP recomputed with sklearn, tolerance1e-12','Global CALIB recipe rule',
                   'Multi-target event union and negative-group counts'],
         'image_inference_calls':0,'scope':'Independent saved-score arithmetic/provenance audit; does not establish annotation correctness or physical independence.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'valid':True,'mode':result['mode'],'events':len(rows),'methods':len(reports),'output':str(output),'sha256':sha(output)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--admission',type=Path)
    a=p.parse_args();audit(a.results,a.output,a.admission)
