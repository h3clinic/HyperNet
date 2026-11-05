"""Verify committed results without training, inference, threshold fitting or resampling."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np

R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dt(x):return datetime.fromisoformat(x)

assert (R/'results.json').exists(),'Wait for completed results; no premature label join'
p=read(R/'protocol.json');pre=read(R/'independent_preflight_audit.json')
c=read(R/'scores_commitment.json');a=read(R/'acquisition.json');result=read(R/'results.json')
assert result['complete'] and c['complete'] and a['complete'] and pre['passed']
assert p['code_sha256']==pre['code_sha256']==c['code_sha256']==result['code_sha256']==sha(R/'run_confirmation.py')
assert sha(R/'protocol.json')==pre['protocol_sha256']==c['protocol_sha256']==a['protocol_sha256']==result['protocol_sha256']
assert sha(R/'scores_commitment.json')==result['score_commitment_sha256']
assert sha(R/'frozen_thresholds.json')==c['thresholds_sha256']
for f,h in c['files'].items():assert sha(Path(f))==h
for f,h in a['files'].items():assert sha(Path(f))==h
assert dt(p['frozen_utc'])<dt(pre['reviewed_utc'])<dt(a['completed_utc'])<dt(c['committed_utc'])<dt(result['completed_utc'])
assert datetime.fromtimestamp((R/'frozen_thresholds.json').stat().st_mtime,timezone.utc)<dt(p['frozen_utc'])
rows=read(R/'records_unlabelled.json');intent=read(R/'intent_unlabelled.json')
assert len(rows)==len(intent)==a['eligible_count']==result['eligible_count']==195
assert a['excluded']==result['excluded']==[]
assert [x['event_id'] for x in rows]==[x['event_id'] for x in intent]
assert all(set(r['windows'])=={'d05','d1','d2','d4'} for r in rows)
bindings=read(R/'source_bindings.json');assert len(bindings)==780 and all('error' not in b for b in bindings)

labels_by_id={r['event_id']:r['label'] for r in read(Path(p['labels_only_for_final_metrics']))['records']}
labels=np.array([labels_by_id[r['event_id']] for r in rows])
targets=list(p['references']);models=['oncenet','random','dino'];alphas=[.001,.01]
cuts=read(R/'frozen_thresholds.json')
def manual_auc(y,s):
    positive=s[y];negative=s[~y]
    return float(((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative)))
recomputed=[];max_error=0.
for m in models:
    z=np.load(R/(m+'_scores.npz'),allow_pickle=False)
    assert set(z.files)==set(targets)
    for alpha in alphas:
        metrics=[];preds=[]
        for target in targets:
            s=z[target];assert s.shape==(195,) and np.isfinite(s).all()
            y=labels==target;threshold=cuts[m][target][str(alpha)]['threshold']
            pred=s>(threshold if threshold is not None else np.inf);preds.append(pred)
            tp=int((pred&y).sum());fp=int((pred&~y).sum());tn=int((~pred&~y).sum());fn=int((~pred&y).sum())
            den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
            row={'tp':tp,'fp':fp,'tn':tn,'fn':fn,'positive':tp+fn,'negative':tn+fp,'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),'mcc':(tp*tn-fp*fn)/math.sqrt(den) if den else 0.,'recall':tp/(tp+fn),'fa_per10000':10000*fp/(fp+tn),'auroc':manual_auc(y,s)}
            saved=next(r for r in result['per_target'] if r['model']==m and r['target']==target and r['alpha']==alpha)
            for k,v in row.items():
                error=abs(v-saved[k]);max_error=max(max_error,error);assert error<1e-12,(m,target,alpha,k,v,saved[k])
            metrics.append(row)
        macro=next(r for r in result['macro'] if r['model']==m and r['alpha']==alpha)
        for k in ['balanced_accuracy','mcc','recall','fa_per10000','auroc']:
            assert abs(np.mean([r[k] for r in metrics])-macro[k])<1e-12
        assert sum(r['tp'] for r in metrics)==macro['tp']
        assert sum(r['fp'] for r in metrics)==macro['fp']
        bg=~np.isin(labels,targets);prediction=np.logical_or.reduce(preds)
        union=next(r for r in result['non_target_union'] if r['model']==m and r['alpha']==alpha)
        assert int(bg.sum())==union['negative']==26
        assert int(prediction[bg].sum())==union['fp']
        assert abs(prediction[bg].mean()*10000-union['fa_per10000'])<1e-12
        recomputed.append({'model':m,'alpha':alpha,'all_confusion_counts_and_macro_metrics_verified':True})
parity=read(R/'consumed_fixture_parity.json')
assert parity['passed'] and all(r['passed'] and all(t['primary_decision_changes']==0 for t in r['checks'].values()) for r in parity['rows'])
groups={(r['ifo'],int(r['event_time']//3600)) for r in rows}
assert result['bootstrap']['groups']==len(groups)
assert result['bootstrap']['requested']==2000 and 0<result['bootstrap']['valid']<=2000
for control,fields in result['paired_primary_intervals'].items():
    for metric,interval in fields.items():assert interval is None or interval['low']<=interval['high']

out={
 'passed':True,'reviewed_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Read-only artifact, chronology, cohort and metric verification of completed scores. No model inference, training, new scoring, threshold fitting or bootstrap repetition.',
 'code_sha256':p['code_sha256'],'protocol_sha256':sha(R/'protocol.json'),'results_sha256':sha(R/'results.json'),'commitment_sha256':sha(R/'scores_commitment.json'),
 'timeline_utc':{'protocol':p['frozen_utc'],'independent_preflight':pre['reviewed_utc'],'acquisition':a['completed_utc'],'all_model_scores':c['committed_utc'],'results':result['completed_utc']},
 'cohort':{'intent':195,'retained':195,'excluded':0,'source_pngs':780,'per_class':{str(k):int((labels==k).sum()) for k in sorted(set(labels))},'non_target_backgrounds':26,'detector_hour_clusters':len(groups)},
 'checks':{'all_committed_score_and_acquisition_hashes_match':True,'fixed_thresholds_precede_acquisition':True,'all_models_share_same_order_and_four_views':True,'all24_target_operating_point_confusion_rows_recomputed':True,'six_macro_rows_recomputed':True,'AUROC_independently_checked_by_pairwise_rank_formula':True,'any_target_background_counts_recomputed':True,'consumed_fixture_primary_alarm_parity_passed':True,'bootstrap_summary_schema_and_group_count_checked':True},
 'maximum_metric_reconstruction_error':max_error,'operating_points':recomputed,
 'not_repeated':['Neural inference','Calibration','Bootstrap intervals','Partial-AUC implementation (reviewed statically and independently exercised in prior descriptive script)'],
 'limitations':['Only26 outside-target backgrounds for union false alarms; target-versus-rest denominators117–183 are also small.','No false-alarm guarantee follows from a zero observed count.','The three models share fixed positive references; this does not test robustness to reference selection.','Comparator CPU numerical portability was checked on12 consumed events; zero primary flips there is not universal feature equivalence.','The source cohort was reserved for model outcomes, but its metadata labels were already visible.'],
}
(R/'independent_completion_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'maximum_metric_reconstruction_error':max_error,'primary_macro':[x for x in result['macro'] if x['alpha']==.001],'cohort':out['cohort']},indent=2))
