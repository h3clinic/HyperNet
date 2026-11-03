"""Independent saved-score arithmetic audit using sklearn ranking references.

Does not import production metrics, read images, run models or fit TEST cutoffs.
"""
from pathlib import Path
from collections import defaultdict
import argparse, hashlib, json, math
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, average_precision_score

METHODS=['v16','v16_no_relations','random','v15','raw48']
RECIPES=['mean','native','context128']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def load(path):
    with np.load(path,allow_pickle=False)as z:return {key:z[key].copy()for key in z.files}
def cp(k,n):return None if not n else [0. if not k else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def upper(k,n):return None if not n else 1. if k==n else float(beta.ppf(.95,k+1,n-k))
def cutoff(negative):
    negative=np.asarray(negative);allow=len(negative)//1000
    return float(np.nextafter(np.partition(negative,len(negative)-1-allow)[len(negative)-1-allow],np.inf))
def equal(actual,expected,path,counter):
    if expected is None:assert actual is None,path
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected),path
        for i,value in enumerate(expected):equal(actual[i],value,path+f'[{i}]',counter)
    elif isinstance(expected,(bool,int,np.integer)):
        assert actual==expected,(path,actual,expected);counter[0]+=1
    else:
        assert np.isclose(actual,expected,rtol=1e-12,atol=1e-12),(path,actual,expected);counter[0]+=1

def check_method(scores,labels,targets,groups,thresholds,reported,fitted,counter):
    n,t=scores.shape;recovered=np.zeros(n,bool);wrong=np.zeros(n,bool);known=np.isin(labels,targets)
    macro=defaultdict(list);accepted=scores>=thresholds[None]
    for j,target in enumerate(targets):
        p=labels==target;neg=~p;k=int(p.sum());nn=int(neg.sum());yes=accepted[:,j]
        tp=int((yes&p).sum());fp=int((yes&neg).sum());rec=tp/k if k else None
        auc=float(roc_auc_score(p,scores[:,j]))if k and nn else None
        ap=float(average_precision_score(p,scores[:,j]))if k else None
        fpr=fp/nn if nn else None;row=reported['per_target'][str(target)]
        expected={'positive_events':k,'negative_events':nn,'true_positive':tp,'false_positive':fp,
                  'false_negative':k-tp,'true_negative':nn-fp,'recall':rec,'precision':tp/(tp+fp)if tp+fp else None,
                  'false_positive_rate':fpr,'false_alarms_per_10000_negatives':10000*fpr if nn else None,
                  'auroc':auc,'average_precision':ap,'threshold':float(thresholds[j]),'available':bool(k and nn)}
        for key,value in expected.items():equal(row[key],value,f'{target}.{key}',counter)
        reference=row['independent_event_binomial_reference']
        equal(reference['recall_ci95'],cp(tp,k),str(target)+'.recall_ci95',counter)
        equal(reference['false_alarms_per_10000_ci95'],[10000*x for x in cp(fp,nn)]if nn else None,str(target)+'.fp_ci95',counter)
        equal(reference['false_alarms_per_10000_upper95'],10000*upper(fp,nn)if nn else None,str(target)+'.fp_upper95',counter)
        clusters=defaultdict(lambda:[0,0,False])
        for g,positive,alarm in zip(groups,p,yes):
            clusters[str(g)][0]+=int(positive);clusters[str(g)][1]+=1;clusters[str(g)][2]|=bool(alarm)
        negative_groups=[v for v in clusters.values()if v[0]==0];gn=len(negative_groups);gfp=sum(v[2]for v in negative_groups)
        group_expected={'groups':gn,'groups_with_any_false_alarm':gfp,
            'mixed_label_groups_excluded_from_group_rate':sum(0<v[0]<v[1]for v in clusters.values()),
            'any_alarm_groups_per_10000_negative_groups':10000*gfp/gn if gn else None,
            'any_alarm_groups_per_10000_upper95':10000*upper(gfp,gn)if gn else None}
        for key,value in group_expected.items():equal(row['negative_time_groups'][key],value,str(target)+'.group.'+key,counter)
        if fitted:
            assert cutoff(scores[neg,j])==thresholds[j],str(target)+' independent cutoff mismatch'
            cut=row['cutoff'];equal(cut['calibration_negatives'],nn,'cutoff.negatives',counter)
            equal(cut['allowed_false_alarms'],nn//1000,'cutoff.allowed',counter);equal(cut['observed_false_alarms'],fp,'cutoff.observed',counter)
        recovered|=yes&p;wrong|=yes&neg
        for key,value in [('recall',rec),('false_alarms_per10000',10000*fpr if nn else None),('auroc',auc),('average_precision',ap)]:macro[key].append(value)
    for key,values in macro.items():equal(reported['macro'][key],None if None in values else float(np.mean(values)),'macro.'+key,counter)
    false=int(wrong.sum());nk=int(known.sum());nr=int(recovered.sum());gall=len(set(groups));gw=len(set(groups[wrong]))
    union={'screened_events':n,'wrong_target_alarm_events':false,'wrong_target_alarm_events_per10000_screened_events':10000*false/n,
        'wrong_target_alarm_event_binomial_ci95_per10000':[10000*x for x in cp(false,n)],
        'events_containing_an_enrolled_target':nk,'target_recovered_events':nr,'target_recovery_recall':nr/nk if nk else None,
        'target_recovery_binomial_ci95':cp(nr,nk),'background_events':n-nk,'background_events_with_any_alarm':int((~known&accepted.any(1)).sum()),
        'screened_time_groups':gall,'time_groups_with_any_wrong_alarm':gw,'wrong_alarm_time_groups_per10000_groups':10000*gw/gall,
        'wrong_alarm_time_group_binomial_upper95_per10000':10000*upper(gw,gall)}
    for key,value in union.items():equal(reported['union'][key],value,'union.'+key,counter)
    equal(reported['thresholds'],thresholds.tolist(),'thresholds',counter)
    return {'macro_recall':None if None in macro['recall']else float(np.mean(macro['recall'])),
            'macro_auroc':None if None in macro['auroc']else float(np.mean(macro['auroc']))}

def audit(folder,calibration=None):
    folder=Path(folder);rp=folder/'results.json';result=read(rp);sp=folder/'scores.npz';data=load(sp)
    assert result['completed']is True and sha(sp)==result['scores_sha256']
    assert data['methods'].tolist()==METHODS and np.isfinite(data['scores']).all()and np.isfinite(data['thresholds']).all()
    assert data['scores'].shape==(5,len(data['labels']),len(data['targets']))
    assert len(set(data['event_ids']))==len(data['labels'])==result['events']
    assert data['targets'].tolist()==result['target_labels']
    bound={str(rp.resolve()):sha(rp),str(sp.resolve()):sha(sp)};counter=[0]
    fitted=result['stage']=='calibration'
    for i,method in enumerate(METHODS):check_method(data['scores'][i],data['labels'],data['targets'],data['group_ids'],data['thresholds'][i],result['metrics'][method],fitted,counter)
    if fitted:
        path=folder/'candidate_scores.npz';assert sha(path)==result['candidate_scores_sha256'];c=load(path);bound[str(path.resolve())]=sha(path)
        names=[m+':'+r for m in METHODS[:3]for r in RECIPES]+['v15','raw48'];assert c['methods'].tolist()==names
        for key in ['event_ids','labels','group_ids','targets','pixel_sha256']:assert np.array_equal(data[key],c[key])
        candidates={}
        for recipe in RECIPES:
            matrix=c['scores'][names.index('v16:'+recipe)];cuts=np.array([cutoff(matrix[data['labels']!=target,j])for j,target in enumerate(data['targets'])])
            candidates[recipe]=check_method(matrix,data['labels'],data['targets'],data['group_ids'],cuts,result['candidate_metrics_primary'][recipe],True,counter)
        selected=max(RECIPES,key=lambda r:(candidates[r]['macro_recall'],candidates[r]['macro_auroc'],-RECIPES.index(r)))
        assert selected==result['selected_recipe']
        selected_names=[m+':'+selected for m in METHODS[:3]]+['v15','raw48']
        for i,name in enumerate(selected_names):assert np.array_equal(data['scores'][i],c['scores'][names.index(name)])
    else:
        assert calibration is not None,'Original CALIB folder is required for TEST audit'
        cp_path=Path(calibration)/'scores.npz';c=load(cp_path);bound[str(cp_path.resolve())]=sha(cp_path)
        cr_path=Path(calibration)/'results.json';cr=read(cr_path);bound[str(cr_path.resolve())]=sha(cr_path)
        assert sha(cp_path)==cr['scores_sha256']and result['selected_recipe']==cr['selected_recipe']
        for key in ['methods','targets','thresholds']:assert np.array_equal(data[key],c[key])
        for key in ['event_ids','group_ids','pixel_sha256']:assert not np.intersect1d(data[key],c[key]).size
        log_path=folder/'event_scores.jsonl';assert sha(log_path)==result['score_log_sha256'];bound[str(log_path.resolve())]=sha(log_path)
        with log_path.open(encoding='utf-8')as f:rows=[json.loads(line)for line in f]
        assert len(rows)==len(data['event_ids'])
        for i,row in enumerate(rows):
            assert row['event_id']==data['event_ids'][i]and row['pixel_sha256']==data['pixel_sha256'][i]
            for j,method in enumerate(METHODS):assert np.array_equal(row['scores'][method],data['scores'][j,i])
    for path,expected in bound.items():assert sha(path)==expected
    return {'format':'hypernet.v16.independent_saved_score_audit.v1','passed':True,'stage':result['stage'],
        'events':len(data['labels']),'methods':METHODS,'targets':data['targets'].tolist(),'numeric_values_checked':counter[0],
        'all_target_score_values':int(data['scores'].size),'sources':{str(Path(__file__).resolve()):sha(__file__)},'bound_files':bound,
        'ranking_reference':'sklearn roc_auc_score and average_precision_score','cutoff_reference':'independent partition order statistic plus nextafter; CALIB only',
        'group_reference':'independently accumulated per-target group counts and event unions','no_model_or_image_calls':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--calibration',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(a.folder,a.calibration);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'output':str(a.output),'sha256':sha(a.output),'numbers':result['numeric_values_checked']}))
