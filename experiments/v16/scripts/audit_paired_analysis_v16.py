"""Independent saved-score bootstrap arithmetic; no production analysis import."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def interval(values):
    values=sorted(map(float,values));out=[]
    for probability in [.025,.975]:
        index=(len(values)-1)*probability;lo=math.floor(index);hi=math.ceil(index);weight=index-lo
        out.append(values[lo]*(1-weight)+values[hi]*weight)
    return out
def run(scores_path,analysis_path,plan_path):
    plan=read(plan_path);reported=read(analysis_path);saved=reported['bootstrap']
    with np.load(scores_path,allow_pickle=False)as z:data={key:z[key].copy()for key in z.files}
    methods=data['methods'].tolist();targets=data['targets'].tolist();labels=data['labels'];groups=data['group_ids']
    assert methods==plan['methods'] and saved['available'] is True
    order={group:i for i,group in enumerate(sorted(set(groups)))};g=len(order);m=len(methods);t=len(targets)
    group_size=np.zeros(g);positive=np.zeros((g,t));correct=np.zeros((g,m,t));incorrect=np.zeros((g,m,t))
    target_index={label:i for i,label in enumerate(targets)}
    for event,(label,group)in enumerate(zip(labels,groups)):
        i=order[group];group_size[i]+=1;truth=np.zeros(t,bool)
        if label in target_index:truth[target_index[label]]=True
        positive[i]+=truth;accepted=data['scores'][:,event]>=data['thresholds']
        correct[i]+=accepted&truth;incorrect[i]+=accepted&~truth
    assert np.all(positive.sum(0)>0)and np.all(group_size.sum()-positive.sum(0)>0)
    rng=np.random.default_rng(plan['bootstrap_seed']);wanted=plan['valid_bootstrap_draws'];batch=plan['bootstrap_batch_size']
    recall=[];fp=[];attempted=rejected=unused=0
    while len(recall)<wanted:
        draws=rng.multinomial(g,np.repeat(1/g,g),size=batch)
        denom_p=draws@positive;denom_n=(draws@group_size)[:,None]-denom_p
        num_tp=(draws@correct.reshape(g,-1)).reshape(batch,m,t)
        num_fp=(draws@incorrect.reshape(g,-1)).reshape(batch,m,t)
        for i in range(batch):
            if len(recall)==wanted:unused+=batch-i;break
            attempted+=1
            if np.any(denom_p[i]==0)or np.any(denom_n[i]==0):rejected+=1;continue
            recall.append((num_tp[i]/denom_p[i]).mean(1));fp.append(10000*(num_fp[i]/denom_n[i]).mean(1))
    samples={'macro_recall':np.array(recall),'macro_false_alarms_per10000_negatives':np.array(fp)}
    point={'macro_recall':(correct.sum(0)/positive.sum(0)).mean(1),
        'macro_false_alarms_per10000_negatives':10000*(incorrect.sum(0)/(group_size.sum()-positive.sum(0))).mean(1)}
    numbers=0
    def compare(a,b):
        nonlocal numbers
        assert np.allclose(a,b,atol=1e-12,rtol=1e-12),(a,b);numbers+=np.asarray(b).size
    for key,actual in [('valid_draws',wanted),('attempted_draws',attempted),('rejected_draws',rejected),('generated_but_unused_draws',unused)]:assert saved[key]==actual
    for index,method in enumerate(methods):
        for metric,values in samples.items():
            row=saved['methods'][method][metric];compare(row['estimate'],point[metric][index]);compare(row['ci95'],interval(values[:,index]))
    primary=methods.index(plan['primary'])
    for control in plan['contrasts']:
        index=methods.index(control)
        for metric,values in samples.items():
            row=saved['contrasts'][plan['primary']+'_minus_'+control][metric]
            compare(row['difference'],point[metric][primary]-point[metric][index]);compare(row['ci95'],interval(values[:,primary]-values[:,index]))
    return {'format':'hypernet.v16.independent_paired_analysis_review.v1','passed':True,'numeric_estimates_and_interval_endpoints_checked':int(numbers),
        'valid_draws':wanted,'attempted_draws':attempted,'rejected_draws':rejected,'generated_but_unused_draws':unused,
        'time_groups':g,'events':len(groups),'fixed_target_count':t,'methods':methods,
        'reference':'Independent event-loop group sufficient statistics; shared seeded group multinomial draws; manually sorted linear percentile interpolation.',
        'source_sha256':sha(__file__),'input_sha256':{str(Path(p).resolve()):sha(p)for p in [scores_path,analysis_path,plan_path]},
        'no_model_or_image_calls':True,'no_threshold_refitting':True,
        'limits':'Conditional on fixed references, trained checkpoints, calibration and chosen target set. Pointwise intervals, not simultaneous superiority statements; observed false-alarm rates differ across methods.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=run(a.scores,a.analysis,a.plan)
    with a.output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'output':str(a.output),'sha256':sha(a.output),'numbers':result['numeric_estimates_and_interval_endpoints_checked']}))
