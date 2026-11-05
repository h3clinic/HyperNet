"""One positive reference plus explicitly counted labeled background calibration.

Fixed group split and operating budgets, never optimized against evaluation labels.
Existing v22 cohorts: retrospective holdout analysis, not fresh project data.
"""
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import beta
R=Path(__file__).resolve().parent;OLD=R.parent/'hypernet_v22'
sys.path.insert(0,str(OLD.parent/'hypernet_v20/evaluation'))
from anomaly_metrics import ranking_metrics
DATASETS=['gravity','hirise','serengeti'];ALPHAS=[.05,.01,.001,.0001]
def read(p):return json.loads(p.read_text())
def save(p,obj):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cutoff(background,alpha):
    n=len(background);k=math.ceil((n+1)*(1-alpha))
    return float(np.sort(background)[k-1]) if 1<=k<=n else None
def upper(k,n):return float(beta.ppf(.95,k+1,n-k)) if n and k<n else (1. if n else None)
def protocol():
    destination=R/'protocol.json'
    if destination.exists():return read(destination)
    plans={}
    for ds in DATASETS:
        p=read(OLD/(ds+'_protocol.json'));rec=p['records'];group={}
        if ds=='gravity':
            for ifo in sorted({x['ifo'] for x in rec}):
                rows=sorted((x for x in rec if x['ifo']==ifo and x['role']=='query'),key=lambda x:x['event_time']);last=-1e30;key=None
                for x in rows:
                    if x['event_time']-last>8:key=ifo+'|'+x['event_id']
                    group[x['event_id']]=key;last=x['event_time']
        else:
            for x in rec:
                if x['role']=='query':group[x['event_id']]=x['source_image'] if ds=='hirise' else x['camera']
        groups=sorted(set(group.values()),key=lambda k:hashlib.sha256(('v23|'+ds+'|'+k).encode()).hexdigest());cal=set(groups[:len(groups)//2])
        plans[ds]={'source_protocol_sha256':digest(OLD/(ds+'_protocol.json')),'records':[{'event_id':x['event_id'],'group':group.get(x['event_id']),'split':'reference' if x['role']=='reference' else ('calibration' if group[x['event_id']] in cal else 'evaluation')} for x in rec],'calibration_groups':len(cal),'evaluation_groups':len(groups)-len(cal)}
    obj={'positive_references_per_target':1,'extra_data':'Labeled non-target examples used only to set alarm thresholds. Not a one-total-image method. No model fitting or extra positive exemplars.','freshness':'Repartition of already inspected v22 cohorts, not a fresh project holdout. Fixed before v23 score inspection.','grouping':'Gravity same detector and successive gaps <=8sec; Mars source exposure; wildlife camera. Wildlife evaluation unit is sequence max over frames.','alpha_per_target':ALPHAS,'decision':'score > sorted negative calibration score at ceil((n+1)*(1-alpha)); reject all if insufficient calibration resolution','guarantee':'Finite-sample marginal interpretation requires exchangeable scores. Within-group dependence and domain shift mean these runs do not certify deployment FPR. Budgets are per target, not simultaneous across target types.','selection':'No threshold or model selection using evaluation scores; all budgets and all available models reported.','source':'https://www.stat.berkeley.edu/~ryantibs/papers/conformal-jasa.pdf','datasets':plans}
    save(destination,obj);return obj
def run():
    p=protocol();results=[]
    for ds in DATASETS:
        rec=read(OLD/(ds+'_protocol.json'))['records'];split=p['datasets'][ds]['records'];assert len(rec)==len(split)
        assert digest(OLD/(ds+'_protocol.json'))==p['datasets'][ds]['source_protocol_sha256']
        files={'meta_dinov2_small':R/'gravity_meta_features.npy' if ds=='gravity' else OLD/(ds+'_meta_features.npy')}
        for arm in ['run_w64_d2','run_w192_d4','run_w384_d6']:
            if (OLD/arm/'retrieval_protocol.json').exists():files[arm]=OLD/arm/(ds+'_features.npy')
        queries=np.array([i for i,x in enumerate(rec) if x['role']=='query']);groups={}
        for i in queries:groups.setdefault(rec[i].get('group_id',rec[i]['event_id']),[]).append(i)
        units=list(groups.values());unit_split=np.array([split[g[0]]['split'] for g in units]);unit_group=[split[g[0]]['group'] for g in units]
        for method,path in files.items():
            if not path.exists():continue
            f=np.load(path);assert len(f)==len(rec) and np.isfinite(f).all()
            for ri,ref in enumerate(rec):
                if ref['role']!='reference':continue
                scores=f@f[ri];s=np.array([max(scores[g]) for g in units]);y=np.array([ref['label'] in rec[g[0]].get('labels',[rec[g[0]]['label']]) for g in units])
                c=(unit_split=='calibration')&~y;e=unit_split=='evaluation';background=s[c];sy=s[e];ey=y[e];metric=ranking_metrics(ey,sy) if ey.any() and (~ey).any() else {'auroc':None,'average_precision':None}
                old=[]
                if method=='meta_dinov2_small':old=[x for x in read(OLD/(ds+'_results.json')) if x['method']=='dinov2_small' and x['target']==ref['label']]
                for alpha in ALPHAS:
                    threshold=cutoff(background,alpha);accept=sy>threshold if threshold is not None else np.zeros(len(sy),bool);tp=int((accept&ey).sum());fp=int((accept&~ey).sum());pos=int(ey.sum());neg=int((~ey).sum())
                    row={'dataset':ds,'method':method,'target':ref['label'],'reference_id':ref['event_id'],'alpha':alpha,'calibration_negatives':len(background),'calibration_groups':len({unit_group[j] for j in np.flatnonzero(c)}),'test_positives':pos,'test_negatives':neg,'auroc':metric['auroc'],'average_precision':metric['average_precision'],'threshold':threshold,'supported_resolution':threshold is not None,'tp':tp,'fp':fp,'recall':tp/pos if pos else None,'false_alarms_per_10000':10000*fp/neg if neg else None,'binomial_upper95_per_10000':10000*upper(fp,neg) if neg else None,'interval_caveat':'Binomial independent-unit assumption; descriptive only for correlated images/sequences','feature_sha256':digest(path)}
                    if old:
                        accepted=sy>=old[0]['threshold'];row['previous_reference_only']={'tp':int((accepted&ey).sum()),'fp':int((accepted&~ey).sum())}
                    results.append(row)
    save(R/'results.json',results)
    return results
if __name__=='__main__':run()
