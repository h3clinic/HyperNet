"""Post hoc paired DEV ranking analysis. No neural inference or policy updates.

Only already-used DEV score arrays are read. Fixed-FPR ROC summaries are
descriptive, not deployable thresholds or an untouched confirmation result.
"""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import hashlib,json,time
from datetime import datetime,timezone
import numpy as np
import sklearn
from sklearn.metrics import roc_curve,roc_auc_score,auc

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TARGETS=['Low_Frequency_Burst','Low_Frequency_Lines','Scattered_Light','Blip']
FPRS=[.001,.01]
BOOTSTRAPS=2000
SEED=20261105
MODELS=['OnceNet_strict_fp32_joint4_s45029','DINOv2_small_late_mean4']
METRICS=['auroc','recall_interpolated_fpr_0.001','recall_empirical_fpr_at_most_0.001','pauc_standardized_0.001',
         'recall_interpolated_fpr_0.01','recall_empirical_fpr_at_most_0.01','pauc_standardized_0.01']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def curve_metrics(y,s,weights=None,details=False):
    fpr,tpr,_=roc_curve(y,s,sample_weight=weights,drop_intermediate=False)
    assert np.isfinite(fpr).all() and np.isfinite(tpr).all()
    result={'auroc':float(auc(fpr,tpr))}; points=[]
    for a in FPRS:
        stop=int(np.searchsorted(fpr,a,side='right'))
        recall=float(np.interp(a,fpr,tpr))
        xx=np.r_[fpr[:stop],a]; yy=np.r_[tpr[:stop],recall]
        partial=float(auc(xx,yy))
        standardized=.5*(1+(partial-.5*a*a)/(a-.5*a*a))
        ix=np.flatnonzero(fpr<=a+1e-15)[-1]
        result[f'recall_interpolated_fpr_{a}']=recall
        result[f'recall_empirical_fpr_at_most_{a}']=float(tpr[ix])
        result[f'pauc_standardized_{a}']=float(standardized)
        if details:
            positive=int(y.sum());negative=int((~y).sum())
            points.append({'fpr_budget':a,'nominal_fa_per10000':a*10000,
                'interpolated_recall':recall,'interpolation_is_not_a_deterministic_threshold':True,
                'empirical_recall':float(tpr[ix]),'empirical_fpr':float(fpr[ix]),
                'empirical_tp':int(round(tpr[ix]*positive)),'empirical_fp':int(round(fpr[ix]*negative)),
                'empirical_fa_per10000':float(fpr[ix]*10000),
                'raw_partial_area':partial,'standardized_partial_auc':float(standardized),
                'negative_count':negative,'positive_count':positive})
    return result,points

def sanity_checks():
    rng=np.random.default_rng(31);count=0
    for s in [np.zeros(100),np.r_[np.zeros(50),np.ones(50)],rng.normal(size=100),rng.integers(0,5,100).astype(float)]:
        y=np.r_[np.zeros(50,bool),np.ones(50,bool)]
        for weights in (None,rng.integers(1,5,100)):
            got,_=curve_metrics(y,s,weights)
            for a in FPRS:
                assert np.isclose(got[f'pauc_standardized_{a}'],roc_auc_score(y,s,max_fpr=a,sample_weight=weights),atol=1e-12)
                count+=1
    tied,_=curve_metrics(y,np.zeros(100));perfect,_=curve_metrics(y,y.astype(float))
    assert all(tied[f'pauc_standardized_{a}']==.5 for a in FPRS)
    assert all(perfect[f'pauc_standardized_{a}']==1 for a in FPRS)
    assert all(perfect[f'recall_empirical_fpr_at_most_{a}']==1 for a in FPRS)
    return {'passed':True,'independent_sklearn_partial_auc_checks':count,'cases':['all ties','perfect separation','continuous scores','mixed ties','nonuniform weights']}

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/'matched_operating_point_development.json'
    plan_path=OUT/'matched_operating_point_development_plan.json'
    assert not output.exists(),'Completed descriptive analysis exists; do not rerun unnecessarily.'
    v46=ROOT/'outputs/hypernet_v46';v47=ROOT/'outputs/hypernet_v47';gs=ROOT/'outputs/cloud_plan_20261105/gs_fp32'
    paths={'records':v46/'data/records.json','splits':v47/'splits.json','old_protocol':v47/'protocol.json',
        'old_results':v47/'evaluation/results.json','once_scores':gs/'scores_decisions.npz',
        'once_binding':gs/'pre_dev_binding.json','once_manifest':gs/'migration_manifest.json',
        'once_existing_metrics':gs/'development_audit.json','dino_scores':v47/'evaluation/dino_small/fused.npz',
        'locked_comparison':ROOT/'outputs/hypernet_locked/comparison.json','script':Path(__file__)}
    hashes={k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in paths.items()}
    # Bind analytical choices before opening score arrays. Already-used metrics
    # and model selection are explicitly known; this is not preregistration.
    plan={'analysis':'post_hoc_development_only','created_utc':datetime.now(timezone.utc).isoformat(),
        'models':MODELS,'targets':TARGETS,'fpr_budgets':FPRS,'bootstraps':BOOTSTRAPS,'rng_seed':SEED,
        'metrics':METRICS,'source_files':hashes,'reserved_data_reads':False,'new_inference':False,
        'deployment_threshold_updates':False,'prespecified_for_this_descriptive_run_not_preregistered_confirmation':True}
    assert not plan_path.exists(),'Existing analytical plan should not be overwritten'
    write(plan_path,plan)
    rows=read(paths['records']);sp=read(paths['splits']);protocol=read(paths['old_protocol']);old=read(paths['old_results'])
    migration=read(paths['once_manifest']);binding=read(paths['once_binding']);old_once=read(paths['once_existing_metrics'])
    assert old['complete'] and old['development_only'] and old['reserve_closed']
    assert old['protocol_sha256']==sha(paths['old_protocol'])
    for p in (paths['records'],paths['splits']):assert protocol['files'][str(p.resolve())]==sha(p)
    assert old['files'][str(paths['dino_scores'].resolve())]==sha(paths['dino_scores'])
    assert binding['files']['scores_decisions.npz']==sha(paths['once_scores'])
    assert binding['migration_manifest_sha256']==sha(paths['once_manifest'])
    assert old_once['migration_manifest_sha256']==sha(paths['once_manifest'])
    assert sp['audits']['passed'] and sp['audits']['no_reserved_id_overlap']
    dev=np.array(sp['indices']['development'],dtype=np.int64)
    assert list(dev)==migration['development_indices'] and len(dev)==1558
    assert [r['index'] for r in rows]==list(range(8250))
    assert all(rows[i]['split']=='development' for i in dev)
    ids=[rows[i]['event_id'] for i in dev];assert len(set(ids))==1558
    labels=np.array([rows[i]['label'] for i in dev]); groups=[rows[i]['group'] for i in dev]
    group_names=sorted(set(groups)); group_index={g:i for i,g in enumerate(group_names)}
    membership=np.array([group_index[g] for g in groups])
    references={}
    for t in TARGETS:
        a=sp['targets'][t];b=migration['targets'][t]
        assert a['reference']==b['reference'] and a['fit']==b['fit'] and a['original_cal']==b['cal']
        assert not set(dev)&(set(a['fit'])|set(a['original_cal'])|{a['reference']})
        references[t]=rows[a['reference']]['event_id']
    # Model configurations were selected previously, not from this analysis.
    compare=read(paths['locked_comparison'])
    dino_line=next(z for z in compare['direct_comparisons'] if z['label'].startswith('DINOv2-small') and 'strongest' in z['label'])
    assert 'late_mean4' in dino_line['note']
    scores={}
    with np.load(paths['once_scores'],allow_pickle=False) as z:
        scores[MODELS[0]]={t:np.array(z['scores__'+t][dev],float) for t in TARGETS}
    with np.load(paths['dino_scores'],allow_pickle=False) as z:
        scores[MODELS[1]]={t:np.array(z['late_mean4__'+t][dev],float) for t in TARGETS}
    assert all(np.isfinite(s).all() for target in scores.values() for s in target.values())
    checks=sanity_checks();target_rows=[];values=np.zeros((2,4,len(METRICS)))
    for mi,model in enumerate(MODELS):
        for ti,target in enumerate(TARGETS):
            y=labels==target;score=scores[model][target];m,points=curve_metrics(y,score,details=True)
            saved=next(x for x in old_once['rows'] if x['version']=='strict_fp32' and x['target']==target and x['alpha']==.001) if mi==0 else next(x for x in old['rows'] if x['model']=='dino_small' and x['target']==target and x['matcher']=='late_mean4' and x['pool']=='original' and x['grouping']=='pooled' and x['alpha']==.001)
            for metric,oldkey in [('auroc','auroc'),('pauc_standardized_0.001','pauc_std_001'),('pauc_standardized_0.01','pauc_std_01')]:
                assert np.isclose(m[metric],saved[oldkey],rtol=0,atol=1e-12),(model,target,metric)
            values[mi,ti]=[m[k] for k in METRICS]
            target_rows.append({'model':model,'target':target,'reference_id':references[target],'positive':int(y.sum()),'negative':int((~y).sum()),'metrics':m,'operating_points':points,'score_sha256':array_hash(score)})
    rng=np.random.default_rng(SEED);reps=np.empty((BOOTSTRAPS,2,4,len(METRICS)),float)
    for rep in range(BOOTSTRAPS):
        draw=rng.integers(0,len(group_names),len(group_names))
        weights=np.bincount(draw,minlength=len(group_names))[membership]
        for mi,model in enumerate(MODELS):
            for ti,target in enumerate(TARGETS):
                m,_=curve_metrics(labels==target,scores[model][target],weights)
                reps[rep,mi,ti]=[m[k] for k in METRICS]
        if (rep+1)%500==0:print(f'Paired DEV bootstrap {rep+1}/{BOOTSTRAPS}',flush=True)
    macros=[];contrasts=[]
    mean_reps=reps.mean(2)
    for mi,model in enumerate(MODELS):
        macros.append({'model':model,'metrics':{k:float(values[mi,:,ki].mean()) for ki,k in enumerate(METRICS)}})
    for ki,k in enumerate(METRICS):
        difference=mean_reps[:,0,ki]-mean_reps[:,1,ki]
        contrasts.append({'metric':k,'difference_oncenet_minus_dino':float(values[0,:,ki].mean()-values[1,:,ki].mean()),
            'paired_percentile95':np.quantile(difference,[.025,.975]).tolist()})
    equal=[]
    for t in TARGETS:
        aa=next(r for r in target_rows if r['model']==MODELS[0] and r['target']==t)
        bb=next(r for r in target_rows if r['model']==MODELS[1] and r['target']==t)
        for ap,bp in zip(aa['operating_points'],bb['operating_points']):
            equal.append({'target':t,'fpr_budget':ap['fpr_budget'],'achieved_fp_equal':ap['empirical_fp']==bp['empirical_fp'],
                'oncenet_fp':ap['empirical_fp'],'dino_fp':bp['empirical_fp'],'negative':aa['negative']})
    result={'complete':True,'analysis':'Post hoc matched-operating-point DEVELOPMENT analysis, not new validation',
        'plan_sha256':sha(plan_path),'source_files':hashes,'seconds':time.monotonic()-started,
        'new_inference':False,'model_fitting':False,'deployment_threshold_updates':False,'reserved_data_reads':False,
        'cohort':{'events':1558,'event_ids_in_shared_order':ids,'event_id_order_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'labels_excluded':[],'all_non_target_labels_remain_negative':True,'references':references,
            'event_groups':len(group_names),'source_split_audits':sp['audits'],'exact_same_ids_labels_reference_fit_cal':True},
        'definitions':{'interpolated_recall':'Linear empirical ROC interpolation at fixed FPR. May require randomization between thresholds; not a single deployed decision rule.',
            'empirical_recall':'Largest empirical TPR on a tied-score-respecting ROC with FPR at most the budget. DEV labels locate this descriptive point; no threshold is exported.',
            'standardized_partial_auc':'McClish correction: 0.5*(1+(partial_area-0.5*m*m)/(m-0.5*m*m)); m is max FPR. Random diagonal=0.5, perfect=1.',
            'macro':'Unweighted mean across four one-versus-rest targets; not a simultaneous four-target archive false-alarm rate.'},
        'targets':target_rows,'macro':macros,'paired_macro_contrasts':contrasts,'achieved_false_alarm_count_comparison':equal,
        'bootstrap':{'repetitions':BOOTSTRAPS,'seed':SEED,'method':'Paired nonparametric event-group bootstrap; identical draws used for both models and all targets; ROC metrics recalculated per draw.',
            'all_groups_singletons':len(group_names)==len(dev),'replicate_array_sha256':array_hash(reps),'interval':'2.5th and97.5th percentile of paired macro differences; exploratory'},
        'implementation_checks':checks,'existing_saved_ranking_metrics_reproduced':True,'library_versions':{'numpy':np.__version__,'sklearn':sklearn.__version__},
        'sources':['https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html'],
        'limitations':['Already-used DEV data and retrospective model selection; bootstrap intervals do not correct selection bias and are not confirmatory evidence.',
            'AtFPR0.001 only about1.3 negative examples per target are expected; finite-sample tail uncertainty is large.',
            'Interpolation and empirical ROC point selection use DEV labels and do not establish prospective calibrated low false alarms.',
            'Existing group/8-second separation does not guarantee independence across observing runs, detectors or longer timescales.',
            'One positive reference per target; labeled negative FIT/CAL and DINO negative-TUNE normalization remain part of the pipelines.',
            'Strict-FP32 OnceNet deployed scores differ slightly from historical TF32 ranks; historical alarms unchanged in migration audit.']}
    for info in hashes.values():assert sha(info['path'])==info['sha256'],'Source changed while analyzing'
    write(output,result)
    print(json.dumps({'macro':macros,'paired_macro_contrasts':contrasts,'exact_empirical_false_alarm_counts_match':all(r['achieved_fp_equal'] for r in equal),'output':str(output)},indent=2))

if __name__=='__main__':main()
