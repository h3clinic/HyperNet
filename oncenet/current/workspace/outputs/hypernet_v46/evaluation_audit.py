"""Independent CPU audit of the consumed-event duration/generator factorial.

No neural inference, test selection, threshold optimization, or reserved-image
access. The evaluator's fitter is NOT imported: positive-definite normal
equations uniquely verify its saved solutions. Frozen v45 metric implementations
independently group ROC ties and cross-check standardized pAUC against sklearn.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent; P=R.parent; V45=P/'hypernet_v45'; V44=P/'hypernet_v44'
sys.path.insert(0,str(V45))
from seed_development_audit import close, independent_fit_indices, independent_references, metric_values
from validation_audit import sha256
DATA=R/'data'; OUT=R/'evaluation'
SEEDS=[45011,45029,45047]; TRAININGS=['random','shapes','spectrogram']
TARGETS=['Low_Frequency_Burst','Low_Frequency_Lines','Scattered_Light','Blip']
DURATIONS=['d05','d1','d2','d4']; VIEWS=['single','multi']; ALPHAS=[.001,.0005]
REFS={'Low_Frequency_Burst':'632xGf8n9x','Low_Frequency_Lines':'Vengh4mj2t','Scattered_Light':'dac0D6ni5R','Blip':'yxumvoPABn'}
FIELDS=['balanced_accuracy','mcc','recall','fa_per10000','auroc','pauc_std_01','pauc_std_001']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(condition,message):
    if not condition:raise AssertionError(message)
def bindings(files):
    for path,digest in files.items():require(sha256(path)==digest,'Changed bound file: '+str(path))
def nested_close(actual,expected,name):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),'Unexpected dictionary fields: '+name)
        for key,value in expected.items():nested_close(actual[key],value,name+'/'+key)
    elif isinstance(expected,(int,float,np.number)) and not isinstance(expected,bool):close(actual,expected,name)
    else:require(actual==expected,'Metadata differs: '+name)

def all_metric_values(labels,scores,indices,threshold):
    values=metric_values(labels,scores,indices,threshold)
    tp,fp,tn,fn=(values[k] for k in ['tp','fp','tn','fn'])
    values.update(precision=tp/(tp+fp) if tp+fp else None,specificity=tn/(tn+fp),
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                  mcc_degenerate=not bool((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return values

def constant_low_collision_allowed(kind,first,second):
    """Only exact spatial constants in the redundant low-frequency subview.

    This admission rule is independent of event labels, scores, model outcomes
    and how many examples it admits. A nonconstant or full-view match fails.
    """
    return (kind=='low_128' and first.shape==(128,128) and second.shape==(128,128)
            and bool(np.all(first==first.flat[0])) and bool(np.all(second==second.flat[0])))

def verify_fixed_dimension(feature,fit_indices,reference_index,state,scores,effective_dimension):
    bg=np.asarray(feature[fit_indices],np.float64);mu=bg.mean(0);x=bg-mu
    beta=.5/(len(bg)-1);ridge=max(.5*np.square(x).sum()/((len(bg)-1)*effective_dimension),1e-12)
    weight=np.asarray(state['weight'],np.float64);target=np.asarray(feature[reference_index],np.float64)-mu
    require(weight.shape==target.shape and np.isfinite(weight).all(),'Invalid weight')
    residual=ridge*weight+beta*(x.T@(x@weight))-target
    relative=float(np.linalg.norm(residual)/max(np.linalg.norm(target),1e-12))
    require(relative<=2e-7,'Fixed-dimension ELDA normal-equation residual exceeds 2e-7')
    close(state['mu'],mu,'ELDA mean',atol=2e-12,rtol=2e-12)
    close(state['beta'],beta,'ELDA beta');close(state['ridge'],ridge,'ELDA ridge',atol=1e-16,rtol=2e-10)
    require(int(state['effective_dimension'])==effective_dimension,'Effective dimension changed')
    close(state['offset'],-mu@weight,'ELDA intercept',atol=2e-7,rtol=2e-8)
    maximum=0.
    for start in range(0,len(feature),512):
        direct=np.asarray(feature[start:start+512],np.float64)@weight+state['offset']
        maximum=max(maximum,float(np.max(np.abs(direct-scores[start:start+512]))))
        close(scores[start:start+512],direct,'All saved scores',atol=2e-7,rtol=2e-8)
    return dict(normal_equation_relative_residual=relative,maximum_score_abs_error=maximum,
                ridge=ridge,beta=beta,effective_dimension=effective_dimension,
                feature_dimensions=feature.shape[1],fit_negatives=len(bg))

def data_audit(protocol):
    rows=read(DATA/'records.json');original=read(V44/'used_records.json');old={r['event_id']:r for r in original}
    plan=read(DATA/'plan.json');bindings(plan['source_hashes'])
    require(plan['code_sha256']==sha256(R/'duration_data.py'),'Acquisition code not frozen')
    require(plan['used_records_sha256']==sha256(V44/'used_records.json'),'Used-source binding')
    reserve=read(V45/'reserve_metadata_audit.json')['records'];reserved_ids={r['event_id'] for r in reserve}
    require(len({r['event_id'] for r in rows})==len(rows),'Duplicate event ID')
    source_indices=[r['source_index'] for r in rows]
    require(source_indices==sorted(set(source_indices)),'Complete-event source order changed')
    inv={r['event_id']:r for r in read(DATA/'metadata_inventory.json')['rows']}
    for i,row in enumerate(rows):
        require(row['index']==i and original[row['source_index']]['event_id']==row['event_id'],'Row alignment')
        prior=old[row['event_id']]
        for key in ['label','split','ifo','event_time','group']:require(row[key]==prior[key],'Changed source role/identity '+key)
        require(row['event_id'] not in reserved_ids,'Reserved event included')
        require(all(row['ifo']!=x['ifo'] or abs(row['event_time']-x['event_time'])>8 for x in reserve),'Reserved temporal neighbor included')
        require(set(row['duration_sources'])==set(DURATIONS),'Missing duration')
        for number,(duration,seconds) in enumerate(zip(DURATIONS,[.5,1.,2.,4.]),1):
            b=row['duration_sources'][duration]
            require(b['url_field']=='url'+str(number) and b['duration_seconds']==seconds,'Wrong duration mapping')
            require(b['url']==inv[row['event_id']]['catalog']['url'+str(number)],'Source URL mapping changed')
            for key in ['event_id','ifo','event_time']:require(b[key]==row[key],'Duration source event mismatch')
            require(sha256(b['path'])==b['sha256'],'Raw source changed')
            if duration=='d1':require(b['sha256']==row['raw_sha256'] and b['reused_original_1s'],'Original 1s replaced')
    acq=read(DATA/'acquisition_audit.json');audit=read(DATA/'audit.json');ib=read(DATA/'inputs_binding.json')
    require(acq['passed'] and not acq['lost_original_fit'] and not acq['lost_references'],'Data eligibility failed')
    require(acq['plan_sha256']==sha256(DATA/'plan.json') and acq['records_sha256']==sha256(DATA/'records.json'),'Acquisition binding')
    expected_audit={'records_sha256':'records.json','inputs_binding_sha256':'inputs_binding.json',
                    'input_pixel_hashes_sha256':'input_pixel_hashes.json','acquisition_audit_sha256':'acquisition_audit.json'}
    for key,path in expected_audit.items():require(audit[key]==sha256(DATA/path),'Data audit binding '+key)
    require(audit['no_reserved_events'] and audit['no_new_real_events'],'Data scope flags')
    review=None
    if audit['cross_role_pixel_collisions']:
        review_path=DATA/'collision_review.json'
        require(protocol['files'].get(str(review_path))==sha256(review_path),'Constant-subview review not frozen')
        review=read(review_path);bindings(review['files'])
        require(review['passed'] and review['eligible_for_evaluation_under_bound_constant_subview_exception'] and review['initial_audit_preserved'] and review['no_data_modified'],'Constant-subview review not admitted')
        require(review['files'].get(str(DATA/'audit.json'))==sha256(DATA/'audit.json'),'Initial false audit not preserved/bound')
        require(audit['passed'] is False,'Original failed gate must remain recorded')
        require(not review['disallowed_cross_role_collisions'],'Review found disallowed collisions')
    else:require(audit['passed'],'No-collision data audit did not pass')
    require(ib['plan_sha256']==sha256(DATA/'plan.json') and ib['records_sha256']==sha256(DATA/'records.json'),'Input plan binding')
    require(ib['code_sha256']==sha256(R/'duration_data.py') and ib['preprocess_sha256']==sha256(P/'hypernet_v41/preprocess.py'),'Input source binding')
    require(ib['original_one_second_bit_exact'] and ib['one_reference_event_four_rendered_windows'] and ib['no_new_events'],'Input scope changed')
    expected={v+'_'+d for d in DURATIONS for v in ['full','low']};require(set(ib['files'])==expected,'Missing input array')
    input_hashes=read(DATA/'input_pixel_hashes.json');require(len(input_hashes)==len(rows),'Input hash row count')
    arrays={k:np.load(DATA/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in expected}
    for key,array in arrays.items():
        require(sha256(DATA/(key+'.npy'))==ib['files'][key],'Input array hash')
        require(array.shape==(len(rows),128,128) and array.dtype==np.float32,'Input shape/dtype')
        require(np.isfinite(array).all() and array.min()>=0 and array.max()<=1,'Input finite/range')
    original_arrays={v:np.load(V44/'features/used'/(v+'128.npy'),mmap_mode='r',allow_pickle=False) for v in ['full','low']}
    seen={};collisions=[];admitted=[]
    for i,(row,record) in enumerate(zip(rows,input_hashes)):
        require(record['event_id']==row['event_id'] and record['index']==i and record['source_index']==row['source_index'],'Input pixel hash row alignment')
        hashes={key:hashlib.sha256(np.ascontiguousarray(a[i]).tobytes()).hexdigest() for key,a in arrays.items()}
        require(hashes==record['hashes'],'Individual input pixel digest')
        combined=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        require(combined==record['combined_sha256'],'Combined input digest')
        for view in ['full','low']:require(np.array_equal(arrays[view+'_d1'][i],original_arrays[view][row['source_index']]),'Original one-second input not bit-exact')
        tokens=[('combined_all_eight_inputs',combined,'all')]
        for d,b in row['duration_sources'].items():
            tokens.extend((k,b[k],d) for k in ['pixel_sha256','crop_pixel_sha256'])
            tokens.extend((view+'_128',hashes[view+'_'+d],d) for view in ['full','low'])
            dual_hash=hashlib.sha256((hashes['full_'+d]+'|'+hashes['low_'+d]).encode()).hexdigest()
            tokens.append(('dual_128',dual_hash,d))
        for kind,digest,duration in tokens:
            token=(kind,digest);current=dict(event_id=row['event_id'],split=row['split'],duration=duration)
            if token in seen:
                first,first_index=seen[token]
                if first['event_id']!=row['event_id'] and first['split']!=row['split']:
                    collision=dict(kind=kind,first=first,second=current);collisions.append(collision)
                    allowed=(kind=='low_128' and constant_low_collision_allowed(kind,arrays['low_'+first['duration']][first_index],arrays['low_'+duration][i]))
                    require(allowed,'Disallowed independent cross-role source/full/nonconstant collision')
                    admitted.append(collision)
            else:seen[token]=(current,i)
    require(collisions==audit['cross_role_pixel_collisions'],'Independent collision list differs from preserved original audit')
    if collisions:
        require(review is not None and len(review['cross_role_low_collisions'])==len(admitted),'Reviewed collision accounting differs')
    require(protocol['row_count']==len(rows) and acq['complete_events']==len(rows),'Population counts')
    require(protocol['split_counts']=={k:sum(r['split']==k for r in rows) for k in sorted({r['split'] for r in rows})},'Role counts')
    ids={r['event_id'] for r in rows};require(set(plan['original_fit_union_event_ids'])<=ids,'Original FIT members lost')
    return rows,dict(events=len(rows),reserved_metadata_rows=len(reserve),native_source_hashes_verified=4*len(rows),
                     input_pixel_hashes_verified=8*len(rows),one_second_bit_exact=True,
                     original_cross_role_collisions=len(collisions),admitted_exact_constant_low_subview_collisions=len(admitted),
                     allowed_constant_low_collisions=len(admitted),unsafe_cross_role_collisions=0,
                     disallowed_cross_role_collisions=0,initial_data_audit_passed=audit['passed'],
                     collision_review_sha256=sha256(DATA/'collision_review.json') if review is not None else None,
                     no_reserved_images_opened=True,data_plan_sha256=sha256(DATA/'plan.json'))

def registry_audit(registry):
    bindings(registry['source_registries']);entries=registry['models']
    require(len(entries)==11 and len({e['id'] for e in entries})==11,'Eleven unique model states required')
    subset=[e for e in entries if e['training'] in TRAININGS]
    require(len(subset)==9 and {(e['seed'],e['training']) for e in subset}=={(s,t) for s in SEEDS for t in TRAININGS},'Missing seed/training factorial cell')
    require({e['id'] for e in entries if e['training'] not in TRAININGS}=={'old_v37','dino_small'},'Historical/DINO control changed')
    sources={}
    for training,path in [('random',V45/'multiseed'),('shapes',V45/'multiseed/diversity'),('spectrogram',R/'spectrogram')]:
        source=read(path/'checkpoints.json');tp=read(path/'protocol.json');sources[training]=source
        require(source['training_complete'] and source['protocol_sha256']==sha256(path/'protocol.json'),'Training registry not complete/bound')
        bindings(tp.get('source_hashes',tp.get('files',{})))
        require(sha256(path/'checks.json')==tp['checks_sha256'] and read(path/'checks.json')['passed'],'Training checks changed/failed')
        for entry in [e for e in subset if e['training']==training]:
            matches=[e for e in source['models'] if e['id']==entry['id']]
            require(len(matches)==1 and all(entry[k]==v for k,v in matches[0].items()),'Copied model registry differs')
            seed=entry['seed'];done=read(path/f'seed_{seed}/completed.json')
            require(done['stage']=='completed' and done['step']==800 and done['protocol_sha256']==source['protocol_sha256'] and done['no_checkpoint_selection'],'Training completion differs')
            if training=='random':
                require(done['weights']['random']==entry['sha256'],'Random completion hash')
                initial=next(x for x in tp['initial_controls'] if x['seed']==seed)
                require(initial['sha256']==entry['sha256']==sha256(initial['path']),'Random is not exact initial checkpoint')
            else:
                require(done['checkpoint_sha256']==entry['sha256'],'Trained completion hash')
                require(done['random_initialization_sha256']==sha256(V45/f'multiseed/seed_{seed}/random.pt'),'Paired initial state differs')
    for entry in entries:
        if 'path' in entry:require(sha256(entry['path'])==entry['sha256'],'Checkpoint bytes differ')
        bindings(entry.get('files',{}))
        require(entry['parameters']==(42480 if entry['kind']=='cnn' else 22056576),'Parameter count metadata differs')
    historical=read(V44/'model_registry.json')
    for model,key in [('old_v37','old'),('dino_small','dino_small')]:
        entry=next(e for e in entries if e['id']==model)
        require(all(entry[k]==v for k,v in historical[key].items()),'Historical control registry changed')
    spec_plan=read(R/'spectrogram/protocol.json')['plan']
    require(spec_plan['borrowed_neural_weights'] is False and spec_plan['real_encoder_training_images']==0,'Own training regime changed')
    require(len({e['sha256'] for e in subset})==9,'Seed/training checkpoint files identical')
    return entries

def verify_table(actual,expected,key_fields,entries=None):
    require(len(actual)==len(expected),'Result table count mismatch: '+str(key_fields));seen=set()
    for row in actual:
        key=tuple(row[k] for k in key_fields);require(key in expected and key not in seen,'Duplicate/unexpected result key');seen.add(key)
        if entries is not None:
            entry=next(e for e in entries if e['id']==row['model'])
            require(row['seed']==entry['seed'] and row['training']==entry['training'],'Model metadata mismatch')
        for k,v in expected[key].items():nested_close(row[k],v,str(key)+'/'+k)

def audit(protocol_sha256):
    threadpool_limits(2);protocol=read(OUT/'protocol.json')
    require(sha256(OUT/'protocol.json')==protocol_sha256,'Wrong frozen protocol');bindings(protocol['files'])
    for path in [Path(__file__),V45/'validation_audit.py',V45/'seed_development_audit.py',V44/'used_records.json',DATA/'plan.json',R/'duration_data.py']:
        require(protocol['files'].get(str(path))==sha256(path),'Audit dependency was not frozen: '+str(path))
    require(protocol['reference_ids']==REFS and protocol['alphas']==ALPHAS and protocol['views']==VIEWS,'Scientific protocol changed')
    require(protocol['duration_seconds']==[.5,1,2,4] and protocol['single_dimension']=={'cnn':4032,'dino':1536},'View/dimension protocol changed')
    require(protocol['one_positive_event_per_target'] and protocol['no_new_real_holdout'],'Scope changed')
    checks=read(R/'matcher_checks.json');require(checks['passed'] and checks['common_sha256']==sha256(R/'common.py'),'Matcher source controls failed')
    result=read(OUT/'results.json');reg=read(OUT/'registry.json')
    require(result['complete'] and result['no_selection'] and result['no_new_real_holdout'],'Results not complete/frozen-development scope')
    require(result['protocol_sha256']==protocol_sha256 and result['registry_sha256']==sha256(OUT/'registry.json'),'Result binding differs');bindings(result['files'])
    rows,data_details=data_audit(protocol);entries=registry_audit(reg)
    require(protocol['models']==[e['id'] for e in entries],'Registry order differs')
    labels=np.array([r['label'] for r in rows]);splits=np.array([r['split'] for r in rows]);dev=np.flatnonzero(splits=='development')
    refs=independent_references(rows,REFS);fits=read(OUT/'fits.json');require(set(fits)==set(protocol['models']),'Fit registry differs')
    expected_rows={};expected_detectors={};expected_macro={};expected_labels={};expected_union={};normal=[];parity=[]
    saved_parity={(r['model'],r['target']):r for r in result['repeated_single_parity']}
    require(len(saved_parity)==len(result['repeated_single_parity'])==44,'Repeated-control completeness')
    for entry in entries:
        model=entry['id'];folder=OUT/'features'/model;b=read(folder/'binding.json');bindings(b['files'])
        require(b['settings']==dict(protocol_sha256=protocol_sha256,registry_sha256=sha256(OUT/'registry.json'),row_sha256=sha256(DATA/'records.json')),'Descriptor source alignment')
        require(b['parameters']==entry['parameters'],'Descriptor parameter metadata')
        dim=protocol['single_dimension'][entry['kind']];arrays={d:np.load(folder/(d+'.npy'),allow_pickle=False) for d in DURATIONS}
        for d,a in arrays.items():
            require(a.shape==(len(rows),dim) and a.dtype==np.float32 and np.isfinite(a).all(),'Invalid descriptor array')
            require(b['files'].get(str(folder/(d+'.npy')))==sha256(folder/(d+'.npy')) and b['shapes'][d]==list(a.shape),'Descriptor file binding')
            close(np.linalg.norm(a,axis=1),1,'Unit descriptor',atol=1e-5,rtol=1e-5)
        with np.load(OUT/(model+'_scores.npz'),allow_pickle=False) as z:scores={k:z[k] for k in z.files}
        with np.load(OUT/(model+'_states.npz'),allow_pickle=False) as z:states={k:z[k] for k in z.files}
        keys={v+'__'+t for v in VIEWS for t in TARGETS};require(set(scores)==set(fits[model])==keys,'Score/fit keys')
        require(set(states)=={k+'__'+s for k in keys for s in ['weight','offset','ridge','beta','mu','effective_dimension']},'State keys')
        for view,feature in [('single',arrays['d1']),('multi',np.concatenate([arrays[d] for d in DURATIONS],1)/2)]:
            for target in TARGETS:
                key=view+'__'+target;fi=independent_fit_indices(rows,target);cal=np.flatnonzero((splits=='calibration')&(labels!=target));ri=refs[target]
                expected_fit=dict(fit_indices=fi.tolist(),calibration_indices=cal.tolist(),reference_index=ri,reference_id=REFS[target],effective_dimension=dim)
                require(fits[model][key]==expected_fit and len(fi)==512,'FIT/CAL/reference membership differs')
                require(not set(fi)&set(dev) and not set(cal)&set(dev) and ri not in set(fi)|set(cal)|set(dev),'Role leakage')
                state={k:states[key+'__'+k] for k in ['weight','offset','ridge','beta','mu','effective_dimension']};s=scores[key]
                require(s.shape==(len(rows),) and np.isfinite(s).all(),'Invalid saved score vector')
                normal.append(dict(model=model,view=view,target=target,**verify_fixed_dimension(feature,fi,ri,state,s,dim)))
                if view=='single':
                    # The lifted solution is exact by isometry: T' T = I. Verify
                    # the same positive-definite equation on the repeated view.
                    rep=np.concatenate([feature]*4,1)/2;lift=dict(state,weight=np.tile(state['weight'],4)/2,mu=np.tile(state['mu'],4)/2)
                    pcheck=verify_fixed_dimension(rep,fi,ri,lift,s,dim);reported=saved_parity[(model,target)]
                    require(reported['all_calibrated_dev_decisions_equal'] and reported['max_absolute_score_error']<2e-6*max(1.,float(np.abs(s).max())),'Evaluator repeated-refit parity failed')
                    close(reported['relative_scale_error'],reported['max_absolute_score_error']/max(1.,float(np.abs(s).max())),'Repeated error accounting')
                    parity.append(dict(model=model,target=target,**pcheck));del rep,lift
                for alpha in ALPHAS:
                    order=math.ceil((len(cal)+1)*(1-alpha));require(1<=order<=len(cal),'Calibration insufficient')
                    cut=float(np.sort(s[cal])[order-1]);y=labels==target;rk=(model,view,target,alpha)
                    expected_rows[rk]=dict(all_metric_values(y,s,dev,cut),threshold=cut,reference_id=REFS[target])
                    for detector in ['H1','L1']:
                        ix=np.array([i for i in dev if rows[i]['ifo']==detector],dtype=int)
                        if len(ix) and 0<y[ix].sum()<len(ix):expected_detectors[rk+(detector,)]=dict(all_metric_values(y,s,ix,cut),threshold=cut,reference_id=REFS[target])
                    for label in sorted(set(labels[dev])-{target}):
                        ix=dev[labels[dev]==label];expected_labels[rk+(label,)]=dict(fp=int((s[ix]>cut).sum()),negative=len(ix),threshold=cut,reference_id=REFS[target])
            for alpha in ALPHAS:
                rr=[expected_rows[(model,view,t,alpha)] for t in TARGETS]
                expected_macro[(model,view,alpha)]={k:float(np.mean([r[k] for r in rr])) for k in FIELDS}|dict(tp=sum(r['tp'] for r in rr),fp=sum(r['fp'] for r in rr))
                alarm=np.logical_or.reduce([scores[view+'__'+t]>expected_rows[(model,view,t,alpha)]['threshold'] for t in TARGETS])
                for label in sorted(set(labels[dev])-set(TARGETS)):
                    ix=dev[labels[dev]==label];expected_union[(model,view,alpha,label)]=dict(fp=int(alarm[ix].sum()),negative=len(ix))
    tables=[('rows',expected_rows,['model','view','target','alpha']),('detector_rows',expected_detectors,['model','view','target','alpha','detector']),('macro',expected_macro,['model','view','alpha']),('false_alarm_by_label',expected_labels,['model','view','target','alpha','negative_label']),('any_target_false_alarm_by_label',expected_union,['model','view','alpha','negative_label'])]
    for field,expected,keys in tables:verify_table(result[field],expected,keys,entries)
    id_for={(e['seed'],e['training']):e['id'] for e in entries};summaries={};duration={};training_pairs={};interaction={}
    for training in TRAININGS:
        for view in VIEWS:
            for alpha in ALPHAS:
                values={k:[expected_macro[(id_for[(seed,training)],view,alpha)][k] for seed in SEEDS] for k in FIELDS}
                summaries[(training,view,alpha)]=dict(metrics={k:dict(mean=float(np.mean(v)),min=min(v),max=max(v)) for k,v in values.items()})
    for e in entries:
        for alpha in ALPHAS:
            a=expected_macro[(e['id'],'multi',alpha)];b=expected_macro[(e['id'],'single',alpha)]
            duration[(e['id'],alpha)]=dict(contrast='multi_minus_single',metrics={k:a[k]-b[k] for k in FIELDS})
    for seed in SEEDS:
        for alpha in ALPHAS:
            for left,right in [('spectrogram','shapes'),('shapes','random'),('spectrogram','random')]:
                for view in VIEWS:
                    a=expected_macro[(id_for[(seed,left)],view,alpha)];b=expected_macro[(id_for[(seed,right)],view,alpha)]
                    training_pairs[(seed,left,right,view,alpha)]=dict(metrics={k:a[k]-b[k] for k in FIELDS})
            for trained in ['shapes','spectrogram']:
                a=duration[(id_for[(seed,trained)],alpha)]['metrics'];b=duration[(id_for[(seed,'random')],alpha)]['metrics']
                interaction[(seed,trained,alpha)]=dict(contrast='trained_duration_gain_minus_random_duration_gain',metrics={k:a[k]-b[k] for k in FIELDS})
    verify_table(result['seed_summary'],summaries,['training','view','alpha'])
    verify_table(result['duration_pairs'],duration,['model','alpha'],entries)
    verify_table(result['training_pairs'],training_pairs,['seed','left','right','view','alpha'])
    verify_table(result['interactions'],interaction,['seed','trained','alpha'])
    return dict(passed=True,protocol_sha256=protocol_sha256,results_sha256=sha256(OUT/'results.json'),
                registry_sha256=sha256(OUT/'registry.json'),audit_code_sha256=sha256(__file__),data=data_details,
                model_count=len(entries),normal_equation_checks=normal,repeated_isometry_checks=parity,
                table_counts={field:len(expected) for field,expected,_ in tables},seed_summaries_verified=len(summaries),
                duration_contrasts_verified=len(duration),training_contrasts_verified=len(training_pairs),interactions_verified=len(interaction),
                scope='Previously consumed development events. One positive reference event, one or four authentic time windows, labeled negative FIT/CAL; no new confirmatory images.',
                source='Independent positive-definite normal equations, source/input hash audit, frozen-CAL order statistics, tied-ROC/pAUC, paired contrasts and detector/class false-alarm accounting. No inference or threshold refit.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--output',default=str(OUT/'audit.json'));args=parser.parse_args()
    destination=Path(args.output).resolve();require(R in destination.parents,'Audit output outside v46');require(not destination.exists(),'Do not overwrite a completed audit')
    result=audit(args.protocol_sha256);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['normal_equation_checks','repeated_isometry_checks']},indent=2))
