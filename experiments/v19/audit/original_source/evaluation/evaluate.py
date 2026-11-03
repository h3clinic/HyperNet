"""Small v19 adapter harness, reusing the unchanged v17 metrics and baselines."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib,json,sys,time
import numpy as np
from data_protocol import(EVAL,ROOT,WORKSPACE,OLD,DATA,PLAN,METHODS,OPERATING_POINTS,read,sha,write_new,verify,load_plan,load_old,load_reference,load_test,verify_protocol)
from vendor_v17.detection_metrics import(global_cutoff,iou,summarize,truth_arrays,clip_box,COMMON_BOX_RULE)
sys.path.insert(0,str(ROOT))
RESULTS=EVAL/'results';DEFAULT_CONFIG=EVAL/'model_config.json'

def canonical(value):return json.loads(json.dumps(value))
def resolve_path(path):
    value=Path(path);return value.resolve()if value.is_absolute()else(WORKSPACE/value).resolve()
def config_read(path):
    config=read(path)
    if sorted(config['methods'])!=sorted(METHODS):raise ValueError('Expected exactly the four planned comparators')
    return config
def model_for(method,entry):
    factory=entry.get('factory')
    if not isinstance(factory,str)or ':'not in factory:raise ValueError('V19 inference adapter factory must be specified after architecture approval')
    module,name=factory.split(':',1);constructor=getattr(importlib.import_module(module),name)
    kwargs=dict(entry.get('kwargs',{}))
    if method=='raw_template':return constructor(**kwargs)
    return constructor(resolve_path(entry['checkpoint']),device=entry['device'],**kwargs)
def calibration(scores,overlaps,rows,targets):
    present,valid=truth_arrays(rows,targets);negative=scores[valid&~present];result={}
    for label,budget in OPERATING_POINTS.items():
        cutoff=global_cutoff(negative,budget)
        result[label]={'cutoff':cutoff,'nominal_dev_negative_fpr':budget,'calibration_negative_pairs':int(negative.size),
          'threshold_source':'All valid DEV absent target/photo pairs, one global cutoff',
          'summary':summarize(scores,overlaps,present,valid,cutoff,targets)}
    return result

def predict_references(model,banks,encoded):
    """Preserve target order while using independent reference-bank batching."""
    batched=getattr(model,'predict_many',None)
    proposals=batched(banks,encoded)if callable(batched)else[model.predict(bank,encoded)for bank in banks]
    if len(proposals)!=len(banks):raise ValueError('Reference batch returned a different target count')
    return proposals

def reuse_baseline_dev(config_path):
    plan=load_plan();old_plan=read(OLD/'data/coco_v17/plan.json');config=config_read(config_path)
    for key in('references','dev','targets'):
        if plan[key]!=old_plan[key]:raise ValueError('Old baseline DEV no longer matches the exact support/queries/GT')
    for method in('v17','raw_template','owlvit_base_patch32'):
        source=OLD/'experiments'/f'{method}_dev_v2';result=read(source/'result.json');path=source/'arrays.npz';arrays=dict(np.load(path,allow_pickle=False))
        if sha(path)!=result['score_file_sha256']:raise ValueError('Original baseline scores changed')
        if arrays['image_ids'].tolist()!=[r['image_id']for r in plan['dev']]or arrays['category_ids'].tolist()!=[t['category_id']for t in plan['targets']]:raise ValueError('Baseline pair ordering differs')
        if method=='v17'and sha(resolve_path(config['methods'][method]['checkpoint']))!=result['provenance']['checkpoint_sha256']:raise ValueError('Not the measured v17 checkpoint')
        entry=config['methods'][method];provenance=result['provenance']
        if entry['device']!=provenance['device']:raise ValueError('Reused baseline device differs from its measured run')
        if method=='v17':
            expected={'max_side':provenance['max_query_side'],'proposal_limit':provenance['proposal_limit'],'nms_iou':provenance['nms_iou'],'pre_nms':provenance['pre_nms_candidates']}
            if entry.get('kwargs',{})!=expected:raise ValueError('Reused v17 preprocessing/proposals differ')
        elif entry.get('kwargs',{}):raise ValueError('Reused OWL/raw baselines used default adapter kwargs')
        if method=='owlvit_base_patch32'and resolve_path(entry['checkpoint'])!=Path(next(iter(provenance['checkpoint_files']))).parent:
            raise ValueError('Reused OWL checkpoint location differs from measured provenance')
        if method=='owlvit_base_patch32':verify(result['provenance']['checkpoint_files'])
        destination=RESULTS/f'{method}_dev'
        if destination.exists():raise FileExistsError('Preserve existing baseline reuse: '+str(destination))
        destination.mkdir(parents=True)
        with(destination/'arrays.npz').open('xb')as f:f.write(path.read_bytes())
        calibrated=calibration(arrays['scores'],arrays['overlaps'],plan['dev'],plan['targets']);write_new(destination/'calibration.json',calibrated)
        reused={**result,'data_plan_sha256':sha(PLAN),'configuration':config['methods'][method],
          'summary':{k:v['summary']for k,v in calibrated.items()},'common_box_rule':COMMON_BOX_RULE,
          'reused_dev':{'inference_rerun':False,'same_supports_queries_truth':True,'original_result_sha256':sha(source/'result.json'),
            'original_arrays_sha256':sha(path),'original_data_plan_sha256':sha(OLD/'data/coco_v17/plan.json'),
            'note':'Original measured scores/boxes reused exactly. Primary5% cutoff unchanged; new optional1% cutoff fits the same DEV negatives.'}}
        if calibrated['primary_5pct']['cutoff']!=read(source/'calibration.json')['primary_5pct']['cutoff']:raise ValueError('Primary baseline calibration changed')
        write_new(destination/'result.json',reused)
        print(json.dumps({'reused':method,'dev_photos':len(plan['dev']),'primary_macro_localized_recall':reused['summary']['primary_5pct']['macro_localized_recall']}),flush=True)

def run(config_path,method,split):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=load_plan();config=config_read(config_path);entry=config['methods'][method];rows=plan[split]
    if split=='test':
        acquisition=read(DATA/'acquisition.json')
        if acquisition.get('completed')is not True or acquisition['data_plan_sha256']!=sha(PLAN):raise ValueError('Completed post-freeze acquisition required')
        protocol=verify_protocol(acquisition['protocol_path'],acquisition['protocol_sha256'])
        if protocol['configuration_sha256']!=sha(config_path):raise ValueError('Frozen model settings changed')
        keep=set(acquisition['eligible_image_ids']);rows=[r for r in rows if r['image_id']in keep]
    directory=RESULTS/f'{method}_{split}'
    if directory.exists():raise FileExistsError('Preserve immutable completed/partial inference output: '+str(directory))
    model=model_for(method,entry)
    if split=='test':
        previous=read(RESULTS/f'{method}_dev/result.json')
        if previous['configuration']!=entry or previous['provenance']!=canonical(model.provenance()):raise ValueError('Model/checkpoint/preprocessing differs from calibrated DEV')
    refs={r['category_id']:load_reference(r)for r in plan['references']};banks=[model.enroll(refs[t['category_id']])for t in plan['targets']]
    directory.mkdir(parents=True);scores=np.zeros((len(rows),len(banks)));boxes=np.zeros((*scores.shape,4));overlaps=np.zeros_like(scores);start=time.monotonic()
    with(directory/'predictions.jsonl').open('x',encoding='utf-8')as f:
        for i,row in enumerate(rows):
            image=load_old(row)if split=='dev'else load_test(row,acquisition);encoded=model.encode_query(image)if hasattr(model,'encode_query')else image
            predictions=predict_references(model,banks,encoded)
            for j,(target,proposals)in enumerate(zip(plan['targets'],predictions)):
                if not proposals:raise ValueError('Every pair requires a finite top1 proposal; no confidence prefilter')
                top=dict(max(proposals,key=lambda r:r['score']));top['raw_box']=list(top['box']);top['box']=clip_box(top['raw_box'],image.shape[1],image.shape[0])
                scores[i,j]=float(top['score']);boxes[i,j]=top['box']
                if not np.isfinite([scores[i,j],*boxes[i,j]]).all():raise ValueError('Nonfinite model output')
                overlaps[i,j]=iou(boxes[i,j],row['boxes_by_target'][str(target['category_id'])])
                f.write(json.dumps({'image_id':row['image_id'],'category_id':target['category_id'],**top,'iou':overlaps[i,j]},allow_nan=False)+'\n')
            f.flush()
            if(i+1)%24==0 or i+1==len(rows):print(json.dumps({'method':method,'split':split,'photos':i+1,'total':len(rows),'seconds':time.monotonic()-start}),flush=True)
    model.assert_unchanged()
    np.savez_compressed(directory/'arrays.npz',scores=scores,boxes=boxes,overlaps=overlaps,image_ids=[r['image_id']for r in rows],category_ids=[t['category_id']for t in plan['targets']])
    if split=='dev':
        calibrated=calibration(scores,overlaps,rows,plan['targets']);write_new(directory/'calibration.json',calibrated);summary={k:v['summary']for k,v in calibrated.items()}
    else:
        calibrated=read(RESULTS/f'{method}_dev/calibration.json');present,valid=truth_arrays(rows,plan['targets'])
        summary={k:summarize(scores,overlaps,present,valid,v['cutoff'],plan['targets'])for k,v in calibrated.items()}
    result={'method':method,'split':split,'data_plan_sha256':sha(PLAN),'configuration':entry,'provenance':canonical(model.provenance()),'summary':summary,
      'score_file_sha256':sha(directory/'arrays.npz'),'predictions_sha256':sha(directory/'predictions.jsonl'),'seconds':time.monotonic()-start,'common_box_rule':COMMON_BOX_RULE}
    if split=='test':result['protocol_sha256']=acquisition['protocol_sha256'];verify_protocol(acquisition['protocol_path'],acquisition['protocol_sha256'])
    write_new(directory/'result.json',result);print(json.dumps({k:{key:val for key,val in value.items()if key!='per_target'}for k,value in summary.items()}),flush=True)

def freeze(config_path,output):
    plan=load_plan();config=config_read(config_path)
    if(DATA/'acquisition_admission.json').exists()or(DATA/'images').exists():raise ValueError('Fresh acquisition already started')
    if not config['methods']['v19'].get('factory')or not config['methods']['v19'].get('checkpoint'):raise ValueError('V19 architecture/checkpoint not specified')
    behavior={str(p.resolve()):sha(p)for p in ROOT.rglob('*.py')}
    behavior[str(Path(config_path).resolve())]=sha(config_path);checkpoints={};methods={}
    for method in METHODS:
        directory=RESULTS/f'{method}_dev';result=read(directory/'result.json');calibrated=read(directory/'calibration.json')
        if result['data_plan_sha256']!=sha(PLAN)or result['configuration']!=config['methods'][method]:raise ValueError('DEV configuration/roster differs')
        if result['score_file_sha256']!=sha(directory/'arrays.npz'):raise ValueError('DEV arrays changed')
        for name in('result.json','calibration.json','arrays.npz'):
            path=directory/name;behavior[str(path.resolve())]=sha(path)
        if method in('v19','v17'):
            path=resolve_path(config['methods'][method]['checkpoint']);checkpoints[str(path)]=sha(path)
            if result['provenance'].get('checkpoint_sha256')!=sha(path):raise ValueError('DEV model must report its exact checkpoint_sha256')
            if result['provenance'].get('natural_training_images')!=0 or result['provenance'].get('borrowed_pretrained_weights')is not False:
                raise ValueError('Own models must report zero natural weight-training images and no borrowed pretrained weights')
        if method=='owlvit_base_patch32':checkpoints.update(result['provenance']['checkpoint_files'])
        for key,budget in OPERATING_POINTS.items():
            if calibrated[key]['nominal_dev_negative_fpr']!=budget:raise ValueError('Calibration budget differs')
        methods[method]={'configuration':config['methods'][method],'cutoffs':{k:v['cutoff']for k,v in calibrated.items()},'dev_result_sha256':sha(directory/'result.json')}
    verify(checkpoints)
    value={'format':'hypernet.v19.fresh_test_admission.v1','created_utc':datetime.now(timezone.utc).isoformat(),'model_selection_complete':True,
      'test_scores_computed_before_freeze':False,'data_plan_sha256':sha(PLAN),'configuration_sha256':sha(config_path),'behavior_files':behavior,'checkpoint_files':checkpoints,
      'methods':methods,'decision_rule':'Higher macro localized recall without higher observed macro FPR; report paired photo-cluster intervals, not a universal superiority claim.',
      'limitations':plan['limitations']}
    write_new(output,value);print(json.dumps({'protocol':str(Path(output).resolve()),'sha256':sha(output),'test_pixels_fetched':False}))

def compare(split):
    plan=load_plan();rows=plan[split]
    if split=='test':
        acquisition=read(DATA/'acquisition.json');keep=set(acquisition['eligible_image_ids']);rows=[r for r in rows if r['image_id']in keep]
    present,valid=truth_arrays(rows,plan['targets']);n=len(rows);rng=np.random.default_rng(190031)
    weights=rng.multinomial(n,np.full(n,1/n),size=5000).astype(float);pos=weights@(present&valid);neg=weights@(~present&valid)
    if np.any(pos==0)or np.any(neg==0):raise ValueError('Bootstrap category denominator zero')
    arrays={};calibrated={};bindings={}
    for method in METHODS:
        directory=RESULTS/f'{method}_{split}';result=read(directory/'result.json');path=directory/'arrays.npz';bound_cal=RESULTS/f'{method}_dev/calibration.json'
        if result['score_file_sha256']!=sha(path)or result['data_plan_sha256']!=sha(PLAN):raise ValueError('Comparison input provenance differs')
        a=dict(np.load(path,allow_pickle=False))
        if a['image_ids'].tolist()!=[r['image_id']for r in rows]:raise ValueError('Paired photo ordering differs')
        arrays[method]=a;calibrated[method]=read(bound_cal);bindings[str(path.resolve())]=sha(path);bindings[str(bound_cal.resolve())]=sha(bound_cal)
    points={}
    for key in OPERATING_POINTS:
        samples={};observed={}
        for method,a in arrays.items():
            accepted=a['scores']>=calibrated[method][key]['cutoff'];hits=accepted&(a['overlaps']>=.5)&present&valid;fp=accepted&~present&valid
            samples[method]=np.stack([((weights@hits)/pos).mean(1),((weights@fp)/neg).mean(1)],1)
            observed[method]=np.array([(hits.sum(0)/(present&valid).sum(0)).mean(),(fp.sum(0)/(~present&valid).sum(0)).mean()])
        contrasts={}
        for other in('v17','raw_template','owlvit_base_patch32'):
            delta=samples['v19']-samples[other];point=observed['v19']-observed[other]
            contrasts['v19_minus_'+other]={label:{'estimate':float(point[j]),'ci95':np.quantile(delta[:,j],[.025,.975]).tolist()}for j,label in enumerate(('macro_localized_recall','macro_false_positive_rate'))}
            contrasts['v19_minus_'+other]['observed_win_gate']=bool(point[0]>0 and point[1]<=0)
        points[key]=contrasts
    result={'data_plan_sha256':sha(PLAN),'split':split,'photos':n,'nominal_pairs':int(present.size),'input_bindings':bindings,'operating_points':points,
      'bootstrap':{'replicates':5000,'seed':190031,'unit':'photo, all12 targets/methods carried together','fixed':'Supports and DEV cutoffs','interval':'95% percentile'},
      'zero_hit_caution':'Percentile bootstrap is degenerate when no observed hits occur; [0,0] does not prove exact population equivalence.'}
    write_new(RESULTS/f'paired_{split}.json',result);print(json.dumps(points),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['reuse-dev','run','freeze','compare']);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--method',choices=METHODS);p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='reuse-dev':reuse_baseline_dev(a.config)
    elif a.action=='freeze':
        if not a.output:p.error('--output is required for immutable protocol')
        freeze(a.config,a.output)
    elif a.action=='compare':compare(a.split)
    else:
        if not a.method:p.error('--method required')
        run(a.config,a.method,a.split)
