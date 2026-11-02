"""Independent metric/split audit and full read-only surface-encoder replay."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def state_sha(state):
    h=hashlib.sha256()
    for name in sorted(state):h.update(name.encode());h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def compare(actual,expected,context,tolerance=1e-12):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),context+': metric keys differ')
        for key in expected:compare(actual[key],expected[key],context+'.'+key,tolerance)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),context+': list length differs')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,context+f'[{i}]',tolerance)
    else:require(math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),abs_tol=tolerance,rel_tol=tolerance),
                 f'{context}: {actual} differs from {expected}')


def fractional_accuracy(scores,labels):
    scores=np.asarray(scores);labels=np.asarray(labels)
    require(np.isfinite(scores).all() and len(scores)==len(labels),'Invalid ranking scores')
    ties=np.abs(scores-scores.max(axis=1)[:,None])<=1e-7
    credit=ties[np.arange(len(labels)),labels]/ties.sum(axis=1)
    return float(sum(credit)/len(credit))


def representation_metrics(features,records):
    objects={}
    for index,row in enumerate(records):
        key=(row['family_index'],row['object_id'])
        slot=(row['view_index'],float(row['surface_strength']))
        require(slot not in objects.setdefault(key,{}),'Duplicate object/view/strength')
        objects[key][slot]=index
    keys=sorted(objects);require(len(keys)==64,'Expected 64 held-out objects')
    for rows in objects.values():require(set(rows)=={(v,s) for v in (0,1) for s in (0.,.03,.07)},'Incomplete held-out object')
    supports=[objects[key][(0,0.)] for key in keys]
    instance_queries=[objects[key][(1,s)] for key in keys for s in (0.,.03,.07)]
    instance_labels=np.repeat(np.arange(64),3)
    instance=fractional_accuracy(features[instance_queries]@features[supports].T,instance_labels)
    family_keys={f:[key for key in keys if key[0]==f] for f in range(8)}
    require(all(len(v)==8 for v in family_keys.values()),'Expected eight held-out objects per family')
    episodes=[];family_decisions=0
    for slot in range(8):
        enrolled={f:family_keys[f][slot] for f in range(8)}
        references=[objects[enrolled[f]][(0,0.)] for f in range(8)]
        query_indices=[];labels=[]
        for key in keys:
            if key==enrolled[key[0]]:continue
            require(key[1] not in {e[1] for e in enrolled.values()},'Exact object leaked into family query')
            for strength in (0.,.03,.07):query_indices.append(objects[key][(1,strength)]);labels.append(key[0])
        family_decisions+=len(labels)
        episodes.append(fractional_accuracy(features[query_indices]@features[references].T,labels))
    detail=[]
    for key in keys:
        for strength,opposite in ((0.,.07),(.07,0.)):
            ref=features[objects[key][(0,strength)]]
            positive=features[objects[key][(1,strength)]];negative=features[objects[key][(1,opposite)]]
            margin=float(ref@positive-ref@negative)
            detail.append(.5 if -1e-7<=margin<=1e-7 else float(margin>0))
    return {'one_support_per_family_accuracy':sum(episodes)/8,'family_episode_accuracies':episodes,
            'family_decisions':family_decisions,'cross_view_instance_retrieval':instance,
            'instance_decisions':len(instance_queries),'balanced_cross_view_surface_match':sum(detail)/len(detail),
            'surface_triplets':len(detail)}


def spectrum_targets(rgb,mask):
    """NumPy FFT powers, retaining the frozen float32 angular-bin boundaries."""
    import torch
    gray=(rgb.astype(np.float64)/255*np.array([.299,.587,.114])).sum(axis=-1)
    mean=(gray*mask).sum((1,2),keepdims=True)/mask.sum((1,2),keepdims=True)
    window=np.hanning(128);weighted=(gray-mean)*mask*window[None,:,None]*window[None,None,:]
    power=np.abs(np.fft.fft2(weighted,axes=(-2,-1),norm='ortho'))**2
    # Coordinate binning is discontinuous at exact diagonals. Preserve the
    # experiment's float32 boundaries rather than silently changing targets by
    # switching atan2/remainder to NumPy float64. The FFT/power aggregation and
    # cosine calculation remain independently implemented in NumPy.
    frequency=torch.from_numpy(np.fft.fftfreq(128).astype(np.float32))
    yy,xx=torch.meshgrid(frequency,frequency,indexing='ij')
    radial=(torch.sqrt(xx*xx+yy*yy)/.707107*8).long().clamp(0,7)
    direction=(torch.remainder(torch.atan2(yy,xx),math.pi)/math.pi*8).long().clamp(0,7)
    ids=(radial*8+direction).numpy().ravel();counts=np.bincount(ids,minlength=64).clip(1)
    values=np.stack([np.bincount(ids,weights=row.ravel(),minlength=64) for row in power])
    values=np.log1p(100*values/counts);return values/np.linalg.norm(values,axis=1,keepdims=True).clip(1e-6)


def evaluate_arrays(arrays,records,spectrum):
    import torch
    result={}
    for name in ('embedding','shape','detail'):
        values=torch.from_numpy(arrays[name].copy())
        normalized=(values/torch.linalg.vector_norm(values,dim=1,keepdim=True).clamp_min(1e-12)).numpy()
        result[name]=representation_metrics(normalized,records)
    result['trained_family_classifier_accuracy']=fractional_accuracy(arrays['family_logits'],[r['family_index'] for r in records])
    prediction=arrays['spectrum'].astype(np.float64)
    prediction=prediction/np.linalg.norm(prediction,axis=1,keepdims=True).clip(1e-8)
    result['spectrum_cosine']=float((prediction*spectrum).sum(1).mean())
    return result


def main():
    import torch
    from PIL import Image
    from hypernet.surface_encoder import SurfaceEncoder,input_for_arm
    output=ROOT/'experiments/surface_encoder_audit.json';require(not output.exists(),'Preserve previous audit')
    freeze_path=ROOT/'experiments/surface_training_freeze.json';freeze=read(freeze_path)
    result_path=ROOT/'results/surface_results.json';result=read(result_path)
    require(result['completed'] and result['sources_unchanged'],'Training did not complete')
    require(sha(freeze_path)==result['freeze_sha256'],'Result freeze binding changed')
    for path,digest in freeze['sources'].items():require(sha(ROOT/path)==digest,'Frozen source changed')
    data=ROOT/'data/surfaces_v10';manifest_path=data/'manifest.json';manifest=read(manifest_path)
    require(sha(manifest_path)==freeze['manifest_sha256'],'Training manifest changed')
    data_audit=read(ROOT/'experiments/surface_data_audit.json')
    require(data_audit['valid'] and data_audit['manifest_sha256']==sha(manifest_path),'Dataset audit binding failed')
    plan_path=ROOT/'experiments/surface_batch_plan.json';plan=read(plan_path)
    require(sha(plan_path)==freeze['batch_plan_sha256'] and len(plan)==400,'Training batch plan changed')
    records=manifest['records'];train={r['object_id'] for r in records if r['split']=='train'}
    held={r['object_id'] for r in records if r['split']=='heldout'}
    require(len(train)==256 and len(held)==64 and not train&held,'Object split leakage')
    for ids in plan:
        require(len(ids)==16 and len(set(ids))==16,'Expected 16 distinct paired training frames')
        require(all(type(i) is int and records[i]['split']=='train' for i in ids),'Held-out gradient-training index')
        pairs=[(records[a],records[b]) for a,b in zip(ids[::2],ids[1::2])]
        require(all(a['object_id']==b['object_id'] and (a['view_index'],b['view_index'])==(0,1) for a,b in pairs),'Broken cross-view positive pair')
        counts=Counter(a['family_index'] for a,b in pairs)
        require(len(counts)==4 and set(counts.values())=={2} and len({a['object_id'] for a,b in pairs})==8,'Unbalanced family/object batch')
    for key,value in {'steps_per_arm':400,'batch_images':16,'image_exposures_per_arm':6400,
            'training_object_programs':256,'heldout_object_programs':64,'natural_training_images':0,
            'external_pretrained_weights':False,'parameters':181525,'seed':731}.items():require(freeze[key]==value,'Unexpected budget '+key)
    torch.set_num_threads(freeze['runtime']['threads']);torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(731);initial=SurfaceEncoder()
    require(state_sha(initial.state_dict())==freeze['initial_state_sha256'],'Initial seed/state mismatch')
    initial_state={k:v.clone() for k,v in initial.state_dict().items()}
    rows=[r for r in records if r['split']=='heldout'];rgb=[];masks=[]
    for row in records:
        for key,digest in [('rgb_path','rgb_sha256'),('target_path','target_sha256'),('metadata_path','metadata_sha256')]:
            require(sha(data/row[key])==row[digest],'Dataset file changed')
        if row['split']=='heldout':
            with Image.open(data/row['rgb_path']) as image:rgb.append(np.array(image.convert('RGB')))
            with np.load(data/row['target_path'],allow_pickle=False) as target:masks.append(target['mask'])
    rgb=np.stack(rgb);masks=np.stack(masks);spectrum=spectrum_targets(rgb,masks)
    images=torch.from_numpy(rgb).permute(0,3,1,2).float()/255
    def replay(state,input_arm):
        model=SurfaceEncoder();model.load_state_dict(state);model.eval().requires_grad_(False)
        before=state_sha(model.state_dict());collected={k:[] for k in ('shape','detail','embedding','family_logits','spectrum')}
        with torch.inference_mode():
            for start in range(0,384,16):
                values=model(input_for_arm(images[start:start+16],input_arm))
                for key in collected:collected[key].append(values[key].numpy())
        require(state_sha(model.state_dict())==before and all(p.grad is None and not p.requires_grad for p in model.parameters()),'Replay changed weights/gradients')
        return {k:np.concatenate(v) for k,v in collected.items()}
    audited={};bound={};replay_max=0.;spectrum_max=0.
    for arm in ('native128','lowpass32'):
        path=ROOT/'checkpoints'/f'surface_{arm}.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        require(checkpoint['format']==freeze['format'] and checkpoint['arm']==arm and checkpoint['freeze_sha256']==sha(freeze_path),'Checkpoint binding mismatch')
        require(checkpoint['natural_training_images']==0 and checkpoint['external_pretrained_weights'] is False,'Checkpoint training budget mismatch')
        require(sha(path)==result['arms'][arm]['checkpoint_sha256'],'Checkpoint file changed')
        state=checkpoint['state_dict'];require(all(torch.isfinite(v).all() for v in state.values()),'Nonfinite learned state')
        require(sum(v.numel() for v in state.values())==181525 and state_sha(state)==result['arms'][arm]['final_state_sha256'],'Checkpoint tensors disagree')
        arm_result=result['arms'][arm];require(arm_result['evaluation_read_only'] is True,'Evaluation read-only claim missing')
        require([h['step'] for h in arm_result['history']]==[1,*range(25,401,25)],'Wrong training history steps')
        require(all(math.isfinite(v) for h in arm_result['history'] for v in h.values()),'Nonfinite logged training losses')
        require(arm_result['actual_distinct_frames_seen']==len({i for batch in plan for i in batch}),'Training exposure count mismatch')
        outputs={}
        for label,state_to_replay,file_prefix,reported_key in [('random',initial_state,'heldout_random_','random_initialization'),
                                                             ('trained',state,'heldout_','trained')]:
            array_path=ROOT/'results'/f'{file_prefix}{arm}.npz'
            with np.load(array_path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
            require(set(arrays)=={'shape','detail','embedding','family_logits','spectrum'},'Saved feature keys differ')
            actual=replay(state_to_replay,arm)
            for key,value in arrays.items():
                require(value.shape==actual[key].shape and np.isfinite(value).all(),'Invalid saved features')
                error=float(np.abs(value-actual[key]).max());replay_max=max(replay_max,error)
                require(np.array_equal(value,actual[key]),f'Saved feature replay differs: {arm}/{label}/{key}, max {error}')
            metrics=evaluate_arrays(arrays,rows,spectrum);recorded=arm_result[reported_key]
            for key in metrics:
                if key=='spectrum_cosine':
                    spectrum_max=max(spectrum_max,abs(metrics[key]-recorded[key]));compare(metrics[key],recorded[key],arm+'/'+label+'/'+key,1e-5)
                else:compare(metrics[key],recorded[key],arm+'/'+label+'/'+key)
            outputs[label]=metrics;bound[str(array_path)]=sha(array_path)
        alternate='lowpass32' if arm=='native128' else 'native128';require(arm_result['alternate_input_arm']==alternate,'Wrong cross-resolution input')
        cross=evaluate_arrays(replay(state,alternate),rows,spectrum)
        for key in cross:
            tolerance=1e-5 if key=='spectrum_cosine' else 1e-12
            compare(cross[key],arm_result['cross_resolution_evaluation'][key],arm+'/cross-resolution/'+key,tolerance)
        outputs['cross_resolution']=cross;audited[arm]=outputs;bound[str(path)]=sha(path)
    for path,digest in freeze['sources'].items():require(sha(ROOT/path)==digest,'Source changed during audit')
    require(sha(manifest_path)==freeze['manifest_sha256'] and sha(plan_path)==freeze['batch_plan_sha256'],'Frozen data/plan changed')
    report={'valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),'auditor_sha256':sha(__file__),
        'freeze_sha256':sha(freeze_path),'results_sha256':sha(result_path),'manifest_sha256':sha(manifest_path),
        'dataset_audit_sha256':sha(ROOT/'experiments/surface_data_audit.json'),'bound_features_and_checkpoints':bound,
        'all_6400_gradient_exposures_per_arm_are_train_only':True,'heldout_object_programs':64,
        'same_initial_state_and_batch_plan_both_arms':True,'balanced_same_object_cross_view_batches':True,
        'family_queries_exclude_all_enrolled_object_ids':True,'clean_and_rough_detail_reference_roles_balanced':True,
        'independent_metric_recomputation':True,'full_random_and_trained_saved_feature_replay_exact':True,
        'replay_forward_frames':2304,'max_saved_feature_absolute_error':replay_max,
        'max_independent_numpy_spectrum_cosine_error':spectrum_max,'new_training_during_audit':False,
        'all_replay_weights_and_gradients_unchanged':True,'arms':audited,
        'limits':['Known supervised mathematical families and unseen procedural objects, not unseen semantic categories.',
                  'No natural-image accuracy claim; renders and targets are synthetic.',
                  'Tensor/history checks combine frozen code and records; they cannot prove arbitrary hidden training history.',
                  'Minute-detail preservation is not guaranteed by global spectrum and 64px geometry auxiliary targets.']}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'valid':True,'audit':str(output),'max_feature_error':replay_max,
                      'max_spectrum_error':spectrum_max},indent=2))


if __name__=='__main__':main()
