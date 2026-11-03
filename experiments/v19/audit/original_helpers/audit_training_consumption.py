"""Independent CPU audit of actual v19 consumption against complete renderer preflight.

Imports no project helpers, performs no rendering/inference/training, and writes
only the explicitly requested audit output. Prefix mode checks immutable commits.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'outputs/hypernet_v19'
PREFLIGHT=ROOT/'experiments/full_preflight_v2'
COMMON=('raw_support_sha256','raw_query_sha256','program_id','near_program_id','replacement_program_id','instance_program_ids','metadata_sha256','family','positive')
GENERATED=('boxes','target_box','support_points','query_points','support_point_valid','query_point_valid','point_valid')
COMPACT=GENERATED+('positive','rois','quality_targets')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(value):
    value=np.ascontiguousarray(value)
    return np.frombuffer(hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).digest(),np.uint8)
def state_sha(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        assert torch.isfinite(tensor).all(),name
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
def box_iou(boxes,target):
    boxes=np.asarray(boxes,np.float64);target=np.asarray(target,np.float64)
    intersection=np.maximum(0,np.minimum(boxes[:,2:],target[2:])-np.maximum(boxes[:,:2],target[:2])).prod(-1)
    areas=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(-1);target_area=np.maximum(0,target[2:]-target[:2]).prod()
    return intersection/np.maximum(areas+target_area-intersection,1e-8)

def audit(run,prefix_steps=None):
    run=(ROOT/run).resolve();assert run.is_relative_to(ROOT)
    admission=read(run/'admission.json');config=admission['config'];batch_size=config['batch_size']
    assert config['natural_training_images']==0 and config['borrowed_pretrained_weights']is False
    assert config['deterministic_algorithms']is True and config['smoke']is False
    assert sha(run/'approval_plan.json')==admission['approval_plan_sha256']
    approval=read(run/'approval_plan.json');assert approval['approved_for_training']is True and approval['config']==config
    for name,h in config['source_sha256'].items():
        assert sha(ROOT/name)==h and sha(run/'source_snapshot'/name)==h,('source',name)
    preflight=read(PREFLIGHT/'result.json');assert preflight['status']=='passed' and preflight['errors']==[]
    assert preflight['source_sha256']['hypernet_v19/curriculum.py']==config['source_sha256']['hypernet_v19/curriculum.py']
    assert preflight['seed']==config['stream_seed'] and preflight['difficulty']==config['difficulty']
    assert sha(PREFLIGHT/'episode_hashes.npz')==preflight['episode_hashes_sha256']
    full=dict(np.load(PREFLIGHT/'episode_hashes.npz',allow_pickle=False));mask=full['split_code']==0
    planned={k:v[mask]for k,v in full.items()};assert len(planned['index'])==config['episodes']
    assert np.array_equal(planned['index'],np.arange(config['episodes']))
    complete=prefix_steps is None
    completion=read(run/'completion.json')if complete else None
    if complete:
        assert completion['status']=='complete' and completion['config']==config
        assert completion['source_sha256']==config['source_sha256']
        assert completion['approval_plan_sha256']==admission['approval_plan_sha256']
        assert completion['natural_training_images']==0 and completion['borrowed_pretrained_weights']is False and completion['deterministic_algorithms']is True
        steps=config['steps'];ledger_paths=[ROOT/s['path']for s in completion['ledger_shards']]
        for p,s in zip(ledger_paths,completion['ledger_shards']):assert sha(p)==s['sha256']
        assert not(run/'failure.json').exists()
    else:
        steps=int(prefix_steps);assert steps>0 and steps%64==0
        ledger_paths=[p for p in sorted((run/'ledger').glob('*.npz'))if int(p.stem.split('_')[1])<=steps]
        assert ledger_paths and int(ledger_paths[-1].stem.split('_')[1])==steps
    offset=0;actual_steps=0;stream=hashlib.sha256();history=[];history_bytes=[];ledger_details=[]
    supports=set();queries=set();programs=set();positive_count=0;point_count=0;families=Counter();tensor_checks=0;generated_checks=0;roi_max_error=0.
    for path in ledger_paths:
        arrays=dict(np.load(path,allow_pickle=False));count=arrays['indices'].shape[0]
        assert arrays['indices'].shape==(count,batch_size)
        assert np.array_equal(arrays['indices'].reshape(-1),np.arange(offset,offset+count*batch_size))
        for step in range(count):
            record={key:value[step]for key,value in arrays.items()}
            for key in sorted(record):
                value=np.ascontiguousarray(record[key]);assert np.isfinite(value).all()
                stream.update(key.encode());stream.update(str(value.dtype).encode());stream.update(str(value.shape).encode());stream.update(value.tobytes())
            indices=record['indices']
            for key in COMMON:assert np.array_equal(record[key],planned[key][indices]),(int(indices[0]),key)
            for j,index in enumerate(indices):
                assert int(record['point_valid'][j].sum())==int(planned['valid_correspondences'][index])
                for key in GENERATED:
                    assert np.array_equal(tensor_sha(record[key][j]),planned['generated_'+key+'_sha256'][index]),(int(index),key)
                    generated_checks+=1
                for key in COMPACT:
                    assert np.array_equal(tensor_sha(record[key][j]),record['tensor_'+key+'_sha256'][j]),(int(index),'tensor_'+key)
                    tensor_checks+=1
                overlap=box_iou(record['rois'][j],record['target_box'][j]);expected=np.where(bool(record['positive'][j])&(overlap>=.5),overlap,0)
                error=float(np.max(abs(expected-record['quality_targets'][j])));roi_max_error=max(roi_max_error,error)
                assert error<2e-6,(int(index),'quality_target',error)
            supports.update(bytes(v)for v in record['raw_support_sha256']);queries.update(bytes(v)for v in record['raw_query_sha256']);programs.update(bytes(v)for v in record['program_id'])
            positive_count+=int(record['positive'].sum());point_count+=int(record['point_valid'].sum());families.update(map(int,record['family']));actual_steps+=1
        offset+=count*batch_size
        history_path=run/'history'/(path.stem+'.jsonl');payload=history_path.read_bytes();rows=[json.loads(line)for line in payload.decode().splitlines()]
        assert len(rows)==count;history.extend(rows);history_bytes.append(payload)
        ledger_details.append({'file':path.name,'sha256':sha(path),'steps':count,'history_sha256':sha(history_path)})
    assert actual_steps==steps and offset==steps*batch_size
    assert [h['step']for h in history]==list(range(1,steps+1))
    for row in history:
        step=row['step'];expected_lr=config['final_learning_rate']+(config['learning_rate']-config['final_learning_rate'])*.5*(1+math.cos(math.pi*(step-1)/max(1,config['steps']-1)))
        assert row['episodes']==step*batch_size and row['raw_rgb_views']==2*step*batch_size
        assert abs(row['learning_rate']-expected_lr)<1e-15 and math.isfinite(row['gradient_norm_before_clip'])
        assert all(math.isfinite(float(v))for v in row['losses'].values())
    checkpoint_proofs=[]
    for name,checkpoint_steps in [('random.pt',0)]+([('final.pt',steps)]if complete else[]):
        path=run/name;saved=torch.load(path,map_location='cpu',weights_only=True);state=state_sha(saved['state_dict'])
        assert saved['config']==config and saved['state_sha256']==state and saved['format']==config['model_format']
        assert saved['approval_plan_sha256']==admission['approval_plan_sha256']
        assert saved['natural_training_images']==0 and saved['borrowed_pretrained_weights']is False
        expected={'optimizer_steps':checkpoint_steps,'training_episodes':checkpoint_steps*batch_size,'training_views':2*checkpoint_steps*batch_size,'raw_rgb_views':2*checkpoint_steps*batch_size,'encoder_scale_views':4*checkpoint_steps*batch_size,'roi_training_examples':64*checkpoint_steps*batch_size}
        assert saved['training']==expected and all(saved[k]==v for k,v in expected.items())
        checkpoint_proofs.append({'file':name,'sha256':sha(path),'state_sha256':state,**expected})
    counters={'optimizer_steps':steps,'training_episodes':offset,'positive_episodes':positive_count,'negative_episodes':offset-positive_count,'valid_point_correspondences':point_count,'unique_support_rgb':len(supports),'unique_query_rgb':len(queries),'unique_rgb_across_input_roles':len(supports|queries),'unique_structural_programs':len(programs),'family_episode_counts':{str(k):v for k,v in families.items()}}
    assert positive_count==offset//2 and len(supports|queries)==2*offset and len(programs)==offset
    if complete:
        assert all(completion[k]==v for k,v in counters.items())
        assert completion['input_stream_sha256']==stream.hexdigest()
        assert b''.join(history_bytes)==(run/'history.jsonl').read_bytes() and sha(run/'history.jsonl')==completion['history_sha256']
        assert len(completion['history_shards'])==len(ledger_details)
        for bound,measured in zip(completion['history_shards'],ledger_details):assert bound['sha256']==measured['history_sha256'] and bound['steps']==measured['steps']
        assert len(completion['checkpoints'])==2
        for bound,measured in zip(completion['checkpoints'],checkpoint_proofs):
            assert bound['sha256']==measured['sha256'] and bound['state_sha256']==measured['state_sha256'] and all(bound[k]==measured[k]for k in ['optimizer_steps','training_episodes','training_views','raw_rgb_views','encoder_scale_views','roi_training_examples'])
        assert completion['state_sha256']==checkpoint_proofs[-1]['state_sha256']
        assert checkpoint_proofs[0]['state_sha256']!=checkpoint_proofs[-1]['state_sha256']
    return {'status':'passed','scope':'Independent complete consumed-stream audit'if complete else'Independent immutable committed-prefix audit; training remains in progress','run':str(run.relative_to(ROOT)).replace('\\','/'),'complete_run_audited':complete,**counters,'preflight_result_sha256':sha(PREFLIGHT/'result.json'),'preflight_episode_hashes_sha256':preflight['episode_hashes_sha256'],'source_sha256':config['source_sha256'],'approval_plan_sha256':sha(run/'approval_plan.json'),'admission_sha256':sha(run/'admission.json'),'completion_sha256':sha(run/'completion.json')if complete else None,'input_stream_sha256':stream.hexdigest(),'preflight_common_hash_fields':list(COMMON),'generated_label_hash_checks':generated_checks,'compact_tensor_hash_checks':tensor_checks,'independently_recomputed_roi_targets':offset*64,'roi_target_max_absolute_error':roi_max_error,'checkpoints':checkpoint_proofs,'ledger_shards':ledger_details,'gpu_used':False,'natural_images_read':0,'renderer_replayed':False,'project_verifier_imported':False,'limit':'Raw image/large-mask input tensors are represented by bound hashes; compact labels/ROIs are independently rehashed. This audit does not independently reproduce optimizer arithmetic.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/primary_4096_r2');p.add_argument('--prefix-steps',type=int);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);result=audit(args.run,args.prefix_steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ['ledger_shards','source_sha256']},indent=2))
