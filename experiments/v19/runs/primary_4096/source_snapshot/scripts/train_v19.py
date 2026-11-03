"""Fixed-budget synthetic-only v19 training, with actual-consumption ledgers."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,math,random,sys,time,warnings
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet_v19.curriculum import episode,FORMAT as DATA_FORMAT,CONFIG as DATA_CONFIG
from hypernet_v19.model import DenseMatchNet,FORMAT,MODEL_CONFIG,state_hash,file_sha,parameter_summary
from hypernet_v19.training import collate,objective,ARRAY_KEYS,TRAINING_CONFIG
from hypernet_v19.proposals import PROPOSAL_CONFIG

SOURCE_NAMES=('hypernet_v19/__init__.py','hypernet_v19/curriculum.py','hypernet_v19/model.py',
              'hypernet_v19/proposals.py','hypernet_v19/training.py','scripts/train_v19.py')
COMPACT_KEYS=('boxes','target_box','positive','support_points','query_points','support_point_valid',
              'query_point_valid','point_valid','rois','quality_targets')

def canonical(value):return json.loads(json.dumps(value,allow_nan=False))
def json_bytes(value):return(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def tensor_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).digest()
def hex_bytes(value):
    value=bytes.fromhex(value)
    if len(value)!=32:raise ValueError('Expected a SHA256 program/image digest')
    return np.frombuffer(value,np.uint8).copy()
def atomic_bytes(path,payload,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.writing')
    with temp.open('xb')as f:f.write(payload);f.flush();os.fsync(f.fileno())
    if exclusive and path.exists():raise FileExistsError('Final file appeared during atomic write: '+str(path))
    temp.replace(path)
def atomic_json(path,value,exclusive=False):atomic_bytes(path,json_bytes(value),exclusive)
def check_sources(sources):
    for name,expected in sources.items():
        if file_sha(ROOT/name)!=expected:raise RuntimeError('Training source changed: '+name)

def setup(seed,threads):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

class Episodes(Dataset):
    def __init__(self,count,seed,difficulty):self.count=count;self.seed=seed;self.difficulty=difficulty
    def __len__(self):return self.count
    def __getitem__(self,index):return episode(index,split='train',seed=self.seed,difficulty=self.difficulty)

def checked_collate(rows):
    data=collate(rows)
    if set(ARRAY_KEYS)-set(data):raise ValueError('Collate omitted declared model/label arrays')
    batch=len(rows);hashes={}
    for key in ARRAY_KEYS:
        value=data[key]
        if not torch.is_tensor(value)or value.device.type!='cpu'or value.shape[0]!=batch:raise ValueError('Expected per-episode CPU tensors: '+key)
        array=value.detach().contiguous().numpy()
        if not np.isfinite(array).all():raise ValueError('Nonfinite collated input/target: '+key)
        hashes['tensor_'+key+'_sha256']=np.stack([np.frombuffer(tensor_hash(a),np.uint8)for a in array])
    for key in('support','query'):
        raw=[]
        recovered=(data[key].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
        for index,row in enumerate(rows):
            image=np.asarray(row[key])
            if image.dtype!=np.uint8 or not np.array_equal(recovered[index],image):raise ValueError('Collated RGB differs from generated image')
            digest=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            if digest!=row['metadata']['image_sha256'][key]:raise ValueError('Renderer RGB hash differs from actual bytes')
            raw.append(hex_bytes(digest))
        hashes['raw_'+key+'_sha256']=np.stack(raw)
    data['ledger_hashes']=hashes
    return data

def worker_init(worker):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    seed=torch.initial_seed()%(2**32);np.random.seed(seed);random.seed(seed)

def config_for(args):
    if args.episodes<1 or args.batch_size<1 or args.episodes%args.batch_size:raise ValueError('Fixed episode budget must be positive and divisible by batch size')
    if args.workers<0 or args.threads<1 or not 0<=args.difficulty<=1:raise ValueError('Invalid worker/thread/difficulty setting')
    run=(ROOT/args.run).resolve()
    if not run.is_relative_to(ROOT)or run==ROOT:raise ValueError('Run directory must be inside v19')
    return canonical({'format':'hypernet.v19.training_configuration.v1','run':str(run.relative_to(ROOT)).replace('\\','/'),
      'episodes':args.episodes,'steps':args.episodes//args.batch_size,'batch_size':args.batch_size,'initialization_seed':args.seed,'stream_seed':args.stream_seed,
      'device':args.device,'workers':args.workers,'threads':args.threads,'difficulty':args.difficulty,'learning_rate':args.lr,'final_learning_rate':args.final_lr,
      'weight_decay':.0001,'gradient_clip':5.,'optimizer':'AdamW','dtype':'float32','amp':False,'deterministic_algorithms':True,
      'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'model_format':FORMAT,'model_config':MODEL_CONFIG,
      'curriculum_format':DATA_FORMAT,'curriculum_config':DATA_CONFIG,'training_objective_config':TRAINING_CONFIG,'proposal_config':PROPOSAL_CONFIG,
      'source_sha256':{name:file_sha(ROOT/name)for name in SOURCE_NAMES},'natural_training_images':0,'borrowed_pretrained_weights':False,
      'checkpoint_rule':'Random initialization and final planned update only; no DEV/TEST checkpoint selection','smoke':bool(args.smoke),
      'ledger_flush_steps':64,'raw_rgb_views_per_episode':2,'encoder_scale_views_per_episode':4})

def save_checkpoint(path,model,step,config,approval_sha):
    if path.exists():raise FileExistsError(path)
    episodes=step*config['batch_size'];training={'optimizer_steps':step,'training_episodes':episodes,'training_views':2*episodes,
      'raw_rgb_views':2*episodes,'encoder_scale_views':4*episodes,'roi_training_examples':episodes*TRAINING_CONFIG['rois_per_episode']}
    value={'format':FORMAT,'state_dict':{k:v.detach().cpu().clone()for k,v in model.state_dict().items()},'state_sha256':state_hash(model),
      'training':training,**training,'config':config,'approval_plan_sha256':approval_sha,'parameter_summary':parameter_summary(model),
      'natural_training_images':0,'borrowed_pretrained_weights':False,'torch_version':str(torch.__version__)}
    buffer=io.BytesIO();torch.save(value,buffer);payload=buffer.getvalue()
    check=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if check['state_sha256']!=value['state_sha256']or any(not torch.equal(t,check['state_dict'][k])for k,t in value['state_dict'].items()):raise RuntimeError('Checkpoint roundtrip changed weights')
    atomic_bytes(path,payload,exclusive=True)
    return {'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':file_sha(path),'state_sha256':value['state_sha256'],**training}

def ledger_record(cpu):
    metadata=cpu['metadata'];record={'indices':np.array([r['index']for r in metadata],np.int64),**cpu['ledger_hashes']}
    for key in COMPACT_KEYS:record[key]=cpu[key].detach().contiguous().numpy().copy()
    for key in('program_id','near_program_id','replacement_program_id'):
        record[key]=np.stack([hex_bytes(r[key])for r in metadata])
    record['instance_program_ids']=np.stack([np.stack([hex_bytes(i['program_id'])for i in r['instances']])for r in metadata])
    record['metadata_sha256']=np.stack([np.frombuffer(hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest(),np.uint8)for r in metadata])
    record['family']=np.array([r['program']['family']for r in metadata],np.int16)
    return record

def update_stream(stream,record):
    for key in sorted(record):
        array=np.ascontiguousarray(record[key]);stream.update(key.encode());stream.update(str(array.dtype).encode());stream.update(str(array.shape).encode());stream.update(array.tobytes())

def train(args):
    config=config_for(args);sources=config['source_sha256'];approval_sha=None
    if not args.smoke:
        if args.approval_plan is None:raise ValueError('Non-smoke training requires the root-approved fixed configuration JSON')
        approval=json.loads(args.approval_plan.read_text())
        if approval.get('approved_for_training')is not True or approval.get('config')!=config:raise ValueError('Root approval plan does not match exact training config/source hashes')
        approval_sha=file_sha(args.approval_plan)
    run=ROOT/config['run']
    if run.exists()and any(run.iterdir()):raise FileExistsError('Preserve prior run: '+str(run))
    run.mkdir(parents=True,exist_ok=True);(run/'ledger').mkdir();(run/'history').mkdir();(run/'source_snapshot').mkdir()
    for name,expected in sources.items():
        source=ROOT/name;destination=run/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        atomic_bytes(destination,source.read_bytes(),exclusive=True)
        if file_sha(destination)!=expected:raise RuntimeError('Source changed while snapshotting')
    if args.approval_plan:atomic_bytes(run/'approval_plan.json',args.approval_plan.read_bytes(),exclusive=True)
    setup(args.seed,args.threads);model=DenseMatchNet().to(args.device)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    atomic_json(run/'admission.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,'approval_plan_sha256':approval_sha,
      'parameters':parameter_summary(model),'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),'opencv_version':cv2.__version__,
      'cuda_device':torch.cuda.get_device_name(0)if args.device=='cuda'else None,
      'ledger_encoding':'Raw RGB SHA256 hashes bytes only. Tensor hashes and stream entries include dtype+shape+bytes. Program IDs parse32bytes from hex.',
      'supervision':'RGB enters the model; synthetic masks/correspondence/ROI-quality targets supervise losses. No natural photos or external pretrained weights.',
      'history_policy':'Immutable atomic64-step history/ledger shards after successful optimizer updates; final history.jsonl assembled atomically.'},exclusive=True)
    checkpoints=[save_checkpoint(run/'random.pt',model,0,config,approval_sha)]
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(args.seed)
    kwargs={'batch_size':args.batch_size,'shuffle':False,'num_workers':args.workers,'collate_fn':checked_collate,'worker_init_fn':worker_init,
            'pin_memory':args.device=='cuda','generator':generator,'drop_last':False}
    if args.workers:kwargs.update(prefetch_factor=2,persistent_workers=True)
    loader=DataLoader(Episodes(args.episodes,args.stream_seed,args.difficulty),**kwargs)
    pending=[];pending_history=[];shards=[];history_shards=[];stream=hashlib.sha256();start=time.perf_counter();last_end=start
    unique_support=set();unique_query=set();unique_programs=set();families={};positive_count=0;valid_correspondences=0;committed_steps=0;step=0
    def flush_pending(last_step):
        nonlocal pending,pending_history,committed_steps
        if not pending:return
        first=last_step-len(pending)+1;stem=f'{first:06d}_{last_step:06d}'
        arrays={key:np.stack([record[key]for record in pending])for key in pending[0]}
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);ledger=run/'ledger'/(stem+'.npz');atomic_bytes(ledger,buffer.getvalue(),exclusive=True)
        history=run/'history'/(stem+'.jsonl');atomic_bytes(history,(''.join(json.dumps(row,allow_nan=False)+'\n'for row in pending_history)).encode(),exclusive=True)
        shards.append({'path':str(ledger.relative_to(ROOT)).replace('\\','/'),'sha256':file_sha(ledger),'steps':len(pending)})
        history_shards.append({'path':str(history.relative_to(ROOT)).replace('\\','/'),'sha256':file_sha(history),'steps':len(pending)})
        committed_steps=last_step;pending=[];pending_history=[]
    try:
        with warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter('always')
            for zero,cpu in enumerate(loader):
                step_start=time.perf_counter();data_wait=step_start-last_end;step=zero+1
                actual=[int(r['index'])for r in cpu['metadata']]
                if actual!=list(range(zero*args.batch_size,step*args.batch_size)):raise RuntimeError('Actual DataLoader order differs from fixed episode budget')
                if not torch.are_deterministic_algorithms_enabled():raise RuntimeError('Deterministic training was disabled')
                batch={key:value.to(args.device,non_blocking=True)if torch.is_tensor(value)else value for key,value in cpu.items()if key!='ledger_hashes'}
                lr=args.final_lr+(args.lr-args.final_lr)*.5*(1+math.cos(math.pi*zero/max(1,config['steps']-1)))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);loss,stats=objective(model,batch,batch['rois'])
                if not math.isfinite(float(loss.detach()))or not all(math.isfinite(float(v))for v in stats.values()):raise RuntimeError('Nonfinite training loss/statistics')
                loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                if args.device=='cuda':torch.cuda.synchronize()
                record=ledger_record(cpu);update_stream(stream,record);pending.append(record)
                for row,support_hash,query_hash in zip(cpu['metadata'],record['raw_support_sha256'],record['raw_query_sha256']):
                    unique_support.add(support_hash.tobytes());unique_query.add(query_hash.tobytes());unique_programs.add(row['program_id'])
                    family=str(row['program']['family']);families[family]=families.get(family,0)+1
                positive_count+=int(cpu['positive'].sum());valid_correspondences+=int(cpu['point_valid'].sum())
                last_end=time.perf_counter();history={'step':step,'episodes':step*args.batch_size,'raw_rgb_views':2*step*args.batch_size,
                  'seconds':last_end-start,'step_seconds':last_end-step_start,'data_wait_seconds':data_wait,'learning_rate':lr,
                  'gradient_norm_before_clip':float(gradient),'losses':stats}
                pending_history.append(history)
                if len(pending)==64 or step==config['steps']:check_sources(sources);flush_pending(step)
                if step==1 or step%16==0 or step==config['steps']:
                    progress={'status':'training','run':config['run'],'smoke':args.smoke,'step':step,'planned_steps':config['steps'],
                      'consumed_episodes':step*args.batch_size,'planned_episodes':args.episodes,'consumed_raw_rgb_views':2*step*args.batch_size,
                      'ledger_committed_steps':committed_steps,'seconds':last_end-start,'estimated_remaining_seconds':(last_end-start)/step*(config['steps']-step),
                      'losses':stats,'updated_utc':datetime.now(timezone.utc).isoformat(),'metrics_scope':'Synthetic training losses, not natural-photo accuracy'}
                    atomic_json(run/'progress.json',progress);atomic_json(run/'history_latest.json',history)
                    if not args.smoke:atomic_json(ROOT/'training_progress.json',progress)
                    print(json.dumps({key:progress[key]for key in('status','run','step','planned_steps','consumed_episodes','seconds')}),flush=True)
            if step!=config['steps']:raise RuntimeError('DataLoader ended before the fixed budget')
            check_sources(sources);flush_pending(step)
            history_payload=b''.join((ROOT/item['path']).read_bytes()for item in history_shards);atomic_bytes(run/'history.jsonl',history_payload,exclusive=True)
            checkpoints.append(save_checkpoint(run/'final.pt',model,step,config,approval_sha))
            warning_records=sorted(set(str(item.message)for item in caught))
        completion={'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'config':config,'approval_plan_sha256':approval_sha,
          'optimizer_steps':step,'training_episodes':args.episodes,'training_views':2*args.episodes,'encoder_scale_views':4*args.episodes,
          'positive_episodes':positive_count,'negative_episodes':args.episodes-positive_count,'valid_point_correspondences':valid_correspondences,
          'unique_structural_programs':len(unique_programs),'unique_support_rgb':len(unique_support),'unique_query_rgb':len(unique_query),
          'unique_rgb_across_input_roles':len(unique_support|unique_query),'family_episode_counts':families,
          'input_stream_sha256':stream.hexdigest(),'source_sha256':sources,'checkpoints':checkpoints,'ledger_shards':shards,'history_shards':history_shards,
          'history_sha256':file_sha(run/'history.jsonl'),'state_sha256':state_hash(model),'seconds':time.perf_counter()-start,
          'peak_cuda_bytes':torch.cuda.max_memory_allocated()if args.device=='cuda'else None,'warnings':warning_records,
          'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'natural_training_images':0,'borrowed_pretrained_weights':False}
        atomic_json(run/'completion.json',completion,exclusive=True)
        progress.update(status='complete',seconds=completion['seconds'],estimated_remaining_seconds=0,ledger_committed_steps=step)
        atomic_json(run/'progress.json',progress)
        if not args.smoke:atomic_json(ROOT/'training_progress.json',progress)
        print(json.dumps({'complete':True,'run':config['run'],'episodes':args.episodes,'seconds':completion['seconds'],'peak_cuda_bytes':completion['peak_cuda_bytes'],
          'final_checkpoint_sha256':checkpoints[-1]['sha256'],'state_sha256':completion['state_sha256']}),flush=True)
    except BaseException as error:
        # Preserve successful updates already logged; never produce a final
        # checkpoint or claim completion after unsupported/nondeterministic ops.
        successful_steps=sum(item['steps']for item in shards)+len(pending)
        if pending:flush_pending(successful_steps)
        failure={'status':'failed','run':config['run'],'step_attempted':step,'successful_steps_recorded':successful_steps,
          'consumed_episodes_recorded':successful_steps*args.batch_size,'error_type':type(error).__name__,'error':str(error),
          'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'determinism_disabled':False,
          'updated_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources,'input_stream_sha256':stream.hexdigest(),
          'ledger_shards':shards,'history_shards':history_shards}
        atomic_json(run/'failure.json',failure,exclusive=True);atomic_json(run/'progress.json',failure)
        if not args.smoke:atomic_json(ROOT/'training_progress.json',failure)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--episodes',type=int,default=32768)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=1919);p.add_argument('--stream-seed',type=int,default=1901)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--difficulty',type=float,default=1.);p.add_argument('--lr',type=float,default=.0005);p.add_argument('--final-lr',type=float,default=.00005)
    p.add_argument('--approval-plan',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--print-config',action='store_true');a=p.parse_args()
    if a.print_config:print(json.dumps(config_for(a),indent=2))
    else:train(a)
