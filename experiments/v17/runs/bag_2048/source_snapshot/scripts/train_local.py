"""Fixed-budget synthetic-only v17 episodes, with actual-consumption hashes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from datetime import datetime,timezone
import sys,argparse,json,hashlib,time,math,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageDraw
from hypernet.scene_curriculum import episode,FORMAT as CURRICULUM_FORMAT,CONFIG as CURRICULUM_CONFIG
from hypernet.local_relation_model import LocalRelationNet,FORMAT,objective,state_hash


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def setup(seed,threads):
    torch.set_num_threads(threads);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


class Episodes(Dataset):
    def __init__(self,count,seed,support_size,query_size):
        self.count=count;self.seed=seed;self.support_size=support_size;self.query_size=query_size
    def __len__(self):return self.count
    def __getitem__(self,index):return episode(index,split='train',seed=self.seed,support_size=self.support_size,query_size=self.query_size)


def collate(rows):
    batch={}
    for key in ('support','query'):
        batch[key]=torch.from_numpy(np.stack([r[key]for r in rows])).permute(0,3,1,2).float().div_(255)
    for key in ('support_mask','target_mask','query_foreground_mask'):
        batch[key]=torch.from_numpy(np.stack([r[key]for r in rows]))[:,None].float()
    batch['bbox']=torch.from_numpy(np.stack([r['bbox']for r in rows])).float()
    batch['present']=torch.tensor([r['present']for r in rows],dtype=torch.float32)
    batch['metadata']=[{**r['metadata'],'program_id':r['program_id'],'family':r['family']}for r in rows]
    batch['input_hashes']={key:np.stack([np.frombuffer(hashlib.sha256(r[key].tobytes()).digest(),np.uint8)for r in rows])
                           for key in ('support','query','support_mask','target_mask','query_foreground_mask')}
    return batch


def worker_init(worker_id):
    torch.set_num_threads(1)


def write_checkpoint(path,model,steps,config):
    if path.exists():raise FileExistsError(path)
    value={'format':FORMAT,'state_dict':{k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
           'state_sha256':state_hash(model),'optimizer_steps':steps,'training_episodes':steps*config['batch_size'],
           'training_views':2*steps*config['batch_size'],'natural_training_images':0,'borrowed_pretrained_weights':False,
           'config':config,'torch_version':str(torch.__version__)}
    torch.save(value,path)
    check=torch.load(path,map_location='cpu',weights_only=True)
    if check['state_sha256']!=value['state_sha256']or any(not torch.equal(v,check['state_dict'][k])for k,v in value['state_dict'].items()):
        raise RuntimeError('Checkpoint roundtrip mismatch')
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'state_sha256':value['state_sha256'],'optimizer_steps':steps}


def preview(batch,path,step,episodes):
    support=(batch['support'].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
    query=(batch['query'].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
    count=min(4,len(support));ss=support.shape[1];qs=query.shape[1];cw=max(ss,qs)+12
    canvas=Image.new('RGB',(cw*count,ss+qs+98),'#101923');draw=ImageDraw.Draw(canvas)
    draw.text((7,8),f'ACTUALLY CONSUMED synthetic episodes: {episodes:,}; optimizer step {step}',fill='white')
    for i in range(count):
        x=i*cw+6;draw.text((x,31),f"{batch['metadata'][i]['family']} / support",fill='white')
        canvas.paste(Image.fromarray(support[i]),(x,49));y=ss+59
        draw.text((x,y),'query / target present'if batch['present'][i]else'query / target absent',fill='white')
        canvas.paste(Image.fromarray(query[i]),(x,y+20))
        if batch['present'][i]:
            x1,y1,x2,y2=batch['bbox'][i].tolist();draw.rectangle((x+x1,y+20+y1,x+x2,y+20+y2),outline='#6effab',width=1)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.writing.png');canvas.save(tmp);tmp.replace(path)


def train(args):
    setup(args.seed,args.threads)
    run=ROOT/args.run
    if run.exists()and any(run.iterdir()):raise FileExistsError('Preserve prior run: '+str(run))
    run.mkdir(parents=True,exist_ok=True);(run/'ledger').mkdir();(run/'source_snapshot').mkdir()
    config={'steps':args.steps,'batch_size':args.batch_size,'support_size':args.support_size,'query_size':args.query_size,
            'seed':args.seed,'stream_seed':args.stream_seed,'threads':args.threads,'workers':args.workers,'device':args.device,
            'variant':args.variant,'geometry':args.variant=='primary','learning_rate':args.lr,'weight_decay':.0001,
            'final_lr':.00003,'gradient_clip':5.,'dtype':'float32','amp':False,'deterministic_algorithms':True,
            'curriculum_format':CURRICULUM_FORMAT,'curriculum_config':CURRICULUM_CONFIG,'model_format':FORMAT,
            'natural_training_images':0,'borrowed_pretrained_weights':False,'smoke':bool(args.smoke)}
    sources={}
    for name in ('hypernet/scene_curriculum.py','hypernet/local_relation_model.py','scripts/train_local.py'):
        source=ROOT/name;target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);sources[name]=sha(source)
    config['source_sha256']=sources
    atomic_json(run/'admission.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,'source_sha256':sources,
        'checkpoint_rule':'Final planned step only; no real TEST or synthetic diagnosis checkpoint selection',
        'ablation':'Identical initialization/episodes/architecture/budget; replaces geometry-aligned tolerant matches with regional log-mean-exp bag aggregation. Different matching operators, not only an auxiliary-loss toggle.',
        'views_planned':2*args.steps*args.batch_size,'episodes_planned':args.steps*args.batch_size,
        'data_scope':'Procedural 2D structural programs, not natural semantic classes or unlimited visual coverage',
        'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),
        'cuda_device':torch.cuda.get_device_name(0)if args.device.startswith('cuda')else None})
    model=LocalRelationNet(geometry=config['geometry']).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    checkpoints=[write_checkpoint(run/'random.pt',model,0,config)]
    dataset=Episodes(args.steps*args.batch_size,args.stream_seed,args.support_size,args.query_size)
    loader_args={'batch_size':args.batch_size,'shuffle':False,'num_workers':args.workers,'collate_fn':collate,
                 'worker_init_fn':worker_init,'pin_memory':args.device.startswith('cuda')}
    if args.workers:loader_args.update(prefetch_factor=2,persistent_workers=True)
    loader=DataLoader(dataset,**loader_args)
    start=time.perf_counter();pending=[];shards=[];stream=hashlib.sha256()
    unique_programs=set();unique_support=set();unique_query=set();families={};present_count=0
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    history=(run/'history.jsonl').open('x',encoding='utf-8')
    for zero_step,cpu in enumerate(loader):
        step=zero_step+1
        expected=list(range(zero_step*args.batch_size,step*args.batch_size))
        actual=[int(r['index'])for r in cpu['metadata']]
        if actual!=expected:raise RuntimeError('Actual episode order differs')
        batch={k:v.to(args.device,non_blocking=True)if isinstance(v,torch.Tensor)else v for k,v in cpu.items()}
        learning_rate=.00003+(args.lr-.00003)*.5*(1+math.cos(math.pi*zero_step/max(1,args.steps-1)))
        for group in optimizer.param_groups:group['lr']=learning_rate
        optimizer.zero_grad(set_to_none=True);out=model(batch['support'],batch['query']);loss,stats=objective(out,batch)
        if not math.isfinite(stats['loss']):raise RuntimeError('Nonfinite loss')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        # Consumption ledger is updated only after the optimizer succeeds.
        record={'indices':np.asarray(actual,np.int64),'present':cpu['present'].numpy().astype(np.uint8),
                'bbox':cpu['bbox'].numpy().astype(np.float32),
                'program_sha256':np.stack([np.frombuffer(hashlib.sha256(r['program_id'].encode()).digest(),np.uint8)for r in cpu['metadata']]),
                **{key+'_sha256':value for key,value in cpu['input_hashes'].items()}}
        for key in sorted(record):stream.update(key.encode());stream.update(record[key].tobytes())
        pending.append(record)
        for i,row in enumerate(cpu['metadata']):
            unique_programs.add(row['program_id']);families[row['family']]=families.get(row['family'],0)+1
            unique_support.add(cpu['input_hashes']['support'][i].tobytes());unique_query.add(cpu['input_hashes']['query'][i].tobytes())
        present_count+=int(cpu['present'].sum())
        history.write(json.dumps({'step':step,'episodes':step*args.batch_size,'learning_rate':learning_rate,
            'gradient_norm_before_clip':float(gradient),'seconds':time.perf_counter()-start,**stats},allow_nan=False)+'\n')
        if len(pending)==64 or step==args.steps:
            ledger=run/'ledger'/f'{step-len(pending)+1:06d}_{step:06d}.npz'
            np.savez_compressed(ledger,**{key:np.stack([r[key]for r in pending])for key in pending[0]})
            shards.append({'path':str(ledger.relative_to(ROOT)),'sha256':sha(ledger),'steps':len(pending)});pending=[]
        if step%32==0 or step==1 or step==args.steps:
            history.flush();elapsed=time.perf_counter()-start
            progress={'status':'training','run':args.run,'variant':args.variant,'step':step,'planned':args.steps,
                'consumed_episodes':step*args.batch_size,'consumed_images':2*step*args.batch_size,'seconds':elapsed,
                'estimated_remaining_seconds':elapsed/step*(args.steps-step),'losses':stats,
                'unique_programs':len(unique_programs),'unique_support_rgb':len(unique_support),'unique_query_rgb':len(unique_query),
                'updated_utc':datetime.now(timezone.utc).isoformat(),'metrics_scope':'Synthetic training loss only; not real accuracy'}
            atomic_json(run/'progress.json',progress);atomic_json(ROOT/'training_progress.json',progress)
            print(json.dumps({k:progress[k]for k in ('status','run','step','planned','consumed_episodes','seconds')}),flush=True)
        if step==1 or step%256==0 or step==args.steps:preview(cpu,ROOT/'assets/training_latest.png',step,step*args.batch_size)
        if step%2048==0 and step!=args.steps:checkpoints.append(write_checkpoint(run/f'intermediate_{step:06d}.pt',model,step,config))
    history.close()
    if any(sha(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('Training source changed during run')
    checkpoints.append(write_checkpoint(run/'final.pt',model,args.steps,config))
    completion={'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'config':config,
        'optimizer_steps':args.steps,'training_episodes':args.steps*args.batch_size,'training_views':2*args.steps*args.batch_size,
        'positive_episodes':present_count,'negative_episodes':args.steps*args.batch_size-present_count,
        'unique_structural_programs':len(unique_programs),'unique_support_rgb':len(unique_support),'unique_query_rgb':len(unique_query),
        'unique_rgb_across_input_roles':len(unique_support|unique_query),'family_episode_counts':families,
        'input_stream_sha256':stream.hexdigest(),'source_sha256':sources,'checkpoints':checkpoints,'ledger_shards':shards,
        'history_sha256':sha(run/'history.jsonl'),'state_sha256':state_hash(model),'seconds':time.perf_counter()-start,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated()if args.device.startswith('cuda')else None,
        'natural_training_images':0,'borrowed_pretrained_weights':False}
    atomic_json(run/'completion.json',completion)
    progress.update(status='complete',seconds=completion['seconds'],estimated_remaining_seconds=0)
    atomic_json(run/'progress.json',progress);atomic_json(ROOT/'training_progress.json',progress)
    print(json.dumps({'complete':True,'run':args.run,'seconds':completion['seconds'],'unique_rgb':completion['unique_rgb_across_input_roles'],
                      'state_sha256':completion['state_sha256']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--steps',type=int,default=8192);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--support-size',type=int,default=96);p.add_argument('--query-size',type=int,default=192)
    p.add_argument('--seed',type=int,default=1717);p.add_argument('--stream-seed',type=int,default=1701)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--variant',choices=('primary','bag'),default='primary');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.steps<1 or args.batch_size<1 or args.workers<0:p.error('Positive step/batch counts and nonnegative workers required')
    if args.support_size!=96:p.error('Frozen support framing requires96px')
    prior_run_exists=(ROOT/args.run).exists()and any((ROOT/args.run).iterdir())
    try:
        train(args)
    except BaseException as exc:
        if prior_run_exists:raise
        progress_path=ROOT/args.run/'progress.json'
        failure=json.loads(progress_path.read_text())if progress_path.exists()else{
            'run':args.run,'variant':args.variant,'step':0,'planned':args.steps,
            'consumed_episodes':0,'consumed_images':0}
        failure.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                       updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(progress_path,failure);atomic_json(ROOT/'training_progress.json',failure)
        raise
