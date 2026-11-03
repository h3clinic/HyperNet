"""Fixed512-update synthetic patch-ranking proof; no real-image weight updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime,timezone
import sys,json,argparse,time,math,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from hypernet_v18.pair_ranker import PairRanker,FORMAT,INPUT_KEYS,objective,parameter_summary,file_sha
from hypernet_v18.pair_curriculum import pair,FORMAT as DATA_FORMAT,CONFIG as DATA_CONFIG
from hypernet_v18.vendor_v17.local_relation_model import state_hash


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class Pairs(Dataset):
    def __init__(self,count,seed):self.count=count;self.seed=seed
    def __len__(self):return self.count
    def __getitem__(self,index):return pair(index,split='train',seed=self.seed)


def collate(rows):
    data={key:torch.from_numpy(np.stack([r[key]for r in rows])).permute(0,3,1,2).float().div_(255)for key in INPUT_KEYS}
    data['metadata']=[r['metadata']for r in rows];return data


def worker_init(worker):torch.set_num_threads(1)


def checkpoint(path,model,step,config):
    if path.exists():raise FileExistsError(path)
    training={'optimizer_steps':step,'quartets':step*config['batch_size'],'rgb_views':4*step*config['batch_size'],
              'cumulative_own_processed_rgb_views':model.initialization['prior_training_views']+4*step*config['batch_size']}
    saved={'format':FORMAT,'state_dict':{k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
           'state_sha256':state_hash(model),'training':training,'config':config,'own_initialization':model.initialization,
           'parameter_summary':parameter_summary(model),'natural_training_images':0,'borrowed_pretrained_weights':False}
    torch.save(saved,path);loaded=torch.load(path,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,loaded['state_dict'][k])for k,v in saved['state_dict'].items())
    return {'path':str(path.relative_to(ROOT)),'sha256':file_sha(path),'state_sha256':saved['state_sha256'],**training}


def main(args):
    torch.set_num_threads(2);torch.manual_seed(1801);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run=ROOT/args.run
    if run.exists()and any(run.iterdir()):raise FileExistsError(run)
    run.mkdir(parents=True,exist_ok=True);(run/'ledger').mkdir();(run/'source_snapshot').mkdir()
    own=Path(args.own_checkpoint)
    if not own.is_absolute():own=(ROOT/own).resolve()
    model=PairRanker(own).to(args.device)
    config={'steps':args.steps,'batch_size':args.batch_size,'stream_seed':1801,'initialization_seed':1801,
            'device':args.device,'workers':args.workers,'threads':2,'learning_rate':.0005,'final_learning_rate':.00005,
            'weight_decay':.0001,'gradient_clip':5.,'dtype':'float32','amp':False,'deterministic_algorithms':True,
            'data_format':DATA_FORMAT,'data_config':DATA_CONFIG,'model_format':FORMAT,
            'checkpoint_rule':'Final planned update only; no real/syntheticTEST checkpoint selection',
            'loss':'0.5 positive logistic +0.25 same-support near-negative logistic +0.25 wrong-support positive-patch logistic'}
    sources={}
    for name in ('hypernet_v18/pair_ranker.py','hypernet_v18/pair_curriculum.py','hypernet_v18/reranker.py',
                 'hypernet_v18/vendor_v17/local_relation_model.py','hypernet_v18/vendor_v17/scene_curriculum.py',
                 'hypernet_v18/vendor_v17/local_matcher.py','scripts/train_pair_ranker.py'):
        source=ROOT/name;target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);sources[name]=file_sha(source)
    config['source_sha256']=sources
    atomic_json(run/'admission.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,
        'own_initialization':model.initialization,'parameters':parameter_summary(model),
        'natural_training_images':0,'borrowed_pretrained_weights':False,'torch_version':str(torch.__version__),
        'numpy_version':str(np.__version__),'ledger_program_encoding':'Raw32byteSHA256program ID, parsedfromhex (not hash-of-hex-string)',
        'training_input_policy':'Four RGB96px patches only; procedural program IDs, masks and boxes used for label/crop generation, never network inputs'})
    checkpoints=[checkpoint(run/'initial.pt',model,0,config)]
    optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001)
    kwargs={'batch_size':args.batch_size,'num_workers':args.workers,'shuffle':False,'collate_fn':collate,
            'worker_init_fn':worker_init,'pin_memory':args.device=='cuda'}
    if args.workers:kwargs.update(persistent_workers=True,prefetch_factor=2)
    loader=DataLoader(Pairs(args.steps*args.batch_size,1801),**kwargs)
    history=(run/'history.jsonl').open('x');start=time.perf_counter();stream=hashlib.sha256();pending=[];shards=[]
    unique_images=set();unique_programs=set();minimum_changed=None
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    for zero,cpu in enumerate(loader):
        step=zero+1;metadata=cpu['metadata'];actual=[r['index']for r in metadata]
        assert actual==list(range(zero*args.batch_size,step*args.batch_size))
        lr=.00005+(.0005-.00005)*.5*(1+math.cos(math.pi*zero/max(1,args.steps-1)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        scores=model(*(cpu[key].to(args.device,non_blocking=True)for key in INPUT_KEYS));loss,stats=objective(scores)
        if not math.isfinite(stats['loss']):raise RuntimeError('Nonfinite rankloss')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        record={'indices':np.asarray(actual,np.int64)}
        for key in INPUT_KEYS:record[key+'_sha256']=np.stack([np.frombuffer(bytes.fromhex(r['image_sha256'][key]),np.uint8)for r in metadata])
        for key in ('program_id','near_negative_program_id','wrong_program_id'):
            record[key]=np.stack([np.frombuffer(bytes.fromhex(r[key]),np.uint8)for r in metadata])
        for key in sorted(record):stream.update(key.encode());stream.update(record[key].tobytes())
        pending.append(record)
        for r in metadata:
            unique_programs.add(r['program_id']);unique_images.update(r['image_sha256'].values())
            minimum_changed=r['final_changed_rgb_pixels']if minimum_changed is None else min(minimum_changed,r['final_changed_rgb_pixels'])
        history.write(json.dumps({'step':step,'quartets':step*args.batch_size,'learning_rate':lr,
                                  'gradient_norm_before_clip':float(gradient),'seconds':time.perf_counter()-start,**stats})+'\n')
        if len(pending)==64 or step==args.steps:
            ledger=run/'ledger'/f'{step-len(pending)+1:06d}_{step:06d}.npz'
            np.savez_compressed(ledger,**{key:np.stack([row[key]for row in pending])for key in pending[0]})
            shards.append({'path':str(ledger.relative_to(ROOT)),'sha256':file_sha(ledger),'steps':len(pending)});pending=[]
        if step==1 or step%32==0 or step==args.steps:
            history.flush();elapsed=time.perf_counter()-start
            progress={'status':'training','run':args.run,'step':step,'planned':args.steps,'consumed_quartets':step*args.batch_size,
                      'consumed_rgb_views':4*step*args.batch_size,'unique_target_programs':len(unique_programs),
                      'unique_input_images':len(unique_images),'seconds':elapsed,'estimated_remaining_seconds':elapsed/step*(args.steps-step),
                      'metrics_scope':'Synthetic training loss/pair rankings only, not heldout accuracy','losses':stats,
                      'updated_utc':datetime.now(timezone.utc).isoformat()}
            atomic_json(run/'progress.json',progress);atomic_json(ROOT/'training_progress.json',progress)
            print(json.dumps({k:progress[k]for k in ('status','step','planned','consumed_quartets','seconds')}),flush=True)
    history.close();assert all(file_sha(ROOT/name)==digest for name,digest in sources.items())
    checkpoints.append(checkpoint(run/'final.pt',model,args.steps,config))
    done={'status':'complete','config':config,'own_initialization':model.initialization,'optimizer_steps':args.steps,
          'training_quartets':args.steps*args.batch_size,'training_rgb_views':4*args.steps*args.batch_size,
          'cumulative_own_processed_rgb_views':model.initialization['prior_training_views']+4*args.steps*args.batch_size,
          'unique_target_programs':len(unique_programs),'unique_input_images':len(unique_images),
          'minimum_final_changed_rgb_pixels':minimum_changed,'input_stream_sha256':stream.hexdigest(),
          'source_sha256':sources,'checkpoints':checkpoints,'ledger_shards':shards,'history_sha256':file_sha(run/'history.jsonl'),
          'state_sha256':state_hash(model),'seconds':time.perf_counter()-start,
          'peak_cuda_bytes':torch.cuda.max_memory_allocated()if args.device=='cuda'else None,
          'natural_training_images':0,'borrowed_pretrained_weights':False}
    atomic_json(run/'completion.json',done);progress.update(status='complete',estimated_remaining_seconds=0)
    atomic_json(run/'progress.json',progress);atomic_json(ROOT/'training_progress.json',progress)
    print(json.dumps({'complete':True,'seconds':done['seconds'],'state_sha256':done['state_sha256']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/proof512')
    p.add_argument('--own-checkpoint',default='../v17/runs/primary_2048/final.pt')
    p.add_argument('--steps',type=int,default=512);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args()
    if args.steps<1 or args.batch_size<1:p.error('Positive steps and batch size required')
    main(args)
