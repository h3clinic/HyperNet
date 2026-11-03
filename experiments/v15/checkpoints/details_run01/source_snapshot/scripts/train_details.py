"""Stream exact mathematical-pattern pairs; count what actually trained.

This script never opens natural photos or downloads weights. A million-scale
run is an explicit CLI action, not the default smoke profile. Pixel and canonical
raster SHA256 ledgers distinguish distinct rendered views from geometry IDs.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.detail_encoder import DetailEncoder,FORMAT,detail_loss,state_hash


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)
def utc():return datetime.now(timezone.utc).isoformat()
def save_torch(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    torch.save(value,temporary);os.replace(temporary,path)


def tensor_batch(batch,device='cpu'):
    result={}
    for source,target in [('view_a','rgb_a'),('view_b','rgb_b')]:
        value=batch[source]
        value=value if torch.is_tensor(value) else torch.from_numpy(np.ascontiguousarray(value))
        if value.dtype!=torch.uint8:raise ValueError('Hash ledger expects original uint8 render bytes')
        result[target]=value.to(device).permute(0,3,1,2).contiguous().float()/255
    for key in ('mask_a','mask_b','detail_a','detail_b','valid_a_to_b','valid_b_to_a','grid_a_to_b','grid_b_to_a'):
        value=batch[key];result[key]=(value if torch.is_tensor(value) else torch.from_numpy(np.ascontiguousarray(value))).to(device)
    return result


def cpu_array(value):return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class PixelLedger:
    def __init__(self,directory,shard_size=8192):
        self.directory=Path(directory);self.directory.mkdir(parents=True)
        self.shard_size=shard_size;self.ids=[];self.hashes=[];self.records=[];self.total=0
    def append(self,ids,batch):
        rasters={key:cpu_array(batch[key]) for key in ('canonical_mask','canonical_raster','view_a','view_b')}
        for index,family in enumerate(ids):
            hashes=[hashlib.sha256(np.ascontiguousarray(rasters[key][index]).tobytes()).digest()
                for key in ('canonical_mask','canonical_raster','view_a','view_b')]
            self.ids.append(int(family));self.hashes.append(b''.join(hashes));self.total+=1
        if len(self.ids)>=self.shard_size:self.flush()
    def flush(self):
        if not self.ids:return
        path=self.directory/f'shard_{len(self.records):05d}.npz'
        ids=np.asarray(self.ids,np.int64);digests=np.frombuffer(b''.join(self.hashes),np.uint8).reshape(-1,4,32)
        np.savez(path,family_ids=ids,sha256=digests)
        self.records.append({'path':path.name,'families':len(ids),'sha256':sha(path)})
        self.ids=[];self.hashes=[]
    def finish(self):
        self.flush();ids=[];digests=[]
        for record in self.records:
            path=self.directory/record['path']
            if sha(path)!=record['sha256']:raise ValueError('Pixel ledger changed')
            with np.load(path,allow_pickle=False) as archive:ids.append(archive['family_ids']);digests.append(archive['sha256'])
        ids=np.concatenate(ids);digests=np.concatenate(digests)
        def unique(values):return len(np.unique(np.ascontiguousarray(values).reshape(-1,32).view('V32')))
        return {'family_visits':len(ids),'unique_family_ids':len(np.unique(ids)),
            'rendered_training_views':2*len(ids),'unique_training_rgb_sha256':unique(digests[:,2:]),
            'unique_canonical_mask_sha256':unique(digests[:,0]),'unique_canonical_raster_sha256':unique(digests[:,1]),
            'digest_columns':['canonical_mask','canonical_material_raster','view_a_rgb','view_b_rgb'],'shards':self.records,
            'definition':'Canonical raster excludes independently varied color and background; RGB hash covers original generated pixels consumed by the optimizer.'}


def save_preview(batch,path):
    from PIL import Image,ImageDraw
    count=min(8,len(batch['view_a']));size=batch['view_a'].shape[1]
    canvas=Image.new('RGB',(count*(size+8),2*(size+22)),(242,244,247));draw=ImageDraw.Draw(canvas)
    for index in range(count):
        for row,key in enumerate(('view_a','view_b')):
            canvas.paste(Image.fromarray(cpu_array(batch[key][index])),(index*(size+8),row*(size+22)))
            draw.text((index*(size+8),row*(size+22)+size+2),f'{index}: {key[-1]}',fill=(25,35,48))
    temporary=Path(path).with_suffix('.writing.png');canvas.save(temporary);os.replace(temporary,path)


def source_hashes(renderer='numpy'):
    names=['hypernet/detail_encoder.py','hypernet/math_curriculum.py','scripts/train_details.py']
    if renderer=='torch':names.append('hypernet/math_curriculum_torch.py')
    return {name:sha(ROOT/name) for name in names}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','train'],default='smoke')
    parser.add_argument('--output',type=Path,default=ROOT/'checkpoints/details_smoke01')
    parser.add_argument('--families',type=int,default=512);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--side',type=int,default=64);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--renderer',choices=['numpy','torch'],default='numpy')
    parser.add_argument('--prefetch-workers',type=int,default=1);parser.add_argument('--seed',type=int,default=1515)
    parser.add_argument('--renderer-seed',type=int,default=15150);parser.add_argument('--anchors',type=int,default=32)
    parser.add_argument('--checkpoint-every',type=int,default=65536);parser.add_argument('--validation-families',type=int,default=256)
    args=parser.parse_args()
    if args.mode=='train' and args.families!=1048576:raise ValueError('The admitted full run contains exactly1,048,576 unique families')
    if not 1<=args.families<=1048576 or args.side<32 or args.side%4 or args.batch_size<2:raise ValueError('Invalid stream configuration')
    if args.output.exists():raise FileExistsError('Preserve previous evidence; choose a new output directory')
    args.output.mkdir(parents=True);(ROOT/'assets').mkdir(exist_ok=True)
    if args.renderer=='torch':from hypernet.math_curriculum_torch import sample_batch
    else:from hypernet.math_curriculum import sample_batch
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but not available; no silent CPU fallback')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=DetailEncoder().to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    steps=(args.families+args.batch_size-1)//args.batch_size
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps,eta_min=.00003)
    sources=source_hashes(args.renderer);initial=state_hash(model)
    admission={'format':FORMAT,'created_utc':utc(),'mode':args.mode,'source_sha256':sources,'seed':args.seed,
        'renderer_seed':args.renderer_seed,'side':args.side,'batch_size':args.batch_size,'threads':args.threads,
        'device':args.device,'renderer_backend':args.renderer,'precision':'float32; no AMP or TF32',
        'deterministic_algorithms':True,'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'runtime':{'torch':str(torch.__version__),'numpy':str(np.__version__),'python':sys.version},
        'prefetch_workers':args.prefetch_workers,'anchors_per_direction':args.anchors,'target_families':args.families,
        'target_training_views':2*args.families,'parameter_count':sum(p.numel() for p in model.parameters()),
        'training_family_interval':[0,args.families],'validation_family_interval':[1048576,1048576+args.validation_families],
        'optimizer':{'name':'AdamW','initial_lr':.001,'weight_decay':.0001,'gradient_clip_norm':5},
        'scheduler':{'name':'cosine','steps':steps,'minimum_lr':.00003},
        'checkpoint_every_families':args.checkpoint_every,
        'primary_model_rule':'Final1,048,576-family checkpoint; no TEST-based checkpoint selection' if args.mode=='train' else 'Smoke only; excluded from full-run fitting budget',
        'declared_scaling_control_families':[65536,262144,1048576] if args.mode=='train' else [],
        'other_milestones_role':'Preserved recovery evidence, not additional TEST candidates',
        'natural_training_images':0,'borrowed_pretrained_weights':False,'semantic_class_labels':False,
        'initialization':'Fresh random seed, no checkpoint loaded','initial_state_sha256':initial,
        'correspondence':'Exact known canonical affine maps; conservative four-neighbor foreground target support; repeated local motifs excluded from negatives.',
        'fine_structure':'Stride-one learned path, pixel-unshuffle context branches, analytic high-frequency skip, learned stride-one reconstruction auxiliary.',
        'loss_weights':{'dense_correspondence':1,'detail_reconstruction':.5,'foreground':.15,'variance':1,'covariance':.05},
        'canonical_hash_scope':'Raster geometry plus mathematical material pattern. IDs alone do not establish unique drawings.'}
    write(args.output/'admission.json',admission)
    for name,digest in sources.items():
        source=ROOT/name;destination=args.output/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        if sha(destination)!=digest:raise ValueError('Source snapshot copy differs')
    save_torch(args.output/'random_initial.pt',{'format':FORMAT,'state_dict':model.state_dict(),'state_sha256':initial,'optimizer_steps':0,
        'natural_training_images':0,'borrowed_pretrained_weights':False,'admission':admission})
    ledger=PixelLedger(args.output/'ledger');anchor_rng=torch.Generator(device=args.device).manual_seed(args.seed+1)
    start=time.perf_counter();consumed=0;generation_wait=0.;optimization_seconds=0.;latest={};history=[]
    def generate(offset):
        ids=np.arange(offset,min(offset+args.batch_size,args.families),dtype=np.int64)
        kwargs={'device':args.device} if args.renderer=='torch' else {}
        return ids,sample_batch(ids,args.renderer_seed,size=args.side,**kwargs)
    def checkpoint(name,step):
        ledger.flush()
        save_torch(args.output/name,{'format':FORMAT,'state_dict':model.state_dict(),'state_sha256':state_hash(model),
            'optimizer_steps':step,'consumed_families':consumed,'rendered_training_views':2*consumed,
            'natural_training_images':0,'borrowed_pretrained_weights':False,'admission':admission,
            'source_sha256':sources,'ledger_shards':ledger.records})
    def progress(stage):
        elapsed=time.perf_counter()-start
        record={'format':FORMAT,'stage':stage,'run':args.output.name,'consumed_families':consumed,
            'rendered_training_views':2*consumed,'optimizer_steps':len(history),'elapsed_seconds':elapsed,
            'training_views_per_second':2*consumed/max(elapsed,1e-9),'target_families':args.families,
            'target_training_views':2*args.families,'generation_wait_seconds':generation_wait,
            'optimization_seconds':optimization_seconds,'latest':latest,'borrowed_weights':False,
            'natural_training_images':0,'preview_path':'assets/training_latest.png','updated_utc':utc()}
        write(args.output/'progress.json',record);write(ROOT/'training_progress.json',record)
        return record
    # Only a bounded number of generated batches exists ahead of the optimizer.
    # The ledger is appended after a successful update, never after prefetch.
    with ThreadPoolExecutor(max_workers=args.prefetch_workers) as pool,(args.output/'history.jsonl').open('x') as log:
        pending={};offsets=list(range(0,args.families,args.batch_size));depth=max(1,2*args.prefetch_workers)
        for position in range(min(depth,len(offsets))):pending[position]=pool.submit(generate,offsets[position])
        for position,offset in enumerate(offsets):
            waited=time.perf_counter();ids,raw=pending.pop(position).result();generation_wait+=time.perf_counter()-waited
            ahead=position+depth
            if ahead<len(offsets):pending[ahead]=pool.submit(generate,offsets[ahead])
            batch=tensor_batch(raw,args.device);model.train();began=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            both=model(torch.cat([batch['rgb_a'],batch['rgb_b']]))
            first={key:value[:len(ids)] for key,value in both.items()};second={key:value[len(ids):] for key,value in both.items()}
            loss,latest=detail_loss(first,second,batch,anchor_rng,args.anchors)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();schedule.step()
            optimization_seconds+=time.perf_counter()-began;consumed+=len(ids);ledger.append(ids,raw)
            row={'optimizer_step':position+1,'consumed_families':consumed,'elapsed_seconds':time.perf_counter()-start,**latest}
            history.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
            if position==0 or (position+1)%16==0 or consumed==args.families:
                log.flush();save_preview(raw,ROOT/'assets/training_latest.png');record=progress(args.mode)
                print(json.dumps(record,allow_nan=False),flush=True)
            if consumed%args.checkpoint_every==0 or consumed==args.families:checkpoint(f'families_{consumed:07d}.pt',position+1)
    model.eval();training_seconds=time.perf_counter()-start
    validation=[]
    with torch.inference_mode():
        val_rng=torch.Generator(device=args.device).manual_seed(151599)
        for offset in range(1048576,1048576+args.validation_families,args.batch_size):
            ids=np.arange(offset,min(offset+args.batch_size,1048576+args.validation_families),dtype=np.int64)
            kwargs={'device':args.device} if args.renderer=='torch' else {}
            batch=tensor_batch(sample_batch(ids,args.renderer_seed+1,size=args.side,**kwargs),args.device)
            encoded=model(torch.cat([batch['rgb_a'],batch['rgb_b']]))
            a={key:value[:len(ids)] for key,value in encoded.items()};b={key:value[len(ids):] for key,value in encoded.items()}
            _,stats=detail_loss(a,b,batch,val_rng,args.anchors);validation.append((len(ids),stats))
    counts=ledger.finish()
    if consumed!=args.families or counts['unique_family_ids']!=args.families:raise RuntimeError('Training stream count mismatch')
    if source_hashes(args.renderer)!=sources:raise RuntimeError('Sources changed during training; preserve incomplete run evidence')
    completed={'format':FORMAT,'completed':True,'mode':args.mode,'created_utc':utc(),'admission_sha256':sha(args.output/'admission.json'),
        'source_sha256':sources,'initial_state_sha256':initial,'final_state_sha256':state_hash(model),
        'optimizer_steps':len(history),'training_seconds':training_seconds,'training_views_per_second':2*consumed/training_seconds,
        'full_2097152_view_time_projection_seconds':2097152/(2*consumed/training_seconds),'counts':counts,
        'validation':{key:sum(n*stats[key] for n,stats in validation)/args.validation_families for key in validation[0][1]},
        'natural_training_images':0,'borrowed_pretrained_weights':False,'validation_updates':0}
    write(args.output/'completed.json',completed);progress('complete');print(json.dumps(completed,allow_nan=False),flush=True)


if __name__=='__main__':main()
