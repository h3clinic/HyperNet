"""Bounded plumbing/profile checks; all fitted weights are discarded."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,argparse,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from hypernet_v18.pair_ranker import PairRanker,INPUT_KEYS,objective,parameter_summary,file_sha
from hypernet_v18.pair_curriculum import batch


def main(args):
    path=ROOT/args.output
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(2);torch.manual_seed(1801);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda':
        free,total=torch.cuda.mem_get_info()
        if total-free>8*1024**3 or free<3*1024**3:raise RuntimeError('Insufficient shared GPU headroom; do not start profile')
    checkpoint=ROOT.parent/'v17/runs/primary_2048/final.pt'
    model=PairRanker(checkpoint).to(args.device);data=batch(range(args.batch_size),device=args.device)
    optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.003,weight_decay=.0001)
    sources={str(p.relative_to(ROOT)):file_sha(p)for p in (ROOT/'hypernet_v18/pair_ranker.py',ROOT/'hypernet_v18/pair_curriculum.py',Path(__file__))}
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    history=[];start=time.perf_counter()
    for step in range(args.steps+1):
        scores=model(*(data[k]for k in INPUT_KEYS));loss,stats=objective(scores)
        history.append({'step':step,**stats})
        if step==args.steps:break
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if step%32==0:print(json.dumps({'step':step,'loss':stats['loss'],'near_accuracy':stats['near_pair_accuracy'],'wrong_accuracy':stats['wrong_family_pair_accuracy']}),flush=True)
    if args.device=='cuda':torch.cuda.synchronize()
    result={'scope':'Fixed training-pair optimization/profile only, NOT heldout performance','weights_discarded':True,
            'device':args.device,'steps':args.steps,'fixed_quartets':args.batch_size,'input_rgb_views_per_step':4*args.batch_size,
            'initial':history[0],'final':history[-1],'history':history,'seconds':time.perf_counter()-start,
            'model':parameter_summary(model),'own_initialization':model.initialization,'source_sha256':sources,
            'sources_unchanged':all(file_sha(ROOT/name)==digest for name,digest in sources.items()),
            'peak_cuda_bytes':torch.cuda.max_memory_allocated()if args.device=='cuda'else None,
            'natural_training_images':0,'borrowed_pretrained_weights':False}
    if args.device=='cuda'and result['peak_cuda_bytes']>2*1024**3:raise RuntimeError('Profile exceeded2GiB own allocation; reduce batch before proof')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('history','own_initialization','source_sha256')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--steps',type=int,default=96);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--output',default='experiments/cpu_fixed_pair_sanity.json');main(p.parse_args())
