"""Bounded pre-training synthetic profile and fixed-episode plumbing check."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from hypernet.scene_curriculum import batch
from hypernet.local_relation_model import LocalRelationNet,objective,decoded_boxes,state_hash

def setup():
    torch.set_num_threads(4);torch.manual_seed(1717);np.random.seed(1717)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
    setup();device='cuda';b=batch(range(16),device=device)
    result={'scope':'Synthetic implementation checks only, not generalization or model selection','arms':{},'natural_images':0}
    for geometry in (True,False):
        setup();model=LocalRelationNet(geometry).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        initial=state_hash(model);torch.cuda.reset_peak_memory_stats();times=[]
        for step in range(4):
            torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            out=model(b['support'],b['query']);loss,stats=objective(out,b);loss.backward()
            assert all(torch.isfinite(p.grad).all().item()for p in model.parameters()if p.grad is not None)
            optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-started)
        result['arms']['primary'if geometry else'bag']={'step_seconds':times,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'initial_state_sha256':initial,'finite_gradients':True,'loss_after4':stats}
        del model,optimizer,out;torch.cuda.empty_cache()
    # A separate fixed four-episode optimization check; these weights are never
    # promoted to training or real evaluation. Both present/absent are included.
    setup();model=LocalRelationNet(True).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    tiny=batch(range(4),device=device);history=[]
    for step in range(32):
        optimizer.zero_grad(set_to_none=True);out=model(tiny['support'],tiny['query']);loss,stats=objective(out,tiny)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        history.append(stats)
    result['fixed_episode_smoke']={'steps':32,'episodes':4,'initial_loss':history[0]['loss'],'final_loss':history[-1]['loss'],
        'loss_decreased':history[-1]['loss']<history[0]['loss'],'history':history,'weights_discarded':True}
    result['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in
        (ROOT/'hypernet/local_relation_model.py',ROOT/'hypernet/scene_curriculum.py',Path(__file__))}
    path=ROOT/'experiments/local_profile.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
