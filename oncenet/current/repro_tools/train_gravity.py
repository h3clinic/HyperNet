"""Portable reproduction of the selected OnceNet procedural training recipe.

No action is run by default. --train is an explicit NEW 800-step experiment,
never an overwrite or resumption of the historical frozen run. --smoke performs
one synthetic CPU forward pass only, with no optimization or accuracy test.
The original mathematical generator, objective, architecture and saved random
initial states are imported unchanged from the packaged workspace.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, hashlib, importlib.util, json, random, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SEEDS=(45011,45029,45047)
OFFSETS=(4600000,4610000,4620000)
DURATIONS=('d05','d1','d2','d4')

def workspace_default():
    candidates=[HERE.parent/'workspace',HERE/'workspace',*HERE.parents]
    return next((p for p in candidates if (p/'outputs/hypernet_v46/spectrogram/generator.py').is_file()),HERE.parent/'workspace')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def verify(workspace,seed):
    bindings=json.loads((HERE/'gravity_training_bindings.json').read_text(encoding='utf-8'))
    required=bindings['training_source_files']+[
        next(x for x in bindings['random_initial_states'] if x['seed']==seed)]
    for row in required:
        path=workspace/row['relative_path']
        if not path.is_file() or sha(path)!=row['sha256']:
            raise RuntimeError('Original training source/initial state mismatch: '+str(path))
    return bindings

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def load_recipe(workspace):
    multiseed=workspace/'outputs/hypernet_v45/multiseed'
    # The historical objective imports sibling modules by their original names.
    # This process is dedicated to this recipe, so explicit path precedence is safe.
    sys.path.insert(0,str(multiseed))
    objective=module('oncenet_original_diversity_objective',multiseed/'diversity/train.py')
    generator=module('oncenet_original_spectral_generator',workspace/'outputs/hypernet_v46/spectrogram/generator.py')
    return objective,generator

def run(args):
    workspace=args.workspace.resolve();bindings=verify(workspace,args.seed)
    import numpy as np
    import torch
    objective,generator=load_recipe(workspace)
    torch.set_num_threads(2)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False
    device='cpu' if args.smoke else args.device
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_per_process_memory_fraction(.45)
    initial=next(x for x in bindings['random_initial_states'] if x['seed']==args.seed)
    net=objective.load_encoder(workspace/initial['relative_path'],device).train()
    count=sum(p.numel() for p in net.parameters())
    if count!=42480:raise RuntimeError('Wrong encoder parameter count')
    if args.smoke:
        # Illustration-only namespace, disjoint from recorded training/test IDs.
        eps=[generator.annotated_episode(99010001,'train')]
        full,low=objective.tr.pack(eps,'cpu')
        with torch.no_grad():loss,stats=objective.objective(net,full,low,eps)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite synthetic objective')
        print(json.dumps({'passed':True,'parameters':count,'synthetic_forward_only':True,
            'optimizer_steps':0,'real_images_read':0,'benchmark_run':False,
            'identity':99010001,'objective_is_finite':True},indent=2))
        return
    if args.output is None:raise ValueError('--train requires --output for a new experiment directory')
    output=args.output.resolve()
    frozen=[workspace/'outputs/hypernet_v46/spectrogram',workspace/'outputs/hypernet_v45/multiseed',
        workspace/'outputs/hypernet_locked',workspace/'outputs/cloud_plan_20261105/gs_fp32',
        workspace/'outputs/oncenet_confirmation_20261105']
    if any(output==p or output.is_relative_to(p) for p in frozen):
        raise ValueError('New training output must be outside frozen experiment directories')
    if output.exists():raise ValueError('Output already exists; no overwrite/resume is performed')
    output.mkdir(parents=True)
    order=np.random.default_rng(args.seed+7919).permutation(3200)+OFFSETS[SEEDS.index(args.seed)]
    config={'purpose':'new reproduction run, not replacement of frozen evidence','seed':args.seed,
        'parameters':count,'steps':800,'batch':4,'device':device,'no_borrowed_weights':True,
        'real_encoder_training_images':0,'initial_state':initial,
        'source_bindings_sha256':sha(HERE/'gravity_training_bindings.json'),
        'launcher_sha256':sha(__file__),'order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
        'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.0001,'clip':1.},
        'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__},
        'limitations':['No bitwise-equivalence claim across hardware/library versions.',
            'No synthetic benchmark or real confirmation is run by this tool.',
            'The final encoder alone is not the fitted one-reference matcher/calibration.']}
    write(output/'reproduction_config.json',config)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    history=[];start=time.time()
    for step in range(800):
        ids=order[step*4:(step+1)*4]
        eps=[generator.annotated_episode(int(i),'train') for i in ids]
        full,low=objective.tr.pack(eps,device)
        opt.zero_grad(set_to_none=True)
        loss,stats=objective.objective(net,full,low,eps)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss at step '+str(step+1))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        opt.step();history.append({'step':step+1,'identities':ids.tolist(),**stats,'gradient_norm_before_clip':float(norm)})
        if step%20==0 or step==799:
            write(output/'history.json',history)
            write(output/'status.json',{'stage':'training','step':step+1,'total':800,'elapsed_seconds':time.time()-start})
            print(args.seed,step+1,stats['loss'],flush=True)
        if (step+1)%100==0:
            objective.atomic_torch(output/'resume.pt',{'model':objective.tr.cpu_state(net),
                'optimizer':opt.state_dict(),'step':step+1,'history':history,
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None,
                'config_sha256':sha(output/'reproduction_config.json')})
    verify(workspace,args.seed)
    objective.atomic_torch(output/'spectrogram.pt',objective.tr.cpu_state(net))
    write(output/'status.json',{'stage':'completed','step':800,'total':800,
        'checkpoint_sha256':sha(output/'spectrogram.pt'),'elapsed_seconds':time.time()-start,
        'new_evaluation_performed':False})
    print('Completed new training reproduction:',str(output),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--train',action='store_true');mode.add_argument('--smoke',action='store_true')
    p.add_argument('--workspace',type=Path,default=workspace_default())
    p.add_argument('--seed',type=int,choices=SEEDS,default=45029)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not(args.train or args.smoke):p.print_help();return
    run(args)

if __name__=='__main__':main()
