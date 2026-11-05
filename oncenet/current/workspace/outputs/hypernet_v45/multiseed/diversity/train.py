"""Separately declared global-diversity rescue; original experiment immutable."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import sys,importlib.util,argparse,time,random,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;M=D.parent;sys.path.insert(0,str(M))
import numpy as np,torch
from torch.nn import functional as F
from model import Encoder,load_encoder,dual,tensor,sha,save,atomic_torch
_s=importlib.util.spec_from_file_location('v45_original_frozen_trainer',M/'train.py');tr=importlib.util.module_from_spec(_s);_s.loader.exec_module(tr)
from generator import annotated_episode,episode,VIEWS,NEGATIVE_TYPES
SEEDS=tr.SEEDS;STEPS=tr.STEPS;BATCH=tr.BATCH;ARM='global_diversity'
def additions(z,n):
 chunks=z.split(n);ref,pos=chunks[:2];edited=chunks[2:];logits=[]
 for i in range(n):
  candidates=torch.stack([pos[i],*[q[i] for q in edited],*[pos[j] for j in range(n) if j!=i]])
  logits.append(candidates@ref[i]/.1)
 logits=torch.stack(logits);contrast=F.cross_entropy(logits,torch.zeros(n,dtype=torch.long,device=z.device))
 centroids=F.normalize(ref+pos,dim=1);g=centroids*(z.shape[1]**.5);floor=F.relu(.5-(g.var(0,unbiased=False)+1e-4).sqrt()).mean()
 return contrast,floor,logits
def objective(net,full,low,eps):
 d=net(torch.cat([full,low]));z=dual(d,'base');chunks=z.split(len(eps));sp=(chunks[0]*chunks[1]).sum(1);sn=torch.stack([(chunks[0]*a).sum(1) for a in chunks[2:]],1);rank=F.relu(.04+sn.max(1).values-sp).mean();attraction=(1-sp).mean();local,consistency,local_floor=tr.local_loss(d,eps);contrast,global_floor,logits=additions(z,len(eps));loss=3*rank+.25*attraction+.5*local+contrast+global_floor
 return loss,dict(loss=float(loss.detach()),rank=float(rank.detach()),attraction=float(attraction.detach()),local_cosine=float(consistency.detach()),local_variance_floor=float(local_floor.detach()),global_infonce=float(contrast.detach()),global_variance_floor=float(global_floor.detach()),positive=float(sp.mean().detach()),hard_negative=float(sn.max(1).values.mean().detach()),logit_candidates=int(logits.shape[1]))
def files():return [Path(__file__),D/'checks.py',D/'intent.json',D/'intent_superseded.json',M/'protocol.json',M/'namespace_audit.py',M/'namespace_audit.json']
def assert_protocol():
 tr.assert_protocol();p=json.loads((D/'protocol.json').read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 assert sha(D/'checks.json')==p['checks_sha256'];assert json.loads((D/'checks.json').read_text())['passed'];return p,sha(D/'protocol.json')
def freeze():
 assert not(D/'protocol.json').exists();tr.assert_protocol();intent=json.loads((D/'intent.json').read_text());checks=json.loads((D/'checks.json').read_text());assert checks['passed'];assert checks['files']=={str(p):sha(p) for p in files()}
 if (M/'synthetic_started.json').exists():assert intent['declared_epoch']<json.loads((M/'synthetic_started.json').read_text())['time']
 assert not(M.parent/'seed_development/results.json').exists(),'No new formula freeze after development results'
 save(D/'protocol.json',dict(frozen_epoch=time.time(),intent_sha256=sha(D/'intent.json'),original_protocol_sha256=sha(M/'protocol.json'),files={str(p):sha(p) for p in files()},checks_sha256=sha(D/'checks.json'),seeds=list(SEEDS),steps=STEPS,batch=BATCH,arm=ARM,parameters=42480,readouts=['base','spatial'],primary_readout='base',formula=intent,original_experiment_unchanged=True));print('DIVERSITY FROZEN',sha(D/'protocol.json'),flush=True)
def registry():
 _,h=assert_protocol();rows=[]
 for seed in SEEDS:
  path=D/f'seed_{seed}/{ARM}.pt'
  if not path.exists():return False
  status=json.loads((path.parent/'status.json').read_text());assert status['stage']=='completed' and status['step']==STEPS and status['protocol_sha256']==h
  rows.append(dict(id=f's{seed}_{ARM}',seed=seed,arm=ARM,path=str(path),sha256=sha(path),parameters=42480))
 save(D/'checkpoints.json',dict(training_complete=True,protocol_sha256=h,original_protocol_sha256=sha(M/'protocol.json'),models=rows));return True
def run(seed,resume=False,device='cuda'):
 _,h=assert_protocol();assert seed in SEEDS;assert (M/'synthetic_completed.json').exists(),'Original9 must finish their single test first';assert not(D/'synthetic_started.json').exists();folder=D/f'seed_{seed}';folder.mkdir(exist_ok=True);assert not(folder/'completed.json').exists(),'Do not repeat a completed seed';torch.set_num_threads(2)
 if device=='cuda':assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.45)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.backends.cudnn.benchmark=False;net=load_encoder(M/f'seed_{seed}/random.pt',device).train();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);first=0;history=[]
 if resume:
  ck=torch.load(folder/'resume.pt',map_location=device,weights_only=False);assert ck['protocol_sha256']==h;net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['step'];history=ck['history'];random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu())
  if device=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
 else:assert not(folder/'status.json').exists(),'Existing run; use --resume'
 order=tr.order_for(seed);start=time.time()
 for step in range(first,STEPS):
  ids=order[step*BATCH:(step+1)*BATCH];eps=[annotated_episode(int(i),'train') for i in ids];full,low=tr.pack(eps,device);opt.zero_grad(set_to_none=True);loss,stats=objective(net,full,low,eps);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();row=dict(step=step+1,identities=ids.tolist(),**stats,gradient_norm_before_clip=float(gn));history.append(row)
  if step%20==0 or step+1==STEPS:
   save(folder/'status.json',dict(stage='training',step=step+1,total=STEPS,seed=seed,protocol_sha256=h,pid=os.getpid(),elapsed_seconds=time.time()-start,last=row));save(folder/'history.json',history);print(seed,step+1,stats['loss'],stats['global_variance_floor'],flush=True)
  if (step+1)%100==0:atomic_torch(folder/'resume.pt',dict(protocol_sha256=h,model=tr.cpu_state(net),optimizer=opt.state_dict(),step=step+1,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None))
 assert_protocol();atomic_torch(folder/f'{ARM}.pt',tr.cpu_state(net));result=dict(stage='completed',step=STEPS,total=STEPS,seed=seed,protocol_sha256=h,checkpoint_sha256=sha(folder/f'{ARM}.pt'),random_initialization_sha256=sha(M/f'seed_{seed}/random.pt'),elapsed_seconds=time.time()-start,no_checkpoint_selection=True);save(folder/'completed.json',result);save(folder/'status.json',result);registry();print('DIVERSITY SEED COMPLETE',seed,flush=True)
def synthetic(device='cuda'):
 _,h=assert_protocol();assert registry();assert not(D/'synthetic_started.json').exists(),'No unreviewed test rerun';old=sha(M/'synthetic_results.json');reg=json.loads((D/'checkpoints.json').read_text());save(D/'synthetic_started.json',dict(protocol_sha256=h,registry_sha256=sha(D/'checkpoints.json'),original_results_sha256=old,time=time.time(),pid=os.getpid()));torch.set_num_threads(2);models={r['id']:load_encoder(r['path'],device) for r in reg['models']};margins={name:{mode:{k:[] for k in NEGATIVE_TYPES} for mode in tr.MODES} for name in models};latent=[]
 with torch.inference_mode():
  for start in range(0,tr.TEST_N,BATCH):
   eps=[episode(tr.TEST_OFFSET+i,'test') for i in range(start,min(start+BATCH,tr.TEST_N))];latent.extend([e['metadata']['latent_seed'] for e in eps]);full,low=tr.pack(eps,device)
   for name,net in models.items():
    d=net(torch.cat([full,low]))
    for mode in tr.MODES:
     chunks=dual(d,mode).split(len(eps));sp=(chunks[0]*chunks[1]).sum(1)
     for kind,q in zip(NEGATIVE_TYPES,chunks[2:]):margins[name][mode][kind].extend((sp-(chunks[0]*q).sum(1)).cpu().tolist())
 result=dict(protocol_sha256=h,registry_sha256=sha(D/'checkpoints.json'),original_results_sha256=old,original_models_rerun=False,identities=tr.TEST_N,paired_cases=tr.TEST_N*4,identity_offset=tr.TEST_OFFSET,latent_seed_sha256=hashlib.sha256(json.dumps(latent).encode()).hexdigest(),no_checkpoint_selection=True,models={name:{mode:dict(overall_accuracy=float(np.mean(np.array(list(edits.values()))>0)),all_four_correct=float(np.mean(np.all(np.array(list(edits.values()))>0,axis=0))),edits={k:dict(accuracy=float(np.mean(np.array(v)>0)),mean_margin=float(np.mean(v)),margins=v) for k,v in edits.items()}) for mode,edits in modes.items()} for name,modes in margins.items()})
 assert sha(M/'synthetic_results.json')==old;assert_protocol();save(D/'synthetic_results.json',result);save(D/'synthetic_completed.json',dict(passed=True,protocol_sha256=h,registry_sha256=sha(D/'checkpoints.json'),synthetic_results_sha256=sha(D/'synthetic_results.json'),original_results_sha256=old,models=3,identities=256,original_models_rerun=False));print('DIVERSITY SYNTHETIC COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',type=int,choices=SEEDS);p.add_argument('--resume',action='store_true');p.add_argument('--synthetic',action='store_true');p.add_argument('--device',default='cuda',choices=['cuda','cpu']);a=p.parse_args();assert sum([a.freeze,a.run is not None,a.synthetic])==1
 if a.freeze:freeze()
 elif a.run is not None:run(a.run,a.resume,a.device)
 else:synthetic(a.device)
