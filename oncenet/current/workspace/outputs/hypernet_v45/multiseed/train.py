"""Bounded, paired from-scratch seed pilot. Root alone launches GPU training.

prepare -> check -> freeze -> run (three seeds) -> synthetic (once).
No checkpoint/readout selection and no real-image encoder training.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import argparse,random,time,copy,hashlib,json
import numpy as np,torch
from torch.nn import functional as F
from model import *
from generator import annotated_episode,episode,VIEWS,NEGATIVE_TYPES,BASE
SEEDS=(45011,45029,45047);STEPS=800;BATCH=4;TRAIN_OFFSETS=(4500000,4510000,4520000);TEST_OFFSET=4590000;TEST_N=256
ARMS=('random','baseline','improved');MODES=('base','spatial')
def pack(eps,device):return tensor([e[k] for k in VIEWS for e in eps],device),tensor([e[k+'_low'] for k in VIEWS for e in eps],device)
def local_loss(d,eps):
 n=len(eps);vectors=[]
 # The input stride is 4 and padded stride-two convolutions center output j at 4*j.
 for which,key in [(0,'reference'),(1,'positive')]:
  xy=torch.as_tensor(np.stack([e['local'][key] for e in eps]),device=d.device,dtype=d.dtype);grid=2*(xy/4+.5)/32-1
  v=F.grid_sample(d[which*n:(which+1)*n],grid.unsqueeze(2),mode='bilinear',padding_mode='zeros',align_corners=False).squeeze(3).permute(0,2,1).reshape(-1,48)
  vectors.append(F.normalize(v,dim=1,eps=1e-8))
 a,b=vectors;consistency=(1-(a*b).sum(1)).mean();floor=.5*sum(F.relu(.05-(v.var(0,unbiased=False)+1e-4).sqrt()).mean()/.05 for v in vectors)
 return consistency+.1*floor,consistency,floor
def objective(net,full,low,eps,arm):
 d=net(torch.cat([full,low]));chunks=dual(d,'base').split(len(eps));sp=(chunks[0]*chunks[1]).sum(1);sn=torch.stack([(chunks[0]*z).sum(1) for z in chunks[2:]],1);rank=F.relu(.04+sn.max(1).values-sp).mean();attraction=(1-sp).mean();loss=3*rank+.25*attraction
 local,consistency,floor=local_loss(d,eps)
 if arm=='improved':loss=loss+.5*local
 return loss,dict(loss=float(loss.detach()),rank=float(rank.detach()),attraction=float(attraction.detach()),local_cosine=float(consistency.detach()),variance_floor=float(floor.detach()),positive=float(sp.mean().detach()),hard_negative=float(sn.max(1).values.mean().detach()))
def order_for(seed):return np.random.default_rng(seed+7919).permutation(STEPS*BATCH)+TRAIN_OFFSETS[SEEDS.index(seed)]
def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def source_files():return [R/'model.py',R/'generator.py',R/'train.py',R/'checks.py',R/'plan.json',BASE,BASE.parent.parent/'generator.py']
def assert_protocol():
 p=R/'protocol.json';protocol=json.loads(p.read_text())
 for path,h in protocol['source_hashes'].items():assert sha(path)==h,('Frozen source changed',path)
 for row in protocol['initial_controls']:assert sha(row['path'])==row['sha256']
 assert sha(R/'checks.json')==protocol['checks_sha256'];assert json.loads((R/'checks.json').read_text())['passed']
 return protocol,sha(p)
def prepare():
 assert not (R/'protocol.json').exists(),'Already frozen'
 plan=dict(experiment='v45 bounded paired from-scratch seed pilot',seeds=list(SEEDS),steps_per_arm=STEPS,batch=BATCH,arms=list(ARMS),trained_arms=['baseline','improved'],parameters=42480,trainable_parameters=42480,optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,gradient_clip=1.,schedule=None),borrowed_weights=False,warm_start=False,teacher=False,real_encoder_training_images=0,
 initialization='Three independent PyTorch random seeds. Within each seed, both trained arms start with exactly its frozen random state. This does not replicate the historical v33/v37 pretraining budget.',paired_data='Identical 3200 identities/order/images for both arms within each seed; distinct disjoint identity ranges across seeds; unchanged corrected-v44 image renderer.',train_identity_ranges={str(s):[o,o+STEPS*BATCH-1] for s,o in zip(SEEDS,TRAIN_OFFSETS)},order_sha256={str(s):hashlib.sha256(order_for(s).astype('<i8').tobytes()).hexdigest() for s in SEEDS},
 baseline_loss='3*mean(relu(.04+max_four_negative_cosine-positive_cosine)) + .25*mean(1-positive_cosine); normalized dual base descriptor',improved_addition='0.5*[mean(1-cosine(aligned foreground vectors)) + 0.1*variance_floor]. Unit-normalized 48D local vectors; variance_floor=mean(relu(.05-sqrt(var+1e-4)))/.05 averaged over both views. 32 points per identity; only full-image reference/positive dense features. No local negatives.',coefficient_choice='Fixed engineering constants before training or new accuracy inspection; no coefficient sweep.',
 readouts=dict(primary='base',secondary='spatial',base_dual_dimension=4032,spatial_dual_dimension=6080,selection=False),synthetic=dict(identities=TEST_N,offset=TEST_OFFSET,split='test',shared_all_nine_checkpoints=True,edits=list(NEGATIVE_TYPES),evaluate_once_after_all_training=True,select_checkpoints=False),real_evaluation='Existing USED development records only, one fixed positive reference/class, separate negative FIT/CAL. No new real holdout; retrospective development, not external confirmation.',
 diagnosis=['Global pooled graph-identity ranking can meet synthetic invariances without stable local correspondences.', 'Earlier local InfoNCE treats other sampled positions as negatives, including potentially equivalent repeated primitives. This is a code-level conflict hypothesis, not a demonstrated causal explanation.', 'V44 teacher anchoring, partial freezing and own-v37 warm start confound whether its objective transfers from independent random initialization. This pilot removes them identically in both arms.', 'V44 development evidence and previously consumed test outcomes are known. Only the new synthetic latent identities are untouched; no claim of fresh real confirmation.'],
 sources=[dict(title='VICReg, ICLR 2022',url='https://arxiv.org/abs/2105.04906'),dict(title='VICRegL, NeurIPS 2022',url='https://arxiv.org/abs/2210.01571')],research_claim='Noncontrastive local invariance and variance regularization are established ideas. This is an adaptation without covariance penalty/projector, not a reproduction, new theorem, or guarantee of semantic transfer.',limitations=['256 test identities share the training grammar/renderer; four edits per identity are dependent.','Three seeds are a bounded seed check, not a large-run generalization guarantee.','Image identity is narrower than a real semantic category; preserved count/orientation need not be true category invariances.','Variance floor is a soft penalty, not a proof against collapse.','Reference/query raster support accounts for stroke discretization; analytic center geometry correspondence is approximate at raster edges.'])
 save(R/'plan.json',plan)
 for seed in SEEDS:
  folder=R/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);path=folder/'random.pt';torch.manual_seed(seed);net=Encoder()
  if path.exists():
   old=torch.load(path,weights_only=True,map_location='cpu');assert all(torch.equal(old[k],v) for k,v in net.state_dict().items())
  else:atomic_torch(path,cpu_state(net))
 save(R/'initialization.json',dict(models=[dict(id=f's{seed}_random',seed=seed,arm='random',path=str(R/f'seed_{seed}/random.pt'),sha256=sha(R/f'seed_{seed}/random.pt'),parameters=42480) for seed in SEEDS]))
 print('Prepared random controls and fixed plan; run checks.py then train.py --freeze',flush=True)
def freeze():
 assert not (R/'protocol.json').exists(),'Already frozen';checks=json.loads((R/'checks.json').read_text());assert checks['passed'];assert checks['source_hashes']=={str(p):sha(p) for p in source_files()};initial=json.loads((R/'initialization.json').read_text())['models'];save(R/'protocol.json',dict(plan=json.loads((R/'plan.json').read_text()),source_hashes={str(p):sha(p) for p in source_files()},checks_sha256=sha(R/'checks.json'),initial_controls=initial));print('Protocol frozen',sha(R/'protocol.json'),flush=True)
def registry():
 protocol,h=assert_protocol();rows=[]
 for seed in SEEDS:
  for arm in ARMS:
   path=R/f'seed_{seed}/{arm}.pt'
   if not path.exists():return False
   rows.append(dict(id=f's{seed}_{arm}',seed=seed,arm=arm,path=str(path),sha256=sha(path),parameters=42480))
 for seed in SEEDS:
  status=json.loads((R/f'seed_{seed}/status.json').read_text());assert status['stage']=='completed' and status['step']==STEPS and status['protocol_sha256']==h
 save(R/'checkpoints.json',dict(training_complete=True,protocol_sha256=h,models=rows));return True
def run(seed,resume=False,device='cuda'):
 protocol,h=assert_protocol();assert seed in SEEDS;torch.set_num_threads(2);folder=R/f'seed_{seed}';assert not (folder/'completed.json').exists(),'This seed is complete; do not rerun';assert not (R/'synthetic_started.json').exists(),'Synthetic test already entered'
 if device=='cuda':assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.45)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.backends.cudnn.benchmark=False
 models={arm:load_encoder(folder/'random.pt',device).train() for arm in ['baseline','improved']};opts={arm:torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4) for arm,net in models.items()};history=[];first=0
 if resume:
  ck=torch.load(folder/'resume.pt',map_location=device,weights_only=False);assert ck['protocol_sha256']==h;first=ck['step'];history=ck['history']
  for arm in models:models[arm].load_state_dict(ck['models'][arm]);opts[arm].load_state_dict(ck['optimizers'][arm])
  random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu())
  if device=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
 else:assert not (folder/'status.json').exists(),'Existing run; use --resume'
 order=order_for(seed);start=time.time()
 for step in range(first,STEPS):
  ids=order[step*BATCH:(step+1)*BATCH];eps=[annotated_episode(int(i),'train') for i in ids];full,low=pack(eps,device);row=dict(step=step+1,identities=ids.tolist())
  for arm,net in models.items():
   opts[arm].zero_grad(set_to_none=True);loss,stats=objective(net,full,low,eps,arm);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opts[arm].step();row[arm]=stats|dict(gradient_norm_before_clip=float(gn))
  history.append(row)
  if step%20==0 or step+1==STEPS:
   save(folder/'status.json',dict(stage='training',seed=seed,step=step+1,total=STEPS,elapsed_seconds=time.time()-start,pid=os.getpid(),protocol_sha256=h,last=row));save(folder/'history.json',history);print(seed,step+1,row['baseline']['loss'],row['improved']['loss'],flush=True)
  if (step+1)%100==0:
   atomic_torch(folder/'resume.pt',dict(protocol_sha256=h,step=step+1,history=history,models={a:cpu_state(m) for a,m in models.items()},optimizers={a:o.state_dict() for a,o in opts.items()},python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None))
 assert_protocol()
 for arm,net in models.items():atomic_torch(folder/f'{arm}.pt',cpu_state(net))
 result=dict(stage='completed',seed=seed,step=STEPS,total=STEPS,protocol_sha256=h,elapsed_seconds=time.time()-start,weights={a:sha(folder/f'{a}.pt') for a in ARMS},no_checkpoint_selection=True)
 save(folder/'completed.json',result);save(folder/'status.json',result);registry();print('SEED COMPLETE',seed,flush=True)
def synthetic(device='cuda'):
 _,h=assert_protocol();assert registry();assert not (R/'synthetic_started.json').exists(),'Test already entered; no unreviewed rerun'
 reg=json.loads((R/'checkpoints.json').read_text());save(R/'synthetic_started.json',dict(protocol_sha256=h,registry_sha256=sha(R/'checkpoints.json'),pid=os.getpid(),time=time.time()));torch.set_num_threads(2)
 models={row['id']:load_encoder(row['path'],device) for row in reg['models']};margins={name:{mode:{k:[] for k in NEGATIVE_TYPES} for mode in MODES} for name in models};latent=[]
 with torch.inference_mode():
  for start in range(0,TEST_N,BATCH):
   eps=[episode(TEST_OFFSET+i,'test') for i in range(start,min(start+BATCH,TEST_N))];latent.extend([e['metadata']['latent_seed'] for e in eps]);full,low=pack(eps,device)
   for name,net in models.items():
    d=net(torch.cat([full,low]))
    for mode in MODES:
     chunks=dual(d,mode).split(len(eps));sp=(chunks[0]*chunks[1]).sum(1)
     for kind,z in zip(NEGATIVE_TYPES,chunks[2:]):margins[name][mode][kind].extend((sp-(chunks[0]*z).sum(1)).cpu().tolist())
 result=dict(protocol_sha256=h,registry_sha256=sha(R/'checkpoints.json'),identities=TEST_N,paired_cases=TEST_N*4,identity_offset=TEST_OFFSET,latent_seed_sha256=hashlib.sha256(json.dumps(latent).encode()).hexdigest(),no_checkpoint_selection=True,limitations='New identities with shared renderer/grammar; dependent four edits per identity; not unseen semantic families.',models={name:{mode:dict(overall_accuracy=float(np.mean(np.array(list(edits.values()))>0)),all_four_correct=float(np.mean(np.all(np.array(list(edits.values()))>0,axis=0))),edits={k:dict(accuracy=float(np.mean(np.array(v)>0)),mean_margin=float(np.mean(v)),margins=v) for k,v in edits.items()}) for mode,edits in modes.items()} for name,modes in margins.items()})
 assert_protocol();save(R/'synthetic_results.json',result);save(R/'synthetic_completed.json',dict(passed=True,protocol_sha256=h,registry_sha256=sha(R/'checkpoints.json'),synthetic_results_sha256=sha(R/'synthetic_results.json'),models=len(models),identities=TEST_N));print('SYNTHETIC TEST COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--run',type=int,choices=SEEDS);p.add_argument('--resume',action='store_true');p.add_argument('--synthetic',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();assert sum([a.prepare,a.freeze,a.run is not None,a.synthetic])==1
 if a.prepare:prepare()
 elif a.freeze:freeze()
 elif a.run is not None:run(a.run,a.resume,a.device)
 else:synthetic(a.device)
