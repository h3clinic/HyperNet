"""Generator-only ablation using the frozen v45 global-diversity objective."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,importlib.util,argparse,time,random,json,hashlib
R=Path(__file__).resolve().parent;V45=R.parents[1]/'hypernet_v45';M=V45/'multiseed';D=M/'diversity'
sys.path.insert(0,str(M))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
dv=module('v46_frozen_v45_objective',D/'train.py')
gen=module('v46_spectrogram_generator',R/'generator.py')
import numpy as np,torch
from model import load_encoder,dual,sha,save,atomic_torch
SEEDS=(45011,45029,45047);STEPS=800;BATCH=4;ARM='spectrogram';OFFSETS=(4600000,4610000,4620000);TEST_OFFSET=4690000;RETENTION_OFFSET=4691000;TEST_N=256
def order_for(seed):return np.random.default_rng(seed+7919).permutation(3200)+OFFSETS[SEEDS.index(seed)]
def sources():return [Path(__file__),R/'generator.py',R/'checks.py',R/'plan.json',R/'research.json',M/'protocol.json',D/'protocol.json']
def assert_protocol():
 dv.assert_protocol();p=json.loads((R/'protocol.json').read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 assert sha(R/'checks.json')==p['checks_sha256']and json.loads((R/'checks.json').read_text())['passed'];return p,sha(R/'protocol.json')
def prepare():
 assert not(R/'protocol.json').exists();dv.assert_protocol()
 save(R/'research.json',dict(sources=[
  dict(title='Detection of gravitational-wave bursts with chirplet-like template families',authors='Chassande-Mottin et al.',year=2010,url='https://arxiv.org/abs/1005.2876',mechanism='Gaussian-windowed sinusoids and chirps motivate finite time-frequency ridges.',implementation='Approximate Gaussian energy ridges, not the paper chirplet search or waveform bank.'),
  dict(title="Reducing Scattered Light in LIGO's Third Observing Run",authors='Soni et al.',year=2020,url='https://arxiv.org/abs/2007.14876',mechanism='Scattered light associated with moving surfaces gives arches; instantaneous frequency scales with surface speed.',implementation='Sinusoidal absolute-frequency arches; no real auxiliary channels or empirical physical parameters fitted.'),
  dict(title='ArchGEM: an Advanced Data Analysis Tool for Analyzing Scattered Light Noise in LIGO',authors='McGowan et al.',year=2026,url='https://arxiv.org/abs/2604.23416',mechanism='Phase-derived Doppler frequency and arch geometry; f_peak=2*A*omega/lambda.',implementation='Motivates arch parameterization only. No GMM, paper code, labels, or pretrained model used.'),
  dict(title='Gravity Spy: lessons learned and a path forward',year=2024,url='https://link.springer.com/article/10.1140/epjp/s13360-023-04795-4',mechanism='Four temporal windows expose different glitch timescales.',implementation='Four-window input is a separate root-owned ablation. This generator training keeps the original one-second full plus low-frequency crop.')],
  mathematical_model='P(t,f)=sum_j A_j e_j(t) exp(-(log(f)-log(f_j(t)))^2/(2*sigma_j^2)); localized bursts use a2D Gaussian. Chirps use f0*(1+k*u+q*u^2), arches fpeak*abs(cos(pi*(t-t0)/period+phase)); colored positive image-space noise has a power-law FFT envelope. Foreground RMS is equalized then shared gain/log1p compression applied.',
  limits=['Not a physical detector or exact Q-transform simulator; correlated noise is an image-space approximation.','Prior development failures motivated the domain-specific families and no claim of untouched domain choice is made.','The papers motivate mechanisms; the parameter distributions and implementation are engineering choices, not published performance guarantees.']))
 plan=dict(experiment='v46 generator-only mathematical spectrogram ablation',seeds=list(SEEDS),parameters=42480,steps=STEPS,batch=BATCH,arm=ARM,initialization='The exact saved v45 random checkpoint for each seed; no trained or external borrowed weights',borrowed_neural_weights=False,real_encoder_training_images=0,trained_changes='Generator only. Frozen v45 global-diversity architecture, base/local/InfoNCE/variance objectives, optimizer and clipping are called unchanged.',optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,clip=1.),
  original_objective_protocol_sha256=sha(D/'protocol.json'),original_model_protocol_sha256=sha(M/'protocol.json'),initial_controls=[dict(seed=s,path=str(M/f'seed_{s}/random.pt'),sha256=sha(M/f'seed_{s}/random.pt'))for s in SEEDS],
  train_ranges={str(s):[o,o+3199]for s,o in zip(SEEDS,OFFSETS)},order_sha256={str(s):hashlib.sha256(order_for(s).astype('<i8').tobytes()).hexdigest()for s in SEEDS},relative_order_same_as_v45=True,
  families=list(gen.FAMILIES),positive_invariances='Same components/count/relative geometry under mild axis-preserving affine pose, linewidth/envelope/gain/colored-noise changes; component frequency geometry unchanged by appearance.',negative_edits=list(gen.NEGATIVE_TYPES),negative_nuisance_control='Positive and all edited queries share background/pose/gain/width; each foreground RMS normalized before shared intensity transform.',
  input='Native256x2561-second time/log-frequency10–2048Hz, downsample128x128full and nativebottomthird enlarged128x128low; RGB channels identical. No axes, timestamps, text or borders.',
  readouts=['base','spatial'],primary_readout='base',synthetic_sets=dict(spectrogram=dict(split='test',offset=TEST_OFFSET,identities=TEST_N,generator_namespace='v46-spectral-power'),retention=dict(split='test',offset=RETENTION_OFFSET,identities=TEST_N,generator_namespace='v44-part-graphs')),synthetic_models='All9 matched random3/v45global3/v46spectrogram3 evaluated once on each new set after training; no test-driven selection.',
  real_scope='Root evaluates only previously consumed USED data with fixed one positive reference and negative FIT/CAL. Separate one-second/four-duration ablation. No new real holdout or claims of confirmed discoveries.',
  limitations=['Domain-motivated power-ridge simulator rather than universal image generator.','New latent test identities share their respective grammars and renderer.','Four edits per identity are dependent and three seeds are not additional test images.','Soft variance regularization does not mathematically remove the exact constant stationary point.','Changes in generator distribution include primitive shapes and nuisance statistics; this ablation cannot separate those internally.'])
 save(R/'plan.json',plan);print('Prepared plan',flush=True)
def freeze():
 assert not(R/'protocol.json').exists();dv.assert_protocol();c=json.loads((R/'checks.json').read_text());assert c['passed'];assert c['files']=={str(p):sha(p)for p in sources()};save(R/'protocol.json',dict(plan=json.loads((R/'plan.json').read_text()),files={str(p):sha(p)for p in sources()},checks_sha256=sha(R/'checks.json'),frozen_epoch=time.time()));print('V46 FROZEN',sha(R/'protocol.json'),flush=True)
def registry():
 _,h=assert_protocol();rows=[]
 for s in SEEDS:
  p=R/f'seed_{s}/{ARM}.pt'
  if not p.exists():return False
  done=json.loads((p.parent/'completed.json').read_text());assert done['stage']=='completed'and done['step']==STEPS and done['protocol_sha256']==h
  rows.append(dict(id=f's{s}_{ARM}',seed=s,arm=ARM,path=str(p),sha256=sha(p),parameters=42480))
 save(R/'checkpoints.json',dict(training_complete=True,protocol_sha256=h,models=rows));return True
def run(seed,resume=False,device='cuda'):
 _,h=assert_protocol();assert seed in SEEDS;folder=R/f'seed_{seed}';folder.mkdir(exist_ok=True);assert not(folder/'completed.json').exists();assert not(R/'synthetic_started.json').exists();torch.set_num_threads(2)
 if device=='cuda':assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.45)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.backends.cudnn.benchmark=False;net=load_encoder(M/f'seed_{seed}/random.pt',device).train();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);first=0;history=[]
 if resume:
  ck=torch.load(folder/'resume.pt',map_location=device,weights_only=False);assert ck['protocol_sha256']==h;net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['step'];history=ck['history'];random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu())
  if device=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
 else:assert not(folder/'status.json').exists(),'Use --resume for existing run'
 order=order_for(seed);start=time.time()
 for step in range(first,STEPS):
  ids=order[step*BATCH:(step+1)*BATCH];eps=[gen.annotated_episode(int(i),'train')for i in ids];full,low=dv.tr.pack(eps,device);opt.zero_grad(set_to_none=True);loss,stats=dv.objective(net,full,low,eps);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();row=dict(step=step+1,identities=ids.tolist(),**stats,gradient_norm_before_clip=float(gn));history.append(row)
  if step%20==0 or step+1==STEPS:
   save(folder/'status.json',dict(stage='training',seed=seed,step=step+1,total=STEPS,protocol_sha256=h,pid=os.getpid(),elapsed_seconds=time.time()-start,last=row));save(folder/'history.json',history);print(seed,step+1,stats['loss'],flush=True)
  if(step+1)%100==0:atomic_torch(folder/'resume.pt',dict(protocol_sha256=h,model=dv.tr.cpu_state(net),optimizer=opt.state_dict(),step=step+1,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()if device=='cuda'else None))
 assert_protocol();atomic_torch(folder/f'{ARM}.pt',dv.tr.cpu_state(net));done=dict(stage='completed',step=STEPS,total=STEPS,seed=seed,protocol_sha256=h,checkpoint_sha256=sha(folder/f'{ARM}.pt'),random_initialization_sha256=sha(M/f'seed_{seed}/random.pt'),elapsed_seconds=time.time()-start,no_checkpoint_selection=True);save(folder/'completed.json',done);save(folder/'status.json',done);registry();print('V46 SEED COMPLETE',seed,flush=True)
def synthetic(device='cuda'):
 _,h=assert_protocol();assert registry();assert not(R/'synthetic_started.json').exists(),'No unreviewed test rerun';new=json.loads((R/'checkpoints.json').read_text())['models'];old=json.loads((D/'checkpoints.json').read_text())['models'];randoms=[r for r in json.loads((M/'checkpoints.json').read_text())['models']if r['arm']=='random'];entries=randoms+old+new
 for r in entries:assert sha(r['path'])==r['sha256']
 bindings=dict(protocol_sha256=h,models=entries,registry_sha256=sha(R/'checkpoints.json'),time=time.time(),pid=os.getpid());save(R/'synthetic_started.json',bindings);torch.set_num_threads(2);models={r['id']:load_encoder(r['path'],device)for r in entries};sets={}
 for domain,offset,renderer in [('spectrogram',TEST_OFFSET,gen.episode),('retention',RETENTION_OFFSET,dv.episode)]:
  margins={name:{mode:{k:[]for k in gen.NEGATIVE_TYPES}for mode in ['base','spatial']}for name in models};latent=[]
  with torch.inference_mode():
   for start in range(0,TEST_N,BATCH):
    eps=[renderer(offset+i,'test')for i in range(start,min(start+BATCH,TEST_N))];latent.extend([e['metadata']['latent_seed']for e in eps]);full,low=dv.tr.pack(eps,device)
    for name,net in models.items():
     dense=net(torch.cat([full,low]))
     for mode in ['base','spatial']:
      chunks=dual(dense,mode).split(len(eps));sp=(chunks[0]*chunks[1]).sum(1)
      for kind,q in zip(gen.NEGATIVE_TYPES,chunks[2:]):margins[name][mode][kind].extend((sp-(chunks[0]*q).sum(1)).cpu().tolist())
  sets[domain]=dict(identities=TEST_N,paired_cases=TEST_N*4,identity_offset=offset,latent_seed_sha256=hashlib.sha256(json.dumps(latent).encode()).hexdigest(),models={name:{mode:dict(overall_accuracy=float(np.mean(np.array(list(edits.values()))>0)),all_four_correct=float(np.mean(np.all(np.array(list(edits.values()))>0,axis=0))),edits={k:dict(accuracy=float(np.mean(np.array(v)>0)),mean_margin=float(np.mean(v)),margins=v)for k,v in edits.items()})for mode,edits in modes.items()}for name,modes in margins.items()})
 assert_protocol();save(R/'synthetic_results.json',dict(protocol_sha256=h,registry_sha256=sha(R/'checkpoints.json'),models=entries,sets=sets,no_checkpoint_selection=True,limits='Two new latent sets, shared9 frozen checkpoints; no semantic-family holdout, no real-image inference.'));save(R/'synthetic_completed.json',dict(passed=True,protocol_sha256=h,registry_sha256=sha(R/'checkpoints.json'),synthetic_results_sha256=sha(R/'synthetic_results.json'),models=9,sets=2,identities_per_set=256));print('V46 SYNTHETIC COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--run',type=int,choices=SEEDS);p.add_argument('--resume',action='store_true');p.add_argument('--synthetic',action='store_true');p.add_argument('--device',default='cuda',choices=['cuda','cpu']);a=p.parse_args();assert sum([a.prepare,a.freeze,a.run is not None,a.synthetic])==1
 if a.prepare:prepare()
 elif a.freeze:freeze()
 elif a.run is not None:run(a.run,a.resume,a.device)
 else:synthetic(a.device)
