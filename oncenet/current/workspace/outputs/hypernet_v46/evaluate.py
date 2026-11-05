"""Controlled, development-only duration and procedural-generator ablations."""
from common import *
import argparse,sys
from threadpoolctl import threadpool_limits
DATA=R/'data';O=R/'evaluation';O.mkdir(exist_ok=True)

def registry():
 old=read(V45/'multiseed/checkpoints.json');div=read(V45/'multiseed/diversity/checkpoints.json')
 spec=read(R/'spectrogram/checkpoints.json');out=[]
 for source,arm in [(old,'random'),(div,'global_diversity')]:
  for e in source['models']:
   if e['arm']==arm:out.append(dict(e,kind='cnn',training='random' if arm=='random' else 'shapes'))
 for e in spec['models']:out.append(dict(e,kind='cnn',training='spectrogram'))
 assert len(out)==9 and {(e['training'],e['seed']) for e in out}=={(a,s) for a in ['random','shapes','spectrogram'] for s in SEEDS}
 assert old['training_complete'] and div['training_complete'] and spec['training_complete']
 v44=read(V44/'model_registry.json')
 out.append(dict(v44['old'],id='old_v37',kind='cnn',training='historical',seed=None,arm='historical'))
 out.append(dict(v44['dino_small'],id='dino_small',kind='dino',parameters=22056576,training='external',seed=None,arm='dino_small'))
 assert len({e['id'] for e in out})==len(out)
 return dict(models=out,source_registries={str(p):sha(p) for p in [V45/'multiseed/checkpoints.json',V45/'multiseed/diversity/checkpoints.json',R/'spectrogram/checkpoints.json',V44/'model_registry.json']})

def freeze():
 assert not (O/'protocol.json').exists() and not (O/'results.json').exists()
 rows=read(DATA/'records.json');refs=references(rows);audit=read(DATA/'audit.json')
 preflight=read(R/'evaluation_data_preflight.json');assert preflight['passed'] and preflight['no_model_scores'];bindings(preflight['files'])
 original={r['event_id']:r for r in read(V44/'used_records.json')}
 assert len({r['event_id'] for r in rows})==len(rows)
 for row in rows:
  prior=original[row['event_id']]
  for k in ['label','split','ifo','event_time']:assert row[k]==prior[k]
  assert set(row['duration_sources'])==set(DURATIONS)
 reg=registry();save(O/'registry.json',reg)
 for e in reg['models']:
  if 'path' in e:assert sha(e['path'])==e['sha256']
  bindings(e.get('files',{}))
 checks=read(R/'matcher_checks.json');assert checks['passed'] and checks['common_sha256']==sha(R/'common.py')
 files=[Path(__file__),R/'common.py',R/'matcher_checks.py',R/'matcher_checks.json',DATA/'records.json',DATA/'inputs_binding.json',DATA/'audit.json',O/'registry.json',P/'hypernet_v41/preprocess.py',V44/'training_model.py',P/'hypernet_v43/matcher.py',P/'hypernet_v23/calibrate.py',P/'hypernet_model_sweep/balanced_metrics.py',R/'spectrogram/protocol.json',R/'evaluation_audit.py']
 files += [DATA/(v+'_'+d+'.npy') for d in DURATIONS for v in ['full','low']]
 files += [V45/'validation_audit.py',V45/'seed_development_audit.py',V44/'used_records.json',DATA/'plan.json',R/'duration_data.py',R/'research.json',R/'protocol_review.json',R/'intent.json']
 files += [DATA/k for k in ['download_failures.json','excluded_events.json','acquisition_audit.json','input_pixel_hashes.json','source_failure_probe.json','supplemental_data_provenance.json','collision_review.json'] if (DATA/k).exists()]
 files += [R/'evaluation_data_preflight.json',R/'data_preflight.py']
 save(O/'protocol.json',dict(frozen_epoch=time.time(),files={str(p):sha(p) for p in files},reference_ids=REFERENCE_IDS,
  one_positive_event_per_target=True,reference_pngs=dict(single=1,multi=4),duration_seconds=[.5,1,2,4],views=['single','multi'],primary_readout='base',
  negative_fit_count=512,fit_policy='unchanged v42 label-balanced deterministic FIT on common complete-event rows',
  alphas=ALPHAS,primary_alpha=.001,primary_ranking_metric='pauc_std_01',secondary_ranking_metric='pauc_std_001',
  matcher='same shrinkage exemplar LDA; effective_dimension fixed to single-duration descriptor dimension for BOTH arms; repeated-single equivalence is checked',
  single_dimension=dict(cnn=4032,dino=1536),cnn_parameters=42480,models=[e['id'] for e in reg['models']],
  control='Same original three independent random starts; generator-only contrast holds training objective/optimizer/800steps fixed. No seed ensemble, reference augmentation, detector normalization, threshold redesign or larger encoder.',
  comparisons=['Within-checkpoint multi minus single','Within-seed spectrogram training minus prior shapes training at single duration','Within-seed trained minus random at each duration','Difference in duration benefit between trained and random'],
  selection='None. All three seeds and all fixed settings reported; existing used DEV only. No confirmation test is authorized by this protocol.',
  no_new_real_holdout=True,row_count=len(rows),split_counts={k:sum(r['split']==k for r in rows) for k in sorted({r['split'] for r in rows})},
  dino='Matched energy full+low inputs, official normalization and CLS+patch-mean pooling, same ELDA. This isolates duration effects, not a claim to its best task-specific pipeline.',
  limitations=['Complete-event selection can alter population; all arms use identical rows.','Four renders provide more observed context than one PNG.','One reference event plus labeled negative FIT/CAL and labeled DEV is not one total real example.','Existing development data cannot provide fresh confirmation.','Three seeds do not multiply the number of independent images.','Detector/site distributions and temporal dependence may shift.']))
 print('V46 EVALUATION FROZEN',sha(O/'protocol.json'),flush=True)

def extract(entry,rows,device):
 import torch
 from torch.nn import functional as F
 arch=module('v46_arch',V44/'training_model.py');folder=O/'features'/entry['id'];folder.mkdir(parents=True,exist_ok=True)
 settings=dict(protocol_sha256=sha(O/'protocol.json'),registry_sha256=sha(O/'registry.json'),row_sha256=sha(DATA/'records.json'))
 binding=folder/'binding.json'
 if binding.exists():
  b=read(binding);assert b['settings']==settings;bindings(b['files']);return b
 if entry['kind']=='cnn':net=arch.load_encoder(entry['path'],device)
 else:
  from transformers import AutoModel
  net=AutoModel.from_pretrained(entry['folder'],local_files_only=True).to(device).eval()
 assert sum(x.numel() for x in net.parameters())==entry['parameters']
 files={};shapes={}
 with torch.inference_mode():
  for duration in DURATIONS:
   parts=[]
   for view in ['full','low']:
    arr=np.load(DATA/(view+'_'+duration+'.npy'),mmap_mode='r');z=[];batch=32 if entry['kind']=='cnn' else 20
    for i in range(0,len(rows),batch):
     x=torch.from_numpy(np.array(arr[i:i+batch])[:,None]).repeat(1,3,1,1).to(device)
     if entry['kind']=='cnn':v=arch.descriptor_base(net.net(x))
     else:
      x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
      x=(x-torch.tensor([.485,.456,.406],device=device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
      zz=net(pixel_values=x).last_hidden_state
      v=F.normalize(torch.cat([F.normalize(zz[:,0],dim=-1),F.normalize(zz[:,1:].mean(1),dim=-1)],dim=-1),dim=-1)
     z.append(v.cpu().numpy())
     if i% (batch*32)==0:save(O/'status.json',dict(stage='extracting',model=entry['id'],duration=duration,view=view,done=i,total=len(rows)))
    parts.append(np.concatenate(z))
   f=unit(np.concatenate(parts,1));assert np.isfinite(f).all()
   p=folder/(duration+'.npy');np.save(p,f);files[str(p)]=sha(p);shapes[duration]=list(f.shape)
 b=dict(settings=settings,files=files,shapes=shapes,parameters=entry['parameters']);save(binding,b);del net
 return b

def run():
 import torch
 threadpool_limits(2);torch.set_num_threads(2);assert not (O/'results.json').exists(),'Results exist; do not rerun'
 protocol=read(O/'protocol.json');bindings(protocol['files']);reg=read(O/'registry.json');bindings(reg['source_registries'])
 rows=read(DATA/'records.json');labels=np.array([r['label'] for r in rows]);splits=np.array([r['split'] for r in rows]);dev=np.flatnonzero(splits=='development');cal=np.flatnonzero(splits=='calibration');refs=references(rows)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 if device=='cuda':torch.cuda.set_per_process_memory_fraction(.45)
 rr=[];mac=[];detectors=[];false_labels=[];unions=[];fits={};sources={};parity=[];start=time.time()
 for entry in reg['models']:
  b=extract(entry,rows,device);sources.update(b['files']);folder=O/'features'/entry['id'];sources[str(folder/'binding.json')]=sha(folder/'binding.json')
  arrays={d:np.load(folder/(d+'.npy')) for d in DURATIONS};single=arrays['d1'];dim=single.shape[1]
  multi=np.concatenate([arrays[d] for d in DURATIONS],1)/2
  assert dim==protocol['single_dimension'][entry['kind']]
  scores={};states={};fits[entry['id']]={}
  for view,f in [('single',single),('multi',multi)]:
   for target in TARGETS:
    fi=MATCH.fit_indices(rows,target);nc=cal[labels[cal]!=target];assert len(fi)==512
    state=fit(f[fi],f[refs[target]],dim);s=f.astype(np.float64)@state['weight']+state['offset'];key=view+'__'+target;scores[key]=s
    for k,v in state.items():states[key+'__'+k]=v
    fits[entry['id']][key]=dict(fit_indices=fi.tolist(),calibration_indices=nc.tolist(),reference_index=refs[target],reference_id=rows[refs[target]]['event_id'],effective_dimension=dim)
    if view=='single':
     # All repeated controls are numerical checks, not separately selected arms.
     rep=np.concatenate([f]*4,1)/2;rs=fit(rep[fi],rep[refs[target]],dim);check=rep.astype(np.float64)@rs['weight']+rs['offset'];error=float(np.max(np.abs(check-s)));scale=max(1.,float(np.max(np.abs(s))));assert error<2e-6*scale
     for alpha in ALPHAS:
      c1=CAL.cutoff(s[nc],alpha);c2=CAL.cutoff(check[nc],alpha);assert c1 is not None and c2 is not None
      assert np.array_equal(s[dev]>c1,check[dev]>c2),'Repeated input changed decisions'
     parity.append(dict(model=entry['id'],target=target,max_absolute_score_error=error,relative_scale_error=error/scale,all_calibrated_dev_decisions_equal=True))
    for alpha in ALPHAS:
     cutoff=CAL.cutoff(s[nc],alpha);assert cutoff is not None
     info=dict(model=entry['id'],seed=entry['seed'],training=entry['training'],view=view,target=target,alpha=alpha,threshold=cutoff,reference_id=REFERENCE_IDS[target])
     rr.append(dict(**info,**metric(labels==target,s,dev,cutoff)))
     for detector in ['H1','L1']:
      ix=np.array([i for i in dev if rows[i]['ifo']==detector],dtype=int)
      if len(ix) and 0<(labels[ix]==target).sum()<len(ix):detectors.append(dict(**info,detector=detector,**metric(labels==target,s,ix,cutoff)))
     for label in sorted(set(labels[dev])-set([target])):
      ix=dev[labels[dev]==label];false_labels.append(dict(**info,negative_label=label,fp=int((s[ix]>cutoff).sum()),negative=len(ix)))
   for alpha in ALPHAS:
    group=[r for r in rr if r['model']==entry['id'] and r['view']==view and r['alpha']==alpha];mac.append(dict(model=entry['id'],seed=entry['seed'],training=entry['training'],view=view,alpha=alpha,**summarize(group)))
    predictions=[]
    for target in TARGETS:
     row=next(x for x in group if x['target']==target);predictions.append(scores[view+'__'+target]>row['threshold'])
    any_alarm=np.logical_or.reduce(predictions)
    for label in sorted(set(labels[dev])-set(TARGETS)):
     ix=dev[labels[dev]==label];unions.append(dict(model=entry['id'],seed=entry['seed'],training=entry['training'],view=view,alpha=alpha,negative_label=label,fp=int(any_alarm[ix].sum()),negative=len(ix)))
  for suffix,value in [('scores',scores),('states',states)]:
   p=O/(entry['id']+'_'+suffix+'.npz');np.savez(p,**value);sources[str(p)]=sha(p)
  save(O/'status.json',dict(stage='scored',model=entry['id'],completed=len(mac)//4,total=len(reg['models']),seconds=time.time()-start));print('V46 SCORED',entry['id'],flush=True)
 save(O/'fits.json',fits);sources[str(O/'fits.json')]=sha(O/'fits.json')
 pairs=[];summaries=[]
 for training in ['random','shapes','spectrogram']:
  for view in ['single','multi']:
   for alpha in ALPHAS:
    group=[x for x in mac if x['training']==training and x['view']==view and x['alpha']==alpha];assert len(group)==3
    summaries.append(dict(training=training,view=view,alpha=alpha,metrics={k:dict(mean=float(np.mean([x[k] for x in group])),min=min(x[k] for x in group),max=max(x[k] for x in group)) for k in FIELDS}))
 for entry in reg['models']:
  for alpha in ALPHAS:
   a=next(x for x in mac if x['model']==entry['id'] and x['view']=='multi' and x['alpha']==alpha);b=next(x for x in mac if x['model']==entry['id'] and x['view']=='single' and x['alpha']==alpha)
   pairs.append(dict(model=entry['id'],seed=entry['seed'],training=entry['training'],alpha=alpha,contrast='multi_minus_single',metrics={k:a[k]-b[k] for k in FIELDS}))
 training_pairs=[];interactions=[]
 for seed in SEEDS:
  for alpha in ALPHAS:
   for left,right in [('spectrogram','shapes'),('shapes','random'),('spectrogram','random')]:
    for view in ['single','multi']:
     a=next(x for x in mac if x['seed']==seed and x['training']==left and x['view']==view and x['alpha']==alpha);b=next(x for x in mac if x['seed']==seed and x['training']==right and x['view']==view and x['alpha']==alpha)
     training_pairs.append(dict(seed=seed,left=left,right=right,view=view,alpha=alpha,metrics={k:a[k]-b[k] for k in FIELDS}))
   for trained in ['shapes','spectrogram']:
    a=next(x for x in pairs if x['seed']==seed and x['training']==trained and x['alpha']==alpha);b=next(x for x in pairs if x['seed']==seed and x['training']=='random' and x['alpha']==alpha)
    interactions.append(dict(seed=seed,trained=trained,alpha=alpha,contrast='trained_duration_gain_minus_random_duration_gain',metrics={k:a['metrics'][k]-b['metrics'][k] for k in FIELDS}))
 save(O/'results.json',dict(complete=True,no_new_real_holdout=True,no_selection=True,protocol_sha256=sha(O/'protocol.json'),registry_sha256=sha(O/'registry.json'),rows=rr,macro=mac,seed_summary=summaries,duration_pairs=pairs,training_pairs=training_pairs,interactions=interactions,detector_rows=detectors,false_alarm_by_label=false_labels,any_target_false_alarm_by_label=unions,repeated_single_parity=parity,files=sources,seconds=time.time()-start))
 save(O/'status.json',dict(complete=True,stage='complete',seconds=time.time()-start));print('V46 EVALUATION COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
