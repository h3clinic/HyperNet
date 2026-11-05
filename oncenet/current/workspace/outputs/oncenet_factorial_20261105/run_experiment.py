"""Prospective, immutable 2x2 OnceNet training/matcher experiment.

All historical artifacts are read-only. New labels are opened for metrics only
after the scores for every prescribed condition have been committed.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, time, traceback
import numpy as np
import factorial_core as C
import factorial_metrics as M

R=Path(__file__).resolve().parent
W=R.parents[1]
O=W/'outputs'
H=O/'hypernet_v46/data'
SEEDS=list(C.SEEDS); TARGETS=list(C.TARGETS); DUR=list(C.DURATIONS)
ALPHAS=[.001,.01]

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return C.sha(p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(x,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
 os.replace(t,p)
def status(stage,**kw): save(R/'status.json',dict(stage=stage,updated_utc=now(),pid=os.getpid(),**kw))
def group(r): return str(r['ifo'])+'_'+str(int(float(r['event_time'])//3600))
def hash_order(r): return hashlib.sha256(('oncenet-factorial-reference-20261105:'+r['event_id']).encode()).hexdigest()

def plan_references():
 assert not (R/'protocol.json').exists()
 rows=read(H/'records.json');split=C.historical_split_plan(W)
 all_negative=set(i for t in TARGETS for key in ['fit','cal'] for i in split['targets'][t][key])
 references={}
 for t in TARGETS:
  first=split['targets'][t]['reference']
  options=sorted([i for i in split['development_indices'] if rows[i]['label']==t and i not in all_negative],key=lambda i:hash_order(rows[i]))
  picked=[first];groups={group(rows[first])}
  for i in options:
   if group(rows[i]) not in groups:
    picked.append(i);groups.add(group(rows[i]))
   if len(picked)==5:break
  assert len(picked)==5,(t,len(picked))
  assert all(i not in set(split['targets'][t]['fit'])|set(split['targets'][t]['cal']) for i in picked)
  references[t]=[dict(index=i,event_id=rows[i]['event_id'],label=t,group=group(rows[i]),original_split=rows[i]['split']) for i in picked]
 plan=dict(created_utc=now(),references=references,split=split,
  policy='Original locked reference plus four SHA256-ordered positive DEV events; five distinct detector-hours within each target; no score/appearance selection. One event per classifier/episode, never pooled.',
  historical_records_sha256=sha(H/'records.json'))
 save(R/'reference_plan.json',plan)
 save(R/'model_registry.json',C.registry(W))
 print('REFERENCE_PLAN', {t:[r['event_id'] for r in references[t]] for t in TARGETS},flush=True)

def inputs(folder):
 result={}
 for d in DUR:
  result[d]={}
  for v in ['full','low']:
   options=[Path(folder)/(v+s+d+'.npy') for s in ['_','__']]
   present=[p for p in options if p.exists()]
   assert len(present)==1,(folder,d,v)
   result[d][v]=np.load(present[0],mmap_mode='r')
 return result

def checks():
 assert not (R/'protocol.json').exists()
 import torch
 from threadpoolctl import threadpool_limits
 torch.set_num_threads(2);threadpool_limits(2)
 models=C.registry(W);a=inputs(H)
 ix=read(R/'reference_plan.json')['split']['development_indices'][:12]
 fixture={d:{v:np.array(a[d][v][ix]) for v in ['full','low']} for d in DUR}
 parity=[]
 for e in models:
  cpu=C.extract_features(e,fixture,'cpu',32)
  gpu=C.extract_features(e,fixture,'cuda',32)
  errors={d:float(np.max(np.abs(cpu[d]-gpu[d]))) for d in DUR}
  assert max(errors.values())<2e-5,errors
  parity.append(dict(model=e['id'],duration_max_absolute_errors=errors))
 result=dict(passed=True,completed_utc=now(),core=C.self_test(W),metrics=M.sanity_checks(),cpu_cuda_fixture=parity,
  fixture_indices=ix,fixture_is_previously_consumed=True,torch_version=torch.__version__,device=torch.cuda.get_device_name(0),
  settings=dict(dtype='float32',matmul_precision='highest',tf32=False,autocast=False,cudnn_benchmark=False,batch=32),
  source_hashes={str(R/n):sha(R/n) for n in ['factorial_core.py','factorial_metrics.py','run_experiment.py']})
 save(R/'implementation_checks.json',result);print('CHECKS_PASSED',flush=True)

def freeze():
 assert not (R/'protocol.json').exists(),'Already frozen'
 assert (R/'data_plan.json').exists(),'Data sampling plan required'
 checks=read(R/'implementation_checks.json');assert checks['passed']
 for p,h in checks['source_hashes'].items(): assert sha(p)==h,p
 sources=[R/n for n in ['run_experiment.py','factorial_core.py','factorial_metrics.py','prepare_data.py','experiment_intent.json','reference_plan.json','model_registry.json','implementation_checks.json','data_plan.json','intent_unlabelled.json','labels.json']]
 sources += [H/'records.json',O/'hypernet_v46/common.py',O/'hypernet_v43/matcher.py',O/'hypernet_v23/calibrate.py',O/'hypernet_model_sweep/balanced_metrics.py']
 sources += [O/'cloud_plan_20261105/gs_fp32'/n for n in ['model.py','kaggle_pilot/model.py','preprocess.py','kaggle_pilot/preprocess.py','migration_manifest.json']]
 sources += [O/'hypernet_v46/spectrogram'/n for n in ['protocol.json','checkpoints.json']]
 sources += [H/(v+'_'+d+'.npy') for d in DUR for v in ['full','low']]
 sources += [Path(e['path']) for e in read(R/'model_registry.json')]
 source_hashes={str(p.resolve()):sha(p) for p in sources}
 p=dict(version='oncenet_factorial_20261105_v1',frozen_utc=now(),intent=read(R/'experiment_intent.json'),
  source_hashes=source_hashes,sampling=read(R/'data_plan.json'),references=read(R/'reference_plan.json'),models=read(R/'model_registry.json'),
  axis_order=dict(training=['random','trained'],matcher=['cosine','negative_aware'],seeds=SEEDS,reference_episodes=list(range(5)),targets=TARGETS,alphas=ALPHAS),
  runtime=checks['settings']|dict(device='cuda',torch_version=checks['torch_version'],gpu=checks['device']),
  feature_storage='Feature arrays held in memory per model; persist SHA256 of dtype/shape-bound array bytes rather than redundant feature copies. Inputs, fitted states, thresholds and all scores retained.',
  eligible_exclusions='Same events for every condition; all four renders required; bounded download/integrity/independent-renderer validation; exact native/full-crop historical collision exclusions. No label/score/appearance-based replacement.',
  primary=dict(endpoint='standardized_pAUC_max_fpr_0.001',contrasts=['training_main','matcher_main'],confidence=.975,bootstrap_draws=2000,bootstrap_seed=51106),
  reporting='All three trained seeds and their paired random initializations, all five single-reference episodes and all four cells; no post-test selection or model changes.',
  test_labels_access='Metadata informed balanced sampling. Scoring runner opens labels only after all-score SHA256 commitment; this is prospective evaluation, not a label-blind sample.',
  scientific_status='New experiment separate from the submitted report. No modifications to historical confirmation, models, manuscript or application.')
 save(R/'protocol.json',p);(R/'protocol.sha256').write_text(sha(R/'protocol.json')+'\n')
 print('FROZEN',sha(R/'protocol.json'),flush=True)

def verify(full=False):
 p=read(R/'protocol.json')
 assert sha(R/'protocol.json')==(R/'protocol.sha256').read_text().strip()
 for f,h in p['source_hashes'].items():
  if full or Path(f).suffix in ['.py','.json','.pt']: assert sha(f)==h,f
 return p

def feature_cache(entry,folder,tag,rows_hash):
 out=R/'features'/tag/entry['id'];out.mkdir(parents=True,exist_ok=True)
 binding=out/'binding.json'
 expected=dict(protocol_sha256=sha(R/'protocol.json'),model_sha256=entry['sha256'],rows_sha256=rows_hash,tag=tag)
 if binding.exists(): assert read(binding)['settings']==expected
 def progress(info):
  if info['done']%512==0 or info['done']==info['total']:status('features',cohort=tag,**info)
 features=C.extract_features(entry,inputs(folder),'cuda',32,progress)
 fingerprints={d:dict(shape=list(f.shape),dtype=str(f.dtype),sha256=hashlib.sha256(np.ascontiguousarray(f).data).hexdigest()) for d,f in features.items()}
 if binding.exists(): assert read(binding)['arrays']==fingerprints,'Feature recomputation differed'
 else: save(binding,dict(settings=expected,arrays=fingerprints,storage='memory_only',completed_utc=now()))
 return C.joint_features(features)

def fit():
 verify(full=True)
 if (R/'fit_commitment.json').exists():
  prior=read(R/'fit_commitment.json');assert prior['protocol_sha256']==sha(R/'protocol.json')
  for f,h in prior['files'].items(): assert sha(f)==h,f
  print('FITS_ALREADY_COMMITTED',flush=True);return
 assert not (R/'acquisition.json').exists(),'Complete historical fitting before fresh acquisition'
 import torch
 from threadpoolctl import threadpool_limits
 torch.set_num_threads(2);threadpool_limits(2)
 plan=read(R/'reference_plan.json');hist=read(H/'records.json');models=read(R/'model_registry.json')
 thresholds=np.empty((2,2,3,5,4,2),dtype=np.float64);paths={};calibration_log=[]
 for e in models:
  tr=['random','trained'].index(e['arm']);se=SEEDS.index(e['seed'])
  x=feature_cache(e,H,'historical',sha(H/'records.json'))
  for ti,t in enumerate(TARGETS):
   sp=plan['split']['targets'][t];fi=sp['fit'];ci=sp['cal']
   assert all(hist[i]['label']!=t for i in fi+ci)
   prepared=C.prepare_elda(x[fi],4032)
   for ri,ref in enumerate(plan['references'][t]):
    assert hist[ref['index']]['event_id']==ref['event_id'] and hist[ref['index']]['label']==t
    for ma,method in enumerate(['cosine','negative_aware']):
     state=C.fit_cosine(x[ref['index']]) if ma==0 else C.fit_prepared_elda(prepared,x[ref['index']])
     cs=C.score(x[ci],state)
     destination=R/'fits'/e['id']/(t+f'_r{ri}_{method}.npz');destination.parent.mkdir(parents=True,exist_ok=True)
     np.savez(destination,**state);paths[str(destination)]=sha(destination)
     for ai,alpha in enumerate(ALPHAS):
      _,_,cut=C.calibrate(cs,np.empty(0),alpha)
      assert cut is not None,'Insufficient original calibration resolution'
      thresholds[tr,ma,se,ri,ti,ai]=cut
     calibration_log.append(dict(model=e['id'],target=t,reference_episode=ri,reference_id=ref['event_id'],matcher=method,fit_count=len(fi) if ma else 0,calibration_count=len(ci),calibration_score_sha256=hashlib.sha256(cs.tobytes()).hexdigest()))
   status('fitting',model=e['id'],target=t)
  print('FITTED',e['id'],flush=True)
  del x
 np.save(R/'thresholds.npy',thresholds);paths[str(R/'thresholds.npy')]=sha(R/'thresholds.npy')
 save(R/'calibration_log.json',calibration_log);paths[str(R/'calibration_log.json')]=sha(R/'calibration_log.json')
 save(R/'fit_commitment.json',dict(complete=True,committed_utc=now(),files=paths,protocol_sha256=sha(R/'protocol.json'),no_fresh_query_images_used=True))

def score_test():
 verify()
 assert not (R/'score_commitment.json').exists(),'All scores already committed; do not rescore'
 audit=read(R/'independent_preflight.json');assert audit['passed'] and audit['protocol_sha256']==sha(R/'protocol.json')
 assert audit['runner_sha256']==sha(__file__)
 acquisition=read(R/'acquisition.json');assert acquisition['complete']
 assert acquisition['protocol_sha256']==sha(R/'protocol.json')
 for f,h in acquisition['files'].items():assert sha(f)==h,f
 fitted=read(R/'fit_commitment.json');assert fitted['protocol_sha256']==sha(R/'protocol.json')
 for f,h in fitted['files'].items():assert sha(f)==h,f
 import torch
 from threadpoolctl import threadpool_limits
 torch.set_num_threads(2);threadpool_limits(2)
 records=read(R/'records_unlabelled.json');assert records
 ids=[r['event_id'] for r in records];assert len(ids)==len(set(ids))
 intent=read(R/'intent_unlabelled.json');keep=set(ids)
 assert ids==[r['event_id'] for r in intent if r['event_id'] in keep]
 assert len({group(r) for r in records})==len(records)
 assert len(records)==acquisition['eligible_count']
 scores=np.empty((2,2,3,5,4,len(records)),dtype=np.float64)
 for e in read(R/'model_registry.json'):
  tr=['random','trained'].index(e['arm']);se=SEEDS.index(e['seed'])
  x=feature_cache(e,R/'inputs','fresh',sha(R/'records_unlabelled.json'))
  for ti,t in enumerate(TARGETS):
   for ri in range(5):
    for ma,method in enumerate(['cosine','negative_aware']):
     source=R/'fits'/e['id']/(t+f'_r{ri}_{method}.npz')
     with np.load(source) as a:state={k:a[k] for k in a.files}
     scores[tr,ma,se,ri,ti]=C.score(x,state)
  print('SCORED',e['id'],flush=True);del x
 assert np.isfinite(scores).all();np.save(R/'scores.npy',scores)
 save(R/'score_commitment.json',dict(committed_utc=now(),protocol_sha256=sha(R/'protocol.json'),files={str(R/n):sha(R/n) for n in ['scores.npy','thresholds.npy','records_unlabelled.json','acquisition.json']},shape=list(scores.shape),all_conditions_scored_before_metrics=True))

def analyze():
 verify();assert not(R/'results.json').exists(),'Results already finalized; no unnecessary rerun'
 commitment=read(R/'score_commitment.json')
 assert commitment['protocol_sha256']==sha(R/'protocol.json')
 for p,h in commitment['files'].items():assert sha(p)==h,p
 records=read(R/'records_unlabelled.json');data=read(R/'labels.json')
 lookup=data if isinstance(data,dict) else {r['event_id']:r['label'] for r in data}
 labels=[lookup[r['event_id']] if isinstance(lookup[r['event_id']],str) else lookup[r['event_id']]['label'] for r in records]
 status('analysis',events=len(records),bootstrap_draws=2000)
 scores=np.load(R/'scores.npy');assert scores.shape[-1]==len(records)
 results=M.evaluate(labels,[group(r) for r in records],scores,np.load(R/'thresholds.npy'),TARGETS)
 results.update(protocol_sha256=sha(R/'protocol.json'),score_commitment_sha256=sha(R/'score_commitment.json'),completed_utc=now(),model_seed_ids=SEEDS,reference_plan=read(R/'reference_plan.json')['references'],acquisition=read(R/'acquisition.json'))
 save(R/'results.json',results)
 with (R/'cell_summary.csv').open('w',newline='',encoding='utf-8') as f:
  keys=['training','matcher']+list(results['cells'][0]['metrics']);writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
  for row in results['cells']:writer.writerow(dict(training=row['training'],matcher=row['matcher'],**{k:v['estimate'] for k,v in row['metrics'].items()}))
 status('complete',complete=True,events=len(records),results_sha256=sha(R/'results.json'))
 print('COMPLETE',json.dumps(results['primary']),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan_references','checks','freeze','fit','score_test','analyze']);args=parser.parse_args()
 try:globals()[args.action]()
 except Exception as e:
  save(R/'failure.json',dict(action=args.action,at_utc=now(),error=str(e),traceback=traceback.format_exc()));status('failed',action=args.action,error=str(e));raise
