"""One frozen CPU-only confirmation. Never changes an encoder, matcher or CAL."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse,csv,hashlib,importlib.util,json,math,time,traceback
import numpy as np
R=Path(__file__).resolve().parent; W=R.parents[1]; O=W/'outputs'
V46=O/'hypernet_v46';V47=O/'hypernet_v47'; GS=O/'cloud_plan_20261105/gs_fp32'
DUR=['d05','d1','d2','d4']; TARGETS=['Low_Frequency_Burst','Low_Frequency_Lines','Scattered_Light','Blip']
MODELS=['oncenet','random','dino'];ALPHAS=[.001,.01]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify(p):
 assert sha(__file__)==p['code_sha256'],'Runner changed after freeze'
 for f,h in p['source_hashes'].items():assert sha(f)==h,f
 assert sha(R/'protocol.json')==(R/'protocol.sha256').read_text().strip()
def review_ok(p):
 a=read(R/'independent_preflight_audit.json')
 assert a['passed'] and a['code_sha256']==p['code_sha256'],'Independent review must cover exact frozen runner'
 if a.get('protocol_sha256'):assert a['protocol_sha256']==sha(R/'protocol.json')
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def cut(cal,a):
 q=Fraction(str(a));k=(len(cal)+1)*q.numerator//q.denominator
 return float(np.sort(cal)[-k]) if k else None
def freeze():
 assert not(R/'protocol.json').exists(),'Protocol already frozen'
 reserve=O/'hypernet_v45/reserve_metadata_audit.json';a=read(reserve)
 assert a['metadata_only'] and a['no_images_opened_downloaded_or_scored']
 candidates=a['records'];assert len(candidates)==195
 used=read(V46/'data/records.json');ids={r['event_id'] for r in used}
 assert not ids&{r['event_id'] for r in candidates}
 gap=min(abs(r['event_time']-u['event_time']) for r in candidates for u in used if r['ifo']==u['ifo']);assert gap>8
 catalog=O/'hypernet_v16/data/gravity_fresh_v16/official_metadata';inventory={}
 for name in sorted({r['source_csv'] for r in candidates}):
  with (catalog/name).open(newline='',encoding='utf-8') as f:
   for row in csv.DictReader(f):inventory[(name,row['gravityspy_id'])]=row
 intent=[]
 for r in candidates:
  c=inventory[(r['source_csv'],r['event_id'])]
  assert c['ifo']==r['ifo'] and abs(float(c['event_time'])-r['event_time'])<1e-4
  assert c['url2']==r['url2'] and c['url4']==r['url4']
  urls={d:c['url'+str(i+1)] for i,d in enumerate(DUR)}
  assert all(u.startswith('https://panoptes-uploads.zooniverse.org/production/subject_location/') for u in urls.values())
  assert len(set(urls.values()))==4,'Four distinct authentic duration URLs required'
  intent.append(dict(event_id=r['event_id'],ifo=r['ifo'],event_time=r['event_time'],source_csv=r['source_csv'],urls=urls))
 save(R/'intent_unlabelled.json',intent)
 reg=read(V46/'evaluation/registry.json');entries={e['id']:e for e in reg['models']}
 sources=[Path(__file__),reserve,R/'intent_unlabelled.json',V46/'data/records.json',V46/'evaluation/registry.json',V46/'evaluation/results.json',V46/'evaluation/audit.json',V46/'evaluation/fits.json',V47/'splits.json',V47/'evaluation/audit.json',O/'hypernet_locked/manifest.json']
 sources +=[catalog/n for n in sorted({r['source_csv'] for r in candidates})]
 sources +=[GS/n for n in ['encoder.pt','model.py','preprocess.py','matcher.npz','calibration.npz','migration_manifest.json','development_audit.json','scores_decisions.npz','kaggle_pilot/events.json']]
 sources +=[Path(entries['s45029_random']['path']),V46/'evaluation/s45029_random_states.npz',V46/'evaluation/s45029_random_scores.npz']
 sources +=[Path(p) for p in entries['dino_small']['files']]
 sources +=[V47/'evaluation/dino_small'/n for n in ['states.npz','normalization.json','fused.npz']]
 sources +=[V46/'data'/(v+'_'+d+'.npy') for d in DUR for v in ['full','low']]
 sources +=[GS/'features'/(d+'.npy') for d in DUR]
 sources +=[V46/'evaluation/features'/m/(d+'.npy') for m in ['s45029_random','dino_small'] for d in DUR]
 fits=read(V46/'evaluation/fits.json');sp=read(V47/'splits.json')
 for m in ['s45029_random','dino_small']:
  for t in TARGETS:
   fit=fits[m]['multi__'+t]
   assert fit['reference_id']==read(O/'hypernet_locked/manifest.json')['targets'][t]['reference_event_id']
   assert set(fit['fit_indices'])==set(sp['targets'][t]['fit'])
   assert set(fit['calibration_indices'])==set(sp['targets'][t]['original_cal'])
 fixed={}
 for name in MODELS:
  cal=calibration(name);fixed[name]={t:{str(a):dict(threshold=cut(cal[t],a),calibration_count=len(cal[t])) for a in ALPHAS} for t in TARGETS}
 save(R/'frozen_thresholds.json',fixed);sources.append(R/'frozen_thresholds.json')
 # Inherited overlap records and source bindings are metadata only.
 historical=[Path(f) for f in a['source_bindings'] if f.endswith('records.json')]
 sources += historical
 sources=list(dict.fromkeys(sources))
 p=dict(version='OnceNet_reserved_195_confirmation_v1',frozen_utc=now(),code_sha256=sha(__file__),source_hashes={str(f):sha(f) for f in sources},candidate_count=195,candidate_identity_file=str(R/'intent_unlabelled.json'),labels_only_for_final_metrics=str(reserve),prior_metadata_labels_were_already_visible=True,minimum_existing_used_same_detector_gap_seconds=gap,
  models=dict(oncenet=dict(checkpoint=str(GS/'encoder.pt'),matcher='strict FP32 deployed joint4, original CAL',parameters=42480),random=dict(checkpoint=entries['s45029_random']['path'],matcher='v46 fixed joint4, original CAL',parameters=42480),dino=dict(folder=entries['dino_small']['folder'],matcher='v47 late_mean4 standardized per-duration ELDA, original pooled CAL',parameters=22056576)),
  references={t:read(O/'hypernet_locked/manifest.json')['targets'][t]['reference_event_id'] for t in TARGETS},positive_reference_events_per_target=1,no_training=True,no_refit=True,borrowed_oncenet_weights=False,
  preprocessing='Exact bound gs_fp32.preprocess: native viridis inversion full+low,128x128; four authentic 0.5/1/2/4-second PNGs per event.',inference=dict(device='cpu',dtype='float32',threads=2,cnn_batch=32,dino_batch=8,no_amp=True,no_gpu=True),
  eligibility='All 195 intent IDs retained in accounting. Exclude an event from all methods if any actual duration fails three bounded download attempts, PNG800x600 integrity or independent legend check; any exact native or native full-plot pixel collision with historical consumed events, or another reserve event (keep first frozen-order event). No replacements or exclusions by label, heatmap appearance, score, confidence, detector or morphology. Constant derived low views are not whole-event collisions.',
  source_pixel_history_files=[str(f) for f in historical],download=dict(workers=8,attempts=3,timeout_seconds=[10,40],max_bytes=10000000),
  primary=dict(alpha=.001,metrics=['macro_balanced_accuracy','macro_mcc','macro_recall','macro_fa_per10000','per_target_TP_FP_TN_FN','union_false_alarms_non_target']),secondary=dict(alpha=.01,note='Newly predeclared analysis of the same frozen scores/CAL; original release advertised .001/.0005. No calibration or threshold fitting change.'),
  decision='Conservative finite-sample rank: score strictly greater than sortedCAL[-floor((N+1)*alpha)]; ties do not alarm.',
  ranking=['AUROC','standardized partial AUROC at max_fpr .001 and .01','descriptive maximum empirical recall with achieved FPR<=.001 and .01; no exported test-selected thresholds'],
  intervals=dict(bootstrap_seed=51105,draws=2000,unit='detector x floor(GPS_seconds/3600) cluster, sampled with replacement',paired=True,comparisons=['oncenet_minus_random','oncenet_minus_dino'],metrics=['balanced_accuracy','mcc','recall','fa_per10000','pauc_std_001','pauc_std_01'],undefined_draws='Skip and report if any target lacks positives or negatives; do not replace draws',confidence=.95),
  score_commitment='Write and SHA256 all three model scores before opening labels for test metrics. All fixed models reported; no winner selection.',
  runtime_parity='Before reserve scoring, compare each CPU encoder on the same 12 already-used portability fixture events against historical feature arrays; record full numerical errors and alarm changes. Require zero primary alarm flips for each model. Halt on nonfinite/wrong shape, OnceNet descriptor max absolute drift>0.00002 or comparator drift>0.02. Any failure requires reported numerical diagnosis; no silent runtime migration. This consumed-data numerical check cannot establish universal feature equality. Thresholds are saved and hash-bound before reserve image access.',
  limitations=['Prospective image/model outcome evaluation, not label-blind metadata sampling; known reserve class counts informed its original selection.','Only195 planned events, only12Blip/13ScatteredLight; insufficient negatives to certify10false alarms/10000.','Detector and time dependence, incomplete cases and consensus-label errors remain.','Pixel separation checks cover available historical native/crop hashes, not unavailable source pixels or all possible perceptual duplicates.','One positive event plus existing labeled negative FIT/CAL and repeated earlier DEV selection.','No claims of new discoveries or universal superiority.'])
 save(R/'protocol.json',p);(R/'protocol.sha256').write_text(sha(R/'protocol.json')+'\n');print('FROZEN',sha(R/'protocol.json'),flush=True)

def acquire():
 p=read(R/'protocol.json');verify(p)
 review_ok(p)
 assert not(R/'acquisition.json').exists(),'Acquisition finalized; do not rerun'
 import requests
 from PIL import Image
 prep=module('frozen_preprocessing',GS/'preprocess.py');intent=read(R/'intent_unlabelled.json')
 start=time.time();raw=R/'raw';raw.mkdir(exist_ok=True)
 def fetch(job):
  r,d=job;dest=raw/d/(r['event_id']+'.png');dest.parent.mkdir(exist_ok=True);err=None
  for attempt in range(3):
   try:
    if not dest.exists():
     resp=requests.get(r['urls'][d],timeout=(10,40));resp.raise_for_status();assert len(resp.content)<=10000000
     temp=dest.with_suffix('.part');temp.write_bytes(resp.content)
     with Image.open(temp) as im:assert im.format=='PNG' and im.size==(800,600);im.verify()
     os.replace(temp,dest)
    with Image.open(dest) as im:assert im.format=='PNG' and im.size==(800,600);rgb=np.asarray(im.convert('RGB'))
    check=prep.validate_renderer(rgb);crop=prep.native_rgb(rgb,'full')
    return dict(event_id=r['event_id'],duration=d,path=str(dest),url=r['urls'][d],sha256=sha(dest),pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),crop_pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),renderer=check)
   except Exception as e:
    err=str(e)
    if attempt<2:time.sleep(1+attempt)
  return dict(event_id=r['event_id'],duration=d,error=err)
 jobs=[(r,d) for r in intent for d in DUR];bindings=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(fetch,j) for j in jobs]
  for i,f in enumerate(as_completed(futures),1):
   bindings.append(f.result())
   if i%25==0 or i==len(jobs):save(R/'status.json',dict(stage='acquisition',done=i,total=len(jobs),elapsed_seconds=time.time()-start,pid=os.getpid(),updated_utc=now()))
 save(R/'source_bindings.json',bindings)
 seen={'pixel_sha256':set(),'crop_pixel_sha256':set()}
 def collect(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in seen and isinstance(v,str):seen[k].add(v)
    else:collect(v)
  elif isinstance(x,list):
   for v in x:collect(v)
 coverage=[]
 def has_pixel(x):
  if isinstance(x,dict):return any(k in seen and isinstance(v,str) or has_pixel(v) for k,v in x.items())
  if isinstance(x,list):return any(has_pixel(v) for v in x)
  return False
 for f in list(dict.fromkeys([str(V46/'data/records.json')]+p['source_pixel_history_files'])):
  data=read(f);collect(data);rr=data if isinstance(data,list) else data.get('records',[])
  coverage.append(dict(path=f,record_count=len(rr),records_with_available_native_or_crop_hash=sum(has_pixel(r) for r in rr)))
 initial_counts={k:len(v) for k,v in seen.items()};by={(r['event_id'],r['duration']):r for r in bindings};kept=[];excluded=[]
 for r in intent:
  bb=[by[(r['event_id'],d)] for d in DUR];reasons=[]
  for b in bb:
   if 'error' in b:reasons.append(dict(duration=b['duration'],error=b['error']))
   else:
    for k in seen:
     if b[k] in seen[k]:reasons.append(dict(duration=b['duration'],collision=k,digest=b[k]))
  if reasons:excluded.append(dict(event_id=r['event_id'],reasons=reasons));continue
  kept.append(dict(r,windows={b['duration']:b for b in bb}))
  for b in bb:
   for k in seen:seen[k].add(b[k])
 arrays={v+'__'+d:[] for d in DUR for v in ['full','low']}
 for i,r in enumerate(kept):
  for d,b in r['windows'].items():
   with Image.open(b['path']) as im:rgb=np.asarray(im.convert('RGB'))
   for v in ['full','low']:arrays[v+'__'+d].append(prep.preprocess(rgb,'energy_'+v,128)[...,0])
  if i%25==0:save(R/'status.json',dict(stage='preprocess',done=i,total=len(kept),updated_utc=now()))
 assert kept,'No eligible records; preserve acquisition failure'
 np.savez(R/'inputs.npz',**{k:np.stack(v).astype(np.float32) for k,v in arrays.items()});save(R/'records_unlabelled.json',kept)
 save(R/'acquisition.json',dict(complete=True,completed_utc=now(),intent_count=len(intent),eligible_count=len(kept),excluded=excluded,source_pixel_history_counts=initial_counts,source_pixel_history_coverage=coverage,pixel_audit_limitation='Only available exact native/crop hashes are checked; no claim to universal perceptual duplicate exclusion.',all_models_same_cohort=True,no_model_inference=True,files={str(R/n):sha(R/n) for n in ['source_bindings.json','records_unlabelled.json','inputs.npz']},protocol_sha256=sha(R/'protocol.json')))
 print('ACQUIRED',len(kept),'EXCLUDED',len(excluded),flush=True)

def features(name,arrays,p,status=True):
 import torch
 from torch.nn import functional as F
 torch.set_num_threads(2);torch.backends.mkldnn.enabled=True
 if name=='dino':
  from transformers import AutoModel
  net=AutoModel.from_pretrained(p['models'][name]['folder'],local_files_only=True).to('cpu').eval();batch=8
 else:
  arch=module('frozen_model_'+name,GS/'model.py');net=arch.load_encoder(p['models'][name]['checkpoint'],'cpu');batch=32
 assert sum(t.numel() for t in net.parameters())==p['models'][name]['parameters']
 out={};start=time.time()
 with torch.inference_mode():
  for d in DUR:
   parts=[]
   for view in ['full','low']:
    a=arrays[view+'__'+d];rows=[]
    for i in range(0,len(a),batch):
     x=torch.from_numpy(np.array(a[i:i+batch])[:,None]).repeat(1,3,1,1)
     if name=='dino':
      x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
      x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
      z=net(pixel_values=x).last_hidden_state;z=F.normalize(torch.cat([F.normalize(z[:,0],dim=-1),F.normalize(z[:,1:].mean(1),dim=-1)],dim=-1),dim=-1)
     else:z=arch.descriptor_base(net.net(x))
     rows.append(z.numpy())
     if status:save(R/'status.json',dict(stage='inference',model=name,duration=d,view=view,done=i+len(x),total=len(a),elapsed_seconds=time.time()-start,updated_utc=now(),pid=os.getpid()))
    parts.append(np.concatenate(rows))
   out[d]=unit(np.concatenate(parts,1)).astype(np.float32)
   assert np.isfinite(out[d]).all()
 del net
 return out
def scores(name,feat):
 out={}
 if name=='dino':
  states=np.load(V47/'evaluation/dino_small/states.npz',allow_pickle=False);norm=read(V47/'evaluation/dino_small/normalization.json')
  for t in TARGETS:
   raw=np.stack([np.einsum('ij,j->i',feat[d].astype(np.float64),states[t+'__'+d+'__weight'],optimize=False)+states[t+'__'+d+'__offset'] for d in DUR],axis=1)
   out[t]=((raw-np.array(norm[t]['location']))/np.array(norm[t]['scale'])).mean(1)
 else:
  states=np.load(GS/'matcher.npz' if name=='oncenet' else V46/'evaluation/s45029_random_states.npz',allow_pickle=False)
  x=(np.concatenate([feat[d] for d in DUR],1)/2).astype(np.float64)
  for t in TARGETS:
   key=t if name=='oncenet' else 'multi__'+t
   out[t]=np.einsum('ij,j->i',x,states[key+'__weight'],optimize=False)+states[key+'__offset']
 return out
def calibration(name):
 sp=read(V47/'splits.json');out={}
 if name=='oncenet':
  z=np.load(GS/'calibration.npz',allow_pickle=False)
  return {t:z[t+'__scores'] for t in TARGETS}
 z=np.load(V46/'evaluation/s45029_random_scores.npz' if name=='random' else V47/'evaluation/dino_small/fused.npz',allow_pickle=False)
 prefix='multi__' if name=='random' else 'late_mean4__'
 return {t:np.sort(z[prefix+t][sp['targets'][t]['original_cal']]) for t in TARGETS}
def parity(p):
 fixture=read(GS/'kaggle_pilot/events.json');used=read(V46/'data/records.json');idx={r['event_id']:i for i,r in enumerate(used)};ii=[idx[r['event_id']] for r in fixture]
 arrays={v+'__'+d:np.load(V46/'data'/(v+'_'+d+'.npy'),mmap_mode='r')[ii] for d in DUR for v in ['full','low']}
 report=[]
 for name,old in [('oncenet','s45029_spectrogram'),('random','s45029_random'),('dino','dino_small')]:
  ft=features(name,arrays,p,status=False);path=GS/'features' if name=='oncenet' else V46/'evaluation/features'/old
  errors={d:float(np.max(np.abs(ft[d]-np.load(path/(d+'.npy'),mmap_mode='r')[ii]))) for d in DUR}
  actual=scores(name,ft);cal=calibration(name)
  saved=np.load(GS/'scores_decisions.npz' if name=='oncenet' else V46/'evaluation/s45029_random_scores.npz' if name=='random' else V47/'evaluation/dino_small/fused.npz',allow_pickle=False)
  prefix='score__' if name=='oncenet' else 'multi__' if name=='random' else 'late_mean4__'
  # Strict-FP32 score schema is verified rather than guessed.
  if name=='oncenet':prefix='scores__' if 'scores__'+TARGETS[0] in saved.files else 'score__'
  checks={t:dict(max_score_error=float(np.max(np.abs(actual[t]-saved[prefix+t][ii]))),primary_decision_changes=int(np.sum((actual[t]>cut(cal[t],.001))!=(saved[prefix+t][ii]>cut(cal[t],.001))))) for t in TARGETS}
  tol=.00002 if name=='oncenet' else .02
  report.append(dict(model=name,feature_max_absolute_errors=errors,feature_tolerance=tol,checks=checks,events=len(ii),passed=max(errors.values())<=tol and all(c['primary_decision_changes']==0 for c in checks.values())))
 save(R/'consumed_fixture_parity.json',dict(passed=all(r['passed'] for r in report),rows=report,used_event_ids=[r['event_id'] for r in fixture],reserved_inference=False))
 assert all(r['passed'] for r in report),'Numerical parity failed; diagnose without reserve scoring'
def run():
 p=read(R/'protocol.json');verify(p);review_ok(p)
 assert not(R/'scores_commitment.json').exists(),'Scores already completed; do not rerun'
 a=read(R/'acquisition.json');assert a['complete'] and a['protocol_sha256']==sha(R/'protocol.json')
 for f,h in a['files'].items():assert sha(f)==h
 parity(p)
 arrays=np.load(R/'inputs.npz',allow_pickle=False);all_scores={};frozen_cuts={};start=time.time()
 for name in MODELS:
  path=R/(name+'_scores.npz');binding=R/(name+'_binding.json')
  if binding.exists():
   b=read(binding);assert b['protocol_sha256']==sha(R/'protocol.json') and b['sha256']==sha(path);ss=dict(np.load(path,allow_pickle=False))
  else:
   ft=features(name,arrays,p);ss=scores(name,ft);np.savez(R/(name+'_features.npz'),**ft);np.savez(path,**ss)
   save(binding,dict(protocol_sha256=sha(R/'protocol.json'),sha256=sha(path),completed_utc=now()))
  all_scores[name]=ss;cal=calibration(name)
  frozen_cuts[name]={t:{str(alpha):dict(threshold=cut(cal[t],alpha),calibration_count=len(cal[t])) for alpha in ALPHAS} for t in TARGETS}
 assert frozen_cuts==read(R/'frozen_thresholds.json'),'Frozen thresholds changed'
 save(R/'scores_commitment.json',dict(complete=True,committed_utc=now(),protocol_sha256=sha(R/'protocol.json'),code_sha256=sha(__file__),label_metrics_not_yet_computed=True,files={str(R/(n+'_scores.npz')):sha(R/(n+'_scores.npz')) for n in MODELS},thresholds_sha256=sha(R/'frozen_thresholds.json'),elapsed_seconds=time.time()-start))
 print('ALL SCORES COMMITTED',flush=True)
 summarize()

def measure(y,s,threshold):
 from sklearn.metrics import roc_auc_score,roc_curve
 y=np.asarray(y,bool);s=np.asarray(s);pos=int(y.sum());neg=len(y)-pos;pred=s>(threshold if threshold is not None else np.inf)
 tp=int((pred&y).sum());fp=int((pred&~y).sum());tn=neg-fp;fn=pos-tp;den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
 out=dict(positive=pos,negative=neg,tp=tp,fp=fp,tn=tn,fn=fn,recall=tp/pos if pos else None,specificity=tn/neg if neg else None,balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,mcc=(tp*tn-fp*fn)/math.sqrt(den) if den else 0.,fa_per10000=10000*fp/neg if neg else None)
 if pos and neg:
  fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
  out.update(auroc=float(roc_auc_score(y,s)),pauc_std_001=float(roc_auc_score(y,s,max_fpr=.001)),pauc_std_01=float(roc_auc_score(y,s,max_fpr=.01)))
  for alpha in [.001,.01]:
   ix=np.where(fpr<=alpha)[0];j=ix[np.argmax(tpr[ix])];out['empirical_at_fpr_'+str(alpha)]=dict(recall=float(tpr[j]),achieved_fpr=float(fpr[j]),allowed_fpr=alpha)
 else:out.update(auroc=None,pauc_std_001=None,pauc_std_01=None)
 return out
def summarize():
 p=read(R/'protocol.json');verify(p);assert not(R/'results.json').exists(),'Completed results exist; no repeat'
 commitment=read(R/'scores_commitment.json');assert commitment['complete'] and commitment['protocol_sha256']==sha(R/'protocol.json')
 for f,h in commitment['files'].items():assert sha(f)==h
 assert commitment['thresholds_sha256']==sha(R/'frozen_thresholds.json')
 # First test-label join follows the immutable score commitment.
 rows=read(R/'records_unlabelled.json');labelmap={r['event_id']:r['label'] for r in read(p['labels_only_for_final_metrics'])['records']};labels=np.array([labelmap[r['event_id']] for r in rows]);cuts=read(R/'frozen_thresholds.json')
 all_s={m:dict(np.load(R/(m+'_scores.npz'),allow_pickle=False)) for m in MODELS};results=[];macro=[];union=[]
 from scipy.stats import beta
 for m in MODELS:
  for a in ALPHAS:
   rr=[];preds=[]
   for t in TARGETS:
    c=cuts[m][t][str(a)]['threshold'];z=measure(labels==t,all_s[m][t],c);preds.append(all_s[m][t]>c)
    n=z['negative'];fp=z['fp'];z['fpr_clopper_pearson_95']=[float(beta.ppf(.025,fp,n-fp+1)) if fp else 0.,float(beta.ppf(.975,fp+1,n-fp)) if fp<n else 1.]
    rr.append(dict(model=m,target=t,alpha=a,threshold=c,**z))
   results+=rr;fields=['balanced_accuracy','mcc','recall','fa_per10000','auroc','pauc_std_001','pauc_std_01']
   macro.append(dict(model=m,alpha=a,**{k:float(np.mean([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in fields},tp=sum(r['tp'] for r in rr),fp=sum(r['fp'] for r in rr)))
   bg=~np.isin(labels,TARGETS);anypred=np.logical_or.reduce(preds);union.append(dict(model=m,alpha=a,negative=int(bg.sum()),fp=int(anypred[bg].sum()),fa_per10000=float(anypred[bg].mean()*10000) if bg.any() else None))
 # Shared hour-detector block bootstrap, fixed scores and CAL.
 groups={}
 for i,r in enumerate(rows):groups.setdefault((r['ifo'],int(r['event_time']//3600)),[]).append(i)
 blocks=list(groups.values());rng=np.random.default_rng(51105);fields=['balanced_accuracy','mcc','recall','fa_per10000','pauc_std_001','pauc_std_01'];draws={b:{k:[] for k in fields} for b in ['random','dino']};valid=0
 for b in range(2000):
  ix=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))]);yy=labels[ix]
  if any(not 0<(yy==t).sum()<len(ix) for t in TARGETS):continue
  mm={}
  for m in MODELS:
   rr=[measure(yy==t,all_s[m][t][ix],cuts[m][t]['0.001']['threshold']) for t in TARGETS]
   mm[m]={k:float(np.mean([r[k] for r in rr])) for k in fields}
  for control in draws:
   for k in fields:draws[control][k].append(mm['oncenet'][k]-mm[control][k])
  valid+=1
  if b%100==0:save(R/'status.json',dict(stage='paired_intervals',draw=b,total=2000,updated_utc=now()))
 ci={c:{k:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975))) if v else None for k,v in dd.items()} for c,dd in draws.items()}
 result=dict(complete=True,completed_utc=now(),protocol_sha256=sha(R/'protocol.json'),score_commitment_sha256=sha(R/'scores_commitment.json'),code_sha256=sha(__file__),intent_count=195,eligible_count=len(rows),excluded=read(R/'acquisition.json')['excluded'],per_target=results,macro=macro,non_target_union=union,paired_primary_intervals=ci,bootstrap=dict(requested=2000,valid=valid,groups=len(blocks),unit=p['intervals']['unit']),limitations=p['limitations'],no_tuning=True,no_refit=True,no_training=True,confirmation_consumed=True)
 save(R/'results.json',result);save(R/'status.json',dict(stage='complete',complete=True,updated_utc=now()));print(json.dumps(macro),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','acquire','run','summarize']);args=ap.parse_args()
 try:globals()[args.action]()
 except Exception:
  save(R/'failure.json',dict(stage=args.action,utc=now(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
