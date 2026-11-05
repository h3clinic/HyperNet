"""Acquire authentic alternate windows of consumed events only; no inference.

All event roles/reference IDs are inherited from v44. Missing windows are never
fabricated from a different duration. Complete-case eligibility is based only on
download/integrity/renderer checks, before any model outcomes.
"""
import argparse,ast,hashlib,importlib.util,io,json,math,os,threading,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import requests
from PIL import Image
R=Path(__file__).resolve().parent;O=R/'data';W=R.parents[1];V44=W/'outputs/hypernet_v44';PREP=W/'outputs/hypernet_v41/preprocess.py'
WINDOWS={'d05':(.5,'url1'),'d1':(1.,'url2'),'d2':(2.,'url3'),'d4':(4.,'url4')}
REFS={'Low_Frequency_Burst':'632xGf8n9x','Low_Frequency_Lines':'Vengh4mj2t','Scattered_Light':'dac0D6ni5R','Blip':'yxumvoPABn'}
MAPPING_SOURCE='https://raw.githubusercontent.com/gwpy/gwpy/c42f7281b1a2ca6098c1e2f7d6921a11b17f2e56/gwpy/table/gravityspy.py'
LOCAL=threading.local();_prep=None

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
 for k in range(30):
  try:os.replace(tmp,p);return
  except PermissionError:
   if k==29:raise
   time.sleep(.1)
def load_prep():
 global _prep
 if _prep is None:
  s=importlib.util.spec_from_file_location('duration_frozen_preprocessor',PREP);_prep=importlib.util.module_from_spec(s);s.loader.exec_module(_prep)
 return _prep
def fit_indices(rows,target):
 labels=sorted({r['label'] for r in rows if r['split']=='fit' and r['label']!=target},key=lambda l:hashlib.sha256(('v42-label|'+l).encode()).hexdigest());pool=[]
 for rank,label in enumerate(labels):
  group=[i for i,r in enumerate(rows) if r['split']=='fit' and r['label']==label];group.sort(key=lambda i:hashlib.sha256(('v42-fit|'+rows[i]['group']+'|'+rows[i]['event_id']).encode()).hexdigest());pool.extend((j,rank,i) for j,i in enumerate(group))
 return [i for _,_,i in sorted(pool)[:512]]

def freeze():
 assert not(O/'plan.json').exists(),'Already frozen'
 inventory=read(O/'metadata_inventory.json');rows=inventory['rows'];original=read(V44/'used_records.json');assert len(rows)==len(original)==8251 and not inventory['missing'] and not inventory['conflicting_urlsets']
 assert inventory['used_records_sha256']==sha(V44/'used_records.json') and [r['event_id'] for r in rows]==[r['event_id'] for r in original]
 reserve=read(W/'outputs/hypernet_v45/reserve_metadata_audit.json')['records'];assert not {r['event_id'] for r in rows}&{r['event_id'] for r in reserve}
 for r in reserve:assert all(r['ifo']!=u['ifo'] or abs(r['event_time']-u['event_time'])>8 for u in rows)
 for r in rows:
  assert r['original_url_field']=='url2' and r['image_url']==r['catalog']['url2'] and r['original_window_seconds']==1.
  assert r['catalog']['ifo']==r['ifo'] and abs(r['catalog']['event_time']-r['event_time'])<1e-4
  assert all(r['catalog'][key].startswith('https://panoptes-uploads.zooniverse.org/production/subject_location/') for _,key in WINDOWS.values())
 assert all(sum(r['event_id']==key for r in rows)==1 for key in REFS.values())
 response=requests.get(MAPPING_SOURCE,timeout=30);response.raise_for_status();parsed=ast.parse(response.text);durations=next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DURATIONS' for t in n.targets));assert durations==(.5,1.,2.,4.)
 (O/'gwpy_duration_mapping.py').write_bytes(response.content)
 files=[Path(p) for p in inventory['source_paths']]+[V44/'used_records.json',O/'metadata_inventory.json',O/'gwpy_duration_mapping.py',PREP,W/'outputs/hypernet_v45/reserve_metadata_audit.json',V44/'features/used/inputs_binding.json']
 files += [V44/'features/used'/f'{k}128.npy' for k in ['full','low']]
 fit={t:fit_indices(original,t) for t in REFS};assert all(len(v)==512 for v in fit.values())
 size=sum(Path(r['raw_path']).stat().st_size for r in rows)
 save(O/'plan.json',dict(frozen_epoch=time.time(),code_sha256=sha(__file__),source_hashes={str(p):sha(p) for p in files},used_records_sha256=sha(V44/'used_records.json'),inventory_sha256=sha(O/'metadata_inventory.json'),windows={k:dict(seconds=v[0],url_field=v[1]) for k,v in WINDOWS.items()},references=REFS,original_fit_indices=fit,original_fit_union_event_ids=sorted({original[i]['event_id'] for ids in fit.values() for i in ids}),
  original_count=8251,roles=dict(Counter(r['split'] for r in rows)),raw_original_bytes=size,estimated_alternate_bytes=3*size,estimated_requests=3*len(rows),
  mapping_authority=MAPPING_SOURCE,mapping_note='GWpy DURATIONS ordered0.5/1/2/4; urlN maps DURATIONS[N-1]. Omicron metadata duration is not PNG window length.',
  population='Only the exact8251 consumed v44 used events. No additional labeled event, no reserved195 image and no v44 fresh event.',
  reference_policy='One labeled reference EVENT per target, now four genuine PNG time windows. More input context than one PNG; not four independent positive events.',
  eligibility='All four actual PNGs must pass URL/provenance/hash/800x600/legend-renderer checks. Exclude incomplete events from every arm; do not use label, heatmap appearance or model scores. Preserve all originalFIT512 members and all4references; otherwise stop for an explicit restricted-cohort protocol.',
  input_contract='records.json compact original order; source_index retains v44 index; duration_sources[d05/d1/d2/d4]={path,sha256,url,duration_seconds,pixel_sha256,crop_pixel_sha256,renderer}. full_KEY.npy and low_KEY.npy arefloat32 Nx128x128. d1 copied bit-exact from bound old cache; others exact v41 preprocessor.',
  network=dict(workers=12,attempts=3,connect_timeout=10,read_timeout=45,max_source_bytes=10000000),no_scores=True,no_new_real_events=True,no_interpolation_of_missing_duration=True))
 print('DURATION PLAN',sha(O/'plan.json'),flush=True)

def check_plan():
 p=read(O/'plan.json');assert p['code_sha256']==sha(__file__)
 for path,h in p['source_hashes'].items():assert sha(path)==h,path
 return p,read(O/'metadata_inventory.json')['rows']

def validate(path,row,key):
 with Image.open(path) as im:
  assert im.format=='PNG' and im.size==(800,600),'Not a native800x600 PNG';raw=np.asarray(im.convert('RGB'))
 prep=load_prep();renderer=prep.validate_renderer(raw);crop=prep.native_rgb(raw,'full')
 return dict(duration_seconds=WINDOWS[key][0],url_field=WINDOWS[key][1],url=row['catalog'][WINDOWS[key][1]],path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size,pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),crop_pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),renderer=renderer,event_id=row['event_id'],ifo=row['ifo'],event_time=row['event_time'])

def acquire_one(row,key):
 event=row['event_id'];folder=O/'raw'/key;folder.mkdir(parents=True,exist_ok=True);path=Path(row['raw_path']) if key=='d1' else folder/(event+'.png');binding=O/'source_bindings'/key/(event+'.json')
 if binding.exists():
  b=read(binding);assert b['event_id']==event and b['url']==row['catalog'][WINDOWS[key][1]] and b['sha256']==sha(path);return b
 if key=='d1':
  assert sha(path)==row['raw_sha256'];b=validate(path,row,key);b['reused_original_1s']=True;save(binding,b);return b
 if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session();LOCAL.session.headers['User-Agent']='HyperNet-research-duration-validation/1.0'
 error=None
 for attempt in range(3):
  try:
   if not path.exists():
    response=LOCAL.session.get(row['catalog'][WINDOWS[key][1]],timeout=(10,45));response.raise_for_status();assert len(response.content)<=10000000
    temp=path.with_suffix('.png.tmp');temp.write_bytes(response.content);b=validate(temp,row,key);os.replace(temp,path)
   b=validate(path,row,key);b['attempts']=attempt+1;b['reused_original_1s']=False;save(binding,b);return b
  except Exception as e:
   error=f'{type(e).__name__}: {e}'
   if attempt<2:time.sleep(1+attempt)
 raise RuntimeError(error)

def download(rows,stage):
 start=time.time();load_prep();jobs=[(r,k) for r in rows for k in WINDOWS];success=[];failures=[];cached=sum((O/'source_bindings'/k/(r['event_id']+'.json')).exists() for r,k in jobs);bytecount=0
 with ThreadPoolExecutor(max_workers=12) as pool:
  pending={pool.submit(acquire_one,r,k):(r,k) for r,k in jobs}
  for i,f in enumerate(as_completed(pending),1):
   r,k=pending[f]
   try:b=f.result();success.append(b);bytecount+=b['bytes']
   except Exception as e:failures.append(dict(event_id=r['event_id'],source_index=r['index'],duration=k,error=str(e)))
   if i%100==0 or i==len(jobs):
    elapsed=time.time()-start;save(O/'status.json',dict(stage=stage,pid=os.getpid(),completed_sources=i,total_sources=len(jobs),cached_before=cached,validated_source_bytes=bytecount,failures=len(failures),seconds=elapsed,eta_seconds=elapsed*(len(jobs)-i)/max(i,1)));print(stage,i,len(jobs),'failures',len(failures),'seconds',round(elapsed,1),flush=True)
 return success,failures,time.time()-start

def probe():
 p,rows=check_plan();assert not(O/'probe.json').exists(),'Probe already completed'
 chosen=[r for r in rows if r['event_id'] in REFS.values()];chosen += sorted([r for r in rows if r['event_id'] not in REFS.values()],key=lambda r:hashlib.sha256(('v46-duration-probe|'+r['event_id']).encode()).hexdigest())[:28]
 success,failed,elapsed=download(chosen,'probe');new=[x for x in success if not x['reused_original_1s']]
 save(O/'probe.json',dict(plan_sha256=sha(O/'plan.json'),metadata_selected_before_download=True,event_ids=[r['event_id'] for r in chosen],sources=len(success),failures=failed,downloaded_alternate_sources=len(new),downloaded_alternate_bytes=sum(x['bytes'] for x in new),seconds=elapsed,estimated_total_alternate_bytes=float(np.mean([x['bytes'] for x in new])*3*len(rows)) if new else None,rough_total_seconds=elapsed*len(rows)/len(chosen),no_scores=True))
 print('PROBE COMPLETE',json.dumps(read(O/'probe.json')),flush=True)

def acquire():
 p,rows=check_plan();assert (O/'probe.json').exists();assert not(O/'records.json').exists(),'Common cohort already finalized'
 success,failures,elapsed=download(rows,'acquiring');by={(r['event_id'],next(k for k,v in WINDOWS.items() if v[0]==r['duration_seconds'])):r for r in success};complete=[];excluded=[]
 for row in rows:
  missing=[k for k in WINDOWS if (row['event_id'],k) not in by]
  if missing:excluded.append(dict(event_id=row['event_id'],source_index=row['index'],split=row['split'],missing=missing));continue
  r=dict(row);r.pop('catalog');r['source_index']=row['index'];r['index']=len(complete);r['duration_sources']={k:by[(row['event_id'],k)] for k in WINDOWS};complete.append(r)
 save(O/'download_failures.json',failures);save(O/'excluded_events.json',excluded)
 ids={r['event_id'] for r in complete};lost_fit=sorted(set(p['original_fit_union_event_ids'])-ids);lost_refs=sorted(set(REFS.values())-ids);calcounts={t:sum(r['split']=='calibration' and r['label']!=t for r in complete) for t in REFS}
 status=dict(plan_sha256=sha(O/'plan.json'),complete_events=len(complete),excluded_events=len(excluded),failed_sources=len(failures),roles=dict(Counter(r['split'] for r in complete)),calibration_negative_counts=calcounts,lost_original_fit=lost_fit,lost_references=lost_refs,seconds=elapsed,no_scores=True)
 save(O/'acquisition_audit.json',status)
 assert not lost_fit and not lost_refs,'Required original FIT/reference missing; no cohort substitution allowed'
 assert min(calcounts.values())>=2000,'Insufficient negative calibration for declared settings'
 save(O/'records.json',complete);status.update(passed=True,records_sha256=sha(O/'records.json'));save(O/'acquisition_audit.json',status)
 print('ACQUISITION COMPLETE',json.dumps({k:v for k,v in status.items() if k not in ['lost_original_fit','lost_references']}),flush=True)

def process_event(row):
 prep=load_prep();out={};pixels=[]
 for key in WINDOWS:
  b=row['duration_sources'][key];assert sha(b['path'])==b['sha256']
  if key=='d1':continue
  with Image.open(b['path']) as im:raw=np.asarray(im.convert('RGB'))
  # Same original palette inversion before AREA resize. Windows are not stretched
  # into substitutes for unavailable durations; each has its own authentic PNG.
  out[key]=(prep.preprocess(raw,'energy_full',128)[...,0],prep.preprocess(raw,'energy_low',128)[...,0])
 return row['index'],out

def prepare():
 p,_=check_plan();assert not(O/'inputs_binding.json').exists(),'Inputs already frozen';rows=read(O/'records.json');acq=read(O/'acquisition_audit.json');assert acq['passed'] and acq['records_sha256']==sha(O/'records.json')
 input_binding=read(V44/'features/used/inputs_binding.json');assert input_binding['rows_sha256']==sha(V44/'used_records.json') and input_binding['preprocess_sha256']==sha(PREP)
 original={kind:np.load(V44/'features/used'/(kind+'128.npy'),mmap_mode='r') for kind in ['full','low']}
 arr={kind+'_'+key:np.lib.format.open_memmap(O/(kind+'_'+key+'.npy'),mode='w+',dtype=np.float32,shape=(len(rows),128,128)) for key in WINDOWS for kind in ['full','low']};start=time.time()
 for i,row in enumerate(rows):
  for kind in ['full','low']:arr[kind+'_d1'][i]=original[kind][row['source_index']]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for count,(i,data) in enumerate(pool.map(process_event,rows,chunksize=8),1):
   for key,(full,low) in data.items():arr['full_'+key][i]=full;arr['low_'+key][i]=low
   if count%250==0 or count==len(rows):save(O/'status.json',dict(stage='preparing_inputs',pid=os.getpid(),events=count,total=len(rows),seconds=time.time()-start));print('PREPARE',count,len(rows),flush=True)
 for a in arr.values():a.flush();assert np.isfinite(a).all() and float(a.min())>=0 and float(a.max())<=1
 for kind in ['full','low']:
  for i,r in enumerate(rows):assert np.array_equal(arr[kind+'_d1'][i],original[kind][r['source_index']])
 files={key:sha(O/(key+'.npy')) for key in arr};save(O/'inputs_binding.json',dict(plan_sha256=sha(O/'plan.json'),records_sha256=sha(O/'records.json'),preprocess_sha256=sha(PREP),code_sha256=sha(__file__),shape=[len(rows),128,128],dtype='float32',files=files,original_one_second_bit_exact=True,one_reference_event_four_rendered_windows=True,no_new_events=True))
 # Source/pixel equality audit across event roles; report and stop on direct
 # cross-role pixel reuse, without class/outcome-based exemptions.
 seen={};collisions=[];within_event=[];same_role=[];input_hashes=[]
 def record_hash(kind,digest,r,key):
  token=(kind,digest);old=seen.get(token);current=dict(event_id=r['event_id'],split=r['split'],duration=key)
  if old:
   if old['event_id']==r['event_id']:within_event.append(dict(kind=kind,first=old,second=current))
   elif old['split']!=r['split']:collisions.append(dict(kind=kind,first=old,second=current))
   else:same_role.append(dict(kind=kind,first=old,second=current))
  else:seen[token]=current
 for r in rows:
  hashes={}
  for key,b in r['duration_sources'].items():
   for kind in ['pixel_sha256','crop_pixel_sha256']:
    record_hash(kind,b[kind],r,key)
   for kind in ['full','low']:
    digest=hashlib.sha256(np.ascontiguousarray(arr[kind+'_'+key][r['index']]).tobytes()).hexdigest();hashes[kind+'_'+key]=digest;record_hash(kind+'_128',digest,r,key)
  combined=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();record_hash('combined_all_eight_inputs',combined,r,'all');input_hashes.append(dict(event_id=r['event_id'],index=r['index'],source_index=r['source_index'],hashes=hashes,combined_sha256=combined))
 save(O/'input_pixel_hashes.json',input_hashes)
 save(O/'audit.json',dict(passed=not collisions,records_sha256=sha(O/'records.json'),inputs_binding_sha256=sha(O/'inputs_binding.json'),input_pixel_hashes_sha256=sha(O/'input_pixel_hashes.json'),acquisition_audit_sha256=sha(O/'acquisition_audit.json'),complete_events=len(rows),reference_ids=REFS,original_one_second_bit_exact=True,cross_role_pixel_collisions=collisions,same_role_cross_event_pixel_duplicates=same_role,within_event_cross_duration_duplicates=within_event,collision_policy='Flag every native/crop/input collision. Constant single-view low-frequency crops are not automatically whole-event leakage; any cross-role collision requires review before inference. No rows silently removed.',no_reserved_events=True,no_new_real_events=True,one_reference_event_four_windows=True,no_model_scores=True,seconds=time.time()-start))
 assert not collisions,'Cross-role exact source/crop pixel collision; do not score without review'
 save(O/'status.json',dict(stage='complete',events=len(rows),seconds=time.time()-start));print('DURATION INPUTS COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','probe','acquire','prepare']);args=parser.parse_args();globals()[args.action]()
