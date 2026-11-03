"""V19 metadata-first COCO roster and post-freeze acquisition; no training code."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,hashlib,json,time,urllib.request
import numpy as np
from PIL import Image

EVAL=Path(__file__).resolve().parent;ROOT=EVAL.parent;WORKSPACE=ROOT.parents[1]
OLD=WORKSPACE/'outputs/hypernet_v17';DATA=EVAL/'data';PLAN=DATA/'plan.json'
SALT='HyperNet-v19-new-natural-scenes-v1:'
METHODS=('v19','v17','raw_template','owlvit_base_patch32')
OPERATING_POINTS={'primary_5pct':.05,'secondary_1pct':.01}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def verify(bindings):
    for p,h in bindings.items():
        if sha(p)!=h:raise ValueError('Bound file changed: '+str(p))
def pixel_sha(rgb):
    a=np.ascontiguousarray(rgb)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def rgb(path):
    with Image.open(path)as image:
        if image.getexif().get(274,1)!=1:raise ValueError('Unexpected EXIF orientation')
        return np.asarray(image.convert('RGB')).copy()
def annotations_for(records,targets):
    boxes={};valid={}
    for target in targets:
        cid=target['category_id'];a=[r for r in records if r['category_id']==cid];key=str(cid)
        boxes[key]=[[float(r['bbox'][0]),float(r['bbox'][1]),float(r['bbox'][0]+r['bbox'][2]),float(r['bbox'][1]+r['bbox'][3])]for r in a if not r['iscrowd']]
        valid[key]=not any(r['iscrowd']for r in a)
    return {'boxes_by_target':boxes,'valid_by_target':valid}
def choose(images,annotations,targets,excluded,positives=16,negatives=32):
    target_ids={t['category_id']for t in targets}
    pool={iid:r for iid,r in images.items()if iid not in excluded and not any(a['iscrowd']and a['category_id']in target_ids for a in annotations[iid])}
    used=set();chosen=[]
    for target in targets:
        cid=target['category_id']
        for present,count in((True,positives),(False,negatives)):
            role=f'{cid:03d}:'+('positive'if present else'negative');candidates=[]
            for iid,image in pool.items():
                if iid in used:continue
                records=[a for a in annotations[iid]if a['category_id']==cid]
                eligible=any(min(a['bbox'][2:])>=16 and a['bbox'][2]*a['bbox'][3]>=.01*image['width']*image['height']for a in records)if present else not records
                if eligible:candidates.append(iid)
            candidates.sort(key=lambda iid:hashlib.sha256((SALT+role+':'+str(iid)).encode()).hexdigest())
            if len(candidates)<count:raise ValueError('Not enough metadata candidates: '+role)
            for iid in candidates[:count]:used.add(iid);chosen.append((iid,cid,present))
    return sorted(chosen),len(pool)

def make_plan():
    if PLAN.exists():raise FileExistsError('Preserve the chosen roster')
    old_path=OLD/'data/coco_v17/plan.json';old_acq_path=OLD/'data/coco_v17/acquisition.json'
    v18_path=WORKSPACE/'outputs/hypernet_v18/data/real_dev_only.json'
    index_path=WORKSPACE/'outputs/hypernet_v14/data/coco_real_v14/sealed/source_annotation_index.json'
    prior_path=WORKSPACE/'outputs/hypernet_v14/data/coco_real_v14/prior_image_inventory.json'
    old=read(old_path);acq=read(old_acq_path);v18=read(v18_path);verify(old['input_bindings'])
    if not acq['completed']or acq['data_plan_sha256']!=sha(old_path)or v18['source_data_plan_sha256']!=sha(old_path):raise ValueError('Prior metadata bindings differ')
    if v18['references']!=old['references']or v18['dev']!=old['dev']:raise ValueError('V18 used new natural sources; extend inventory explicitly')
    excluded=set(old['prior_excluded_image_ids'])|{r['image_id']for r in old['references']+old['dev']+old['test']}
    if len(excluded)!=3648:raise ValueError('Expected all3648 earlier COCO IDs; review later image usage')
    index=read(index_path);annotations=defaultdict(list)
    for row in index['annotations']:annotations[row['image_id']].append(row)
    images={r['id']:r for r in index['images']};licenses={r['id']:r for r in index['licenses']};targets=old['targets']
    selected,pool=choose(images,annotations,targets,excluded);test=[]
    for iid,cid,present in selected:
        im=images[iid];lic=licenses[im['license']]
        test.append({'id':f'coco_v19_{iid:012d}','image_id':iid,'split':'test','width':im['width'],'height':im['height'],'rgb_path':f'images/{iid:012d}.jpg',
          'selection_category_id':cid,'selection_present':present,'source':{'dataset':'COCO2017','official_source_split':'train2017','file_name':im['file_name'],
          'coco_url':im['coco_url'],'flickr_url':im.get('flickr_url'),'date_captured':im.get('date_captured'),'license_id':im['license'],'license_name':lic['name'],'license_url':lic['url']},
          **annotations_for(annotations[iid],targets)})
    if len(test)!=576 or len({r['image_id']for r in test})!=576 or set(r['image_id']for r in test)&excluded:raise ValueError('Fresh roster dimensions/overlap differ')
    counts={}
    for split,rows in(('dev',old['dev']),('test',test)):
        counts[split]={str(t['category_id']):{'positive':sum(bool(r['boxes_by_target'][str(t['category_id'])])and r['valid_by_target'][str(t['category_id'])]for r in rows),
             'negative':sum(not r['boxes_by_target'][str(t['category_id'])]and r['valid_by_target'][str(t['category_id'])]for r in rows),
             'ignored_crowd':sum(not r['valid_by_target'][str(t['category_id'])]for r in rows)}for t in targets}
    bindings={str(p.resolve()):sha(p)for p in(old_path,old_acq_path,v18_path,index_path,prior_path,EVAL/'vendor_manifest.json')}
    plan={'format':'hypernet.v19.coco.metadata_plan.v1','created_utc':datetime.now(timezone.utc).isoformat(),'targets':targets,'references':old['references'],'dev':old['dev'],'test':test,
      'input_bindings':bindings,'selection_source_sha256':sha(__file__),'prior_excluded_image_ids':sorted(excluded),'per_target_counts':counts,
      'selection':{'salt':SALT,'pool_photos':pool,'positive_sources_per_category':16,'negative_sources_per_category':32,'test_pixels_acquired':False,'model_scores_used':False,
        'test_crowd_rule':'Exclude whole candidate photo if any of12 target categories has crowd annotations.',
        'positive_stratum':'At least one noncrowd target box min edge16px and area>=1%; score against all boxes including smaller incidental objects.',
        'reference_crop':'Same one v17 boxed support per target; floor/ceil clipped crop; no query annotation is supplied to models.'},
      'protocol':{'methods':list(METHODS),'operating_points':OPERATING_POINTS,'primary':'Macro top1 localized recall at IoU>=.5, with one global cutoff from all valid DEV absent pairs.',
        'all_targets_on_all_photos':True,'dev_photos':288,'fresh_test_photos':576,'test_pairs':6912,'bootstrap':'5000 paired photo-cluster draws; fixed supports and DEV thresholds.',
        'metric_is_standard_coco_AP':False,'one_support_per_type':True,'natural_weight_training_images':0,'borrowed_hypernet_pretraining':False},
      'limitations':old['limitations']+['V17 fresh TEST photos are now excluded from this new cohort, along with every earlier COCO source ID.','V18 used only48 of the already-known288 DEV photos; its exploratory thresholds are not reused.']}
    write_new(PLAN,plan)
    print(json.dumps({'plan_sha256':sha(PLAN),'excluded_prior_ids':len(excluded),'fresh_photos':576,'test_positive_pairs':sum(x['positive']for x in counts['test'].values()),
      'test_negative_pairs':sum(x['negative']for x in counts['test'].values()),'per_target_counts':counts}),flush=True)

def load_plan():
    plan=read(PLAN);verify(plan['input_bindings'])
    for name,record in read(EVAL/'vendor_manifest.json').items():
        if sha(EVAL/'vendor_v17'/name)!=record['sha256']:raise ValueError('Vendored baseline changed: '+name)
    return plan
def load_old(row):
    path=WORKSPACE/row['workspace_path']
    if sha(path)!=row['sha256']:raise ValueError('Old support/DEV bytes changed')
    a=rgb(path)
    if pixel_sha(a)!=row['pixel_sha256']:raise ValueError('Old support/DEV pixels changed')
    return a
def load_reference(row):
    a=load_old(row);x1,y1,x2,y2=row['bbox_xyxy']
    return a[max(0,int(np.floor(y1))):min(a.shape[0],int(np.ceil(y2))),max(0,int(np.floor(x1))):min(a.shape[1],int(np.ceil(x2)))].copy()
def load_test(row,acquisition):
    record=acquisition['records'][str(row['image_id'])];path=DATA/row['rgb_path']
    if sha(path)!=record['sha256']:raise ValueError('TEST bytes changed')
    a=rgb(path)
    if pixel_sha(a)!=record['pixel_sha256']:raise ValueError('TEST pixels changed')
    return a

def verify_protocol(path,expected):
    path=Path(path).resolve()
    if sha(path)!=expected:raise ValueError('Frozen protocol hash differs')
    value=read(path)
    if value.get('model_selection_complete')is not True or value.get('test_scores_computed_before_freeze')is not False:raise ValueError('Model and DEV calibration must be frozen')
    if value['data_plan_sha256']!=sha(PLAN):raise ValueError('Frozen protocol uses another cohort')
    if sorted(value['methods'])!=sorted(METHODS):raise ValueError('Frozen comparator roster differs')
    verify(value['behavior_files']);verify(value['checkpoint_files']);return value

def fetch(path,expected,workers):
    plan=load_plan();protocol=verify_protocol(path,expected);gate=DATA/'acquisition_admission.json'
    if (DATA/'acquisition.json').exists():raise FileExistsError('Preserve completed acquisition')
    if gate.exists():
        if read(gate)['protocol_sha256']!=expected:raise ValueError('Existing acquisition belongs to another freeze')
    else:
        if (DATA/'images').exists()and any((DATA/'images').iterdir()):raise ValueError('TEST pixels existed before freeze')
        write_new(gate,{'protocol_path':str(Path(path).resolve()),'protocol_sha256':expected,'data_plan_sha256':sha(PLAN),'created_utc':datetime.now(timezone.utc).isoformat(),'before_first_test_image_write':True})
    (DATA/'images').mkdir(exist_ok=True);(DATA/'records').mkdir(exist_ok=True)
    def one(row):
        destination=DATA/row['rgb_path'];record_path=DATA/'records'/f"{row['image_id']:012d}.json"
        if record_path.exists():
            record=read(record_path)
            if sha(destination)!=record['sha256']or pixel_sha(rgb(destination))!=record['pixel_sha256']:raise ValueError('Cached TEST changed')
            return record
        if destination.exists():raise ValueError('Uncommitted image; do not overwrite')
        url='https://s3.amazonaws.com/images.cocodataset.org/train2017/'+row['source']['file_name']
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=45)as response:payload=response.read()
                break
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
        with destination.open('xb')as f:f.write(payload)
        a=rgb(destination)
        if a.shape!=(row['height'],row['width'],3):raise ValueError('TEST dimensions differ')
        record={'image_id':row['image_id'],'sha256':sha(destination),'pixel_sha256':pixel_sha(a),'transport_url':url};write_new(record_path,record);return record
    records={};failures=[]
    with ThreadPoolExecutor(max_workers=workers)as pool:
        futures={pool.submit(one,row):row['image_id']for row in plan['test']}
        for future in as_completed(futures):
            try:records[str(futures[future])]=future.result()
            except Exception as error:failures.append({'image_id':futures[future],'error':str(error)})
    if failures:raise RuntimeError(json.dumps(failures))
    old=read(OLD/'data/coco_v17/plan.json');old_acq=read(OLD/'data/coco_v17/acquisition.json')
    prior=read(WORKSPACE/'outputs/hypernet_v14/data/coco_real_v14/prior_image_inventory.json')
    previous=prior['files']+old['references']+old['dev']+list(old_acq['records'].values())
    raw_hashes={r['sha256']for r in previous};pixel_hashes={r['pixel_sha256']for r in previous}
    raw_groups=defaultdict(list);pixel_groups=defaultdict(list)
    for iid,record in records.items():raw_groups[record['sha256']].append(iid);pixel_groups[record['pixel_sha256']].append(iid)
    excluded=[]
    for row in plan['test']:
        record=records[str(row['image_id'])];reasons=[]
        if record['sha256']in raw_hashes or record['pixel_sha256']in pixel_hashes:reasons.append('exact_prior_pixels')
        if len(raw_groups[record['sha256']])>1 or len(pixel_groups[record['pixel_sha256']])>1:reasons.append('within_test_duplicate')
        if reasons:excluded.append({'image_id':row['image_id'],'reasons':reasons})
    removed={r['image_id']for r in excluded};verify_protocol(path,expected)
    result={'completed':True,'data_plan_sha256':sha(PLAN),'protocol_path':str(Path(path).resolve()),'protocol_sha256':expected,'acquisition_admission_sha256':sha(gate),
      'requested':576,'downloaded':len(records),'eligible_image_ids':[r['image_id']for r in plan['test']if r['image_id']not in removed],'excluded':excluded,'records':records,'failures':[],
      'replacement_sampling':False,'model_scores_used_for_quality':False}
    write_new(DATA/'acquisition.json',result);print(json.dumps({'eligible':len(result['eligible_image_ids']),'excluded':excluded,'acquisition_sha256':sha(DATA/'acquisition.json')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['plan','fetch-test']);p.add_argument('--protocol',type=Path);p.add_argument('--expected-protocol-sha');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.stage=='plan':make_plan()
    else:
        if not a.protocol or not a.expected_protocol_sha or not 1<=a.workers<=8:p.error('Frozen protocol/hash and workers1..8 required')
        fetch(a.protocol,a.expected_protocol_sha,a.workers)
