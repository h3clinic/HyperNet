"""All planned synthetic scenes, exact source binding, deterministic ordered CPU preflight."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib,io,json,multiprocessing,sys,time,traceback
import cv2
import numpy as np

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'outputs/hypernet_v19'
DEST=ROOT/'experiments/full_preflight_v2'
sys.path.insert(0,str(ROOT))
from hypernet_v19 import curriculum as c
SPLIT_NAMES=('train','dev','test','compositional')
SPLIT_COUNTS=(32768,256,256,128)
SOURCE_NAMES=('hypernet_v19/__init__.py','hypernet_v19/curriculum.py')
ARRAY_NAMES=('support_mask','query_target_mask','query_object_mask','boxes','target_box','support_points','query_points','support_point_valid','query_point_valid','point_valid')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest_bytes(value):return np.frombuffer(bytes.fromhex(value),np.uint8).copy()
def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
def atomic_bytes(path,data):
    temp=path.with_name(path.name+'.writing')
    with temp.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    temp.replace(path)
def atomic_json(path,value):atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
def now():return datetime.now(timezone.utc).isoformat()
def init_worker(hashes):
    cv2.setNumThreads(1)
    for name,expected in hashes.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen source differs in worker: '+name)

def evaluate(task):
    split_number,index=task;split=SPLIT_NAMES[split_number]
    base={'split_code':np.int8(split_number),'index':np.int64(index)}
    try:
        row=c.episode(index,split,1901,1.);m=row['metadata'];positive=bool(row['positive'])
        assert m['natural_images']==0 and m['borrowed_pretrained_weights']is False
        assert m['index']==index and m['split']==split and m['seed']==1901
        assert row['program_id']==c.digest(c.program(index,split,1901)), 'Retry changed program'
        assert positive==(index%2==0) and positive==bool(row['query_target_mask'].sum()>0)
        assert np.array_equal(row['point_valid'],row['support_point_valid']&row['query_point_valid'])
        assert np.all(row['query_target_mask']<=row['query_object_mask']+1e-6)
        assert min(m['visible_edit_pixels'])>=8 and min(m['support_visible_edit_pixels'])>=8
        if positive:assert m['target_visibility']>=.65 and int(row['point_valid'].sum())>=16
        assert len(set(i['program_id']for i in m['instances']))==4
        assert (row['program_id']in[i['program_id']for i in m['instances']])==positive
        for key in ('support','query'):
            image=np.asarray(row[key]);assert image.dtype==np.uint8
            assert hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()==m['image_sha256'][key]
            base['raw_'+key+'_sha256']=digest_bytes(m['image_sha256'][key])
        for key in ARRAY_NAMES:
            assert np.isfinite(row[key]).all()
            base['generated_'+key+'_sha256']=digest_bytes(array_hash(row[key]))
        for key in ('program_id','near_program_id','replacement_program_id'):base[key]=digest_bytes(m[key])
        base['instance_program_ids']=np.stack([digest_bytes(i['program_id'])for i in m['instances']])
        base['metadata_sha256']=digest_bytes(hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
        base.update(valid=np.bool_(True),positive=np.bool_(positive),family=np.int8(m['program']['family']),nuisance_attempt=np.int16(m['nuisance_attempt']),valid_correspondences=np.int16(row['point_valid'].sum()),
                    visibility=np.float32(m['target_visibility']if positive else -1.),support_edit_pixels=np.asarray(m['support_visible_edit_pixels'],np.int32),query_edit_pixels=np.asarray(m['visible_edit_pixels'],np.int32))
        return base,None,{'near':m['edit']['operation'],'replacement':m['replacement_edit']['operation']}
    except Exception as error:
        return base,{'split':split,'index':index,'seed':1901,'difficulty':1.,'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(limit=4)},None

def main():
    if DEST.exists():raise FileExistsError('Preserve existing preflight: '+str(DEST))
    DEST.mkdir(parents=True);(DEST/'source_snapshot').mkdir();(DEST/'shards').mkdir()
    hashes={name:sha(ROOT/name)for name in SOURCE_NAMES}
    for name,expected in hashes.items():
        path=DEST/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);atomic_bytes(path,(ROOT/name).read_bytes());assert sha(path)==expected
    atomic_bytes(DEST/'source_snapshot/preflight.py',Path(__file__).read_bytes())
    plan={'format':'hypernet.v19.full_renderer_preflight.v2','created_utc':now(),'splits':dict(zip(SPLIT_NAMES,SPLIT_COUNTS)),'planned_episodes':sum(SPLIT_COUNTS),'seed':1901,'difficulty':1.,'workers':8,'chunksize':8,'ordered':True,'source_sha256':hashes,'script_sha256':sha(__file__),'natural_images':0,'gpu_used':False,'model_training':False,'scope':'Actual renderer and its visibility guards for every planned synthetic index. No index/program replacement and no independent full-scene rerender; raw image hashes, program IDs and metadata hashes bind the realized scenes for later training-consumption comparison.'}
    atomic_json(DEST/'plan.json',plan)
    tasks=[(code,index)for code,count in enumerate(SPLIT_COUNTS)for index in range(count)]
    errors=[];all_records=[];pending=[];shards=[];retry=Counter();near_ops=Counter();replacement_ops=Counter();by_split={s:{'completed':0,'valid':0,'errors':0}for s in SPLIT_NAMES}
    start=time.perf_counter();min_support=10**9;min_query=10**9;min_visibility=1.;min_points=64;completed=0
    def progress(status):
        elapsed=time.perf_counter()-start
        return {'status':status,'updated_utc':now(),'completed':completed,'completed_episodes':completed,'planned_episodes':len(tasks),'valid_count':len(all_records),'error_count':len(errors),'last_split':SPLIT_NAMES[tasks[completed-1][0]]if completed else None,'last_index':tasks[completed-1][1]if completed else None,'seconds':elapsed,'estimated_remaining_seconds':elapsed/completed*(len(tasks)-completed)if completed else None,'by_split':by_split,'retry_histogram':dict(retry),'min_support_edit_pixels':min_support if all_records else None,'min_query_edit_pixels':min_query if all_records else None,'min_positive_visibility':min_visibility if all_records else None,'min_positive_valid_points':min_points if all_records else None,'source_sha256':hashes,'natural_images':0,'gpu_used':False}
    atomic_json(DEST/'progress.json',progress('running'))
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(hashes,))as pool:
        for task,(record,error,ops)in zip(tasks,pool.map(evaluate,tasks,chunksize=8)):
            code,index=task;assert int(record['index'])==index and int(record['split_code'])==code
            completed+=1;split=SPLIT_NAMES[code];by_split[split]['completed']+=1
            if error:
                errors.append(error);by_split[split]['errors']+=1
                atomic_json(DEST/'errors.json',errors)
                print(json.dumps({'error':error,'completed':completed}),flush=True)
            else:
                all_records.append(record);pending.append(record);by_split[split]['valid']+=1
                retry[int(record['nuisance_attempt'])]+=1;near_ops[ops['near']]+=1;replacement_ops[ops['replacement']]+=1
                min_support=min(min_support,int(record['support_edit_pixels'].min()));min_query=min(min_query,int(record['query_edit_pixels'].min()))
                if bool(record['positive']):min_visibility=min(min_visibility,float(record['visibility']));min_points=min(min_points,int(record['valid_correspondences']))
            if completed%256==0 or completed==len(tasks):
                for name,expected in hashes.items():
                    if sha(ROOT/name)!=expected:raise RuntimeError('Frozen source changed during preflight: '+name)
                if pending:
                    arrays={key:np.stack([r[key]for r in pending])for key in pending[0]};buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
                    name=f'{completed-len(pending):06d}_{completed:06d}.npz';path=DEST/'shards'/name;atomic_bytes(path,buffer.getvalue());shards.append({'file':name,'sha256':sha(path),'valid_rows':len(pending)});pending=[]
                p=progress('running');atomic_json(DEST/'progress.json',p);print(json.dumps({k:p[k]for k in ('completed','planned_episodes','error_count','seconds','estimated_remaining_seconds')}),flush=True)
    atomic_json(DEST/'errors.json',errors)
    arrays={key:np.stack([r[key]for r in all_records])for key in all_records[0]};buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);atomic_bytes(DEST/'episode_hashes.npz',buffer.getvalue())
    programs={s:set()for s in SPLIT_NAMES};images={s:set()for s in SPLIT_NAMES}
    for record in all_records:
        name=SPLIT_NAMES[int(record['split_code'])];programs[name].add(record['program_id'].tobytes());images[name].update([record['raw_support_sha256'].tobytes(),record['raw_query_sha256'].tobytes()])
    collisions=[]
    for i,a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i+1:]:
            if programs[a]&programs[b]or images[a]&images[b]:collisions.append({'splits':[a,b],'program_collisions':len(programs[a]&programs[b]),'rgb_collisions':len(images[a]&images[b])})
    result={**plan,**progress('passed'if not errors and not collisions else'failed'),'errors':errors,'split_collisions':collisions,'near_edit_operation_counts':dict(near_ops),'replacement_edit_operation_counts':dict(replacement_ops),'unique_target_programs':{s:len(v)for s,v in programs.items()},'unique_raw_rgb_by_split':{s:len(v)for s,v in images.items()},'episode_hashes_sha256':sha(DEST/'episode_hashes.npz'),'shards':shards,'hash_schema':'raw_support/query SHA256 uses raw RGB bytes, matching trainer ledger. Program fields are32byte decoded SHA256. metadata_sha256 uses sorted canonical JSON exactly as trainer. generated_* hashes include native generated dtype, shape and bytes; trainer tensor hashes are a separate encoding.'}
    atomic_json(DEST/'result.json',result);atomic_json(DEST/'progress.json',progress(result['status']))
    print(json.dumps({k:v for k,v in result.items()if k not in ['shards','errors']},indent=2),flush=True)
if __name__=='__main__':main()
