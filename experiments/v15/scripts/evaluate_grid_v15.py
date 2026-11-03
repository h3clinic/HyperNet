"""Calibrate one-reference Grid anomaly scores, freeze, then evaluate TEST."""
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import argparse,json,math,os,platform,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/mvtec_grid_v15'
from hypernet.mvtec_grid import GridDataset
from hypernet.grid_anomaly import GridAnomaly,SAMPLING_STRIDE,MAX_BANK,BANK_SEED,TOP_FRACTION,original_pixel_map
from hypernet.rare_event_metrics import summary,upper_bound

ALIASES=['learned_final','random','analytic_detail']
SOURCES=['hypernet/detail_encoder.py','hypernet/detail_features.py','hypernet/grid_anomaly.py',
         'hypernet/mvtec_grid.py','hypernet/rare_event_metrics.py','scripts/evaluate_grid_v15.py','scripts/prepare_mvtec_grid.py']
def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def require(condition,message):
    if not condition:raise ValueError(message)
def resolve(path):return Path(path).resolve() if Path(path).is_absolute() else (ROOT/path).resolve()
def source_hashes():return {name:digest(ROOT/name) for name in SOURCES}

def validate_checkpoint(path,expected_families=None):
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=True)
    require(saved.get('natural_training_images')==0 and saved.get('borrowed_pretrained_weights') is False,'Own synthetic-only checkpoint required')
    if expected_families is not None:require(saved.get('consumed_families',0)==expected_families,'Wrong declared training-family milestone')
    return {'sha256':digest(path),'state_sha256':saved['state_sha256'],'families':saved.get('consumed_families',0),'optimizer_steps':saved['optimizer_steps']}

def operating_cutoff(scores):
    values=np.asarray(scores,dtype=np.float64)
    require(values.ndim==1 and len(values)==64 and np.isfinite(values).all(),'Exactly64 finite normal DEV scores required')
    cutoff=float(np.nextafter(values.max(),math.inf))
    return {'threshold':cutoff,'normal_calibration_images':64,'observed_calibration_false_positives':int(np.count_nonzero(values>=cutoff)),
            'rule':'Strict maximum normal DEV score plus nextafter toward +infinity',
            'calibration_zero_alarm_one_sided95_upper_per_10000':10000*upper_bound(0,64),
            'claim':'Zero empirical calibration alarms; no guaranteed future false-positive rate.'}

def verify_protocol(protocol):
    require(protocol['model_selection_complete'] and protocol['test_scores_computed_before_freeze'] is False,'Invalid TEST freeze')
    for group in ('behavior_files','checkpoint_files'):
        for path,value in protocol[group].items():require(digest(resolve(path))==value,'Frozen file changed: '+path)
    require(digest(DATA/'manifest.json')==protocol['grid_manifest_sha256'],'Grid manifest changed')
    require(digest(DATA/'roster_freeze.json')==protocol['grid_roster_freeze_sha256'],'Grid roster changed')
    require(digest(ROOT/'experiments/mvtec_grid_data_audit.json')==protocol['grid_data_audit_sha256'],'Grid data audit changed')
    require(digest(resolve(protocol['dev_results_path']))==protocol['dev_results_sha256'],'DEV calibration changed')

def verify_test_admission_matches(protocol_path,admission_path):
    admission=read(admission_path)
    require(admission['protocol_sha256']==digest(protocol_path),'TEST admission is for a different CLI protocol')

def validate_final_dev_rosters(dev):
    require(dev['method_aliases']==ALIASES and dev['query_images']==64 and dev['primary_method']=='learned_final','Final model/control roster differs')
    for name in ('score_files','calibration','thresholds'):
        require(isinstance(dev.get(name),dict) and set(dev[name])==set(ALIASES),'Incomplete final DEV '+name+' roster')
    for name in ('checkpoints','checkpoint_provenance'):
        require(isinstance(dev.get(name),dict) and set(dev[name])=={'learned_final','random'},'Incomplete checkpoint roster')

def save_prediction(output,alias,image_id,prediction,record):
    relative=Path('maps')/alias/(image_id.replace('/','__')+'.npz');path=output/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez_compressed(stream,patch_scores=prediction['patch_scores'],xy=prediction['xy'])
    row={key:record[key] for key in ('id','role','official_split')}
    row.update({key:value for key,value in prediction.items() if key not in ('patch_scores','xy')})
    row.update(method=alias,patch_shape=list(prediction['patch_scores'].shape),map_path=relative.as_posix(),map_sha256=digest(path))
    return row

def pixel_metrics(output,alias,rows,dataset):
    """Exact full-resolution rank metrics; temporary mmap arrays limit RAM use."""
    from sklearn.metrics import roc_auc_score,average_precision_score
    total=sum(dataset.records[row['id']]['width']*dataset.records[row['id']]['height'] for row in rows)
    folder=output/'metric_work';folder.mkdir(exist_ok=True)
    score_path=folder/(alias+'_scores.f32');target_path=folder/(alias+'_targets.u8')
    require(not score_path.exists() and not target_path.exists(),'Preserve incomplete metric workspace')
    scores=np.memmap(score_path,mode='w+',dtype=np.float32,shape=(total,));targets=np.memmap(target_path,mode='w+',dtype=np.uint8,shape=(total,))
    cursor=0
    for row in rows:
        path=output/row['map_path'];require(digest(path)==row['map_sha256'],'Saved anomaly map changed')
        with np.load(path,allow_pickle=False) as saved:
            score=original_pixel_map(saved['patch_scores'],saved['xy'],row['source_shape'])
        target=dataset.load_evaluation_target(row['id'])['mask']
        require(target.shape==score.shape,'Pixel metric shape mismatch')
        count=score.size;scores[cursor:cursor+count]=score.ravel();targets[cursor:cursor+count]=target.ravel();cursor+=count
    scores.flush();targets.flush();positive=int(np.count_nonzero(targets))
    require(0<positive<total,'Pixel AUROC/AP require both classes')
    result={'auroc':float(roc_auc_score(targets,scores)),'average_precision':float(average_precision_score(targets,scores)),
            'positive_pixels':positive,'negative_pixels':total-positive,'total_pixels':total,
            'scope':'All original1024x1024 pixels across all78officialTEST images; normal masks are zero. Score maps are interpolated at native sampled centers; GT masks never resized.',
            'approximate_histogram':False}
    del scores,targets
    # Only these two explicit files inside this run output are ephemeral.
    require(score_path.resolve().is_relative_to(output.resolve()) and target_path.resolve().is_relative_to(output.resolve()),'Metric path escaped output')
    score_path.unlink();target_path.unlink()
    return result

def freeze(dev_path):
    dev_path=Path(dev_path).resolve();dev=read(dev_path)
    require(dev.get('completed') and dev['stage']=='dev' and not dev['prototype'],'Require final complete64normal DEV calibration')
    validate_final_dev_rosters(dev)
    require(source_hashes()==dev['source_sha256'],'Source changed after DEV')
    require(digest(DATA/'development_manifest.json')==dev['development_manifest_sha256'],'DEV manifest changed')
    dataset=GridDataset(DATA);ids=dataset.ids('dev_normal')
    for alias,info in dev['score_files'].items():
        path=dev_path.parent/info['path'];require(digest(path)==info['sha256'],'DEV log changed')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows)==64 and [r['id'] for r in rows]==ids,'DEV roster changed')
        require(operating_cutoff([r['score'] for r in rows])==dev['calibration'][alias],'DEV threshold changed')
    for name,path in dev['checkpoints'].items():
        require(validate_checkpoint(path,1048576 if name=='learned_final' else 0)==dev['checkpoint_provenance'][name],'Selected checkpoint changed')
    require(dev['state_unchanged'] and dev['banks_unchanged'],'DEV changed model/reference state')
    files=source_hashes()
    for name in ('data/mvtec_grid_v15/roster_freeze.json','experiments/mvtec_grid_data_audit.json','experiments/grid_method_plan.json'):
        files[name]=digest(ROOT/name)
    files[str(dev_path)]=digest(dev_path)
    for info in dev['score_files'].values():
        path=dev_path.parent/info['path'];files[str(path)]=digest(path)
    protocol={'format':'hypernet.grid_anomaly.protocol.v15','created_utc':now(),'model_selection_complete':True,
        'test_scores_computed_before_freeze':False,'grid_manifest_sha256':digest(DATA/'manifest.json'),
        'grid_roster_freeze_sha256':digest(DATA/'roster_freeze.json'),
        'grid_data_audit_sha256':digest(ROOT/'experiments/mvtec_grid_data_audit.json'),
        'behavior_files':files,'checkpoint_files':{path:dev['checkpoint_provenance'][name]['sha256'] for name,path in dev['checkpoints'].items()},
        'checkpoints':dev['checkpoints'],'method_aliases':ALIASES,'primary_method':'learned_final','device':dev['device'],'batch_size':dev['batch_size'],
        'thresholds':{name:value['threshold'] for name,value in dev['calibration'].items()},'calibration':dev['calibration'],
        'dev_results_path':str(dev_path),'dev_results_sha256':digest(dev_path),'reference_id':dev['reference_id'],
        'method_plan_sha256':digest(ROOT/'experiments/grid_method_plan.json'),
        'real_data_budget':{'normal_reference_images':1,'extra_threshold_calibration_normals':64,'natural_weight_training_images':0},
        'test_policy':'All78official Grid TEST images once, afterfreeze; no threshold/model/reference/map tuning from TEST outcomes.'}
    path=ROOT/'experiments/grid_protocol_freeze.json';write_new(path,protocol)
    print(json.dumps({'protocol':str(path),'sha256':digest(path)}),flush=True)

def run(args):
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=Path(args.output).resolve();require(not output.exists(),'Preserve existing evaluation output')
    protocol=None
    if args.stage=='test':
        require(args.protocol is not None,'TEST needs a frozen protocol')
        protocol_path=Path(args.protocol).resolve();protocol=read(protocol_path);verify_protocol(protocol)
        verify_test_admission_matches(protocol_path,DATA/'test_admission.json')
        dataset=GridDataset(DATA,development_only=False);dataset._test_gate()
        checkpoints=protocol['checkpoints'];device=protocol['device'];batch_size=protocol['batch_size'];prototype=False
        ids=dataset.ids('test');require(len(ids)==78,'Unexpected officialTEST roster')
    else:
        prototype=args.prototype;dataset=GridDataset(DATA);ids=dataset.ids('dev_normal')
        require(args.random_checkpoint is not None,'Specify the same-initialization random checkpoint')
        checkpoints={'random':str(Path(args.random_checkpoint).resolve())}
        if not prototype:
            require(args.checkpoint is not None,'FinalDEV needs the finalmillion-family checkpoint')
            checkpoints={'learned_final':str(Path(args.checkpoint).resolve()),**checkpoints}
        if args.limit_dev is not None:
            require(prototype and 1<=args.limit_dev<=64,'Only a labeled prototype may limit DEV');ids=ids[:args.limit_dev]
        device=args.device;batch_size=args.batch_size
    require(1<=batch_size<=64,'Use1..64 native tiles per batch')
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    checkpoint_provenance={name:validate_checkpoint(path,1048576 if name=='learned_final' else 0) for name,path in checkpoints.items()}
    before_sources=source_hashes();reference_ids=dataset.ids('reference');require(len(reference_ids)==1,'Exactly one reference required')
    audit=read(ROOT/'experiments/mvtec_grid_data_audit.json')
    require(audit['valid'] and audit['development_manifest_sha256']==digest(DATA/'development_manifest.json'),'Grid data audit changed')
    output.mkdir(parents=True)
    write_new(output/'admission.json',{'created_utc':now(),'stage':args.stage,'prototype':prototype,'query_ids':ids,
        'source_sha256':before_sources,'checkpoint_provenance':checkpoint_provenance,
        'protocol_sha256':digest(args.protocol) if protocol else None,'model_calls_started':False})
    engine=GridAnomaly(checkpoints,device=device,batch_size=batch_size);engine.enroll(dataset.load_rgb(reference_ids[0]))
    before=engine.provenance();aliases=[*checkpoints,'analytic_detail'];rows={name:[] for name in aliases};start=time.perf_counter()
    if protocol:
        dev=read(resolve(protocol['dev_results_path']))
        require(before==dev['provenance'],'TEST model/reference bank differs from frozen DEV')
    for index,image_id in enumerate(ids):
        predictions=engine.predict(dataset.load_rgb(image_id));require(list(predictions)==aliases,'Prediction aliases differ')
        for alias,prediction in predictions.items():
            row=save_prediction(output,alias,image_id,prediction,dataset.records[image_id])
            # Official labels/masks enter after all method predictions above.
            if protocol:row.update(is_anomaly=dataset.records[image_id]['is_anomaly'],defect_type=dataset.records[image_id]['defect_type'])
            rows[alias].append(row)
        if (index+1)%4==0 or index+1==len(ids):print(json.dumps({'stage':args.stage,'images':index+1,'total':len(ids),'seconds':round(time.perf_counter()-start,2)}),flush=True)
    after=engine.provenance();require(before==after,'CNN or reference bank changed')
    require(source_hashes()==before_sources,'Inference source changed during run')
    for name,path in checkpoints.items():require(validate_checkpoint(path)==checkpoint_provenance[name],'Checkpoint changed during run')
    score_files={}
    for alias,records in rows.items():
        path=output/(alias+'_scores.jsonl')
        with path.open('x',encoding='utf-8') as stream:
            for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
        score_files[alias]={'path':path.name,'sha256':digest(path)}
    calibration={alias:operating_cutoff([row['score'] for row in records]) for alias,records in rows.items()} if args.stage=='dev' and len(ids)==64 else None
    metrics=None
    if protocol:
        metrics={}
        for alias,records in rows.items():
            print(json.dumps({'pixel_metrics':alias,'original_resolution':True}),flush=True)
            metrics[alias]={'image':summary([r['score'] for r in records],[r['is_anomaly'] for r in records],protocol['thresholds'][alias]),
                            'pixel':pixel_metrics(output,alias,records,dataset)}
        verify_protocol(protocol);require(digest(protocol_path)==read(DATA/'test_admission.json')['protocol_sha256'],'TEST admission changed')
    result={'format':'hypernet.grid_anomaly.results.v15','completed':True,'stage':args.stage,'prototype':prototype,
        'created_utc':now(),'method_aliases':aliases,'primary_method':'learned_final' if 'learned_final' in checkpoints else None,
        'reference_id':reference_ids[0],'query_images':len(ids),'checkpoints':checkpoints,'checkpoint_provenance':checkpoint_provenance,
        'device':device,'batch_size':batch_size,'source_sha256':before_sources,'provenance':before,'state_unchanged':before==after,'banks_unchanged':before['banks']==after['banks'],
        'development_manifest_sha256':digest(DATA/'development_manifest.json'),'manifest_sha256':digest(DATA/'manifest.json'),
        'protocol_sha256':digest(args.protocol) if protocol else None,'score_files':score_files,'calibration':calibration,
        'thresholds':protocol['thresholds'] if protocol else ({a:v['threshold'] for a,v in calibration.items()} if calibration else None),
        'methods':metrics,'elapsed_seconds':time.perf_counter()-start,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'threads':4,'tf32':False,
            'own_process_peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
            'own_process_peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0},
        'data_budget':{'normal_reference_images':1,'additional_normals_for_bank_fitting':0,'extra_normal_calibration_images':64,'natural_weight_training_images':0},
        'scope':'Grid-only one-reference anomaly pilot, not fullMVTec or classificationaccuracy.64extra realnormal images set thresholds; all79reference+TEST images are disjoint from calibration. FP/10000 is scaled from21normalTEST images with binomial uncertainty.'}
    write_new(output/'grid_results.json',result)
    print(json.dumps({'completed':True,'results':str(output/'grid_results.json'),'sha256':digest(output/'grid_results.json')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['dev','freeze','test'],required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--random-checkpoint',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--prototype',action='store_true');parser.add_argument('--limit-dev',type=int)
    parser.add_argument('--dev-results',type=Path);parser.add_argument('--protocol',type=Path);args=parser.parse_args()
    if args.stage=='freeze':require(args.dev_results is not None,'Provide DEV result');freeze(args.dev_results)
    else:require(args.output is not None,'Provide new output directory');run(args)
