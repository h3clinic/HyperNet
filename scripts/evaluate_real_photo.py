"""Frozen real-photo transfer evaluation; no reference training or model tuning.

Primary: unchanged synthetic thresholds. Secondary: additional real DEV photos
calibrate one pooled threshold per method. Both modes reuse identical scores.
"""
from __future__ import annotations
import argparse,collections,contextlib,datetime,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageOps
import torch
from hypernet.real_photo import RealPhotoDetector,ALIASES,PRIMARY,TRANSFERRED_CUTOFFS
from hypernet.detail_pipeline import state_hash
from hypernet.real_photo_metrics import max_box_iou,cutoff_from_real_dev,summarize,category_bootstrap

DATA_BINDINGS={
    'data_audit_sha256':'experiments/real_photo_data_audit.json',
    'data_freeze_sha256':'experiments/real_photo_data_freeze.json',
    'fetch_admission_sha256':'experiments/real_photo_fetch_admission.json',
    'transport_recovery_admission_sha256':'experiments/real_photo_transport_recovery_admission.json'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def load_rgb(data,record):
    path=(Path(data)/record['rgb_path']).resolve()
    if not path.is_relative_to(Path(data).resolve()) or sha(path)!=record['sha256']:
        raise ValueError('Image path/file hash differs from manifest')
    with Image.open(path) as image:
        rgb=np.array(ImageOps.exif_transpose(image).convert('RGB'))
    expected=(int(record['height']),int(record['width']),3)
    if rgb.dtype!=np.uint8 or rgb.shape!=expected:
        raise ValueError('Decoded image dimensions differ from manifest')
    digest=hashlib.sha256();digest.update(str(rgb.dtype).encode());digest.update(str(rgb.shape).encode());digest.update(rgb.tobytes())
    if digest.hexdigest()!=record['pixel_sha256']:
        raise ValueError('Decoded image hash differs from manifest')
    return rgb

def verify_protocol(protocol_path,data,checkpoint):
    protocol=read(protocol_path)
    if protocol.get('model_selection_complete') is not True or protocol.get('photo_scores_computed_before_freeze') is not False:
        raise ValueError('Require frozen design before real-photo scoring')
    if protocol.get('primary_mode')!='transferred' or protocol.get('primary_method')!=PRIMARY:
        raise ValueError('Primary experiment must remain transferred/native_patch')
    if protocol.get('transferred_thresholds')!=TRANSFERRED_CUTOFFS:
        raise ValueError('Transferred synthetic thresholds changed')
    tracked={Path(protocol_path).resolve():sha(protocol_path)}
    for name,expected in protocol['behavior_files'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(path)!=expected:raise ValueError('Frozen behavior changed: '+name)
        tracked[path.resolve()]=expected
    for key,name in DATA_BINDINGS.items():
        path=ROOT/name
        if sha(path)!=protocol[key]:raise ValueError('Frozen data evidence changed: '+name)
        tracked[path.resolve()]=protocol[key]
    manifest_path=Path(data)/'manifest.json';manifest=read(manifest_path)
    if sha(manifest_path)!=protocol['data_manifest_sha256'] or manifest.get('schema_version')!='hypernet-real-photo-v13-v1':
        raise ValueError('Require exact frozen real-photo manifest')
    if len(manifest.get('references',[]))!=12 or len(manifest.get('queries',[]))!=288:
        raise ValueError('Require complete12-reference/288-query manifest')
    audit=read(ROOT/DATA_BINDINGS['data_audit_sha256'])
    if audit.get('valid') is not True or audit.get('manifest_sha256')!=sha(manifest_path):
        raise ValueError('Require passing independent data audit for this manifest')
    for key in ('data_freeze_sha256','fetch_admission_sha256'):
        if audit.get(key)!=protocol[key]:raise ValueError('Data audit evidence binding differs: '+key)
    recovery=read(ROOT/DATA_BINDINGS['transport_recovery_admission_sha256'])
    if manifest.get('transport_recovery_admission_sha256')!=protocol['transport_recovery_admission_sha256']:
        raise ValueError('Manifest transport recovery binding differs')
    if recovery.get('data_freeze_sha256')!=protocol['data_freeze_sha256'] or recovery.get('original_fetch_admission_sha256')!=protocol['fetch_admission_sha256']:
        raise ValueError('Recovery must preserve original selection/fetch evidence')
    if recovery.get('selection_annotations_and_criteria_unchanged') is not True or recovery.get('model_runs')!=0 or recovery.get('original_dataset_images_written')!=0:
        raise ValueError('Recovery must be transport-only before any model runs')
    recovery_script=ROOT/'scripts/recover_real_photo_transport.py'
    if sha(recovery_script)!=recovery.get('recovery_script_sha256') or tracked.get(recovery_script.resolve())!=sha(recovery_script):
        raise ValueError('Recovery script must match admission and frozen behavior files')
    if sha(checkpoint)!=protocol['checkpoint_sha256']:
        raise ValueError('Own synthetic checkpoint changed')
    tracked[manifest_path.resolve()]=sha(manifest_path);tracked[Path(checkpoint).resolve()]=sha(checkpoint)
    return protocol,manifest,tracked

def validate_prediction(prediction,width,height):
    for required in ('score','bbox_xyxy','proposals','proposal_scores','evidence','diagnostics'):
        if required not in prediction:raise ValueError('Prediction missing '+required)
    if not math.isfinite(float(prediction['score'])):
        raise ValueError('Nonfinite prediction score')
    boxes=prediction['proposals'];scores=prediction['proposal_scores']
    if len(boxes)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError('Proposal/score mismatch')
    for box in boxes+([prediction['bbox_xyxy']] if prediction['bbox_xyxy'] is not None else []):
        a=np.asarray(box,dtype=np.float64)
        if a.shape!=(4,) or not np.isfinite(a).all() or not (0<=a[0]<a[2]<=width and 0<=a[1]<a[3]<=height):
            raise ValueError('Output box is not in original query coordinates')
    if boxes:
        best=int(np.argmax(scores))
        if prediction['bbox_xyxy']!=boxes[best] or prediction['score']!=scores[best]:
            raise ValueError('Output must select one highest-scoring declared proposal')
    elif prediction['bbox_xyxy'] is not None:
        raise ValueError('Cannot predict a box without proposals')

def evaluate_queries(engine,manifest,data,on_row=None):
    references=manifest['references'];queries=manifest['queries']
    if len(references)!=12 or len(queries)!=288:
        raise ValueError('Expected12 references and288 queries')
    by_category={int(r['category_id']):r for r in references}
    if len(by_category)!=12 or len({q['id'] for q in queries})!=len(queries):
        raise ValueError('Duplicate reference category or query ID')
    if len({r['image_id'] for r in references+queries})!=300:
        raise ValueError('All300 original photos must be globally disjoint')
    if collections.Counter(q['split'] for q in queries)!=dict(dev=96,test=192):
        raise ValueError('Expected96 real DEV and192 real TEST queries')
    if {int(q['category_id']) for q in queries}!=set(by_category):
        raise ValueError('Query category has no unique enrolled reference')
    enrolled={}
    for category,record in by_category.items():
        enrolled[category]=engine.enroll(load_rgb(data,record),record['bbox_xyxy'])
    rows={alias:[] for alias in ALIASES}
    for index,query in enumerate(queries):
        category=int(query['category_id']);rgb=load_rgb(data,query)
        # Category identifies which single reference is enrolled for this task;
        # neither category ID nor query annotation is passed to predict().
        predictions=engine.predict(enrolled[category],rgb)
        if set(predictions)!=set(ALIASES):raise ValueError('Unexpected inference aliases')
        for prediction in predictions.values():validate_prediction(prediction,query['width'],query['height'])
        for first,second in [('old_cnn','native_patch'),('old_cnn_grid','native_patch_grid')]:
            if predictions[first]['proposals']!=predictions[second]['proposals']:
                raise ValueError('Matcher comparison must share exact proposals')
        # Only after all four methods return do we consult query ground truth.
        present=bool(query['present']);targets=query['gt_boxes_xyxy']
        if present!=bool(targets):raise ValueError('Presence label/target list mismatch')
        for box in targets:
            a=np.asarray(box,dtype=np.float64)
            if a.shape!=(4,) or not np.isfinite(a).all() or not (0<=a[0]<a[2]<=query['width'] and 0<=a[1]<a[3]<=query['height']):
                raise ValueError('Target boxes must use original image coordinates')
        for alias in ALIASES:
            prediction=predictions[alias]
            row=dict(id=query['id'],category_id=category,name=query['name'],split=query['split'],present=present,
                method=alias,image_id=query['image_id'],reference_image_id=by_category[category]['image_id'],
                reference_path=by_category[category]['rgb_path'],query_path=query['rgb_path'],
                original_query_size=[query['width'],query['height']],**prediction,
                iou=max_box_iou(prediction['bbox_xyxy'],targets),
                proposal_best_iou=max((max_box_iou(box,targets) for box in prediction['proposals']),default=0.))
            # Validate nested evidence/diagnostics now, before appending a row.
            json.dumps(row,allow_nan=False)
            rows[alias].append(row)
            if on_row is not None:on_row(alias,row)
        if (index+1)%24==0:print(f'Scored{index+1}/{len(queries)} real queries',flush=True)
    for category in by_category:
        counts=collections.Counter((r['split'],r['present']) for r in rows[ALIASES[0]] if r['category_id']==category)
        if counts!={('dev',True):4,('dev',False):4,('test',True):8,('test',False):8}:
            raise ValueError('Category-specific query counts differ from frozen design')
    return rows

def evaluate_modes(rows):
    thresholds={'transferred':dict(TRANSFERRED_CUTOFFS),
        'real_dev_calibrated':{alias:cutoff_from_real_dev([r for r in records if r['split']=='dev']) for alias,records in rows.items()}}
    modes={}
    for mode,cuts in thresholds.items():
        methods={alias:{split:summarize([r for r in records if r['split']==split],cuts[alias])
            for split in ('dev','test')} for alias,records in rows.items()}
        categories=sorted({r['category_id'] for r in rows[ALIASES[0]]})
        per_category={str(category):{alias:{split:summarize([r for r in records
            if r['category_id']==category and r['split']==split],cuts[alias]) for split in ('dev','test')}
            for alias,records in rows.items()} for category in categories}
        modes[mode]=dict(primary=mode=='transferred',thresholds=cuts,methods=methods,per_category=per_category,
            real_calibration_photos=0 if mode=='transferred' else 96,
            real_calibration_negative_photos=0 if mode=='transferred' else 48,
            threshold_source='Unchanged v12 synthetic DEV cutoffs' if mode=='transferred' else
                'Pooled96 additional real DEV photos; strict cutoff accepts at most2 of48 negatives; no weight fitting',
            bootstrap=category_bootstrap(rows,cuts,samples=10000,seed=1313))
    return modes

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/coco_real_v13')
    parser.add_argument('--checkpoint',type=Path,default=ROOT.parent/'hypernet_v11/checkpoints/detector_micro_aware.pt')
    parser.add_argument('--protocol-freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/real_eval')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Preserve existing evaluation evidence')
    protocol,manifest,tracked=verify_protocol(args.protocol_freeze,args.data,args.checkpoint)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    engine=RealPhotoDetector(args.checkpoint);before=state_hash(engine.model);start=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'run_status.json',dict(completed=False,protocol_sha256=sha(args.protocol_freeze),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state_before=before))
    with contextlib.ExitStack() as stack:
        handles={alias:stack.enter_context((args.output/f'{alias}_scores.jsonl').open('x',encoding='utf-8')) for alias in ALIASES}
        def persist(alias,row):handles[alias].write(json.dumps(row,allow_nan=False)+'\n')
        rows=evaluate_queries(engine,manifest,args.data,persist)
    after=state_hash(engine.model)
    if before!=after or any(p.grad is not None for p in engine.model.parameters()):
        raise RuntimeError('Inference changed weights or produced parameter gradients')
    if any(sha(path)!=digest for path,digest in tracked.items()):raise RuntimeError('Frozen input/source hash changed')
    modes=evaluate_modes(rows)
    summary=dict(format='hypernet.real_photo_transfer.v13',completed=True,
        scope='One boxed natural-photo reference per category; full-scene category detection on disjoint COCO val2017 photos. Not official COCO AP.',
        primary_mode='transferred',primary_method=PRIMARY,primary_metric='localized_recall_at_iou50',
        diagnostic_methods=['old_cnn_grid','native_patch_grid'],method_aliases=list(ALIASES),modes=modes,
        reference_categories=12,reference_images=12,reference_images_per_enrollment=1,
        query_images=288,dev_photos=96,test_photos=192,total_unique_photos=300,
        natural_training_images=0,learned_weight_updates_this_version=0,borrowed_pretrained_weights=False,
        model_provenance=engine.provenance(),model_state_before=before,model_state_after=after,
        gradients_created=False,protocol_sha256=sha(args.protocol_freeze),
        manifest_sha256=sha(args.data/'manifest.json'),checkpoint_sha256=sha(args.checkpoint),
        frozen_sources_and_inputs_unchanged=True,seconds=time.perf_counter()-start,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        score_files={alias:dict(path=f'{alias}_scores.jsonl',sha256=sha(args.output/f'{alias}_scores.jsonl')) for alias in ALIASES},
        metric_notes=['A positive scene succeeds when the selected box overlaps any eligible same-category instance at IoU>=.5 and its score passes the mode threshold.',
            'Top1 IoU recall and proposal recall ignore score thresholds and diagnose localization coverage.',
            'Single-box AP50 is tied-score scene ranking, not COCO multi-detection/multi-IoU AP.',
            'The primary mode uses no real calibration labels. Secondary realDEV calibration uses96 extra photos and does not modify weights.',
            'Fixed51-window grid methods are diagnostic proposal controls, not learned model improvements.',
            'Category-cluster bootstrap intervals condition on selected support images and fixed thresholds.'])
    write(args.output/'real_results.json',summary)
    write(args.output/'run_status.json',dict(completed=True,protocol_sha256=sha(args.protocol_freeze),
        results_sha256=sha(args.output/'real_results.json'),state_before=before,state_after=after))
    print(json.dumps({mode:{alias:summary['modes'][mode]['methods'][alias]['test'] for alias in ALIASES} for mode in modes},indent=2),flush=True)

if __name__=='__main__':main()
