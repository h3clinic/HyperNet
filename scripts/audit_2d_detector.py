"""Independent saved-result metrics, provenance and predeclared inference replay."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/shapes2d_v11'
REPLAY_PLAN=ROOT/'experiments/detector_replay_plan.json'
OUTPUT=ROOT/'experiments/detector_run_audit.json'


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(condition,message):
    if not condition:raise ValueError(message)
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def overlap(a,b):
    if a is None or b is None:return 0.
    w=max(0.,min(a[2],b[2])-max(a[0],b[0]));h=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    intersection=w*h
    union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-intersection
    return intersection/max(union,1e-12)


def dev_cutoff(rows):
    scores=np.sort(np.array([r['score'] for r in rows if not r['present']],dtype=np.float64))
    require(len(scores)>0,'DEV negatives required')
    allowed=len(scores)//20
    return float(np.nextafter(scores[-allowed-1],np.inf))


def independent_metrics(rows,cutoff):
    positives=[r for r in rows if r['present']];negatives=[r for r in rows if not r['present']]
    p=np.array([r['score'] for r in positives]);n=np.sort([r['score'] for r in negatives])
    require(len(p)>0 and len(n)>0 and np.isfinite(p).all() and np.isfinite(n).all(),'Finite balanced labels required')
    tp=int(np.sum(p>=cutoff));fp=int(np.sum(n>=cutoff))
    localized=sum(r['score']>=cutoff and r['iou']>=.5 for r in positives)
    lo=np.searchsorted(n,p,side='left');hi=np.searchsorted(n,p,side='right')
    auc=float(np.mean((lo+hi)/(2*len(n))))
    groups=defaultdict(list);pairs=defaultdict(dict)
    for row in rows:
        groups[row['score']].append(row)
        require(bool(row['present']) not in pairs[row['episode_id']],'Duplicate episode label')
        pairs[row['episode_id']][bool(row['present'])]=row['score']
    correct=count=0;ap=0.
    for score in sorted(groups,reverse=True):
        group=groups[score];new=sum(r['present'] and r['iou']>=.5 for r in group)
        correct+=new;count+=len(group);ap+=(correct/count)*(new/len(positives))
    require(all(set(pair)=={True,False} for pair in pairs.values()),'Incomplete paired episode')
    ranking=float(np.mean([(pair[True]>pair[False])+.5*(pair[True]==pair[False]) for pair in pairs.values()]))
    return {'positive_scenes':len(positives),'negative_scenes':len(negatives),'threshold':cutoff,
            'accepted_positive':tp,'false_positive':fp,'localized_true_positive':localized,
            'recall':tp/len(positives),'false_positive_rate':fp/len(negatives),
            'localized_recall_at_iou50':localized/len(positives),
            'proposal_recall_at_iou50':sum(r['proposal_best_iou']>=.5 for r in positives)/len(positives),
            'auroc':auc,'single_box_ap50':ap,'paired_scene_ranking_accuracy':ranking,
            'balanced_accuracy':.5*(tp/len(positives)+1-fp/len(negatives))}


def plan_stage():
    require(not (ROOT/'results/detector_results.json').exists(),'Replay selection must precede model results')
    data_plan=read(DATA/'plan.json')
    families=data_plan['family_partitions']['test']
    chosen=sorted(families[:2]+families[-2:])
    ids=[q['id'] for e in data_plan['episodes'] if e['split']=='test' and e['family_id'] in chosen for q in e['queries']]
    require(len(ids)==16,'Expected four families, two enrollments, two query rows')
    value={'created_utc':datetime.now(timezone.utc).isoformat(),'selection':'First two and last two sorted TEST family IDs; both enrollments and both queries',
           'selected_family_ids':chosen,'selected_query_ids':ids,'planned_comparisons':32,
           'data_plan_sha256':digest(DATA/'plan.json'),'data_freeze_sha256':digest(ROOT/'experiments/shapes2d_data_freeze.json'),
           'model_results_existed_at_selection':False}
    write_new(REPLAY_PLAN,value)
    print(json.dumps(value),flush=True)


def state_hash(model):
    h=sha256()
    for key,tensor in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_stage():
    import torch
    from PIL import Image
    from hypernet.one_shot_detector import OneReferenceDetector,FORMAT,detect
    freeze_path=ROOT/'experiments/detector_training_freeze.json'
    freeze=read(freeze_path);result_path=ROOT/'results/detector_results.json';results=read(result_path)
    manifest=read(DATA/'manifest.json');data_audit=read(ROOT/'experiments/shapes2d_data_audit.json')
    replay_plan=read(REPLAY_PLAN)
    require(results['completed'] and results['freeze_sha256']==digest(freeze_path),'Incomplete/unbound results')
    require(manifest['completed'] and data_audit['valid'],'Incomplete/unaudited data')
    for key,path in {'manifest_sha256':DATA/'manifest.json','batch_plan_sha256':ROOT/'experiments/detector_batch_plan.json',
                     'data_audit_sha256':ROOT/'experiments/shapes2d_data_audit.json','data_freeze_sha256':ROOT/'experiments/shapes2d_data_freeze.json'}.items():
        require(digest(path)==freeze[key],'Frozen input changed: '+key)
    for path,expected in freeze['sources'].items():require(digest(ROOT/path)==expected,'Frozen source changed: '+path)
    require(replay_plan['data_plan_sha256']==digest(DATA/'plan.json') and replay_plan['data_freeze_sha256']==freeze['data_freeze_sha256'],'Replay plan binding mismatch')
    require(digest(REPLAY_PLAN)==freeze['replay_plan_sha256'],'Training freeze did not bind this exact replay selection')
    episodes=manifest['episodes'];batch_plan=read(ROOT/'experiments/detector_batch_plan.json')
    train_ids={i for i,e in enumerate(episodes) if e['split']=='train'}
    flattened=[i for batch in batch_plan for i in batch]
    require(all(len(batch)==4 and len(set(batch))==4 for batch in batch_plan),'Wrong gradient batch composition')
    require(set(flattened)==train_ids and len(set(Counter(flattened).values()))==1,'Training coverage/leakage mismatch')
    require(len(batch_plan)==freeze['steps_per_arm'] and 4*len(batch_plan)==freeze['reference_exposures_per_arm'],'Gradient exposure budget mismatch')
    require(freeze['natural_training_images']==0 and freeze['externally_pretrained_weights'] is False,'Unexpected external training')
    require(freeze['query_annotations_at_inference'] is False,'Query annotation inference access declared')
    torch.set_num_threads(freeze['runtime']['threads']);torch.use_deterministic_algorithms(True)
    torch.manual_seed(freeze['seed']);initial=OneReferenceDetector()
    require(state_hash(initial)==freeze['initial_state_sha256'],'Initial state does not reproduce')
    require(sum(p.numel() for p in initial.parameters())==freeze['parameter_count'],'Model size mismatch')
    annotations={q['id']:(e,q) for e in episodes for q in e['queries'] if e['split'] in ('dev','test')}
    protected={str(p):digest(p) for p in [freeze_path,result_path,DATA/'manifest.json',DATA/'plan.json',REPLAY_PLAN,
        ROOT/'experiments/detector_batch_plan.json',ROOT/'experiments/shapes2d_data_audit.json',ROOT/'experiments/shapes2d_data_freeze.json']}
    protected.update({str(ROOT/p):h for p,h in freeze['sources'].items()})
    data_freeze=read(ROOT/'experiments/shapes2d_data_freeze.json')
    for path,expected in data_freeze['behavior_files'].items():
        require(digest(path)==expected,'Frozen generation source changed: '+path)
        protected[path]=expected
    metric_checks=0;localization_checks=0;replayed=[];arms={}
    def blocked(*args,**kwargs):raise RuntimeError('Training is forbidden during inference audit')
    torch.Tensor.backward=blocked
    torch.optim.AdamW.step=blocked
    for arm in freeze['arms']:
        checkpoint_path=ROOT/'checkpoints'/f'detector_{arm}.pt'
        prediction_path=ROOT/'results'/f'{arm}_predictions.jsonl'
        arm_result=results['arms'][arm]
        require(digest(checkpoint_path)==arm_result['checkpoint_sha256'] and digest(prediction_path)==arm_result['predictions_sha256'],'Model/result hash mismatch')
        protected[str(checkpoint_path)]=digest(checkpoint_path);protected[str(prediction_path)]=digest(prediction_path)
        rows=[json.loads(line) for line in prediction_path.read_text(encoding='utf-8').splitlines()]
        require(len(rows)==len(annotations)==1024 and {r['id'] for r in rows}==set(annotations),'Wrong heldout prediction roster')
        row_map={r['id']:r for r in rows}
        for row in rows:
            episode,target=annotations[row['id']]
            require(row['episode_id']==episode['episode_id'] and row['split']==episode['split'] and row['family_id']==episode['family_id'], 'Prediction identity mismatch')
            require(row['enrolled_variant']==episode['enrolled_variant'] and row['present']==target['present'],'Prediction labels mismatch')
            require(row['support_path']==episode['support']['rgb_path'] and row['query_path']==target['rgb_path'],'Prediction image binding mismatch')
            gt=target['target_bbox_xyxy']
            iou=overlap(row['bbox_xyxy'],gt)
            proposal=max((overlap(box,gt) for box in row['proposals']),default=0.)
            require(abs(iou-row['iou'])<1e-12 and abs(proposal-row['proposal_best_iou'])<1e-12,'Saved localization mismatch')
            localization_checks+=1
        cutoff=dev_cutoff([r for r in rows if r['split']=='dev'])
        for split in ('dev','test'):
            recomputed=independent_metrics([r for r in rows if r['split']==split],cutoff)
            for key,value in recomputed.items():
                require(abs(value-arm_result['metrics'][split][key])<1e-12,'Independent metric mismatch: '+arm+'/'+split+'/'+key)
                metric_checks+=1
        require(arm_result['metrics']['dev']['false_positive_rate']<=.05,'DEV calibration FPR exceeded')
        payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        require(payload['format']==FORMAT and payload['arm']==arm and payload['freeze_sha256']==digest(freeze_path),'Checkpoint identity mismatch')
        require(payload['natural_training_images']==0 and payload['externally_pretrained_weights'] is False,'Unexpected external checkpoint provenance')
        model=OneReferenceDetector();model.load_state_dict(payload['state_dict']);model.eval()
        before=state_hash(model);require(before==arm_result['final_state_sha256'],'Final tensor state mismatch')
        for query_id in replay_plan['selected_query_ids']:
            episode,target=annotations[query_id];expected=row_map[query_id]
            for record in (episode['support'],target):require(digest(DATA/record['rgb_path'])==record['rgb_sha256'],'Replay image hash mismatch')
            with Image.open(DATA/episode['support']['rgb_path']) as image:reference=np.array(image.convert('RGB'))
            with Image.open(DATA/target['rgb_path']) as image:query=np.array(image.convert('RGB'))
            reference=torch.from_numpy(reference.copy()).permute(2,0,1)[None].float()/255
            query=torch.from_numpy(query.copy()).permute(2,0,1)[None].float()/255
            actual=detect(model,reference,query,episode['support']['bbox_xyxy'])
            for key,expected_key in [('score','score'),('bbox_xyxy','bbox_xyxy'),('proposals','proposals'),('scores','proposal_scores')]:
                require(actual[key]==expected[expected_key],'Inference replay mismatch: '+arm+'/'+query_id+'/'+key)
            replayed.append({'arm':arm,'id':query_id,'score':actual['score'],'bbox_xyxy':actual['bbox_xyxy'],'exact':True})
        require(state_hash(model)==before and not any(p.grad is not None for p in model.parameters()),'Audit changed weights/gradients')
        arms[arm]={'checkpoint_sha256':digest(checkpoint_path),'predictions_sha256':digest(prediction_path),
                   'metrics':arm_result['metrics'],'final_state_sha256':before,'replayed_queries':16}
    require(len({results['arms'][arm]['proposal_training_roster_sha256'] for arm in freeze['arms']})==1,'Training proposal controls differ')
    for path,expected in protected.items():require(digest(path)==expected,'Audit modified protected input: '+path)
    audit={'valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),'auditor_sha256':digest(__file__),
           'freeze_sha256':digest(freeze_path),'results_sha256':digest(result_path),'manifest_sha256':digest(DATA/'manifest.json'),
           'replay_plan_sha256':digest(REPLAY_PLAN),'independent_metric_values_checked':metric_checks,
           'independent_localization_rows_checked':localization_checks,'exact_inference_replays':len(replayed),
           'all_training_indices_train_only':True,'all_training_episodes_used_equally':True,
           'same_initial_state_and_training_proposals':True,'new_training':False,'weights_and_gradients_unchanged':True,
           'protected_files':protected,'arms':arms,'replayed':replayed,
           'limitations':['One initial seed and fixed final checkpoint.','Whole grammar combinations are heldout but building blocks are shared.',
                          'Counterbalanced scene reuse creates correlated rows; independent test unit is the family.',
                          'single_box_ap50 is a tied-score-group, one-box-per-scene metric, not full COCO AP.']}
    write_new(OUTPUT,audit);print(json.dumps({k:v for k,v in audit.items() if k not in ('arms','protected_files','replayed')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('plan','audit'),required=True);args=parser.parse_args()
    plan_stage() if args.stage=='plan' else audit_stage()
