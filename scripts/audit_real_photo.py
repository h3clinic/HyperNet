"""Independent arithmetic, provenance and predeclared deterministic v13 replay.

Does not import production metric/runner functions. Runtime replay uses only
the frozen RGB/support-box inference API after a completed original run exists.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
DATA=ROOT/'data/coco_real_v13'
RESULTS=ROOT/'results/real_eval'
ALIASES=('old_cnn','native_patch','old_cnn_grid','native_patch_grid')
TRANSFERRED={'old_cnn':.9806613922119142,'native_patch':.41308703128766816,
             'old_cnn_grid':.9806613922119142,'native_patch_grid':.41308703128766816}


def require(value,message):
    if not value:raise AssertionError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def load(row):
    path=(DATA/row['rgb_path']).resolve()
    require(path.is_relative_to(DATA.resolve()) and digest(path)==row['sha256'],'Input path/raw hash changed')
    with Image.open(path) as image:
        require(image.getexif().get(274,1)==1,'EXIF coordinates invalid')
        array=np.array(ImageOps.exif_transpose(image).convert('RGB'),dtype=np.uint8)
    require(array.shape==(row['height'],row['width'],3),'Input dimensions changed')
    value=sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()
    require(value==row['pixel_sha256'],'Input decoded pixels changed')
    return array


def iou(box,targets):
    if box is None:return 0.
    best=0.
    for target in targets:
        x=max(0.,min(box[2],target[2])-max(box[0],target[0]))
        y=max(0.,min(box[3],target[3])-max(box[1],target[1]))
        intersection=x*y
        union=(box[2]-box[0])*(box[3]-box[1])+(target[2]-target[0])*(target[3]-target[1])-intersection
        best=max(best,intersection/union)
    return best


def metric(rows,threshold):
    positives=[r for r in rows if r['present']]
    negatives=[r for r in rows if not r['present']]
    p,n=len(positives),len(negatives)
    require(p>0 and n>0,'Empty metric class')
    accepted=sum(r['score']>=threshold for r in positives)
    false=sum(r['score']>=threshold for r in negatives)
    local=sum(r['score']>=threshold and r['iou']>=.5 for r in positives)
    # Mann–Whitney rank statistic with average tied ranks.
    ordered=sorted(rows,key=lambda r:r['score'])
    rank_sum=0.;cursor=0
    while cursor<len(ordered):
        end=cursor+1
        while end<len(ordered) and ordered[end]['score']==ordered[cursor]['score']:end+=1
        rank_sum+=sum(r['present'] for r in ordered[cursor:end])*(cursor+1+end)/2
        cursor=end
    auc=(rank_sum-p*(p+1)/2)/(p*n)
    # AP of one selected box per scene, treating localization failures as misses.
    thresholds=sorted(set(r['score'] for r in rows),reverse=True)
    ap=0.;seen=found=0
    for value in thresholds:
        group=[r for r in rows if r['score']==value]
        hits=sum(r['present'] and r['iou']>=.5 for r in group)
        seen+=len(group);found+=hits
        ap+=(hits/p)*(found/seen)
    return {'positive_scenes':p,'negative_scenes':n,'threshold':threshold,
        'accepted_positive':accepted,'false_positive':false,'localized_true_positive':local,
        'recall':accepted/p,'false_positive_rate':false/n,'localized_recall_at_iou50':local/p,
        'top1_iou_recall_at_iou50':sum(r['iou']>=.5 for r in positives)/p,
        'proposal_recall_at_iou50':sum(r['proposal_best_iou']>=.5 for r in positives)/p,
        'balanced_accuracy':(accepted/p+1-false/n)/2,'auroc':auc,'single_box_ap50':ap}


def dev_cutoff(rows):
    values=sorted((r['score'] for r in rows if r['split']=='dev' and not r['present']),reverse=True)
    require(len(values)==48,'Expected48 DEV negatives')
    cutoff=math.nextafter(values[2],math.inf)
    require(sum(x>=cutoff for x in values)<=2,'Calibration exceeded2/48 negatives')
    return cutoff


def percentile(values,q):
    ordered=sorted(float(x) for x in values)
    index=(len(ordered)-1)*q;low=math.floor(index);high=math.ceil(index)
    return ordered[low]*(high-index)+ordered[high]*(index-low) if high!=low else ordered[low]


def compare_numeric(actual,expected,path='root'):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),'Keys differ: '+path)
        return sum(compare_numeric(actual[k],v,path+'.'+k) for k,v in expected.items())
    if isinstance(expected,list):
        require(len(actual)==len(expected),'Length differs: '+path)
        return sum(compare_numeric(a,b,path+f'[{i}]') for i,(a,b) in enumerate(zip(actual,expected)))
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        require(math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),abs_tol=1e-12,rel_tol=0),f'Numeric mismatch {path}: {actual} != {expected}')
        return 1
    require(actual==expected,'Value differs: '+path)
    return 0


def bootstrap(rows_by_alias,thresholds,categories):
    draws=np.random.default_rng(1313).integers(0,12,size=(10000,12))
    methods={};local_distributions={}
    for alias,rows in rows_by_alias.items():
        test=[r for r in rows if r['split']=='test']
        local=[];false=[]
        for category in categories:
            subset=[r for r in test if r['category_id']==category]
            require(sum(r['present'] for r in subset)==8 and len(subset)==16,'Unbalanced TEST category')
            local.append(sum(r['present'] and r['score']>=thresholds[alias] and r['iou']>=.5 for r in subset)/8)
            false.append(sum(not r['present'] and r['score']>=thresholds[alias] for r in subset)/8)
        samples_local=np.asarray(local)[draws].mean(axis=1)
        samples_false=np.asarray(false)[draws].mean(axis=1)
        local_distributions[alias]=samples_local
        methods[alias]={
            'localized_recall_at_iou50':{'estimate':sum(local)/12,'ci95':[percentile(samples_local,q) for q in [.025,.975]]},
            'false_positive_rate':{'estimate':sum(false)/12,'ci95':[percentile(samples_false,q) for q in [.025,.975]]}}
    differences={}
    for treatment,control in [('native_patch','old_cnn'),('native_patch_grid','native_patch'),('old_cnn_grid','old_cnn')]:
        distribution=local_distributions[treatment]-local_distributions[control]
        differences[treatment+'_minus_'+control]={
            'estimate':methods[treatment]['localized_recall_at_iou50']['estimate']-methods[control]['localized_recall_at_iou50']['estimate'],
            'ci95':[percentile(distribution,q) for q in [.025,.975]]}
    return methods,differences


def tensor_state_hash(state):
    h=sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(protocol_path):
    summary_path=RESULTS/'real_results.json'
    summary=read(summary_path);protocol=read(protocol_path);manifest=read(DATA/'manifest.json')
    require(summary['completed'] is True and summary['primary_mode']=='transferred','Missing complete frozen primary run')
    require(summary['protocol_sha256']==digest(protocol_path),'Protocol binding changed')
    require(summary['manifest_sha256']==digest(DATA/'manifest.json'),'Manifest binding changed')
    require(summary['borrowed_pretrained_weights'] is False and summary['natural_training_images']==0 and summary['learned_weight_updates_this_version']==0,'Training budget changed')
    tracked={Path(protocol_path).resolve():digest(protocol_path),summary_path:digest(summary_path),DATA/'manifest.json':digest(DATA/'manifest.json')}
    for name,value in protocol['behavior_files'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        require(digest(path)==value,'Frozen behavior changed: '+name);tracked[path]=value
    for key,name in [('data_audit_sha256','real_photo_data_audit.json'),('data_freeze_sha256','real_photo_data_freeze.json'),('fetch_admission_sha256','real_photo_fetch_admission.json')]:
        path=ROOT/'experiments'/name
        require(digest(path)==protocol[key],'Data evidence changed');tracked[path]=digest(path)
    dataaudit=read(ROOT/'experiments/real_photo_data_audit.json')
    require(dataaudit['valid'] and dataaudit['manifest_sha256']==digest(DATA/'manifest.json'),'Data audit mismatch')
    recovery_path=ROOT/'experiments/real_photo_transport_recovery_admission.json'
    recovery=read(recovery_path)
    require(digest(recovery_path)==manifest['transport_recovery_admission_sha256'],'Transport recovery binding')
    require(digest(ROOT/'scripts/recover_real_photo_transport.py')==recovery['recovery_script_sha256'],'Transport source changed')
    tracked[recovery_path]=digest(recovery_path)
    for row in manifest['references']+manifest['queries']:
        load(row);tracked[DATA/row['rgb_path']]=row['sha256']
    all_ids=[r['image_id'] for r in manifest['references']+manifest['queries']]
    require(len(all_ids)==len(set(all_ids))==300,'Photo split identity leakage')
    categories=sorted(r['category_id'] for r in manifest['references'])
    queries={r['id']:r for r in manifest['queries']}
    references={r['category_id']:r for r in manifest['references']}
    rows={};by_alias={};iou_checks=0
    for alias in ALIASES:
        score_path=RESULTS/(alias+'_scores.jsonl')
        require(digest(score_path)==summary['score_files'][alias]['sha256'],'Raw predictions changed')
        tracked[score_path]=digest(score_path)
        records=[json.loads(line) for line in score_path.read_text().splitlines()]
        require(len(records)==288 and {r['id'] for r in records}==set(queries),'Prediction roster mismatch')
        for row in records:
            q=queries[row['id']];ref=references[q['category_id']]
            for key in ['category_id','name','split','present','image_id']:require(row[key]==q[key],'Prediction labels mismatch')
            require(row['reference_image_id']==ref['image_id'] and row['reference_path']==ref['rgb_path'] and row['query_path']==q['rgb_path'],'Reference/query binding mismatch')
            require(row['method']==alias and row['diagnostics']['learned_weight_updates']==0,'Wrong method/training flag')
            exact=iou(row['bbox_xyxy'],q['gt_boxes_xyxy'])
            proposal=max((iou(b,q['gt_boxes_xyxy']) for b in row['proposals']),default=0.)
            compare_numeric([row['iou'],row['proposal_best_iou']],[exact,proposal]);iou_checks+=2
            require(len(row['proposals'])==len(row['proposal_scores'])==len(row['all_evidence']),'Proposal length mismatch')
            if row['proposals']:
                index=max(range(len(row['proposal_scores'])),key=row['proposal_scores'].__getitem__)
                require(row['bbox_xyxy']==row['proposals'][index] and row['score']==row['proposal_scores'][index],'Non-argmax chosen box')
                require(row['evidence']==row['all_evidence'][index],'Winning evidence mismatch')
            else:require(row['bbox_xyxy'] is None,'Box without proposals')
        rows[alias]=records;by_alias[alias]={r['id']:r for r in records}
    for qid in queries:
        for first,second in [('old_cnn','native_patch'),('old_cnn_grid','native_patch_grid')]:
            require(by_alias[first][qid]['proposals']==by_alias[second][qid]['proposals'],'Matched methods use different proposals')
    numeric_checks=0;bootstrap_checks=0
    for mode in ['transferred','real_dev_calibrated']:
        thresholds=TRANSFERRED if mode=='transferred' else {alias:dev_cutoff(records) for alias,records in rows.items()}
        saved=summary['modes'][mode]
        numeric_checks+=compare_numeric(saved['thresholds'],thresholds)
        for alias,records in rows.items():
            for split in ['dev','test']:
                subset=[r for r in records if r['split']==split]
                numeric_checks+=compare_numeric(saved['methods'][alias][split],metric(subset,thresholds[alias]))
                for category in categories:
                    selected=[r for r in subset if r['category_id']==category]
                    numeric_checks+=compare_numeric(saved['per_category'][str(category)][alias][split],metric(selected,thresholds[alias]))
        methods,differences=bootstrap(rows,thresholds,categories)
        boot=saved['bootstrap']
        require(boot['category_ids']==categories and boot['seed']==1313 and boot['samples']==10000 and boot['categories']==12,'Bootstrap cluster settings changed')
        bootstrap_checks+=compare_numeric(boot['methods'],methods)+compare_numeric(boot['localized_recall_differences'],differences)
    plan_path=ROOT/'experiments/real_photo_replay_plan.json'
    supplement_path=ROOT/'experiments/real_photo_positive_replay_supplement.json'
    replay_plan=read(plan_path);supplement=read(supplement_path)
    require(replay_plan['manifest_sha256']==supplement['manifest_sha256']==digest(DATA/'manifest.json'),'Replay data binding changed')
    require(supplement['original_replay_plan_sha256']==digest(plan_path),'Original replay plan changed')
    replay_ids=replay_plan['query_ids']+supplement['query_ids']
    require(len(replay_ids)==len(set(replay_ids))==24,'Replay roster changed')
    require(all(queries[qid]['split']=='test' for qid in replay_ids),'Replay is not TEST-only')
    tracked[plan_path]=digest(plan_path);tracked[supplement_path]=digest(supplement_path)
    checkpoint=ROOT.parent/'hypernet_v11/checkpoints/detector_micro_aware.pt'
    require(digest(checkpoint)==summary['checkpoint_sha256']==protocol['checkpoint_sha256'],'Checkpoint file changed')
    tracked[checkpoint]=digest(checkpoint)
    # Actual replay starts only after all offline evidence checks pass.
    sys.path.insert(0,str(ROOT))
    import torch
    from hypernet.real_photo import RealPhotoDetector
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(payload['natural_training_images']==0 and payload['externally_pretrained_weights'] is False,'Checkpoint natural/pretrained provenance changed')
    engine=RealPhotoDetector(checkpoint)
    state_before=tensor_state_hash(engine.model.state_dict())
    require(state_before==tensor_state_hash(payload['state_dict'])==summary['model_state_before']==summary['model_state_after'],'Model tensors differ from checkpoint')
    require(not engine.model.training and not any(p.requires_grad for p in engine.model.parameters()),'Model is trainable')
    enrolled={c:engine.enroll(load(row),row['bbox_xyxy']) for c,row in references.items()}
    exact_replays=0
    for qid in replay_ids:
        q=queries[qid]
        predictions=engine.predict(enrolled[q['category_id']],load(q))
        require(set(predictions)==set(ALIASES),'Replay aliases differ')
        for alias,prediction in predictions.items():
            recorded=by_alias[alias][qid]
            require(prediction=={key:recorded[key] for key in prediction},'Full prediction replay differs: '+qid+'/'+alias)
            exact_replays+=1
    state_after=tensor_state_hash(engine.model.state_dict())
    require(state_before==state_after and all(p.grad is None for p in engine.model.parameters()),'Replay modified model or produced gradients')
    require(all(digest(path)==value for path,value in tracked.items()),'Source/input/result changed during audit')
    report={'valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'auditor_sha256':digest(__file__),'results_sha256':digest(summary_path),'protocol_sha256':digest(protocol_path),
        'manifest_sha256':digest(DATA/'manifest.json'),'data_audit_sha256':digest(ROOT/'experiments/real_photo_data_audit.json'),
        'transport_recovery_admission_sha256':digest(recovery_path),'checkpoint_sha256':digest(checkpoint),
        'model_state_before':state_before,'model_state_after':state_after,'gradients_created':False,
        'method_rows_verified':sum(len(r) for r in rows.values()),'scene_and_proposal_iou_checks':iou_checks,
        'metric_numbers_independently_recomputed':numeric_checks,'bootstrap_numbers_independently_recomputed':bootstrap_checks,
        'bootstrap_categories':categories,'bootstrap_seed':1313,'bootstrap_samples':10000,
        'exact_full_prediction_replays':exact_replays,'replay_query_ids':replay_ids,'maximum_replay_difference':0,
        'all300_raw_and_decoded_images_verified':True,'identical_proposals_within_matched_comparisons':True,
        'frozen_sources_data_scores_and_checkpoint_unchanged':True,'tracked_files':len(tracked),
        'replay_plan_sha256':digest(plan_path),'positive_replay_supplement_sha256':digest(supplement_path),
        'arithmetic_note':'Independent scalar IoU, Mann-Whitney rank AUROC, tie-group scene AP, and manually interpolated percentile bootstrap; no production metric functions imported.',
        'scope':'Natural COCO-category transfer on new photographs; primary synthetic thresholds unchanged, secondary96-photo realDEV calibration distinct; not official COCO AP.'}
    out=ROOT/'experiments/real_photo_independent_audit.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-freeze',type=Path,required=True)
    args=parser.parse_args();run(args.protocol_freeze)
