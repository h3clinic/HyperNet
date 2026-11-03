"""CPU-only synthetic DEVELOPMENT interventions; not a benchmark method.

Oracle variants use generated SUPPORT ground truth solely to diagnose the
foreground bottleneck. Query ground truth is used only after model inference
for diagnostic measurements. Frozen weights and the benchmark are unchanged.
"""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
import hypernet.scene_curriculum as sc
from hypernet.local_relation_model import decoded_boxes,detection_targets,part_pool
from hypernet.local_matcher import FrozenLocalMatcher,nms,box_iou_one
from train_local import setup,collate,sha,atomic_json

PLAN={'scope':'Synthetic DEVELOPMENT diagnostic; no checkpoint selection or benchmark-method change',
      'indices':[0,96],'split':'development','seed':1777,'device':'cpu','threads':2,
      'variants':['correct','wrong_family','one_edit','oracle_extent','oracle_extent_and_weights'],
      'oracle_warning':'Oracle variants use support GT only and are NOT deployable benchmark scores',
      'one_edit_warning':'One-edit reference names a deliberately present hard-negative program; global high score can be correct for that new reference',
      'proposal_limit':100,'pre_nms':1000,'nms_iou':.4}


def redraw_reference(row,program):
    meta=row['metadata'];attempt=meta['ambiguous_nuisance_draws_rejected']
    values=[1777,meta['index'],sc.SPLIT_NUISANCE_CODES['development'],419]
    if attempt:values.append(attempt)
    rng=sc._rng(*values);support_seed=int(rng.integers(1,2**31));rng.integers(1,2**31)
    instance=sc.render_instance(program,support_seed,96)
    rgb=sc._background(rng,96);color=sc._color(rng,rgb);sc._paint(rgb,instance,0,0,color)
    rgb,parts,_=sc._frame_support(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8),instance['parts'],rng,96)
    return rgb,(parts>0).astype(np.uint8)


def reference_variants(row):
    target=row['program'];negative,_=sc.hard_negative(target);replacement,_=sc._second_negative(target,negative)
    assert np.array_equal(redraw_reference(row,target)[0],row['support'])
    # All complete/contained scene programs are reconstructed from deterministic
    # generation metadata, so the wrong reference is truly absent by program.
    scene=[target,negative,replacement]
    for j,record in enumerate(row['metadata']['instances']):
        if record['role']=='distractor':scene.append(sc.sample_program(row['metadata']['index']*13+j+104729,'development',1777))
    for k in range(10000+row['metadata']['index']*31,20000+row['metadata']['index']*31):
        wrong=sc.sample_program(k,'development',1777)
        if wrong['family']!=target['family']and all(not sc.contains_program(p,wrong)for p in scene):break
    else:raise RuntimeError('No absent different-family program')
    return {'correct':row['support'],'wrong_family':redraw_reference(row,wrong)[0],
            'one_edit':redraw_reference(row,negative)[0]},sc.program_id(wrong)


def oracle_bank(encoded,mask,oracle_weights=False):
    # Duplicates only the model's deterministic bank bookkeeping, with oracle
    # bounds (and optionally weights). No network layer or tensor is changed.
    raw=encoded['raw'];logits=encoded['foreground_logits'];h,w=raw.shape[-2:]
    truth=F.interpolate(mask.float(),(h,w),mode='area')
    extent_mask=truth[0,0]>0;ys,xs=extent_mask.nonzero(as_tuple=True)
    if len(xs):
        left,right=int(xs.min()),int(xs.max())+1;top,bottom=int(ys.min()),int(ys.max())+1
        if right-left<4:left=max(0,min(w-4,(left+right-4)//2));right=left+4
        if bottom-top<4:top=max(0,min(h-4,(top+bottom-4)//2));bottom=top+4
    else:left,top,right,bottom=0,0,w,h
    crop=raw[:,:,top:bottom,left:right]
    weight=.05+.95*(truth[:,:,top:bottom,left:right]if oracle_weights else logits[:,:,top:bottom,left:right].sigmoid())
    pooled=part_pool(weight);descriptors=part_pool(crop*weight)/pooled.clamp_min(1e-6)
    return {'parts':F.normalize(descriptors.flatten(2),dim=1,eps=1e-6),
            'weights':pooled.flatten(2).squeeze(1),
            'prototype':F.normalize((crop*weight).sum((-2,-1))/weight.sum((-2,-1)),dim=1,eps=1e-6),
            'predicted_extent':torch.tensor([[left,top,right,bottom]])}


def measure(out,row,hard_negative_box):
    scores=out['center_logits'][0].flatten().numpy();boxes=decoded_boxes(out)[0].numpy().clip(0,192)
    valid=((boxes[:,2:]-boxes[:,:2])>0).all(1)&np.isfinite(scores)
    candidates=np.flatnonzero(valid);candidates=candidates[np.argsort(-scores[candidates],kind='stable')[:1000]]
    selected=candidates[nms(boxes[candidates],scores[candidates],.4,100)]
    ious=box_iou_one(row['bbox'],boxes[selected])if row['present']else np.zeros(len(selected))
    mask=F.interpolate(out['mask_logits'],(192,192),mode='bilinear',align_corners=False).sigmoid()[0,0].numpy()>=.5
    target=row['target_mask']>0
    result={'score':float(scores[selected[0]]),'box':boxes[selected[0]].tolist(),'original_target_top1_iou':float(ious[0]),
            'original_target_max_iou_at100':float(ious.max()),
            'original_target_mask_iou':float((mask&target).sum()/max(1,(mask|target).sum())),
            'one_edit_object_amodal_box_top1_iou':float(box_iou_one(hard_negative_box,boxes[selected[:1]])[0])}
    if row['present']:
        _,anchor,x,y,_=detection_targets(out,torch.from_numpy(row['bbox'])[None],torch.ones(1))
        result['original_target_assigned_center_logit']=float(out['center_logits'][0,anchor[0],y[0],x[0]])
    return result


def aggregate(rows,name):
    positives=[r for r in rows if r['present']];values=[r['variants'][name]for r in positives]
    result={'episodes':len(rows),'positives':len(positives),
        'original_target_top1_recall':float(np.mean([v['original_target_top1_iou']>=.5 for v in values])),
        'original_target_proposal_recall_at100':float(np.mean([v['original_target_max_iou_at100']>=.5 for v in values])),
        'original_target_mask_mean_iou':float(np.mean([v['original_target_mask_iou']for v in values])),
        'mean_top1_score':float(np.mean([r['variants'][name]['score']for r in rows])),
        'mean_original_target_center_logit':float(np.mean([v['original_target_assigned_center_logit']for v in values])),
        'mean_score_minus_correct':float(np.mean([r['variants'][name]['score']-r['variants']['correct']['score']for r in rows])),
        'mean_original_target_center_logit_minus_correct':float(np.mean([r['variants'][name]['original_target_assigned_center_logit']-r['variants']['correct']['original_target_assigned_center_logit']for r in positives])),
        'fraction_original_target_center_logit_below_correct':float(np.mean([r['variants'][name]['original_target_assigned_center_logit']<r['variants']['correct']['original_target_assigned_center_logit']for r in positives])),
        'one_edit_object_amodal_box_top1_recall':float(np.mean([r['variants'][name]['one_edit_object_amodal_box_top1_iou']>=.5 for r in rows]))}
    return result


@torch.inference_mode()
def main():
    output=ROOT/'experiments/conditioning_dev/results.json'
    if output.exists():raise FileExistsError(output)
    setup(1717,2);matcher=FrozenLocalMatcher(ROOT/'runs/primary_2048/final.pt',device='cpu',max_side=192)
    model=matcher.model;rows=[];start=time.perf_counter()
    sources={str(p.relative_to(ROOT)):sha(p)for p in (Path(__file__),ROOT/'hypernet/local_relation_model.py',ROOT/'hypernet/scene_curriculum.py')}
    for index in range(96):
        row=sc.episode(index,split='development',seed=1777);refs,wrong_id=reference_variants(row)
        cpu=collate([row]);query=model.encode(cpu['query']);encoded={}
        for name,rgb in refs.items():encoded[name]=model.encode(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255)
        oracle=oracle_bank(encoded['correct'],cpu['support_mask']);predicted=model.support_bank(encoded['correct'])
        truth=F.interpolate(cpu['support_mask'],encoded['correct']['foreground_logits'].shape[-2:],mode='area')>0
        predicted_mask=encoded['correct']['foreground_logits']>0
        negative,_=sc.hard_negative(row['program']);record=next(r for r in row['metadata']['instances']if r['role']=='hard_negative')
        nx,ny,side,_=record['placement_xywh'];inst=sc.render_instance(negative,record['seed'],side)
        negative_box=sc.bbox_from_mask(inst['mask'])+np.array([nx,ny,nx,ny],np.float32)
        result={'index':index,'family':row['family'],'present':row['present'],'program_id':row['program_id'],'wrong_program_id':wrong_id,
                'support_mask_iou':float((truth&predicted_mask).sum()/(truth|predicted_mask).sum().clamp_min(1)),
                'support_extent_iou':float(box_iou_one(oracle['predicted_extent'][0].numpy(),predicted['predicted_extent'].numpy())[0]),
                'variants':{}}
        banks={name:model.support_bank(value)for name,value in encoded.items()}
        banks['oracle_extent']=oracle;banks['oracle_extent_and_weights']=oracle_bank(encoded['correct'],cpu['support_mask'],True)
        for name,bank in banks.items():result['variants'][name]=measure(model.match_encoded(bank,query),row,negative_box)
        rows.append(result)
        if (index+1)%16==0:print(json.dumps({'episodes':index+1,'planned':96,'seconds':time.perf_counter()-start}),flush=True)
    matcher.assert_unchanged();assert all(sha(ROOT/name)==digest for name,digest in sources.items())
    result={'plan':PLAN,'provenance':matcher.provenance(),'source_sha256':sources,'rows':rows,
            'support_mask_mean_iou':float(np.mean([r['support_mask_iou']for r in rows])),
            'support_extent_mean_iou':float(np.mean([r['support_extent_iou']for r in rows])),
            'metrics':{name:aggregate(rows,name)for name in PLAN['variants']},'seconds':time.perf_counter()-start,'weights_unchanged':True}
    atomic_json(output,result);print(json.dumps({k:v for k,v in result.items()if k in ('metrics','support_mask_mean_iou','support_extent_mean_iou','seconds')}),flush=True)

if __name__=='__main__':main()
