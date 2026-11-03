"""Frozen final-model synthetic diagnosis; never used for checkpoint selection.

One cutoff is selected on256DEV episodes, then applied unchanged to512 whole-
program TEST and512 compositional holdouts. These are synthetic generalization
results, not estimates of COCO/Gravity Spy performance or unlimited categories.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys,argparse,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from hypernet.scene_curriculum import episode
from hypernet.local_relation_model import decoded_boxes
from hypernet.local_matcher import FrozenLocalMatcher
from train_local import setup,collate,atomic_json,sha

PLAN={'format':'hypernet.synthetic_diagnosis.v17.1','seed':1777,'support_size':96,'query_size':192,
      'splits':{'development':256,'test':512,'compositional':512},'nominal_dev_absent_fpr':.05,
      'iou_threshold':.5,'mask_threshold':.5,'checkpoint_selection':False,
      'scope':'Frozen-model synthetic diagnosis; no real performance claim'}


def cutoff(rows):
    negatives=np.asarray([r['score']for r in rows if not r['present']],np.float64)
    # A strict-above-next-value cutoff keeps ties together, with <=floor(.05*N)
    # empirical false alarms. Evaluation uses score>=cutoff, always documented.
    allowed=int(np.floor(PLAN['nominal_dev_absent_fpr']*len(negatives)))
    ordered=np.sort(negatives)[::-1]
    return float(np.nextafter(ordered[allowed],np.inf))


def summarize(rows,threshold):
    positives=[r for r in rows if r['present']];negatives=[r for r in rows if not r['present']]
    p=np.array([r['score']for r in positives]);n=np.array([r['score']for r in negatives])
    true=sum(r['score']>=threshold and r['iou']>=.5 for r in positives)
    false=sum(r['score']>=threshold for r in negatives)
    return {'episodes':len(rows),'positives':len(positives),'negatives':len(negatives),
        'localized_true_positives':true,'false_alarms':false,
        'localized_recall':true/len(positives)if positives else None,
        'false_positive_rate':false/len(negatives)if negatives else None,
        'false_alarms_per_10000':10000*false/len(negatives)if negatives else None,
        'unthresholded_localized_recall':float(np.mean([r['iou']>=.5 for r in positives]))if positives else None,
        'positive_top1_mean_iou':float(np.mean([r['iou']for r in positives]))if positives else None,
        'positive_target_mask_mean_iou':float(np.mean([r['mask_iou']for r in positives]))if positives else None,
        'presence_roc_auc':float(((p[:,None]>n[None]).sum()+.5*(p[:,None]==n[None]).sum())/(len(p)*len(n)))if len(p)*len(n)else None,
        'reference_swap_mean_absolute_logit_change':float(np.mean([r['reference_change']for r in rows]))}


@torch.inference_mode()
def evaluate(matcher,split,count,batch_size):
    rows=[];model=matcher.model;digest=hashlib.sha256()
    for start in range(0,count,batch_size):
        examples=[episode(i,split=split,seed=PLAN['seed'],support_size=96,query_size=192)for i in range(start,min(count,start+batch_size))]
        cpu=collate(examples);b={k:v.to(matcher.device)if isinstance(v,torch.Tensor)else v for k,v in cpu.items()}
        for key in sorted(cpu['input_hashes']):digest.update(key.encode());digest.update(cpu['input_hashes'][key].tobytes())
        out=model(b['support'],b['query']);all_boxes=decoded_boxes(out).clamp(0,192)
        valid=((all_boxes[:,:,2:]-all_boxes[:,:,:2])>0).all(-1)
        scores,indices=out['center_logits'].flatten(1).masked_fill(~valid,-torch.inf).max(1)
        boxes=all_boxes[torch.arange(len(indices),device=indices.device),indices]
        gt=b['bbox'];inter=(torch.minimum(boxes[:,2:],gt[:,2:])-torch.maximum(boxes[:,:2],gt[:,:2])).clamp_min(0).prod(-1)
        union=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(-1)+(gt[:,2:]-gt[:,:2]).prod(-1)-inter
        iou=inter/union.clamp_min(1e-8)
        predicted=torch.nn.functional.interpolate(out['mask_logits'],size=(192,192),mode='bilinear',align_corners=False).sigmoid()>=.5
        target=b['target_mask']>.5;mask_iou=(predicted&target).sum((1,2,3))/(predicted|target).sum((1,2,3)).clamp_min(1)
        # Cycling the support bank changes only the reference, not query features.
        bank=model.support_bank(out['support_encoded']);wrong={key:value.roll(1,0)for key,value in bank.items()}
        changed=model.match_encoded(wrong,out['query_encoded'])['center_logits']
        reference_change=(changed-out['center_logits']).abs().mean((1,2,3))
        for j,e in enumerate(examples):
            rows.append({'index':start+j,'split':split,'program_id':e['program_id'],'family':e['family'],
                         'present':bool(e['present']),'score':float(scores[j]),'box':boxes[j].cpu().tolist(),
                         'truth_box':e['bbox'].tolist(),'iou':float(iou[j]),'mask_iou':float(mask_iou[j]),
                         'reference_change':float(reference_change[j])})
    matcher.assert_unchanged();return rows,digest.hexdigest()


def main(args):
    setup(1717,4);matcher=FrozenLocalMatcher(args.checkpoint,device=args.device,max_side=192)
    destination=ROOT/args.output
    if destination.exists():raise FileExistsError(destination)
    started=time.perf_counter();result={'plan':PLAN,'provenance':matcher.provenance(),'splits':{},'rows':{}}
    threshold=None
    for split,count in PLAN['splits'].items():
        rows,digest=evaluate(matcher,split,count,args.batch_size)
        if split=='development':threshold=cutoff(rows)
        metrics=summarize(rows,threshold);metrics['families']={family:summarize([r for r in rows if r['family']==family],threshold)for family in sorted({r['family']for r in rows})}
        result['splits'][split]={'input_stream_sha256':digest,'threshold':threshold,'metrics':metrics};result['rows'][split]=rows
        print(json.dumps({'split':split,**{k:metrics[k]for k in ('localized_recall','false_positive_rate','unthresholded_localized_recall','presence_roc_auc')}}),flush=True)
    result['seconds']=time.perf_counter()-started
    result['source_sha256']={str(p.relative_to(ROOT)):sha(p)for p in (Path(__file__),ROOT/'hypernet/scene_curriculum.py',ROOT/'hypernet/local_relation_model.py',ROOT/'scripts/train_local.py')}
    atomic_json(destination,result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16);main(p.parse_args())
