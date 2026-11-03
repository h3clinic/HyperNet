"""Old-DEV isolated-crop and reference-self diagnostics, with no natural training."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from data_protocol import PLAN,load_plan,load_old,load_reference,sha,write_new
from diagnose_old_dev import oracle_scores
from hypernet_v19.matcher import FrozenDenseMatcher

CASE_SHA='78effc35fe7a902fe94bff1c585e97ca96c61aead8ff2510c01ca14034cc73c0'


def crop(rgb,box):
    x1,y1,x2,y2=box
    integer=[max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),min(rgb.shape[1],int(np.ceil(x2))),min(rgb.shape[0],int(np.ceil(y2)))]
    x1,y1,x2,y2=integer
    if x2<=x1 or y2<=y1:raise ValueError('Empty oracle crop')
    return rgb[y1:y2,x1:x2],integer


def intersects(a,b):return min(a[2],b[2])>max(a[0],b[0])and min(a[3],b[3])>max(a[1],b[1])


def measures(matrix,cases):
    successes=[];pairwise=[];true=[];wrong=[];ranks=[]
    for scores,case in zip(matrix,cases):
        t=float(scores[case['target_index']]);n=np.asarray(scores)[case['negative_indices']]
        if not len(n):raise ValueError('No eligible wrong-reference comparisons')
        true.append(t);wrong.extend(n.tolist());successes.append(bool(t>n.max()));pairwise.append(float(np.mean(t>n)));ranks.append(int(1+(n>=t).sum()))
    negatives=np.sort(wrong);positives=np.asarray(true)
    return {'correct_top1':sum(successes),'cases':len(cases),'top1_accuracy':float(np.mean(successes)),
      'mean_pairwise_correct':float(np.mean(pairwise)),
      'pooled_correct_vs_wrong_reference_auroc':float(np.mean((np.searchsorted(negatives,positives,'left')+np.searchsorted(negatives,positives,'right'))/(2*len(negatives)))),
      'median_true_rank':float(np.median(ranks)),'mean_true_score':float(np.mean(true)),'mean_wrong_score':float(np.mean(wrong)),
      'mean_eligible_wrong_references':float(np.mean([len(case['negative_indices'])for case in cases]))}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    destination=args.output.resolve()
    if not destination.is_relative_to(ROOT.resolve()):raise ValueError('Output must be inside v19')
    if destination.exists():raise FileExistsError('Preserve existing control result')
    plan=load_plan();targets=plan['targets'];rows=plan['dev'][:48]
    model=FrozenDenseMatcher(args.checkpoint,device=args.device,max_side=args.max_side)
    refmap={row['category_id']:load_reference(row)for row in plan['references']};refs=[refmap[t['category_id']]for t in targets]
    banks=[model.enroll(ref)for ref in refs];patches=[];cases=[]
    for row in rows:
        rgb=load_old(row)
        for j,target in enumerate(targets):
            key=str(target['category_id']);truth=row['boxes_by_target'][key]
            if not row['valid_by_target'][key]or not truth:continue
            gt=max(truth,key=lambda b:(b[2]-b[0])*(b[3]-b[1]));patch,box=crop(rgb,gt)
            eligible=[k for k,t in enumerate(targets)if k!=j and row['valid_by_target'][str(t['category_id'])]and all(not intersects(box,b)for b in row['boxes_by_target'][str(t['category_id'])])]
            cases.append({'image_id':row['image_id'],'target_index':j,'category_id':target['category_id'],'box':box,'negative_indices':eligible});patches.append(patch)
    case_sha=hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if len(cases)!=42 or case_sha!=CASE_SHA:raise ValueError('Cases differ from the exact42-case v18 old-DEV oracle audit')
    sources={str(path.relative_to(ROOT)):sha(path)for path in [Path(__file__),HERE/'diagnose_old_dev.py',ROOT/'hypernet_v19/model.py',ROOT/'hypernet_v19/matcher.py']}
    destination.mkdir(parents=True);start=time.monotonic()
    support={key:torch.cat([bank['bank'][key]for bank in banks],0)for key in banks[0]['bank']}
    def score(images):
        pipeline=[];symmetric=[]
        for rgb in images:
            pipeline.append(oracle_scores(model,banks,rgb,[[0,0,rgb.shape[1],rgb.shape[0]]])[:,0])
            # Both branches use the identical predicted-extent support encoding.
            # This is an explicit representation control, not deployed detection.
            q=model.enroll(rgb)['bank'];regions={key:value.expand(len(banks),*value.shape[1:])for key,value in q.items()}
            symmetric.append(model.model.score_regions(support,regions).float().cpu().numpy()[:,0])
        return np.stack(pipeline),np.stack(symmetric)
    pipeline,symmetric=score(patches);self_pipeline,self_symmetric=score(refs)
    if not all(np.isfinite(value).all()for value in(pipeline,symmetric,self_pipeline,self_symmetric)):raise ValueError('Nonfinite controls')
    model.assert_unchanged()
    if any(sha(ROOT/name)!=value for name,value in sources.items()):raise ValueError('Control source changed during inference')
    self_cases=[{'target_index':j,'negative_indices':[k for k in range(len(targets))if k!=j]}for j in range(len(targets))]
    np.savez_compressed(destination/'scores.npz',pipeline=pipeline,symmetric=symmetric,self_pipeline=self_pipeline,self_symmetric=self_symmetric)
    result={'scope':'Exploratory old48DEV and12existing references only. Explicit GT crops and self controls; not deployed detection or heldout accuracy.',
      'case_selection':'Largest GT crop per positive valid target/photo pair; wrong references exclude every category whose GT intersects that crop, and invalid/crowd categories.',
      'cases_sha256':case_sha,'cases_match_v18_oracle_audit_exactly':True,'data_plan_sha256':sha(PLAN),'source_sha256':sources,
      'provenance':model.provenance(),'cases':cases,
      'methods':{'isolated_crop_pipeline':measures(pipeline,cases),'symmetric_predicted_extent_control':measures(symmetric,cases)},
      'reference_self':{'isolated_crop_pipeline':measures(self_pipeline,self_cases),'symmetric_predicted_extent_control':measures(self_symmetric,self_cases)},
      'mode_definitions':{'isolated_crop_pipeline':'Standard v19 query preprocessing/pyramid; one full-crop oracle ROI; standard reference bank.',
        'symmetric_predicted_extent_control':'Query crop encoded exactly as support; predicted-foreground extent for both branches; same learned scoring head.'},
      'arrays_sha256':sha(destination/'scores.npz'),'seconds':time.monotonic()-start,'fresh_test_photos_accessed':0,'real_weight_updates':0}
    write_new(destination/'result.json',result)
    print(json.dumps({key:result[key]for key in('methods','reference_self','seconds')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--max-side',type=int,choices=[384,640],default=384);run(p.parse_args())
