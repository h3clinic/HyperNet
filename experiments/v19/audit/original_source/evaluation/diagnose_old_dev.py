"""Development-only resolution/coverage/oracle diagnostic on the first48 old photos.

Oracle boxes are annotation-assisted diagnostic inputs, never normal detections.
No fresh TEST image loader or acquisition path is imported or called here.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from data_protocol import PLAN,load_plan,load_old,load_reference,sha,write_new,read
from evaluate import calibration,predict_references
from vendor_v17.detection_metrics import clip_box,iou,truth_arrays
from hypernet_v19.matcher import FrozenDenseMatcher,prepare_image
from hypernet_v19.proposals import box_iou


def atomic_progress(path,value):
    temp=path.with_suffix('.writing');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


@torch.inference_mode()
def oracle_scores(matcher,banks,image,boxes):
    """Score supplied GT regions without inserting them in ordinary proposals."""
    if not len(boxes):return np.empty((len(banks),0),np.float32)
    canvas,meta=prepare_image(image,max_side=matcher.max_side)
    regions=np.asarray(boxes,np.float32).copy()
    regions[:,(0,2)]*=meta['scale_x'];regions[:,(1,3)]*=meta['scale_y']
    tensor=torch.from_numpy(canvas.copy()).permute(2,0,1)[None].to(matcher.device,dtype=torch.float32)/255
    pyramid=matcher.model.encode_query(tensor)
    support={key:torch.cat([bank['bank'][key]for bank in banks],0)for key in banks[0]['bank']}
    scores=[]
    for start in range(0,len(regions),matcher.region_batch):
        sampled=matcher.model.sample_query_regions(pyramid,torch.from_numpy(regions[start:start+matcher.region_batch]).to(matcher.device)[None])
        sampled={key:value.expand(len(banks),*value.shape[1:])for key,value in sampled.items()}
        scores.append(matcher.model.score_regions(support,sampled).float().cpu().numpy())
    result=np.concatenate(scores,1)
    if not np.isfinite(result).all():raise RuntimeError('Nonfinite oracle score')
    return result


def run(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=load_plan();rows=plan['dev'][:48];targets=plan['targets'];destination=args.output.resolve()
    if not destination.is_relative_to(ROOT.resolve()):raise ValueError('Diagnostic output must be inside v19')
    if len(rows)!=48:raise ValueError('Expected fixed first48 old DEV photos')
    matcher=FrozenDenseMatcher(args.checkpoint,device=args.device,max_side=args.max_side)
    references={row['category_id']:load_reference(row)for row in plan['references']}
    banks=[matcher.enroll(references[target['category_id']])for target in targets]
    bound={str(path.relative_to(ROOT)):sha(path)for path in [ROOT/'hypernet_v19/model.py',ROOT/'hypernet_v19/matcher.py',ROOT/'hypernet_v19/proposals.py',Path(__file__),HERE/'evaluate.py']}
    config={'scope':'Exploratory old DEV only; no weight updates; no fresh TEST photos',
      'roster':'Exact first48 rows of the existing288-photo DEV plan',
      'data_plan_sha256':sha(PLAN),'image_ids':[row['image_id']for row in rows],
      'targets':targets,'provenance':matcher.provenance(),'source_sha256':bound,
      'oracle_warning':'GT boxes supplied only to the separately named oracle diagnostic. They never enter normal proposal generation/detection.',
      'calibration_warning':'Small-pilot cutoffs fit and evaluate the same48 DEV photos; not heldout accuracy or final calibration.'}
    config=json.loads(json.dumps(config))
    if destination.exists():
        if not args.resume:raise FileExistsError('Preserve existing pilot output; --resume requires exact admission')
        if read(destination/'admission.json')!=config:raise ValueError('Resume admission/source/checkpoint differs')
        if(destination/'result.json').exists():raise FileExistsError('Completed pilot is immutable')
    else:destination.mkdir(parents=True);write_new(destination/'admission.json',config)
    start=time.monotonic();measured=[]
    for number,row in enumerate(rows):
        path=destination/f"photo_{row['image_id']:012d}.json"
        if path.exists():
            result=read(path)
            if result['image_id']!=row['image_id']or result['index']!=number:raise ValueError('Saved pilot photo mismatch')
            measured.append(result);continue
        image=load_old(row);encoded=matcher.encode_query(image);proposals=predict_references(matcher,banks,encoded)
        oracle_boxes=[]
        for target in targets:
            if row['valid_by_target'][str(target['category_id'])]:
                oracle_boxes.extend(row['boxes_by_target'][str(target['category_id'])])
        oracle_boxes=np.unique(np.asarray(oracle_boxes,np.float32).reshape(-1,4),axis=0)
        oscores=oracle_scores(matcher,banks,image,oracle_boxes)
        labels=np.zeros((len(targets),len(oracle_boxes)),bool);pairs=[]
        for j,target in enumerate(targets):
            cid=str(target['category_id']);truth=row['boxes_by_target'][cid]
            if len(oracle_boxes)and truth:labels[j]=box_iou(oracle_boxes,truth).max(1)>=.5
            top=dict(max(proposals[j],key=lambda item:item['score']));top['raw_box']=list(top['box'])
            top['box']=clip_box(top['box'],image.shape[1],image.shape[0])
            if not np.isfinite([top['score'],*top['box']]).all():raise RuntimeError('Nonfinite ordinary prediction')
            pairs.append({'category_id':target['category_id'],'valid':row['valid_by_target'][cid],
              'present':bool(truth),'score':top['score'],'box':top['box'],'raw_box':top['raw_box'],'iou':iou(top['box'],truth),
              'proposal_max_iou':float(box_iou(encoded['boxes'],truth).max())if truth else 0.,
              'oracle_best_true_box_score':float(oscores[j,labels[j]].max())if labels[j].any()else None})
        result={'image_id':row['image_id'],'index':number,'generic_proposal_count':encoded['proposal_count'],'pairs':pairs,
          'oracle_boxes':oracle_boxes.tolist(),'oracle_scores_by_reference':oscores.tolist(),'oracle_type_labels':labels.tolist()}
        write_new(path,result);measured.append(result)
        progress={'status':'running','photos':number+1,'total':len(rows),'max_side':args.max_side,'seconds_this_session':time.monotonic()-start}
        atomic_progress(destination/'progress.json',progress)
        if(number+1)%4==0:print(json.dumps(progress),flush=True)
    matcher.assert_unchanged()
    if any(sha(ROOT/name)!=expected for name,expected in bound.items()):raise RuntimeError('Diagnostic source changed')
    scores=np.array([[pair['score']for pair in row['pairs']]for row in measured]);overlaps=np.array([[pair['iou']for pair in row['pairs']]for row in measured])
    coverage=np.array([[pair['proposal_max_iou']for pair in row['pairs']]for row in measured]);present,valid=truth_arrays(rows,targets)
    positive=present&valid;per_target=[]
    for j,target in enumerate(targets):
        mask=positive[:,j];n=int(mask.sum())
        per_target.append({**target,'positives':n,'proposal_coverage_iou50':float((coverage[mask,j]>=.5).mean())if n else None,
          'uncut_top1_localized_recall':float((overlaps[mask,j]>=.5).mean())if n else None})
    oracle_good=oracle_total=oracle_beats_wrong=wrong_top1_count=0
    for original,record in zip(rows,measured):
        labels=np.asarray(record['oracle_type_labels'],bool);oscores=np.asarray(record['oracle_scores_by_reference'],float)
        admissible=np.array([original['valid_by_target'][str(t['category_id'])]for t in targets],bool)
        for k in range(len(record['oracle_boxes'])):
            if labels[admissible,k].any():
                ranked=np.where(admissible,oscores[:,k],-np.inf);oracle_good+=int(labels[np.argmax(ranked),k]);oracle_total+=1
        for pair in record['pairs']:
            if pair['valid']and pair['present']and pair['iou']<.5 and pair['oracle_best_true_box_score']is not None:
                wrong_top1_count+=1;oracle_beats_wrong+=int(pair['oracle_best_true_box_score']>pair['score'])
    summary={'admission':config,'pilot_calibration':calibration(scores,overlaps,rows,targets),'per_target':per_target,
      'proposal_coverage_micro':float((coverage[positive]>=.5).mean()),'uncut_top1_recall_micro':float((overlaps[positive]>=.5).mean()),
      'oracle_type_classification':{'correct':oracle_good,'total_gt_boxes':oracle_total,'accuracy':oracle_good/oracle_total if oracle_total else None},
      'oracle_score_beats_wrong_generic_top1':{'count':oracle_beats_wrong,'wrong_top1_positive_pairs':wrong_top1_count},
      'photo_file_sha256':{f"photo_{r['image_id']:012d}.json":sha(destination/f"photo_{r['image_id']:012d}.json")for r in rows},
      'seconds_this_session':time.monotonic()-start,'fresh_test_photos_accessed':0,'real_weight_updates':0}
    write_new(destination/'result.json',summary);atomic_progress(destination/'progress.json',{'status':'complete','photos':48,'total':48})
    print(json.dumps({key:summary[key]for key in('proposal_coverage_micro','uncut_top1_recall_micro','oracle_type_classification','oracle_score_beats_wrong_generic_top1','seconds_this_session')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-side',type=int,choices=[384,640],default=384)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--resume',action='store_true');run(parser.parse_args())
