"""Fixed48-photo oldDEV pilot only. Never reads the mixed plan or real TEST."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));WORKSPACE=ROOT.parents[1]
import numpy as np
import torch
from PIL import Image
from hypernet_v18.reranker import FrozenPatchReranker
from hypernet_v18.pair_ranker import file_sha
from hypernet_v18.vendor_v17.detection_metrics import iou,clip_box,truth_arrays,summarize,auc,global_cutoff,COMMON_BOX_RULE
from train_pair_ranker import atomic_json

MANIFEST_SHA='c773ac5592d0f1dcfe4a61f623077f7e58f22aa4a7a5401587b417cab80ca9cf'


def load_rgb(row):
    path=WORKSPACE/row['workspace_path']
    if file_sha(path)!=row['sha256']:raise ValueError('OldDEV source hash changed')
    with Image.open(path)as image:
        if image.getexif().get(274,1)!=1:raise ValueError('Unexpected orientation')
        rgb=np.asarray(image.convert('RGB')).copy()
    digest=hashlib.sha256(str(rgb.dtype).encode()+str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    if digest!=row['pixel_sha256']:raise ValueError('OldDEV pixel hash changed')
    return rgb


def support(row):
    rgb=load_rgb(row);x1,y1,x2,y2=row['bbox_xyxy']
    left,top=max(0,int(np.floor(x1))),max(0,int(np.floor(y1)))
    right,bottom=min(rgb.shape[1],int(np.ceil(x2))),min(rgb.shape[0],int(np.ceil(y2)))
    if right<=left or bottom<=top:raise ValueError('Empty reference box')
    return rgb[top:bottom,left:right].copy()


def main():
    manifest=ROOT/'data/real_dev_only.json'
    if file_sha(manifest)!=MANIFEST_SHA:raise ValueError('DEV-only manifest changed')
    plan=json.loads(manifest.read_text());assert 'test'not in plan and len(plan['references'])==12
    assert all(row['split']=='dev'for row in plan['dev'])
    rows=sorted(plan['dev'],key=lambda r:r['image_id'])[:48]
    assert [r['image_id']for r in rows]==plan['pilot_selection']['image_ids']
    targets=plan['targets'];category_ids=[r['category_id']for r in targets]
    output=ROOT/'experiments/real_dev_pilot'
    if output.exists()and any(output.iterdir()):raise FileExistsError('Preserve previous pilot')
    output.mkdir(parents=True,exist_ok=True)
    baseline=ROOT.parent/'v17/experiments/v17_dev_v2'
    baseline_result=json.loads((baseline/'result.json').read_text())
    assert file_sha(baseline/'arrays.npz')==baseline_result['score_file_sha256']
    with np.load(baseline/'arrays.npz',allow_pickle=False)as z:
        assert list(z['category_ids'])==category_ids
        lookup={int(v):i for i,v in enumerate(z['image_ids'])};selection=[lookup[r['image_id']]for r in rows]
        base_scores=z['scores'][selection].copy();base_overlaps=z['overlaps'][selection].copy();base_boxes=z['boxes'][selection].copy()
    present,valid=truth_arrays(rows,targets)
    # Verify stored baseline IoU against the DEV-only annotations before reuse.
    for i,row in enumerate(rows):
        for j,target in enumerate(targets):
            assert abs(iou(base_boxes[i,j],row['boxes_by_target'][str(target['category_id'])])-base_overlaps[i,j])<1e-5
    admission={'scope':'EXPLORATORY DEVELOPMENT only; not fresh validation','manifest_sha256':MANIFEST_SHA,
        'selection':plan['pilot_selection'],'checkpoint_rule':'Frozen final512updates; no retraining or checkpoint selection',
        'metrics':'Ranking AUROC and unthresholded localization; within-pilot5%absentFPR calibration is descriptive only',
        'baseline_arrays_sha256':file_sha(baseline/'arrays.npz'),'baseline_result_sha256':file_sha(baseline/'result.json'),
        'evaluation_source_sha256':file_sha(__file__),'common_box_rule':COMMON_BOX_RULE,
        'max_side':384,'proposals':100,'patch_batch':32,'real_weight_updates':0,'real_TEST_access':False}
    atomic_json(output/'admission.json',admission)
    torch.set_num_threads(2);torch.manual_seed(1801);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=FrozenPatchReranker(ROOT/'runs/proof512/final.pt',ROOT.parent/'v17/runs/primary_2048/final.pt',
                             device='cuda',max_side=384,proposal_limit=100,patch_batch=32)
    ref_map={r['category_id']:r for r in plan['references']}
    banks=[model.enroll(support(ref_map[target['category_id']]))for target in targets]
    scores=np.zeros((48,12),np.float64);overlaps=np.zeros_like(scores);boxes=np.zeros((48,12,4),np.float64)
    proposal_overlap=np.zeros_like(scores);start=time.perf_counter()
    with(output/'predictions.jsonl').open('x')as history:
        for i,row in enumerate(rows):
            rgb=load_rgb(row);query=model.encode_query(rgb);predictions=model.predict_many(banks,query)
            for j,target in enumerate(targets):
                best=predictions[j][0];box=clip_box(best['box'],row['width'],row['height']);truth=row['boxes_by_target'][str(target['category_id'])]
                scores[i,j]=best['score'];boxes[i,j]=box;overlaps[i,j]=iou(box,truth)
                proposal_overlap[i,j]=max(iou(p['box'],truth)for p in predictions[j])
                history.write(json.dumps({'image_id':row['image_id'],'category_id':target['category_id'],'score':best['score'],'box':box,
                                          'overlap':overlaps[i,j],'proposal_overlap':proposal_overlap[i,j]})+'\n')
            if (i+1)%8==0:
                history.flush();elapsed=time.perf_counter()-start;estimate=elapsed/(i+1)*48
                progress={'photos':i+1,'planned':48,'seconds':elapsed,'projected_total_seconds':estimate,
                          'unique_query_crop_features_last_photo':len(query['patch_tokens'])}
                atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
                if i==7 and estimate>600:raise RuntimeError('First8 timing projects>10minutes; coordinate with root before continuing')
    model.assert_unchanged()
    np.savez_compressed(output/'arrays.npz',scores=scores,overlaps=overlaps,boxes=boxes,proposal_overlap=proposal_overlap,
                        baseline_scores=base_scores,baseline_overlaps=base_overlaps,baseline_boxes=base_boxes,
                        image_ids=[r['image_id']for r in rows],category_ids=category_ids,present=present,valid=valid)
    results={}
    for name,score,overlap in [('v17_saved_dev_subset',base_scores,base_overlaps),('v18_proof512',scores,overlaps)]:
        threshold=global_cutoff(score[valid&~present],.05)
        metrics=summarize(score,overlap,present,valid,threshold,targets)
        metrics['micro_presence_auroc']=auc(present[valid],score[valid])
        metrics['micro_localized_recall']=metrics['localized_hits']/int((valid&present).sum())
        metrics['unthresholded_micro_localized_recall']=float((overlap[valid&present]>=.5).mean())
        for j,item in enumerate(metrics['per_target']):
            active=valid[:,j]&present[:,j]
            item['unthresholded_localized_recall']=float((overlap[active,j]>=.5).mean())if active.any()else None
        results[name]=metrics
    result={'scope':admission['scope'],'admission':admission,'metrics':results,
            'proposal_recall_at100':float((proposal_overlap[valid&present]>=.5).mean()),
            'seconds':time.perf_counter()-start,'provenance':model.provenance(),
            'arrays_sha256':file_sha(output/'arrays.npz'),'predictions_sha256':file_sha(output/'predictions.jsonl')}
    atomic_json(output/'result.json',result);print(json.dumps({'complete':True,'seconds':result['seconds'],
        'metrics':{name:{k:m[k]for k in ('micro_presence_auroc','macro_presence_auroc','micro_localized_recall','micro_false_positive_rate','unthresholded_micro_localized_recall')}for name,m in results.items()}}),flush=True)

if __name__=='__main__':main()
