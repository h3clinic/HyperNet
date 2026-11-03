"""Frozen full-DEV ranking/box-size diagnosis only; never opens v19 TEST metadata or pixels."""
from pathlib import Path
import hashlib,json,math
import numpy as np

BASE=Path(__file__).resolve().parents[2];ROOT=BASE/'outputs/hypernet_v19';HERE=Path(__file__).resolve().parent
METHODS=('v19','v17','raw_template','owlvit_base_patch32')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def q(values):
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()),'min':float(a.min()),'p10':float(np.quantile(a,.1)),'p25':float(np.quantile(a,.25)),'median':float(np.median(a)),'p75':float(np.quantile(a,.75)),'p90':float(np.quantile(a,.9)),'max':float(a.max())}if len(a)else None
def geometry(box):
    box=np.asarray(box,float);wh=np.maximum(box[2:]-box[:2],0)
    return float(wh.prod()),float(wh[0]/max(wh[1],1e-12)),(box[:2]+box[2:])/2
def overlaps(box,truth):
    box=np.asarray(box,float);truth=np.asarray(truth,float).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(box[2:],truth[:,2:])-np.maximum(box[:2],truth[:,:2])).prod(1)
    target_area=np.maximum(truth[:,2:]-truth[:,:2],0).prod(1);pred_area=geometry(box)[0]
    return intersection/np.maximum(pred_area+target_area-intersection,1e-15),intersection,target_area
def main():
    manifest_path=BASE/'outputs/hypernet_v18/data/real_dev_only.json';manifest=json.loads(manifest_path.read_text());assert 'test'not in manifest
    by_id={r['image_id']:r for r in manifest['dev']};targets=manifest['targets'];summaries={};all_records={};bindings={};shared_shape=None
    for method in METHODS:
        directory=ROOT/'evaluation/results'/f'{method}_dev';path=directory/'arrays.npz';result=json.loads((directory/'result.json').read_text());cal=json.loads((directory/'calibration.json').read_text())
        assert sha(path)==result['score_file_sha256'];bindings[method]={'arrays_sha256':sha(path),'result_sha256':sha(directory/'result.json'),'calibration_sha256':sha(directory/'calibration.json')}
        a=dict(np.load(path,allow_pickle=False));assert set(map(int,a['image_ids']))==set(by_id)and len(a['image_ids'])==288
        assert a['category_ids'].tolist()==[t['category_id']for t in targets]
        rows=[by_id[int(i)]for i in a['image_ids']];present=np.array([[bool(r['boxes_by_target'][str(t['category_id'])])for t in targets]for r in rows]);valid=np.array([[r['valid_by_target'][str(t['category_id'])]for t in targets]for r in rows]);pos=present&valid;neg=~present&valid
        assert int(pos.sum())==255 and int(neg.sum())==3197
        cutoff=cal['primary_5pct']['cutoff'];records=[];all_area=[];all_aspect=[];all_score=[];box_diversity=[];recomputed=np.zeros_like(a['overlaps'])
        for i,row in enumerate(rows):
            image_area=row['width']*row['height'];box_diversity.append(len(np.unique(np.round(a['boxes'][i],3),axis=0)))
            for j,target in enumerate(targets):
                box=a['boxes'][i,j];area,aspect,center=geometry(box)
                if valid[i,j]:all_area.append(area/image_area);all_aspect.append(aspect);all_score.append(float(a['scores'][i,j]))
                truth=row['boxes_by_target'][str(target['category_id'])]
                if not truth:continue
                io,intersection,truth_area=overlaps(box,truth);recomputed[i,j]=float(io.max())
                if not pos[i,j]:continue
                best=int(io.argmax())if io.max()>0 else int(truth_area.argmax());largest=int(truth_area.argmax());gt=np.asarray(truth[best],float);big=np.asarray(truth[largest],float)
                gt_area,gt_aspect,gt_center=geometry(gt);largest_area,largest_aspect,_=geometry(big)
                center_inside=any(b[0]<=center[0]<=b[2]and b[1]<=center[1]<=b[3]for b in truth)
                records.append({'image_id':row['image_id'],'target_index':j,'category_id':target['category_id'],'name':target.get('name',target.get('category_name')),'score':float(a['scores'][i,j]),'accepted':bool(a['scores'][i,j]>=cutoff),'iou':float(io.max()),'unthresholded_hit':bool(io.max()>=.5),'pred_area_fraction':area/image_area,'pred_aspect':aspect,'largest_gt_area_fraction':largest_area/image_area,'largest_gt_aspect':largest_aspect,'best_iou_gt_area_fraction':gt_area/image_area,'best_iou_gt_aspect':gt_aspect,'pred_area_to_largest_gt':area/max(largest_area,1e-15),'pred_area_to_best_iou_gt':area/max(gt_area,1e-15),'aspect_ratio_to_best_iou_gt':aspect/max(gt_aspect,1e-15),'absolute_log_aspect_error':abs(math.log(max(aspect,1e-15)/max(gt_aspect,1e-15))),'center_distance_over_gt_diagonal':float(np.linalg.norm(center-gt_center)/max(np.linalg.norm(gt[2:]-gt[:2]),1e-15)),'prediction_center_inside_any_gt':bool(center_inside),'any_intersection':bool(intersection.max()>0),'best_gt_fraction_covered':float(intersection[best]/max(gt_area,1e-15)),'prediction_fraction_on_best_gt':float(intersection[best]/max(area,1e-15)),'gt_object_count':len(truth),'pred_box':box.tolist(),'best_iou_gt_box':gt.tolist()})
        assert np.max(abs(recomputed-a['overlaps']))<2e-12
        fields=['pred_area_fraction','pred_aspect','largest_gt_area_fraction','largest_gt_aspect','best_iou_gt_area_fraction','pred_area_to_largest_gt','pred_area_to_best_iou_gt','aspect_ratio_to_best_iou_gt','absolute_log_aspect_error','center_distance_over_gt_diagonal','best_gt_fraction_covered','prediction_fraction_on_best_gt']
        size_bins=[]
        for low,high in [(0,.01),(.01,.05),(.05,.15),(.15,.4),(.4,1.000001)]:
            selected=[r for r in records if low<=r['largest_gt_area_fraction']<high]
            size_bins.append({'largest_gt_area_fraction_range':[low,high],'positive_pairs':len(selected),'unthresholded_hits':sum(r['unthresholded_hit']for r in selected),'calibrated_hits':sum(r['unthresholded_hit']and r['accepted']for r in selected),'median_pred_to_largest_gt_area':float(np.median([r['pred_area_to_largest_gt']for r in selected]))if selected else None})
        per_target=[]
        for target in targets:
            group=[r for r in records if r['category_id']==target['category_id']]
            per_target.append({'category_id':target['category_id'],'name':target.get('name',target.get('category_name')),'positive_pairs':len(group),'unthresholded_hits':sum(r['unthresholded_hit']for r in group),'calibrated_hits':sum(r['unthresholded_hit']and r['accepted']for r in group),'median_pred_area_fraction':float(np.median([r['pred_area_fraction']for r in group])),'median_largest_gt_area_fraction':float(np.median([r['largest_gt_area_fraction']for r in group]))})
        summary={'positive_pairs':255,'negative_pairs':3197,'primary_cutoff':cutoff,'unthresholded_iou50_hits':sum(r['unthresholded_hit']for r in records),'calibrated_iou50_hits':sum(r['unthresholded_hit']and r['accepted']for r in records),'accepted_positive_pairs':sum(r['accepted']for r in records),'false_alarms':int(((a['scores']>=cutoff)&neg).sum()),'positive_pair_distributions':{f:q([r[f]for r in records])for f in fields},'all_valid_pred_area_fraction':q(all_area),'all_valid_pred_aspect':q(all_aspect),'unique_top1_boxes_among12_references_per_photo':q(box_diversity),'score_vs_log_area_correlation':float(np.corrcoef(all_score,np.log(np.maximum(all_area,1e-12)))[0,1]),'pred_area_over_4x_largest_gt':sum(r['pred_area_to_largest_gt']>4 for r in records),'pred_area_below_quarter_largest_gt':sum(r['pred_area_to_largest_gt']<.25 for r in records),'prediction_center_inside_any_gt':sum(r['prediction_center_inside_any_gt']for r in records),'any_intersection_with_gt':sum(r['any_intersection']for r in records),'covers_at_least90pct_of_gt_but_gt_at_most25pct_of_prediction':sum(r['best_gt_fraction_covered']>=.9 and r['prediction_fraction_on_best_gt']<=.25 for r in records),'inside_gt_at_least90pct_but_covers_at_most25pct_of_gt':sum(r['prediction_fraction_on_best_gt']>=.9 and r['best_gt_fraction_covered']<=.25 for r in records),'size_bins':size_bins,'per_target':per_target}
        assert summary['calibrated_iou50_hits']==result['summary']['primary_5pct']['localized_hits']and summary['false_alarms']==result['summary']['primary_5pct']['false_presence']
        summaries[method]=summary;all_records[method]=records
    report={'scope':'Post-freeze diagnosis of existing full288 DEV arrays only. All metadata from v18 real_dev_only.json; no v19 TEST plan/results or images opened. No tuning, inference, rendering, GPU or behavioral edits.','old_dev_manifest_sha256':sha(manifest_path),'methods':summaries,'input_bindings':bindings,'interpretation_limits':['Unthresholded top1 IoU is the ranking ceiling for these saved boxes; it is not oracle proposal coverage.','GT size distributions use the largest annotation per positive target/photo unless explicitly labeled best-IoU GT. Best-IoU ties with zero overlap choose largest GT.','Observed size/shape biases diagnose associations and do not establish a causal remedy.'],'records':all_records}
    path=HERE/'full_dev_ranking_diagnosis.json'
    with path.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({name:{k:v for k,v in s.items()if k not in ['positive_pair_distributions','all_valid_pred_area_fraction','all_valid_pred_aspect','size_bins','per_target']}for name,s in summaries.items()},indent=2))
    print(json.dumps({name:{f:s['positive_pair_distributions'][f]['median']for f in ['pred_area_fraction','largest_gt_area_fraction','pred_aspect','largest_gt_aspect','pred_area_to_largest_gt','absolute_log_aspect_error']}for name,s in summaries.items()},indent=2))
if __name__=='__main__':main()
