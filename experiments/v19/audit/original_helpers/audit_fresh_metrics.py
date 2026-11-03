"""Independent post-GO audit of frozen v19 metrics, boxes, calibration and paired CIs.

No model/project metric imports, image decoding, inference or network calls.
Reading fresh TEST records requires an explicit --authorized flag after root GO.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'outputs/hypernet_v19';EVAL=ROOT/'evaluation'
METHODS=('v19','v17','raw_template','owlvit_base_patch32')
POINTS={'primary_5pct':.05,'secondary_1pct':.01}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def iou_one(box,truth):
    best=0.;left,top,right,bottom=map(float,box);area=max(0.,right-left)*max(0.,bottom-top)
    for a,b,c,d in truth:
        intersection=max(0.,min(right,c)-max(left,a))*max(0.,min(bottom,d)-max(top,b))
        union=area+max(0.,c-a)*max(0.,d-b)-intersection
        best=max(best,intersection/max(union,1e-15))
    return best
def rank_auc(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float);p=int(labels.sum());n=len(labels)-p
    if not p or not n:return None
    # Mann–Whitney statistic with average ranks for exact score ties.
    order=np.argsort(scores,kind='stable');rank_sum=0.;start=0
    while start<len(order):
        stop=start+1
        while stop<len(order)and scores[order[stop]]==scores[order[start]]:stop+=1
        rank_sum+=float(labels[order[start:stop]].sum())*(start+1+stop)/2;start=stop
    return (rank_sum-p*(p+1)/2)/(p*n)
def labels(rows,targets):
    p=np.array([[len(r['boxes_by_target'][str(t['category_id'])])>0 for t in targets]for r in rows],bool)
    v=np.array([[bool(r['valid_by_target'][str(t['category_id'])])for t in targets]for r in rows],bool)
    return p,v
def cutoff(scores,budget):
    scores=np.sort(np.asarray(scores,np.float64))[::-1];allowed=math.floor(len(scores)*budget)
    return float(np.nextafter(scores[allowed],float('inf')))
def summary(scores,overlap,present,valid,threshold,targets):
    accepted=scores>=threshold;details=[]
    for j,target in enumerate(targets):
        pos=np.logical_and(valid[:,j],present[:,j]);neg=np.logical_and(valid[:,j],~present[:,j]);p=int(pos.sum());n=int(neg.sum())
        tp=int(accepted[pos,j].sum());fp=int(accepted[neg,j].sum());hit=int(np.logical_and(accepted[pos,j],overlap[pos,j]>=.5).sum())
        recall=tp/p if p else None;fpr=fp/n if n else None
        details.append({'category_id':target['category_id'],'name':target.get('category_name',target.get('name')),'positive_pairs':p,'negative_pairs':n,'ignored_pairs':int((~valid[:,j]).sum()),'true_presence':tp,'false_presence':fp,'localized_hits':hit,'localized_recall':hit/p if p else None,'presence_recall':recall,'false_positive_rate':fpr,'false_alarms_per_10000':10000*fpr if n else None,'presence_balanced_accuracy':(recall+1-fpr)/2 if p and n else None,'presence_auroc':rank_auc(present[valid[:,j],j],scores[valid[:,j],j])})
    avg=lambda name:float(np.mean([r[name]for r in details if r[name]is not None]))
    n=sum(r['negative_pairs']for r in details);fp=sum(r['false_presence']for r in details)
    return {'cutoff':threshold,'photos':len(scores),'nominal_pairs':scores.size,'valid_pairs':int(valid.sum()),'positive_pairs':int((present&valid).sum()),'negative_pairs':n,'ignored_pairs':int((~valid).sum()),'localized_hits':sum(r['localized_hits']for r in details),'false_presence':fp,'macro_localized_recall':avg('localized_recall'),'macro_presence_recall':avg('presence_recall'),'macro_false_positive_rate':avg('false_positive_rate'),'macro_presence_balanced_accuracy':avg('presence_balanced_accuracy'),'macro_presence_auroc':avg('presence_auroc'),'micro_false_positive_rate':fp/n,'micro_false_alarms_per_10000':10000*fp/n,'per_target':details}
def compare_values(actual,expected,path='',stats=None):
    if stats is None:stats={'numeric_values':0,'maximum_absolute_error':0.}
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(path,'keys')
        for key in expected:compare_values(actual[key],expected[key],path+'/'+key,stats)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),(path,'length')
        for j,(a,b)in enumerate(zip(actual,expected)):compare_values(a,b,path+'/'+str(j),stats)
    elif isinstance(expected,(float,int))and not isinstance(expected,bool):
        error=abs(float(actual)-float(expected));assert error<2e-12,(path,actual,expected)
        stats['numeric_values']+=1;stats['maximum_absolute_error']=max(stats['maximum_absolute_error'],error)
    else:assert actual==expected,(path,actual,expected)
    return stats

def audit():
    plan_path=EVAL/'data/plan.json';plan=read(plan_path);acquisition=read(EVAL/'data/acquisition.json')
    assert acquisition['completed']is True and acquisition['data_plan_sha256']==sha(plan_path)
    protocol_path=Path(acquisition['protocol_path']);assert sha(protocol_path)==acquisition['protocol_sha256'];protocol=read(protocol_path)
    assert protocol['model_selection_complete']is True and protocol['test_scores_computed_before_freeze']is False and protocol['data_plan_sha256']==sha(plan_path)
    for group in ('behavior_files','checkpoint_files'):
        for path,h in protocol[group].items():assert sha(path)==h,('frozen_binding',path)
    keep=set(acquisition['eligible_image_ids']);rows=[r for r in plan['test']if r['image_id']in keep];targets=plan['targets'];dev=plan['dev']
    assert len(rows)==len(keep)and len({r['image_id']for r in rows})==len(rows)
    assert not keep&{r['image_id']for r in dev}and not keep&{r['image_id']for r in plan['references']}
    assert len(targets)==12 and len(plan['references'])==12
    present,valid=labels(rows,targets);dp,dv=labels(dev,targets)
    arrays={};thresholds={};summaries={};value_stats={'numeric_values':0,'maximum_absolute_error':0.};ious=0;iou_error=0.;result_bindings={}
    for method in METHODS:
        summaries[method]={};loaded={}
        for split,selected,p,v in [('dev',dev,dp,dv),('test',rows,present,valid)]:
            directory=EVAL/'results'/f'{method}_{split}';result=read(directory/'result.json');path=directory/'arrays.npz'
            assert sha(path)==result['score_file_sha256']and result['data_plan_sha256']==sha(plan_path)
            a=dict(np.load(path,allow_pickle=False));assert a['image_ids'].tolist()==[r['image_id']for r in selected]
            assert a['category_ids'].tolist()==[t['category_id']for t in targets]
            assert a['scores'].shape==p.shape and a['boxes'].shape==(*p.shape,4)and a['overlaps'].shape==p.shape
            assert all(np.isfinite(a[k]).all()for k in ('scores','boxes','overlaps'))
            measured=np.empty(p.shape,np.float64)
            for i,row in enumerate(selected):
                for j,target in enumerate(targets):
                    box=a['boxes'][i,j];assert np.all(box>=0)and box[0]<=row['width']and box[2]<=row['width']and box[1]<=row['height']and box[3]<=row['height']
                    measured[i,j]=iou_one(box,row['boxes_by_target'][str(target['category_id'])]);ious+=1
            error=float(np.max(abs(measured-a['overlaps'])));iou_error=max(iou_error,error);assert error<2e-12
            a['overlaps']=measured;loaded[split]=(a,result);result_bindings[str(path)]=sha(path)
            if split=='test':
                assert result['protocol_sha256']==acquisition['protocol_sha256']
                assert result['configuration']==loaded['dev'][1]['configuration']and result['provenance']==loaded['dev'][1]['provenance']
                prediction_path=directory/'predictions.jsonl';assert sha(prediction_path)==result['predictions_sha256']
                lines=[json.loads(line)for line in prediction_path.read_text().splitlines()];assert len(lines)==p.size
                for k,line in enumerate(lines):
                    i,j=divmod(k,len(targets));row=selected[i];assert line['image_id']==row['image_id']and line['category_id']==targets[j]['category_id']
                    raw=np.array(line['raw_box'],float);clipped=np.minimum(np.maximum(raw,0),[row['width'],row['height'],row['width'],row['height']])
                    assert np.array_equal(clipped,a['boxes'][i,j])and line['score']==a['scores'][i,j]and abs(line['iou']-measured[i,j])<2e-12
        cal_path=EVAL/'results'/f'{method}_dev/calibration.json';cal=read(cal_path);thresholds[method]={}
        for key,budget in POINTS.items():
            value=cutoff(loaded['dev'][0]['scores'][dv&~dp],budget);thresholds[method][key]=value
            assert value==cal[key]['cutoff']==protocol['methods'][method]['cutoffs'][key]
            assert cal[key]['calibration_negative_pairs']==int((dv&~dp).sum())and cal[key]['nominal_dev_negative_fpr']==budget
            assert int((loaded['dev'][0]['scores'][dv&~dp]>=value).sum())<=math.floor(budget*int((dv&~dp).sum()))
            for split,p,v in [('dev',dp,dv),('test',present,valid)]:
                a,result=loaded[split];m=summary(a['scores'],a['overlaps'],p,v,value,targets)
                compare_values(m,result['summary'][key],method+'/'+split+'/'+key,value_stats)
                if split=='dev':compare_values(m,cal[key]['summary'],method+'/cal/'+key,value_stats)
                else:summaries[method][key]=m
        arrays[method]=loaded['test'][0];result_bindings[str(cal_path)]=sha(cal_path)
    paired=read(EVAL/'results/paired_test.json');assert paired['data_plan_sha256']==sha(plan_path)
    for path,h in paired['input_bindings'].items():assert sha(path)==h
    assert paired['photos']==len(rows)and paired['nominal_pairs']==present.size
    assert paired['bootstrap']['replicates']==5000 and paired['bootstrap']['seed']==190031
    # Reuse only the declared random resampling sequence; recompute statistics
    # from independently recomputed IoUs and all-category photo clusters.
    w=np.random.default_rng(190031).multinomial(len(rows),np.full(len(rows),1/len(rows)),size=5000).astype(float)
    positives=present&valid;negatives=~present&valid;posden=w@positives;negden=w@negatives;assert np.all(posden>0)and np.all(negden>0)
    comparison={};ci_count=0
    for key in POINTS:
        boot={};point={}
        for method,a in arrays.items():
            yes=a['scores']>=thresholds[method][key];hit=yes&(a['overlaps']>=.5)&positives;fp=yes&negatives
            boot[method]=np.column_stack([(w@hit/posden).mean(1),(w@fp/negden).mean(1)])
            point[method]=np.array([(hit.sum(0)/positives.sum(0)).mean(),(fp.sum(0)/negatives.sum(0)).mean()])
        comparison[key]={}
        for other in METHODS[1:]:
            delta=boot['v19']-boot[other];estimate=point['v19']-point[other]
            value={label:{'estimate':float(estimate[j]),'ci95':np.percentile(delta[:,j],[2.5,97.5]).tolist()}for j,label in enumerate(['macro_localized_recall','macro_false_positive_rate'])}
            value['observed_win_gate']=bool(estimate[0]>0 and estimate[1]<=0);comparison[key]['v19_minus_'+other]=value;ci_count+=4
    compare_values(comparison,paired['operating_points'],'paired',value_stats)
    return {'status':'passed','scope':'Independent post-freeze CPU recomputation from completed result arrays and annotations; no images, inference, project metric helpers or GPU.','protocol_sha256':sha(protocol_path),'verified_behavior_files':len(protocol['behavior_files']),'verified_checkpoint_files':len(protocol['checkpoint_files']),'photos':len(rows),'targets':len(targets),'nominal_pairs':int(present.size),'positive_pairs':int(positives.sum()),'negative_pairs':int(negatives.sum()),'ignored_pairs':int((~valid).sum()),'recomputed_box_ious':ious,'box_iou_max_absolute_error':iou_error,'metric_and_ci_value_checks':value_stats,'bootstrap_replicates':5000,'bootstrap_seed':190031,'ci_endpoints_checked':ci_count,'methods':summaries,'paired':comparison,'result_input_bindings':result_bindings,'uncertainty_limit':'A zero-hit percentile interval may be degenerate; it does not prove exact zero population recall or population equivalence.','gpu_used':False,'natural_images_decoded':0}

def self_test():
    assert iou_one([0,0,10,10],[[5,0,15,10]])==1/3
    assert iou_one([0,0,10,10],[])==0
    assert rank_auc([1,0,1,0],[2,1,1,0])==.875
    assert rank_auc([1,0],[1,1])==.5
    assert cutoff([1,1,2,3],.25)>2 and cutoff([1,1,2,3],0)>3
    print('Independent numerical self-checks passed; no result/data files read.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--authorized',action='store_true');p.add_argument('--self-test',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.self_test:self_test()
    else:
        if not a.authorized or a.output is None:p.error('After root GO, require --authorized and --output; do not inspect fresh records beforehand')
        result=audit();a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in result.items()if k not in ['methods','result_input_bindings']},indent=2))
