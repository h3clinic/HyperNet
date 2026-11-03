"""Top-one localization at a DEV-fitted global presence threshold."""
import numpy as np

COMMON_BOX_RULE='Select highest-scoring proposal first, clip xyxy to the original image bounds for every method, then compute half-open IoU; raw box retained.'

def clip_box(box,width,height):
    a=np.asarray(box,dtype=float).copy()
    if a.shape!=(4,) or not np.isfinite(a).all() or width<1 or height<1: raise ValueError('Invalid box/image size')
    a[[0,2]]=np.clip(a[[0,2]],0,width);a[[1,3]]=np.clip(a[[1,3]],0,height)
    return a.tolist()

def global_cutoff(negative_scores, budget):
    a=np.sort(np.asarray(negative_scores,dtype=np.float64))
    if a.size==0 or not np.isfinite(a).all() or not 0<=budget<1: raise ValueError('Invalid calibration')
    allowed=int(np.floor(budget*len(a)))
    # >= cutoff; nextafter makes ties conservative, never exceeding the DEV budget.
    return float(np.nextafter(a[len(a)-allowed-1],np.inf))

def iou(box,truth):
    box=np.asarray(box,dtype=float); truth=np.asarray(truth,dtype=float).reshape(-1,4)
    if not len(truth): return 0.
    inter=np.maximum(0,np.minimum(box[2:],truth[:,2:])-np.maximum(box[:2],truth[:,:2])).prod(1)
    union=np.maximum(0,box[2:]-box[:2]).prod()+np.maximum(0,truth[:,2:]-truth[:,:2]).prod(1)-inter
    return float(np.max(inter/np.maximum(union,1e-15)))

def auc(y,s):
    y=np.asarray(y,dtype=bool); s=np.asarray(s,dtype=float)
    p=s[y]; n=s[~y]
    if not len(p) or not len(n): return None
    ordered=np.sort(n)
    return float(np.mean((np.searchsorted(ordered,p,'left')+np.searchsorted(ordered,p,'right'))/(2*len(n))))

def truth_arrays(rows,targets):
    keys=[str(t['category_id']) for t in targets]
    present=np.array([[bool(r['boxes_by_target'][k]) for k in keys] for r in rows])
    valid=np.array([[r['valid_by_target'][k] for k in keys] for r in rows])
    return present,valid

def summarize(scores, overlaps, present, valid, cutoff, targets):
    scores=np.asarray(scores); overlaps=np.asarray(overlaps); accepted=scores>=cutoff
    if scores.shape!=present.shape or overlaps.shape!=present.shape or not np.isfinite(scores).all(): raise ValueError('Invalid score matrix')
    per=[]
    for j,t in enumerate(targets):
        pos=valid[:,j]&present[:,j]; neg=valid[:,j]&~present[:,j]
        tp=int((accepted[:,j]&pos).sum()); fp=int((accepted[:,j]&neg).sum())
        loc=int((accepted[:,j]&pos&(overlaps[:,j]>=.5)).sum()); p=int(pos.sum()); n=int(neg.sum())
        rec=tp/p if p else None; fpr=fp/n if n else None
        per.append({'category_id':t['category_id'],'name':t.get('category_name',t.get('name')),
                    'positive_pairs':p,'negative_pairs':n,'ignored_pairs':int((~valid[:,j]).sum()),
                    'true_presence':tp,'false_presence':fp,'localized_hits':loc,
                    'localized_recall':loc/p if p else None,'presence_recall':rec,'false_positive_rate':fpr,
                    'false_alarms_per_10000':10000*fpr if fpr is not None else None,
                    'presence_balanced_accuracy':.5*(rec+1-fpr) if rec is not None and fpr is not None else None,
                    'presence_auroc':auc(present[valid[:,j],j],scores[valid[:,j],j])})
    avg=lambda k:float(np.mean([r[k] for r in per if r[k] is not None]))
    p=sum(r['positive_pairs'] for r in per); n=sum(r['negative_pairs'] for r in per)
    return {'cutoff':cutoff,'photos':len(scores),'nominal_pairs':int(scores.size),'valid_pairs':int(valid.sum()),
            'positive_pairs':p,'negative_pairs':n,'ignored_pairs':int((~valid).sum()),
            'localized_hits':sum(r['localized_hits'] for r in per),'false_presence':sum(r['false_presence'] for r in per),
            'macro_localized_recall':avg('localized_recall'),'macro_presence_recall':avg('presence_recall'),
            'macro_false_positive_rate':avg('false_positive_rate'),'macro_presence_balanced_accuracy':avg('presence_balanced_accuracy'),
            'macro_presence_auroc':avg('presence_auroc'),
            'micro_false_positive_rate':sum(r['false_presence'] for r in per)/n,
            'micro_false_alarms_per_10000':10000*sum(r['false_presence'] for r in per)/n,'per_target':per}

def calibration(scores, overlaps, rows, targets):
    present,valid=truth_arrays(rows,targets); negative=scores[valid&~present]
    result={}
    for key,budget in [('primary_5pct',.05),('secondary_10_per_10000',.001)]:
        cutoff=global_cutoff(negative,budget)
        result[key]={'nominal_dev_negative_fpr':budget,'calibration_negative_pairs':int(negative.size),
                     'threshold_source':'All valid DEV absent target/photo pairs, one global threshold',
                     'summary':summarize(scores,overlaps,present,valid,cutoff,targets),'cutoff':cutoff}
    return result
