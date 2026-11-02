"""Scene-level one-reference category detection metrics; not official COCO AP."""
from __future__ import annotations
import math
import numpy as np


def max_box_iou(predicted, targets):
    if predicted is None or not targets:
        return 0.
    a=np.asarray(predicted,dtype=np.float64)
    b=np.asarray(targets,dtype=np.float64).reshape(-1,4)
    if a.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('IoU requires finite xyxy boxes')
    if np.any(a[2:]<=a[:2]) or np.any(b[:,2:]<=b[:,:2]):
        raise ValueError('IoU requires nonempty boxes')
    intersection=np.maximum(0.,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(1)
    union=np.prod(a[2:]-a[:2])+np.prod(b[:,2:]-b[:,:2],axis=1)-intersection
    return float(np.max(intersection/np.maximum(union,1e-20)))


def cutoff_from_real_dev(rows,max_fpr=.05):
    if not rows or any(row['split']!='dev' for row in rows):
        raise ValueError('Calibration accepts real DEV rows only')
    if not 0<=max_fpr<1:
        raise ValueError('max_fpr must lie in[0,1)')
    values=[float(row['score']) for row in rows if not row['present']]
    if not values or not np.isfinite(values).all():
        raise ValueError('Calibration requires finite negative-scene scores')
    values.sort(reverse=True)
    return float(np.nextafter(values[math.floor(max_fpr*len(values))],math.inf))


def summarize(rows,cutoff):
    if not rows or not np.isfinite(cutoff):
        raise ValueError('Metrics require rows and a finite cutoff')
    if len({row['id'] for row in rows})!=len(rows):
        raise ValueError('Duplicate query IDs')
    positives=[r for r in rows if r['present']]
    negatives=[r for r in rows if not r['present']]
    if not positives or not negatives:
        raise ValueError('Metrics require both positive and negative scenes')
    scores=np.array([r['score'] for r in rows],dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite')
    tp=int(sum(r['score']>=cutoff for r in positives))
    fp=int(sum(r['score']>=cutoff for r in negatives))
    localized=int(sum(r['score']>=cutoff and r['iou']>=.5 for r in positives))
    p=np.array([r['score'] for r in positives]);n=np.array([r['score'] for r in negatives])
    auc=float(np.mean((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])))
    ordered=sorted(rows,key=lambda r:r['score'],reverse=True)
    found=seen=cursor=0;ap=0.
    while cursor<len(ordered):
        end=cursor+1
        while end<len(ordered) and ordered[end]['score']==ordered[cursor]['score']:end+=1
        hits=sum(r['present'] and r['iou']>=.5 for r in ordered[cursor:end])
        found+=hits;seen+=end-cursor
        ap+=found/seen*hits/len(positives)
        cursor=end
    return dict(positive_scenes=len(positives),negative_scenes=len(negatives),threshold=float(cutoff),
        accepted_positive=tp,false_positive=fp,localized_true_positive=localized,
        recall=tp/len(positives),false_positive_rate=fp/len(negatives),
        localized_recall_at_iou50=localized/len(positives),
        top1_iou_recall_at_iou50=sum(r['iou']>=.5 for r in positives)/len(positives),
        proposal_recall_at_iou50=sum(r['proposal_best_iou']>=.5 for r in positives)/len(positives),
        balanced_accuracy=.5*(tp/len(positives)+1-fp/len(negatives)),
        auroc=auc,single_box_ap50=float(ap))


def category_bootstrap(rows_by_method,cutoffs,samples=10000,seed=1313):
    if samples<1:raise ValueError('Bootstrap samples must be positive')
    test={alias:[r for r in rows if r['split']=='test'] for alias,rows in rows_by_method.items()}
    aliases=list(test)
    if not aliases:raise ValueError('No methods supplied')
    baseline={r['id']:(r['category_id'],bool(r['present'])) for r in test[aliases[0]]}
    if len(baseline)!=len(test[aliases[0]]) or not baseline:
        raise ValueError('Empty or duplicated TEST roster')
    for alias,rows in test.items():
        roster={r['id']:(r['category_id'],bool(r['present'])) for r in rows}
        if roster!=baseline or len(roster)!=len(rows):raise ValueError('Methods must share the exact TEST roster')
    categories=sorted({category for category,present in baseline.values()})
    generator=np.random.default_rng(seed)
    draws=generator.integers(0,len(categories),size=(samples,len(categories)))
    distributions={};methods={}
    for alias,rows in test.items():
        numerators=[];positive_counts=[];negative_counts=[];false_counts=[]
        for category in categories:
            positive=[r for r in rows if r['category_id']==category and r['present']]
            negative=[r for r in rows if r['category_id']==category and not r['present']]
            if not positive or not negative:raise ValueError('Every category needs positive and negative TEST scenes')
            numerators.append(sum(r['score']>=cutoffs[alias] and r['iou']>=.5 for r in positive))
            false_counts.append(sum(r['score']>=cutoffs[alias] for r in negative))
            positive_counts.append(len(positive));negative_counts.append(len(negative))
        numerators=np.array(numerators);positive_counts=np.array(positive_counts)
        false_counts=np.array(false_counts);negative_counts=np.array(negative_counts)
        local=numerators[draws].sum(1)/positive_counts[draws].sum(1)
        fpr=false_counts[draws].sum(1)/negative_counts[draws].sum(1)
        distributions[alias]=local
        methods[alias]=dict(localized_recall_at_iou50=dict(estimate=float(numerators.sum()/positive_counts.sum()),
            ci95=np.quantile(local,[.025,.975]).tolist()),false_positive_rate=dict(
            estimate=float(false_counts.sum()/negative_counts.sum()),ci95=np.quantile(fpr,[.025,.975]).tolist()))
    differences={}
    for treatment,control in [('native_patch','old_cnn'),('native_patch_grid','native_patch'),('old_cnn_grid','old_cnn')]:
        if treatment in methods and control in methods:
            differences[treatment+'_minus_'+control]=dict(
                estimate=methods[treatment]['localized_recall_at_iou50']['estimate']-methods[control]['localized_recall_at_iou50']['estimate'],
                ci95=np.quantile(distributions[treatment]-distributions[control],[.025,.975]).tolist())
    return dict(unit='category',categories=len(categories),category_ids=categories,samples=samples,seed=seed,
        methods=methods,localized_recall_differences=differences,
        scope='Percentile cluster intervals conditional on the frozen operating thresholds and selected reference images; thresholds are not refitted in resamples.')
