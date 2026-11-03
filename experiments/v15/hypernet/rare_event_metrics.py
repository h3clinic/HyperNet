"""Count false alarms at a fixed calibration-selected operating point.

Rates per10,000 are normalized rates, never a claim that10,000 independent
negative examples were observed. Event and time-group counts stay separate.
"""
import math
import numpy as np
from scipy.stats import beta,rankdata


def binomial_interval(hits,total,confidence=.95):
    if not 0<=hits<=total or total<1:raise ValueError('Expected nonempty binomial counts')
    alpha=1-confidence
    return [0. if hits==0 else float(beta.ppf(alpha/2,hits,total-hits+1)),
            1. if hits==total else float(beta.ppf(1-alpha/2,hits+1,total-hits))]


def upper_bound(hits,total,confidence=.95):
    if not 0<=hits<=total or total<1:raise ValueError('Expected nonempty binomial counts')
    return 1. if hits==total else float(beta.ppf(confidence,hits+1,total-hits))


def calibration_cutoff(negative_scores,target_false_alarms_per_10000=10.):
    values=np.asarray(negative_scores,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Finite calibration negative scores required')
    if not 0<=target_false_alarms_per_10000<10000:raise ValueError('Invalid false-alarm budget')
    allowed=math.floor(len(values)*target_false_alarms_per_10000/10000)
    threshold=float(np.nextafter(np.sort(values)[::-1][allowed],math.inf))
    actual=int(np.count_nonzero(values>=threshold))
    return {'threshold':threshold,'calibration_negatives':len(values),'allowed_false_alarms':allowed,
            'observed_false_alarms':actual,'target_false_alarms_per_10000':target_false_alarms_per_10000,
            'claim':'Empirical calibration constraint, not a guaranteed TEST false-alarm rate.'}


def summary(scores,present,threshold,group_ids=None):
    s=np.asarray(scores,dtype=np.float64);y=np.asarray(present,dtype=bool)
    if s.ndim!=1 or y.shape!=s.shape or not np.isfinite(s).all() or not np.isfinite(threshold):raise ValueError('Invalid score/label arrays')
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:raise ValueError('Both positive and negative events are needed')
    hit=s>=threshold;tp=int((hit&y).sum());fp=int((hit&~y).sum())
    ranks=rankdata(s,method='average');auc=float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))
    # Tie-block average precision: a tied block enters as one decision level.
    order=np.argsort(-s,kind='stable');found=seen=cursor=0;ap=0.
    while cursor<len(order):
        end=cursor+1
        while end<len(order) and s[order[end]]==s[order[cursor]]:end+=1
        count=int(y[order[cursor:end]].sum());found+=count;seen=end
        ap+=found/seen*count/pos;cursor=end
    out={'positive_events':pos,'negative_events':neg,'true_positive':tp,'false_positive':fp,
         'recall':tp/pos,'precision':tp/(tp+fp) if tp+fp else None,'threshold':float(threshold),
         'false_positive_rate':fp/neg,'false_alarms_per_10000_negatives':10000*fp/neg,
         'auroc':auc,'average_precision':ap,
         'independent_event_binomial_reference':{'recall_ci95':binomial_interval(tp,pos),
             'false_alarms_per_10000_ci95':[10000*v for v in binomial_interval(fp,neg)],
             'false_alarms_per_10000_upper95':10000*upper_bound(fp,neg),
             'assumption':'Independent Bernoulli events; not a dependence-adjusted guarantee.'},
         'rate_scope':'Normalized per10,000 negatives; denominator is the actual negative_events count above.'}
    if group_ids is not None:
        groups=np.asarray(group_ids,dtype=str)
        if groups.shape!=s.shape:raise ValueError('One time-group ID required per event')
        negative_groups=[];mixed=0
        for key in np.unique(groups):
            take=groups==key
            if y[take].any():
                mixed+=int((~y[take]).any());continue
            negative_groups.append(bool(hit[take].any()))
        if not negative_groups:raise ValueError('No wholly negative time groups')
        gfp=sum(negative_groups);gn=len(negative_groups)
        out['negative_time_groups']={'groups':gn,'groups_with_any_false_alarm':gfp,
            'mixed_label_groups_excluded_from_group_rate':mixed,
            'any_alarm_groups_per_10000_negative_groups':10000*gfp/gn,
            'any_alarm_groups_per_10000_upper95':10000*upper_bound(gfp,gn),
            'assumption':'Independent time groups. A group alarms if any member alarms; this is a group rate, not an image rate.'}
    return out


def projected_search(recall,false_positive_rate,images=1_000_000,prevalence=.0001):
    if not 0<=recall<=1 or not 0<=false_positive_rate<=1 or not 0<prevalence<1 or images<1:raise ValueError('Invalid search scenario')
    targets=images*prevalence;tp=targets*recall;fp=images*(1-prevalence)*false_positive_rate
    return {'images':images,'assumed_prevalence':prevalence,'targets_present':targets,
            'expected_recovered_targets':tp,'expected_false_alarms':fp,
            'expected_precision':tp/(tp+fp) if tp+fp else None,
            'scope':'Illustration assuming measured rates transfer unchanged; not a measured archive-search result.'}
