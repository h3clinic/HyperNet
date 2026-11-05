"""Confusion-matrix metrics for binary presence, not localized-box recall."""
import math
def metrics(tp,fp,positives,negatives):
 if tp is None or fp is None or not positives or not negatives:return {'balanced_accuracy':None,'mcc':None,'precision':None,'recall':None,'specificity':None,'f1':None}
 tn=negatives-fp;fn=positives-tp
 assert min(tp,fp,tn,fn)>=0
 recall=tp/positives;specificity=tn/negatives;den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
 return dict(tp=tp,fp=fp,tn=tn,fn=fn,balanced_accuracy=.5*(recall+specificity),mcc=(tp*tn-fp*fn)/math.sqrt(den) if den else 0.,precision=tp/(tp+fp) if tp+fp else None,recall=recall,specificity=specificity,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,mcc_degenerate=not bool(den))
if __name__=='__main__':
 assert metrics(10,0,10,100)['balanced_accuracy']==1
 assert metrics(10,0,10,100)['mcc']==1
 assert metrics(0,0,10,100)['balanced_accuracy']==.5
 assert metrics(0,0,10,100)['mcc']==0
 assert metrics(10,100,10,100)['balanced_accuracy']==.5
 assert metrics(0,100,10,100)['mcc']==-1
 assert metrics(8,2,10,100)['balanced_accuracy']>metrics(8,20,10,100)['balanced_accuracy']
 assert metrics(None,None,10,100)['mcc'] is None
 print('PASS: perfect, constant, inverted, false-alarm penalty, unsupported threshold')
