"""Fixed, development-only per-duration matching and outer calibration."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import hashlib,importlib.util,json,time
from pathlib import Path
from fractions import Fraction
import numpy as np
R=Path(__file__).resolve().parent;V46=R.parent/'hypernet_v46'
def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=load_module('v47_frozen_v46',V46/'common.py')
read=C.read;save=C.save;sha=C.sha;bindings=C.bindings
TARGETS=C.TARGETS;DURATIONS=C.DURATIONS;ALPHAS=[.001,.0005]
MATCHERS=['single1s','joint4','late_mean4','late_max4']
POOLS=['original','expanded'];GROUPINGS=['pooled','detector']
FIELDS=C.FIELDS

def scale_parameters(values):
 """Use target-negative TUNE only. Every fallback is predeclared."""
 x=np.asarray(values,np.float64)
 assert x.ndim==2 and len(x)>1 and np.isfinite(x).all()
 loc=np.median(x,axis=0);mad=1.4826*np.median(np.abs(x-loc),axis=0)
 std=np.std(x,axis=0,ddof=1)
 scale=np.where(mad>1e-12,mad,np.where(std>1e-12,std,1.))
 fallback=['mad' if a>1e-12 else ('std' if b>1e-12 else 'unit') for a,b in zip(mad,std)]
 return dict(location=loc,scale=scale,fallback=fallback)

def fuse(raw,joint,params):
 x=np.asarray(raw,np.float64);assert x.ndim==2 and x.shape[1]==4 and np.isfinite(x).all()
 z=(x-params['location'])/params['scale']
 return dict(single1s=x[:,1].copy(),joint4=np.asarray(joint,np.float64).copy(),late_mean4=z.mean(1),late_max4=z.max(1))

def budget(n,alpha):
 a=Fraction(str(alpha));return (int(n)+1)*a.numerator//a.denominator

def threshold(negative_scores,alpha):
 x=np.sort(np.asarray(negative_scores,np.float64));assert len(x)>0 and np.isfinite(x).all()
 k=budget(len(x),alpha)
 return float(x[-k]) if k>=1 else None

def tails(negative_scores,scores):
 cal=np.sort(np.asarray(negative_scores,np.float64));x=np.asarray(scores,np.float64)
 assert len(cal) and np.isfinite(cal).all() and np.isfinite(x).all()
 numerator=1+len(cal)-np.searchsorted(cal,x,side='left')
 return numerator/(len(cal)+1),numerator

def calibration(scores,rows,cal_indices,grouping):
 """Fit final-score cutoffs only; no per-view p-value combination."""
 assert grouping in GROUPINGS and set(r['ifo'] for r in rows)<=set(['H1','L1'])
 assert len(cal_indices)==len(set(cal_indices)) and all(0<=i<len(rows) for i in cal_indices)
 n=len(rows);p=np.empty(n,np.float64);pred={a:np.zeros(n,bool) for a in ALPHAS};details={}
 for group in (['all'] if grouping=='pooled' else ['H1','L1']):
  eligible=np.arange(n) if group=='all' else np.array([i for i,r in enumerate(rows) if r['ifo']==group],int)
  ci=np.array([i for i in cal_indices if group=='all' or rows[i]['ifo']==group],int)
  assert len(ci)>0 and len(eligible)>0
  pp,num=tails(np.asarray(scores)[ci],np.asarray(scores)[eligible]);p[eligible]=pp
  cuts={}
  for a in ALPHAS:
   cut=threshold(np.asarray(scores)[ci],a);pr=num<=budget(len(ci),a)
   assert np.array_equal(pr,np.asarray(scores)[eligible]>(cut if cut is not None else np.inf))
   pred[a][eligible]=pr;cuts[str(a)]=dict(threshold=cut,rank_budget=budget(len(ci),a),supported=cut is not None)
  details[group]=dict(calibration_indices=ci.tolist(),n=len(ci),minimum_p=1/(len(ci)+1),cutoffs=cuts)
 assert np.isfinite(p).all() and np.all((p>0)&(p<=1))
 return p,pred,details

def metrics(y,scores,pred,indices):
 from sklearn.metrics import roc_auc_score
 ix=np.asarray(indices,int);yy=np.asarray(y,bool)[ix];v=np.asarray(scores)[ix];pr=np.asarray(pred,bool)[ix]
 pos=int(yy.sum());neg=int((~yy).sum());tp=int((yy&pr).sum());fp=int((~yy&pr).sum())
 return dict(positive=pos,negative=neg,**C.METRIC.metrics(tp,fp,pos,neg),fa_per10000=fp/neg*10000 if neg else None,
  auroc=float(roc_auc_score(yy,v)) if pos and neg else None,
  pauc_std_01=float(roc_auc_score(yy,v,max_fpr=.01)) if pos and neg else None,
  pauc_std_001=float(roc_auc_score(yy,v,max_fpr=.001)) if pos and neg else None)

def summarize(rows):return {k:float(np.mean([r[k] for r in rows])) for k in FIELDS}|dict(tp=sum(r['tp'] for r in rows),fp=sum(r['fp'] for r in rows))
