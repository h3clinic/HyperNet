"""Outcome-free numerical and adversarial checks for v47 implementation."""
from core import *
from fractions import Fraction
from threadpoolctl import threadpool_limits

def main():
 threadpool_limits(2);rng=np.random.default_rng(47019);counts={};checks=[]
 def record(name):checks.append(name)
 # All ranks and ties checked against a slow, independent direct counting rule.
 total=0
 for n in [1,2,9,19,99,998,999,1000,1998,1999,2000,4049,5000]:
  for tie in [False,True]:
   cal=rng.normal(size=n);cal=np.round(cal,1) if tie else cal
   query=np.r_[rng.normal(size=25),cal[:min(n,15)],[-100,100]]
   p,num=tails(cal,query);direct=np.array([1+sum(c>=q for c in cal) for q in query])
   assert np.array_equal(num,direct);np.testing.assert_array_equal(p,direct/(n+1))
   for alpha in [.0005,.001,.01,.05,.2,.5]:
    cut=threshold(cal,alpha);fr=Fraction(str(alpha))
    truth=direct*fr.denominator<=(n+1)*fr.numerator
    assert np.array_equal(query>(np.inf if cut is None else cut),truth);total+=1
 counts['tail_boundary_cases']=total;record('Exact direct rank, ties, finite-sample boundary and unsupported alpha')
 # Leave-one-out exchangeable toy enumeration, with and without ties.
 for n in [4,9,19,49]:
  for values in [rng.normal(size=n+1),rng.integers(0,4,size=n+1).astype(float)]:
   for alpha in [.05,.1,.2,.5]:
    alarms=[]
    for i in range(n+1):
     cut=threshold(np.delete(values,i),alpha);alarms.append(values[i]>(np.inf if cut is None else cut))
    assert np.mean(alarms)<=alpha+1e-12
 record('Exhaustive exchangeable toy rank control; conservative ties')
 # Median/MAD and the exact fixed zero-MAD fallback.
 x=np.array([[1,0,3,1],[1,0,5,2],[1,0,7,3],[1,10,9,4],[1,0,11,5]],float)
 params=scale_parameters(x);assert params['fallback']==['unit','std','mad','mad']
 assert params['scale'][0]==1 and params['location'][0]==1
 np.testing.assert_allclose(params['scale'][1],np.std(x[:,1],ddof=1))
 z=(x-params['location'])/params['scale'];f=fuse(x,np.arange(5),params)
 assert np.array_equal(f['single1s'],x[:,1]) and np.array_equal(f['joint4'],np.arange(5))
 np.testing.assert_allclose(f['late_mean4'],z.sum(1)/4);np.testing.assert_allclose(f['late_max4'],np.array([max(v) for v in z]))
 assert np.all(f['late_max4']>=z[:,1]);f2=fuse(np.vstack([x,100*np.ones(4)]),np.arange(6),params)
 for k in f:assert np.array_equal(f[k],f2[k][:-1])
 record('Fixed TUNE scaling, MAD/std/unit fallbacks, view axis and query-batch invariance')
 # Detector memberships and an unsupported secondary threshold must be explicit.
 rows=[dict(ifo='H1') for _ in range(1700)]+[dict(ifo='L1') for _ in range(2300)]
 scores=np.r_[rng.normal(0,1,1700),rng.normal(100,1,2300)];cal=list(range(1690))+list(range(1700,3990))
 p,pred,detail=calibration(scores,rows,cal,'detector')
 assert not detail['H1']['cutoffs']['0.0005']['supported'] and not pred[.0005][:1700].any()
 for group,ix in [('H1',np.arange(1700)),('L1',np.arange(1700,4000))]:
  ci=detail[group]['calibration_indices'];pp=np.array([(1+sum(scores[j]>=scores[i] for j in ci))/(len(ci)+1) for i in ix]);np.testing.assert_array_equal(pp,p[ix])
 changed=scores.copy();changed[1700:]+=12345;p2,pr2,_=calibration(changed,rows,cal,'detector')
 assert np.array_equal(p[:1700],p2[:1700])
 for a in ALPHAS:assert np.array_equal(pred[a][:1700],pr2[a][:1700])
 record('Detector calibration isolation, conservative ties, unsupported H1 cutoff rejects all')
 # Direct dense solution checks the frozen shrinkage/Woodbury implementation.
 residuals=[]
 for n,d in [(17,9),(37,61),(27,201)]:
  bg=rng.normal(size=(n,d));ref=rng.normal(size=d);s=C.fit(bg,ref,d);xc=bg-bg.mean(0)
  direct=np.linalg.solve(float(s['ridge'])*np.eye(d)+float(s['beta'])*(xc.T@xc),ref-bg.mean(0))
  np.testing.assert_allclose(s['weight'],direct,atol=1e-8,rtol=1e-8);residuals.append(float(np.max(np.abs(s['weight']-direct))))
 record('Frozen per-duration ELDA matches independent dense solve')
 y=np.array([1,1,0,0,0],bool);s=np.array([5,3,4,2,1.]);pr=s>2.5;m=metrics(y,s,pr,np.arange(5))
 assert m['tp']==2 and m['fp']==1 and m['fn']==0 and m['tn']==2
 assert abs(m['balanced_accuracy']-(1+2/3)/2)<1e-12
 assert abs(m['mcc']-4/np.sqrt(3*2*3*2))<1e-12
 record('Confusion counts, balanced accuracy and MCC against hand calculation')
 files=[Path(__file__),R/'core.py',R/'evaluate.py']
 save(R/'checks.json',dict(passed=True,epoch=time.time(),core_sha256=sha(R/'core.py'),files={str(p):sha(p) for p in files},checks=checks,counts=counts,dense_solution_max_errors=residuals,no_real_scores=True,no_neural_inference=True))
 print('V47 NUMERICAL CHECKS PASSED',len(checks),counts,flush=True)

if __name__=='__main__':main()
