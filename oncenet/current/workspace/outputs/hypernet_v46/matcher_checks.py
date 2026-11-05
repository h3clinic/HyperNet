"""Outcome-free ELDA and repeated-input checks before protocol freezing."""
from common import *
from threadpoolctl import threadpool_limits

def main():
 threadpool_limits(2);rng=np.random.default_rng(460017);checks=[]
 for n,d in [(41,17),(80,67),(100,4032)]:
  for contracted in [False,True]:
   z=rng.normal(size=(n+39,d));z=unit(z if not contracted else np.ones_like(z)+.001*z)
   bg=z[:n];ref=z[n];q=z[n+1:];state=fit(bg,ref,d);s=q@state['weight']+state['offset']
   previous=MATCH.predict(q,MATCH.fit(bg,ref))['exemplar_lda']
   np.testing.assert_allclose(s,previous,atol=1e-7,rtol=1e-8)
   repeated=np.concatenate([z]*4,1)/2;rep=fit(repeated[:n],repeated[n],d);sr=repeated[n+1:]@rep['weight']+rep['offset']
   np.testing.assert_allclose(sr,s,atol=1e-7,rtol=1e-8)
   x=bg-bg.mean(0);res=float(np.linalg.norm(float(state['ridge'])*state['weight']+float(state['beta'])*(x.T@(x@state['weight']))-(ref-bg.mean(0)))/max(1e-12,np.linalg.norm(ref-bg.mean(0))))
   assert res<1e-6
   if d<100:
    inverse=np.linalg.inv(.5*np.cov(bg,rowvar=False)+float(state['ridge'])*np.eye(d));direct=(q-bg.mean(0))@inverse@(ref-bg.mean(0));np.testing.assert_allclose(s,direct,atol=1e-7,rtol=1e-8)
   checks.append(dict(n=n,d=d,contracted=contracted,max_historical_error=float(np.max(np.abs(s-previous))),max_repeat_error=float(np.max(np.abs(sr-s))),normal_equation_relative_residual=res))
 save(R/'matcher_checks.json',dict(passed=True,common_sha256=sha(R/'common.py'),check_code_sha256=sha(__file__),historical_matcher_sha256=sha(P/'hypernet_v43/matcher.py'),checks=checks,real_images_used=False))
 print('MATCHER CHECKS PASSED',flush=True)
if __name__=='__main__':main()
