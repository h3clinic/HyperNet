"""CPU-only implementation checks; no accuracy-guided choices or heldout test."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import importlib.util
import numpy as np,torch
from model import *
import generator as gen
import train as tr
def main():
 assert not (R/'protocol.json').exists(),'Checks must precede freeze';torch.set_num_threads(2)
 oldpath=R.parents[1]/'hypernet_v44/training_model.py';s=importlib.util.spec_from_file_location('v45_parity_readouts',oldpath);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
 torch.manual_seed(459999);d=torch.randn(3,48,32,32);base=descriptor_base(d);spatial=descriptor_spatial(d);expected_spatial=F.normalize(old.descriptor_relation(d)[:,:3040],dim=1)
 parity=dict(base_max_error=float((base-old.descriptor_base(d)).abs().max()),spatial_max_error=float((spatial-expected_spatial).abs().max()),base_dimension=base.shape[1],spatial_dimension=spatial.shape[1]);assert parity['base_max_error']==0;assert parity['spatial_max_error']<2e-7
 geometry=[];raster=[]
 for seed in tr.SEEDS:
  ids=tr.order_for(seed)[:24]
  for i,identity in enumerate(ids):
   q=gen.local_points(int(identity),'train');pred=q['reference']@q['affine'].T+q['offset'];err=float(np.max(np.abs(pred-q['positive'])));assert err<2e-5 and q['minimum_positive_support']>.05
   geometry.append(dict(identity=int(identity),available=q['available'],replacement=q['replacement'],affine_error=err))
   if i<2:
    a=gen.annotated_episode(int(identity),'train');b=gen._c.episode(int(identity),'train');assert all(np.array_equal(a[k+suffix],b[k+suffix]) for k in gen.VIEWS for suffix in ['', '_low']);raster.append(int(identity))
 # Compare actual trainable objectives only on TRAIN images; no synthetic holdout.
 eps=[gen.annotated_episode(int(tr.order_for(tr.SEEDS[0])[0]),'train')];full,low=tr.pack(eps,'cpu');start=R/f'seed_{tr.SEEDS[0]}/random.pt';nets={arm:load_encoder(start).train() for arm in ['baseline','improved']};grads={};stats={}
 for arm,net in nets.items():
  loss,record=tr.objective(net,full,low,eps,arm);loss.backward();assert torch.isfinite(loss);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());assert sum(p.numel() for p in net.parameters())==42480;grads[arm]=torch.cat([p.grad.flatten() for p in net.parameters()]);stats[arm]=record|dict(gradient_norm=float(grads[arm].norm()),all_parameters_trainable=all(p.requires_grad for p in net.parameters()))
 assert torch.equal(torch.cat([p.flatten() for p in nets['baseline'].parameters()]),torch.cat([p.flatten() for p in nets['improved'].parameters()]));difference=float((grads['improved']-grads['baseline']).norm());assert difference>0
 states=[torch.cat([v.flatten() for v in torch.load(R/f'seed_{seed}/random.pt',weights_only=True).values()]) for seed in tr.SEEDS];initdist=[float((states[i]-states[j]).norm()) for i in range(3) for j in range(i+1,3)];assert min(initdist)>0
 # A perfectly collapsed direction must receive a nonzero variance penalty.
 constant=torch.zeros(1,48,32,32);constant[:,0]=1;loc,con,var=tr.local_loss(constant.repeat(6,1,1,1),eps);assert float(var)>.7
 result=dict(passed=True,cpu_only=True,heldout_test_consumed=False,source_hashes={str(p):sha(p) for p in tr.source_files()},readout_parity=parity,geometry=dict(identities=len(geometry),maximum_affine_error=max(z['affine_error'] for z in geometry),minimum_available_points=min(z['available'] for z in geometry),replacement_identities=sum(z['replacement'] for z in geometry)),unchanged_image_identities=raster,paired_initial_weights_exact=True,independent_initialization_pairwise_l2=initdist,objective_checks=stats,added_objective_gradient_difference_l2=difference,collapsed_variance_penalty=float(var),limitations='Geometry is exact for analytic centers but rendered stroke edges are discretized. Soft variance penalty detects collapse but does not prove optimization avoids it. CPU checks use train identities only.')
 save(R/'checks.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
