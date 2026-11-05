"""CPU TRAIN-only checks for the separately declared formula."""
import train as dv
import numpy as np,torch,json
from torch.nn import functional as F
def main():
 assert not(dv.D/'protocol.json').exists();torch.set_num_threads(2);dv.tr.assert_protocol();seed=dv.SEEDS[0];eps=[dv.annotated_episode(int(i),'train') for i in dv.tr.order_for(seed)[:4]];full,low=dv.tr.pack(eps,'cpu');path=dv.M/f'seed_{seed}/random.pt';net=dv.load_encoder(path).train();original=dv.load_encoder(path).train();loss,stats=dv.objective(net,full,low,eps);old,oldstats=dv.tr.objective(original,full,low,eps,'improved')
 for k in ['rank','attraction','local_cosine','positive','hard_negative']:assert abs(stats[k]-oldstats[k])<1e-8,k
 assert abs(float(loss.detach()-old.detach())-stats['global_infonce']-stats['global_variance_floor'])<1e-6
 loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());assert sum(p.numel() for p in net.parameters())==42480
 torch.manual_seed(45099);z=F.normalize(torch.randn(24,4032),dim=1);ce,var,logits=dv.additions(z,4);assert logits.shape==(4,8)
 expected=[]
 for i in range(4):expected.append(torch.stack([z[i]@z[4+i],*[z[i]@z[k*4+i] for k in range(2,6)],*[z[i]@z[4+j] for j in range(4) if j!=i]])/.1)
 assert torch.equal(logits,torch.stack(expected)) or torch.allclose(logits,torch.stack(expected),rtol=1e-6,atol=1e-6)
 c=F.normalize(z[:4]+z[4:8],dim=1)*np.sqrt(4032);manual=np.maximum(.5-np.sqrt(c.numpy().var(0)+1e-4),0).mean();assert abs(float(var)-float(manual))<1e-7
 # Explicitly retain the limitation: the exact constant normalized representation is stationary.
 x=torch.ones(24,4032,requires_grad=True);constant=F.normalize(x,dim=1);cce,cv,_=dv.additions(constant,4);(cce+cv).backward();cg=float(x.grad.abs().max());assert abs(float(cce)-np.log(8))<1e-6 and abs(float(cv)-.49)<1e-6 and cg<1e-7
 near=(torch.ones(24,4032)+.001*torch.randn(24,4032)).requires_grad_();nce,nvar,_=dv.additions(F.normalize(near,dim=1),4);nvar.backward();ng=float(near.grad.norm());assert ng>0 and torch.isfinite(near.grad).all()
 result=dict(passed=True,cpu_only=True,heldout_images_rendered=0,real_outcomes_read=0,files={str(p):dv.sha(p) for p in dv.files()},exact_original_loss_components=True,logit_shape=list(logits.shape),variance_formula_numpy_error=abs(float(var)-float(manual)),parameter_count=42480,all_gradients_finite=True,gradient_norm=float(torch.cat([p.grad.flatten() for p in net.parameters()]).norm()),training_example_stats=stats,exact_collapse_gradient_max=cg,exact_collapse_infonce=float(cce),exact_collapse_variance_penalty=float(cv),near_collapse_variance_gradient_norm=ng,limitation='This is a soft near-collapse safeguard, not mathematical exclusion of the exact constant stationary point.')
 dv.save(dv.D/'checks.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
