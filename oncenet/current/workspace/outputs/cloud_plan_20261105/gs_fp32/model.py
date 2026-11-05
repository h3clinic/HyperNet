"""Own 42,480-parameter encoder and fixed relation readout for v44.

The fixed readout is applied identically to old, new and random backbones.
No pretrained external weights or extra learned projection is introduced.
"""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parent;P=R.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False))
 for i in range(30):
  try:os.replace(t,p);return
  except PermissionError:
   if i==29:raise
   time.sleep(.1)
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.GroupNorm(4,24),nn.GELU(),nn.Conv2d(24,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.GroupNorm(6,48),nn.GELU(),nn.Conv2d(48,48,3,padding=1),nn.GELU())
 def forward(self,x):return self.net(x)
def load_encoder(path,device='cpu'):
 state=torch.load(path,map_location='cpu',weights_only=True);state=state.get('model',state)
 net=Encoder();net.load_state_dict({k:v for k,v in state.items() if k.startswith('net.')},strict=True);return net.to(device).eval()
def export_state(model,template_path):
 state=torch.load(template_path,map_location='cpu',weights_only=True)
 state.update({k:v.detach().cpu() for k,v in model.state_dict().items()});return state
def descriptor_base(d):
 blocks=[]
 for size in (1,2,4):
  m=F.adaptive_avg_pool2d(d,size);s=(F.adaptive_avg_pool2d(d.square(),size)-m.square()).clamp_min(1e-8).sqrt()
  blocks.append(F.normalize(torch.cat([m.flatten(1),s.flatten(1)],1),dim=1))
 return F.normalize(torch.cat(blocks,1),dim=1)
def descriptor_relation(d):
 """3424D/view: base2016 + fine grouped moments1024 + lag correlations384."""
 base=descriptor_base(d);g=F.adaptive_avg_pool2d(d,(8,8));group=g.reshape(len(g),8,6,8,8)
 spatial=torch.cat([group.mean(2).flatten(1),group.std(2,unbiased=False).flatten(1)],1)
 centered=g-g.mean((2,3),keepdim=True);centered=centered/(centered.square().mean((2,3),keepdim=True).sqrt()+1e-4)
 corr=[]
 for dy,dx in [(0,1),(0,2),(0,4),(1,0),(2,0),(4,0),(1,1),(2,2)]:
  h,w=g.shape[-2:];a=centered[:,:,:h-dy,:w-dx];b=centered[:,:,dy:,dx:]
  corr.append((a*b).mean((2,3)))
 return F.normalize(torch.cat([base,.5*F.normalize(spatial,dim=1),.5*F.normalize(torch.cat(corr,1),dim=1)],1),dim=1)
def paired_descriptor(net,full,low,mode='relation'):
 d=net.net(torch.cat([full,low]));f=(descriptor_relation if mode=='relation' else descriptor_base)(d);a,b=f.chunk(2)
 return F.normalize(torch.cat([a,b],1),dim=1)
def tensor(arrays,device):return torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).to(device)
