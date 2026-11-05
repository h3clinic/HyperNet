"""42,480 independently initialized weights; fixed v44-compatible readouts."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
 for attempt in range(30):
  try:os.replace(tmp,path);return
  except PermissionError:
   if attempt==29:raise
   time.sleep(.1)
def atomic_torch(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp)
 for attempt in range(30):
  try:os.replace(tmp,path);return
  except PermissionError:
   if attempt==29:raise
   time.sleep(.1)
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.GroupNorm(4,24),nn.GELU(),nn.Conv2d(24,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.GroupNorm(6,48),nn.GELU(),nn.Conv2d(48,48,3,padding=1),nn.GELU())
 def forward(self,x):return self.net(x)
def load_encoder(path,device='cpu'):
 state=torch.load(path,map_location='cpu',weights_only=True);state=state.get('model',state);net=Encoder();net.load_state_dict({k:v for k,v in state.items() if k.startswith('net.')},strict=True);return net.to(device).eval()
def descriptor_base(d):
 blocks=[]
 for size in (1,2,4):
  m=F.adaptive_avg_pool2d(d,size);s=(F.adaptive_avg_pool2d(d.square(),size)-m.square()).clamp_min(1e-8).sqrt();blocks.append(F.normalize(torch.cat([m.flatten(1),s.flatten(1)],1),dim=1))
 return F.normalize(torch.cat(blocks,1),dim=1)
def descriptor_spatial(d):
 g=F.adaptive_avg_pool2d(d,(8,8)).reshape(len(d),8,6,8,8);spatial=torch.cat([g.mean(2).flatten(1),g.std(2,unbiased=False).flatten(1)],1)
 return F.normalize(torch.cat([descriptor_base(d),.5*F.normalize(spatial,dim=1)],1),dim=1)
def dual(d,mode='base'):
 a,b=(descriptor_base if mode=='base' else descriptor_spatial)(d).chunk(2);return F.normalize(torch.cat([a,b],1),dim=1)
def paired_descriptor(net,full,low,mode='base'):return dual(net(torch.cat([full,low])),mode)
def tensor(arrays,device):return torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).to(device)
