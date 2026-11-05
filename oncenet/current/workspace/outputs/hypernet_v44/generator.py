"""Mathematical part graphs with counterfactual relational edits.

An identity is a graph of analytic curved/rippled strokes. Positives preserve
primitive count, connectivity, ordering, orientation and curvature sign, while
pose, stroke width, gain and background vary. Four negatives edit repetition,
connectivity, curvature or orientation/layout. No real image or dataset label
is read. This is an explicit invariance hypothesis, not natural class truth.
"""
import hashlib,copy
import numpy as np
import cv2
SIZE=128;NATIVE=256
NEGATIVE_TYPES=('repetition','connectivity','curvature','orientation')
VIEWS=('reference','positive',*NEGATIVE_TYPES)
def seed_for(identity,split,component):
 if split not in ['train','dev','test']:raise ValueError(split)
 return int.from_bytes(hashlib.sha256(f'v44-part-graphs|{split}|{identity}|{component}'.encode()).digest()[:8],'little')
def rng(identity,split,key):return np.random.default_rng(seed_for(identity,split,key))
def graph(r):
 n=int(r.integers(1,7));layout=int(r.integers(3));center=r.uniform([45,43],[83,94]);radius=r.uniform(13,31)
 if layout==0:
  t=np.linspace(-1,1,n);nodes=np.stack([center[0]+radius*t,center[1]+r.uniform(-8,8)*t],1)
 elif layout==1:
  t=np.linspace(0,2*np.pi,n,endpoint=False)+r.uniform(-.3,.3);nodes=center+radius*np.stack([np.cos(t),.65*np.sin(t)],1)
 else:
  nodes=center+np.stack([(np.arange(n)%3-1)*radius*.6,(np.arange(n)//3-.5)*radius*.6],1)
 nodes+=r.uniform(-1.5,1.5,nodes.shape)
 connect=int(r.integers(3));edges=[]
 if connect==1:edges=[(i,i+1) for i in range(n-1)]
 if connect==2:edges=[(0,i) for i in range(1,n)]
 return dict(nodes=nodes,edges=edges,theta=float(r.choice([0,np.pi/2,np.pi/4])+r.uniform(-.10,.10)),curvature=float(r.choice([-1,0,1])*r.uniform(.25,.6)),ripple=float(r.uniform(.02,.10)),cycles=int(r.integers(2,7)),radius=float(r.uniform(4.5,9)),width=float(r.uniform(.65,1.5)),layout=layout)
def mutate(g,kind,r):
 z=copy.deepcopy(g);n=len(z['nodes'])
 if kind=='repetition':
  # Add a detached copy; total foreground RMS is normalized after rendering.
  new=z['nodes'][int(r.integers(n))]+r.choice([-1,1],2)*r.uniform(13,21,2);new=np.clip(new,12,116)
  z['nodes']=np.concatenate([z['nodes'],new[None]],0)
 elif kind=='connectivity':
  if n>1:
   if z['edges']:z['edges']=z['edges'][:-1]
   else:z['edges']=[(0,n-1)]
  else:z['curvature']=.6 if abs(g['curvature'])<.1 else 0.
 elif kind=='curvature':z['curvature']=float(r.choice([-1,1])*.65) if abs(g['curvature'])<.1 else 0.
 elif kind=='orientation':
  z['theta']+=np.pi/2;center=z['nodes'].mean(0);d=z['nodes']-center;z['nodes']=center+np.stack([-d[:,1],d[:,0]],1)
 else:raise ValueError(kind)
 return z
def pose(r):
 scale=r.uniform(.90,1.06,2);angle=r.uniform(-.035,.035);c,s=np.cos(angle),np.sin(angle);a=np.array([[c,-s],[s,c]])@np.diag(scale);shift=r.uniform(-2.5,2.5,2)
 return a,63.5+shift-a@np.array([63.5,63.5])
def appearance(r):
 yy,xx=np.mgrid[:NATIVE,:NATIVE]/NATIVE;b=np.full((NATIVE,NATIVE),r.uniform(.005,.05),np.float32)
 for _ in range(3):b+=r.uniform(.001,.006)*np.sin(r.uniform(-15,15)*xx+r.uniform(-15,15)*yy+r.uniform(-np.pi,np.pi))
 b+=r.normal(0,r.uniform(.0004,.003),b.shape)
 return dict(background=b,gain=float(np.exp(r.uniform(np.log(.012),np.log(.06)))),width=float(r.uniform(.8,1.2)))
def render(g,p,a):
 field=np.zeros((NATIVE,NATIVE),np.uint8);A,t=p
 def stroke(points,width):
  points=points@A.T+t;points=np.rint(2*(points+.5)-.5).astype(np.int32)
  cv2.polylines(field,[points],False,255,max(1,round(2*width)),cv2.LINE_AA)
 u=np.linspace(-1,1,65);c,s=np.cos(g['theta']),np.sin(g['theta'])
 local=np.stack([g['radius']*u,g['radius']*(g['curvature']*(1-u*u)+g['ripple']*np.sin(np.pi*g['cycles']*u))],1)@np.array([[c,s],[-s,c]])
 for point in g['nodes']:stroke(local+point,g['width']*a['width'])
 tt=np.linspace(0,1,81)
 for i,j in g['edges']:
  start,end=g['nodes'][i],g['nodes'][j];delta=end-start;normal=np.array([-delta[1],delta[0]])
  points=start+tt[:,None]*delta+g['curvature']*.25*np.sin(np.pi*tt[:,None])*normal
  stroke(points,g['width']*a['width'])
 f=field.astype(np.float32)/255
 # Equal RMS removes a total-energy shortcut between query alternatives.
 f=f/max(float(np.sqrt(np.mean(f*f))),1e-5)
 image=np.clip(a['background']+a['gain']*f,0,1).astype(np.float32)
 small=cv2.resize(image,(SIZE,SIZE),interpolation=cv2.INTER_AREA);low=cv2.resize(image[2*NATIVE//3:],(SIZE,SIZE),interpolation=cv2.INTER_AREA)
 rgb=lambda z:np.repeat(z[...,None],3,axis=-1)
 return rgb(small),rgb(low),dict(foreground_rms=float(np.sqrt(np.mean(f*f))),clipped_fraction=float(np.mean(a['background']+a['gain']*f>1)))
def episode(identity,split='train'):
 g=graph(rng(identity,split,'graph'));rp=pose(rng(identity,split,'reference_pose'));qp=pose(rng(identity,split,'query_pose'));ra=appearance(rng(identity,split,'reference_appearance'));qa=appearance(rng(identity,split,'query_appearance'))
 out={};metas={}
 out['reference'],out['reference_low'],metas['reference']=render(g,rp,ra)
 out['positive'],out['positive_low'],metas['positive']=render(g,qp,qa)
 for kind in NEGATIVE_TYPES:out[kind],out[kind+'_low'],metas[kind]=render(mutate(g,kind,rng(identity,split,kind)),qp,qa)
 out['metadata']=dict(identity=identity,split=split,latent_seed=seed_for(identity,split,'graph'),primitive_count=len(g['nodes']),edges=len(g['edges']),layout=g['layout'],positive_preserves_count_topology_orientation=True,query_alternatives_share_background_pose_gain_and_width=True,foreground_normalization='RMS one before shared gain',render_checks=metas,limitations='Exact graph identity views, not broad semantic category variation; count and orientation may or may not define real category membership.')
 return out
