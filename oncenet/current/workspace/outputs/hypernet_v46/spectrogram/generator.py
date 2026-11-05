"""Analytic time-frequency power ridges with exact affine part coordinates.

These are domain-informed 2D approximations, not simulated detector strain or
an exact Q-transform. No real image, class annotation, or neural model is read.
"""
import copy,hashlib
import numpy as np
import cv2
SIZE=128;NATIVE=256;FMIN=10.;FMAX=2048.;LOGR=np.log(FMAX/FMIN)
NEGATIVE_TYPES=('repetition','connectivity','curvature','orientation')
VIEWS=('reference','positive',*NEGATIVE_TYPES)
FAMILIES=('chirp','line','arch','burst','repeated')
COMPONENT_KEYS=('scene','reference_pose','query_pose','reference_appearance','query_appearance',*NEGATIVE_TYPES,'local_points')
_yy,_xx=np.mgrid[:NATIVE,:NATIVE].astype(np.float32)
TIME=(_xx+.5)/NATIVE;LOGF=1-(_yy+.5)/NATIVE
def seed_for(identity,split,component):
 assert split in ['train','dev','test'];return int.from_bytes(hashlib.sha256(f'v46-spectral-power|{split}|{identity}|{component}'.encode()).digest()[:8],'little')
def rng(identity,split,key):return np.random.default_rng(seed_for(identity,split,key))
def scene(identity,split='train'):
 r=rng(identity,split,'scene');family=FAMILIES[identity%len(FAMILIES)];n=int(r.integers(1,5));components=[]
 if family=='repeated':n=int(r.integers(2,5))
 base_frequency=float(np.exp(r.uniform(np.log(18),np.log(700))));time0=float(r.uniform(.3,.7));spacing=float(r.uniform(.10,.18))
 for j in range(n):
  kind='burst'if family=='repeated'else family
  if family not in ['repeated']and j>0 and r.random()<.20:kind=str(r.choice(['chirp','line','arch','burst']))
  t0=float(np.clip(time0+(j-(n-1)/2)*spacing,.12,.88))if family=='repeated'else float(r.uniform(.18,.82))
  f0=base_frequency if family=='repeated'else float(np.exp(r.uniform(np.log(18),np.log(1200))))
  width=float(r.uniform(.07,.24))
  if kind=='line':width=float(r.uniform(.24,.60))
  if kind=='arch':width=float(r.uniform(.20,.52))
  if kind=='burst':width=float(r.uniform(.009,.045))
  components.append(dict(kind=kind,t0=t0,f0=f0,time_width=width,q=float(r.uniform(9,48)),slope=float(r.choice([-1,1])*r.uniform(.10,.42)),curve=float(r.choice([-1,1])*r.uniform(.05,.23)),period=float(r.uniform(.28,1.15)),phase=float(r.uniform(-.2,.2)),burst_bandwidth=float(r.uniform(.035,.14)),gap=False,gap_center=float(r.uniform(-.35,.35)),gap_width=float(r.uniform(.10,.25)),amplitude=float(r.uniform(.65,1.3))))
 return dict(family=family,components=components)
def mutate(g,kind,r):
 z=copy.deepcopy(g);i=int(r.integers(len(z['components'])));c=z['components'][i]
 if kind=='repetition':
  new=copy.deepcopy(c);delta=float(r.uniform(.14,.24));sgn=float(r.choice([-1,1]));new['t0']=c['t0']+sgn*delta
  if not(.08<new['t0']<.92):new['t0']=c['t0']-sgn*delta
  if any(abs(new['t0']-o['t0'])<.04 and abs(np.log(new['f0']/o['f0']))<.12 for o in z['components']):new['f0']*=1.45
  z['components'].append(new)
 elif kind=='connectivity':c['gap']=True
 elif kind=='curvature':
  if c['kind']=='arch':c['period']*=.55
  elif c['kind']=='burst':c['burst_bandwidth']*=.42;c['time_width']*=1.8
  else:c['curve']=-np.sign(c['curve'])*.48;c['kind']='chirp'
 elif kind=='orientation':
  if c['kind']=='line':c['kind']='chirp';c['slope']=float(r.choice([-1,1])*.6);c['curve']=0.
  elif c['kind']=='arch':c['kind']='chirp';c['slope']=float(r.choice([-1,1])*.5);c['curve']=.2
  elif c['kind']=='burst':c['kind']='line';c['time_width']=max(.16,c['time_width']*4)
  else:c['slope']=-c['slope'];c['curve']=-c['curve']
 else:raise ValueError(kind)
 return z
def pose(r):
 # Axis-preserving time dilation/frequency-scale changes; no nonphysical rotation.
 A=np.diag(r.uniform(.92,1.06,2));shift=r.uniform(-2.5,2.5,2);return A,63.5+shift-A@np.array([63.5,63.5])
def appearance(r):
 # Positive colored speckle plus a slowly changing spectral floor. This is an
 # image-space noise approximation, not a fitted detector PSD or strain sample.
 white=r.normal(size=(NATIVE,NATIVE));ft=np.fft.rfft2(white);fy=np.fft.fftfreq(NATIVE)[:,None];fx=np.fft.rfftfreq(NATIVE)[None,:]
 spectrum=(.003**2+(fx/r.uniform(.6,1.8))**2+(fy/r.uniform(.6,1.8))**2)**(-r.uniform(.2,.65)/2)
 noise=np.fft.irfft2(ft*spectrum,s=white.shape).real;noise=(noise-noise.mean())/max(noise.std(),1e-8)
 envelope=np.exp(r.uniform(-.8,.8)*(LOGF-.5));background=r.uniform(.004,.035)*envelope*np.exp(r.uniform(.12,.45)*noise)
 return dict(background=background.astype(np.float32),gain=float(np.exp(r.uniform(np.log(.018),np.log(.10)))),width=float(r.uniform(.85,1.15)),time_width=float(r.uniform(.92,1.08)))
def foreground(g,p,a):
 field=np.zeros((NATIVE,NATIVE),np.float32)
 for c in g['components']:
  u=(TIME-c['t0'])/c['time_width'];envelope=np.exp(-.5*(u/(.65*a['time_width']))**2)
  if c['kind']=='burst':
   # Localized sine-Gaussian-like power patch; bandwidth is bounded by the
   # time/frequency uncertainty scale, then broadened procedurally.
   center=np.log(c['f0']/FMIN)/LOGR;width=max(c['burst_bandwidth'],1/(4*np.pi*c['time_width']*c['f0']*LOGR));ridge=np.exp(-.5*((LOGF-center)/(width*a['width']))**2)
  else:
   if c['kind']=='line':f=np.full_like(TIME,c['f0'])
   elif c['kind']=='arch':f=c['f0']*np.abs(np.cos(np.pi*(TIME-c['t0'])/c['period']+c['phase']))
   else:f=c['f0']*np.maximum(.06,1+c['slope']*u+c['curve']*u*u)
   center=np.log(np.maximum(f,.5)/FMIN)/LOGR;width=a['width']/(c['q']*LOGR);ridge=np.exp(-.5*((LOGF-center)/width)**2)
  if c['gap']:envelope*=1-.99*np.exp(-.5*((u-c['gap_center'])/c['gap_width'])**8)
  field+=c['amplitude']*envelope*ridge
 A,t=p;native_t=2*t+.5-A@np.array([.5,.5]);matrix=np.concatenate([A,native_t[:,None]],1).astype(np.float32)
 field=cv2.warpAffine(field,matrix,(NATIVE,NATIVE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 field/=max(float(np.sqrt(np.mean(field*field))),1e-6)
 return field
def render(g,p,a):
 f=foreground(g,p,a);power=a['background']+a['gain']*f;image=np.log1p(8*power)/np.log(9);image=np.clip(image,0,1).astype(np.float32)
 full=cv2.resize(image,(SIZE,SIZE),interpolation=cv2.INTER_AREA);low=cv2.resize(image[2*NATIVE//3:],(SIZE,SIZE),interpolation=cv2.INTER_AREA);rgb=lambda x:np.repeat(x[...,None],3,axis=-1)
 return rgb(full),rgb(low),dict(foreground_rms=float(np.sqrt(np.mean(f*f))),clipped_fraction=float(np.mean(power>1)),time_seconds=1.,frequency_bounds_hz=[FMIN,FMAX])
def local_points(g,rp,qp,ra,qa,identity,split,count=32):
 masks=[cv2.resize(foreground(g,p,a),(SIZE,SIZE),interpolation=cv2.INTER_AREA)for p,a in [(rp,ra),(qp,qa)]]
 masks=[m/max(float(m.max()),1e-6)for m in masks];yy,xx=np.where(masks[0]>.12);xy=np.stack([xx,yy],1).astype(np.float64);A=qp[0]@np.linalg.inv(rp[0]);b=qp[1]-A@rp[1];mapped=xy@A.T+b
 valid=np.all((xy>=4)&(xy<=120),1)&np.all((mapped>=4)&(mapped<=120),1);support=cv2.remap(masks[1],mapped[:,0].astype(np.float32)[:,None],mapped[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel();valid&=support>.12;xy=xy[valid];mapped=mapped[valid]
 if len(xy)==0:raise RuntimeError(f'No foreground correspondence: {identity}/{split}')
 indices=rng(identity,split,'local_points').choice(len(xy),count,replace=len(xy)<count)
 return dict(reference=xy[indices].astype(np.float32),positive=mapped[indices].astype(np.float32),available=len(xy),replacement=bool(len(xy)<count),affine=A,offset=b,minimum_positive_support=float(support[valid].min()))
def episode(identity,split='train',annotate=False):
 g=scene(identity,split);rp=pose(rng(identity,split,'reference_pose'));qp=pose(rng(identity,split,'query_pose'));ra=appearance(rng(identity,split,'reference_appearance'));qa=appearance(rng(identity,split,'query_appearance'));out={};metas={}
 out['reference'],out['reference_low'],metas['reference']=render(g,rp,ra);out['positive'],out['positive_low'],metas['positive']=render(g,qp,qa)
 for kind in NEGATIVE_TYPES:out[kind],out[kind+'_low'],metas[kind]=render(mutate(g,kind,rng(identity,split,kind)),qp,qa)
 out['metadata']=dict(identity=identity,split=split,latent_seed=seed_for(identity,split,'scene'),family=g['family'],component_count=len(g['components']),positive_preserves_component_count_and_layout=True,query_alternatives_share_background_pose_gain_and_width=True,render_checks=metas,limits='Analytic power-ridge approximation; not a waveform/Q-transform simulator or semantic class label.')
 if annotate:out['local']=local_points(g,rp,qp,ra,qa,identity,split)
 return out
def annotated_episode(identity,split='train'):return episode(identity,split,True)
