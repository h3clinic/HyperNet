"""Frozen mathematical images plus analytic foreground correspondence labels.

No images are changed from v44's corrected generator. Correspondences use
known affine poses, then require foreground support in both rasterized views.
This module is sequential: the frozen correction module has mutable episode ID.
"""
from pathlib import Path
import importlib.util
import numpy as np
import cv2
BASE=Path(__file__).resolve().parents[2]/'hypernet_v44/correction/generator.py'
_spec=importlib.util.spec_from_file_location('v45_frozen_correction_generator',BASE);_c=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_c)
VIEWS=_c.VIEWS;NEGATIVE_TYPES=_c.NEGATIVE_TYPES
def episode(identity,split='train'):return _c.episode(identity,split)
def local_points(identity,split='train',count=32):
 g=_c._g;graph=g.graph(g.rng(identity,split,'graph'));rp=g.pose(g.rng(identity,split,'reference_pose'));qp=g.pose(g.rng(identity,split,'query_pose'));ra=g.appearance(g.rng(identity,split,'reference_appearance'));qa=g.appearance(g.rng(identity,split,'query_appearance'))
 masks=[]
 for p,a in [(rp,ra),(qp,qa)]:
  masks.append(g.render(graph,p,dict(background=np.zeros((256,256),np.float32),gain=1.,width=a['width']))[0][...,0])
 yy,xx=np.where(masks[0]>.05);xy=np.stack([xx,yy],1).astype(np.float64)
 A=qp[0]@np.linalg.inv(rp[0]);b=qp[1]-A@rp[1];mapped=xy@A.T+b
 valid=np.all((xy>=4)&(xy<=120),1)&np.all((mapped>=4)&(mapped<=120),1)
 support=cv2.remap(masks[1],mapped[:,0].astype(np.float32)[:,None],mapped[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel();valid&=support>.05
 xy=xy[valid];mapped=mapped[valid]
 if len(xy)==0:raise RuntimeError(f'No valid foreground correspondence for {identity}/{split}')
 # Private deterministic draw does not consume any image-generator RNG.
 indices=g.rng(identity,split,'v45_local_foreground_points').choice(len(xy),size=count,replace=len(xy)<count)
 return dict(reference=xy[indices].astype(np.float32),positive=mapped[indices].astype(np.float32),available=int(len(xy)),sampled=count,replacement=bool(len(xy)<count),affine=A,offset=b,minimum_positive_support=float(support[valid].min()))
def annotated_episode(identity,split='train'):
 result=episode(identity,split);result['local']=local_points(identity,split);return result
