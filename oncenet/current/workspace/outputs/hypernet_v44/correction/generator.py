"""Only repetition placement differs from the frozen v44 generator."""
from pathlib import Path
import importlib.util,copy
import numpy as np
BASE=Path(__file__).resolve().parent.parent/'generator.py'
_s=importlib.util.spec_from_file_location('correction_frozen_generator',BASE);_g=importlib.util.module_from_spec(_s);_s.loader.exec_module(_g)
_mutate=_g.mutate;VIEWS=_g.VIEWS;NEGATIVE_TYPES=_g.NEGATIVE_TYPES
_identity=None;_last=None
def repetition(g,r,identity):
 z=copy.deepcopy(g);n=len(g['nodes']);source=int(r.integers(n));signs=r.choice([-1,1],2);magnitudes=r.uniform(13,21,2);nominal=signs*magnitudes;length=float(np.linalg.norm(nominal));axis=('horizontal','vertical','diagonal')[(identity-4400000)%3]
 if axis=='horizontal':offsets=[np.array([signs[0]*length,0.]),np.array([-signs[0]*length,0.])]
 elif axis=='vertical':offsets=[np.array([0.,signs[1]*length]),np.array([0.,-signs[1]*length])]
 else:offsets=[nominal,nominal*np.array([-1,1]),nominal*np.array([1,-1]),-nominal]
 for anchor in [source]+[i for i in range(n) if i!=source]:
  for direction,offset in enumerate(offsets):
   candidate=g['nodes'][anchor]+offset
   if np.all(candidate>=12) and np.all(candidate<=116) and np.linalg.norm(g['nodes']-candidate,axis=1).min()>=2:
    z['nodes']=np.concatenate([z['nodes'],candidate[None]],0)
    return z,dict(axis=axis,nominal_length=length,actual_length=float(np.linalg.norm(offset)),dx=float(offset[0]),dy=float(offset[1]),anchor_changed=anchor!=source,direction_reflected=direction!=0,minimum_node_separation=float(np.linalg.norm(g['nodes']-candidate,axis=1).min()))
 raise RuntimeError('No valid nonduplicate repetition placement')
def mutate(g,kind,r):
 global _last
 if kind!='repetition':return _mutate(g,kind,r)
 z,_last=repetition(g,r,_identity);return z
_g.mutate=mutate
def episode(identity,split='train'):
 global _identity,_last
 _identity=identity;_last=None;result=_g.episode(identity,split);result['metadata']['correction_repetition']=_last
 return result
