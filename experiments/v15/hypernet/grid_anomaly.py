"""One-normal-reference anomaly scoring with no query adaptation or mask input."""
from hashlib import sha256
import math
import cv2
import numpy as np
import torch

SAMPLING_STRIDE=2
MAX_BANK=4096
BANK_SEED=1515
QUERY_CHUNK=2048
TOP_FRACTION=.005

def array_hash(array):
    array=np.ascontiguousarray(array)
    return sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()

def bank_indices(count,maximum=MAX_BANK):
    if count<1 or maximum<1:raise ValueError('Reference bank cannot be empty')
    if count<=maximum:return np.arange(count,dtype=np.int64)
    return np.sort(np.random.default_rng(BANK_SEED).choice(count,maximum,replace=False))

def normalized(values):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all():raise ValueError('Finite nonempty descriptor matrix required')
    norm=np.linalg.norm(values,axis=1,keepdims=True)
    return np.ascontiguousarray(values/np.maximum(norm,1e-8)),norm[:,0]<=1e-8

def validate_features(features,key):
    values=np.asarray(features[key]);xy=np.asarray(features['xy']);h,w=features['source_shape']
    if values.ndim!=3 or xy.shape!=(*values.shape[:2],2) or not np.isfinite(values).all() or not np.isfinite(xy).all():
        raise ValueError('Invalid dense feature map')
    if min(h,w)<1 or features.get('source_pixels_resized') is not False:raise ValueError('Original-resolution features required')
    if features.get('sampling_stride')!=SAMPLING_STRIDE:raise ValueError('Frozen stride differs')
    gh,gw=values.shape[:2]
    expected_x=(np.arange(gw,dtype=np.float64)*SAMPLING_STRIDE+.5)/w
    expected_y=(np.arange(gh,dtype=np.float64)*SAMPLING_STRIDE+.5)/h
    if (gh,gw)!=((h+SAMPLING_STRIDE-1)//SAMPLING_STRIDE,(w+SAMPLING_STRIDE-1)//SAMPLING_STRIDE):raise ValueError('Feature grid dimensions differ')
    if not np.allclose(xy[...,0],expected_x[None,:],rtol=0,atol=1e-7) or not np.allclose(xy[...,1],expected_y[:,None],rtol=0,atol=1e-7):
        raise ValueError('Feature coordinates do not follow native pixel centers')
    return values

def image_score(patch_scores):
    values=np.asarray(patch_scores,dtype=np.float32).ravel()
    if not len(values) or not np.isfinite(values).all():raise ValueError('Finite anomaly scores required')
    count=max(1,math.ceil(TOP_FRACTION*len(values)))
    return float(np.partition(values,len(values)-count)[-count:].mean(dtype=np.float64))

def original_pixel_map(patch_scores,xy,source_shape):
    """Interpolate scores at actual sampled centers; never resize target masks."""
    scores=np.asarray(patch_scores,dtype=np.float32);xy=np.asarray(xy,dtype=np.float32);h,w=source_shape
    if scores.ndim!=2 or xy.shape!=(*scores.shape,2):raise ValueError('Score/coordinate shape mismatch')
    xcenters=xy[0,:,0].astype(np.float64)*w-.5
    ycenters=xy[:,0,1].astype(np.float64)*h-.5
    if (len(xcenters)>1 and np.any(np.diff(xcenters)<=0)) or (len(ycenters)>1 and np.any(np.diff(ycenters)<=0)):
        raise ValueError('Sample coordinates must be ordered')
    ix=np.interp(np.arange(w),xcenters,np.arange(len(xcenters))).astype(np.float32)
    iy=np.interp(np.arange(h),ycenters,np.arange(len(ycenters))).astype(np.float32)
    map_x,map_y=np.meshgrid(ix,iy)
    return cv2.remap(scores,map_x,map_y,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

class ReferenceBank:
    """Fit once using reference descriptors only; score calls never alter it."""
    def __init__(self,features,*,key='descriptors',device='cpu'):
        values=validate_features(features,key);flat=values.reshape(-1,values.shape[-1])
        self.indices=bank_indices(len(flat));bank,zeros=normalized(flat[self.indices])
        self.key=key;self.device=device;self.bank_np=bank;self.bank=torch.from_numpy(bank).to(device)
        self.reference_has_zero=bool(zeros.any());self.reference_grid=list(values.shape[:2])
        self.bank_sha256=array_hash(bank);self.indices_sha256=array_hash(self.indices)

    @torch.inference_mode()
    def score(self,features):
        values=validate_features(features,self.key)
        if values.shape[-1]!=self.bank.shape[1]:raise ValueError('Reference/query feature dimensions differ')
        query,zeros=normalized(values.reshape(-1,values.shape[-1]));scores=np.empty(len(query),dtype=np.float32)
        for start in range(0,len(query),QUERY_CHUNK):
            tensor=torch.from_numpy(query[start:start+QUERY_CHUNK]).to(self.device)
            similarity=(tensor@self.bank.T).amax(1).clamp(-1,1)
            residual=(1-similarity).cpu().numpy()
            # An exactly constant zero feature is normal if the reference has
            # the same feature; otherwise its cosine similarity remains zero.
            if self.reference_has_zero:residual[zeros[start:start+QUERY_CHUNK]]=0
            scores[start:start+len(residual)]=residual
        patch=scores.reshape(values.shape[:2])
        return {'score':image_score(patch),'patch_scores':patch,'xy':np.asarray(features['xy'],dtype=np.float32),
                'source_shape':list(features['source_shape']),'bank_sha256':self.bank_sha256,
                'bank_indices_sha256':self.indices_sha256,'bank_tokens':len(self.indices),
                'query_tokens':len(query),'image_score_rule':'mean largest ceil(0.005*N) patch cosine residuals'}

    def provenance(self):
        if array_hash(self.bank_np)!=self.bank_sha256 or array_hash(self.bank.detach().cpu().numpy())!=self.bank_sha256:
            raise RuntimeError('Reference bank changed')
        return {'feature_key':self.key,'reference_grid':self.reference_grid,'bank_tokens':len(self.indices),
            'bank_sha256':self.bank_sha256,'bank_indices_sha256':self.indices_sha256,
            'selection':'Uniform without replacement, NumPy seed1515, sorted indices, max4096; no query fitting',
            'normal_reference_images':1,'query_adaptation':False}

class GridAnomaly:
    """FrozenDetails extraction plus separate learned/random/analytic banks."""
    def __init__(self,checkpoints,device='cuda',batch_size=32):
        from .detail_features import FrozenDetails
        if not checkpoints or 'analytic_detail' in checkpoints:raise ValueError('Named own checkpoints required')
        self.extractors={name:FrozenDetails(path,device=device,batch_size=batch_size) for name,path in checkpoints.items()}
        self.device=device;self.banks={};self.initial={name:model.provenance() for name,model in self.extractors.items()}

    def enroll(self,reference_rgb):
        if self.banks:raise ValueError('Exactly one reference enrollment is permitted')
        for index,(name,extractor) in enumerate(self.extractors.items()):
            features=extractor.encode(reference_rgb,sampling_stride=SAMPLING_STRIDE)
            self.banks[name]=ReferenceBank(features,device=self.device)
            if index==0:self.banks['analytic_detail']=ReferenceBank(features,key='detail',device=self.device)

    def predict(self,query_rgb):
        if not self.banks:raise ValueError('Enroll the normal reference before scoring')
        predictions={}
        for index,(name,extractor) in enumerate(self.extractors.items()):
            features=extractor.encode(query_rgb,sampling_stride=SAMPLING_STRIDE)
            predictions[name]=self.banks[name].score(features)
            if index==0:predictions['analytic_detail']=self.banks['analytic_detail'].score(features)
        return {name:predictions[name] for name in [*self.extractors,'analytic_detail']}

    def provenance(self):
        current={name:model.provenance() for name,model in self.extractors.items()}
        if current!=self.initial:raise RuntimeError('Frozen CNN state/checkpoint changed')
        return {'models':current,'banks':{name:bank.provenance() for name,bank in self.banks.items()},
            'sampling_stride':SAMPLING_STRIDE,'top_fraction':TOP_FRACTION,'max_bank_tokens':MAX_BANK,
            'feature_normalization':'Per-token L2; no DEV/query mean/covariance fitting',
            'analytic_control':'Fixed local contrast, gx, gy and Laplacian from native encoder preprocessing; no learned mask'}
