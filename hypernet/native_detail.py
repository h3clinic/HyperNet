"""Native-pixel geometric detail skip path for our own CNN region detector.

This is a disclosed analytic prior for high-contrast2D shapes, not a learned
general semantic representation. It segments each RGB proposal, aligns filled
silhouettes, then compares internal connected features without a global128D
projection. No renderer mask, family ID, query label or target box is accepted.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.optimize import linear_sum_assignment

ANGLES = 360
RADII = 48
EXTENT = 2.6
CANONICAL = 128


def _unit_mask(rgb, box):
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected RGB uint8')
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    if not np.isfinite([x1,y1,x2,y2]).all() or x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid candidate box')
    pad = max(4., .12*max(x2-x1,y2-y1))
    left, top = max(0,int(math.floor(x1-pad))), max(0,int(math.floor(y1-pad)))
    right, bottom = min(width,int(math.ceil(x2+pad))), min(height,int(math.ceil(y2+pad)))
    patch = image[top:bottom,left:right].astype(np.float32)
    if min(patch.shape[:2]) < 8:
        return None
    # Fit a smooth background plane to the region perimeter. No dataset-specific
    # color, renderer background or ground-truth foreground is consulted.
    yy,xx = np.mgrid[:patch.shape[0],:patch.shape[1]].astype(np.float32)
    edge = (xx<2)|(yy<2)|(xx>=patch.shape[1]-2)|(yy>=patch.shape[0]-2)
    basis = np.stack([np.ones_like(xx), xx/max(1,patch.shape[1]-1), yy/max(1,patch.shape[0]-1)],-1)
    coefficients = np.linalg.lstsq(basis[edge],patch[edge],rcond=None)[0]
    residual = np.linalg.norm(patch-basis@coefficients,axis=2)
    upper = float(np.percentile(residual,99))
    if upper < 12:
        return None
    contrast = np.clip(residual/upper*255,0,255).astype(np.uint8)
    cutoff,_ = cv2.threshold(contrast,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    mask = contrast > max(float(cutoff),25.)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    if count <= 1:
        return None
    # The input is already a CNN proposal. Select its central connected object,
    # penalizing components whose centroid falls outside the proposal interior.
    candidates = []
    for index in range(1,count):
        area = int(stats[index,cv2.CC_STAT_AREA])
        if area < 20:
            continue
        py,px = np.nonzero(labels==index)
        cx,cy = float(px.mean())+left,float(py.mean())+top
        inside = x1 <= cx <= x2 and y1 <= cy <= y2
        candidates.append((area if inside else area*.1,index))
    if not candidates:
        return None
    mask = labels==max(candidates)[1]
    filled = binary_fill_holes(mask)
    py,px = np.nonzero(filled)
    center = np.array([px.mean(),py.mean()],np.float64)
    radius = float(np.sqrt(np.mean((px-center[0])**2+(py-center[1])**2)))
    if radius < 3:
        return None
    return dict(mask=mask,filled=filled,center=center,radius=radius,
        offset=np.array([left,top]), contrast=upper)


def _polar(filled,center,radius):
    theta = np.arange(ANGLES,dtype=np.float32)*2*np.pi/ANGLES
    radial = (np.arange(RADII,dtype=np.float32)+.5)/RADII*EXTENT
    x = center[0]+radius*radial[:,None]*np.cos(theta)[None]
    y = center[1]+radius*radial[:,None]*np.sin(theta)[None]
    return cv2.remap(filled.astype(np.float32),x.astype(np.float32),y.astype(np.float32),
        cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def _canonical(filled,center,radius,angle=0):
    coordinate = ((np.arange(CANONICAL,dtype=np.float32)+.5)/CANONICAL*2-1)*EXTENT
    xx,yy = np.meshgrid(coordinate,coordinate)
    a=math.radians(angle); c,s=math.cos(a),math.sin(a)
    # Output is the reference-oriented frame; query sampling rotates forward.
    x=center[0]+radius*(c*xx-s*yy); y=center[1]+radius*(s*xx+c*yy)
    return cv2.remap(filled.astype(np.float32),x.astype(np.float32),y.astype(np.float32),
        cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def _features(mask,filled,center,radius,detail_stride=1):
    if detail_stride != 1:
        h,w=mask.shape
        reduced=cv2.resize(mask.astype(np.float32),(max(1,w//detail_stride),max(1,h//detail_stride)),interpolation=cv2.INTER_AREA)
        mask=cv2.resize(reduced,(w,h),interpolation=cv2.INTER_NEAREST)>.5
        filled=binary_fill_holes(mask)
    holes=filled&~mask
    count,labels,stats,_=cv2.connectedComponentsWithStats(holes.astype(np.uint8),connectivity=8)
    features=[]
    for index in range(1,count):
        area=int(stats[index,cv2.CC_STAT_AREA])
        if area < 2:
            continue
        yy,xx=np.nonzero(labels==index)
        points=np.column_stack([xx,yy]).astype(np.float64)
        centroid=points.mean(0)
        centered=points-centroid
        covariance=centered.T@centered/len(points)+np.eye(2)/12
        trace=float(np.trace(covariance))
        eigen=np.linalg.eigvalsh(covariance)
        features.append(dict(position=(centroid-center)/radius,
            covariance=covariance/trace, size=math.sqrt(trace)/radius,
            area=area/(radius*radius), pixels=area,
            anisotropy=float((eigen[-1]-eigen[0])/max(trace,1e-12))))
    return features


def describe_native(rgb,box,detail_stride=1):
    if isinstance(detail_stride,(bool,np.bool_)) or not isinstance(detail_stride,(int,np.integer)) or detail_stride<1:
        raise ValueError('detail_stride must be a positive integer')
    item=_unit_mask(rgb,box)
    if item is None:
        return None
    item['polar']=_polar(item['filled'],item['center'],item['radius'])
    item['canonical']=_canonical(item['filled'],item['center'],item['radius'])
    item['features']=_features(item['mask'],item['filled'],item['center'],item['radius'],detail_stride)
    return item


def align_outline(reference,query):
    # The chosen alignment is independent of hole orientation/identity. Filled
    # silhouettes prevent the micro-pattern from steering its own alignment.
    r=reference['polar']; q=query['polar']
    corr=np.fft.ifft(np.conj(np.fft.fft(r,axis=1))*np.fft.fft(q,axis=1),axis=1).real.sum(0)
    peak=int(corr.argmax())
    candidates=[(peak+offset)%ANGLES for offset in range(-3,4)]
    comparisons=[]
    for angle in candidates:
        canonical=_canonical(query['filled'],query['center'],query['radius'],angle)
        agreement=float(np.minimum(reference['canonical'],canonical).sum()/np.maximum(reference['canonical'],canonical).sum())
        comparisons.append((agreement,angle))
    agreement,angle=max(comparisons,key=lambda pair:(pair[0],-pair[1]))
    return agreement,float(angle)


def compare_native(reference,query,use_detail=True,use_covariance=True):
    if reference is None or query is None:
        return dict(score=0.,outline=0.,detail=0.,angle=None,feature_count=0,reason='blank_or_unresolved')
    outline,angle=align_outline(reference,query)
    if not use_detail:
        return dict(score=outline,outline=outline,detail=None,angle=angle,feature_count=len(query['features']))
    a,b=reference['features'],query['features']
    if len(a)<1 or len(b)<1:
        return dict(score=0.,outline=outline,detail=0.,angle=angle,feature_count=len(b),reason='no_resolved_interior_structure')
    theta=math.radians(-angle); c,s=math.cos(theta),math.sin(theta)
    rotation=np.array([[c,-s],[s,c]])
    positions=np.stack([rotation@f['position'] for f in b])
    distances=np.linalg.norm(np.stack([f['position'] for f in a])[:,None]-positions[None],axis=-1)
    ii,jj=linear_sum_assignment(distances)
    token_errors=[]
    for i,j in zip(ii,jj):
        x,y=a[i],b[j]
        cov=rotation@y['covariance']@rotation.T
        cov_error=float(np.linalg.norm(x['covariance']-cov))
        # Tiny features are positional anchors. If either larger feature is
        # anisotropic, compare the covariance tensors as shape evidence. This is
        # not a guarantee that both estimated orientations are reliable.
        eligible=min(x['pixels'],y['pixels'])>=8 and max(x['anisotropy'],y['anisotropy'])>=.15
        position=float(distances[i,j])
        size=abs(math.log(max(x['size'],1e-9)/max(y['size'],1e-9)))
        token_errors.append((position/.10)**2+.15*(size/.35)**2+((cov_error/.25)**2 if eligible and use_covariance else 0.))
    missing=abs(len(a)-len(b))
    detail=math.exp(-float(np.mean(token_errors))-.75*missing)
    score=(outline**4)*detail
    return dict(score=float(score),outline=outline,detail=detail,angle=angle,
        feature_count=len(b),matched_features=len(ii),missing_features=missing,
        local_errors=[float(value) for value in token_errors])


def match_native(reference_rgb,reference_box,query_rgb,proposals,use_detail=True,detail_stride=1,use_covariance=True):
    reference=describe_native(reference_rgb,reference_box,detail_stride)
    evidence=[compare_native(reference,describe_native(query_rgb,box,detail_stride),use_detail,use_covariance)
        for box in proposals]
    scores=[item['score'] for item in evidence]
    best=int(np.argmax(scores)) if scores else None
    return dict(score=scores[best] if best is not None else 0.,
        bbox_xyxy=list(proposals[best]) if best is not None else None,
        proposals=[list(box) for box in proposals],proposal_scores=scores,
        evidence=evidence[best] if best is not None else {},all_evidence=evidence)
