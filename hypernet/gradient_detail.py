"""Reference-selected native gradient patches after filled-silhouette alignment.

This is a disclosed local-image prior, not an external pretrained model.
No interior connected-feature/covariance descriptor is extracted or consulted.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from .native_detail import _unit_mask, _polar, _canonical, align_outline

PATCH_SIDE=7
MAX_PATCHES=12
SUPPRESSION_RADIUS=5
TRANSLATIONS=((0,0),(-1,0),(1,0),(0,-1),(0,1))
MIN_PATCH_RMS=.1

def gray(rgb):return np.asarray(rgb,np.float32)@np.array([.299,.587,.114],np.float32)
def patch(gray_image,x,y):
    return cv2.remap(gray_image,x.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101)
def unit(v):
    v=np.asarray(v,np.float64).ravel();v=v-v.mean();norm=np.linalg.norm(v)
    return v/norm if norm>=MIN_PATCH_RMS*math.sqrt(v.size) else np.zeros_like(v)

def detail_rgb(rgb,fine_stride=1):
    """Degrade image evidence only; all native locations/outline geometry stay fixed."""
    if isinstance(fine_stride,bool) or not isinstance(fine_stride,(int,np.integer)) or fine_stride<1:
        raise ValueError('fine_stride must be a positive integer')
    if fine_stride==1:return rgb
    image=np.asarray(rgb,np.float32);h,w=image.shape[:2]
    small=cv2.resize(image,(max(1,w//fine_stride),max(1,h//fine_stride)),interpolation=cv2.INTER_AREA)
    return cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR)

def reference_patches(rgb,description):
    if description is None:return []
    image=gray(rgb);ox,oy=map(int,description['offset']);h,w=description['filled'].shape
    local=image[oy:oy+h,ox:ox+w]
    dx=cv2.Sobel(local,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(local,cv2.CV_32F,0,1,ksize=3)
    strength=np.sqrt(dx*dx+dy*dy)
    # Interior exclusion avoids selecting the macro outline repeatedly. This
    # selects arbitrary gradients/textures/strokes; no hole component is read.
    interior=cv2.distanceTransform(description['filled'].astype(np.uint8),cv2.DIST_L2,5)>3
    strength=np.where(interior,strength,0.)
    cutoff=max(4.,float(strength.max())*.03)
    yy,xx=np.mgrid[:h,:w];records=[]
    u,v=np.meshgrid(np.arange(PATCH_SIDE)-PATCH_SIDE//2,np.arange(PATCH_SIDE)-PATCH_SIDE//2)
    while len(records)<MAX_PATCHES and float(strength.max())>=cutoff:
        y,x=np.unravel_index(int(strength.argmax()),strength.shape)
        pixels=patch(image,x+ox+u,y+oy+v)
        records.append(dict(position=(np.array([x,y])-description['center'])/description['radius'],
            vector=unit(pixels),u=u/description['radius'],v=v/description['radius']))
        strength[(xx-x)**2+(yy-y)**2<=SUPPRESSION_RADIUS**2]=0
    return records

def compare_patches(reference,query,patches,query_rgb):
    if reference is None or query is None or not patches:
        return dict(score=0.,outline=0.,detail=0.,patch_count=len(patches))
    outline,angle=align_outline(reference,query)
    theta=math.radians(angle);c,s=math.cos(theta),math.sin(theta)
    image=gray(query_rgb);correlations=[]
    origin=query['center']+query['offset'];radius=query['radius']
    for record in patches:
        x=record['position'][0]+record['u'];y=record['position'][1]+record['v']
        xq=origin[0]+radius*(c*x-s*y);yq=origin[1]+radius*(s*x+c*y)
        values=[float(record['vector']@unit(patch(image,xq+dx,yq+dy))) for dx,dy in TRANSLATIONS]
        correlations.append(float(np.clip(max(values),-1,1)))
    error=float(np.mean([(1-value)/2 for value in correlations]))
    detail=math.exp(-8*error)
    return dict(score=outline**4*detail,outline=outline,detail=detail,angle=angle,
        patch_count=len(patches),patch_correlations=correlations)

def describe_gradient_region(rgb,box):
    """Extract RGB foreground/filled outline only; never enumerate interior holes."""
    item=_unit_mask(rgb,box)
    if item is None:return None
    item['polar']=_polar(item['filled'],item['center'],item['radius'])
    item['canonical']=_canonical(item['filled'],item['center'],item['radius'])
    return item

def reencode_fixed_patches(rgb,reference,patches):
    """Resample patch vectors at unchanged native-selected coordinates and sizes."""
    image=gray(rgb);origin=reference['center']+reference['offset'];radius=reference['radius']
    result=[]
    for record in patches:
        x=origin[0]+radius*(record['position'][0]+record['u'])
        y=origin[1]+radius*(record['position'][1]+record['v'])
        result.append(dict(record,vector=unit(patch(image,x,y))))
    return result

def make_reference(rgb,box):
    reference=describe_gradient_region(rgb,box)
    if reference is None:return None
    reference['patches']=reference_patches(rgb,reference)
    reference['patches_by_stride']={1:reference['patches']}
    reference['reference_rgb']=np.asarray(rgb).copy()
    return reference

def compare(reference,query_description,query_rgb,fine_stride=1):
    # Validate even when a reference has no resolved region.
    evidence_rgb=detail_rgb(query_rgb,fine_stride)
    if reference is None:
        return compare_patches(None,query_description,[],evidence_rgb)
    if fine_stride not in reference['patches_by_stride']:
        reference['patches_by_stride'][fine_stride]=reencode_fixed_patches(
            detail_rgb(reference['reference_rgb'],fine_stride),reference,reference['patches'])
    return compare_patches(reference,query_description,reference['patches_by_stride'][fine_stride],evidence_rgb)

def match_gradient(reference_rgb,reference_box,query_rgb,proposals,fine_stride=1):
    """One enrollment RGB+box, query RGB, and CNN-predicted proposal boxes."""
    reference=make_reference(reference_rgb,reference_box)
    evidence=[compare(reference,describe_gradient_region(query_rgb,box),query_rgb,fine_stride)
        for box in proposals]
    scores=[item['score'] for item in evidence]
    best=int(np.argmax(scores)) if scores else None
    return dict(score=scores[best] if best is not None else 0.,
        bbox_xyxy=list(proposals[best]) if best is not None else None,
        proposals=[list(box) for box in proposals],proposal_scores=scores,
        evidence=evidence[best] if best is not None else {},all_evidence=evidence)
