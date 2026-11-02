"""Training-free affine alignment with reserved reference cells.

This is a deterministic local registration heuristic, not a semantic recognizer.
Fit and validation pixels are spatial subsets of the SAME image pair, not an
independent statistical validation set. Only fit cells choose the transformation.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from ..types import MatchResult, rgb_uint8
from ..geometry import project_points

def _cv2():
    import cv2
    return cv2

def _ncc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a=a-a.mean(); b=b-b.mean()
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(a@b/denom,-1,1)) if denom>1e-9 else 0.

def _normalized_matrix(shape):
    h,w=shape[:2]
    return np.array([[2/max(w-1,1),0,-1],[0,2/max(h-1,1),-1],[0,0,1]],float)

def _gray(image,max_size):
    cv2=_cv2();h,w=image.shape[:2]
    scale=min(1.,max_size/max(h,w))
    resized=cv2.resize(image,(max(16,int(round(w*scale))),max(16,int(round(h*scale)))),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(resized,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    return gray

def _masks(shape, cells=6, margin=2):
    h,w=shape
    y,x=np.mgrid[:h,:w]
    cx=np.minimum(x*cells//w,cells-1);cy=np.minimum(y*cells//h,cells-1)
    # Two-pixel guard around checkerboard boundaries prevents the validation
    # gradient stencil from including neighboring fit-cell pixels.
    edges=np.zeros(shape,bool)
    for v in np.linspace(0,w,cells+1).astype(int):
        edges |= np.abs(x-v)<=margin
    for v in np.linspace(0,h,cells+1).astype(int):
        edges |= np.abs(y-v)<=margin
    fit=((cx+cy)%2==0)&~edges
    validation=((cx+cy)%2==1)&~edges
    return fit,validation,cy*cells+cx

class AlignmentMatcher:
    name="reserved_cell_alignment"
    def __init__(self,max_size=160,min_overlap=.45,min_validation_ncc=.78,
                 min_gradient_ncc=.50,max_iterations=35,motion="affine",cache_size=512):
        if max_size<48 or not 0<min_overlap<=1 or max_iterations<1:
            raise ValueError("Invalid alignment size, overlap or iterations")
        if motion!="affine":
            raise ValueError("Only the documented bounded affine model is supported")
        if not -1<=min_validation_ncc<=1 or not -1<=min_gradient_ncc<=1:
            raise ValueError("Correlation gates must lie in [-1,1]")
        self.max_size=int(max_size);self.min_overlap=float(min_overlap)
        self.min_validation_ncc=float(min_validation_ncc)
        self.min_gradient_ncc=float(min_gradient_ncc)
        self.max_iterations=int(max_iterations)
        if cache_size<0:
            raise ValueError("cache_size must be nonnegative")
        self.cache_size=int(cache_size)
        self.cache=OrderedDict()

    def provenance(self):
        return {"method":self.name,"target_reference_images":1,"external_training_images":0,
                "target_weight_updates":0,"model":"bounded affine normalized-intensity registration",
                "max_size":self.max_size,"fit":"fixed checkerboard reference cells",
                "verification":"reserved reference cells; intensity, gradient and local NCC",
                "min_overlap":self.min_overlap,"min_validation_ncc":self.min_validation_ncc,
                "min_gradient_ncc":self.min_gradient_ncc,"max_iterations":self.max_iterations,
                "pair_cache_size":self.cache_size,
                "score_is_probability":False,"paper_reproduction":False,
                "sources":["https://docs.opencv.org/4.x/dc/d6b/group__video__track.html",
                           "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html"],
                "limitations":["Same-pair spatial reserve, not independent validation data.",
                               "Bounded affine alignment cannot model unseen surfaces or large general perspective changes.",
                               "Similar smooth or repeating appearances can still match; rejection is heuristic.",
                               "No segmentation: background can provide alignment evidence."]}

    @staticmethod
    def _sample(query,points,params):
        transform=np.array([[params[0],params[1],params[2]],
                            [params[3],params[4],params[5]]],float)
        mapped=points@transform.T
        valid=np.all(np.abs(mapped)<=.985,axis=1)
        h,w=query.shape
        coords=np.stack(((mapped[:,1]+1)*(h-1)/2,(mapped[:,0]+1)*(w-1)/2))
        values=map_coordinates(query,coords,order=1,mode="constant",cval=0.,prefilter=False)
        return values,valid

    @staticmethod
    def _information(image):
        cv2=_cv2()
        dx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)/8
        dy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)/8
        gradients=np.stack([dx.ravel(),dy.ravel()],1)
        eig=np.linalg.eigvalsh(gradients.T@gradients/max(1,len(gradients)))
        return {"intensity_std":float(image.std()),
                "gradient_rms":float(np.sqrt(np.mean(dx*dx+dy*dy))),
                "gradient_2d_ratio":float(eig[0]/max(eig.sum(),1e-12))}

    def predict(self,reference,query):
        reference,query=rgb_uint8(reference),rgb_uint8(query)
        key=tuple(sha256(str(image.shape).encode()+image.tobytes()).hexdigest() for image in (reference,query))
        if key in self.cache:
            self.cache.move_to_end(key)
            return deepcopy(self.cache[key])
        result=self._predict(reference,query)
        if self.cache_size:
            self.cache[key]=deepcopy(result)
            while len(self.cache)>self.cache_size:
                self.cache.popitem(last=False)
        return result

    def _predict(self,reference,query):
        reference,query=rgb_uint8(reference),rgb_uint8(query)
        a,b=_gray(reference,self.max_size),_gray(query,self.max_size)
        info_a,info_b=self._information(a),self._information(b)
        evidence={"geometry_valid":False,"alignment_valid":False,
                  "reference_information":info_a,"query_information":info_b,
                  "score_is_probability":False,"fit_pixels_used_for_verification":False}
        def reject(reason,**extra):
            return MatchResult(0.,{**evidence,"reason":reason,**extra})
        if min(a.shape)<48 or min(b.shape)<48:
            return reject("insufficient_resolution")
        if any(info["intensity_std"]<.025 or info["gradient_rms"]<.003 or info["gradient_2d_ratio"]<.005 for info in (info_a,info_b)):
            return reject("insufficient_spatial_information")
        fit,validation,cells=_masks(a.shape)
        y,x=np.nonzero(fit)
        # Deterministic regular subsampling controls optimization cost.
        if len(x)>2400:
            take=np.linspace(0,len(x)-1,2400,dtype=int);x,y=x[take],y[take]
        points=np.column_stack([2*x/(a.shape[1]-1)-1,2*y/(a.shape[0]-1)-1,np.ones(len(x))])
        targets=a[y,x].astype(float)
        target_z=(targets-targets.mean())/max(targets.std(),.02)
        def objective(params):
            values,valid=self._sample(b,points,params)
            if valid.sum()<max(20,int(.35*len(valid))):
                return np.full(len(valid),3.)
            mean,std=float(values[valid].mean()),max(float(values[valid].std()),.02)
            residual=target_z-(values-mean)/std
            residual[~valid]=2.
            return residual
        def fit_score(params):
            values,valid=self._sample(b,points,params)
            if valid.mean()<self.min_overlap:return -2.
            return _ncc(targets[valid],values[valid])-.3*(1-valid.mean())
        starts=[]
        for degrees in (-45.,-30.,-15.,0.,15.,30.,45.):
            angle=math.radians(degrees)
            for scale in (.8,1.,1.2):
                for tx in (-.15,0.,.15):
                    for ty in (-.15,0.,.15):
                        starts.append(np.array([scale*math.cos(angle),-scale*math.sin(angle),tx,
                                                scale*math.sin(angle),scale*math.cos(angle),ty]))
        starts.sort(key=fit_score,reverse=True)
        candidates=[]
        # Multiple starts are chosen only by fit-cell evidence.
        for start in starts[:1] if fit_score(starts[0])>.995 else starts[:2]:
            candidates.append((fit_score(start),start))
            if fit_score(start)>.995:
                continue
            try:
                optimized=least_squares(objective,start,
                    bounds=([.40,-.85,-.65,-.85,.40,-.65],[1.65,.85,.65,.85,1.65,.65]),
                    max_nfev=self.max_iterations,diff_step=1e-3,loss="soft_l1",f_scale=.7,
                    ftol=2e-4,xtol=2e-4,gtol=2e-4)
                candidates.append((fit_score(optimized.x),optimized.x))
            except (ValueError,FloatingPointError):
                continue
        fit_ncc,params=max(candidates,key=lambda item:item[0])
        hn=np.array([[params[0],params[1],params[2]],
                     [params[3],params[4],params[5]],[0,0,1]],float)
        singular=np.linalg.svd(hn[:2,:2],compute_uv=False)
        if np.linalg.det(hn[:2,:2])<=0 or singular.min()<.35 or singular.max()>2.:
            return reject("implausible_affine",fit_score=float(fit_ncc))
        yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]
        fullpoints=np.column_stack([2*xx.ravel()/(a.shape[1]-1)-1,
                                    2*yy.ravel()/(a.shape[0]-1)-1,np.ones(a.size)])
        warped,valid=self._sample(b,fullpoints,params)
        warped=warped.reshape(a.shape).astype(np.float32);valid=valid.reshape(a.shape)
        holdout=validation&valid
        overlap=float(valid.mean())
        if overlap<self.min_overlap or holdout.sum()<150:
            return reject("insufficient_overlap",overlap=overlap,fit_score=float(fit_ncc))
        validation_ncc=_ncc(a[holdout],warped[holdout])
        cv2=_cv2()
        ga=np.stack([cv2.Sobel(a,cv2.CV_32F,1,0),cv2.Sobel(a,cv2.CV_32F,0,1)],-1)
        gb=np.stack([cv2.Sobel(warped,cv2.CV_32F,1,0),cv2.Sobel(warped,cv2.CV_32F,0,1)],-1)
        safe=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&holdout
        gradient_ncc=_ncc(ga[safe].ravel(),gb[safe].ravel()) if safe.sum()>100 else 0.
        local=[]
        for cell in np.unique(cells[holdout]):
            mask=holdout&(cells==cell)
            if mask.sum()>=25 and a[mask].std()>=.012 and warped[mask].std()>=.012:
                local.append(_ncc(a[mask],warped[mask]))
        good_cells=sum(value>=.55 for value in local)
        local_median=float(np.median(local)) if local else 0.
        verification={"fit_score":float(fit_ncc),"overlap":overlap,
                      "validation_ncc":validation_ncc,"gradient_ncc":gradient_ncc,
                      "validation_informative_cells":len(local),"validation_good_cells":good_cells,
                      "validation_local_median_ncc":local_median,
                      "fit_pixels":len(x),"reserved_pixels":int(holdout.sum())}
        if validation_ncc<self.min_validation_ncc or gradient_ncc<self.min_gradient_ncc or good_cells<3 or local_median<.55:
            return reject("reserved_cell_verification_failed",**verification)
        # Convert the normalized coordinate mapping to original OpenCV pixels.
        def resize_matrix(original,small):
            sx,sy=small[1]/original[1],small[0]/original[0]
            return np.array([[sx,0,.5*sx-.5],[0,sy,.5*sy-.5],[0,0,1]],float)
        sr,sq=resize_matrix(reference.shape,a.shape),resize_matrix(query.shape,b.shape)
        h=np.linalg.inv(sq)@np.linalg.inv(_normalized_matrix(b.shape))@hn@_normalized_matrix(a.shape)@sr
        corners=np.array([[0,0],[reference.shape[1]-1,0],
                          [reference.shape[1]-1,reference.shape[0]-1],[0,reference.shape[0]-1]],float)
        projected=project_points(corners,h)
        x0,y0=np.maximum(projected.min(axis=0),[0,0])
        x1,y1=np.minimum(projected.max(axis=0),[query.shape[1],query.shape[0]])
        if not np.isfinite(projected).all() or x1<=x0 or y1<=y0:
            return reject("no_query_overlap",**verification)
        score=float(np.clip((.4*validation_ncc+.35*gradient_ncc+.25*local_median)*math.sqrt(overlap),0,1))
        return MatchResult(score,{**evidence,**verification,"geometry_valid":True,"alignment_valid":True,
                           "reason":"reserved_cells_pass","correspondence_kind":"model-projected footprint, not independent keypoint matches"},
                           homography=h.tolist(),bbox_xyxy=[float(x0),float(y0),float(x1),float(y1)])

class EvidenceFallbackMatcher:
    """Use reserved-cell-checked alignment when primary geometry is absent."""
    name="geometry_with_alignment_fallback"
    calibration_mode="route_budget"
    def __init__(self,primary_model,fallback_model=None,fallback_weight=1.):
        if not math.isfinite(fallback_weight) or not 0<fallback_weight<=1:
            raise ValueError("fallback_weight must lie in (0,1]")
        self.primary_model=primary_model
        self.fallback_model=fallback_model or AlignmentMatcher()
        self.fallback_weight=float(fallback_weight)

    def prepare(self,reference):
        for child in (self.primary_model,self.fallback_model):
            if hasattr(child,"prepare"):
                child.prepare(reference)

    def predict(self,reference,query):
        primary=self.primary_model.predict(reference,query)
        if not math.isfinite(float(primary.score)):
            raise ValueError("Nonfinite primary score")
        geometry_valid=(primary.evidence.get("geometry_valid") is True or
                        primary.evidence.get("geometry_verified") is True)
        if geometry_valid and primary.score>0:
            result=deepcopy(primary)
            result.evidence={**result.evidence,"route":"primary","fallback_skipped":True,
                             "primary_score":float(primary.score),"alignment_score":None,
                             "route_score":float(primary.score),"fallback_weight":self.fallback_weight}
            return result
        fallback=self.fallback_model.predict(reference,query)
        if not math.isfinite(float(fallback.score)):
            raise ValueError("Nonfinite fallback score")
        accepted=fallback.evidence.get("alignment_valid") is True
        return MatchResult(float(self.fallback_weight*fallback.score) if accepted else 0.,
              {"route":"alignment_fallback" if accepted else "both_rejected",
               "geometry_valid":accepted,"alignment_valid":accepted,"primary":primary.evidence,
               "primary_score":float(primary.score),"alignment_score":float(fallback.score),
               "route_score":float(self.fallback_weight*fallback.score) if accepted else 0.,
               "alignment":fallback.evidence,"fallback_weight":self.fallback_weight,
               "score_is_probability":False},
              points_reference=fallback.points_reference,points_query=fallback.points_query,
              inliers=fallback.inliers,homography=fallback.homography if accepted else None,
              bbox_xyxy=fallback.bbox_xyxy if accepted else None)

    def provenance(self):
        children=[self.primary_model.provenance(),self.fallback_model.provenance()]
        return {"method":self.name,"target_reference_images":1,
                "target_weight_updates":"Inherited from primary; alignment performs no weight updates",
                "external_training_images":0 if all(c.get("external_training_images")==0 for c in children) else None,
                "fallback_weight":self.fallback_weight,
                "route":"Primary verified geometry; otherwise reserved-cell affine alignment",
                "components":children,
                "score_is_probability":False,
                "limitation":"Adds a second false-acceptance route; full combined dev calibration is required."}

