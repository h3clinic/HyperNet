"""Reference-conditioned CNN descriptor normalization and verified partial matching.

The CNN checkpoint is unchanged. Whitening is fitted to one reference's patch
descriptors, then applied unchanged to each query. This is not population
covariance estimation and introduces no extra visual observations.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np

from .cnn import TinyPatchMatcher, image_digest
from ..types import MatchResult, rgb_uint8
from ..geometry import estimate_homography, spatial_coverage


def fit_reference_whitener(features, shrinkage=.1, eigen_floor=1e-6):
    """Return reference mean, regularized symmetric inverse square root, diagnostics."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("Need at least two reference descriptors")
    if not np.isfinite(features).all():
        raise ValueError("Reference descriptors must be finite")
    if not 0 < shrinkage <= 1 or eigen_floor <= 0:
        raise ValueError("shrinkage must be in (0,1] and eigen_floor positive")
    mean = features.mean(axis=0)
    centered = features-mean
    covariance = centered.T@centered/max(1,len(features)-1)
    average_variance = float(np.trace(covariance)/features.shape[1])
    regularized = (1-shrinkage)*covariance + shrinkage*average_variance*np.eye(features.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    floor = max(float(eigen_floor), average_variance*1e-4)
    eigenvalues = np.maximum(eigenvalues,floor)
    transform = (eigenvectors/np.sqrt(eigenvalues)[None,:])@eigenvectors.T
    return mean, transform, {
        "shrinkage": float(shrinkage), "reference_descriptor_count": len(features),
        "descriptor_dimension": features.shape[1], "mean_variance": average_variance,
        "eigenvalue_floor": floor, "regularized_condition_number": float(eigenvalues.max()/eigenvalues.min()),
        "fit_observations": "patch descriptors from the target reference only",
    }


def apply_reference_whitener(features, mean, transform):
    features = np.asarray(features,dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(mean) or not np.isfinite(features).all():
        raise ValueError("Query/reference descriptors must be finite and dimension-compatible")
    projected = (features-mean)@transform
    norms = np.linalg.norm(projected,axis=1,keepdims=True)
    return np.divide(projected,norms,out=np.zeros_like(projected),where=norms>1e-10)


def distinctive_reciprocal_matches(reference, query, ratio=.85, min_similarity=.15):
    """Ratio test on Euclidean distances of unit vectors, applied in both directions."""
    a,b=np.asarray(reference,float),np.asarray(query,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:
        raise ValueError("Descriptor arrays must be two dimensional with equal width")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Descriptors must be finite")
    if not 0<ratio<1 or not -1<=min_similarity<=1:
        raise ValueError("Invalid match thresholds")
    empty=(np.empty(0,int),np.empty(0,int),np.empty(0,float),np.empty(0,float))
    if len(a)<2 or len(b)<2:return empty
    similarity=np.clip(a@b.T,-1.,1.)
    squared=np.maximum(2.-2.*similarity,0.)
    forward=np.argmin(squared,axis=1); backward=np.argmin(squared,axis=0)
    first=squared[np.arange(len(a)),forward]
    second=np.partition(squared,1,axis=1)[:,1]
    reverse_second=np.partition(squared,1,axis=0)[1,:]
    mutual=backward[forward]==np.arange(len(a))
    # Repeated identical descriptors fail the strict 0 < ratio*0 test.
    valid=mutual & (first<ratio**2*second) & (first<ratio**2*reverse_second[forward])
    valid &= similarity[np.arange(len(a)),forward]>=min_similarity
    valid &= (np.linalg.norm(a,axis=1)>1e-8) & (np.linalg.norm(b[forward],axis=1)>1e-8)
    ri=np.flatnonzero(valid);qi=forward[ri]
    ratios=np.sqrt(first[ri]/np.maximum(second[ri],1e-12))
    return ri,qi,similarity[ri,qi],np.clip(1.-ratios,0.,1.)


class WhitenedTinyMatcher:
    """Reference-condition existing v1 or supplied reference-bound descriptors."""
    def __init__(self, checkpoint_path: str|Path|None=None, device="cpu", geometry=True,
                 shrinkage=.1, ratio=.85, min_similarity=.15, min_inliers=6,
                 reprojection_threshold=10., keep_fraction=.35, stride=16,
                 base_matcher=None, boundary_centers=True):
        if not 0<keep_fraction<=1 or min_inliers<4 or reprojection_threshold<=0:
            raise ValueError("Invalid partial support or geometry settings")
        if not 0<ratio<1 or not 0<shrinkage<=1 or not -1<=min_similarity<=1:
            raise ValueError("Invalid whitening or match settings")
        if checkpoint_path is None and base_matcher is None:
            raise ValueError("Supply a checkpoint_path or a reference-bound base_matcher")
        self.base=base_matcher if base_matcher is not None else TinyPatchMatcher(checkpoint_path,device=device,stride=stride)
        self.center_offset=.5 if boundary_centers else 0.
        self.geometry=bool(geometry);self.shrinkage=float(shrinkage)
        self.ratio=float(ratio);self.min_similarity=float(min_similarity)
        self.min_inliers=int(min_inliers);self.reprojection_threshold=float(reprojection_threshold)
        self.keep_fraction=float(keep_fraction)
        self.name="tiny_whitened_geometry" if self.geometry else "tiny_whitened"
        self._reference=None

    def provenance(self):
        base=self.base.provenance()
        return {**base,"method":self.name,"base_method":base.get("method"),
                "checkpoint_retrained":False,"extra_visual_observations":0,
                "normalization":"Reference mean centering + shrunk-covariance ZCA + per-descriptor L2 normalization",
                "normalization_fit":"Reference patches only; no query or development/test labels",
                "shrinkage":self.shrinkage,"bidirectional_ratio":self.ratio,
                "min_similarity":self.min_similarity,"geometry":self.geometry,
                "reprojection_threshold_px":self.reprojection_threshold,
                "min_inliers":self.min_inliers,"keep_fraction":self.keep_fraction,
                "coordinate_center_offset_subtracted":self.center_offset,
                "source_inspiration":["https://scikit-learn.org/stable/modules/covariance.html",
                                      "https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf"],
                "limitation":"Reference-conditioned scores require calibration. Dense patch centers give coarse geometry; no new invariant features or unseen views are learned."}

    def predict(self,reference,query):
        reference,query=rgb_uint8(reference),rgb_uint8(query)
        if image_digest(reference)!=self.base.metadata["reference_rgb_sha256"]:
            raise ValueError("Checkpoint was trained on a different reference image")
        if self.base.metadata.get("insufficient_informative_patches",False):
            return MatchResult(0.,{"reason":"insufficient reference texture for descriptor learning",
                                  "training_images":1,"external_pretraining":False,
                                  "checkpoint_retrained":False,"score_is_probability":False})
        if self._reference is None:
            features,points=self.base._features(reference)
            mean,transform,diagnostics=fit_reference_whitener(features,self.shrinkage)
            normalized=apply_reference_whitener(features,mean,transform)
            self._reference=(normalized,np.asarray(points,float)-self.center_offset,mean,transform,diagnostics)
        a,pa,mean,transform,diagnostics=self._reference
        features,points=self.base._features(query)
        b=apply_reference_whitener(features,mean,transform);pb=np.asarray(points,float)-self.center_offset
        ri,qi,similarities,margins=distinctive_reciprocal_matches(a,b,self.ratio,self.min_similarity)
        p0,p1=pa[ri],pb[qi]
        geometry=None
        if self.geometry:
            geometry=estimate_homography(p0,p1,reference.shape,query.shape,
                                        self.reprojection_threshold,self.min_inliers)
            selected=geometry.inliers
        else:selected=np.ones(len(p0),bool)
        n=int(selected.sum())
        keep=max(self.min_inliers,math.ceil(len(a)*self.keep_fraction))
        quality=np.maximum(similarities[selected],0.)*(.5+.5*margins[selected])
        order=np.argsort(-quality,kind="stable")[:keep]
        support=min(1.,n/keep)
        fraction=n/max(1,len(p0))
        score=float(quality[order].mean()*math.sqrt(support)*(math.sqrt(fraction) if self.geometry else 1.)) if n>=self.min_inliers else 0.
        return MatchResult(
            score=score,
            evidence={"training_images":1,"external_pretraining":False,"checkpoint_retrained":False,
                      "reference_descriptors":len(a),"query_descriptors":len(b),
                      "distinctive_mutual_matches":len(p0),"inlier_count":n,"support":support,
                      "inlier_fraction":fraction,"geometry_valid":geometry.valid if geometry else None,
                      "geometry_reason":geometry.reason if geometry else "disabled",
                      "reference_coverage":spatial_coverage(p0[selected],reference.shape),
                      "normalization":diagnostics,"score_is_probability":False},
            points_reference=p0.tolist(),points_query=p1.tolist(),inliers=selected.tolist(),
            homography=geometry.homography.tolist() if geometry and geometry.valid else None,
            bbox_xyxy=geometry.bbox_xyxy if geometry else None)

