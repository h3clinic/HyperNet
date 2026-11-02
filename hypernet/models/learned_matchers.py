"""Optional published-model adapters, with explicit weight acquisition."""
from __future__ import annotations
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import math
import numpy as np
from PIL import Image

from ..types import MatchResult, rgb_uint8
from ..geometry import estimate_homography, spatial_coverage
from .cnn import image_digest, _cached

LOFTR_URL = "https://huggingface.co/kornia/loftr/resolve/main/loftr_outdoor.ckpt"


def restore_resize_coordinates(points, scale_xy):
    """Undo PIL resize for zero-based pixel-center coordinates.

    The installed Kornia 0.9 LoFTR implementation emits stride*grid-index
    locations and learned offsets without a half-pixel shift. PIL resize maps
    a resized center x to (x + .5) * (original_width / resized_width) - .5.
    In particular, an identity resize must leave coordinates unchanged.
    """
    points = np.asarray(points, dtype=np.float64)
    scale_xy = np.asarray(scale_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or scale_xy.shape != (2,):
        raise ValueError("Expected N x 2 points and two resize scale factors")
    if not np.isfinite(points).all() or not np.isfinite(scale_xy).all() or np.any(scale_xy <= 0):
        raise ValueError("Coordinates must be finite and resize scales positive")
    return (points + .5) * scale_xy - .5


class LoFTRMatcher:
    """Kornia's published outdoor LoFTR + guarded planar verification.

    The correspondences come from the published model. The instance score and
    acceptance calibration are this workbench's wrapper, not a published metric.
    """
    name = "loftr"

    def __init__(self, checkpoint_path: str | Path | None = None, allow_download: bool = False,
                 device: str = "cpu", max_size: int = 480, confidence_threshold: float = .2,
                 reprojection_threshold: float = 4., max_matches: int = 2000):
        if max_size < 32 or not 0 <= confidence_threshold < 1 or max_matches < 6:
            raise ValueError("Invalid LoFTR dimensions, confidence threshold or match limit")
        try:
            import torch
            import kornia
            from kornia.feature import LoFTR
        except ImportError as exc:
            raise RuntimeError("LoFTRMatcher requires torch and kornia; install the learned extras") from exc
        self.device, self.torch = str(device), torch
        self.max_size = int(max_size)
        self.confidence_threshold = float(confidence_threshold)
        self.reprojection_threshold = float(reprojection_threshold)
        self.max_matches = int(max_matches)
        self.kornia_version = kornia.__version__
        candidate = Path(checkpoint_path) if checkpoint_path else Path(torch.hub.get_dir()) / "checkpoints/loftr_outdoor.ckpt"
        if not candidate.is_file():
            if checkpoint_path is not None:
                raise FileNotFoundError(f"LoFTR checkpoint not found: {candidate}")
            if not allow_download:
                raise FileNotFoundError("LoFTR weights absent. Pass checkpoint_path or explicitly set allow_download=True.")
            checkpoint = torch.hub.load_state_dict_from_url(LOFTR_URL, map_location="cpu", progress=False, weights_only=True)
        else:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        self.model = LoFTR(pretrained=None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.checkpoint_sha256 = sha256(candidate.read_bytes()).hexdigest() if candidate.exists() else None
        self.checkpoint_path = str(candidate.resolve())
        self.weight_origin = "explicit local checkpoint" if checkpoint_path else "official outdoor URL / torch hub cache"
        self.cache: OrderedDict = OrderedDict()

    def provenance(self):
        return {"method": self.name, "implementation": "Kornia LoFTR outdoor",
                "kornia_version": self.kornia_version,
                "checkpoint_sha256": self.checkpoint_sha256,
                "resolved_checkpoint_path": self.checkpoint_path, "weight_origin": self.weight_origin,
                "weight_source": LOFTR_URL,
                "external_pretraining": "MegaDepth for published outdoor checkpoint; verify provenance of custom checkpoints",
                "external_training_images": None,
                "target_reference_images": 1, "target_weight_updates": 0,
                "max_size": self.max_size, "confidence_threshold": self.confidence_threshold,
                "coordinate_mapping": "PIL inverse resize: (keypoint + 0.5) * original/resized - 0.5",
                "coordinate_mapping_revision": 2,
                "score_is_probability": False,
                "source": "https://arxiv.org/abs/2104.00680",
                "limitation": "Published correspondence model with custom planar-verification score; not one-image total training."}

    def _image(self, image):
        key = image_digest(image)
        existing = _cached(self.cache, key)
        if existing is not None:
            return existing
        h, w = image.shape[:2]
        ratio = min(1., self.max_size / max(h, w))
        nh, nw = max(32, int(h * ratio) // 8 * 8), max(32, int(w * ratio) // 8 * 8)
        resized = Image.fromarray(image).convert("L").resize((nw, nh), Image.Resampling.BILINEAR)
        tensor = self.torch.from_numpy(np.asarray(resized).copy()).float()[None, None].div(255).to(self.device)
        return _cached(self.cache, key, (tensor, np.array([w / nw, h / nh])), 64)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        a, scale_a = self._image(reference)
        b, scale_b = self._image(query)
        with self.torch.inference_mode():
            output = self.model({"image0": a, "image1": b})
        confidence = output["confidence"].cpu().numpy()
        valid = np.isfinite(confidence) & (confidence >= self.confidence_threshold)
        order = np.flatnonzero(valid)
        order = order[np.argsort(-confidence[order], kind="stable")[:self.max_matches]]
        confidence = confidence[order]
        # Kornia's executable output path applies no half-pixel offset. Undo
        # our PIL resize using pixel-center alignment, including identity scale.
        pa = restore_resize_coordinates(output["keypoints0"].cpu().numpy()[order], scale_a)
        pb = restore_resize_coordinates(output["keypoints1"].cpu().numpy()[order], scale_b)
        geometry = estimate_homography(pa, pb, reference.shape, query.shape,
                                       reprojection_threshold=self.reprojection_threshold)
        inliers = geometry.inliers
        n = int(inliers.sum()) if geometry.valid else 0
        ratio = float(n / max(1, len(pa)))
        coverage = spatial_coverage(pa[inliers], reference.shape) if n else 0.
        mean_confidence = float(confidence[inliers].mean()) if n else 0.
        score = float((1. - math.exp(-n / 20.)) * math.sqrt(ratio) * (.5 + .5 * coverage) * mean_confidence) if n else 0.
        return MatchResult(
            score,
            {"matches": len(pa), "verified_inliers": n, "inlier_ratio": ratio,
             "reference_coverage": coverage, "mean_inlier_confidence": mean_confidence,
             "geometry_verified": geometry.valid, "geometry_reason": geometry.reason,
             "score_is_probability": False, "wrapper_score": True},
            pa.astype(float).tolist(), pb.astype(float).tolist(),
            inliers.tolist(),
            geometry.homography.tolist() if geometry.homography is not None else None,
            geometry.bbox_xyxy,
        )
