"""Frozen neural baselines and a reference-trained patch matcher.

These are new adapters, not reproductions of a published one-shot recognizer.
No constructor downloads weights unless ``allow_download=True`` is explicit.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from typing import Any
import math
import numpy as np
from PIL import Image

from ..types import MatchResult, rgb_uint8

RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def image_digest(image: np.ndarray) -> str:
    image = rgb_uint8(image)
    return sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def _tensor(image: np.ndarray, size: int, device: str, normalize: bool = True):
    import torch
    image = rgb_uint8(image)
    resized = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    if normalize:
        mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
        std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
        tensor = (tensor - mean) / std
    return tensor


def _cached(cache: OrderedDict, key: str, value=None, max_items: int = 256):
    if value is None:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)
    return value


def _local_result(ref_features: np.ndarray, query_features: np.ndarray,
                  ref_points: np.ndarray, query_points: np.ndarray,
                  evidence: dict[str, Any], keep_fraction: float = .35) -> MatchResult:
    """Reciprocal descriptor matches, with a visible-subset score.

    Correspondences are appearance proposals only. No homography is inferred here.
    Scores are similarities and explicitly are not probabilities.
    """
    if len(ref_features) == 0 or len(query_features) == 0:
        return MatchResult(0., {**evidence, "matches": 0})
    similarity = np.clip(ref_features @ query_features.T, -1., 1.)
    forward = similarity.argmax(axis=1)
    backward = similarity.argmax(axis=0)
    ref_indices = np.flatnonzero(backward[forward] == np.arange(len(ref_features)))
    query_indices = forward[ref_indices]
    values = similarity[ref_indices, query_indices]
    count = len(values)
    # Require several mutually consistent appearance proposals. This is a
    # heuristic, whose rejection threshold is calibrated on development objects.
    keep = max(4, math.ceil(len(ref_features) * keep_fraction))
    order = np.argsort(-values, kind="stable")[:keep]
    top_values = np.maximum(values[order], 0.)
    support = min(1., count / keep)
    score = float(top_values.mean() * math.sqrt(support)) if count else 0.
    return MatchResult(
        score=score,
        evidence={**evidence, "matches": int(count), "retained_matches": len(order),
                  "mutual_fraction": float(count / len(ref_features)),
                  "visible_subset_fraction": keep_fraction,
                  "score_is_probability": False,
                  "geometry_verified": False},
        points_reference=ref_points[ref_indices[order]].astype(float).tolist(),
        points_query=query_points[query_indices[order]].astype(float).tolist(),
    )


class FrozenCNNMatcher:
    """ImageNet ResNet18 global or local-feature matching baseline.

    Images are resized as full frames to a square (no crop). That preserves
    coverage for matching but differs from ImageNet classification preprocessing.
    """
    def __init__(self, mode: str = "global", weights_path: str | Path | None = None,
                 allow_download: bool = False, device: str = "cpu", image_size: int = 224,
                 grid_size: int = 7, keep_fraction: float = .35, cache_size: int = 256):
        if mode not in {"global", "local"}:
            raise ValueError("mode must be global or local")
        if image_size < 32 or grid_size < 2 or not 0 < keep_fraction <= 1 or cache_size < 1:
            raise ValueError("Invalid image size, grid size, keep fraction or cache size")
        try:
            import torch
            from torchvision.models import resnet18
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("FrozenCNNMatcher requires a compatible torch + torchvision installation") from exc
        self.torch = torch
        self.device = str(device)
        self.mode = mode
        self.image_size = int(image_size)
        self.grid_size = int(grid_size)
        self.keep_fraction = float(keep_fraction)
        self.cache_size = int(cache_size)
        self.name = "cnn_global" if mode == "global" else "cnn_local"
        candidate = Path(weights_path) if weights_path is not None else Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
        if not candidate.is_file():
            if weights_path is not None:
                raise FileNotFoundError(f"ResNet18 checkpoint not found: {candidate}")
            if not allow_download:
                raise FileNotFoundError("ResNet18 weights are absent. Pass weights_path or explicitly set allow_download=True.")
            state = torch.hub.load_state_dict_from_url(RESNET18_URL, map_location="cpu", progress=False, check_hash=True, weights_only=True)
        else:
            state = torch.load(candidate, map_location="cpu", weights_only=True)
        self.checkpoint_sha256 = sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else None
        self.checkpoint_path = str(candidate.resolve())
        self.is_official_checkpoint = bool(self.checkpoint_sha256 and self.checkpoint_sha256.startswith("f37072fd"))
        model = resnet18(weights=None)
        model.load_state_dict(state, strict=True)
        self.encoder = torch.nn.Sequential(*list(model.children())[:-2]).eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def provenance(self) -> dict[str, Any]:
        return {
            "method": self.name, "architecture": "ResNet18",
            "weight_source": RESNET18_URL if self.is_official_checkpoint else "user-provided checkpoint; prior training unverified",
            "checkpoint_sha256": self.checkpoint_sha256,
            "resolved_checkpoint_path": self.checkpoint_path,
            "external_pretraining": "ImageNet-1K supervised" if self.is_official_checkpoint else "unknown",
            "external_training_images": 1281167 if self.is_official_checkpoint else None,
            "target_reference_images": 1, "target_weight_updates": 0,
            "image_size": self.image_size, "grid_size": self.grid_size if self.mode == "local" else None,
            "preprocessing": "Full-frame square resize, ImageNet mean/std; no center crop",
            "score": "cosine similarity" if self.mode == "global" else "partial reciprocal local cosine similarity",
            "score_is_probability": False,
            "source": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html",
            "limitation": "Pretrained representation; not one-image total training. Patch proposals have no geometric verification.",
        }

    def _features(self, image: np.ndarray):
        key = image_digest(image)
        cached = _cached(self.cache, key)
        if cached is not None:
            return cached
        torch = self.torch
        with torch.inference_mode():
            features = self.encoder(_tensor(image, self.image_size, self.device))
            if self.mode == "global":
                array = torch.nn.functional.normalize(features.mean(dim=(2, 3)), dim=1)[0].cpu().numpy()
                result = (array, None)
            else:
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
                array = torch.nn.functional.normalize(pooled.flatten(2).transpose(1, 2), dim=2)[0].cpu().numpy()
                y, x = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing="ij")
                points = np.column_stack([(x.ravel() + .5) * image.shape[1] / self.grid_size,
                                          (y.ravel() + .5) * image.shape[0] / self.grid_size])
                result = (array, points)
        return _cached(self.cache, key, result, self.cache_size)

    def predict(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        ref_features, ref_points = self._features(reference)
        query_features, query_points = self._features(query)
        if self.mode == "global":
            return MatchResult(float(np.clip(ref_features @ query_features, -1., 1.)),
                               {"descriptor_dimensions": len(ref_features), "score_is_probability": False,
                                "geometry_verified": False, "external_pretraining": True})
        return _local_result(ref_features, query_features, ref_points, query_points,
                             {"reference_descriptors": len(ref_features), "query_descriptors": len(query_features),
                              "external_pretraining": True}, self.keep_fraction)


class TinyPatchMatcher:
    """Use a CNN trained only on patches of exactly one reference image.

    A checkpoint is reference-bound: passing another reference is an error.
    This avoids accidentally using a test object's model on unrelated objects.
    """
    name = "tiny_patch_cnn"

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", image_size: int | None = None,
                 stride: int = 16, patch_size: int | None = None, keep_fraction: float = .35):
        import torch
        from ..training import TinyPatchCNN
        if stride < 1 or not 0 < keep_fraction <= 1:
            raise ValueError("stride must be positive and keep_fraction must be in (0, 1]")
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "hypernet.single_reference_patch_cnn.v1":
            raise ValueError("Unsupported TinyPatchCNN checkpoint format")
        self.metadata = checkpoint["metadata"]
        self.image_size = int(image_size or self.metadata["image_size"])
        self.patch_size = int(patch_size or self.metadata["patch_size"])
        if self.patch_size != self.metadata["patch_size"]:
            raise ValueError("Inference patch size must match the training checkpoint")
        if self.image_size < self.patch_size:
            raise ValueError("image_size must be at least patch_size")
        self.stride, self.keep_fraction = int(stride), float(keep_fraction)
        self.device, self.torch = str(device), torch
        self.model = TinyPatchCNN(self.metadata["num_patch_classes"], self.metadata["embedding_dim"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.checkpoint_sha256 = sha256(checkpoint_path.read_bytes()).hexdigest()
        self.checkpoint_path = str(checkpoint_path.resolve())
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def provenance(self):
        return {**self.metadata, "method": self.name, "checkpoint_sha256": self.checkpoint_sha256,
                "resolved_checkpoint_path": self.checkpoint_path,
                "score_is_probability": False,
                "limitation": "Reference patch discrimination is not proof of instance recognition or unseen-view generalization."}

    def _features(self, image):
        from ..training import patch_grid
        key = image_digest(image)
        cached = _cached(self.cache, key)
        if cached is not None:
            return cached
        torch = self.torch
        image_tensor = _tensor(image, self.image_size, self.device, normalize=False)
        patches, points = patch_grid(image_tensor, self.patch_size, self.stride)
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(patches), 128):
                embeddings.append(self.model.embed(patches[start:start + 128]).cpu())
        array = torch.cat(embeddings).numpy()
        points = points.cpu().numpy()
        points[:, 0] *= image.shape[1] / self.image_size
        points[:, 1] *= image.shape[0] / self.image_size
        return _cached(self.cache, key, (array, points), 256)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        if image_digest(reference) != self.metadata["reference_rgb_sha256"]:
            raise ValueError("Checkpoint was trained on a different reference image")
        a, pa = self._features(reference)
        b, pb = self._features(query)
        return _local_result(a, b, pa, pb,
                             {"external_pretraining": False, "training_images": 1,
                              "reference_descriptors": len(a), "query_descriptors": len(b)}, self.keep_fraction)
