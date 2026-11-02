"""Reference-bound matching with the scratch multiscale shape CNN."""
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import torch

from ..types import MatchResult, rgb_uint8
from ..training_v3 import FORMAT, TinyShapeCNN, multiscale_patch_grid
from .cnn import _cached, _local_result, _tensor, image_digest


class TinyV3Matcher:
    name = "tiny_shape_cnn_v3"

    def __init__(self, checkpoint_path, device="cpu", image_size=None, keep_fraction=.35):
        if not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != FORMAT:
            raise ValueError("Unsupported TinyShapeCNN checkpoint format")
        self.metadata = checkpoint["metadata"]
        self.image_size = int(image_size or self.metadata["image_size"])
        self.patch_size = int(self.metadata["patch_size"])
        self.scale_factors = tuple(self.metadata["scale_factors"])
        if self.image_size < self.patch_size * max(self.scale_factors):
            raise ValueError("image_size must accommodate every trained slice scale")
        self.keep_fraction, self.device = float(keep_fraction), str(device)
        self.model = TinyShapeCNN(self.metadata["embedding_dim"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)
        self.checkpoint_sha256 = sha256(checkpoint_path.read_bytes()).hexdigest()
        self.checkpoint_path = str(checkpoint_path.resolve())
        self.cache = OrderedDict()

    def provenance(self):
        return {**self.metadata, "method": self.name, "checkpoint_sha256": self.checkpoint_sha256,
                "resolved_checkpoint_path": self.checkpoint_path, "keep_fraction": self.keep_fraction,
                "inference_image_size": self.image_size, "geometry_verified": False,
                "score_is_probability": False,
                "feature_coordinates": "zero-based pixel centers",
                "matching": "partial reciprocal cosine matching across multiscale CNN slices"}

    def _features(self, image):
        image = rgb_uint8(image)
        key = image_digest(image)
        cached = _cached(self.cache, key)
        if cached is not None:
            return cached
        with torch.inference_mode():
            tensor = _tensor(image, self.image_size, self.device, normalize=False)
            patches, points, _ = multiscale_patch_grid(tensor, self.patch_size, self.scale_factors)
            embedding = torch.cat([self.model.embed(patches[start:start + 128]).cpu()
                                   for start in range(0, len(patches), 128)]).numpy()
            points = points.cpu().numpy()
        points[:, 0] = points[:, 0] * image.shape[1] / self.image_size - .5
        points[:, 1] = points[:, 1] * image.shape[0] / self.image_size - .5
        return _cached(self.cache, key, (embedding, points), 256)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        if image_digest(reference) != self.metadata["reference_rgb_sha256"]:
            raise ValueError("Checkpoint was trained on a different reference image")
        if self.metadata.get("insufficient_informative_patches"):
            return MatchResult(0., {"reason": "insufficient_informative_reference_patches",
                                   "external_pretraining": False, "training_images": 1,
                                   "training_revision": 3, "score_is_probability": False,
                                   "geometry_verified": False})
        a, pa = self._features(reference)
        b, pb = self._features(query)
        return _local_result(a, b, pa, pb,
                             {"external_pretraining": False, "training_images": 1,
                              "training_revision": 3, "reference_descriptors": len(a),
                              "query_descriptors": len(b), "slice_scale_factors": self.scale_factors},
                             self.keep_fraction)

