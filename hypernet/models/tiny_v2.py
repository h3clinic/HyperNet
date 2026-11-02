"""Reference-bound matcher for the contrastive Tiny CNN v2 experiment."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from collections import OrderedDict
from ..types import MatchResult, rgb_uint8
from .cnn import TinyPatchMatcher, image_digest

class TinyV2Matcher(TinyPatchMatcher):
    name = "tiny_descriptor_cnn_v2"
    feature_coordinate_convention = "zero_based_pixel_centers"

    def __init__(self,checkpoint_path,device="cpu",image_size=None,stride=16,patch_size=None,keep_fraction=.35):
        import torch
        from ..training_v2 import TinyDescriptorCNN, FORMAT
        if stride<1 or not 0<keep_fraction<=1:
            raise ValueError("stride must be positive and keep_fraction in (0,1]")
        path=Path(checkpoint_path)
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        if checkpoint.get("format")!=FORMAT:
            raise ValueError("Unsupported TinyDescriptorCNN v2 checkpoint format")
        self.metadata=checkpoint["metadata"]
        self.image_size=int(image_size or self.metadata["image_size"])
        self.patch_size=int(patch_size or self.metadata["patch_size"])
        if self.patch_size!=self.metadata["patch_size"] or self.image_size<self.patch_size:
            raise ValueError("Invalid inference patch or image size")
        self.stride,self.keep_fraction=int(stride),float(keep_fraction)
        self.device,self.torch=str(device),torch
        self.model=TinyDescriptorCNN(self.metadata["embedding_dim"])
        self.model.load_state_dict(checkpoint["state_dict"],strict=True)
        self.model.eval().to(self.device)
        self.checkpoint_sha256=sha256(path.read_bytes()).hexdigest()
        self.checkpoint_path=str(path.resolve())
        self.cache=OrderedDict()

    def _features(self,image):
        descriptors,points=super()._features(image)
        # v1 patch_grid produces centers in pixel-boundary coordinates. Its
        # resize multiplication preserves that boundary convention; subtracting
        # .5 converts the final points to OpenCV's zero-based pixel centers.
        return descriptors,points-.5

    def predict(self,reference,query):
        reference,query=rgb_uint8(reference),rgb_uint8(query)
        if image_digest(reference)!=self.metadata["reference_rgb_sha256"]:
            raise ValueError("Checkpoint was trained on a different reference image")
        if self.metadata["insufficient_informative_patches"]:
            return MatchResult(0.,{"reason":"insufficient reference texture for descriptor learning",
                                    "training_images":1,"external_pretraining":False,"score_is_probability":False})
        result=super().predict(reference,query)
        result.evidence["training_revision"]=2
        return result

    def provenance(self):
        return {**super().provenance(),"method":self.name,
                "feature_coordinate_convention":self.feature_coordinate_convention,
                "training_revision":2,
                "limitation":"Reference-only contrastive patch descriptors; object identity and unseen-view generalization remain unproven."}

