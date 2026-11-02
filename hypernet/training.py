"""Small, reference-only patch-discrimination experiment.

The only visual observations used are patches of the supplied reference image.
Synthetic jitter is recorded separately; no teacher or pretrained encoder is used.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .types import rgb_uint8


class TinyPatchCNN(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 64):
        super().__init__()
        if num_classes < 2 or embedding_dim < 4:
            raise ValueError("At least two patch classes and four embedding dimensions are required")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, image):
        return F.normalize(self.encoder(image), dim=1)

    def forward(self, image):
        embedding = self.embed(image)
        # Normalized embeddings make descriptor scale irrelevant. A fixed
        # logit multiplier preserves a useful classification gradient.
        return self.classifier(embedding) * 8., embedding


def patch_grid(image: torch.Tensor, patch_size: int = 32, stride: int = 16):
    """Extract full patches; return NCHW patches and N x 2 center coordinates."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("patch_grid expects one RGB BCHW image")
    if patch_size < 8 or stride < 1 or min(image.shape[-2:]) < patch_size:
        raise ValueError("Invalid patch size or stride")
    h, w = image.shape[-2:]
    patches = F.unfold(image, kernel_size=patch_size, stride=stride).transpose(1, 2)
    patches = patches.reshape(-1, 3, patch_size, patch_size)
    y, x = torch.meshgrid(torch.arange((h - patch_size) // stride + 1, device=image.device),
                          torch.arange((w - patch_size) // stride + 1, device=image.device), indexing="ij")
    points = torch.stack([x.flatten() * stride + patch_size / 2,
                          y.flatten() * stride + patch_size / 2], dim=1).float()
    return patches, points


def _augment(patches: torch.Tensor, generator: torch.Generator):
    batch = len(patches)
    def random(shape):
        return torch.rand(shape, generator=generator, device=patches.device)
    angle = (random((batch,)) - .5) * (40. * math.pi / 180.)
    scale = .85 + .3 * random((batch,))
    theta = torch.zeros(batch, 2, 3, device=patches.device)
    theta[:, 0, 0] = torch.cos(angle) * scale
    theta[:, 1, 1] = torch.cos(angle) * scale
    theta[:, 0, 1] = -torch.sin(angle) * scale
    theta[:, 1, 0] = torch.sin(angle) * scale
    theta[:, :, 2] = (random((batch, 2)) - .5) * .24
    grid = F.affine_grid(theta, patches.shape, align_corners=False)
    warped = F.grid_sample(patches, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    brightness = .75 + .5 * random((batch, 1, 1, 1))
    contrast = .75 + .5 * random((batch, 1, 1, 1))
    color = .9 + .2 * random((batch, 3, 1, 1))
    mean = warped.mean(dim=(2, 3), keepdim=True)
    warped = ((warped - mean) * contrast + mean) * brightness * color
    noise = torch.randn(warped.shape, generator=generator, device=patches.device) * .015
    return (warped + noise).clamp(0, 1)


def train_reference_cnn(reference: np.ndarray, checkpoint_path: str | Path,
                        steps: int = 120, seed: int = 17, batch_size: int = 32,
                        patch_size: int = 32, max_patches: int = 64,
                        device: str = "cpu", image_size: int = 256,
                        embedding_dim: int = 64, learning_rate: float = .002) -> dict:
    """Train from exactly one array and save a reference-bound safe-load checkpoint.

    Patch locations provide surrogate labels, not semantic parts or extra images.
    This function never sees queries, validation labels, or another reference.
    """
    from .models.cnn import image_digest
    reference = rgb_uint8(reference)
    if steps < 1 or batch_size < 2 or max_patches < 2 or learning_rate <= 0:
        raise ValueError("Positive steps/lr, batch_size>=2 and max_patches>=2 are required")
    if patch_size < 8 or image_size < patch_size * 2:
        raise ValueError("Image size must accommodate at least two patches per axis")
    device = str(device)
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError("A non-CPU device was requested but CUDA is unavailable")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resized = np.asarray(Image.fromarray(reference).resize((image_size, image_size), Image.Resampling.BILINEAR)).copy()
    image = torch.from_numpy(resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    patches, points = patch_grid(image, patch_size, max(4, patch_size // 2))
    # Evenly spread classes across the reference. Their location is a training
    # label only; the matcher later compares learned descriptors, not class IDs.
    selection = torch.linspace(0, len(patches) - 1, min(max_patches, len(patches)), device=device).round().long().unique()
    patches, points = patches[selection], points[selection]
    cuda_devices = [] if device == "cpu" else [torch.device(device).index or 0]
    start = time.perf_counter()
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model = TinyPatchCNN(len(patches), embedding_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        losses = []
        classification_losses = []
        model.train()
        for step in range(steps):
            labels = torch.randint(len(patches), (batch_size,), generator=generator, device=device)
            selected = patches[labels]
            view0, view1 = _augment(selected, generator), _augment(selected, generator)
            logits, embeddings = model(torch.cat([view0, view1], dim=0))
            classification = F.cross_entropy(logits, torch.cat([labels, labels]))
            consistency = (1. - (embeddings[:batch_size] * embeddings[batch_size:]).sum(1)).mean()
            loss = classification + .2 * consistency
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            classification_losses.append(float(classification.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            clean_logits, _ = model(patches)
            patch_accuracy = float((clean_logits.argmax(1) == torch.arange(len(patches), device=device)).float().mean().cpu())
        metadata = {
            "method": "tiny_patch_cnn", "architecture": f"3-convolution patch encoder + {embedding_dim}-dimensional embedding",
            "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "target_weight_updates": steps,
            "reference_rgb_sha256": image_digest(reference),
            "reference_original_shape": list(reference.shape), "image_size": image_size,
            "patch_size": patch_size, "num_patch_classes": len(patches), "embedding_dim": embedding_dim,
            "selected_patch_centers": points.detach().cpu().tolist(),
            "seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
            "synthetic_patch_views_seen": steps * batch_size * 2,
            "augmentations": "affine rotation +/-20 degrees, translation +/-12%, scale .85-1.15, brightness/contrast/color/noise",
            "objective": "patch-identity cross entropy + 0.2 paired-view cosine consistency",
            "training_loss": losses, "classification_loss": classification_losses,
            "clean_training_patch_accuracy": patch_accuracy,
            "elapsed_seconds": time.perf_counter() - start,
            "torch_version": str(torch.__version__), "device": device,
            "parameters": sum(p.numel() for p in model.parameters()),
            "score_is_probability": False,
            "source_inspiration": "https://arxiv.org/abs/1312.5242",
            "limitation": "Training accuracy concerns surrogate patch labels, not object recognition. No unseen view information is created.",
        }
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"format": "hypernet.single_reference_patch_cnn.v1", "metadata": metadata, "state_dict": state}, checkpoint_path)
    checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
