"""One-reference CNN with multiscale slices and analytic shape prediction.

The analytic target is a disclosed hand-coded prior, not a pretrained model.
All learned parameters are initialized from scratch and updated using the one
reference image only. The learned descriptor, rather than the analytic target,
is used during matching.
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
from .training import patch_grid
from .training_v2 import select_diverse_patches

FORMAT = "hypernet.single_reference_shape_cnn.v3"


def _gray_and_edges(patches):
    gray = (patches * patches.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    centered = gray - gray.mean((2, 3), keepdim=True)
    contrast = (centered.square().mean((2, 3), keepdim=True) + 1e-6).sqrt().clamp_min(.05)
    normalized = centered / contrast
    kernel = patches.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
    kernels = torch.stack([kernel, kernel.t()])[:, None]
    edges = F.conv2d(F.pad(normalized, (1, 1, 1, 1), mode="reflect"), kernels)
    return normalized, edges


def shape_channels(patches):
    """Eight channels: RGB, low-pass gray, signed edges, magnitude, detail."""
    normalized, edges = _gray_and_edges(patches)
    low = F.avg_pool2d(F.pad(normalized, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    magnitude = (edges.square().sum(1, keepdim=True) + 1e-8).sqrt()
    detail = normalized - low
    return torch.cat([patches * 2. - 1., low, edges, magnitude, detail], dim=1)


def analytic_shape_target(patches):
    """64 values: centered HOG (32), coarse gray (16), edge energy (16).

    HOG bins are pooled over 2x2 spatial cells and normalized within each cell.
    Each descriptor block is independently normalized, preventing raw contrast
    from dominating. These are mathematical image measurements, with no learned
    parameters, category labels, or external training images.
    """
    gray, edges = _gray_and_edges(patches)
    magnitude = edges.square().sum(1, keepdim=True).sqrt()
    angle = torch.remainder(torch.atan2(edges[:, 1], edges[:, 0]), math.pi)
    centers = torch.arange(8, device=patches.device, dtype=patches.dtype) * (math.pi / 8.)
    delta = (angle[:, None] - centers[None, :, None, None]).abs()
    delta = torch.minimum(delta, math.pi - delta)
    votes = (1. - delta / (math.pi / 8.)).clamp_min(0.) * magnitude
    hog = F.adaptive_avg_pool2d(votes, (2, 2))
    hog = hog / hog.square().sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    hog = (hog - hog.mean(1, keepdim=True)).flatten(1)
    coarse = F.adaptive_avg_pool2d(gray, (4, 4)).flatten(1)
    coarse = coarse - coarse.mean(1, keepdim=True)
    energy = F.adaptive_avg_pool2d(magnitude, (4, 4)).flatten(1)
    energy = energy - energy.mean(1, keepdim=True)
    blocks = [F.normalize(block, dim=1, eps=1e-6) for block in (hog, coarse, energy)]
    return F.normalize(torch.cat(blocks, dim=1), dim=1, eps=1e-6)


class TinyShapeCNN(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        if embedding_dim != 64:
            raise ValueError("The analytic shape target requires embedding_dim=64")
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, embedding_dim),
        )

    def embed(self, patches):
        return F.normalize(self.encoder(shape_channels(patches)), dim=1)

    def forward(self, patches):
        return self.embed(patches)


def multiscale_patch_grid(image, patch_size=32, scale_factors=(1, 2, 4)):
    """Half-overlapping slices resized to a common CNN input size.

    Points use the boundary-coordinate convention of patch_grid; callers
    converting to image pixel centers subtract one half after resizing.
    """
    if patch_size < 8 or not scale_factors or any(int(s) != s or s < 1 for s in scale_factors):
        raise ValueError("Use patch_size>=8 and positive integral scale_factors")
    if len(set(scale_factors)) != len(scale_factors):
        raise ValueError("scale_factors must be unique")
    patches, points, sizes = [], [], []
    for factor in scale_factors:
        size = int(patch_size * factor)
        if size > min(image.shape[-2:]):
            continue
        p, xy = patch_grid(image, size, max(1, size // 2))
        if size != patch_size:
            p = F.interpolate(p, size=(patch_size, patch_size), mode="bilinear", align_corners=False, antialias=True)
        patches.append(p)
        points.append(xy)
        sizes.append(torch.full((len(p),), size, device=image.device, dtype=torch.long))
    if not patches:
        raise ValueError("No pyramid slices fit inside the image")
    return torch.cat(patches), torch.cat(points), torch.cat(sizes)


def augment_shape_patches(patches, generator):
    batch = len(patches)
    def random(shape):
        return torch.rand(shape, generator=generator, device=patches.device)
    angle = (random((batch,)) - .5) * (50. * math.pi / 180.)
    sx, sy = .8 + .4 * random((batch,)), .8 + .4 * random((batch,))
    theta = torch.zeros(batch, 2, 3, device=patches.device)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(angle) * sx, -torch.sin(angle) * sy
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(angle) * sx, torch.cos(angle) * sy
    theta[:, :, 2] = (random((batch, 2)) - .5) * .2
    grid = F.affine_grid(theta, patches.shape, align_corners=False)
    elastic = F.interpolate(random((batch, 2, 4, 4)) * 2. - 1., size=patches.shape[-2:],
                            mode="bicubic", align_corners=False).permute(0, 2, 3, 1) * .025
    transformed = F.grid_sample(patches, grid + elastic, padding_mode="reflection", align_corners=False)
    # Compression changes effective detail resolution without adding image data.
    if random(()).item() < .5:
        divisor = 2 if random(()).item() < .7 else 4
        reduced = F.interpolate(transformed, size=max(4, patches.shape[-1] // divisor), mode="bilinear",
                                align_corners=False, antialias=True)
        transformed = F.interpolate(reduced, size=patches.shape[-2:], mode="bilinear", align_corners=False)
    mean = transformed.mean((2, 3), keepdim=True)
    transformed = (transformed - mean) * (.65 + .7 * random((batch, 1, 1, 1))) + mean
    transformed = transformed * (.75 + .5 * random((batch, 1, 1, 1)))
    transformed = transformed * (.85 + .3 * random((batch, 3, 1, 1)))
    gray = transformed.mean(1, keepdim=True)
    transformed = torch.where(random((batch, 1, 1, 1)) < .25, gray.expand_as(transformed), transformed)
    blurred = F.avg_pool2d(F.pad(transformed, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    transformed = torch.where(random((batch, 1, 1, 1)) < .3, blurred, transformed)
    quantized = torch.round(transformed * 15.) / 15.
    transformed = torch.where(random((batch, 1, 1, 1)) < .25, quantized, transformed)
    noise = torch.randn(transformed.shape, generator=generator, device=patches.device) * .01
    return (transformed + noise).clamp(0., 1.)


def shape_descriptor_loss(z0, z1, target0, target1):
    """Predict shape measurements, align views, preserve their feature relations."""
    target0, target1 = target0.detach(), target1.detach()
    prediction = 1. - .5 * ((z0 * target0).sum(1).mean() + (z1 * target1).sum(1).mean())
    consistency = 1. - (z0 * z1).sum(1).mean()
    z = torch.cat([z0, z1])
    target = torch.cat([target0, target1])
    relational = F.mse_loss(z @ z.t(), target @ target.t())
    # The targets anchor descriptor orientation and variance: no patch-identity
    # labels or artificial repulsion of naturally similar reference parts.
    loss = prediction + .1 * consistency + .2 * relational
    return loss, {"prediction": prediction, "view_consistency": consistency, "relations": relational}


def train_reference_cnn_v3(reference, checkpoint_path, steps=120, seed=61, device="cpu",
                           batch_size=32, patch_size=32, max_patches=64, image_size=256,
                           embedding_dim=64, learning_rate=.002, scale_factors=(1, 2, 4)):
    from .models.cnn import image_digest
    reference = rgb_uint8(reference)
    if steps < 1 or int(steps) != steps or batch_size < 2 or max_patches < 2:
        raise ValueError("steps must be positive; batch_size and max_patches must be at least two")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if patch_size < 8 or image_size < patch_size * 2:
        raise ValueError("Use patch_size>=8 and image_size>=2*patch_size")
    if embedding_dim != 64:
        raise ValueError("embedding_dim must be 64")
    checkpoint_path = Path(checkpoint_path)
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if checkpoint_path.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint or provenance file")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    started = time.perf_counter()
    resized = np.asarray(Image.fromarray(reference).resize((image_size, image_size), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(resized).permute(2, 0, 1)[None].to(dev, dtype=torch.float32) / 255.
    patches, points, sizes = multiscale_patch_grid(tensor, patch_size, scale_factors)
    selections, selection_report = [], []
    unique_sizes = torch.unique(sizes).tolist()
    quotas = [max_patches // len(unique_sizes) + (int(i < max_patches % len(unique_sizes)))
              for i in range(len(unique_sizes))]
    for size, quota in zip(unique_sizes, quotas):
        candidates = torch.where(sizes == size)[0]
        if quota < 1:
            continue
        chosen, report = select_diverse_patches(patches[candidates], points[candidates], max(2, quota))
        chosen = chosen[:quota]
        selections.append(candidates[chosen])
        selection_report.append({"slice_size": size, **report})
    chosen = torch.cat(selections) if selections else torch.empty(0, device=dev, dtype=torch.long)
    selected = patches[chosen]
    count = len(selected)
    insufficient = count < 2
    batch_n = min(batch_size, count)
    history, components = [], []
    rng_devices = [] if dev.type == "cpu" else [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        model = TinyShapeCNN(embedding_dim).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device=dev).manual_seed(seed + 1)
        model.train()
        if not insufficient:
            for _ in range(int(steps)):
                indices = torch.randperm(count, generator=generator, device=dev)[:batch_n]
                first = augment_shape_patches(selected[indices], generator)
                second = augment_shape_patches(selected[indices], generator)
                with torch.no_grad():
                    target0, target1 = analytic_shape_target(first), analytic_shape_target(second)
                z0, z1 = model(torch.cat([first, second])).chunk(2)
                loss, pieces = shape_descriptor_loss(z0, z1, target0, target1)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite shape descriptor loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                history.append(float(loss.detach()))
                components.append({key: float(value.detach()) for key, value in pieces.items()})
        model.eval()
        with torch.inference_mode():
            if count >= 2:
                clean = model(selected)
                similarities = clean @ clean.t()
                mask = ~torch.eye(count, dtype=torch.bool, device=dev)
                clean_mean = float(similarities[mask].mean())
                clean_std = float(similarities[mask].std(unbiased=False))
                target_cosine = float((clean * analytic_shape_target(selected)).sum(1).mean())
            else:
                clean_mean = clean_std = target_cosine = None
        metadata = {
            "method": "tiny_shape_cnn_v3", "architecture": "8-channel scratch CNN / 64-D shape descriptor",
            "training_revision": 3, "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "target_weight_updates": len(history),
            "requested_steps": int(steps), "reference_rgb_sha256": image_digest(reference),
            "reference_shape": list(reference.shape), "image_size": int(image_size),
            "patch_size": int(patch_size), "scale_factors": [int(size // patch_size) for size in unique_sizes],
            "requested_scale_factors": list(scale_factors),
            "embedding_dim": embedding_dim, "num_reference_patches": count,
            "selected_original_patch_indices": chosen.cpu().tolist(),
            "selected_patch_centers": points[chosen].cpu().tolist(),
            "selected_slice_sizes": sizes[chosen].cpu().tolist(), "selection": selection_report,
            "insufficient_informative_patches": insufficient, "seed": int(seed),
            "batch_size": int(batch_size), "effective_batch_size": batch_n,
            "learning_rate": learning_rate, "synthetic_patch_views_seen": len(history) * batch_n * 2,
            "input_channels": ["red", "green", "blue", "low_frequency_gray",
                               "sobel_x", "sobel_y", "edge_magnitude", "high_frequency_gray"],
            "augmentations": {"rotation_degrees": [-25, 25], "anisotropic_scale": [.8, 1.2],
                              "translation_normalized": [-.1, .1], "elastic_displacement_normalized": .025,
                              "resolution_divisors": [1, 2, 4], "posterization_levels": 16,
                              "brightness_contrast_color": True, "grayscale_probability": .25,
                              "blur_probability": .3, "gaussian_noise_std": .01},
            "objective": "analytic HOG/coarse-gray/edge target + 0.1 paired-view consistency + 0.2 relational MSE",
            "analytic_target": "64-D hand-coded image measurement; no learned teacher; not a semantic label",
            "descriptor_at_inference": "CNN output only; analytic target is not concatenated",
            "training_loss": history, "loss_components": components,
            "clean_descriptor_offdiag_cosine_mean": clean_mean,
            "clean_descriptor_offdiag_cosine_std": clean_std, "clean_target_cosine": target_cosine,
            "elapsed_seconds": time.perf_counter() - started, "torch_version": str(torch.__version__),
            "device": str(dev), "parameters": sum(p.numel() for p in model.parameters()),
            "score_is_probability": False, "paper_reproduction": False,
            "sources": ["https://arxiv.org/abs/2112.09133"],
            "limitation": "Hand-coded shape priors and one-image augmentation cannot supply unseen category variation.",
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": FORMAT, "metadata": metadata,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
                   checkpoint_path)
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

