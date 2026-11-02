"""Reference-only shape restoration for an information-preserving hybrid.

The exact analytic representation is retained separately at inference. This
CNN is trained from random initialization to restore same-geometry shape
measurements from mildly degraded reference patches. It never sees queries,
category strings, external images, or a learned teacher.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .types import rgb_uint8
from .training_v2 import select_diverse_patches
from .training_v3 import TinyShapeCNN, analytic_shape_target, multiscale_patch_grid

FORMAT = "hypernet.single_reference_shape_restoration_cnn.v4"


def shared_geometry(patches, generator):
    """Apply geometry once, before producing a target and its corrupted views.

    Different rotated patches have different targets. The loss never asks the
    network to identify two different orientations with the same descriptor.
    """
    n = len(patches)
    def random(shape):
        return torch.rand(shape, generator=generator, device=patches.device, dtype=patches.dtype)
    angle = (random((n,)) - .5) * (30. * math.pi / 180.)
    sx, sy = .9 + .2 * random((n,)), .9 + .2 * random((n,))
    theta = torch.zeros(n, 2, 3, device=patches.device, dtype=patches.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = angle.cos() * sx, -angle.sin() * sy
    theta[:, 1, 0], theta[:, 1, 1] = angle.sin() * sx, angle.cos() * sy
    theta[:, :, 2] = (random((n, 2)) - .5) * .1
    grid = F.affine_grid(theta, patches.shape, align_corners=False)
    return F.grid_sample(patches, grid, padding_mode="reflection", align_corners=False)


def corrupt_same_geometry(clean, generator):
    """Change detail/illumination, never the coordinate frame of the target."""
    n = len(clean)
    def random(shape):
        return torch.rand(shape, generator=generator, device=clean.device, dtype=clean.dtype)
    mean = clean.mean((2, 3), keepdim=True)
    view = (clean - mean) * (.8 + .4 * random((n, 1, 1, 1))) + mean
    view = view * (.9 + .2 * random((n, 1, 1, 1)))
    view = view * (.95 + .1 * random((n, 3, 1, 1)))
    reduced = F.interpolate(view, size=max(4, clean.shape[-1] // 2), mode="bilinear",
                            align_corners=False, antialias=True)
    restored = F.interpolate(reduced, size=clean.shape[-2:], mode="bilinear", align_corners=False)
    view = torch.where(random((n, 1, 1, 1)) < .3, restored, view)
    blurred = F.avg_pool2d(F.pad(view, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    view = torch.where(random((n, 1, 1, 1)) < .2, blurred, view)
    noise = torch.randn(view.shape, generator=generator, device=view.device, dtype=view.dtype) * .005
    return (view + noise).clamp(0., 1.)


def restoration_views(patches, generator):
    """Return a clean shared-geometry target and two independently degraded views."""
    clean = shared_geometry(patches, generator)
    return clean, corrupt_same_geometry(clean, generator), corrupt_same_geometry(clean, generator)


def restoration_loss(z0, z1, clean_target):
    """Two predictions of ONE target; there is no cross-geometry consistency loss."""
    target = clean_target.detach()
    prediction = 1. - .5 * ((z0 * target).sum(1).mean() + (z1 * target).sum(1).mean())
    z = torch.cat([z0, z1])
    targets = torch.cat([target, target])
    relations = F.mse_loss(z @ z.t(), targets @ targets.t())
    return prediction + .2 * relations, {"prediction": prediction, "relations": relations}


def _validate_configuration(steps, batch_size, max_patches, image_size, patch_size,
                            learning_rate, scale_factors):
    for name, value, minimum in (("steps", steps, 1), ("batch_size", batch_size, 2),
                                 ("max_patches", max_patches, 2), ("image_size", image_size, 8),
                                 ("patch_size", patch_size, 8)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (not scale_factors or any(isinstance(s, bool) or int(s) != s or s < 1 for s in scale_factors)
            or len(set(scale_factors)) != len(scale_factors)):
        raise ValueError("scale_factors must be unique positive integers")
    if image_size < patch_size * max(scale_factors):
        raise ValueError("image_size must accommodate every slice scale")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")


def train_reference_shape_v4(reference, checkpoint_path, steps=120, seed=73, device="cpu",
                             batch_size=32, patch_size=32, max_patches=64, image_size=256,
                             learning_rate=.002, scale_factors=(1, 2, 4)):
    """Train on exactly one RGB array; all patches/views derive from that array."""
    from .models.cnn import image_digest
    _validate_configuration(steps, batch_size, max_patches, image_size, patch_size,
                            learning_rate, scale_factors)
    reference = rgb_uint8(reference)
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
    unique_sizes = torch.unique(sizes).tolist()
    selections, selection_report = [], []
    for i, size in enumerate(unique_sizes):
        quota = max_patches // len(unique_sizes) + int(i < max_patches % len(unique_sizes))
        if quota < 1:
            continue
        candidates = torch.where(sizes == size)[0]
        chosen, report = select_diverse_patches(patches[candidates], points[candidates], max(2, quota))
        selections.append(candidates[chosen[:quota]])
        selection_report.append({"slice_size": size, **report})
    chosen = torch.cat(selections) if selections else torch.empty(0, device=dev, dtype=torch.long)
    selected = patches[chosen]
    count = len(selected)
    insufficient = count < 2 or not np.any(reference != reference[0, 0])
    batch_n = min(batch_size, count)
    history, components = [], []
    rng_devices = [] if dev.type == "cpu" else [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        # Architecture reuse is not weight reuse: every parameter starts fresh.
        model = TinyShapeCNN(64).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device=dev).manual_seed(seed + 1)
        model.train()
        if not insufficient:
            for _ in range(int(steps)):
                ids = torch.randperm(count, generator=generator, device=dev)[:batch_n]
                clean, first, second = restoration_views(selected[ids], generator)
                with torch.no_grad():
                    target = analytic_shape_target(clean)
                z0, z1 = model(torch.cat([first, second])).chunk(2)
                loss, pieces = restoration_loss(z0, z1, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite shape restoration loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
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
            "method": "shape_restoration_cnn_v4", "architecture": "TinyShapeCNN; 8 input channels, 64-D CNN output",
            "training_revision": 4, "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "query_weight_updates": 0,
            "target_weight_updates": len(history), "requested_steps": int(steps),
            "reference_rgb_sha256": image_digest(reference), "reference_shape": list(reference.shape),
            "image_size": int(image_size), "patch_size": int(patch_size),
            "scale_factors": list(scale_factors), "embedding_dim": 64,
            "num_reference_patches": count, "selected_original_patch_indices": chosen.cpu().tolist(),
            "selected_patch_centers": points[chosen].cpu().tolist(), "selected_slice_sizes": sizes[chosen].cpu().tolist(),
            "selection": selection_report, "insufficient_informative_patches": insufficient,
            "seed": int(seed), "batch_size": int(batch_size), "effective_batch_size": batch_n,
            "learning_rate": float(learning_rate), "synthetic_patch_views_seen": len(history) * batch_n * 2,
            "input_channels": ["red", "green", "blue", "low_frequency_gray", "sobel_x", "sobel_y",
                               "edge_magnitude", "high_frequency_gray"],
            "geometry": {"rotation_degrees": [-15, 15], "anisotropic_scale": [.9, 1.1],
                         "translation_normalized": [-.05, .05],
                         "coupling": "ONE geometric warp is shared by the clean target and both degraded inputs"},
            "corruptions": {"contrast": [.8, 1.2], "brightness": [.9, 1.1], "color_scale": [.95, 1.05],
                            "resolution_divisor": 2, "resolution_probability": .3,
                            "blur_probability": .2, "gaussian_noise_std": .005},
            "objective": "same-geometry clean analytic target cosine prediction + 0.2 target-relation MSE",
            "cross_geometry_invariance_loss": False,
            "analytic_target": "unchanged fixed 64-D HOG/coarse-gray/edge measurement; no learned teacher",
            "analytic_branch_at_inference": "retained exactly by GuardedShapeMatcher; this checkpoint stores only the scratch CNN",
            "training_loss": history, "loss_components": components,
            "clean_descriptor_offdiag_cosine_mean": clean_mean,
            "clean_descriptor_offdiag_cosine_std": clean_std, "clean_target_cosine": target_cosine,
            "elapsed_seconds": time.perf_counter() - started,
            "torch_version": str(torch.__version__), "device": str(dev),
            "parameters": sum(p.numel() for p in model.parameters()),
            "score_is_probability": False, "paper_reproduction": False,
            "limitation": "One-reference restoration can still overfit; retaining analytic features does not guarantee unchanged matches or better category accuracy.",
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": FORMAT, "metadata": metadata,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, checkpoint_path)
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
