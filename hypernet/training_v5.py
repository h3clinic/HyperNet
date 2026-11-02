"""One-reference restoration with complete scale and canonical-region coverage.

The CNN begins at random weights. Analytic supervision consists only of fixed
image measurements made on the supplied reference and its synthetic views.
Region matching is a separate inference operation, never a training source.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .types import rgb_uint8
from .training_v3 import TinyShapeCNN, analytic_shape_target, multiscale_patch_grid
from .training_v4 import _validate_configuration, restoration_loss, restoration_views

FORMAT = "hypernet.single_reference_spatial_restoration_cnn.v5"


def reference_patch_bank(reference, *, image_size=256, patch_size=32,
                         scale_factors=(1, 2, 4), device="cpu"):
    """Keep every pyramid patch and the exact 14 tiles used by region inference."""
    from .models.region_shape import canonical_region_tiles
    reference = rgb_uint8(reference)
    if patch_size != 32:
        raise ValueError("Region inference uses 32-pixel tiles; patch_size must be 32")
    resized = np.asarray(Image.fromarray(reference).resize(
        (image_size, image_size), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(resized).permute(2, 0, 1)[None].to(
        device=device, dtype=torch.float32) / 255.
    patches, points, sizes = multiscale_patch_grid(tensor, patch_size, scale_factors)
    groups = [torch.where(sizes == size)[0] for size in torch.unique(sizes)]
    group_names = [f"pyramid_slice_{int(size)}" for size in torch.unique(sizes).tolist()]
    canonical = canonical_region_tiles(reference, device=device)
    if canonical.shape != (14, 3, 32, 32):
        raise ValueError("Canonical region API must return 14 RGB 32x32 tiles")
    groups.append(torch.arange(len(patches), len(patches) + len(canonical), device=device))
    group_names.append("canonical_whole_2x2_3x3")
    bank = torch.cat([patches, canonical])
    metadata = {
        "pyramid_patch_count": len(patches), "canonical_tile_count": len(canonical),
        "pyramid_patch_centers": points.cpu().tolist(),
        "pyramid_slice_sizes": sizes.cpu().tolist(),
        "bank_groups": [{"name": name, "indices": ids.cpu().tolist()}
                        for name, ids in zip(group_names, groups)],
        "canonical_tiles": "whole crop, 2x2 row-major cells, 3x3 row-major cells; same function as inference",
        "selection": "all pyramid patches retained; no diversity/energy filtering",
    }
    return bank, groups, metadata


class BalancedPatchCycle:
    """Cycle each group without replacement before reshuffling, with equal quotas.

    Complete coverage is a measured outcome in metadata, not assumed when the
    requested number of updates is too small to visit all patches.
    """
    def __init__(self, groups, generator):
        if not groups or any(len(group) == 0 for group in groups):
            raise ValueError("Patch groups must be nonempty")
        self.groups, self.generator = groups, generator
        self.orders = [group[torch.randperm(len(group), generator=generator,
                                           device=group.device)] for group in groups]
        self.offsets, self.step = [0] * len(groups), 0

    def take(self, batch_size):
        n = len(self.groups)
        selected = []
        for i, group in enumerate(self.groups):
            quota = batch_size // n + int((i - self.step) % n < batch_size % n)
            pieces = []
            while quota:
                if self.offsets[i] == len(group):
                    self.orders[i] = group[torch.randperm(len(group), generator=self.generator,
                                                          device=group.device)]
                    self.offsets[i] = 0
                count = min(quota, len(group) - self.offsets[i])
                pieces.append(self.orders[i][self.offsets[i]:self.offsets[i] + count])
                self.offsets[i] += count
                quota -= count
            selected.extend(pieces)
        self.step += 1
        return torch.cat(selected)


def train_reference_shape_v5(reference, checkpoint_path, steps=120, seed=73, device="cpu",
                             batch_size=32, patch_size=32, image_size=256,
                             learning_rate=.002, scale_factors=(1, 2, 4)):
    """Fit a fresh CNN using one image; accept no queries, labels or teacher."""
    from .models.cnn import image_digest
    _validate_configuration(steps, batch_size, 2, image_size, patch_size,
                            learning_rate, scale_factors)
    reference = rgb_uint8(reference)
    path = Path(checkpoint_path)
    sidecar = path.with_suffix(path.suffix + ".json")
    if path.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite checkpoint or provenance")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    started = time.perf_counter()
    patches, groups, bank_metadata = reference_patch_bank(
        reference, image_size=image_size, patch_size=patch_size,
        scale_factors=scale_factors, device=dev)
    insufficient = not np.any(reference != reference[0, 0])
    counts = torch.zeros(len(patches), dtype=torch.long, device=dev)
    history, components = [], []
    rng_devices = [] if dev.type == "cpu" else [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        model = TinyShapeCNN(64).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device=dev).manual_seed(seed + 1)
        sampler = BalancedPatchCycle(groups, generator)
        model.train()
        if not insufficient:
            for _ in range(int(steps)):
                ids = sampler.take(batch_size)
                counts += torch.bincount(ids, minlength=len(patches))
                clean, first, second = restoration_views(patches[ids], generator)
                with torch.no_grad():
                    target = analytic_shape_target(clean)
                z0, z1 = model(torch.cat([first, second])).chunk(2)
                loss, parts = restoration_loss(z0, z1, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite shape restoration loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                history.append(float(loss.detach()))
                components.append({key: float(value.detach()) for key, value in parts.items()})
        model.eval()
        with torch.inference_mode():
            predicted = torch.cat([model(patches[start:start + 128])
                                   for start in range(0, len(patches), 128)])
            target = analytic_shape_target(patches)
            fidelity = float((predicted * target).sum(1).mean())
            canonical_fidelity = float((predicted[-14:] * target[-14:]).sum(1).mean())
        metadata = {
            "method": "complete_reference_shape_restoration_cnn_v5",
            "architecture": "TinyShapeCNN; 8 input channels; 64-D learned descriptor",
            "training_revision": 5, "external_pretraining": False,
            "external_training_images": 0, "target_reference_images": 1,
            "learned_teacher": False, "query_weight_updates": 0,
            "target_weight_updates": len(history), "requested_steps": int(steps),
            "reference_rgb_sha256": image_digest(reference), "reference_shape": list(reference.shape),
            "image_size": int(image_size), "patch_size": int(patch_size),
            "scale_factors": list(scale_factors), "embedding_dim": 64,
            "num_reference_patches": len(patches), **bank_metadata,
            "sampling": "equal group quotas; each group shuffled cyclically without replacement",
            "patch_training_draw_counts": counts.cpu().tolist(),
            "unique_patches_trained": int((counts > 0).sum()),
            "all_reference_patches_trained": bool((counts > 0).all()),
            "insufficient_informative_patches": bool(insufficient),
            "seed": int(seed), "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "synthetic_patch_views_seen": len(history) * batch_size * 2,
            "objective": "unchanged v4 same-geometry analytic target cosine + 0.2 target-relation MSE",
            "cross_geometry_invariance_loss": False,
            "augmentation_recipe": "unchanged v4 shared_geometry and corrupt_same_geometry",
            "analytic_target": "fixed HOG/coarse-gray/edge energy; no learned teacher",
            "inference": "learned descriptors replace analytic descriptors in identical structured region search",
            "training_loss": history, "loss_components": components,
            "clean_target_cosine": fidelity, "canonical_tile_target_cosine": canonical_fidelity,
            "elapsed_seconds": time.perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "torch_version": str(torch.__version__), "device": str(dev),
            "score_is_probability": False, "paper_reproduction": False,
            "limitation": "One-image shape restoration supplies no unseen category examples; region search does not supply pretrained semantics.",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": FORMAT, "metadata": metadata,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, path)
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
