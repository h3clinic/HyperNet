"""Native-resolution procedural surface encoder; all weights start randomly.

Shape invariance and surface sensitivity use separate heads. Auxiliary targets
come from our renderer and mathematical image measurements, never a teacher.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .training_v3 import shape_channels

FORMAT = 'hypernet.mathematical_surface_encoder.v10'


class SurfaceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.GroupNorm(4, 24), nn.GELU())
        self.coarse = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.GroupNorm(8, 48), nn.GELU())
        self.shape_head = nn.Linear(48*4*4, 64)
        self.detail_head = nn.Linear(24*8*8, 64)
        self.family_head = nn.Linear(64, 8)
        self.spectrum_head = nn.Linear(64, 64)
        self.geometry_head = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1),
            nn.GELU(), nn.Conv2d(16, 5, 1))

    def forward(self, images, auxiliary=True):
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError('SurfaceEncoder requires native N x 3 x 128 x 128 input')
        fine = self.stem(shape_channels(images))
        coarse = self.coarse(fine)
        shape = F.normalize(self.shape_head(F.adaptive_avg_pool2d(coarse, 4).flatten(1)), dim=1)
        detail = F.normalize(self.detail_head(F.adaptive_avg_pool2d(fine, 8).flatten(1)), dim=1)
        result = {'shape': shape, 'detail': detail,
            'embedding': torch.cat([math.sqrt(.75)*shape, math.sqrt(.25)*detail], 1)}
        if auxiliary:
            result.update(family_logits=self.family_head(shape),
                spectrum=self.spectrum_head(detail), geometry=self.geometry_head(fine))
        return result

    def embed(self, images):
        return self(images, auxiliary=False)['embedding']


def input_for_arm(images, arm):
    if arm == 'native128':
        return images
    if arm == 'lowpass32':
        return F.interpolate(F.interpolate(images, size=32, mode='area'),
            size=128, mode='bilinear', align_corners=False)
    raise ValueError('Unknown input arm')


def surface_spectrum(images, mask):
    """64 log-power targets: 8 radial x 8 unsigned orientation frequency bins.

    Computed from native128 RGB, even for the low-resolution control. Bins above
    the32px Nyquist limit explicitly retain fine surface-frequency supervision.
    The foreground mask suppresses background shortcuts but boundaries remain.
    """
    gray = (images*images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    mask = mask.to(gray.dtype)
    mean = (gray*mask).sum((1, 2), keepdim=True)/mask.sum((1, 2), keepdim=True).clamp_min(1)
    window = torch.hann_window(128, periodic=False, device=images.device)
    gray = (gray-mean)*mask*window[:, None]*window[None, :]
    power = torch.fft.fft2(gray, norm='ortho').abs().square()
    freq = torch.fft.fftfreq(128, device=images.device)
    yy, xx = torch.meshgrid(freq, freq, indexing='ij')
    radius = (xx.square()+yy.square()).sqrt()
    radial = (radius/.707107*8).long().clamp(0, 7)
    orientation = (torch.remainder(torch.atan2(yy, xx), math.pi)/math.pi*8).long().clamp(0, 7)
    ids = (radial*8+orientation).flatten()
    values = images.new_zeros((len(images), 64))
    values.scatter_add_(1, ids[None].expand(len(images), -1), power.flatten(1))
    counts = torch.bincount(ids, minlength=64).to(images.dtype).clamp_min(1)
    values = torch.log1p(100*values/counts)
    return F.normalize(values, dim=1, eps=1e-6)


def geometry_targets(mask, normal, depth):
    """Mask-weighted pooling keeps background zeros out of normal/depth targets."""
    weight = F.avg_pool2d(mask[:, None].float(), 2)
    normal = F.avg_pool2d(normal*mask[:, None], 2)/weight.clamp_min(1e-6)
    normal = F.normalize(normal, dim=1, eps=1e-6)
    # Fixed camera depth range; no per-frame min-max target instability.
    depth = F.avg_pool2d(((depth[:, None]-2.)/2.)*mask[:, None], 2)/weight.clamp_min(1e-6)
    return weight, normal, depth


def surface_loss(outputs, families, spectrum, mask64, normal64, depth64):
    """Consecutive pairs are the same program in different views/strengths."""
    n = len(families)
    if n % 2:
        raise ValueError('A batch must contain consecutive positive pairs')
    z = outputs['shape']
    similarities = z@z.T/.15
    similarities = similarities.masked_fill(torch.eye(n, device=z.device, dtype=torch.bool), -1e4)
    partners = torch.arange(n, device=z.device)^1
    contrastive = F.cross_entropy(similarities, partners)
    family = F.cross_entropy(outputs['family_logits'], families)
    detail = (1-F.cosine_similarity(outputs['spectrum'], spectrum, dim=1)).mean()
    geometry = outputs['geometry']
    mask_loss = F.binary_cross_entropy_with_logits(geometry[:, :1], mask64)
    weight = (mask64 > .75).float()
    normals = 1-(F.normalize(geometry[:, 1:4], dim=1)*normal64).sum(1, keepdim=True)
    normal_loss = (normals*weight).sum()/weight.sum().clamp_min(1)
    depth_loss = ((geometry[:, 4:5].sigmoid()-depth64).square()*weight).sum()/weight.sum().clamp_min(1)
    terms = dict(family=family, contrastive=contrastive, spectrum=detail,
        mask=mask_loss, normal=normal_loss, depth=depth_loss)
    total = family+.3*contrastive+.5*detail+.2*mask_loss+.2*normal_loss+.2*depth_loss
    return total, terms
