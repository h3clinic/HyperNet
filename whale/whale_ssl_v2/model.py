"""HyperNet-G v2: scratch spectrogram transformer for one-reference glitch retrieval.

No borrowed weights. Tokens are 10x10 pixel patches. Rows carry a learned
frequency embedding because frequency is physically meaningful. There is no
absolute time embedding: a depthwise convolution supplies local position.

v2 changes from v1:
- Input has two channels: standardized intensity, which keeps loudness, and a
  per-image rank transform, which is the same under any monotone brightness
  mapping (a different rendering pipeline or colour scale).
- A mask token and a pixel-reconstruction head support masked-patch prediction.
"""
import torch
from torch import nn
from torch.nn import functional as F

PATCH = 10


def rank_transform(x):
    """Per-image rank of each pixel scaled to zero mean, unit variance."""
    b = x.shape[0]
    flat = x.reshape(b, -1)
    ranks = flat.argsort(1).argsort(1).to(x.dtype) / (flat.shape[1] - 1)
    return ((ranks - .5) * 3.4641).reshape(x.shape)  # uniform -> unit std


def two_channel(x):
    """Channel 0: per-image standardized intensity (keeps loudness). Channel 1: rank (rendering-invariant)."""
    m = x.mean((-1, -2), keepdim=True)
    s = x.std((-1, -2), keepdim=True).clamp_min(1e-4)
    return torch.cat([(x - m) / s, rank_transform(x)], 1)


class SpectroEncoder(nn.Module):
    def __init__(self, width=144, depth=4, heads=6, rows=14, desc=64, proj=512):
        super().__init__()
        self.patch = nn.Conv2d(2, width, PATCH, PATCH)
        self.mask_token = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.freq = nn.Parameter(torch.zeros(1, width, rows, 1))
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 3, dropout=0, batch_first=True,
                                           norm_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.descriptor = nn.Linear(width, desc)
        self.dense_proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, proj))
        self.global_proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, proj))
        self.reconstruct = nn.Linear(width, PATCH * PATCH)
        nn.init.normal_(self.freq, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

    def tokens(self, x, mask=None, transformed=False):
        """x: (B,1,140,W) -> tokens (B, rows, cols, width). mask: (B, rows, cols) bool."""
        z = self.patch(x if transformed else two_channel(x))
        if mask is not None:
            z = torch.where(mask[:, None], self.mask_token.expand_as(z), z)
        z = z + self.freq
        z = z + self.local(z)
        b, c, h, w = z.shape
        t = self.blocks(z.flatten(2).transpose(1, 2))
        return self.norm(t).reshape(b, h, w, c)

    def describe(self, x):
        """Normalized token descriptors (B, rows*cols, desc) and pooled descriptor (B, desc)."""
        t = self.tokens(x)
        d = F.normalize(self.descriptor(t), dim=-1).flatten(1, 2)
        return d, F.normalize(d.mean(1), dim=-1)


def token_nn_score(query_desc, reference_desc, weights=None):
    """Weighted mean over reference tokens of the best-matching query token cosine."""
    best = torch.einsum('qnc,mc->qnm', query_desc, reference_desc).amax(1)
    if weights is None:
        return best.mean(1)
    return (best * weights).sum(1) / weights.sum()


def novelty_weights(reference_desc, bank, keep=.25):
    """Keep the reference tokens least like the unlabeled corpus bank (the glitch, not the background)."""
    novelty = 1 - (reference_desc @ bank.T).amax(1)
    k = max(1, int(round(keep * len(novelty))))
    w = torch.zeros_like(novelty)
    w[novelty.topk(k).indices] = 1
    return w
