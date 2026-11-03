"""Dense, locally supervised matching without borrowed weights.

The local and contextual channels remain separate until their cosine evidence is
computed. Each ROI retains all 64 sampled descriptors; no cellwise spatial mean
or top-k descriptor selection is performed. This combines established matching
ideas; architectural novelty and natural-image transfer are hypotheses to test.
"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'hypernet.dense_match.v19.1'
MODEL_CONFIG = {'local_channels': 32, 'context_channels': 32, 'roi_side': 8,
                'initialization': 'random', 'borrowed_pretrained_weights': False,
                'sampler': 'Explicit bilinear gathers with border clamping',
                'support_extent': 'Predicted foreground >0.5; full support if empty; minimum4featurecells',
                'descriptor_stride': 2, 'descriptor_pixel_origin': .5,
                'query_pyramid_scales': (.5, 1., 2.), 'normalized_roi_side': 96.}


def state_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode()); value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def file_sha(path):
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ChannelNorm(nn.Module):
    """Per-pixel channel normalization, independent of other images/regions."""
    def __init__(self, channels):
        super().__init__(); self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        value = x-x.mean(1, keepdim=True)
        return value*torch.rsqrt(value.square().mean(1, keepdim=True)+1e-4)*self.weight+self.bias


class Block(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=dilation,
            dilation=dilation, groups=channels, bias=False), ChannelNorm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False), ChannelNorm(channels))

    def forward(self, x): return F.silu(x+self.layers(x))


def bilinear_sample(feature, points, stride=2., origin=.5):
    """Sample BxCxHxW at Bx...x2 original-image pixel centers.

    Integer corner indices are detached, but interpolation weights and features
    retain gradients. Avoids CUDA grid_sample backward nondeterminism. Border
    clamping matches grid_sample(..., padding_mode='border').
    """
    if points.shape[0] != feature.shape[0] or points.shape[-1] != 2:
        raise ValueError('Expected matching batch dimension and xy point coordinates')
    b, c, h, w = feature.shape; shape = points.shape[1:-1]
    xy = (points.reshape(b, -1, 2)-origin)/stride
    x = xy[..., 0].clamp(0, w-1); y = xy[..., 1].clamp(0, h-1)
    x0 = x.floor().long(); y0 = y.floor().long()
    x1 = (x0+1).clamp(max=w-1); y1 = (y0+1).clamp(max=h-1)
    dx = (x-x0).unsqueeze(1); dy = (y-y0).unsqueeze(1)
    flat = feature.flatten(2)
    def take(xx, yy): return flat.gather(2, (yy*w+xx).unsqueeze(1).expand(-1, c, -1))
    result = (take(x0, y0)*(1-dx)+take(x1, y0)*dx)*(1-dy)
    result = result+(take(x0, y1)*(1-dx)+take(x1, y1)*dx)*dy
    return result.reshape(b, c, *shape)


def roi_points(boxes, side=8):
    if boxes.ndim != 3 or boxes.shape[-1] != 4: raise ValueError('Expected BxNx4 xyxy boxes')
    axis = (torch.arange(side, device=boxes.device, dtype=boxes.dtype)+.5)/side
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    unit = torch.stack([xx, yy], -1).reshape(1, 1, side*side, 2)
    # Boxes have half-open pixel boundaries; integer coordinate0 is pixel0 center.
    return boxes[..., None, :2]+unit*(boxes[..., None, 2:]-boxes[..., None, :2])-.5


def resize_by_two(rgb, scale):
    if scale == 1.: return rgb
    if scale == .5: return F.avg_pool2d(rgb, 2, 2)
    if scale != 2.: raise ValueError('Fixed pyramid supports .5,1,2 only')
    b, _, h, w = rgb.shape
    y = (torch.arange(2*h, device=rgb.device, dtype=rgb.dtype)+.5)/2-.5
    x = (torch.arange(2*w, device=rgb.device, dtype=rgb.dtype)+.5)/2-.5
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    points = torch.stack([xx, yy], -1)[None].expand(b, -1, -1, -1)
    return bilinear_sample(rgb, points, stride=1., origin=0.)


def pyramid_indices(boxes):
    sizes = (boxes[..., 2:]-boxes[..., :2]).clamp_min(1).prod(-1).sqrt()
    scales = boxes.new_tensor(MODEL_CONFIG['query_pyramid_scales'])
    return ((sizes[..., None]*scales/96).log().abs()).argmin(-1)


def predicted_extent(logits, image_height, image_width):
    selected = logits[:, 0].detach() > 0; b, h, w = selected.shape
    xs = torch.arange(w, device=logits.device)[None].expand(b, -1)
    ys = torch.arange(h, device=logits.device)[None].expand(b, -1)
    xx = selected.any(1); yy = selected.any(2)
    x0 = xs.masked_fill(~xx, w).amin(1); x1 = xs.masked_fill(~xx, -1).amax(1)+1
    y0 = ys.masked_fill(~yy, h).amin(1); y1 = ys.masked_fill(~yy, -1).amax(1)+1
    empty = (x1 <= x0) | (y1 <= y0)
    # Integer bounds are coordinates in stride2 cells; expand rather than discard
    # small predicted objects. No ground-truth mask enters this computation.
    cx = (x0+x1).float()/2; cy = (y0+y1).float()/2
    bw = (x1-x0).clamp(min=min(4, w)); bh = (y1-y0).clamp(min=min(4, h))
    left = (cx-bw/2).floor().clamp(min=0); top = (cy-bh/2).floor().clamp(min=0)
    left = torch.minimum(left, w-bw); top = torch.minimum(top, h-bh)
    boxes = 2*torch.stack([left, top, left+bw, top+bh], 1)
    full = boxes.new_tensor([0, 0, image_width, image_height])[None].expand(b, -1)
    boxes = torch.where(empty[:, None], full, boxes)
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, image_width)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, image_height)
    return boxes[:, None]


class DenseMatchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(7, 24, 3, padding=1, bias=False), ChannelNorm(24),
                                  nn.SiLU(), Block(24), Block(24))
        self.local = nn.Sequential(nn.Conv2d(96, 32, 1), nn.SiLU())
        self.context = nn.Sequential(nn.Conv2d(32, 32, 1, bias=False), ChannelNorm(32),
                                      nn.SiLU(), Block(32, 1), Block(32, 2))
        self.foreground = nn.Sequential(nn.Conv2d(64, 24, 1), nn.SiLU(), nn.Conv2d(24, 1, 1))
        self.token_head = nn.Sequential(nn.Conv2d(16, 24, 1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1, groups=24), nn.SiLU())
        self.score_head = nn.Sequential(nn.Linear(24*8*8, 64), nn.SiLU(), nn.Linear(64, 1))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(.10)))
        self.dustbin = nn.Parameter(torch.tensor(0.))
        self.register_buffer('sobel', torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]/8)
        self.register_buffer('laplace', torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]])[None, None])
        axis = (torch.arange(8).float()+.5)/8
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('unit_points', torch.stack([xx, yy], -1).reshape(64, 2))

    @property
    def temperature(self): return self.log_temperature.exp().clamp(.03, .3)

    def encode(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 16:
            raise ValueError('Expected float RGB BCHW with dimensions >=16')
        h, w = rgb.shape[-2:]; ph, pw = (-h)%4, (-w)%4
        if ph or pw: rgb = F.pad(rgb, (0, pw, 0, ph), mode='reflect')
        gray = rgb.mean(1, keepdim=True)
        high = gray-F.avg_pool2d(gray, 5, 1, 2)
        contrast = (high/(.06+F.avg_pool2d(high.square(), 5, 1, 2).clamp_min(1e-8).sqrt())).clamp(-3, 3)/3
        edges = torch.cat([contrast, 4*F.conv2d(gray, self.sobel, padding=1),
                           4*F.conv2d(gray, self.sobel.transpose(2, 3), padding=1),
                           F.conv2d(gray, self.laplace, padding=1)], 1)
        fine = self.stem(torch.cat([2*rgb-1, edges], 1))
        local = self.local(F.pixel_unshuffle(fine, 2))
        context = self.context(F.avg_pool2d(local, 2, 2))
        context = context.repeat_interleave(2, -2).repeat_interleave(2, -1)
        raw = torch.cat([local, context], 1)[:, :, :(h+1)//2, :(w+1)//2]
        # Neither branch can suppress the other's total norm during matching.
        dense = torch.cat([F.normalize(raw[:, :32], dim=1, eps=1e-6),
                           F.normalize(raw[:, 32:], dim=1, eps=1e-6)], 1)/math.sqrt(2)
        return {'raw': raw, 'dense': dense, 'foreground_logits': self.foreground(raw),
                'height': h, 'width': w}

    def sample_regions(self, encoded, boxes):
        points = roi_points(boxes)
        # Interpolate the final descriptors, then renormalize each path separately.
        raw = bilinear_sample(encoded['raw'], points)
        descriptor = torch.cat([F.normalize(raw[:, :32], dim=1, eps=1e-6),
            F.normalize(raw[:, 32:], dim=1, eps=1e-6)], 1)/math.sqrt(2)
        fg = bilinear_sample(encoded['foreground_logits'], points).sigmoid()
        norms = raw.square().sum(1).clamp_min(1e-8).sqrt().log1p()
        return {'descriptors': descriptor.permute(0, 2, 1, 3).contiguous(),
                'foreground': fg[:, 0], 'norms': norms, 'boxes': boxes}

    def support_bank(self, encoded):
        box = predicted_extent(encoded['foreground_logits'], encoded['height'], encoded['width'])
        return self.sample_regions(encoded, box)

    def encode_query(self, rgb):
        return {'levels': [self.encode(resize_by_two(rgb, scale)) for scale in MODEL_CONFIG['query_pyramid_scales']],
                'height': rgb.shape[-2], 'width': rgb.shape[-1]}

    def sample_query_regions(self, pyramid, boxes):
        indices = pyramid_indices(boxes); result = {}
        for i, (level, scale) in enumerate(zip(pyramid['levels'], MODEL_CONFIG['query_pyramid_scales'])):
            sampled = self.sample_regions(level, boxes*scale)
            for key in ('descriptors', 'foreground', 'norms'):
                active = (indices == i).to(boxes.dtype)
                while active.ndim < sampled[key].ndim: active = active.unsqueeze(-1)
                result[key] = result.get(key, 0)+sampled[key]*active
        result['boxes'] = boxes
        return result

    def score_regions(self, support, regions, diagnostics=False):
        s = support['descriptors']; q = regions['descriptors']
        if s.shape[1] != 1 or s.shape[-2:] != (64, 64) or q.shape[-2:] != (64, 64):
            raise ValueError('Expected one support bank and N regions with64channels/64tokens')
        if s.shape[0] == 1 and q.shape[0] > 1:
            support = {k: v.expand(q.shape[0], *v.shape[1:]) for k, v in support.items()}
            s = support['descriptors']
        b, n = q.shape[:2]
        corr = torch.einsum('bcx,bncy->bnxy', s[:, 0], q)
        logits = corr/self.temperature
        dust = self.dustbin+math.log(64)
        row_dust = dust.expand(b, n, 64, 1)
        col_dust = dust.expand(b, n, 1, 64)
        forward = torch.cat([logits, row_dust], -1).softmax(-1)[..., :64]
        backward = torch.cat([logits, col_dust], -2).softmax(-2)[..., :64, :]
        mutual = forward*backward
        mass = mutual.sum(-1).clamp_min(1e-8)
        matched = mutual/mass[..., None]
        coords = self.unit_points.to(q.dtype)
        expected = torch.einsum('bnxy,yd->bnxd', matched, coords)
        displacement = expected-coords[None, None]
        # Pairwise distance consistency is rotation invariant. All source tokens
        # participate; uncertain/foreground-negative matches receive low weight.
        reference_distance = (coords[:, None]-coords[None]).square().sum(-1).clamp_min(1e-8).sqrt()
        query_distance = (expected[..., :, None, :]-expected[..., None, :, :]).square().sum(-1).clamp_min(1e-8).sqrt()
        residual = (query_distance-reference_distance).abs()
        sf = support['foreground'][:, 0, None].expand(-1, n, -1)
        qf = regions['foreground']
        pairweight = mass*sf
        residual_mean = (residual*pairweight[..., None, :]).sum(-1)/pairweight.sum(-1, keepdim=True).clamp_min(1e-8)
        entropy = -(matched*matched.clamp_min(1e-8).log()).sum(-1)/math.log(64)
        evidence = torch.stack([corr.amax(-1), (matched*corr).sum(-1), forward.sum(-1), mass,
            entropy, expected[..., 0], expected[..., 1], displacement[..., 0], displacement[..., 1],
            residual_mean, residual.amax(-1), sf, (matched*qf[..., None, :]).sum(-1),
            support['norms'][:, 0, None].expand(-1, n, -1)/4,
            (matched*regions['norms'][..., None, :]).sum(-1)/4,
            (corr*backward).sum(-1)/backward.sum(-1).clamp_min(1e-8)], 2)
        hidden = self.token_head(evidence.reshape(b*n, 16, 8, 8))
        score = self.score_head(hidden.flatten(1)).reshape(b, n)
        if diagnostics: return score, {'correlation': corr, 'mutual': mutual, 'evidence': evidence,
                                       'expected_coordinates': expected, 'geometry_residual': residual_mean}
        return score

    def forward(self, support, query, rois):
        encoded_support = self.encode(support); encoded_query = self.encode_query(query)
        bank = self.support_bank(encoded_support)
        regions = self.sample_query_regions(encoded_query, rois)
        return {'scores': self.score_regions(bank, regions), 'support_encoded': encoded_support,
                'query_encoded': encoded_query, 'support_bank': bank, 'regions': regions}


def parameter_summary(model):
    return {'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'descriptor_channels': 64, 'reference_tokens': 64, 'region_tokens': 64,
            'correlation_entries_per_roi': 4096, 'config': dict(MODEL_CONFIG)}
