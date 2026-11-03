"""Synthetic-only dense exemplar matching with partial spatial agreement.

The encoder consumes RGB alone. A 4x4 bank of predicted-foreground-weighted
support descriptors votes at relative positions in candidate boxes. The matched
ablation replaces spatial votes with a bag of locally pooled part similarities.
These are engineering adaptations of established dense matching ideas, not a
claim that correlation, geometry voting, or conditioned regression is novel.
"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'hypernet.local_relations.v17.2'
SCALES = (.16, .28, .44, .66, .90)
ASPECTS = (.5, 1., 2.)
PART_SIDE = 4


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ChannelNorm(nn.Module):
    """Normalize channels at each pixel; no batch/image statistics leak."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        centered = x - x.mean(1, keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(1, keepdim=True) + 1e-4) * self.weight + self.bias


class Block(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
            ChannelNorm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 1, bias=False), ChannelNorm(channels))

    def forward(self, x): return F.silu(x + self.layers(x))


def stage(cin, cout, dilations=()):
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), ChannelNorm(cout), nn.SiLU(),
                         *(Block(cout, d) for d in dilations))


def anchor_sizes(h, w, device=None, dtype=torch.float32):
    """Candidate width/height in native pixels, tied to the query short side."""
    side = min(h, w)
    return torch.tensor([(side * scale * math.sqrt(aspect), side * scale / math.sqrt(aspect))
                         for scale in SCALES for aspect in ASPECTS], device=device, dtype=dtype)


def integral_box_means(x, sizes):
    """Zero-outside, valid-area rectangle means via one summed-area table.

    sizes are integer (height,width), always odd. All windows retain the source
    spatial shape and ignore pixels outside it. No learned sampling coordinates.
    """
    h, w = x.shape[-2:]
    py = max(kh // 2 for kh, kw in sizes); px = max(kw // 2 for kh, kw in sizes)
    # exp(8*cosine) spans exp(-8)..exp(8). Subtracting large float32
    # prefixes can erase small positive window sums or make them negative.
    # Accumulate AND difference in float64, then return native descriptor dtype.
    padded = F.pad(x.to(torch.float64), (px, px, py, py))
    sat = F.pad(deterministic_cumsum(deterministic_cumsum(padded, -1), -2), (1, 0, 1, 0))
    yy = torch.arange(h, device=x.device); xx = torch.arange(w, device=x.device)
    result = []
    for kh, kw in sizes:
        y0, x0 = py - kh // 2, px - kw // 2
        total = sat[..., y0+kh:y0+kh+h, x0+kw:x0+kw+w] - sat[..., y0:y0+h, x0+kw:x0+kw+w]
        total = total - sat[..., y0+kh:y0+kh+h, x0:x0+w] + sat[..., y0:y0+h, x0:x0+w]
        count_y = ((yy + kh//2 + 1).clamp(max=h) - (yy - kh//2).clamp(min=0)).to(x.dtype)
        count_x = ((xx + kw//2 + 1).clamp(max=w) - (xx - kw//2).clamp(min=0)).to(x.dtype)
        result.append((total / (count_y[:, None] * count_x[None, :])).to(x.dtype))
    return result


def deterministic_cumsum(x, dim):
    """Fixed-order doubling scan, including deterministic CUDA backward.

    Native CUDA floating cumsum has no deterministic implementation in some
    supported PyTorch versions. This explicit scan trades speed for a declared
    reproducible summation order and requires no global determinism exception.
    """
    result=x;offset=1;length=x.shape[dim]
    while offset<length:
        if dim==-1:shifted=F.pad(result[...,:-offset],(offset,0))
        elif dim==-2:shifted=F.pad(result[...,:-offset,:],(0,0,offset,0))
        else:raise ValueError('Only spatial dimensions are supported')
        result=result+shifted;offset*=2
    return result


def part_pool(x):
    """Explicit disjoint spatial cells avoid nondeterministic adaptive pooling."""
    h,w=x.shape[-2:]
    if min(h,w)<PART_SIDE:raise ValueError('Support feature map is too small')
    cells=[x[...,y*h//PART_SIDE:(y+1)*h//PART_SIDE,z*w//PART_SIDE:(z+1)*w//PART_SIDE].mean((-2,-1))
           for y in range(PART_SIDE)for z in range(PART_SIDE)]
    return torch.stack(cells,-1).reshape(x.shape[0],x.shape[1],PART_SIDE,PART_SIDE)


class LocalRelationNet(nn.Module):
    def __init__(self, geometry=True):
        super().__init__(); self.geometry = bool(geometry)
        self.stem = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1, bias=False), ChannelNorm(16), nn.SiLU(), Block(16))
        self.mid = stage(64, 32, (1, 2))
        self.context = stage(128, 48, (1, 2, 4))
        self.deep = stage(192, 64, (1, 2))
        self.project = stage(64+32+48+64, 40, (1,))
        self.descriptor = nn.Conv2d(40, 32, 1)
        self.foreground = nn.Conv2d(32, 1, 1)
        self.detail = nn.Conv2d(16, 4, 1)
        # Per-anchor evidence channels: mean, partial, spread, best, coverage,
        # prototype cosine, query foreground, log width, log height.
        self.match_head = nn.Sequential(nn.Conv2d(9, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.SiLU(), nn.Conv2d(32, 5, 1))
        self.seg_head = nn.Sequential(nn.Conv2d(32+4, 24, 1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1, groups=24), nn.SiLU(), nn.Conv2d(24, 1, 1))
        self.register_buffer('sobel', torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])[None,None] / 8)
        self.register_buffer('laplace', torch.tensor([[0.,1,0],[1,-4,1],[0,1,0]])[None,None])
        nn.init.constant_(self.match_head[-1].bias[0], -4.6)
        nn.init.zeros_(self.match_head[-1].weight[1:]); nn.init.zeros_(self.match_head[-1].bias[1:])

    def encode(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 32:
            raise ValueError('Expected float RGB BCHW, dimensions >=32')
        h, w = rgb.shape[-2:]; ph, pw = (-h) % 8, (-w) % 8
        if ph or pw: rgb = F.pad(rgb, (0, pw, 0, ph), mode='reflect')
        gray = rgb.mean(1, keepdim=True)
        high = gray - F.avg_pool2d(gray, 5, 1, 2)
        contrast = (high / (.06 + F.avg_pool2d(high.square(), 5, 1, 2).clamp_min(1e-8).sqrt())).clamp(-3, 3) / 3
        target = torch.cat([contrast, F.conv2d(gray, self.sobel, padding=1)*4,
                            F.conv2d(gray, self.sobel.transpose(2,3), padding=1)*4,
                            F.conv2d(gray, self.laplace, padding=1)], 1)
        fine = self.stem(torch.cat([2*rgb-1, target], 1))
        phase = F.pixel_unshuffle(fine, 2); mid = self.mid(phase)
        context = self.context(F.pixel_unshuffle(mid, 2)); deep = self.deep(F.pixel_unshuffle(context, 2))
        fused = torch.cat([phase, mid, F.interpolate(context, mid.shape[-2:], mode='nearest'),
                           F.interpolate(deep, mid.shape[-2:], mode='nearest')], 1)
        raw = self.descriptor(self.project(fused))[:, :, :(h+1)//2, :(w+1)//2]
        return {'raw': raw, 'dense': F.normalize(raw, dim=1, eps=1e-6),
                'foreground_logits': self.foreground(raw), 'height': h, 'width': w,
                'detail_prediction': self.detail(fine)[:, :, :h, :w], 'detail_target': target[:, :, :h, :w]}

    def support_bank(self, encoded):
        raw=encoded['raw'];logits=encoded['foreground_logits'];b,_,h,w=raw.shape
        # The scene anchors denote tight object boxes, so reference part
        # coordinates must describe the predicted object, not its letterbox.
        # No GT enters this path. Bounds are a detached hard proposal; descriptor
        # and foreground-weight gradients remain active inside the crop.
        selected=logits[:,0].detach()>0
        xs=torch.arange(w,device=raw.device)[None].expand(b,-1)
        ys=torch.arange(h,device=raw.device)[None].expand(b,-1)
        anyx=selected.any(1);anyy=selected.any(2)
        x0=xs.masked_fill(~anyx,w).amin(1);x1=xs.masked_fill(~anyx,-1).amax(1)+1
        y0=ys.masked_fill(~anyy,h).amin(1);y1=ys.masked_fill(~anyy,-1).amax(1)+1
        # One synchronization for the complete batch, rather than four per item.
        bounds=torch.stack([x0,y0,x1,y1],1).cpu().tolist()
        parts=[];weights=[];prototypes=[];extents=[]
        for i,(left,top,right,bottom)in enumerate(bounds):
            if right<=left or bottom<=top:left,top,right,bottom=0,0,w,h
            else:
                if right-left<PART_SIDE:
                    left=max(0,min(w-PART_SIDE,(left+right-PART_SIDE)//2));right=left+PART_SIDE
                if bottom-top<PART_SIDE:
                    top=max(0,min(h-PART_SIDE,(top+bottom-PART_SIDE)//2));bottom=top+PART_SIDE
            crop=raw[i:i+1,:,top:bottom,left:right]
            weight=.05+.95*logits[i:i+1,:,top:bottom,left:right].sigmoid()
            pooled_weight=part_pool(weight)
            descriptors=part_pool(crop*weight)/pooled_weight.clamp_min(1e-6)
            parts.append(F.normalize(descriptors.flatten(2),dim=1,eps=1e-6))
            weights.append(pooled_weight.flatten(2).squeeze(1))
            prototypes.append(F.normalize((crop*weight).sum((-2,-1))/weight.sum((-2,-1)),dim=1,eps=1e-6))
            extents.append([left,top,right,bottom])
        return {'parts':torch.cat(parts),'weights':torch.cat(weights),'prototype':torch.cat(prototypes),
                'predicted_extent':torch.tensor(extents,device=raw.device,dtype=torch.int64)}

    def relation_evidence(self, bank, query):
        raw = query['dense']; b, c, h, w = raw.shape
        parts, weight = bank['parts'], bank['weights']
        if parts.shape[0] == 1 and b != 1:
            parts = parts.expand(b, -1, -1); weight = weight.expand(b, -1)
        if parts.shape[0] != b: raise ValueError('Support/query batch dimensions differ')
        correlation = torch.einsum('bcp,bchw->bphw', parts, raw)
        sizes = anchor_sizes(query['height'], query['width'], raw.device, raw.dtype)
        geometry_sizes = [(float(x)/2, float(y)/2) for x, y in sizes.detach().cpu().tolist()]
        proto = bank['prototype']
        if proto.shape[0] == 1 and b != 1: proto = proto.expand(b, -1)
        prototype = (raw * proto[:, :, None, None]).sum(1)
        qforeground = query['foreground_logits'].sigmoid().squeeze(1)
        if self.geometry:
            # One-pixel tolerance at descriptor resolution admits bounded
            # independent part deformation, without arbitrary part reordering.
            local = F.max_pool2d(correlation, 3, 1, 1)
            maximum = max(max(wh) for wh in geometry_sizes)
            padding = int(math.ceil(.5 * maximum)) + 2
            padded = F.pad(local, (padding,)*4)
            valid = F.pad(torch.ones((1,1,h,w), device=raw.device, dtype=raw.dtype), (padding,)*4)
        else:
            windows = [(max(1, 2*int(y/2)+1), max(1, 2*int(x/2)+1)) for x, y in geometry_sizes]
            pooled = integral_box_means((8*correlation).exp(), windows)
        features = []
        for anchor_index, (aw, ah) in enumerate(geometry_sizes):
            if self.geometry:
                samples, visible = [], []
                for p in range(PART_SIDE * PART_SIDE):
                    dx = int(round(((p % PART_SIDE + .5) / PART_SIDE - .5) * aw))
                    dy = int(round(((p // PART_SIDE + .5) / PART_SIDE - .5) * ah))
                    samples.append(padded[:, p, padding+dy:padding+dy+h, padding+dx:padding+dx+w])
                    visible.append(valid[:, 0, padding+dy:padding+dy+h, padding+dx:padding+dx+w])
                aligned = torch.stack(samples, 1); visibility = torch.stack(visible, 1)
            else:
                aligned = pooled[anchor_index].clamp_min(1e-8).log() / 8
                visibility = torch.ones((1, PART_SIDE*PART_SIDE, h, w), device=raw.device, dtype=raw.dtype)
            effective = weight[:, :, None, None] * visibility
            denom = effective.sum(1).clamp_min(1e-5)
            mean = (aligned * effective).sum(1) / denom
            variance = ((aligned-mean[:,None]).square() * effective).sum(1) / denom
            # A supported, highest-scoring half can survive partial occlusion.
            ranked = aligned + .15 * effective.clamp_min(1e-6).log()
            chosen = ranked.topk(PART_SIDE*PART_SIDE//2, dim=1).indices
            partial_weights = effective.expand(-1,-1,h,w).gather(1, chosen)
            partial = (aligned.gather(1,chosen)*partial_weights).sum(1) / partial_weights.sum(1).clamp_min(1e-5)
            best = aligned.masked_fill(effective <= .01, -1).amax(1)
            coverage = denom / weight.sum(1)[:,None,None].clamp_min(1e-5)
            logw = torch.full_like(mean, math.log(aw*2/min(query['height'], query['width'])))
            logh = torch.full_like(mean, math.log(ah*2/min(query['height'], query['width'])))
            features.append(torch.stack([mean, partial, (variance+1e-6).sqrt(), best, coverage,
                                          prototype, qforeground, logw, logh], 1))
        return torch.stack(features, 1), sizes

    def match_encoded(self, bank, query):
        evidence, sizes = self.relation_evidence(bank, query)
        b, anchors, channels, h, w = evidence.shape
        predictions = self.match_head(evidence.reshape(b*anchors, channels, h, w)).reshape(b, anchors, 5, h, w)
        summary = torch.cat([evidence[:,:,0].amax(1,keepdim=True), evidence[:,:,1].amax(1,keepdim=True),
                             evidence[:,0,5:7]], 1)
        mask = self.seg_head(torch.cat([query['raw'], summary], 1))
        return {'center_logits': predictions[:,:,0], 'box_deltas': predictions[:,:,1:],
                'mask_logits': mask, 'anchor_sizes': sizes, 'height': query['height'], 'width': query['width']}

    def forward(self, support, query):
        support_encoded = self.encode(support); query_encoded = self.encode(query)
        result = self.match_encoded(self.support_bank(support_encoded), query_encoded)
        result['support_encoded'] = support_encoded; result['query_encoded'] = query_encoded
        return result


def detection_targets(out, bbox, present):
    """Fixed anchor assignment; actual GT coordinates enter only this loss."""
    b, anchors, h, w = out['center_logits'].shape; device = bbox.device
    sizes = out['anchor_sizes']; boxwh = (bbox[:,2:]-bbox[:,:2]).clamp_min(1)
    center = (bbox[:,:2]+bbox[:,2:]) / 2
    intersection = torch.minimum(boxwh[:,None], sizes[None]).prod(-1)
    iou = intersection / (boxwh.prod(-1)[:,None] + sizes.prod(-1)[None] - intersection).clamp_min(1)
    best = iou.argmax(1)
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    centers = torch.stack([(x+.5)*2, (y+.5)*2], -1).to(bbox.dtype)
    # CenterNet-like Gaussian target. Exactly one positive pixel per target;
    # nearby cells receive reduced negative weights, not positive class labels.
    ix = (center[:,0]/2-.5).round().long().clamp(0,w-1)
    iy = (center[:,1]/2-.5).round().long().clamp(0,h-1)
    sigma = (boxwh.min(-1).values/12).clamp(1,4)
    distance = (x[None]-ix[:,None,None]).square() + (y[None]-iy[:,None,None]).square()
    gaussian = torch.exp(-distance/(2*sigma[:,None,None].square()))
    heat = out['center_logits'].new_zeros((b,anchors,h,w))
    heat[torch.arange(b,device=device),best] = gaussian * present[:,None,None]
    regression = torch.cat([(center[:,None,None]-centers[None]) / sizes[best][:,None,None],
                            (boxwh/sizes[best]).log()[:,None,None].expand(-1,h,w,-1)], -1)
    return heat, best, ix, iy, regression


def objective(out, batch):
    present = batch['present'].float(); boxes = batch['bbox'].float()
    target, assigned, ix, iy, regression = detection_targets(out, boxes, present)
    logits = out['center_logits']; probability = logits.sigmoid().clamp(1e-5,1-1e-5)
    positive = target == 1; negative = target < 1
    center_loss = (-(1-probability).square()*probability.log()*positive -
        probability.square()*(1-probability).log()*(1-target).pow(4)*negative).sum() / max(1, len(present))
    active = present > .5; rows = torch.arange(len(present),device=present.device)
    predicted = out['box_deltas'][rows,assigned,:,iy,ix]
    truth = regression[rows,iy,ix]
    regression_loss = F.smooth_l1_loss(predicted[active], truth[active]) if active.any() else predicted.sum()*0
    target_mask = F.interpolate(batch['target_mask'].float(), out['mask_logits'].shape[-2:], mode='area')
    mask_bce = F.binary_cross_entropy_with_logits(out['mask_logits'],target_mask,reduction='none')
    # Foreground weighting is bounded so tiny masks do not dominate gradients.
    fg_weight = (1/(target_mask.mean((-2,-1),keepdim=True)+.03)).clamp(1,12)
    mask_loss = (mask_bce*(1+target_mask*(fg_weight-1))).mean()
    prediction = out['mask_logits'].sigmoid()
    dice = 1-(2*(prediction*target_mask).sum((-2,-1))+1)/(prediction.sum((-2,-1))+target_mask.sum((-2,-1))+1)
    support_mask = F.interpolate(batch['support_mask'].float(),out['support_encoded']['foreground_logits'].shape[-2:],mode='area')
    support_loss = F.binary_cross_entropy_with_logits(out['support_encoded']['foreground_logits'],support_mask)
    if 'query_foreground_mask' in batch:
        query_fg = F.interpolate(batch['query_foreground_mask'].float(),out['query_encoded']['foreground_logits'].shape[-2:],mode='area')
        support_loss = support_loss + F.binary_cross_entropy_with_logits(out['query_encoded']['foreground_logits'],query_fg)
    detail_loss = sum(F.smooth_l1_loss(out[key]['detail_prediction'],out[key]['detail_target'])
                      for key in ('support_encoded','query_encoded')) / 2
    loss = center_loss + 2*regression_loss + mask_loss + .5*dice.mean() + .25*support_loss + .1*detail_loss
    scores = logits.flatten(1).amax(1)
    stats = {'loss':loss,'center_loss':center_loss,'box_loss':regression_loss,'target_mask_loss':mask_loss,
             'target_mask_dice_loss':dice.mean(),'support_foreground_loss':support_loss,'detail_loss':detail_loss,
             'positive_fraction':present.mean(),'maximum_logit_mean':scores.mean()}
    return loss, {k:float(v.detach()) for k,v in stats.items()}


def decoded_boxes(out):
    b, anchors, h, w = out['center_logits'].shape
    yy, xx = torch.meshgrid(torch.arange(h,device=out['center_logits'].device),
                           torch.arange(w,device=out['center_logits'].device),indexing='ij')
    centers = torch.stack([(xx+.5)*2,(yy+.5)*2],0)[None,None].to(out['box_deltas'].dtype)
    sizes = out['anchor_sizes'][None,:,:,None,None]
    delta = out['box_deltas']; xy = centers + delta[:,:,:2].clamp(-2,2)*sizes
    wh = delta[:,:,2:].clamp(-2,2).exp()*sizes
    return torch.cat([xy-wh/2,xy+wh/2],2).permute(0,1,3,4,2).reshape(b,-1,4)
