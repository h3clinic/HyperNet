"""Synthetic-only ROI quality and final-descriptor correspondence objectives."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .model import bilinear_sample, roi_points, pyramid_indices, MODEL_CONFIG
from .proposals import grid_boxes, box_iou

TRAINING_CONFIG = {'rois_per_episode': 64, 'reference_slot_jitters': 16,
    'other_object_jitters': 4, 'target_match_iou': .5,
    'quality_target': 'IoU if target present and IoU>=0.5, otherwise0',
    'roi_loss_weight': 1., 'correspondence_loss_weight': .2, 'foreground_loss_weight': .25,
    'correspondence_candidate_grid_side': 16, 'correspondence_sigma_pixels': 2.,
    'point_target': 'Gaussian distribution around visible canonical counterpart, otherwise dustbin',
    'correspondence_scene_filter': 'Positive scenes only; global type absence never labels all shared local parts unmatched',
    'correspondence_negative_filter': 'Exclude256-grid sites within2pixels of non-target objects; canonical sites plus target/background grid remain',
    'foreground_use': 'Auxiliary predicted weighting only; no ground-truth mask in model forward'}
ARRAY_KEYS = ('support', 'query', 'support_mask', 'query_target_mask', 'query_object_mask',
    'boxes', 'target_box', 'positive', 'support_points', 'query_points',
    'support_point_valid', 'query_point_valid', 'point_valid', 'rois', 'quality_targets')


def _jitter(box, rng, height, width, amount=.3):
    size = np.maximum(box[2:]-box[:2], 2); center = (box[2:]+box[:2])/2
    center += rng.uniform(-amount, amount, 2)*size
    size *= np.exp(rng.uniform(-amount, amount, 2))
    result = np.concatenate([center-size/2, center+size/2])
    result[(0, 2),] = np.clip(result[(0, 2),], 0, width)
    result[(1, 3),] = np.clip(result[(1, 3),], 0, height)
    result[2:] = np.maximum(result[2:], result[:2]+1)
    return result.astype(np.float32)


def training_rois(row, index=None, seed=1901, count=64):
    if count < 32: raise ValueError('At least32 ROIs required for declared mixture')
    if index is None: index = row['metadata']['index']
    # Independent stream from procedural rendering. Same deterministic policy
    # for present/absent scenes: the first object may be target or replacement.
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(index), 1919]))
    h, w = row['query'].shape[:2]; objects = np.asarray(row['boxes'], np.float32)
    result = []
    for j, box in enumerate(objects):
        number = 16 if j == 0 else 4
        result.append(box)
        result.extend(_jitter(box, rng, h, w, amount=.12 if k < number//2 else .38)
                      for k in range(number-1))
    grid = grid_boxes(h, w)
    needed = count-len(result)
    if needed < 0: raise ValueError('Too many scene objects for fixed ROI budget')
    result.extend(grid[rng.choice(len(grid), needed, replace=False)])
    return np.asarray(result, np.float32)


def collate(rows):
    data = {}
    for key in ('support', 'query'):
        data[key] = torch.from_numpy(np.stack([r[key] for r in rows])).permute(0, 3, 1, 2).float()/255
    for key in ('support_mask', 'query_target_mask', 'query_object_mask'):
        data[key] = torch.from_numpy(np.stack([r[key] for r in rows])).float()[:, None]
    for key in ('boxes', 'target_box', 'support_points', 'query_points'):
        data[key] = torch.from_numpy(np.stack([r[key] for r in rows])).float()
    for key in ('positive', 'support_point_valid', 'query_point_valid', 'point_valid'):
        data[key] = torch.from_numpy(np.asarray([r[key] for r in rows], bool))
    rois = [training_rois(r, seed=int(r['metadata'].get('seed', 1901))) for r in rows]
    qualities = []
    for r, region in zip(rows, rois):
        overlap = box_iou(region, np.asarray(r['target_box']).reshape(1, 4))[:, 0]
        qualities.append(np.where(bool(r['positive']) & (overlap >= .5), overlap, 0).astype(np.float32))
    data['rois'] = torch.from_numpy(np.stack(rois))
    data['quality_targets'] = torch.from_numpy(np.stack(qualities))
    data['metadata'] = [r['metadata'] for r in rows]
    return data


def _descriptor_points(encoded, points):
    raw = bilinear_sample(encoded['raw'], points)
    return torch.cat([F.normalize(raw[:, :32], dim=1, eps=1e-6),
                      F.normalize(raw[:, 32:], dim=1, eps=1e-6)], 1)/math.sqrt(2)


def _point_direction(model, source, target, source_points, target_points, source_valid, target_valid, target_scale=1., excluded_grid_mask=None):
    """Bounded InfoNCE candidate set: all canonical points plus256 grid sites.

    These use the exact final ROI descriptors, not an earlier reconstruction
    branch. Nearby candidates receive soft positive mass; invalid canonical
    candidates are masked. Missing counterparts map to a learned dustbin.
    """
    b, k = source_valid.shape
    full = target_points.new_tensor([0, 0, target['width'], target['height']])[None, None].expand(b, -1, -1)
    grid = roi_points(full, side=16)[:, 0]
    locations = torch.cat([target_points, grid], 1)
    grid_valid = torch.ones((b, 256), device=target_valid.device, dtype=torch.bool)
    if excluded_grid_mask is not None:
        grid_valid = bilinear_sample(excluded_grid_mask, grid, stride=1., origin=0.)[:, 0] < .1
    candidate_valid = torch.cat([target_valid, grid_valid], 1)
    a = _descriptor_points(source, source_points)
    z = _descriptor_points(target, locations)
    similarity = torch.einsum('bck,bcn->bkn', a, z)/model.temperature
    similarity = similarity.masked_fill(~candidate_valid[:, None], -1e4)
    dust = (model.dustbin+candidate_valid.sum(-1).float().log())[:, None, None].expand(b, k, 1)
    logp = torch.cat([similarity, dust], -1).log_softmax(-1)
    distance = (target_points[:, :, None]-locations[:, None]).square().sum(-1)
    sigma = 2.*target_scale
    soft = torch.exp(-distance/(2*sigma*sigma))*candidate_valid[:, None]
    # Avoid negligible tails turning known nonmatches into positive labels.
    soft = soft.masked_fill(distance > (3*sigma)**2, 0)
    soft = soft/soft.sum(-1, keepdim=True).clamp_min(1e-8)
    counterpart = target_valid[:, :, None]
    target_distribution = torch.cat([soft*counterpart, (~target_valid)[:, :, None].to(soft.dtype)], -1)
    value = -(target_distribution*logp).sum(-1)
    weight = source_valid.to(value.dtype)
    count = weight.sum().clamp_min(1)
    loss = (value*weight).sum()/count
    predicted = logp.argmax(-1)
    correct = torch.where(target_valid,
        (predicted < soft.shape[-1]) & (soft.gather(-1, predicted.clamp(max=soft.shape[-1]-1)[..., None]).squeeze(-1) > 0),
        predicted == soft.shape[-1])
    accuracy = (correct*source_valid).sum().float()/count
    return loss, accuracy, int(source_valid.sum().detach())


def correspondence_loss(model, output, batch):
    levels = pyramid_indices(batch['boxes'][:, :1])[:, 0]
    terms = []; accuracy = []; counts = []
    for i, scale in enumerate(MODEL_CONFIG['query_pyramid_scales']):
        # Absent scenes contain near graphs sharing valid local parts. Whole-type
        # absence does not mean every local descriptor has no counterpart.
        selected = (levels == i) & batch['positive']
        if not bool(selected.any()): continue
        def subset(encoded):
            return {key: value[selected] if torch.is_tensor(value) else value for key, value in encoded.items()}
        support = subset(output['support_encoded']); query = subset(output['query_encoded']['levels'][i])
        sp = batch['support_points'][selected]; qp = (batch['query_points'][selected]+.5)*scale-.5
        sv = batch['support_point_valid'][selected]; qv = batch['query_point_valid'][selected]
        # Other graph instances may contain the same valid local edge/node.
        # Excluding their grid sites avoids an arbitrary instance-identity loss;
        # whole-graph negative supervision remains in the ROI objective.
        other = ((batch['query_object_mask'][selected] > .1) & (batch['query_target_mask'][selected] < .1)).float()
        other = F.max_pool2d(other, 5, 1, 2)
        if scale == .5: other = F.max_pool2d(other, 2, 2)
        elif scale == 2.: other = other.repeat_interleave(2, -2).repeat_interleave(2, -1)
        for values in ((support, query, sp, qp, sv, qv, scale, other), (query, support, qp, sp, qv, sv, 1., None)):
            loss, acc, count = _point_direction(model, *values)
            if count: terms.append(loss*count); accuracy.append(acc*count); counts.append(count)
    if not counts: return output['scores'].sum()*0, output['scores'].new_tensor(0.)
    return sum(terms)/sum(counts), sum(accuracy)/sum(counts)


def foreground_loss(logits, target):
    # Area targets agree with descriptor stride and retain thin-part occupancy.
    target = F.avg_pool2d(target, 2, 2)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
    positive = target; negative = 1-target
    positive_loss = (F.softplus(-logits)*positive).flatten(1).sum(1)/positive.flatten(1).sum(1).clamp_min(1)
    negative_loss = (F.softplus(logits)*negative).flatten(1).sum(1)/negative.flatten(1).sum(1).clamp_min(1)
    return .5*(positive_loss+negative_loss).mean()


def objective(model, batch, rois=None):
    if rois is None: rois = batch['rois']
    if rois.shape != batch['rois'].shape or not torch.equal(rois, batch['rois']):
        raise ValueError('Objective expects the exact collated ROIs bound to quality targets')
    output = model(batch['support'], batch['query'], rois)
    score = output['scores']; target = batch['quality_targets']
    element = F.binary_cross_entropy_with_logits(score, target, reduction='none')
    positive = target > 0; negative = ~positive
    pos_count = positive.sum(1); neg_count = negative.sum(1)
    pos = (element*positive).sum(1)/pos_count.clamp_min(1)
    neg = (element*negative).sum(1)/neg_count.clamp_min(1)
    roi_loss = torch.where(pos_count > 0, .5*(pos+neg), neg).mean()
    point_loss, point_accuracy = correspondence_loss(model, output, batch)
    fg_terms = [foreground_loss(output['support_encoded']['foreground_logits'], batch['support_mask'])]
    for level, scale in zip(output['query_encoded']['levels'], MODEL_CONFIG['query_pyramid_scales']):
        mask = batch['query_object_mask']
        if scale == .5: mask = F.avg_pool2d(mask, 2, 2)
        elif scale == 2.: mask = mask.repeat_interleave(2, -2).repeat_interleave(2, -1)
        fg_terms.append(foreground_loss(level['foreground_logits'], mask))
    fg_loss = sum(fg_terms)/len(fg_terms)
    loss = roi_loss+.2*point_loss+.25*fg_loss
    stats = {'loss': loss, 'roi_quality_loss': roi_loss, 'point_loss': point_loss,
             'foreground_loss': fg_loss, 'point_candidate_accuracy': point_accuracy,
             'positive_roi_mean_score': (score*positive).sum()/positive.sum().clamp_min(1),
             'negative_roi_mean_score': (score*negative).sum()/negative.sum().clamp_min(1),
             'positive_rois': positive.sum(), 'temperature': model.temperature,
             'dustbin_logit': model.dustbin}
    return loss, {key: float(value.detach()) for key, value in stats.items()}
