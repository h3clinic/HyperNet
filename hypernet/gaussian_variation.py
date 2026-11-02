"""Bounded Gaussian-driven variations of one reference's shape and detail.

Independent normal fields are Gaussian-smoothed, combined and tapered to zero
at the image boundary. Per-image scaling enforces a displacement and derivative
bound. The resulting bounded fields are no longer an unconditioned Gaussian
distribution. They are an explicit augmentation prior, not a human-vision model.

Only input pixels are resampled. The optional pattern branch multiplies existing
high-frequency detail by a smooth gain; it supplies no external texture/image.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

SIGMAS_PX = (3., 6.)
TARGET_VECTOR_RMS_PX = .75
MAX_VECTOR_DISPLACEMENT_PX = 2.
MAX_AXIS_DERIVATIVE = .35
MAX_DETAIL_GAIN_DEVIATION = .18


def gaussian_variation_recipe():
    return {
        'prior': 'bounded Gaussian-smoothed displacement; correlated existing-detail gain',
        'gaussian_sigmas_px': list(SIGMAS_PX), 'field_mixture_weights': [.35, .65],
        'target_vector_rms_px_before_bounds': TARGET_VECTOR_RMS_PX,
        'maximum_vector_displacement_px': MAX_VECTOR_DISPLACEMENT_PX,
        'maximum_absolute_displacement_axis_derivative': MAX_AXIS_DERIVATIVE,
        'boundary_taper': 'sin(pi*x/(W-1))*sin(pi*y/(H-1)); exact zero boundary',
        'normalization': 'per-image component centering, vector RMS normalization, global safety scaling',
        'pattern_gain': '1 + 0.18*tanh(Gaussian-smoothed unit-RMS normal field)',
        'pattern_gain_sigma_px': 3., 'existing_detail_sigma_px': 1.,
        'resampling': 'bilinear, pixel-center coordinates, align_corners=False',
        'external_images': 0, 'external_learned_parameters': 0,
        'interpretation': 'synthetic local variation from one image; no new category observations',
    }


def _smooth(value, sigma):
    radius = int(math.ceil(3*sigma))
    offsets = torch.arange(-radius, radius+1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-.5*(offsets/sigma).square())
    kernel = kernel/kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.view(1,1,1,-1).expand(channels,1,1,-1)
    vertical = kernel.view(1,1,-1,1).expand(channels,1,-1,1)
    result = F.conv2d(F.pad(value,(radius,radius,0,0),mode='reflect'),
                      horizontal,groups=channels)
    return F.conv2d(F.pad(result,(0,0,radius,radius),mode='reflect'),
                    vertical,groups=channels)


def _axis_derivative_max(field):
    dx=(field[:,:,:,1:]-field[:,:,:,:-1]).abs().flatten(1).amax(1)
    dy=(field[:,:,1:,:]-field[:,:,:-1,:]).abs().flatten(1).amax(1)
    return torch.maximum(dx,dy)


def sampling_jacobian_determinant(field):
    """Forward-cell determinant of the input sampling map p -> p+u(p).

The global partial-derivative bound applies across each bilinear cell, not just
    these reported corners. With all four |du_i/dp_j| <= .35, det(I+Du) >= .3.
"""
    if field.ndim!=4 or field.shape[1]!=2:
        raise ValueError('Expected N x 2 x H x W displacement field')
    dx=field[:,:,:-1,1:]-field[:,:,:-1,:-1]
    dy=field[:,:,1:,:-1]-field[:,:,:-1,:-1]
    return (1+dx[:,0])*(1+dy[:,1])-dy[:,0]*dx[:,1]


def _fields(field, gain, strength, kind):
    return {
        'displacement_px':field,
        'displacement_vector_rms_px':field.square().sum(1).mean((1,2)).sqrt(),
        'displacement_max_norm_px':field.square().sum(1).sqrt().flatten(1).amax(1),
        'maximum_axis_derivative':_axis_derivative_max(field),
        'sampling_jacobian_det':sampling_jacobian_determinant(field),
        'pattern_gain':gain,
        'recipe':{**gaussian_variation_recipe(),'kind':kind,'strength':float(strength)},
    }


def gaussian_variation(patches, generator, kind='shape', strength=1., return_fields=False):
    """Vary RGB patches using only their existing pixels.

    Inputs are N x 3 x 32 x 32 float32/float64 values in [0,1]. Interpolation
    roundoff within four dtype epsilons is clipped without mutating the input.
    Return a tensor,
    or (tensor, fields) when requested. ``none`` and zero strength are exact
    identity paths for valid RGB and consume no randomness. Shape and shape_pattern consume
    identical geometry draws; only the latter draws the extra detail-gain field.

    The geometric bounds always hold; a requested RMS can be reduced by the
    safety bounds. Targets for a same-geometry restoration loss should be
    measured on this returned clean variation, before nuisance corruptions.
    """
    if not isinstance(patches,torch.Tensor) or patches.ndim!=4 or patches.shape[1:]!=(3,32,32) or len(patches)==0:
        raise ValueError('Expected a nonempty N x 3 x 32 x 32 RGB tensor')
    if patches.dtype not in (torch.float32,torch.float64):
        raise ValueError('Patches must use float32 or float64')
    tolerance=4*torch.finfo(patches.dtype).eps
    if not torch.isfinite(patches).all() or patches.min() < -tolerance or patches.max() > 1+tolerance:
        raise ValueError('Patches must be finite RGB values in [0,1]')
    patches=patches.clamp(0.,1.)
    if kind not in ('none','shape','shape_pattern'):
        raise ValueError('kind must be none, shape, or shape_pattern')
    if isinstance(strength,bool) or not math.isfinite(strength) or strength<0:
        raise ValueError('strength must be finite and nonnegative')
    n,_,h,w=patches.shape
    field=patches.new_zeros((n,2,h,w))
    gain=patches.new_ones((n,1,h,w))
    if strength==0 or kind=='none':
        result=patches.clone()
        return (result,_fields(field,gain,strength,kind)) if return_fields else result

    parts=[]
    for sigma in SIGMAS_PX:
        noise=torch.randn((n,2,h,w),generator=generator,device=patches.device,dtype=patches.dtype)
        smooth=_smooth(noise,sigma)
        smooth=smooth-smooth.mean((2,3),keepdim=True)
        smooth=smooth/(smooth.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-8))
        parts.append(smooth)
    field=.35*parts[0]+.65*parts[1]
    x=torch.linspace(0,math.pi,w,device=patches.device,dtype=patches.dtype).sin()
    y=torch.linspace(0,math.pi,h,device=patches.device,dtype=patches.dtype).sin()
    # Set endpoints explicitly: sin(pi) need not round to exact zero.
    x[0]=x[-1]=0.;y[0]=y[-1]=0.
    field=field*(y[:,None]*x[None,:])[None,None]
    rms=field.square().sum(1).mean((1,2)).sqrt().clamp_min(1e-8)
    field=field*(TARGET_VECTOR_RMS_PX*strength/rms)[:,None,None,None]
    max_norm=field.square().sum(1).sqrt().flatten(1).amax(1).clamp_min(1e-8)
    max_derivative=_axis_derivative_max(field).clamp_min(1e-8)
    scale=torch.minimum(torch.ones_like(max_norm),MAX_VECTOR_DISPLACEMENT_PX/max_norm)
    scale=torch.minimum(scale,MAX_AXIS_DERIVATIVE/max_derivative)
    field=field*scale[:,None,None,None]
    yy,xx=torch.meshgrid(torch.arange(h,device=patches.device,dtype=patches.dtype),
                        torch.arange(w,device=patches.device,dtype=patches.dtype),indexing='ij')
    base=torch.stack([(xx+.5)*2/w-1,(yy+.5)*2/h-1],dim=-1)
    offset=field.permute(0,2,3,1)*field.new_tensor([2/w,2/h])
    result=F.grid_sample(patches,base[None]+offset,mode='bilinear',
                         padding_mode='border',align_corners=False)
    if kind=='shape_pattern':
        noise=torch.randn((n,1,h,w),generator=generator,device=patches.device,dtype=patches.dtype)
        smooth=_smooth(noise,3.)
        smooth=smooth-smooth.mean((2,3),keepdim=True)
        smooth=smooth/smooth.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-8)
        gain=1+MAX_DETAIL_GAIN_DEVIATION*torch.tanh(smooth*strength)
        detail=result-_smooth(result,1.)
        result=result+(gain-1)*detail
    # Exact constant-color preservation avoids interpolation/blur roundoff being
    # mistaken for generated texture. No input tensor is modified in place.
    constant=(patches.amax((2,3))==patches.amin((2,3))).all(1)
    result=torch.where(constant[:,None,None,None],patches,result).clamp(0.,1.)
    return (result,_fields(field,gain,strength,kind)) if return_fields else result
