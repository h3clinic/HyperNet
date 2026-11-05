"""Fixed one-image Gravity Spy PNG preprocessing, without labels or learned weights.

The verified renderer uses viridis and log-frequency axes. Palette inversion
recovers displayed relative intensity, NOT exact Q-transform or strain values.
Only 800x600 images with this plotting convention are accepted by preprocess.
"""
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from matplotlib import colormaps

LEGACY_CROP=(105,66,671,532)
# Plot geometry verified on36 already-used 1s/4s PNGs and exact-palette references.
# Coordinates are half-open. Some later PNGs smooth RGB between palette entries;
# small inward tick marks can remain and are mapped near minimum energy.
PLOT_CROP=(102,62,673,539)
COLORBAR_CROP=(703,62,709,539)
PALETTE=np.rint(colormaps['viridis'](np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
_TREE=cKDTree(PALETTE.astype(np.float32))
# 32MiB, allocated once. Encode twice the palette index so duplicate RGB bins
# can map to their midpoint without a systematic endpoint choice.
_LUT=np.full(1<<24,-1,dtype=np.int16)
_BAD=np.zeros(1<<24,dtype=np.uint8)  # bit0: distance>3; bit1: distance>12
_codes=(PALETTE[:,0].astype(np.uint32)<<16)|(PALETTE[:,1].astype(np.uint32)<<8)|PALETTE[:,2]
for _code in np.unique(_codes):
    _LUT[_code]=int(np.rint(2*np.flatnonzero(_codes==_code).mean()))


def _rgb(raw):
    if isinstance(raw,Image.Image):
        raw=np.asarray(raw.convert('RGB'))
    a=np.asarray(raw)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[-1]!=3:
        raise ValueError('Expected uint8 RGB PNG pixels, not grayscale or normalized arrays')
    return a


def native_rgb(raw, crop='full'):
    a=_rgb(raw)
    if a.shape!=(600,800,3):
        raise ValueError('Unknown plotting layout; expected800x600 RGB, do not silently resize')
    box=LEGACY_CROP if crop=='legacy'else PLOT_CROP
    if crop not in {'legacy','full','low'}:raise ValueError(crop)
    x1,y1,x2,y2=box
    a=a[y1:y2,x1:x2]
    if crop=='low':a=a[(2*a.shape[0])//3:]
    return a


def validate_renderer(raw):
    """Verify an independent legend, never reject by target heatmap appearance.

    The renderer's middle colorbar strip is independent of the glitch shape.
    It validates only the plotting convention and is never passed to a model.
    Heatmap RGB smoothing can legitimately move colors far from the palette.
    """
    a=_rgb(raw)
    if a.shape!=(600,800,3):raise ValueError('Unknown plotting layout: expected800x600 RGB')
    x1,y1,x2,y2=COLORBAR_CROP
    bar=np.median(a[y1:y2,x1:x2],axis=1).astype(np.float32)
    distances,indices=_TREE.query(bar,workers=1)
    agreement=float(np.mean(distances<=3))
    correlation=float(np.corrcoef(np.arange(len(indices)),indices)[0,1])if np.std(indices)>0 else 0.
    span=float((indices.max()-indices.min())/255)
    if agreement<.9 or correlation>-.99 or span<.9:
        raise ValueError(f'Unverified renderer/colorbar: agreement={agreement:.3f}, correlation={correlation:.3f}, span={span:.3f}')
    return dict(colorbar_crop=list(COLORBAR_CROP),palette_agreement_fraction=agreement,
                colorbar_order_correlation=correlation,colorbar_palette_span=span,
                criterion='>=90% legend pixels within RGBdistance3, decreasing index correlation<=-.99, palette span>=.9',
                colorbar_enters_model=False,heatmap_quality_used_for_exclusion=False)


def to_energy(rgb,return_metadata=False):
    """Nearest viridis inversion with cached24-bit RGB lookup; no per-image fit."""
    a=_rgb(rgb)
    codes=(a[...,0].astype(np.uint32)<<16)|(a[...,1].astype(np.uint32)<<8)|a[...,2]
    indices=_LUT[codes]
    unseen=indices<0
    if unseen.any():
        keys=np.unique(codes[unseen])
        colors=np.stack([(keys>>16)&255,(keys>>8)&255,keys&255],axis=-1).astype(np.float32)
        distances,nearest=_TREE.query(colors,workers=1)
        _LUT[keys]=(2*nearest).astype(np.int16)
        _BAD[keys]=(distances>3).astype(np.uint8)+2*(distances>12).astype(np.uint8)
        indices=_LUT[codes]
    energy=indices.astype(np.float32)/510
    flags=_BAD[codes]
    bad_fraction=float((flags&1).mean())
    far_fraction=float(((flags&2)>0).mean())
    if return_metadata:
        return energy,dict(bad_palette_fraction=bad_fraction,palette_rgb_distance_limit=3,
                          far_palette_fraction=far_fraction,validation_rgb_distance_limit=12,
                          validation_required_fraction=.9,
                          inversion_is_approximate=True,
                          colorbar_units='relative displayed palette index',normalization='linear index /255; no per-image fit')
    return energy


def preprocess(rawRGB,mode,size=128,return_metadata=False):
    """Return float32 size×size×3; modes legacy, energy_full, energy_low.

    Energy modes invert native pixels BEFORE their only area resize. Low mode
    uses the native bottom third (~10–59Hz), not a crop of a small input. The
    caller must retain full-image context if needed; this function never fuses
    temporal views or adds another positive source image.
    """
    size=int(size)
    if size<8:raise ValueError('Output size must be>=8')
    if mode=='legacy':
        native=native_rgb(rawRGB,'legacy')
        gray=np.asarray(Image.fromarray(native).convert('L'))
        small=cv2.resize(gray,(170,140),interpolation=cv2.INTER_AREA)
        gray=cv2.resize(small,(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
        meta=dict(mode=mode,crop=list(LEGACY_CROP),legacy_intermediate=[170,140],size=size)
    elif mode in {'energy_full','energy_low'}:
        renderer=validate_renderer(rawRGB)
        native=native_rgb(rawRGB,'low'if mode=='energy_low'else 'full')
        energy,meta=to_energy(native,True)
        gray=cv2.resize(energy,(size,size),interpolation=cv2.INTER_AREA)
        meta.update(mode=mode,crop=list(PLOT_CROP),native_shape=list(native.shape),size=size,
                    frequency_extent_approx_hz=[10.1,59]if mode=='energy_low'else[10.1,2000],
                    renderer=renderer,heatmap_quality_is_diagnostic_only=True)
    else:raise ValueError(mode)
    result=np.repeat(np.clip(gray,0,1)[...,None],3,axis=-1).astype(np.float32)
    return (result,meta)if return_metadata else result
