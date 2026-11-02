"""Named parametric surfaces with geometric detail and native-resolution rendering.

No image assets, learned models, natural images, Gaussian image filtering, or
post-render resizing are used. Gaussian functions displace surface vertices;
finite-difference normals are recomputed on that displaced surface. The renderer
uses an orthographic camera, parametric triangulation, and a depth buffer.
It is a small deterministic graphics simulator, not a ray tracer.

The signed-power superquadric construction follows Alan Barr (1981),
https://authors.library.caltech.edu/records/rtr62-f2882 . The torus-knot centerline
is the standard (2,3) trigonometric parameterization; the other families are
explicit engineered combinations documented in FORMULAS, not claimed new
mathematical discoveries. All images and geometry are generated here.
"""
from functools import lru_cache
import hashlib
import json
import math

import numpy as np

REVISION = 'mathematical_surfaces_v10_1'
FAMILY_NAMES = ('superellipsoid', 'supertoroid', 'lobed_vase', 'trefoil_tube',
                'corrugated_torus', 'saddle_shell', 'radial_rosette', 'multipart_spindle')
FAMILY_TO_INDEX = {family: index for index, family in enumerate(FAMILY_NAMES)}
FAMILY_LABELS = {family: family.replace('_', ' ').title() for family in FAMILY_NAMES}
FORMULAS = {
    'superellipsoid': 'S=(a C(v,e1) C(u,e2), b C(v,e1) S(u,e2), c S(v,e1)); signed powers C,S; v in [-pi/2,pi/2]',
    'supertoroid': 'S=((R+r C(v,e2)) C(u,e1), (R+r C(v,e2)) S(u,e1), h S(v,e2))',
    'lobed_vase': 'r(t,u)=[.48+.16 cos(2pi t)+.15 exp(-((t-.18)/.2)^2)] [1+a cos(k u) sin(pi t)^2]; z=2t-1; toroidal rim and foot',
    'trefoil_tube': 'c(u)=((R+r cos(3u)) cos(2u),(R+r cos(3u)) sin(2u),r sin(3u)); S=c+rho [cos(v) N(u)+sin(v) B(u)]',
    'corrugated_torus': 'rho(u,v)=r [1+a cos(k u+2v)]; S=((R+rho cos(v)) cos(u),(R+rho cos(v)) sin(u),rho sin(v))',
    'saddle_shell': 'S=(a x,b y,c(x^2-y^2)+d sin(3x) cos(2y)); x,y in [-1,1]; two-sided open shell',
    'radial_rosette': 'r=1+a cos(k u) cos(v)^2+b cos(m v); S=r(cos(v)cos(u),cos(v)sin(u),h sin(v))',
    'multipart_spindle': 'Lathed r(t)=r0+b exp(-(t/w)^2), z=h t, t in [-1,1], with three intersecting toroidal collars; separate components, not a Boolean union',
}


def _seed(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**63:
        raise ValueError(name+' must be an integer in [0,2**63)')
    return int(value)


def _unit(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def _spow(value, exponent):
    return np.sign(value)*np.abs(value)**exponent


def _superangle(angle, exponent, periodic=True):
    """Redistribute samples near signed-power cusps, without altering the surface.

    Uniform angles leave wide gaps where e<1 makes d(sin(t)^e)/dt large.
    Inverse signed-power angular reparameterization keeps spatial steps bounded
    near those axes. This is geometric sampling, not image-space filling.
    """
    warped = np.arctan2(_spow(np.sin(angle), 1/exponent),
                       _spow(np.cos(angle), 1/exponent))
    return np.mod(warped, 2*np.pi) if periodic else warped


def _normals(points, periodic_u, periodic_v):
    du = (np.roll(points, -1, axis=0)-np.roll(points, 1, axis=0) if periodic_u
          else np.gradient(points, axis=0))
    dv = (np.roll(points, -1, axis=1)-np.roll(points, 1, axis=1) if periodic_v
          else np.gradient(points, axis=1))
    normal = _unit(np.cross(du, dv))
    if np.median(np.sum(normal*points, axis=-1)) < 0:
        normal = -normal
    return normal


def _component(points, u, v, periodic=(True, True), name='body'):
    return {'points': np.asarray(points, dtype=np.float32), 'u': u, 'v': v,
            'periodic': periodic, 'name': name}


@lru_cache(maxsize=24)
def _grid_triangles(count_u, count_v, periodic_u, periodic_v):
    uu, vv = np.meshgrid(np.arange(count_u if periodic_u else count_u-1),
                         np.arange(count_v if periodic_v else count_v-1), indexing='ij')
    a = uu*count_v+vv
    b = ((uu+1)%count_u)*count_v+vv
    c = ((uu+1)%count_u)*count_v+(vv+1)%count_v
    d = uu*count_v+(vv+1)%count_v
    return np.concatenate((np.stack((a, b, c), -1).reshape(-1, 3),
                           np.stack((a, c, d), -1).reshape(-1, 3))).astype(np.int32)


def _rasterize_triangles(xyz, normals, triangles, component_ids, px, py, size):
    """Exact pixel-center triangle coverage and barycentric depth/normals.

    Triangle interpolation is part of rasterizing the mathematical mesh; no
    pixel colors or silhouettes are smoothed, splatted, blurred, or resized.
    """
    x, y = px[triangles], py[triangles]
    left = np.maximum(np.ceil(x.min(1)-.5).astype(np.int32), 0)
    right = np.minimum(np.floor(x.max(1)-.5).astype(np.int32), size-1)
    top = np.maximum(np.ceil(y.min(1)-.5).astype(np.int32), 0)
    bottom = np.minimum(np.floor(y.max(1)-.5).astype(np.int32), size-1)
    area = (x[:, 1]-x[:, 0])*(y[:, 2]-y[:, 0])-(y[:, 1]-y[:, 0])*(x[:, 2]-x[:, 0])
    width, height = np.maximum(0, right-left+1), np.maximum(0, bottom-top+1)
    count = width*height
    count[np.abs(area) < 1e-10] = 0
    face = np.repeat(np.arange(len(triangles), dtype=np.int32), count)
    offsets = np.arange(len(face))-np.repeat(np.cumsum(count)-count, count)
    xx, yy = left[face]+offsets%width[face], top[face]+offsets//width[face]
    dx, dy = xx+.5-x[face, 0], yy+.5-y[face, 0]
    w1 = (dx*(y[face, 2]-y[face, 0])-dy*(x[face, 2]-x[face, 0]))/area[face]
    w2 = ((x[face, 1]-x[face, 0])*dy-(y[face, 1]-y[face, 0])*dx)/area[face]
    w0 = 1-w1-w2
    inside = (w0 >= -1e-6)&(w1 >= -1e-6)&(w2 >= -1e-6)
    face, xx, yy = face[inside], xx[inside], yy[inside]
    weights = np.column_stack((w0[inside], w1[inside], w2[inside])).astype(np.float32)
    vertices = triangles[face]
    depths = (xyz[vertices, 2]*weights).sum(1)
    flat = yy*size+xx
    zbuffer = np.full(size*size, -np.inf, dtype=np.float32)
    np.maximum.at(zbuffer, flat, depths)
    front = depths == zbuffer[flat]
    selected = np.full(size*size, -1, dtype=np.int32)
    np.maximum.at(selected, flat[front], np.flatnonzero(front))
    mask = selected >= 0
    chosen = selected[mask]
    visible_normals = _unit((normals[vertices[chosen]]*weights[chosen, :, None]).sum(1))
    visible_normals *= np.where(visible_normals[:, 2:3] < 0, -1., 1.)
    return mask, depths[chosen], visible_normals, component_ids[vertices[chosen, 0]]


def _torus(u, v, major, minor, z=0):
    return np.stack(((major+minor*np.cos(v))*np.cos(u),
                     (major+minor*np.cos(v))*np.sin(u),
                     z+minor*np.sin(v)+np.zeros_like(u)), axis=-1)


def _base_components(family, seed, resolution):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4129, FAMILY_TO_INDEX[family]]))
    # Dense sampling is independent of image values; each output pixel is shaded
    # directly, rather than obtained by blurring or enlarging a 32px raster.
    count = int(resolution*2)
    line = np.arange(count, dtype=np.float32)/count
    u, v = np.meshgrid(line*2*np.pi, line*2*np.pi, indexing='ij')
    parameters = {'family': family, 'object_seed': seed, 'formula': FORMULAS[family]}
    if family in ('superellipsoid', 'radial_rosette'):
        # Half-step avoids exactly degenerate poles while leaving <1 pixel cap.
        vv = ((np.arange(count, dtype=np.float32)+.5)/count-.5)*np.pi
        u, v = np.meshgrid(line*2*np.pi, vv, indexing='ij')
    if family == 'superellipsoid':
        e1, e2 = rng.uniform(.55, 1.7, 2)
        axes = rng.uniform(.7, 1.1, 3)
        u, v = np.meshgrid(_superangle(line*2*np.pi, e2), _superangle(vv, e1, False), indexing='ij')
        points = np.stack((axes[0]*_spow(np.cos(v), e1)*_spow(np.cos(u), e2),
                           axes[1]*_spow(np.cos(v), e1)*_spow(np.sin(u), e2),
                           axes[2]*_spow(np.sin(v), e1)), axis=-1)
        parameters.update(exponents=[float(e1), float(e2)], axes=axes.tolist())
        components = [_component(points, u, v, (True, False))]
    elif family == 'supertoroid':
        e1, e2 = rng.uniform(.65, 1.5, 2)
        u, v = np.meshgrid(_superangle(line*2*np.pi, e1), _superangle(line*2*np.pi, e2), indexing='ij')
        major, minor = float(rng.uniform(.67, .8)), float(rng.uniform(.24, .34))
        r = major+minor*_spow(np.cos(v), e2)
        points = np.stack((r*_spow(np.cos(u), e1), r*_spow(np.sin(u), e1),
                           minor*_spow(np.sin(v), e2)), axis=-1)
        parameters.update(exponents=[float(e1), float(e2)], major_radius=major, minor_radius=minor)
        components = [_component(points, u, v)]
    elif family == 'lobed_vase':
        t = v/(2*np.pi)
        lobes, amplitude = int(rng.integers(3, 8)), float(rng.uniform(.06, .15))
        radius = (.48+.16*np.cos(2*np.pi*t)+.15*np.exp(-((t-.18)/.2)**2))*(1+amplitude*np.cos(lobes*u)*np.sin(np.pi*t)**2)
        points = np.stack((radius*np.cos(u), radius*np.sin(u), 2*t-1), axis=-1)
        components = [_component(points, u, v, (True, False), 'vase_wall'),
                      _component(_torus(u, v, .64, .045, .99), u, v, name='rim'),
                      _component(_torus(u, v, .70, .055, -.99), u, v, name='foot')]
        parameters.update(lobes=lobes, lobe_amplitude=amplitude, surface_kind='open wall with rim and foot')
    elif family == 'trefoil_tube':
        major, minor, tube = 1., float(rng.uniform(.35, .45)), float(rng.uniform(.14, .19))
        angle = line*2*np.pi
        center = np.column_stack(((major+minor*np.cos(3*angle))*np.cos(2*angle),
                                  (major+minor*np.cos(3*angle))*np.sin(2*angle), minor*np.sin(3*angle)))
        tangent = _unit(np.roll(center, -1, axis=0)-np.roll(center, 1, axis=0))
        # Project radial xy direction into each curve's normal plane. This frame
        # is periodic and nondegenerate for the declared torus-knot parameter range.
        radial = _unit(center*np.array([1., 1., 0.]))
        normal = _unit(radial-(radial*tangent).sum(-1, keepdims=True)*tangent)
        binormal = _unit(np.cross(tangent, normal))
        points = center[:, None, :]+tube*(np.cos(v)[..., None]*normal[:, None, :]+np.sin(v)[..., None]*binormal[:, None, :])
        components = [_component(points, u, v)]
        parameters.update(torus_windings=[2, 3], major_radius=major, centerline_minor_radius=minor, tube_radius=tube)
    elif family == 'corrugated_torus':
        ribs, amplitude = int(rng.integers(9, 17)), float(rng.uniform(.1, .22))
        minor = .29*(1+amplitude*np.cos(ribs*u+2*v))
        points = np.stack(((.75+minor*np.cos(v))*np.cos(u), (.75+minor*np.cos(v))*np.sin(u), minor*np.sin(v)), axis=-1)
        components = [_component(points, u, v)]
        parameters.update(ribs=ribs, helical_winding=2, corrugation_amplitude=amplitude)
    elif family == 'saddle_shell':
        x, y = np.meshgrid(np.linspace(-1, 1, count, dtype=np.float32), np.linspace(-1, 1, count, dtype=np.float32), indexing='ij')
        a, b, c, d = rng.uniform(.85, 1.05), rng.uniform(.75, 1.), rng.uniform(.35, .6), rng.uniform(.06, .14)
        points = np.stack((a*x, b*y, c*(x*x-y*y)+d*np.sin(3*x)*np.cos(2*y)), axis=-1)
        components = [_component(points, (x+1)*np.pi, (y+1)*np.pi, (False, False))]
        parameters.update(coefficients=[float(a), float(b), float(c), float(d)], surface_kind='two-sided open shell')
    elif family == 'radial_rosette':
        lobes, bands = int(rng.integers(4, 9)), int(rng.integers(2, 5))
        amplitude, band_amp, height = float(rng.uniform(.17, .29)), float(rng.uniform(.05, .1)), float(rng.uniform(.65, .9))
        radius = 1+amplitude*np.cos(lobes*u)*np.cos(v)**2+band_amp*np.cos(bands*v)
        points = np.stack((radius*np.cos(v)*np.cos(u), radius*np.cos(v)*np.sin(u), height*radius*np.sin(v)), axis=-1)
        components = [_component(points, u, v, (True, False))]
        parameters.update(lobes=lobes, latitude_bands=bands, lobe_amplitude=amplitude, band_amplitude=band_amp, height=height)
    else:
        stem, bulge, width, height = [float(value) for value in
            (rng.uniform(.18, .25), rng.uniform(.35, .5), rng.uniform(.55, .75), rng.uniform(.85, 1.15))]
        collar_radius = float(rng.uniform(.055, .09))
        t = v/np.pi-1
        r = stem+bulge*np.exp(-(t/width)**2)
        points = np.stack((r*np.cos(u), r*np.sin(u), height*t), axis=-1)
        components = [_component(points, u, v, (True, False), 'spindle')]
        collars = [-.65, 0., .65]
        for index, zz in enumerate(collars):
            radius = stem+bulge*np.exp(-(zz/width)**2)
            components.append(_component(_torus(u, v, radius+.04, collar_radius, height*zz), u, v, name=f'collar_{index}'))
        parameters.update(stem_radius=stem, bulge_radius=bulge, bulge_width=width, height=height,
            collar_heights=[height*value for value in collars], collar_radius=collar_radius,
            composition='four intersecting parametric surfaces; no CSG union')
    return components, parameters


@lru_cache(maxsize=16)
def _geometry(family, object_seed, strength, resolution):
    components, parameters = _base_components(family, object_seed, resolution)
    rng = np.random.default_rng(np.random.SeedSequence([object_seed, 7187, FAMILY_TO_INDEX[family]]))
    bumps = [{'u': float(rng.uniform(0, 2*np.pi)), 'v': float(rng.uniform(0, 1)),
              'sigma_u': float(rng.uniform(.14, .38)), 'sigma_v': float(rng.uniform(.045, .12)),
              'amplitude': float(rng.uniform(-1, 1))} for _ in range(6)]
    frequency, phase = int(rng.integers(13, 27)), float(rng.uniform(-np.pi, np.pi))
    points, normals, component_ids, triangles = [], [], [], []
    point_offset = 0
    maximum_displacement = 0.
    for index, component in enumerate(components):
        base, u, v = component['points'], component['u'], component['v']
        periodic = component['periodic']
        normal = _normals(base, *periodic)
        vv = ((v+np.pi/2)/np.pi if family in ('superellipsoid', 'radial_rosette')
              else v/(2*np.pi))
        field = np.zeros(u.shape, dtype=np.float32)
        # Normalize by absolute coefficient sum: absolute Gaussian component <=.8
        # and periodic component <=.2, hence displacement is bounded by strength.
        denominator = max(sum(abs(b['amplitude']) for b in bumps), 1e-8)
        for bump in bumps:
            du = np.abs(u-bump['u'])
            if periodic[0]:
                du = np.minimum(du, 2*np.pi-du)
            dv = np.abs(vv-bump['v'])
            if periodic[1]:
                dv = np.minimum(dv, 1-dv)
            field += .8*bump['amplitude']/denominator*np.exp(-.5*((du/bump['sigma_u'])**2+(dv/bump['sigma_v'])**2))
        groove = np.cos(frequency*u+3*v+phase)
        envelope = np.ones_like(u) if periodic[1] else np.sin(np.pi*vv)**2
        field += .2*groove*envelope
        displacement = strength*field
        displaced = base+displacement[..., None]*normal
        normal = _normals(displaced, *periodic)
        maximum_displacement = max(maximum_displacement, float(np.max(np.abs(displacement))))
        points.append(displaced.reshape(-1, 3))
        normals.append(normal.reshape(-1, 3))
        component_ids.append(np.full(displaced.shape[0]*displaced.shape[1], index, dtype=np.uint8))
        triangles.append(_grid_triangles(displaced.shape[0], displaced.shape[1], *periodic)+point_offset)
        point_offset += displaced.shape[0]*displaced.shape[1]
    points, normals = np.concatenate(points), np.concatenate(normals)
    # Fixed base-dependent scale ensures same-view 0/>0 treatment differs through
    # geometric perturbation, not a separate fit-to-view camera normalization.
    base_points = np.concatenate([component['points'].reshape(-1, 3) for component in components])
    center = (base_points.max(0)+base_points.min(0))/2
    radius = float(np.linalg.norm(base_points-center, axis=1).max())
    points = (points-center)/radius
    geometry_hash = hashlib.sha256(points.astype(np.float32).tobytes()).hexdigest()
    parameters.update(surface_strength=float(strength), gaussian_bumps=bumps,
        groove_frequency=frequency, groove_phase=phase, displacement_bound=float(strength),
        maximum_actual_displacement=maximum_displacement,
        components=[component['name'] for component in components],
        base_bounding_sphere_radius=radius, base_center=center.tolist())
    # IDs describe the continuous program, independent of camera and raster size.
    identity = {key: value for key, value in parameters.items()
                if key not in ('maximum_actual_displacement', 'base_bounding_sphere_radius', 'base_center')}
    program_id = hashlib.sha256(json.dumps({'revision': REVISION, **identity}, sort_keys=True).encode()).hexdigest()
    base_id = hashlib.sha256(json.dumps({'revision': REVISION, 'family': family, 'object_seed': object_seed}, sort_keys=True).encode()).hexdigest()
    return points.astype(np.float32), normals.astype(np.float32), np.concatenate(component_ids), parameters, program_id, base_id, geometry_hash, np.concatenate(triangles)


def render_surface(family, object_seed, view_seed=0, surface_strength=.06, size=128):
    """Render a named shape. Object RNG and view RNG are strictly independent.

    Returns RGB uint8 [H,W,3], mask bool [H,W], camera-space normal float32
    [H,W,3], positive depth float32 [H,W] (zero outside), component uint8,
    parameters and stable program IDs. `normal` faces the camera for two-sided
    shading; it is not an intrinsic orientation/closed-mesh guarantee. Surface
    strength controls *geometry*, measured before bounding-sphere normalization.
    This is a deterministic triangle z-buffer renderer, without cast shadows,
    perspective, physical material simulation, topology validation, or image blur.
    """
    if family not in FAMILY_TO_INDEX:
        raise ValueError('Unknown family; choose one of '+', '.join(FAMILY_NAMES))
    object_seed, view_seed = _seed(object_seed, 'object_seed'), _seed(view_seed, 'view_seed')
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or not 32 <= int(size) <= 384:
        raise ValueError('size must be an integer between32and384')
    if isinstance(surface_strength, (bool, np.bool_)) or not np.isfinite(surface_strength) or not 0 <= float(surface_strength) <= .2:
        raise ValueError('surface_strength must be finite in [0,.2]')
    size, strength = int(size), float(surface_strength)
    points, normals, component_ids, params, program_id, base_id, geometry_hash, triangles = _geometry(family, object_seed, strength, size)
    rng = np.random.default_rng(np.random.SeedSequence([view_seed, 9313]))
    azimuth, elevation = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(.22, .95))
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.], dtype=np.float32)
    eye = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)], dtype=np.float32)
    up = np.cross(eye, right)
    rotation = np.column_stack((right, up, eye))
    xyz, normal = points@rotation, normals@rotation
    scale = float(rng.uniform(.36, .41))*size
    translation = rng.uniform(-.025, .025, 2)*size
    px = xyz[:, 0]*scale+size/2+translation[0]
    py = -xyz[:, 1]*scale+size/2+translation[1]
    mask, visible_depth, visible_normal, visible_components = _rasterize_triangles(
        xyz, normal, triangles, component_ids, px, py, size)
    light = _unit(np.array([rng.uniform(-.6, .6), rng.uniform(.2, .8), rng.uniform(.7, 1.2)], dtype=np.float32))
    half_vector = _unit(light+np.array([0., 0., 1.], dtype=np.float32))
    diffuse = np.clip(visible_normal@light, 0., 1.)
    specular = np.clip(visible_normal@half_vector, 0., 1.)**28
    material = rng.uniform(.32, .86, 3)
    # Independent low-frequency background colors are rendered analytically;
    # no noise filtering is used, and the object is not assigned a family color.
    yy, xx = np.mgrid[0:size, 0:size]/max(size-1, 1)
    background = rng.uniform(.07, .2, 3)[None, None, :]+(.025*xx+.035*yy)[..., None]
    colors = np.clip(material[None, :]*(.22+.76*diffuse[:, None])+.18*specular[:, None], 0., 1.)
    rgb = np.rint(background*255).astype(np.uint8).reshape(-1, 3)
    rgb[mask] = np.rint(colors*255).astype(np.uint8)
    normal_image = np.zeros((size*size, 3), dtype=np.float32)
    normal_image[mask] = visible_normal
    depth = np.zeros(size*size, dtype=np.float32)
    depth[mask] = 3.-visible_depth
    parts = np.full(size*size, 255, dtype=np.uint8)
    parts[mask] = visible_components
    view = {'view_seed': view_seed, 'azimuth': azimuth, 'elevation': elevation,
            'scale_pixels': scale, 'translation_pixels': translation.tolist(),
            'light_camera': light.tolist(), 'material_rgb': material.tolist()}
    return {'rgb': rgb.reshape(size, size, 3), 'mask': mask.reshape(size, size),
        'normal': normal_image.reshape(size, size, 3), 'depth': depth.reshape(size, size),
        'component': parts.reshape(size, size), 'parameters': json.loads(json.dumps(params)),
        'family': family, 'family_index': FAMILY_TO_INDEX[family], 'family_name': FAMILY_LABELS[family],
        'object_name': FAMILY_LABELS[family]+' '+program_id[:8], 'program_id': program_id,
        'base_program_id': base_id, 'geometry_sha256': geometry_hash, 'view': view,
        'render': {'revision': REVISION, 'native_resolution': size, 'surface_samples': len(points), 'surface_triangles': len(triangles),
                   'gaussian_image_blur': False, 'image_resizing': False, 'external_assets': 0,
                   'natural_images': 0, 'pretrained_models': False, 'projection': 'orthographic',
                   'rasterizer': 'pixel-center triangle coverage + barycentric depth/normals + z-buffer; no image filtering'}}
