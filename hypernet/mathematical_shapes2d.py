"""2D-only compositional shapes and matched reference-conditioned scenes.

4096 discrete program compositions, not4096 natural semantic categories.
Pixels are sampled from analytic/vector masks at the requested resolution:
there are no3D surfaces, image assets, blur filters, or pretrained generators.
"""
from functools import lru_cache
import hashlib
import json
import math
import numpy as np

REVISION = 'mathematical_shapes2d_v11_1'
FAMILY_COUNT = 4096
OUTLINES = ('ellipse', 'rounded_rectangle', 'hexagon', 'shield', 'capsule', 'diamond', 'five_lobed', 'droplet')
LAYOUTS = tuple(f'{number}_attached_parts' for number in range(1, 9))
STROKE_PROFILES = ('uniform', 'taper_outward', 'taper_inward', 'stepped', 'beaded', 'triangular_ribs', 'waisted', 'scalloped')
MICROPROGRAMS = ('rising_pair', 'falling_pair', 'diagonal_pair', 'vertical_pair', 'offset_pair', 'wide_pair', 'close_pair', 'staggered_pair')
MICRO_SITES = (
    ((-.18, -.07), (.18, .07), 0., 90.),
    ((-.18, .07), (.18, -.07), 0., 90.),
    ((-.15, -.13), (.15, .13), 45., 135.),
    ((-.07, -.16), (.07, .16), 0., 90.),
    ((-.19, -.04), (.13, .12), 25., 115.),
    ((-.22, .01), (.22, .01), 0., 90.),
    ((-.13, -.09), (.13, .09), 35., 125.),
    ((-.17, .12), (.16, -.03), 65., 155.),
)
MICRO_WIDTH, MICRO_LENGTH = .12, .19


def _integer(value, name, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(name+' must be a nonnegative integer')
    value = int(value)
    if maximum is not None and value >= maximum:
        raise ValueError(name+' is outside its declared range')
    return value


def decode_family(family_id):
    value = _integer(family_id, 'family_id', FAMILY_COUNT)
    return [value//512, (value//64)%8, (value//8)%8, value%8]


def family_spec(family_id):
    digits = decode_family(family_id)
    names = [choices[index] for choices, index in zip((OUTLINES, LAYOUTS, STROKE_PROFILES, MICROPROGRAMS), digits)]
    identity = {'revision': REVISION, 'family_id': int(family_id), 'family_digits': digits, 'grammar_names': names}
    return {**identity, 'grammar_digits': digits, 'family_name': ' / '.join(names),
            'canonical_program_id': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            'family_scope': 'Discrete 2D composition program; not a natural semantic category'}


def _rng(*seeds):
    return np.random.default_rng(np.random.SeedSequence([int(seed) for seed in seeds]))


def _polygon(x, y, vertices):
    inside = np.zeros_like(x, dtype=bool)
    last = vertices[-1]
    for current in vertices:
        x0, y0 = last
        x1, y1 = current
        if y0 != y1:
            crosses = ((y0 > y) != (y1 > y)) & (x < (x1-x0)*(y-y0)/(y1-y0)+x0)
            inside ^= crosses
        last = current
    return inside


def _macro_mask(x, y, digits, instance_seed, canonical=False):
    rng = _rng(instance_seed, 7411)
    axes = np.ones(2) if canonical else rng.uniform(.97, 1.03, 2)
    connector_gain = 1. if canonical else float(rng.uniform(.97, 1.03))
    xx, yy = x/axes[0], y/axes[1]
    outline, layout, profile, _ = digits
    if outline == 0:
        mask = (xx/.49)**2+(yy/.45)**2 <= 1
    elif outline == 1:
        mask = (np.abs(xx)/.48)**4+(np.abs(yy)/.43)**4 <= 1
    elif outline == 2:
        mask = _polygon(xx, yy, [(math.cos(a)*.53, math.sin(a)*.49) for a in np.arange(6)*np.pi/3])
    elif outline == 3:
        mask = _polygon(xx, yy, [(-.45,-.40),(.45,-.40),(.49,.02),(.30,.34),(0,.55),(-.30,.34),(-.49,.02)])
    elif outline == 4:
        mask = np.maximum(np.abs(xx)-.20, 0)**2+yy**2 <= .32**2
    elif outline == 5:
        mask = np.abs(xx)/.54+np.abs(yy)/.52 <= 1
    elif outline == 6:
        angle = np.arctan2(yy, xx)
        mask = xx*xx+yy*yy <= (.43+.065*np.cos(5*angle))**2
    else:
        angle = np.arctan2(yy, xx)
        mask = xx*xx+yy*yy <= (.45*(1+.20*np.sin(angle)+.08*np.cos(2*angle)))**2
    # Layouts alter part count/intrinsic arrangement, never merely rotate a shape.
    for part in range(layout+1):
        angle = .28+part*2.399963229728653
        along, across = xx*np.cos(angle)+yy*np.sin(angle), -xx*np.sin(angle)+yy*np.cos(angle)
        t = np.clip((along-.30)/.46, 0, 1)
        if profile == 0: width = .085+0*t
        elif profile == 1: width = .045+.09*t
        elif profile == 2: width = .13-.075*t
        elif profile == 3: width = .055+.07*(t > .5)
        elif profile == 4: width = .065+.035*np.cos(6*np.pi*t)
        elif profile == 5: width = .055+.04*(1-np.abs(2*np.mod(3*t, 1)-1))
        elif profile == 6: width = .11-.06*np.sin(np.pi*t)**2
        else: width = .065+.04*np.sin(4*np.pi*t)**2
        mask |= (along >= .30)&(along <= .76)&(np.abs(across) <= width*connector_gain)
        mask |= (along-.76)**2+across**2 <= .105**2
    # A fixed off-center orientation key is part of every macro program. It
    # prevents rotationally symmetric bodies from making internal orientations
    # contradictory identities when scenes rotate the whole object.
    mask &= xx**2+(yy+.285)**2 > .037**2
    return mask, {'axis_jitter': axes.tolist(), 'connector_gain': connector_gain}


def _micro_mask(x, y, program, variant):
    first, second, angle_a, angle_b = MICRO_SITES[program]
    holes = np.zeros_like(x, dtype=bool)
    for index, (cx, cy) in enumerate((first, second)):
        angle = (angle_a, angle_b)[index ^ variant]*np.pi/180
        xx, yy = x-cx, y-cy
        along, across = xx*np.cos(angle)+yy*np.sin(angle), -xx*np.sin(angle)+yy*np.cos(angle)
        holes |= (np.abs(along) <= MICRO_LENGTH/2)&(np.abs(across) <= MICRO_WIDTH/2)
    return holes


def raster_shape(family_id, instance_seed, micro_variant, height=96, width=96,
                 center=None, scale=None, angle=0., canonical=False):
    """Native pixel-center samples of a connected macro silhouette with holes."""
    digits = decode_family(family_id)
    instance_seed = _integer(instance_seed, 'instance_seed')
    micro_variant = _integer(micro_variant, 'micro_variant', 2)
    if min(height, width) < 16:
        raise ValueError('Raster dimensions must be at least16')
    center = (width/2, height/2) if center is None else center
    scale = min(height, width)*.475 if scale is None else float(scale)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xx, yy = (xx+.5-center[0])/scale, (yy+.5-center[1])/scale
    x, y = xx*np.cos(angle)+yy*np.sin(angle), -xx*np.sin(angle)+yy*np.cos(angle)
    macro, parameters = _macro_mask(x, y, digits, instance_seed, canonical)
    micro = _micro_mask(x, y, digits[3], micro_variant)&macro
    mask = macro&~micro
    locations = np.argwhere(mask)
    if not len(locations):
        raise ValueError('Object fell outside raster')
    y0, x0 = locations.min(0)
    y1, x1 = locations.max(0)+1
    stats = {'candidate_foreground_pixels': int(mask.sum()), 'macro_foreground_pixels': int(macro.sum()),
        'micro_removed_pixels': int(micro.sum()), 'primitive_count': 2,
        'continuous_removed_area': 2*MICRO_WIDTH*MICRO_LENGTH,
        'minimum_nominal_feature_width_pixels': MICRO_WIDTH*scale,
        'maximum_nominal_feature_length_pixels': MICRO_LENGTH*scale,
        'micro_control': 'Two equal-area rectangular holes exchange orientations between fixed sites; count and continuous area identical',
        'raster_ink_note': 'Pixel counts can differ slightly after rotation; exact area differences are reported'}
    return {'mask': mask, 'macro_mask': macro, 'micro_mask': micro,
            'bbox_xyxy': [int(x0),int(y0),int(x1),int(y1)], 'parameters': parameters, 'micro_statistics': stats}


def canonical_mask(family_id, micro_variant=0, size=96):
    return raster_shape(family_id, 0, micro_variant, size, size, canonical=True)['mask']


def _background(size, rng):
    yy, xx = np.mgrid[:size, :size]/max(size-1, 1)
    color = rng.uniform(.10, .28, 3)
    ramp = rng.uniform(-.025, .025)*xx+rng.uniform(-.025, .025)*yy
    wave = .008*np.sin(2*np.pi*(rng.uniform(.5, 1.5)*xx+rng.uniform(.5, 1.5)*yy))
    return np.rint(np.clip(color[None,None,:]+ramp[...,None]+wave[...,None],0,1)*255).astype(np.uint8)


def _paint(background, mask, color):
    result = background.copy()
    result[mask] = color
    return result


@lru_cache(maxsize=16)
def _scene_pair(family_id, object_seed, scene_seed, distractors, query_size):
    rng = _rng(scene_seed, 8117)
    background = _background(query_size, rng)
    count = len(distractors)+1
    rows = 2 if count <= 6 else 3
    cells = [(x,y) for y in range(rows) for x in range(3)]
    rng.shuffle(cells)
    candidate_slot = int(rng.integers(0, count))
    families = list(distractors)
    families.insert(candidate_slot, family_id)
    common = background.copy()
    union = np.zeros((query_size,query_size), dtype=bool)
    objects, candidate_args, candidate_color = [], None, None
    for index, current_family in enumerate(families):
        column,row = cells[index]
        center = ((column+.5)*query_size/3+rng.uniform(-2.5,2.5),
                  (row+.5)*query_size/rows+rng.uniform(-2.5,2.5))
        scale = float(rng.uniform(36,40))*query_size/256
        angle = float(rng.uniform(-np.pi,np.pi))
        color = np.rint(rng.uniform(.48,.94,3)*255).astype(np.uint8)
        instance_seed = int(rng.integers(0,2**31-1))
        variant = int(rng.integers(0,2))
        args = dict(family_id=current_family, instance_seed=instance_seed, height=query_size, width=query_size,
                    center=center, scale=scale, angle=angle)
        shape = raster_shape(micro_variant=0 if index==candidate_slot else variant, **args)
        if np.any(union&shape['macro_mask']):
            raise RuntimeError('Nonoverlapping pilot object placement failed')
        record = {'bbox_xyxy': shape['bbox_xyxy'], 'role': 'candidate' if index==candidate_slot else 'distractor',
            'family_id': current_family, 'micro_variant': 0 if index==candidate_slot else variant,
            'object_seed': object_seed if index==candidate_slot else instance_seed,
            'instance_seed': instance_seed,
            'instance_key': hashlib.sha256(json.dumps({'family': current_family,'seed':instance_seed},sort_keys=True).encode()).hexdigest(),
            'pose': {'center_xy': list(center), 'scale': scale, 'angle_radians': angle},
            'color_rgb': color.tolist(), 'parameters': shape['parameters']}
        objects.append(record)
        if index == candidate_slot:
            candidate_args, candidate_color = args, color
        else:
            common[shape['mask']] = color
            union |= shape['mask']
    queries = {}
    for variant in (0,1):
        shape = raster_shape(micro_variant=variant, **candidate_args)
        records = json.loads(json.dumps(objects))
        records[candidate_slot]['micro_variant'] = variant
        records[candidate_slot]['bbox_xyxy'] = shape['bbox_xyxy']
        queries[str(variant)] = {'query_rgb': _paint(common,shape['mask'],candidate_color),
            'union_mask': union|shape['mask'], 'candidate_mask': shape['mask'],
            'candidate_bbox_xyxy': shape['bbox_xyxy'], 'object_records': records,
            'object_boxes': [record['bbox_xyxy'] for record in records], 'all_boxes': records,
            'micro_statistics': shape['micro_statistics'], 'candidate_variant': variant}
    changed = np.any(queries['0']['query_rgb'] != queries['1']['query_rgb'],axis=-1)
    candidate_union = queries['0']['candidate_mask']|queries['1']['candidate_mask']
    assert not np.any(changed&~candidate_union)
    pair = {'candidate_changed_pixels': int(changed.sum()),
        'candidate_area_difference_pixels': abs(int(queries['0']['candidate_mask'].sum())-int(queries['1']['candidate_mask'].sum())),
        'query_pixels_unchanged_outside_candidate': True, 'candidate_slot': candidate_slot,
        'scene_object_count': count, 'nonoverlap': True,
        'query_candidate_is_new_macro_instance': True,
        'new_instance_scope': 'Independent +/-3% macro-axis/connector jitter; same discrete family grammar and local hole program'}
    return queries, pair


def render_episode(family_id, object_seed, enrolled_variant, scene_seed, distractor_family_ids,
                   support_size=96, query_size=256):
    """One support and both query microstates, exactly counterbalanced by reference.

    Each query's labels depend on enrolled_variant but its pixels never do.
    Nonmatching target_mask is zero; candidate_mask still identifies the hard
    negative slot for auditing, and must not be supplied to model inference.
    """
    spec = family_spec(family_id)
    object_seed, scene_seed = _integer(object_seed,'object_seed'), _integer(scene_seed,'scene_seed')
    enrolled_variant = _integer(enrolled_variant,'enrolled_variant',2)
    if support_size != 96 or query_size != 256:
        raise ValueError('This fixed pilot requires support96/query256')
    distractors = tuple(_integer(value,'distractor_family_id',FAMILY_COUNT) for value in distractor_family_ids)
    if not 4 <= len(distractors) <= 7 or len(set(distractors)) != len(distractors) or family_id in distractors:
        raise ValueError('Provide4to7distinct other-family distractors')
    support_rng = _rng(object_seed,scene_seed,9007)
    background = _background(support_size,support_rng)
    color = np.rint(support_rng.uniform(.48,.94,3)*255).astype(np.uint8)
    angle = float(support_rng.uniform(-.18,.18))
    shape = raster_shape(family_id,object_seed,enrolled_variant,support_size,support_size,angle=angle)
    base_queries,pair = _scene_pair(int(family_id),object_seed,scene_seed,distractors,query_size)
    queries = {}
    for variant,query in base_queries.items():
        # Caller mutation must never poison the cached paired scene.
        copy = {key: value.copy() if isinstance(value,np.ndarray) else json.loads(json.dumps(value))
                for key,value in query.items()}
        matches = int(variant)==enrolled_variant
        copy.update(matches_reference=matches, present=matches,
            target_mask=copy['candidate_mask'].copy() if matches else np.zeros_like(copy['candidate_mask']),
            target_bbox_xyxy=copy['candidate_bbox_xyxy'] if matches else None)
        queries[variant]=copy
    return {'support_rgb':_paint(background,shape['mask'],color), 'support_mask':shape['mask'],
        'support_bbox_xyxy':shape['bbox_xyxy'], 'support_micro_statistics':shape['micro_statistics'],
        'queries':queries, 'family_spec':spec, 'family_id':int(family_id),
        'object_seed':object_seed,'scene_seed':scene_seed,'enrolled_variant':enrolled_variant,
        'pair_statistics':json.loads(json.dumps(pair)),
        'parameters':{'revision':REVISION,'support_parameters':shape['parameters'],
            'support_instance_seed':object_seed,
            'support_angle_radians':angle,'support_color_rgb':color.tolist(),
            'support_size':support_size,'query_size':query_size,'external_assets':0,
            'natural_images':0,'pretrained_generator':False,'image_blur':False,'three_dimensional':False,
            'label_rule':'presence iff candidate microvariant equals enrolled reference microvariant',
            'limitations':'Isolated nonoverlapping objects; synthetic composition families; no natural-category claim'}}
