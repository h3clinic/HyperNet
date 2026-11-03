"""Deterministic, entirely 2D structural-program reference-to-scene episodes.

Programs define synthetic identity; seeds only vary its permitted appearance.
The grammar is deliberately finite and is not a catalogue of natural semantics.
Masks use centre-sampled occupancy. RGB boundaries are antialiased at 2x scale.
Different same-type instances have part labels, not a dense pixel correspondence.
"""
from functools import lru_cache
import hashlib
import json
import math
import cv2
import numpy as np

FORMAT = 'hypernet.compositional_2d_curriculum.v17.1'
FAMILIES = ('branch', 'contour', 'repetition', 'ridge', 'texture', 'composite')
TRAIN_PAIRS = (('branch', 'contour'), ('repetition', 'texture'), ('contour', 'ridge'), ('branch', 'repetition'))
HELDOUT_PAIRS = (('branch', 'ridge'), ('contour', 'texture'), ('ridge', 'repetition'), ('branch', 'texture'))
SPLIT_NUISANCE_CODES = {'train': 0, 'development': 1, 'test': 2, 'compositional': 3}
CONFIG = {'format': FORMAT, 'families': list(FAMILIES), 'support_size': 96, 'query_size': 192,
          'supersampling': 2, 'target_probability': .5, 'visibility_floor': .65,
          'minimum_line_width_final_pixels': 1,
          'matched_negative_selection': 'Program-hash seeded random ordering of admissible single-attribute edits, independently for scene distractor and absent replacement',
          'material_rendering': 'Modulate foreground contrast against the local background, preserving texture in either polarity',
          'ambiguity_rejection': 'At most 16 deterministic nuisance draws of the same program; reject only explicit unresolved edit geometry or final presence counterfactuals with fewer than 8 RGB-distinguishing pixels; retry counts and reasons recorded',
          'family_sampling': 'Round-robin by episode index, with program hashing within each family',
          'support_framing': 'Known synthetic mask crop with margins 0,.05,.15,.35,.6 then aspect-preserving letterbox; padding is crop border median RGB',
          'train_composite_pairs': TRAIN_PAIRS, 'construction_holdout_pairs': HELDOUT_PAIRS,
          'split': 'SHA256 of the complete canonical structural program modulo10: 0=test,1=development,2..9=train',
          'subprogram_overlap': 'Hash disjointness applies to whole programs; primitive subprograms can recur inside composites across splits. The separate compositional split reserves constituent-pair constructions, not unseen primitive categories.',
          'nuisance_split_codes': SPLIT_NUISANCE_CODES,
          'natural_images_used': 0, 'borrowed_weights_used': False,
          'mask_convention': 'Binary occupancy and part labels sampled at final pixel centres; RGB antialiased separately',
          'occlusion_identity_floor': 'At least 8 geometric/material edit pixels for each negative and 8 counterfactual RGB pixels with channel difference >=3 remain untouched by any occluder alpha',
          'bbox_convention': 'Tight visible centre-sampled target mask, pixelxyxy half-open; absent is fourzeros',
          'positive_correspondence': 'Part identity only across independently rendered instances; not pixel correspondence'}


def canonical(program):
    return json.dumps(program, sort_keys=True, separators=(',', ':'))


def program_id(program):
    return hashlib.sha256(canonical(program).encode()).hexdigest()


def config_hash():
    return hashlib.sha256(canonical(CONFIG).encode()).hexdigest()


def contains_program(container, target):
    """Exclude exact target subprograms from every non-target scene instance."""
    return program_id(container)==program_id(target) or any(contains_program(part,target) for part in container.get('parts',[]))


def program_split(program):
    if program['family'] == 'composite':
        pair = tuple(p['family'] for p in program['parts'])
        if pair in HELDOUT_PAIRS:
            return 'compositional'
    bucket = int(program_id(program)[:16], 16) % 10
    return 'test' if bucket == 0 else 'development' if bucket == 1 else 'train'


def _rng(*values):
    return np.random.default_rng(np.random.SeedSequence([int(v) & 0xffffffff for v in values]))


def _candidate(rng, family=None, heldout=False):
    family = family or FAMILIES[int(rng.integers(len(FAMILIES)))]
    if family == 'branch':
        n = int(rng.integers(2, 8))
        spec = [n, int(rng.integers(1 << n)), int(rng.integers(1 << n)), int(rng.integers(3))]
    elif family == 'contour':
        holes = int(rng.integers(4))
        spec = [int(rng.integers(3, 9)), int(rng.integers(2, 6)), int(rng.integers(8)), holes,
                int(rng.integers(3)) if holes > 1 else 0]
    elif family == 'repetition':
        spec = [int(rng.integers(3, 8)), int(rng.integers(4)), int(rng.integers(3)), int(rng.integers(4))]
    elif family == 'ridge':
        n = int(rng.integers(2, 6))
        spacing = int(rng.choice([0, 2])) if n == 3 else int(rng.integers(3)) if n > 3 else 0
        spec = [n, int(rng.integers(6)), spacing, int(rng.integers(3))]
    elif family == 'texture':
        spec = [int(rng.integers(8)), int(rng.integers(3, 9)), int(rng.integers(4)), int(rng.integers(3))]
    elif family == 'composite':
        pairs = HELDOUT_PAIRS if heldout else TRAIN_PAIRS
        pair = pairs[int(rng.integers(len(pairs)))]
        return {'family': family, 'layout': int(rng.integers(3)),
                'parts': [_candidate(rng, f) for f in pair]}
    else:
        raise ValueError('Unknown structural family: ' + str(family))
    return {'family': family, 'spec': spec}


@lru_cache(maxsize=8192)
def _program_json(index, split, seed, family):
    if split not in ('train', 'development', 'test', 'compositional'):
        raise ValueError('Unknown split')
    rng = _rng(seed, index, 191)
    family = family or FAMILIES[(index+seed)%len(FAMILIES)]
    for _ in range(10000):
        p = _candidate(rng, 'composite' if split == 'compositional' else family,
                       heldout=split == 'compositional')
        if program_split(p) == split:
            available=0
            for candidate,_ in _single_edits(p):
                if program_split(candidate)==split:
                    available+=1
                    if available==2:
                        return canonical(p)
    raise RuntimeError('Could not find a program in the declared split')


def sample_program(index, split='train', seed=1701, family=None):
    """Return a fresh dict; callers cannot mutate the cached canonical program."""
    if index < 0 or (family is not None and family not in FAMILIES):
        raise ValueError('Use a nonnegative index and known family')
    return json.loads(_program_json(int(index), split, int(seed), family))


def _single_edits(program):
    """Each proposal alters one explicit structural attribute, never just a seed."""
    family = program['family']
    if family == 'composite':
        for layout in range(3):
            if layout != program['layout']:
                yield {**program, 'layout': layout}, 'relative_part_layout'
        for i, part in enumerate(program['parts']):
            for altered, reason in _single_edits(part):
                parts = list(program['parts']); parts[i] = altered
                yield {**program, 'parts': parts}, 'part' + str(i) + ':' + reason
        return
    spec = program['spec']
    if family == 'branch':
        ranges = [(), [spec[1]^(1<<j) for j in range(spec[0])], [spec[2]^(1<<j) for j in range(spec[0])], range(3)]
        names = ['branch_count', 'branch_side_order', 'branch_fork_pattern', 'trunk_bend']
    elif family == 'contour':
        ranges = [range(3, 9), range(2, 6), range(8), range(4), range(3) if spec[3] > 1 else ()]
        names = ['outline_lobes', 'secondary_outline_harmonic', 'outline_phase', 'hole_count', 'hole_arrangement']
    elif family == 'repetition':
        ranges = [range(3, 8), range(4), range(3), range(4)]
        names = ['motif_count', 'motif_shape', 'motif_arrangement', 'spacing_order']
    elif family == 'ridge':
        ranges = [range(2, 6), range(6), [0, 2] if spec[0] == 3 else range(3) if spec[0] > 3 else (), range(3)]
        names = ['ridge_count', 'ridge_curve', 'ridge_spacing', 'ridge_window']
    else:
        ranges = [range(8), range(3, 9), range(4), range(3)]
        names = ['texture_orientation', 'texture_frequency', 'texture_modulation', 'patch_shape']
    for i, values in enumerate(ranges):
        for value in values:
            if value == spec[i]:
                continue
            changed = list(spec); changed[i] = value
            if family == 'contour' and changed[3] < 2:
                changed[4] = 0  # inactive field has one canonical value
            if family == 'ridge' and changed[0] == 2:
                changed[2] = 0
            if family == 'ridge' and changed[0] == 3 and changed[2] == 1:
                changed[2] = 0  # symmetric spacing0/1 both give[-.53,0,.53]
            yield {'family': family, 'spec': changed}, names[i]


def hard_negative(program):
    candidates = list(_single_edits(program))
    rng = _rng(int(program_id(program)[:8], 16), 1723)
    order = rng.permutation(len(candidates))
    split = program_split(program)
    for i in order:
        candidate, reason = candidates[int(i)]
        if program_split(candidate) == split and program_id(candidate) != program_id(program):
            return candidate, reason
    raise RuntimeError('No single-attribute hard negative in the same split')


def _second_negative(program, first):
    candidates=list(_single_edits(program))
    order=_rng(int(program_id(program)[:8],16),1747).permutation(len(candidates))
    for i in order:
        candidate,reason=candidates[int(i)]
        if program_split(candidate)==program_split(program) and program_id(candidate) not in (program_id(program),program_id(first)):
            return candidate,reason
    raise RuntimeError('A second distinct matched negative is required')


def _curve(points, count=48):
    points = np.asarray(points, np.float32); t = np.linspace(0, 1, count, dtype=np.float32)[:, None]
    return (1-t)**3 * points[0] + 3*(1-t)**2*t*points[1] + 3*(1-t)*t**2*points[2] + t**3*points[3]


def _geometry(program, seed, size, factor=2):
    """Draw centre-labelled geometry at higher resolution; texture stays separate."""
    rng = _rng(seed, 179); side = size * factor
    labels = np.zeros((side, side), np.int16)
    family = program['family']; material = np.ones((side, side), np.float32)
    if family == 'composite':
        layout = program['layout']; child_side = int(round(size * (.55 if layout < 2 else .48)))
        centers = ((.28, .5), (.72, .5)) if layout == 0 else ((.5, .28), (.5, .72)) if layout == 1 else ((.33, .33), (.67, .67))
        for j, (child, center) in enumerate(zip(program['parts'], centers)):
            child_labels, child_material = _geometry(child, int(seed) + 1709 * (j+1), child_side, factor)
            width = child_labels.shape[0]
            x = int(np.clip(center[0]*side-width/2+rng.uniform(-.025,.025)*side, 0, side-width))
            y = int(np.clip(center[1]*side-width/2+rng.uniform(-.025,.025)*side, 0, side-width))
            region = labels[y:y+width, x:x+width]; region[child_labels > 0] = child_labels[child_labels > 0] + j * 64
            mat = material[y:y+width, x:x+width]; mat[child_labels > 0] = child_material[child_labels > 0]
        return labels, material
    angle = rng.uniform(-.10, .10) if family != 'ridge' else rng.uniform(-.025, .025)
    sx, sy = rng.uniform(.88, 1.06, 2); shift = rng.uniform(-.035, .035, 2)
    matrix = np.array([[math.cos(angle)*sx, -math.sin(angle)*sy],
                       [math.sin(angle)*sx, math.cos(angle)*sy]], np.float32)
    def xy(points):
        p = np.asarray(points, np.float32) @ matrix.T + shift
        return np.rint((p + 1) * (side - 1) / 2).astype(np.int32)
    def line(points, part, radius=.028):
        # A subpixel high-resolution line can miss every final pixel centre.
        thickness = max(factor, int(round(radius * side)))
        pixels = xy(points)
        cv2.polylines(labels, [pixels], False, int(part), thickness, cv2.LINE_8)
        cv2.circle(labels, tuple(pixels[0]), max(1, thickness//2), int(part), -1)
        cv2.circle(labels, tuple(pixels[-1]), max(1, thickness//2), int(part), -1)
    def polygon(points, part):
        cv2.fillPoly(labels, [xy(points)], int(part), cv2.LINE_8)
    def disk(center, radius, part):
        theta = np.linspace(0, 2*np.pi, 40, endpoint=False)
        polygon(np.asarray(center) + radius * np.stack([np.cos(theta),np.sin(theta)],1), part)
    s = program['spec']; width = rng.uniform(.020, .037)
    if family == 'branch':
        n, sides, forks, bend = s
        curvature = (bend - 1) * .18
        trunk = _curve([[0,.79],[curvature,.25],[-curvature,-.25],[0,-.79]])
        line(trunk, 1, width)
        for j, t in enumerate(np.linspace(.15,.85,n)):
            k = int(t*(len(trunk)-1)); start = trunk[k]
            sign = 1 if sides & (1 << j) else -1
            gap = 1.10 / (n-1)
            end = np.array([sign*rng.uniform(.50,.74), start[1]+rng.uniform(-.07,.07)*gap])
            points = _curve([start, start+[sign*.20,-.10*gap], end+[-sign*.12,.10*gap], end])
            line(points, j+2, width*.85)
            if forks & (1 << j):
                mid = points[len(points)*2//3]
                line(_curve([mid, mid+[0,-.10*gap], end+[0,-.22*gap], end+[0,-.34*gap]]), j+2+n, width*.70)
    elif family == 'contour':
        lobes, harmonic, phase, holes, pattern = s
        theta = np.linspace(0, 2*np.pi, 160, endpoint=False)
        radius = .64 + .105*np.cos(lobes*theta+phase*np.pi/4) + .055*np.sin(harmonic*theta)
        radius += rng.uniform(.006,.018)*np.sin((lobes+harmonic)*theta+rng.uniform(0,2*np.pi))
        polygon(np.stack([radius*np.cos(theta),radius*np.sin(theta)],1), 1)
        if holes:
            for j in range(holes):
                a = 2*np.pi*j/holes + pattern*np.pi/5
                c = np.zeros(2) if holes == 1 else .30*np.array([np.cos(a),np.sin(a)])
                if pattern == 2 and holes > 1:
                    c = np.array([(j-(holes-1)/2)*.30, 0])
                disk(c, rng.uniform(.105,.135), 0)
    elif family == 'repetition':
        n, motif, arrangement, spacing = s
        gaps = np.array([1 + (.32 if (j+spacing)%3 == 0 else -.12 if (j+spacing)%3 == 1 else 0)
                         for j in range(n-1)]) if spacing < 3 else np.linspace(.6,1.4,n-1)
        positions = np.r_[0,np.cumsum(gaps)]; positions = 1.42*positions/positions[-1]-.71
        radius = min(.14,.48/n)
        for j,x in enumerate(positions):
            y = 0 if arrangement == 0 else .25*(1-(x/.72)**2) if arrangement == 1 else .15*(-1)**j
            c = np.array([x,y]); part=j+1
            if motif == 0:
                disk(c, radius, part)
            elif motif == 1:
                pts=c+radius*1.25*np.array([[0,-1],[.86,.5],[-.86,.5],[0,-1]])
                line(pts, part, width*.8)
            elif motif == 2:
                line([c+[-radius,0],c+[radius,0]], part, width)
                line([c+[0,-radius],c+[0,radius]], part, width)
            else:
                theta=np.linspace(0,2*np.pi,40,endpoint=False)
                rad=radius*(1+.30*np.cos(theta))
                polygon(c+np.stack([rad*np.cos(theta),rad*np.sin(theta)],1),part)
    elif family == 'ridge':
        n, kind, spacing, window = s
        offsets = np.linspace(-.53,.53,n)
        if spacing == 1: offsets = np.sign(offsets)*np.abs(offsets)**1.4/.53**.4
        if spacing == 2: offsets = np.linspace(0,1,n)**1.5*1.06-.53
        x = np.linspace(-.78,.78,80,dtype=np.float32)
        bend = (np.zeros_like(x) if kind == 0 else .22*x if kind == 1 else -.22*x if kind == 2
                else .22*(1-(x/.78)**2) if kind == 3 else .15*np.sin(4*x)
                if kind == 4 else .12*np.sin(5*x+3*x*x))
        for j,offset in enumerate(offsets):
            points=np.stack([x,offset+bend+rng.uniform(-.012,.012)],1)
            if window == 0: line(points,j+1,width*.7)
            elif window == 1: line(points[15:65],j+1,width*.8)
            else:
                line(points[4:33],j+1,width*.8); line(points[47:76],j+1,width*.8)
    else:  # texture: micro-pattern is identity, not an erased nuisance
        orientation, density, modulation, shape = s
        theta=np.linspace(0,2*np.pi,100,endpoint=False)
        if shape == 0:
            polygon(np.stack([.73*np.cos(theta),.57*np.sin(theta)],1),1)
        elif shape == 1:
            r=.65+.07*np.cos(5*theta); polygon(np.stack([r*np.cos(theta),r*np.sin(theta)],1),1)
        else: polygon([[-.70,-.57],[.70,-.57],[.70,.57],[-.70,.57]],1)
        y,x=np.mgrid[:side,:side].astype(np.float32); x=(x/(side-1)*2-1-shift[0]); y=(y/(side-1)*2-1-shift[1])
        a=orientation*np.pi/8+angle; u=x*np.cos(a)+y*np.sin(a); v=-x*np.sin(a)+y*np.cos(a)
        phase=density*np.pi*u+rng.uniform(-.16,.16)
        if modulation == 1: phase += 1.2*u*u
        bright=np.sin(phase)>0
        if modulation == 2: bright &= np.cos(3*np.pi*v)>-.25
        if modulation == 3: bright |= np.sin(density*np.pi*v)>0
        material=np.where(bright,1.,.30).astype(np.float32)
        labels[(labels>0)&bright] = 2+np.floor((phase[(labels>0)&bright]+density*np.pi+1)/(2*np.pi)).astype(np.int16).clip(0,30)
    return labels, material


def render_instance(program, instance_seed, size=96):
    if size < 24:
        raise ValueError('Shapes need at least24 pixels to retain declared details')
    labels_hi, mat_hi = _geometry(program, int(instance_seed), int(size))
    alpha = cv2.resize((labels_hi>0).astype(np.float32), (size,size), interpolation=cv2.INTER_AREA)
    parts = labels_hi[1::2,1::2].copy()
    material = cv2.resize(mat_hi,(size,size),interpolation=cv2.INTER_AREA)
    # A small material ripple is a nuisance for geometry families only.
    if program['family'] not in ('texture','composite'):
        rng=_rng(instance_seed,181); y,x=np.mgrid[:size,:size].astype(np.float32)
        material *= .86+.14*np.cos((x*rng.uniform(.08,.20)+y*rng.uniform(.08,.20))+rng.uniform(0,6.28))
    mask=parts>0
    if not mask.any(): raise RuntimeError('Empty structural raster')
    return {'alpha':alpha,'mask':mask.astype(np.uint8),'parts':parts,'material':material,
            'program_id':program_id(program),'family':program['family'],'seed':int(instance_seed)}


def _background(rng, size):
    base=rng.uniform(.18,.82,3).astype(np.float32)
    small=rng.normal(0,.055,(7,7,3)).astype(np.float32)
    field=cv2.resize(small,(size,size),interpolation=cv2.INTER_CUBIC)
    noise=rng.normal(0,rng.uniform(.003,.035),(size,size,1)).astype(np.float32)
    # Random correlated oriented field is independent of the program label.
    y,x=np.mgrid[:size,:size].astype(np.float32); a=rng.uniform(0,2*np.pi)
    wave=rng.uniform(0,.06)*np.sin((np.cos(a)*x+np.sin(a)*y)*rng.uniform(.035,.5)+rng.uniform(0,6.28))
    return np.clip(base+field+noise+wave[...,None],0,1)


def _color(rng, background):
    mean=np.asarray(background).mean(axis=(0,1))
    direction=1 if rng.random()<.5 else -1
    contrast=rng.uniform(.17,.48)
    value=np.clip(mean+direction*contrast+rng.uniform(-.10,.10,3),.02,.98)
    # Avoid accidental near-invisibility without tying polarity to identity.
    if np.abs(value-mean).mean()<.10:
        value=np.clip(mean-direction*contrast,.02,.98)
    return value.astype(np.float32)


def _paint(canvas, instance, x, y, color):
    side=instance['alpha'].shape[0]; region=canvas[y:y+side,x:x+side]
    # Multiplying dark foreground RGB erases fine textures. Instead vary its
    # contrast against the local background, equally for either polarity.
    alpha=instance['alpha'][...,None]*instance['material'][...,None]
    region[:]=region*(1-alpha)+color[None,None,:]*alpha


def bbox_from_mask(mask):
    yy,xx=np.nonzero(mask)
    return np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],np.float32) if len(xx) else np.zeros(4,np.float32)


def _frame_support(rgb,parts,rng,size):
    x1,y1,x2,y2=bbox_from_mask(parts>0).astype(int)
    margin=float(rng.choice([0.,.05,.15,.35,.6]))
    dx=int(round((x2-x1)*margin));dy=int(round((y2-y1)*margin))
    crop=[max(0,x1-dx),max(0,y1-dy),min(size,x2+dx),min(size,y2+dy)]
    x1,y1,x2,y2=crop; source=rgb[y1:y2,x1:x2]; labels=parts[y1:y2,x1:x2]
    scale=min(size/source.shape[1],size/source.shape[0])
    width=max(1,int(round(source.shape[1]*scale)));height=max(1,int(round(source.shape[0]*scale)))
    # Mirror real input preparation: isotropic fit and crop-border median padding.
    fitted=cv2.resize(source,(width,height),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    fitted_labels=cv2.resize(labels,(width,height),interpolation=cv2.INTER_NEAREST)
    border=np.concatenate([source[0],source[-1],source[:,0],source[:,-1]],axis=0)
    pad=np.median(border,axis=0).astype(np.uint8)
    framed=np.broadcast_to(pad,(size,size,3)).copy(); framed_labels=np.zeros((size,size),np.int16)
    left=(size-width)//2;top=(size-height)//2
    framed[top:top+height,left:left+width]=fitted
    framed_labels[top:top+height,left:left+width]=fitted_labels
    return framed,framed_labels,{'crop_xyxy':[int(v) for v in crop],'margin_fraction':margin,'scale':scale,'letterbox_xywh':[left,top,width,height],
                                 'pad_rgb':pad.tolist(),'synthetic_ground_truth_used_for_training_crop':True}


class AmbiguousSceneError(RuntimeError):
    """A valid program whose sampled scene does not visibly distinguish labels."""


def _episode_once(index, split='train', seed=1701, support_size=96, query_size=192,
                  positive_probability=.5, visibility_floor=.65, scene_mode='cluttered', nuisance_attempt=0):
    """One independently rendered support; query has zero/one matching instance.

    The same-type target has new geometry and material. It is never a pasted
    copy of the support raster. Distractors include a one-attribute hard negative.
    """
    if index < 0 or support_size<48 or query_size<96 or not 0<=positive_probability<=1 or not .3<=visibility_floor<=1:
        raise ValueError('Invalid episode dimensions/probabilities')
    if scene_mode not in ('cluttered','isolated'):
        raise ValueError('Unknown scene mode')
    p=sample_program(index,split,seed); negative,edit=hard_negative(p); replacement,replacement_edit=_second_negative(p,negative)
    rng=_rng(seed,index,SPLIT_NUISANCE_CODES[split],419) if nuisance_attempt==0 else _rng(seed,index,SPLIT_NUISANCE_CODES[split],419,nuisance_attempt)
    support_seed=int(rng.integers(1,2**31)); target_seed=int(rng.integers(1,2**31))
    support_inst=render_instance(p,support_seed,support_size)
    support=_background(rng,support_size); support_color=_color(rng,support)
    _paint(support,support_inst,0,0,support_color)
    support_rgb,support_parts,support_frame=_frame_support(np.rint(np.clip(support,0,1)*255).astype(np.uint8),support_inst['parts'],rng,support_size)
    query=_background(rng,query_size); unpainted_query=query.copy(); present=bool(rng.random()<positive_probability)
    target_mask=np.zeros((query_size,query_size),np.uint8); query_parts=np.zeros_like(target_mask,np.int16)
    amodal=target_mask.copy(); occupancy=np.zeros_like(target_mask,np.int16)
    target_side=int(rng.uniform(.12,.68)*query_size); target_side=max(24,target_side)
    # The selected hard-negative edit must survive final raster sampling.
    # This guards against labelling indistinguishable tiny rasters differently.
    while True:
        target_inst=render_instance(p,target_seed,target_side)
        negative_inst=render_instance(negative,target_seed,target_side)
        replacement_inst=render_instance(replacement,target_seed,target_side)
        changed=(target_inst['mask']!=negative_inst['mask']) | ((target_inst['mask']>0)&
                  (np.abs(target_inst['material']-negative_inst['material'])>.15))
        changed2=(target_inst['mask']!=replacement_inst['mask']) | ((target_inst['mask']>0)&
                   (np.abs(target_inst['material']-replacement_inst['material'])>.15))
        if min(int(changed.sum()),int(changed2.sum()))>=8 and min(changed.sum(),changed2.sum())/max(1,target_inst['mask'].sum())>=.003:
            break
        target_side+=8
        if target_side>query_size-8:
            raise AmbiguousSceneError('Declared hard-negative edit is not visually resolvable')
    target_x=int(rng.integers(2,query_size-target_side-1)); target_y=int(rng.integers(2,query_size-target_side-1))
    target_color=_color(rng,query)
    instance_records=[]; negative_id=program_id(negative)
    count=int(rng.integers(2,6)) if scene_mode=='cluttered' else 1
    # Background instances appear before the target; one explicit hard negative.
    for j in range(count):
        other=negative if j==0 else sample_program(index*13+j+104729,split,seed)
        if contains_program(other,p): other=negative
        side=target_side if j==0 else max(28,int(rng.uniform(.19,.49)*query_size))
        x=int(rng.integers(0,query_size-side+1)); y=int(rng.integers(0,query_size-side+1))
        inst_seed=target_seed if j==0 else int(rng.integers(1,2**31)); inst=negative_inst if j==0 else render_instance(other,inst_seed,side)
        _paint(query,inst,x,y,target_color if j==0 else _color(rng,query[y:y+side,x:x+side]))
        occupied=occupancy[y:y+side,x:x+side]; occupied[inst['mask']>0]=j+1
        instance_records.append({'program_id':program_id(other),'family':other['family'],'role':'hard_negative' if j==0 else 'distractor',
                                 'seed':inst_seed,'placement_xywh':[x,y,side,side]})
    # The slot always contains an object. Absence is a relationship-changing
    # replacement, not an easier query with one fewer object.
    slot_background=query[target_y:target_y+target_side,target_x:target_x+target_side].copy()
    counterfactual_rgb=[]
    for candidate_inst in (target_inst,replacement_inst):
        patch=slot_background.copy();_paint(patch,candidate_inst,0,0,target_color)
        counterfactual_rgb.append(np.rint(np.clip(patch,0,1)*255).astype(np.int16))
    rgb_changed=np.max(np.abs(counterfactual_rgb[0]-counterfactual_rgb[1]),axis=2)>=3
    if rgb_changed.sum()<8:
        raise AmbiguousSceneError('Matched presence counterfactual is not visibly distinct in final RGB')
    inst=target_inst if present else replacement_inst
    _paint(query,inst,target_x,target_y,target_color)
    occupied=occupancy[target_y:target_y+target_side,target_x:target_x+target_side]
    occupied[inst['mask']>0]=count+1
    target_mask[target_y:target_y+target_side,target_x:target_x+target_side]=inst['mask']
    query_parts[target_y:target_y+target_side,target_x:target_x+target_side]=inst['parts']
    amodal=target_mask.copy()
    gate_positive=np.zeros_like(target_mask); gate_negative=np.zeros_like(target_mask)
    gate_positive[target_y:target_y+target_side,target_x:target_x+target_side]=target_inst['mask']
    gate_negative[target_y:target_y+target_side,target_x:target_x+target_side]=replacement_inst['mask']
    gate_positive_area=int(gate_positive.sum()); gate_negative_area=int(gate_negative.sum())
    detail_gates=[]
    for local_detail in (changed,changed2,rgb_changed):
        global_detail=np.zeros_like(target_mask)
        global_detail[target_y:target_y+target_side,target_x:target_x+target_side]=local_detail
        detail_gates.append(global_detail)
    instance_records.append({'program_id':program_id(p if present else replacement),'family':p['family'],
                             'role':'target' if present else 'matched_absence_replacement','seed':target_seed,
                             'placement_xywh':[target_x,target_y,target_side,target_side]})
    occluders=0
    if scene_mode=='cluttered':
        # Accept only occlusion whose final visible mask satisfies the fixed floor.
        for j in range(int(rng.integers(0,3))):
            other=negative if j==0 else sample_program(index*17+j+982451653,split,seed)
            if contains_program(other,p): other=negative
            side=max(28,int(rng.uniform(.16,.38)*query_size))
            if rng.random()<.6:
                x=int(np.clip(target_x+rng.uniform(-.3,.9)*target_side,0,query_size-side))
                y=int(np.clip(target_y+rng.uniform(-.3,.9)*target_side,0,query_size-side))
            else:
                x=int(rng.integers(0,query_size-side+1)); y=int(rng.integers(0,query_size-side+1))
            inst_seed=int(rng.integers(1,2**31)); inst=render_instance(other,inst_seed,side)
            covered=np.zeros_like(target_mask); covered[y:y+side,x:x+side]=inst['mask']
            remaining=target_mask & (1-covered)
            next_positive=gate_positive & (1-covered); next_negative=gate_negative & (1-covered)
            if min(next_positive.sum()/max(1,gate_positive_area),next_negative.sum()/max(1,gate_negative_area))<visibility_floor:
                continue
            # All antialiased occluder pixels count as potentially destructive.
            # Apply this same gate to both presence counterfactuals so neither
            # object count nor occluder placement can reveal the label.
            affected=np.zeros_like(target_mask);affected[y:y+side,x:x+side]=inst['alpha']>0
            next_details=[detail & (1-affected) for detail in detail_gates]
            if min(int(detail.sum()) for detail in next_details)<8:
                continue
            _paint(query,inst,x,y,_color(rng,unpainted_query[y:y+side,x:x+side]))
            occupied=occupancy[y:y+side,x:x+side]; occupied[inst['mask']>0]=count+2+j
            target_mask=remaining; query_parts[covered>0]=0; occluders+=1
            gate_positive=next_positive; gate_negative=next_negative
            detail_gates=next_details
            instance_records.append({'program_id':program_id(other),'family':other['family'],'role':'occluder','seed':inst_seed,
                                     'placement_xywh':[x,y,side,side]})
    # No artificial blur. Both palettes and background fields are independent.
    query_rgb=np.rint(np.clip(query,0,1)*255).astype(np.uint8)
    slot_visibility=float(target_mask.sum()/amodal.sum())
    visibility=slot_visibility if present else 0.
    if not present:
        target_mask[:]=0; amodal[:]=0; query_parts[:]=0
    return {'support':support_rgb,'query':query_rgb,'support_mask':(support_parts>0).astype(np.uint8),
            'query_foreground_mask':(occupancy>0).astype(np.uint8),
            'target_mask':target_mask,'amodal_target_mask':amodal,'support_parts':support_parts,
            'query_parts':query_parts,'bbox':bbox_from_mask(target_mask),'amodal_bbox':bbox_from_mask(amodal),
            'present':present,'visibility':visibility,'program_id':program_id(p),'family':p['family'],'program':p,
            'metadata':{'format':FORMAT,'index':int(index),'split':split,'seed':int(seed),'config_sha256':config_hash(),
                        'support_instance_seed':support_seed,'query_instance_seed':target_seed,
                        'support_frame':support_frame,
                        'same_raster_copy':False,'hard_negative_id':negative_id,'hard_negative_edit':edit,
                        'support_color':support_color.tolist(),'target_color':target_color.tolist(),
                        'matched_negative_nuisance': 'same instance seed, scale, material seed and color as the target counterfactual; independent placement',
                        'hard_negative_changed_pixels':int(changed.sum()),'target_render_side':target_side,
                        'absence_replacement_id':program_id(replacement),'absence_replacement_edit':replacement_edit,
                        'absence_replacement_changed_pixels':int(changed2.sum()),'slot_visibility':slot_visibility,
                        'visible_hard_negative_detail_pixels':int(detail_gates[0].sum()),
                        'visible_absence_detail_pixels':int(detail_gates[1].sum()),
                        'visible_counterfactual_rgb_difference_pixels_lower_bound':int(detail_gates[2].sum()),
                        'occluders':occluders,'instances':instance_records,'scene_mode':scene_mode,
                        'pixel_correspondence_claimed':False,'natural_images_used':0}}


def episode(index, split='train', seed=1701, support_size=96, query_size=192,
            positive_probability=.5, visibility_floor=.65, scene_mode='cluttered'):
    """Render one valid reference/scene episode; record bounded ambiguity retries.

    Rejected nuisance draws keep exactly the same structural program. Rejection
    tests both potential presence labels before either is painted, so positive
    and absent counterfactuals use identical draw/acceptance decisions.
    """
    rejected_reasons=[]
    for attempt in range(16):
        try:
            row=_episode_once(index,split,seed,support_size,query_size,positive_probability,
                              visibility_floor,scene_mode,nuisance_attempt=attempt)
            row['metadata']['ambiguous_nuisance_draws_rejected']=attempt
            row['metadata']['ambiguous_nuisance_rejection_reasons']=rejected_reasons
            return row
        except AmbiguousSceneError as exc:
            rejected_reasons.append(str(exc))
            if attempt==15:
                raise


def batch(indices, split='train', seed=1701, support_size=96, query_size=192, device=None, **kwargs):
    """Convenience torch adapter; CPU episode() is the canonical renderer."""
    import torch
    rows=[episode(int(i),split,seed,support_size,query_size,**kwargs) for i in indices]
    if not rows: raise ValueError('Empty batch')
    out={}
    for key in ('support','query'):
        out[key]=torch.from_numpy(np.stack([r[key] for r in rows])).permute(0,3,1,2).float().div_(255)
    for key in ('support_mask','target_mask','amodal_target_mask','query_foreground_mask'):
        out[key]=torch.from_numpy(np.stack([r[key] for r in rows]))[:,None].float()
    for key in ('support_parts','query_parts'):
        out[key]=torch.from_numpy(np.stack([r[key] for r in rows])).long()
    for key in ('bbox','amodal_bbox'):
        out[key]=torch.from_numpy(np.stack([r[key] for r in rows])).float()
    out['present']=torch.tensor([r['present'] for r in rows],dtype=torch.float32)
    out['visibility']=torch.tensor([r['visibility'] for r in rows],dtype=torch.float32)
    out['metadata']=[{**r['metadata'],'program_id':r['program_id'],'family':r['family'],'program':r['program']} for r in rows]
    if device is not None:
        out={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in out.items()}
    return out
