"""Procedural 2D graph objects with independent geometry and rendering.

The label is a construction program, not a natural semantic category. Paired
instances share connectivity and part shape parameters, never an RGB raster.
Canonical sample points supervise final descriptors during training only.
"""
import hashlib
import json
import math
from functools import lru_cache
import cv2
import numpy as np

FORMAT = 'hypernet.graph_curriculum.v19.1'
POINTS = 64
CONFIG = {'format': FORMAT, 'support_size': 128, 'query_size': 256,
          'point_count': POINTS, 'natural_images': 0, 'borrowed_weights': False,
          'identity': 'Typed planar construction graph and part parameters',
          'positive': 'Independent geometry, material, lighting and background draws',
          'negative': 'One edge addition/deletion or one local micro-hole pattern change, under matched rendering',
          'split': 'Disjoint program seed namespaces, no natural category names',
          'canonical_edit_geometry_seeds': [1971901, 1971907, 1971913, 1971919, 1971923, 1971929, 1971933, 1971937],
          'canonical_edit_min_alpha_pixels_every_seed': 12,
          'max_nuisance_attempts': 32}
SPLITS = {'train': 101, 'dev': 307, 'test': 509, 'compositional': 701}


def rng_for(*values):
    return np.random.default_rng(np.random.SeedSequence([int(x) for x in values]))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def program(index, split='train', seed=1901):
    """Each seed specifies a reusable graph type, independent of render seeds."""
    rng = rng_for(seed, SPLITS[split], index, 11)
    n = int(rng.integers(4, 10))
    # Distinct radial/branch, ladder, and contour layouts. All remain 2D.
    family = int(rng.integers(0, 4))
    if family == 0:
        angles = np.linspace(0, 2*np.pi, n-1, endpoint=False) + rng.uniform(-.15, .15, n-1)
        nodes = np.concatenate([np.zeros((1, 2)), .62*np.stack([np.cos(angles), np.sin(angles)], 1)])
        edges = [[0, i] for i in range(1, n)]
    elif family == 1:
        nodes = np.stack([np.where(np.arange(n)%2, .45, -.45), np.linspace(-.7, .7, n)], 1)
        edges = [[i, i+2] for i in range(n-2)] + [[i, i+1] for i in range(0, n-1, 2)]
    elif family == 2:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = rng.uniform(.45, .72, n)
        nodes = np.stack([np.cos(angles)*radii, np.sin(angles)*radii], 1)
        edges = [[i, (i+1)%n] for i in range(n)]
    else:
        nodes = np.stack([np.linspace(-.7, .7, n), rng.uniform(-.5, .5, n)], 1)
        edges = [[i, i+1] for i in range(n-1)]
        if n > 5: edges.append([1, n-2])
    nodes += rng.uniform(-.055, .055, nodes.shape)
    radii = rng.uniform(.055, .16, (n, 2))
    shapes = rng.integers(0, 3, n).tolist()
    marks = rng.choice(4, n, p=[.45, .25, .20, .10]).tolist()
    curves = rng.uniform(-.25, .25, len(edges)).tolist()
    value = {'family': family, 'nodes': nodes.round(5).tolist(), 'edges': edges,
             'radii': radii.round(5).tolist(), 'shapes': shapes, 'marks': marks,
             'curves': [round(x, 5) for x in curves],
             'edge_width': round(float(rng.uniform(.022, .065)), 5)}
    if split == 'compositional':
        # A denser graph composition than training: extra long-range chords.
        for i in range(min(3, n-3)):
            edge = [i, (i+3)%n]
            if edge not in value['edges'] and edge[::-1] not in value['edges']:
                value['edges'].append(edge); value['curves'].append(float(rng.uniform(-.25, .25)))
    return value


def _near_program_raw(value, variant=0):
    changed = json.loads(json.dumps(value))
    rng = rng_for(int(digest(value)[:8], 16), 37)
    existing = {tuple(sorted(edge)) for edge in value['edges']}
    possible = [(i, j) for i in range(len(value['nodes'])) for j in range(i+1, len(value['nodes']))
                if (i, j) not in existing]
    choices = [('remove', i) for i in range(len(value['edges']))]+[('add', edge) for edge in possible]
    choices += [('mark', (i, m)) for i, current in enumerate(value['marks']) for m in range(4) if m != current]
    choice = choices[int(rng.permutation(len(choices))[variant % len(choices)])]
    if choice[0] == 'add':
        edge = choice[1]
        changed['edges'].append(list(edge)); changed['curves'].append(0.)
        edit = {'operation': 'add_edge', 'edge': list(edge)}
    elif choice[0] == 'remove':
        k = choice[1]
        edge = changed['edges'].pop(k); changed['curves'].pop(k)
        edit = {'operation': 'remove_edge', 'edge': edge}
    else:
        i, mark = choice[1]
        changed['marks'][i] = mark
        edit = {'operation': 'change_node_mark', 'node': i, 'from': value['marks'][i], 'to': mark}
    return changed, edit


@lru_cache(maxsize=1024)
def _visible_near_json(serialized, variant):
    value = json.loads(serialized)
    seeds = CONFIG['canonical_edit_geometry_seeds']
    baselines = [instance(value, seed, 96, geometry_only=True) for seed in seeds]
    count = len(value['nodes'])*(len(value['nodes'])-1)//2+3*len(value['nodes'])
    accepted = 0
    for offset in range(count):
        candidate, edit = _near_program_raw(value, offset)
        robust = True
        for seed, baseline in zip(seeds, baselines):
            other = instance(candidate, seed, 96, geometry_only=True)
            if int((np.abs(baseline['alpha']-other['alpha']) > .25).sum()) < CONFIG['canonical_edit_min_alpha_pixels_every_seed']:
                robust = False
                break
        if not robust:
            continue
        if accepted == variant:
            return json.dumps(candidate, sort_keys=True), json.dumps(edit, sort_keys=True)
        accepted += 1
    raise ValueError('Construction has insufficient visibly distinct structural edits')


def near_program(value, variant=0):
    candidate, edit = _visible_near_json(json.dumps(value, sort_keys=True), variant)
    return json.loads(candidate), json.loads(edit)


def field(rng, side):
    """Multiscale Gaussian random field; includes both coarse and fine bands."""
    result = np.zeros((side, side), np.float32)
    for grid, amplitude in ((3, .45), (7, .25), (17, .16), (43, .09)):
        source = rng.normal(size=(grid, grid)).astype(np.float32)
        result += amplitude*cv2.resize(source, (side, side), interpolation=cv2.INTER_LINEAR)
    return result


def microtexture(rng, side):
    """Localized, sharp 2D wave packets; structured variation rather than blur."""
    y, x = np.mgrid[0:1:complex(side), 0:1:complex(side)]
    result = np.zeros((side, side), np.float32)
    for _ in range(4):
        angle = float(rng.uniform(-np.pi, np.pi))
        frequency = float(rng.uniform(max(4, side/30), max(6, side/7)))
        phase = float(rng.uniform(-np.pi, np.pi))
        wave = np.tanh(rng.uniform(1., 3.)*np.sin(2*np.pi*frequency*(x*np.cos(angle)+y*np.sin(angle))+phase))
        center = rng.uniform(0, 1, 2); width = float(rng.uniform(.18, .65))
        envelope = np.exp(-((x-center[0])**2+(y-center[1])**2)/(2*width*width))
        result += (wave*envelope*rng.uniform(.25, .55)).astype(np.float32)
    return result


def background(rng, side, difficulty=1.):
    base = np.clip(rng.uniform(.2, .8)+rng.normal(0, .055, 3), .08, .92).astype(np.float32)
    texture = field(rng, side)
    image = np.broadcast_to(base, (side, side, 3)).copy()
    image += texture[..., None]*rng.uniform(.035, .20)*difficulty
    image += microtexture(rng, side)[..., None]*rng.uniform(.025, .14)*difficulty
    # Independent occluding planar shapes supply edges, junctions and textures.
    for _ in range(int(rng.integers(8, 30)*difficulty)):
        color = np.clip(base+rng.normal(0, .15, 3), .03, .97)
        center = tuple(rng.integers(0, side, 2).tolist())
        axes = tuple(rng.integers(max(2, side//35), max(4, side//3), 2).tolist())
        if rng.random() < .55:
            cv2.ellipse(image, center, axes, float(rng.uniform(0, 180)), 0, 360, color.tolist(), -1)
        else:
            endpoint = tuple(np.clip(np.asarray(center)+rng.integers(-side//2, side//2, 2), 0, side-1).tolist())
            cv2.line(image, center, endpoint, color.tolist(), int(rng.integers(1, max(2, side//14))))
    image += rng.normal(0, .008+.012*difficulty, image.shape).astype(np.float32)
    return np.clip(image, 0, 1)


def instance(value, seed, side, difficulty=1., geometry_only=False):
    """Render an independent instance, with exact homologous sample locations."""
    rng = rng_for(seed, 83)
    factor = 2; high = side*factor
    nodes = np.asarray(value['nodes'], np.float32).copy()
    nodes += rng.normal(0, .012+.05*difficulty, nodes.shape)
    angle = rng.uniform(-.08-.25*difficulty, .08+.25*difficulty)
    stretch = np.exp(rng.uniform(-.10-.2*difficulty, .10+.2*difficulty, 2))
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])*stretch
    nodes = nodes@matrix.T
    nodes *= min(1., .79/(np.abs(nodes).max()+.16))
    labels = np.zeros((high, high), np.int16)
    points = []; point_parts = []

    def xy(coords):
        return np.rint((np.asarray(coords)+1)*.5*(high-1)).astype(np.int32)

    for e, (a, b) in enumerate(value['edges']):
        edge_rng = rng_for(seed, min(a, b), max(a, b), 97)
        start, end = nodes[a], nodes[b]; delta = end-start
        normal = np.array([-delta[1], delta[0]])
        bend = value['curves'][e]+edge_rng.normal(0, .04*difficulty)
        t = np.linspace(0, 1, 41)[:, None]
        curve = start+(end-start)*t+np.sin(np.pi*t)*normal*bend
        width = max(2, round(value['edge_width']*high*edge_rng.uniform(.8, 1.3)))
        part = e+1
        cv2.polylines(labels, [xy(curve)], False, part, width, cv2.LINE_8)
        for k in (8, 14, 20, 26, 32):
            points.append(curve[k]); point_parts.append(part)
    edge_count = len(value['edges'])
    for i, center in enumerate(nodes):
        node_rng = rng_for(seed, i, 103)
        radii = np.asarray(value['radii'][i])*node_rng.uniform(.8, 1.25, 2)
        theta = np.linspace(0, 2*np.pi, 49)[:-1]
        shape = value['shapes'][i]
        power = (1., .55, 1.6)[shape]
        outline = np.stack([np.sign(np.cos(theta))*np.abs(np.cos(theta))**power,
                            np.sign(np.sin(theta))*np.abs(np.sin(theta))**power], 1)*radii
        outline = outline@matrix.T+center
        part = edge_count+i+1
        cv2.fillPoly(labels, [xy(outline)], part, cv2.LINE_8)
        # Type-defining small details coexist with independently randomized
        # surface texture. Learning to ignore every tiny pattern cannot solve
        # the micro-hole counterfactuals.
        mark = value['marks'][i]
        if mark:
            offsets = [np.zeros(2)] if mark == 1 else [np.array([-.36, 0]), np.array([.36, 0])] if mark == 2 else [
                .38*np.array([np.cos(t), np.sin(t)]) for t in (0, 2*np.pi/3, 4*np.pi/3)]
            hole_scale = .32 if mark == 1 else .21 if mark == 2 else .17
            for offset in offsets:
                hole = (np.stack([np.cos(theta), np.sin(theta)], 1)*hole_scale+offset)*radii
                hole = hole@matrix.T+center
                cv2.fillPoly(labels, [xy(hole)], 0, cv2.LINE_8)
        for k in (0, 6, 12, 18, 24, 30, 36, 42):
            # Slightly inside the boundary; stable part visibility after resize.
            points.append(.88*outline[k]+.12*center); point_parts.append(part)
    all_points = np.asarray(points, np.float32)
    # Same graph -> identical canonical sampling indices across instances.
    selected = np.linspace(0, len(points)-1, POINTS).round().astype(int)
    point_xy = (all_points[selected]+1)*.5*(side-1)
    point_parts = np.asarray(point_parts, np.int16)[selected]
    mask_high = labels > 0
    if geometry_only:
        return {'alpha': cv2.resize(mask_high.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)}
    # Shared material direction but independently drawn instance appearance.
    material = np.empty((high, high, 3), np.float32)
    base = np.clip(rng.uniform(.08, .92)+rng.normal(0, .075, 3), .03, .97).astype(np.float32)
    material[:] = base
    texture = field(rng, high)
    y, x = np.mgrid[-1:1:complex(high), -1:1:complex(high)]
    phi = float(rng.uniform(-np.pi, np.pi)); freq = float(rng.uniform(3, 28))
    stripe = np.sin(freq*(x*np.cos(phi)+y*np.sin(phi))+rng.uniform(-np.pi, np.pi))
    lighting = rng.uniform(-.17, .17)*x+rng.uniform(-.17, .17)*y
    material += (texture*rng.uniform(.02, .16)+stripe*rng.uniform(0, .06)+lighting)[..., None]
    material += microtexture(rng, high)[..., None]*rng.uniform(.035, .18)*difficulty
    # Per-part coloration is a nuisance, independent across the two instances.
    if rng.random() < .5:
        for part in range(1, edge_count+len(nodes)+1):
            if part <= edge_count:
                a, b = value['edges'][part-1]
                color_rng = rng_for(seed, min(a, b), max(a, b), 107)
            else:
                color_rng = rng_for(seed, part-edge_count-1, 109)
            material[labels == part] += color_rng.normal(0, .055, 3)
    mask = cv2.resize(mask_high.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(np.clip(material, 0, 1), (side, side), interpolation=cv2.INTER_AREA)
    part_map = cv2.resize(labels, (side, side), interpolation=cv2.INTER_NEAREST)
    px = np.clip(np.rint(point_xy[:, 0]).astype(int), 0, side-1)
    py = np.clip(np.rint(point_xy[:, 1]).astype(int), 0, side-1)
    valid = part_map[py, px] == point_parts
    # Fixed-size padding must not turn repeated sample indices into additional
    # correspondence supervision. The consumer ignores invalid slots.
    valid &= np.r_[True, selected[1:] != selected[:-1]]
    return {'rgb': rgb, 'alpha': mask, 'parts': part_map, 'points': point_xy,
            'point_parts': point_parts, 'point_valid': valid}


def mask_box(mask):
    y, x = np.nonzero(mask > .25)
    if not len(x): return np.zeros(4, np.float32)
    return np.array([x.min(), y.min(), x.max()+1, y.max()+1], np.float32)


def paint(canvas, item, x, y):
    side = item['alpha'].shape[0]
    region = canvas[y:y+side, x:x+side]
    alpha = item['alpha'][..., None]
    region[:] = region*(1-alpha)+item['rgb']*alpha


def finish(image, rng, difficulty):
    # Camera-like effects contain no photographic assets or learned generators.
    out = np.clip(image, 0, 1)
    gamma = float(rng.uniform(.75, 1.3))
    out = out**gamma
    out += rng.normal(0, rng.uniform(0, .018)*difficulty, out.shape)
    out = np.rint(np.clip(out, 0, 1)*255).astype(np.uint8)
    if rng.random() < .25*difficulty:
        out = cv2.GaussianBlur(out, (3, 3), float(rng.uniform(.25, .65)))
    if rng.random() < .3*difficulty:
        ok, data = cv2.imencode('.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        if not ok: raise RuntimeError('Synthetic JPEG encoding failed')
        out = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


class AmbiguousRenderError(RuntimeError):
    pass


def _episode_once(index, split='train', seed=1901, difficulty=1., attempt=0):
    if index < 0 or split not in SPLITS or not 0 <= difficulty <= 1:
        raise ValueError('Invalid procedural episode request')
    rng = rng_for(seed, SPLITS[split], index, 173, attempt)
    p = program(index, split, seed); near, edit = near_program(p)
    replacement, replacement_edit = near_program(p, 1)
    positive = bool(index % 2 == 0)
    support = background(rng_for(seed, SPLITS[split], index, 181, attempt), 128, difficulty)
    support_background = support.copy()
    query = background(rng_for(seed, SPLITS[split], index, 191, attempt), 256, difficulty)
    counterfactual_queries = [query.copy(), query.copy()]
    support_seed = int(rng.integers(1, 2**31))
    sside = int(rng.integers(108, 125)); sx = (128-sside)//2; sy = (128-sside)//2
    support_item = instance(p, support_seed, sside, difficulty)
    paint(support, support_item, sx, sy)
    support_mask = np.zeros((128, 128), np.float32)
    support_mask[sy:sy+sside, sx:sx+sside] = support_item['alpha']
    support_points = support_item['points']+np.array([sx, sy])
    support_valid = support_item['point_valid'].copy()
    programs = [p if positive else replacement, near,
                program(index*11+1000003, split, seed), program(index*13+2000003, split, seed)]
    roles = ['target' if positive else 'replacement', 'near', 'other', 'other']
    target_mask = np.zeros((256, 256), np.float32)
    object_mask = np.zeros_like(target_mask)
    boxes = []; placement_records = []
    query_points = np.zeros((POINTS, 2), np.float32)
    query_valid = np.zeros(POINTS, bool)
    target_amodal_area = 0.
    for j, value in enumerate(programs):
        side = int(rng.integers(48, 177))
        object_seed = int(rng.integers(1, 2**31))
        item = instance(value, object_seed, side, difficulty)
        local_box = mask_box(item['alpha'])
        best = None
        # Free positions and broad scale support. Pick the least-overlapping
        # draw if all32 candidates are crowded; no type or episode is replaced.
        for placement_attempt in range(32):
            x = int(rng.integers(0, 257-side)); y = int(rng.integers(0, 257-side))
            box = local_box+np.array([x, y, x, y])
            overlap = 0.
            for old in boxes:
                area = np.maximum(0, np.minimum(box[2:], old[2:])-np.maximum(box[:2], old[:2])).prod()
                overlap = max(overlap, float(area/min(np.prod(box[2:]-box[:2]), np.prod(old[2:]-old[:2]))))
            if best is None or overlap < best[0]: best = (overlap, x, y, box)
            if overlap <= .12: break
        overlap, x, y, box = best
        paint(query, item, x, y)
        for cf_index, cf_image in enumerate(counterfactual_queries):
            if j == cf_index:
                alternate = replacement if j == 0 and positive else p
                cf_item = instance(alternate, object_seed, side, difficulty)
                paint(cf_image, cf_item, x, y)
            else:
                paint(cf_image, item, x, y)
        object_mask[y:y+side, x:x+side] = np.maximum(object_mask[y:y+side, x:x+side], item['alpha'])
        if j > 0:
            target_mask[y:y+side, x:x+side] *= 1-item['alpha']
        boxes.append(box)
        placement_records.append({'role': roles[j], 'program_id': digest(value), 'seed': object_seed,
                                  'placement_xywh': [x, y, side, side], 'box_overlap': overlap})
        if j == 0 and positive:
            target_mask[y:y+side, x:x+side] = item['alpha']
            target_amodal_area = float(item['alpha'].sum())
            query_points = item['points']+np.array([x, y])
            query_valid = item['point_valid'].copy()
    if positive:
        px = np.clip(np.rint(query_points[:, 0]).astype(int), 0, 255)
        py = np.clip(np.rint(query_points[:, 1]).astype(int), 0, 255)
        query_valid &= target_mask[py, px] > .5
    support_rgb = finish(support, rng_for(seed, SPLITS[split], index, 211, attempt), difficulty)
    query_rgb = finish(query, rng_for(seed, SPLITS[split], index, 223, attempt), difficulty)
    support_changed_pixels = []
    for alternate in (near, replacement):
        cf_support = support_background.copy()
        cf_item = instance(alternate, support_seed, sside, difficulty)
        paint(cf_support, cf_item, sx, sy)
        cf_rgb = finish(cf_support, rng_for(seed, SPLITS[split], index, 211, attempt), difficulty)
        support_changed_pixels.append(int((np.max(np.abs(support_rgb.astype(np.int16)-cf_rgb.astype(np.int16)), axis=2) >= 3).sum()))
    changed_pixels = []
    for cf_image in counterfactual_queries:
        cf_rgb = finish(cf_image, rng_for(seed, SPLITS[split], index, 223, attempt), difficulty)
        changed_pixels.append(int((np.max(np.abs(query_rgb.astype(np.int16)-cf_rgb.astype(np.int16)), axis=2) >= 3).sum()))
    visibility = float(target_mask.sum()/max(target_amodal_area, 1.)) if positive else None
    if min(changed_pixels) < 8:
        raise AmbiguousRenderError('A structural edit is not visible after rendering/occlusion/camera effects')
    if min(support_changed_pixels) < 8:
        raise AmbiguousRenderError('A defining structural edit is not visible in the single support')
    if positive and (visibility < .65 or int((support_valid & query_valid).sum()) < 16):
        raise AmbiguousRenderError('Too little visible target/correspondence evidence')
    result = {'support': support_rgb, 'query': query_rgb,
              'support_mask': support_mask, 'query_target_mask': target_mask,
              'query_object_mask': object_mask, 'boxes': np.asarray(boxes, np.float32), 'roles': roles,
              'target_box': np.asarray(boxes[0] if positive else [0, 0, 0, 0], np.float32),
              'positive': positive, 'support_points': support_points.astype(np.float32),
              'query_points': query_points.astype(np.float32), 'support_point_valid': support_valid,
              'query_point_valid': query_valid, 'point_valid': support_valid & query_valid,
              'program_id': digest(p)}
    result['metadata'] = {'format': FORMAT, 'index': int(index), 'split': split, 'seed': int(seed),
                          'difficulty': float(difficulty), 'program': p, 'program_id': digest(p),
                          'near_program_id': digest(near), 'edit': edit, 'support_seed': support_seed,
                          'replacement_program_id': digest(replacement), 'replacement_edit': replacement_edit,
                          'nuisance_attempt': attempt, 'visible_edit_pixels': changed_pixels,
                          'support_visible_edit_pixels': support_changed_pixels,
                          'target_visibility': visibility,
                          'instances': placement_records, 'positive': positive,
                          'valid_correspondences': int(result['point_valid'].sum()),
                          'image_sha256': {k: hashlib.sha256(result[k].tobytes()).hexdigest() for k in ('support', 'query')},
                          'natural_images': 0, 'borrowed_pretrained_weights': False}
    return result


def episode(index, split='train', seed=1901, difficulty=1.):
    """Retry ambiguous nuisance draws without changing the structural program."""
    rejected = []
    for attempt in range(CONFIG['max_nuisance_attempts']):
        try:
            row = _episode_once(index, split, seed, difficulty, attempt)
            row['metadata']['rejected_nuisance_reasons'] = rejected
            return row
        except AmbiguousRenderError as exc:
            rejected.append(str(exc))
    raise AmbiguousRenderError(f'No valid draw for {split}:{index}; {rejected[-1:]}')
