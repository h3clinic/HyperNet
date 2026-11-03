"""Reference-independent region coverage, with no learned proposal filter."""
import math
import cv2
import numpy as np

SCALES = (.10, .16, .24, .36, .54, .80, 1.)
ASPECTS = (.33, .5, .75, 1., 1.5, 2., 3.)
PROPOSAL_CONFIG = {'scales': SCALES, 'aspects': ASPECTS, 'spacing_fraction': .4,
                   'minimum_side': 8, 'edge_limit': 256,
                   'edge_thresholds': (60, 140), 'edge_dilation_sides': (1, 3, 7),
                   'selection': 'All deduplicated generic anchors and bounded edge boxes; no learned filtering'}


def grid_boxes(height, width):
    """Half-open xyxy in the same pixel coordinates as the supplied image."""
    if min(height, width) < 8: raise ValueError('Image dimensions must be >=8')
    result = []
    for scale in SCALES:
        for aspect in ASPECTS:
            w = min(width, max(8, min(height, width)*scale*math.sqrt(aspect)))
            h = min(height, max(8, min(height, width)*scale/math.sqrt(aspect)))
            nx = max(1, int(math.ceil((width-w)/(.4*w)))+1)
            ny = max(1, int(math.ceil((height-h)/(.4*h)))+1)
            for y in np.linspace(0, height-h, ny):
                for x in np.linspace(0, width-w, nx): result.append((x, y, x+w, y+h))
    result.append((0, 0, width, height))
    return deduplicate(np.asarray(result, np.float32))


def deduplicate(boxes):
    if not len(boxes): return np.zeros((0, 4), np.float32)
    # Millipixel key preserves the geometric grid while removing repeated full frames.
    _, index = np.unique(np.rint(np.asarray(boxes)*1000).astype(np.int64), axis=0, return_index=True)
    return np.asarray(boxes, np.float32)[np.sort(index)]


def edge_boxes(rgb, limit=256):
    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    result = []
    for size in (1, 3, 7):
        active = edges if size == 1 else cv2.dilate(edges, np.ones((size, size), np.uint8))
        contours, _ = cv2.findContours(active, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) >= 8 and .1 <= w/h <= 10:
                result.append((x, y, x+w, y+h))
    boxes = deduplicate(np.asarray(result, np.float32).reshape(-1, 4))
    if len(boxes) > limit:
        area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
        # Fixed size-stratified sampling, not target-dependent ranking.
        order = np.argsort(area, kind='stable')
        boxes = boxes[order[np.linspace(0, len(order)-1, limit).round().astype(int)]]
    return boxes


def propose(rgb):
    return deduplicate(np.concatenate([grid_boxes(*rgb.shape[:2]), edge_boxes(rgb)], 0))


def box_iou(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4); b = np.asarray(b, np.float32).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(rb-lt, 0).prod(-1)
    aa = np.maximum(a[:, 2:]-a[:, :2], 0).prod(-1); ba = np.maximum(b[:, 2:]-b[:, :2], 0).prod(-1)
    return inter/np.maximum(aa[:, None]+ba[None, :]-inter, 1e-8)


def nms(boxes, scores, limit=100, threshold=.5):
    order = np.argsort(-np.asarray(scores), kind='stable'); keep = []
    while len(order) and len(keep) < limit:
        i = int(order[0]); keep.append(i)
        if len(order) == 1: break
        remaining = order[1:]
        order = remaining[box_iou(np.asarray(boxes)[i:i+1], np.asarray(boxes)[remaining])[0] <= threshold]
    return np.asarray(keep, np.int64)
