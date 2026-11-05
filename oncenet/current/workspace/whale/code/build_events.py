"""Label-free event bank for HyperNet-W v2 training. For each unlabeled corpus
window, keep the largest connected blob of pixels clearly above the whitened
background (threshold 60/255, 2x3 opening, area >= 40 px, 0.2 to 3 s long,
at least 3 rows tall). Stores the blob's pixels (masked to the blob, dilated
by one pixel) and its row/column box. No labels are read."""
import numpy as np
from scipy import ndimage as nd

C = np.load('../corpus.npy', mmap_mode='r')
TH, MIN_AREA = 60, 40
crops, boxes, src = [], [], []
for i0 in range(0, len(C), 2000):
    X = np.asarray(C[i0:i0 + 2000])
    for k, w in enumerate(X):
        m = nd.binary_opening(w > TH, np.ones((2, 3)))
        lab, n = nd.label(m)
        if n == 0:
            continue
        area = nd.sum(m, lab, range(1, n + 1)); j = int(area.argmax())
        if area[j] < MIN_AREA:
            continue
        sl = nd.find_objects(lab)[j]
        h, l = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h < 3 or not (8 <= l <= 120):
            continue
        r0, r1 = max(0, sl[0].start - 1), min(140, sl[0].stop + 1)
        c0, c1 = max(0, sl[1].start - 1), min(160, sl[1].stop + 1)
        mk = nd.binary_dilation(lab == j + 1)[r0:r1, c0:c1]
        crops.append((w[r0:r1, c0:c1] * mk).astype(np.uint8)); boxes.append((r0, r1, c0, c1)); src.append(i0 + k)
    print(i0, len(crops), flush=True)
np.save('../events_boxes.npy', np.array(boxes, np.int16)); np.save('../events_src.npy', np.array(src, np.int32))
np.save('../events_crops.npy', np.array(crops, dtype=object), allow_pickle=True)
print('events', len(crops))
