"""Fresh Gravity Spy O3 cohort: tables, time grouping, sampling, image fetch and conversion.

Source: Zenodo 5649212 (CC BY 4.0), the machine-learning classifications of
every Omicron trigger from O1 to O3b, with four omega-scan PNG links per event.
The v1.1 training set used by v0.1-v0.4 predates O3, so every O3 event is new.

Labels here are the Gravity Spy CNN's predictions (`ml_label`, `ml_confidence`),
not volunteer or expert labels. Only events with confidence at or above the
declared threshold are used, and that circularity is disclosed in the protocol.

Images are the publisher's 800x600 PNGs. Crop x 105:671, y 66:532 (the plot
area), skimage rgb2gray weights (0.2125, 0.7154, 0.0721) and a bilinear resize
to 170x140 reproduce the v1.1 HDF5 arrays exactly (correlation 1.0, zero mean
difference on 25 non-blank v1.1 events), so PNG-derived O3 images and the
HDF5-derived v1.1 references are on one footing.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, time
import numpy as np

ZENODO = 'https://zenodo.org/records/5649212'
TABLE_MD5 = {'H1_O3b.csv': '590290fe3c7ddf8cd9fa85c3a09cd5e4', 'L1_O3b.csv': '90d0bc16448bca72818ed9abf267979c'}
CROP = (105, 66, 671, 532)  # x0, y0, x1, y1 in the 800x600 PNG
GRAY = np.array([.2125, .7154, .0721])
URL_COLUMNS = {'0.5': 'url1', '1.0': 'url2', '2.0': 'url3', '4.0': 'url4'}
GROUP_SECONDS = 4.0
PNG_SIGNATURE = bytes([137, 80, 78, 71, 13, 10, 26, 10])
PNG_IEND = bytes([73, 69, 78, 68, 174, 66, 96, 130])


def png_to_array(path, resample='linear'):
    """800x600 PNG -> float32 (140, 170) in [0, 1], matching the training-set arrays."""
    from PIL import Image
    import cv2
    im = Image.open(path).convert('RGB')
    if im.size != (800, 600):
        raise ValueError(f'{path}: unexpected PNG size {im.size}')
    rgb = np.asarray(im).astype(np.float64) / 255
    x0, y0, x1, y1 = CROP
    gray = (rgb[y0:y1, x0:x1] @ GRAY).astype(np.float32)
    flag = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA}[resample]
    return np.clip(cv2.resize(gray, (170, 140), interpolation=flag), 0, 1)


def load_tables(folder, names=tuple(TABLE_MD5)):
    import pandas as pd
    frames = []
    for name in names:
        path = Path(folder) / name
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 22), b''):
                h.update(block)
        if h.hexdigest() != TABLE_MD5[name]:
            raise ValueError(f'{name} checksum differs from Zenodo')
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def time_groups(table, seconds=GROUP_SECONDS):
    """Group ids: events on the same detector whose event_time differ by <= seconds form a chain."""
    order = np.lexsort((table['event_time'].to_numpy(), table['ifo'].to_numpy()))
    ifo = table['ifo'].to_numpy()[order]; t = table['event_time'].to_numpy()[order]
    new = np.r_[True, (ifo[1:] != ifo[:-1]) | (t[1:] - t[:-1] > seconds)]
    gid = np.cumsum(new) - 1
    out = np.empty(len(table), int); out[order] = gid
    return out


def sample_cohort(table, classes, *, confidence, per_class, calib_per_class, seed, exclude_ids=()):
    """One event per time group; stratified by ml_label; TEST and CALIB from disjoint groups.

    Events whose ml_label is not in `classes` are kept as distractors ('other') at a capped count.
    """
    rng = np.random.default_rng(seed)
    t = table.copy()
    t = t[~t['gravityspy_id'].isin(set(exclude_ids))]
    t = t[t['ml_confidence'] >= confidence]
    t['group'] = time_groups(t)
    t = t.groupby('group', group_keys=False).apply(lambda g: g.sample(1, random_state=int(rng.integers(1 << 31))))
    t['target'] = np.where(t['ml_label'].isin(classes), t['ml_label'], 'other')
    test, calib = [], []
    for label, cap, cap_c in [(c, per_class, calib_per_class) for c in classes] + [('other', per_class, calib_per_class)]:
        rows = t[t['target'] == label]
        rows = rows.iloc[rng.permutation(len(rows))]
        n_test = min(cap, int(np.ceil(len(rows) * .75))); n_calib = min(cap_c, len(rows) - n_test)
        test.append(rows.iloc[:n_test]); calib.append(rows.iloc[n_test:n_test + n_calib])
    import pandas as pd
    test = pd.concat(test); calib = pd.concat(calib)
    assert not set(test['group']) & set(calib['group'])
    return test, calib


def complete_png(data):
    """A PNG that starts with the signature and ends with the IEND chunk."""
    return len(data) > 10_000 and data[:8] == PNG_SIGNATURE and data.rstrip().endswith(PNG_IEND)


def fetch_images(rows, folder, durations=('0.5', '1.0', '2.0', '4.0'), pause=.25, retries=4, log=print):
    """Download each event's PNGs to folder/<gravityspy_id>_<duration>.png; keep only complete files; record sha256."""
    import requests
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    records = {}; failures = []
    session = requests.Session(); session.headers['User-Agent'] = 'HyperNet-research/0.5 (Gravity Spy O3 fresh test)'
    for n, (_, row) in enumerate(rows.iterrows()):
        for d in durations:
            name = f"{row['gravityspy_id']}_{d}.png"; path = folder / name
            if path.exists() and complete_png(path.read_bytes()):
                records[name] = hashlib.sha256(path.read_bytes()).hexdigest(); continue
            url = row[URL_COLUMNS[d]]
            for attempt in range(retries):
                try:
                    r = session.get(url, timeout=60)
                    if r.status_code == 200 and complete_png(r.content):
                        path.write_bytes(r.content); records[name] = hashlib.sha256(r.content).hexdigest(); break
                    raise IOError(f'status {r.status_code}' if r.status_code != 200 else 'incomplete PNG')
                except Exception as e:
                    if attempt == retries - 1:
                        failures.append({'file': name, 'url': url, 'error': str(e)})
                    else:
                        time.sleep(2 * (attempt + 1))
                time.sleep(pause)
        if (n + 1) % 100 == 0:
            log(f'fetched {n + 1}/{len(rows)} events, {len(failures)} failures')
    return records, failures


def write_manifest(path, **fields):
    Path(path).write_text(json.dumps(fields, indent=2, default=str))
