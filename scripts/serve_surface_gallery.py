"""Local, read-only viewer for every v10 mathematical-surface frame.

Generated display PNGs live outside the frozen dataset. The server binds only
127.0.0.1 and never changes renderer, dataset, model, or experiment evidence.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import threading
import time
from urllib.parse import quote, unquote, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet.mathematical_surfaces import FAMILY_NAMES, FAMILY_LABELS, FORMULAS, render_surface

DATA = ROOT/'data/surfaces_v10'
DISPLAY = ROOT/'viewer_assets'
_INDEX_CACHE = {}
_INDEX_LOCK = threading.Lock()


def _url(path):
    return '/'+quote(Path(path).resolve().relative_to(ROOT).as_posix(), safe='/')


def make_preview():
    """Initial review samples are explicitly separate from the training dataset."""
    directory = DISPLAY/'preview_samples'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        return
    rows = []
    for index, family in enumerate(FAMILY_NAMES):
        for view_index, view_seed in enumerate((18, 22)):
            for strength in (0., .03, .07):
                frame = render_surface(family, 410+index, view_seed, strength, 128)
                identifier = f'preview_{family}_v{view_index}_s{round(strength*100):02d}'
                rgb = directory/f'{identifier}.png'
                targets = directory/f'{identifier}.npz'
                metadata = directory/f'{identifier}.json'
                Image.fromarray(frame['rgb']).save(rgb)
                np.savez_compressed(targets, **{key: frame[key] for key in ('mask', 'normal', 'depth', 'component')})
                metadata.write_text(json.dumps({key: value for key, value in frame.items()
                    if key not in ('rgb', 'mask', 'normal', 'depth', 'component')}, indent=2), encoding='utf-8')
                rows.append({'id': identifier, 'split': 'preview', 'family': family,
                    'family_index': index, 'object_id': frame['base_program_id'], 'object_seed': 410+index,
                    'view_index': view_index, 'view_seed': view_seed, 'surface_strength': strength,
                    'rgb_path': rgb.name, 'target_path': targets.name, 'metadata_path': metadata.name})
    manifest_path.write_text(json.dumps({'completed': True, 'preview_only': True, 'records': rows}), encoding='utf-8')


def _build_gallery_index():
    choices = ((DATA/'manifest.json', False, True), (DATA/'plan.json', False, False),
               (DISPLAY/'preview_samples/manifest.json', True, False))
    selected = next(((path, preview, completed) for path, preview, completed in choices if path.exists()), None)
    if selected is None:
        return {'records': [], 'expected_frames': 1920, 'completed': False, 'preview_only': True}, {}
    path, preview, completed = selected
    manifest = json.loads(path.read_text(encoding='utf-8'))
    records = manifest.get('records', manifest.get('renders', []))
    rows, targets = [], {}
    for row in records:
        rgb = path.parent/row['rgb_path']
        target = path.parent/row['target_path']
        metadata = path.parent/row['metadata_path']
        if not all(file.is_file() for file in (rgb, target, metadata)):
            continue
        identifier = str(row['id'])
        targets[identifier] = target
        urls = {'rgb': _url(rgb)}
        urls.update({key: f'/preview/{quote(identifier, safe="")}/{key}.png' for key in ('mask', 'normal', 'depth')})
        rows.append({key: row[key] for key in ('id', 'split', 'family', 'family_index', 'object_id',
                    'object_seed', 'view_index', 'view_seed', 'surface_strength')} |
            {'name': FAMILY_LABELS[row['family']], 'formula': FORMULAS[row['family']],
             'view_seed': str(row['view_seed']), 'object_seed': str(row['object_seed']),
             'channels': urls, 'metadata_url': _url(metadata), 'target_url': _url(target)})
    index = {'records': rows, 'expected_frames': 1920, 'completed': bool(completed and len(rows) == 1920),
        'preview_only': preview, 'source_manifest': _url(path),
        'source_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'native_resolution': 128, 'family_names': FAMILY_LABELS,
        'scope': 'All generated frames are browsable. Masks, normal maps and depth here are display diagnostics.',
        'depth_display': 'Linear grayscale: distance1.8=white,4.2=black; background black.'}
    return index, targets


def gallery_index():
    # A channel switch can request thousands of thumbnails. Do not rescan every
    # dataset file for every thumbnail; refresh discovery only once per2seconds.
    now = time.monotonic()
    with _INDEX_LOCK:
        if _INDEX_CACHE.get('value', ({}, {}))[0].get('completed'):
            return _INDEX_CACHE['value']
        if now-_INDEX_CACHE.get('time', -100.) > 2.:
            value = _build_gallery_index()
            _INDEX_CACHE.update(time=time.monotonic(), value=value)
        return _INDEX_CACHE['value']


def target_preview(path, channel):
    with np.load(path, allow_pickle=False) as archive:
        mask = archive['mask'].astype(bool)
        if channel == 'mask':
            pixels = mask.astype(np.uint8)*255
        elif channel == 'normal':
            pixels = np.rint(np.clip((archive['normal'].astype(np.float32)+1)*.5, 0, 1)*255).astype(np.uint8)
            pixels[~mask] = 0
        elif channel == 'depth':
            pixels = np.rint(np.clip((4.2-archive['depth'].astype(np.float32))/2.4, 0, 1)*255).astype(np.uint8)
            pixels[~mask] = 0
        else:
            raise ValueError('Unknown display channel')
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return stream.getvalue()


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        route = urlparse(self.path).path
        if route in ('/', '/gallery'):
            self.path = '/surface_gallery.html'
        elif route == '/api/gallery':
            try:
                index, _ = gallery_index()
                content = json.dumps(index).encode('utf-8')
            except (OSError, ValueError, KeyError) as error:
                self.send_error(503, str(error))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        elif route.startswith('/preview/'):
            pieces = route.split('/')
            if len(pieces) != 4 or pieces[3] not in ('mask.png', 'normal.png', 'depth.png'):
                self.send_error(404)
                return
            identifier, channel = unquote(pieces[2]), pieces[3][:-4]
            _, targets = gallery_index()
            if identifier not in targets:
                self.send_error(404)
                return
            target = targets[identifier]
            key = hashlib.sha256((str(target.resolve())+str(target.stat().st_mtime_ns)+channel).encode()).hexdigest()
            cache = DISPLAY/'target_png'/f'{key}.png'
            try:
                if cache.exists():
                    content = cache.read_bytes()
                else:
                    content = target_preview(target, channel)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    # Identical concurrent requests may write identical bytes;
                    # this cache is outside all experiment evidence.
                    cache.write_bytes(content)
            except (OSError, ValueError) as error:
                self.send_error(503, str(error))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8791)
    parser.add_argument('--prepare-preview', action='store_true')
    args = parser.parse_args()
    if args.prepare_preview:
        make_preview()
        print('Prepared48 review frames outside the training dataset', flush=True)
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT)))
    print(f'Surface gallery: http://127.0.0.1:{args.port}/surface_gallery.html', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
