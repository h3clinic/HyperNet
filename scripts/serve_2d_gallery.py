"""Paginated read-only gallery of every v11 2D reference/query pair."""
from __future__ import annotations
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
import argparse
import hashlib
import io
import json
import sys
import threading

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFESTS = (ROOT/'data/shapes2d_v11/manifest.json', ROOT/'assets/grammar_preview/manifest.json')
_LOCK = threading.Lock()
_CACHE = {}
_PREDICTIONS = {}


def prediction_index():
    """Expose completed, hash-bound evaluation outputs; never run model inference."""
    result_path = ROOT/'results/detector_results.json'
    if not result_path.is_file():
        return {}
    bypass_path = ROOT/'results/analytic_detail_bypass_results.json'
    stamp = (str(result_path),result_path.stat().st_mtime_ns,result_path.stat().st_size,
             bypass_path.stat().st_mtime_ns if bypass_path.is_file() else None)
    with _LOCK:
        if _PREDICTIONS.get('stamp') == stamp:
            return _PREDICTIONS['arms']
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if not result.get('completed') or not result.get('sources_unchanged'):
            return {}
        arms = {}
        methods = dict(result.get('arms',{}))
        if bypass_path.is_file():
            bypass = json.loads(bypass_path.read_text(encoding='utf-8'))
            if bypass.get('completed') and bypass.get('weights_and_frozen_sources_unchanged'):
                methods['analytic_detail_bypass'] = bypass
        for arm,info in methods.items():
            path = safe_path(ROOT/'results',arm+'_predictions.jsonl')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != info['predictions_sha256']:
                raise ValueError('Prediction log hash does not match completed results')
            rows = {}
            for line in raw.splitlines():
                if not line:continue
                row = json.loads(line)
                cutoff = info['metrics'][row['split']]['threshold']
                rows[row['id']] = {'score':row['score'],'threshold':cutoff,
                    'accepted':row['score']>=cutoff,'bbox_xyxy':row['bbox_xyxy'],
                    'iou':row['iou']}
            arms[arm] = rows
        _PREDICTIONS.update(stamp=stamp,arms=arms)
        return arms


def safe_path(base, value):
    path = (base/value).resolve()
    path.relative_to(base.resolve())
    return path


def url(path):
    return '/'+quote(path.resolve().relative_to(ROOT).as_posix(), safe='/')


def load_index():
    manifest = next((p for p in MANIFESTS if p.is_file()), None)
    if manifest is None:
        return {'records':[], 'by_id':{}, 'families':{}, 'source':None, 'preview':True,
                'completed':False, 'manifest_sha256':None}
    stamp = (str(manifest), manifest.stat().st_mtime_ns, manifest.stat().st_size)
    with _LOCK:
        if _CACHE.get('stamp') == stamp:
            return _CACHE['index']
        raw = manifest.read_bytes()
        data = json.loads(raw)
        rows, families = [], {}
        for ep in data.get('episodes', []):
            support = ep['support']
            support_path = safe_path(manifest.parent, support['rgb_path'])
            family_id = int(ep['family_id'])
            family_name = ep.get('family_name')
            if not family_name:
                from hypernet.mathematical_shapes2d import family_spec
                family_name = family_spec(family_id)['family_name']
            families[str(family_id)] = {'name':family_name, 'digits':ep.get('grammar_digits')}
            for query in ep['queries']:
                rgb_path = safe_path(manifest.parent, query['rgb_path'])
                if not support_path.is_file() or not rgb_path.is_file():
                    continue
                target_path = safe_path(manifest.parent, query['target_path']) if query.get('target_path') else None
                row = {'id':query['id'], 'episode_id':ep['episode_id'],
                    'split':ep.get('split','preview'), 'family_id':family_id, 'family_name':family_name,
                    'grammar_digits':ep.get('grammar_digits'), 'enrolled_variant':int(ep['enrolled_variant']),
                    'candidate_variant':int(query['candidate_variant']), 'present':bool(query['present']),
                    'support_url':url(support_path), 'query_url':url(rgb_path),
                    'support_bbox_xyxy':support.get('bbox_xyxy'),
                    'target_bbox_xyxy':query.get('target_bbox_xyxy'),
                    'candidate_bbox_xyxy':query.get('candidate_bbox_xyxy'), 'all_boxes':query.get('all_boxes', []),
                    'masks_available':target_path is not None and target_path.is_file(),
                    'target_path':target_path,
                    'support_sha256':support.get('rgb_sha256'), 'query_sha256':query.get('rgb_sha256')}
                rows.append(row)
        rows.sort(key=lambda r:(r['split'], r['family_id'], r['enrolled_variant'], not r['present'], r['id']))
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate gallery query IDs')
        index = {'records':rows, 'by_id':{r['id']:r for r in rows}, 'families':families,
            'source':url(manifest), 'preview':manifest == MANIFESTS[-1],
            'completed':bool(data.get('completed',False)),
            'manifest_sha256':hashlib.sha256(raw).hexdigest()}
        _CACHE.update(stamp=stamp,index=index)
        return index


def public_row(row, predictions=None):
    value = {k:v for k,v in row.items() if k!='target_path'}
    value['all_boxes']=[{k:b[k] for k in ('bbox_xyxy','role','family_id','micro_variant') if k in b}
                        for b in row['all_boxes']]
    value['mask_urls']={key:f'/api/mask/{quote(row["id"],safe="")}/{key}.png'
                        for key in ('union_mask','target_mask','candidate_mask')} if row['masks_available'] else {}
    value['predictions']={arm:records[row['id']] for arm,records in (predictions or {}).items() if row['id'] in records}
    return value


def page(params):
    index = load_index()
    predictions = prediction_index()
    first=lambda key,default='':params.get(key,[default])[0]
    page_number=max(0,int(first('page','0')))
    limit=min(48,max(1,int(first('limit','24'))))
    split,family,present,micro=first('split'),first('family'),first('present'),first('micro')
    exact=first('episode')
    rows=[r for r in index['records'] if (not split or r['split']==split)
          and (not family or r['family_id']==int(family))
          and (not present or r['present']==(present=='yes'))
          and (not micro or r['enrolled_variant']==int(micro))
          and (not exact or r['episode_id']==exact)]
    pages=max(1,(len(rows)+limit-1)//limit)
    page_number=min(page_number,pages-1)
    selected=rows[page_number*limit:(page_number+1)*limit]
    return {'rows':[public_row(r,predictions) for r in selected], 'page':page_number,'pages':pages,'limit':limit,
        'filtered_count':len(rows),'total_count':len(index['records']),
        'available_family_count':len(index['families']), 'families':index['families'] if first('catalog')=='1' else None,
        'splits':sorted({r['split'] for r in index['records']}),
        'unique_query_images':len({r['query_url'] for r in index['records']}),
        'unique_support_images':len({r['support_url'] for r in index['records']}),
        'preview':index['preview'],'completed':index['completed'],'source_manifest':index['source'],
        'manifest_sha256':index['manifest_sha256'],'prediction_methods':list(predictions)}


def mask_png(identifier, channel):
    if channel not in ('union_mask','target_mask','candidate_mask'):
        raise KeyError('Unknown mask')
    row=load_index()['by_id'][identifier]
    if not row['masks_available']:
        raise KeyError('Mask not yet generated')
    with np.load(row['target_path'],allow_pickle=False) as archive:
        mask=archive[channel]
        if mask.ndim!=2 or mask.dtype!=np.bool_:
            raise ValueError('Unexpected mask schema')
        pixels=mask.astype(np.uint8)*255
    stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='PNG')
    return stream.getvalue()


class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, format, *args):
        if len(args)>1 and str(args[1])=='200':
            return
        return super().log_message(format,*args)

    def do_GET(self):
        parsed=urlsplit(self.path)
        if parsed.path=='/api/scenes':
            try:
                body=json.dumps(page(parse_qs(parsed.query)),allow_nan=False).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
            except (ValueError,KeyError,OSError) as error:
                self.send_error(400,str(error))
            return
        if parsed.path.startswith('/api/mask/'):
            parts=parsed.path.split('/')
            try:
                if len(parts)!=5 or not parts[-1].endswith('.png'):
                    raise KeyError('Invalid mask route')
                body=mask_png(unquote(parts[3]),parts[4][:-4])
                self.send_response(200);self.send_header('Content-Type','image/png')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except (ValueError,KeyError,OSError) as error:
                self.send_error(404,str(error))
            return
        if parsed.path=='/':
            self.path='/two_dimensional_gallery.html'
        return super().do_GET()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8792)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(ROOT)))
    print(f'2D gallery: http://127.0.0.1:{args.port}/two_dimensional_gallery.html',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
