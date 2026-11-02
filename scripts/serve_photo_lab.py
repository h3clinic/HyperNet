"""Local photo enrollment and read-only real-photo evaluation gallery.

Uploads are decoded in memory and never written to disk. Only explicitly
selected reference/query RGB and the reference box enter the inference API.
"""
from __future__ import annotations
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,unquote,urlsplit
import argparse,base64,binascii,hashlib,io,json,math,secrets,sys,threading,time
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/coco_real_v13'
RESULTS=ROOT/'results/real_eval'
MAX_IMAGE_BYTES=8*1024*1024
MAX_IMAGE_PIXELS=16_000_000
MAX_REQUEST_BYTES=24*1024*1024
TOKEN=secrets.token_urlsafe(32)
METHOD_LABELS={'old_cnn':'Previous CNN · learned proposals','native_patch':'Native patch hybrid · learned proposals',
 'old_cnn_grid':'CNN · fixed-grid diagnostic','native_patch_grid':'Native patches · fixed-grid diagnostic'}
INFERENCE=None
INFERENCE_PROVENANCE={}
TRANSFERRED_CUTOFFS={}
_INFERENCE_LOCK=threading.Lock()
_CACHE_LOCK=threading.Lock()
_CACHE={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def finite_json(raw):
    def invalid(value):raise ValueError('Nonfinite JSON values are not allowed')
    return json.loads(raw,parse_constant=invalid)
def safe_path(base,value):
    path=(base/value).resolve()
    if not path.is_relative_to(base.resolve()):raise ValueError('Path must stay inside the cohort')
    return path

def decode_upload(value):
    if not isinstance(value,str) or not value.startswith('data:image/') or ';base64,' not in value:
        raise ValueError('Expected an image file encoded by the local page')
    header,encoded=value.split(',',1)
    if header not in ('data:image/png;base64','data:image/jpeg;base64','data:image/webp;base64'):
        raise ValueError('Use a PNG, JPEG or WebP image')
    if len(encoded)>4*((MAX_IMAGE_BYTES+2)//3):raise ValueError('Each image must be at most 8 MB')
    try:raw=base64.b64decode(encoded,validate=True)
    except (ValueError,binascii.Error) as error:raise ValueError('Invalid image encoding') from error
    if not raw or len(raw)>MAX_IMAGE_BYTES:raise ValueError('Each image must be at most 8 MB')
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ('PNG','JPEG','WEBP'):raise ValueError('Unsupported image format')
            if getattr(source,'n_frames',1)!=1:raise ValueError('Choose a still image')
            if source.width*source.height>MAX_IMAGE_PIXELS or min(source.size)<8:
                raise ValueError('Use an image with 8 or more pixels per side and at most 16 million pixels')
            return np.array(ImageOps.exif_transpose(source).convert('RGB'),dtype=np.uint8)
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise ValueError('Could not decode this image') from error

def reference_box(value,rgb):
    h,w=rgb.shape[:2]
    if value is None:return [0,0,w,h]
    if not isinstance(value,list) or len(value)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in value):
        raise ValueError('Reference box must contain four finite pixel coordinates')
    x1,y1,x2,y2=map(float,value)
    if not(0<=x1<x2<=w and 0<=y1<y2<=h) or x2-x1<2 or y2-y1<2:
        raise ValueError('Reference box must cover at least 2 by 2 pixels inside the reference')
    return [x1,y1,x2,y2]

def result_source():
    return next((p for p in (RESULTS/'real_results.json',ROOT/'benchmarks/v0.13/aggregate_metrics.json') if p.is_file()),None)

def completed_results():
    """Prefer a local run; otherwise read the published, hash-bound benchmark."""
    path=result_source()
    if path is None:return None
    if path.parent==ROOT/'benchmarks/v0.13':
        public=read(path.parent/'public_export_provenance.json')
        if sha(path)!=public['files']['aggregate_metrics.json']['public_sha256']:
            raise ValueError('Published aggregate hash changed')
    data=read(path)
    return data if data.get('completed') is True else None

def prediction_index(mode='transferred'):
    results=completed_results()
    if not results:return {}
    if mode not in results['modes']:raise ValueError('Unknown operating mode')
    methods={}
    for alias,info in results.get('score_files',{}).items():
        path=safe_path(result_source().parent,info['path'])
        if sha(path)!=info['sha256']:raise ValueError('Saved score log hash changed')
        rows={}
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line:continue
            row=json.loads(line);threshold=results['modes'][mode]['methods'][alias][row['split']]['threshold']
            rows[row['id']]={'score':row['score'],'threshold':threshold,'accepted':row['score']>=threshold,
                'bbox_xyxy':row['bbox_xyxy'],'iou':row['iou'],'evidence':row.get('evidence',{})}
        methods[alias]=rows
    return methods

def load_cohort():
    manifest_path=DATA/'manifest.json'
    if not manifest_path.is_file():return {'records':[],'by_id':{},'manifest_sha256':None}
    stamp=(manifest_path.stat().st_mtime_ns,manifest_path.stat().st_size)
    with _CACHE_LOCK:
        if _CACHE.get('stamp')==stamp:return _CACHE['index']
        data=read(manifest_path);references={r['category_id']:r for r in data['references']};rows=[]
        if len(references)!=len(data['references']):raise ValueError('Expected one reference per category')
        for q in data['queries']:
            ref=references[q['category_id']];support_path=safe_path(DATA,ref['rgb_path']);query_path=safe_path(DATA,q['rgb_path'])
            if not support_path.is_file() or not query_path.is_file():continue
            row={'id':q['id'],'split':q['split'],'category':q['name'],'category_id':q['category_id'],
                'present':bool(q['present']),'support_path':support_path,'query_path':query_path,
                'support_sha256':ref['sha256'],'query_sha256':q['sha256'],'support_box':ref['bbox_xyxy'],
                'query_size':[q['width'],q['height']],'target_boxes':q['gt_boxes_xyxy'],
                'reference_source':ref.get('source',{}),'query_source':q.get('source',{}),
                'reference_image_id':ref['image_id'],'query_image_id':q['image_id']}
            rows.append(row)
        rows.sort(key=lambda r:(r['split'],r['category'],not r['present'],r['id']))
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate query IDs')
        index={'records':rows,'by_id':{r['id']:r for r in rows},'manifest_sha256':sha(manifest_path)}
        _CACHE.update(stamp=stamp,index=index)
        return index

def scene_page(params):
    first=lambda key,default='':params.get(key,[default])[0]
    index=load_cohort();predictions=prediction_index(first('mode','transferred'));limit=min(36,max(1,int(first('limit','18'))))
    selected=[r for r in index['records'] if (not first('split') or r['split']==first('split'))
        and (not first('category') or r['category']==first('category'))
        and (not first('present') or r['present']==(first('present')=='yes'))]
    pages=max(1,(len(selected)+limit-1)//limit);page=min(pages-1,max(0,int(first('page','0'))));public=[]
    for row in selected[page*limit:(page+1)*limit]:
        out={k:v for k,v in row.items() if k not in ('support_path','query_path')}
        out['support_url']=f'/api/image/{row["id"]}/reference.png';out['query_url']=f'/api/image/{row["id"]}/query.png'
        out['predictions']={alias:records[row['id']] for alias,records in predictions.items() if row['id'] in records};public.append(out)
    return {'rows':public,'page':page,'pages':pages,'total_count':len(index['records']),'filtered_count':len(selected),
        'categories':sorted({r['category'] for r in index['records']}),'prediction_methods':list(predictions),
        'unique_query_images':len({r['query_image_id'] for r in index['records']}),
        'unique_support_images':len({r['reference_image_id'] for r in index['records']}),
        'manifest_sha256':index['manifest_sha256'],
        'message':'Completed saved evaluation.' if predictions else ('Cohort available; frozen evaluation results are pending.' if selected else 'Fresh photo acquisition is pending. No results are being inferred here.')}

def cohort_image(identifier,kind):
    row=load_cohort()['by_id'][identifier]
    if kind not in ('reference.png','query.png'):raise ValueError('Unknown image kind')
    prefix='support' if kind=='reference.png' else 'query';path=row[prefix+'_path']
    if sha(path)!=row[prefix+'_sha256']:raise ValueError('Cohort image hash changed')
    with Image.open(path) as source:
        image=source.convert('RGB')
        if prefix=='support':
            from hypernet.real_photo import reference_image
            normalized,_,_=reference_image(np.array(image),row['support_box']);image=Image.fromarray(normalized)
        stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()

def compare_request(payload):
    if INFERENCE is None:raise RuntimeError('The frozen real-photo adapter is not available yet')
    if not isinstance(payload,dict) or set(payload)-{'reference','query','reference_box','operating_mode'}:
        raise ValueError('Only reference image, reference box and query image are accepted')
    if not _INFERENCE_LOCK.acquire(blocking=False):raise RuntimeError('Another local comparison is running; try again shortly')
    try:
        mode=payload.get('operating_mode','transferred')
        if mode not in ('transferred','real_dev_calibrated'):raise ValueError('Unknown operating mode')
        result=completed_results()
        if mode=='real_dev_calibrated' and (result is None or mode not in result['modes']):
            raise ValueError('Real-photo DEV calibration is not available yet')
        reference=decode_upload(payload.get('reference'));query=decode_upload(payload.get('query'))
        box=reference_box(payload.get('reference_box'),reference);start=time.perf_counter()
        predictions=INFERENCE(reference,box,query)
        for alias,prediction in predictions.items():
            threshold=result['modes'][mode]['methods'][alias]['dev']['threshold'] if result and alias in result['modes'][mode]['methods'] else TRANSFERRED_CUTOFFS.get(alias)
            prediction['threshold']=threshold;prediction['accepted']=None if threshold is None else prediction['score']>=threshold
        # Return scores and geometry, never uploaded pixels or original names.
        return {'predictions':predictions,'reference_size':[reference.shape[1],reference.shape[0]],
            'query_size':[query.shape[1],query.shape[0]],'reference_box':box,'seconds':time.perf_counter()-start,
            'enrollment_images':1,'weight_updates':0,'saved_uploads':False,'provenance':INFERENCE_PROVENANCE,
            'operating_mode':mode,'additional_real_calibration_photos':96 if mode=='real_dev_calibrated' else 0,
            'message':'Comparison complete. '+('Decisions reuse the frozen synthetic cutoff; no real-photo threshold fitting.' if mode=='transferred' else 'Secondary mode uses 96 additional real DEV photographs for cutoff calibration.')+' Scores are not probabilities; accuracy on your photos is not established.'}
    finally:_INFERENCE_LOCK.release()

class Handler(SimpleHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,format,*args):
        if len(args)>1 and str(args[1])=='200':return
        return super().log_message(format,*args)
    def local_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
    def body(self,status,body,kind='application/json'):
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(body)
    def json(self,status,value):self.body(status,json.dumps(value,allow_nan=False).encode())
    def do_GET(self):
        if not self.local_host():self.json(403,{'error':'Localhost access only'});return
        parsed=urlsplit(self.path)
        try:
            if parsed.path=='/api/config':
                self.json(200,{'token':TOKEN,'inference_available':INFERENCE is not None,'method_labels':METHOD_LABELS,
                    'max_image_bytes':MAX_IMAGE_BYTES,'max_image_pixels':MAX_IMAGE_PIXELS});return
            if parsed.path=='/api/scenes':self.json(200,scene_page(parse_qs(parsed.query)));return
            if parsed.path.startswith('/api/image/'):
                parts=parsed.path.split('/')
                if len(parts)!=5:raise ValueError('Invalid image route')
                self.body(200,cohort_image(unquote(parts[3]),parts[4]),'image/png');return
            if parsed.path=='/index.html' and not (ROOT/'index.html').is_file() and (ROOT/'benchmarks/v0.13/report.html').is_file():
                self.send_response(302);self.send_header('Location','/benchmarks/v0.13/report.html');self.send_header('Content-Length','0');self.end_headers();return
            relative='photo_lab.html' if parsed.path=='/' else unquote(parsed.path).lstrip('/')
            path=safe_path(ROOT,relative)
            allowed=relative in ('photo_lab.html','index.html','data/coco_real_v13/manifest.json','data/coco_real_v13/source_licenses.json','results/real_eval/real_results.json') or (relative.split('/')[0] in ('assets','figures','research','experiments') and path.suffix in ('.html','.json','.png','.svg')) or (relative.startswith('benchmarks/v0.13/') and path.suffix in ('.html','.json','.png','.md'))
            if not allowed or not path.is_file():self.json(404,{'error':'Unknown local page'});return
            self.path='/'+relative;super().do_GET()
        except (ValueError,KeyError,OSError) as error:self.json(400,{'error':str(error)})
    def do_POST(self):
        origin=self.headers.get('Origin')
        if not self.local_host() or origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}') or not secrets.compare_digest(self.headers.get('X-HyperNet-Token',''),TOKEN):
            self.close_connection=True;self.json(403,{'error':'Use the local photo page to submit images'});return
        if urlsplit(self.path).path!='/api/compare':self.close_connection=True;self.json(404,{'error':'Unknown endpoint'});return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=MAX_REQUEST_BYTES:raise ValueError('Upload request is too large or empty')
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected JSON image upload')
            self.connection.settimeout(30)
            raw=self.rfile.read(length)
            if len(raw)!=length:raise ValueError('Incomplete upload')
            self.json(200,compare_request(finite_json(raw)))
        except (ValueError,KeyError,OSError) as error:self.close_connection=True;self.json(400,{'error':str(error)})
        except RuntimeError as error:self.json(503,{'error':str(error)})

def main():
    global INFERENCE,INFERENCE_PROVENANCE,TRANSFERRED_CUTOFFS
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8794)
    parser.add_argument('--enable-uploads',action='store_true',help='Enable manual photo inference after the benchmark protocol is frozen')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/detector_micro_aware.pt')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.enable_uploads:
        import torch
        from hypernet.real_photo import RealPhotoDetector,TRANSFERRED_CUTOFFS as CUTS
        from hypernet.detail_pipeline import state_hash
        if not 1<=args.threads<=8:raise ValueError('Use between 1 and 8 CPU threads')
        torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
        model=RealPhotoDetector(args.checkpoint);INFERENCE_PROVENANCE=model.provenance();TRANSFERRED_CUTOFFS=dict(CUTS)
        def infer(reference,box,query):
            before=state_hash(model.model);predictions=model.detect(reference,box,query)
            if state_hash(model.model)!=before or any(p.grad is not None for p in model.model.parameters()):
                raise RuntimeError('Read-only inference contract was violated')
            return predictions
        INFERENCE=infer
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(ROOT)))
    print(f'Local photo lab: http://127.0.0.1:{args.port}/photo_lab.html',flush=True);server.serve_forever()
if __name__=='__main__':main()
