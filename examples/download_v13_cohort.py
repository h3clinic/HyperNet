"""Acquire original licensed COCO photos and this project's own CNN for replay.

No natural photos are served by this repository. Original attribution metadata
is retained beside locally downloaded files. URLs are the audited official S3
object paths with ordinary TLS verification, never an insecure fallback.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,io,json,re,shutil,urllib.request
from urllib.parse import urlsplit
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT_SHA='15c64b81747f3ebec224ef19621793c2d24ed9dd6f93de25c6ad5f7c5c04d49a'
CHECKPOINT_URL='https://github.com/h3clinic/HyperNet/releases/download/v0.12.0/detector_micro_aware.pt'
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def source_url(record):
    name=record['source']['file_name']
    if not re.fullmatch(r'\d{12}\.jpg',name) or name!=f"{int(record['image_id']):012d}.jpg":raise ValueError('Unexpected COCO object name')
    original=urlsplit(record['source']['coco_url'])
    if original.hostname!='images.cocodataset.org' or original.path!='/val2017/'+name:raise ValueError('Unexpected original source URL')
    return 'https://s3.amazonaws.com/images.cocodataset.org/val2017/'+name
def safe_target(data,name):
    path=(data/name).resolve()
    if not path.is_relative_to(data.resolve()):raise ValueError('Image path escapes output directory')
    return path
def validate_photo(raw,record):
    if sha_bytes(raw)!=record['sha256']:raise ValueError('Original photo file hash differs')
    with Image.open(io.BytesIO(raw)) as image:
        if image.getexif().get(274,1)!=1:raise ValueError('Photo orientation differs from frozen cohort')
        if image.size!=(record['width'],record['height']):raise ValueError('Original photo dimensions differ')
        rgb=np.array(ImageOps.exif_transpose(image).convert('RGB'),dtype=np.uint8)
    digest=sha_bytes(str(rgb.dtype).encode()+str(rgb.shape).encode()+rgb.tobytes())
    if digest!=record['pixel_sha256']:raise ValueError('Decoded image hash differs; check decoder/runtime rather than bypassing this guard')
def fetch_photo(record,data):
    path=safe_target(data,record['rgb_path'])
    if path.exists():validate_photo(path.read_bytes(),record);return 'verified_existing'
    with urllib.request.urlopen(source_url(record),timeout=45) as response:raw=response.read(16*1024*1024+1)
    if len(raw)>16*1024*1024:raise ValueError('Unexpected oversized photo response')
    validate_photo(raw,record);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as output:output.write(raw)
    return 'downloaded_original_source'
def checkpoint(path):
    if path.exists():
        if sha_bytes(path.read_bytes())!=CHECKPOINT_SHA:raise ValueError('Existing checkpoint is not the project-trained CNN')
        return
    with urllib.request.urlopen(CHECKPOINT_URL,timeout=60) as response:raw=response.read(4_000_001)
    if len(raw)!=3_598_069 or sha_bytes(raw)!=CHECKPOINT_SHA:raise ValueError('Checkpoint asset hash/size mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as output:output.write(raw)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=ROOT/'data/coco_real_v13')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/detector_micro_aware.pt')
    parser.add_argument('--checkpoint-only',action='store_true',help='Prepare only our own synthetic-trained CNN for manual image comparison')
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('Use 1–8 download workers')
    checkpoint(args.checkpoint)
    if args.checkpoint_only:print(json.dumps({'checkpoint_verified':True,'natural_photos_downloaded':0}));return
    source=ROOT/'benchmarks/v0.13/cohort_manifest.json';raw=source.read_bytes();manifest=json.loads(raw)
    provenance=json.loads((ROOT/'benchmarks/v0.13/public_export_provenance.json').read_text())
    if sha_bytes(raw)!=provenance['files']['cohort_manifest.json']['public_sha256']:raise ValueError('Published cohort manifest changed')
    records=manifest['references']+manifest['queries']
    if len(records)!=300 or len({r['image_id'] for r in records})!=300:raise ValueError('Expected exactly 300 distinct original photos')
    args.data.mkdir(parents=True,exist_ok=True)
    target=args.data/'manifest.json'
    if target.exists() and target.read_bytes()!=raw:raise FileExistsError('Different manifest already exists')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:outcomes=list(pool.map(lambda r:fetch_photo(r,args.data),records))
    if not target.exists():target.write_bytes(raw)
    attribution=args.data/'SOURCE_ATTRIBUTIONS.json'
    metadata={'dataset':'COCO2017 val2017','scope':'Original photos acquired from their official public source; individual licenses apply.',
        'images':[{'image_id':r['image_id'],'local_path':r['rgb_path'],'source':r['source'],'transport_url':source_url(r)} for r in records]}
    content=(json.dumps(metadata,indent=2)+'\n').encode()
    if attribution.exists() and attribution.read_bytes()!=content:raise FileExistsError('Different attribution record exists')
    if not attribution.exists():attribution.write_bytes(content)
    print(json.dumps({'verified_photos':300,'downloaded':outcomes.count('downloaded_original_source'),
        'verified_existing':outcomes.count('verified_existing'),'checkpoint_sha256':CHECKPOINT_SHA,'photo_attributions':str(attribution)},indent=2))
if __name__=='__main__':main()
