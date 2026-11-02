"""Transport-only recovery after official virtual-host TLS certificate failure.

Preserves original immutable plan, builder, freeze and first fetch admission.
Uses the same official public S3 bucket through its certificate-valid path URL.
No image selection or model execution is performed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import shutil
import urllib.request
import time
from collections import Counter
from PIL import Image
import prepare_real_photos as d


def main():
    d.verify_freeze()
    d.require(not (d.DATA/'manifest.json').exists(), 'Completed manifest already exists')
    existing=list((d.DATA/'images').glob('*.jpg'))
    d.require(not existing,'Recovery expects zero partial image writes; inspect manually if this changes')
    admission={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'data_freeze_sha256':d.digest(d.EXPERIMENTS/'real_photo_data_freeze.json'),
        'original_fetch_admission_sha256':d.digest(d.EXPERIMENTS/'real_photo_fetch_admission.json'),
        'plan_sha256':d.digest(d.DATA/'plan.json'),
        'recovery_script_sha256':d.digest(__file__),
        'reason':'Original virtual-host HTTPS endpoint fails certificate hostname verification; official S3 bucket path endpoint has valid TLS.',
        'original_dataset_images_written':len(existing),
        'transport_url_template':'https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}',
        'selection_annotations_and_criteria_unchanged':True,'model_runs':0}
    d.write_new(d.EXPERIMENTS/'real_photo_transport_recovery_admission.json',admission)
    plan=d.read(d.DATA/'plan.json')

    def fetch(row):
        dest=d.DATA/row['rgb_path']
        cached=d.CACHE/row['source']['file_name']
        transport='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+row['source']['file_name']
        if cached.is_file():
            shutil.copyfile(cached,dest);method='existing_unselected_coco_cache'
        else:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(transport,timeout=45) as response:
                        payload=response.read()
                    with dest.open('xb') as stream:stream.write(payload)
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(attempt+1)
            method='download_official_coco_s3_path_https'
        with Image.open(dest) as image:orientation=image.getexif().get(274,1)
        d.require(orientation==1,'Nontrivial EXIF orientation invalidates frozen original-coordinate protocol')
        array=d.rgb(dest)
        d.require(array.shape==(row['height'],row['width'],3),'Original dimensions changed')
        result={**row,'sha256':d.digest(dest),'pixel_sha256':d.pixels(array),'exif_orientation':orientation}
        return result,{'image_id':row['image_id'],'method':method,'url':row['source']['coco_url'],
                       'transport_url':transport,'sha256':result['sha256']}

    with ThreadPoolExecutor(max_workers=6) as pool:
        loaded=list(pool.map(fetch,plan['references']+plan['queries']))
    rows=[r for r,_ in loaded];provenance=[p for _,p in loaded]
    manifest={**plan,'schema_version':'hypernet-real-photo-v13-v1',
        'references':rows[:12],'queries':rows[12:],
        'pixel_hash_convention':'SHA256(str(dtype)+str(shape)+contiguous RGB bytes); dtype uint8; EXIF transposed',
        'plan_sha256':d.digest(d.DATA/'plan.json'),
        'data_freeze_sha256':d.digest(d.EXPERIMENTS/'real_photo_data_freeze.json'),
        'annotation_sha256':d.digest(d.ANNOTATIONS),
        'transport_recovery_admission_sha256':d.digest(d.EXPERIMENTS/'real_photo_transport_recovery_admission.json')}
    d.write_new(d.DATA/'manifest.json',manifest)
    d.write_new(d.DATA/'download_provenance.json',{'manifest_sha256':d.digest(d.DATA/'manifest.json'),'images':provenance,
        'transport_recovery_admission_sha256':manifest['transport_recovery_admission_sha256']})
    d.write_new(d.DATA/'source_licenses.json',{'dataset':'COCO2017 val2017',
        'annotation_license':'CC BY4.0; this does not supersede individual photograph licenses',
        'annotation_terms':'https://cocodataset.org/#termsofuse','photo_licenses':d.read(d.ANNOTATIONS)['licenses'],
        'images':[{'image_id':r['image_id'],**r['source'],'sha256':r['sha256'],'pixel_sha256':r['pixel_sha256']} for r in rows]})
    d.verify_freeze()
    print(json.dumps({'status':'fetched_recovered_transport','methods':dict(Counter(p['method'] for p in provenance)),
                      'manifest_sha256':d.digest(d.DATA/'manifest.json')}),flush=True)


if __name__=='__main__':main()
