"""Acquire and freeze the official MVTec AD Grid pilot without model scoring."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path,PurePosixPath
import argparse,json,tarfile,urllib.request
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/mvtec_grid_v15'
URL='https://www.mydrive.ch/shares/163644/8c7fc7cba43ef5275c87ca750a6d0004/download/420937487-1629959044/grid.tar.xz'
ARCHIVE_BYTES=160763852
SALT='HyperNet-v15-MVTec-Grid-one-reference-normal-calibration-v1:'
CALIBRATION_IMAGES=64

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    value=sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):value.update(chunk)
    return value.hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def require(condition,message):
    if not condition:raise ValueError(message)
def pixel_hash(array):return sha256(str(array.dtype).encode()+str(array.shape).encode()+np.ascontiguousarray(array).tobytes()).hexdigest()

def safe_member(name):
    require('\\' not in name and ':' not in name,'Nonportable archive path')
    path=PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and path.parts,'Unsafe archive path')
    return path.as_posix()

def acquire():
    plan_path=DATA/'acquisition_plan.json';archive=DATA/'archive/grid.tar.xz'
    require(not plan_path.exists() and not archive.exists(),'Preserve existing acquisition evidence')
    write_new(plan_path,{'format':'hypernet.mvtec_grid.acquisition.v15','created_utc':now(),
        'dataset':'MVTec AD','category':'grid','official_download_page':'https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads',
        'official_dataset_page':'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
        'url':URL,'expected_bytes':ARCHIVE_BYTES,'official_checksum_available':False,
        'license':{'name':'CC BY-NC-SA 4.0','url':'https://creativecommons.org/licenses/by-nc-sa/4.0/'},
        'selection':{'normal_filename_sha256_salt':SALT,'reference_count':1,'normal_calibration_count':CALIBRATION_IMAGES,
            'remaining_train_normals':'unused','test':'every official test image, unchanged labels and split'},
        'no_images_previously_downloaded_for_this_pilot':True,'no_model_scores_consulted':True,
        'test_policy':'No model/UI inspection before frozen method and checkpoint admission. Mechanical pixel/hash audit only.',
        'real_weight_training_images':0,'threshold_calibration_images_are_extra_data':True})
    archive.parent.mkdir(parents=True,exist_ok=True);partial=archive.with_suffix(archive.suffix+'.partial')
    with urllib.request.urlopen(URL,timeout=90) as response:
        length=response.headers.get('Content-Length')
        require(length is None or int(length)==ARCHIVE_BYTES,'Unexpected official archive length')
        headers=dict(response.headers.items());final_url=response.geturl();count=0
        with partial.open('xb') as stream:
            while chunk:=response.read(4*1024*1024):
                count+=len(chunk);require(count<=ARCHIVE_BYTES,'Archive exceeds declared length');stream.write(chunk)
    require(count==ARCHIVE_BYTES,'Incomplete archive');partial.rename(archive)
    write_new(DATA/'download_provenance.json',{'created_utc':now(),'acquisition_plan_sha256':digest(plan_path),
        'request_url':URL,'final_url':final_url,'response_headers':headers,'archive_bytes':count,'archive_sha256':digest(archive),
        'transport':'HTTPS with default certificate validation','digest_scope':'Locally measured immutable identity; no publisher-supplied checksum was available'})
    print(json.dumps({'downloaded_bytes':count,'archive_sha256':digest(archive)}),flush=True)

def inventory():
    archive=DATA/'archive/grid.tar.xz';download=read(DATA/'download_provenance.json')
    require(digest(archive)==download['archive_sha256'],'Archive changed')
    members=[];seen=set()
    with tarfile.open(archive,'r|xz') as tar:
        for member in tar:
            name=safe_member(member.name)
            require(name not in seen,'Duplicate archive member');seen.add(name)
            require(member.isdir() or member.isfile(),'Links/devices are forbidden')
            require(member.size<=16*1024*1024,'Oversized member')
            members.append({'name':name,'type':'directory' if member.isdir() else 'file','bytes':member.size})
    require(sum(m['bytes'] for m in members)<=1024**3,'Archive extraction exceeds one GiB')
    images=[m['name'] for m in members if m['type']=='file' and m['name'].endswith('.png') and '/ground_truth/' not in m['name']]
    train=sorted(p for p in images if p.startswith('grid/train/good/'))
    test=sorted(p for p in images if p.startswith('grid/test/'))
    require(len(images)==len(train)+len(test) and len(train)>CALIBRATION_IMAGES and test,'Unexpected official image layout')
    ordered=sorted(train,key=lambda p:(sha256((SALT+p).encode()).hexdigest(),p))
    roles={p:('reference' if i==0 else 'dev_normal' if i<=CALIBRATION_IMAGES else 'unused_normal') for i,p in enumerate(ordered)}
    roles.update({p:'test' for p in test})
    record={'created_utc':now(),'archive_sha256':digest(archive),'image_pixels_decoded':False,'members':members,
        'image_counts':{'train_good':len(train),'test':len(test)},
        'published_test_types':dict(Counter(PurePosixPath(p).parts[2] for p in test))}
    write_new(DATA/'archive_inventory.json',record)
    write_new(DATA/'roster_freeze.json',{'format':'hypernet.mvtec_grid.roster.v15','created_utc':now(),
        'archive_sha256':digest(archive),'inventory_sha256':digest(DATA/'archive_inventory.json'),
        'acquisition_plan_sha256':digest(DATA/'acquisition_plan.json'),'builder_sha256':digest(__file__),
        'loader_sha256':digest(ROOT/'hypernet/mvtec_grid.py'),'selection_salt':SALT,
        'reference_path':ordered[0],'normal_calibration_paths':ordered[1:1+CALIBRATION_IMAGES],
        'roles':roles,'counts':dict(Counter(roles.values())),
        'selection_inputs':'Archive member paths and official split labels only; no decoded pixels, masks, model scores or visual curation',
        'threshold_rule':'Must be frozen by the evaluation protocol and fitted only on separate normal DEV scores; never on TEST labels',
        'data_budget':{'normal_reference_images':1,'extra_normal_calibration_images':CALIBRATION_IMAGES,
            'natural_weight_training_images':0,'additional_normals_for_reference_subspace':0},
        'metrics_required':['image_AUROC','image_average_precision','pixel_AUROC','pixel_average_precision',
            'false_positives_per_10000_normal_images','binomial_confidence_limits_from_actual_normal_TEST_N'],
        'interpretation':'Grid-only pilot, not the full 15-category MVTec AD benchmark. One-normal enrollment plus separate calibration; not one image total.'})
    print(json.dumps({'metadata_inventory':record['image_counts'],'test_types':record['published_test_types'],
        'roster_sha256':digest(DATA/'roster_freeze.json'),'roles':dict(Counter(roles.values()))}),flush=True)

def verify_roster():
    roster=read(DATA/'roster_freeze.json')
    for path,key in [(DATA/'archive/grid.tar.xz','archive_sha256'),(DATA/'archive_inventory.json','inventory_sha256'),
        (DATA/'acquisition_plan.json','acquisition_plan_sha256'),(Path(__file__),'builder_sha256'),(ROOT/'hypernet/mvtec_grid.py','loader_sha256')]:
        require(digest(path)==roster[key],'Roster-bound source changed: '+str(path))
    return roster

def prepare():
    roster=verify_roster();folder=DATA/'files';require(not folder.exists(),'Preserve original extracted files')
    folder.mkdir(parents=True)
    with tarfile.open(DATA/'archive/grid.tar.xz','r|xz') as tar:
        for member in tar:
            name=safe_member(member.name);path=(folder/name).resolve()
            require(path.is_relative_to(folder.resolve()) and (member.isfile() or member.isdir()),'Unsafe extraction')
            if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,path.open('xb') as dest:
                while chunk:=source.read(1024*1024):dest.write(chunk)
            require(path.stat().st_size==member.size,'Incomplete extracted file')
    records=[];raw_seen={};pixel_seen={};duplicates=[]
    for name,role in sorted(roster['roles'].items()):
        path=folder/name;parts=PurePosixPath(name).parts;anomaly=parts[1]=='test' and parts[2]!='good'
        with Image.open(path) as image:
            require(image.format=='PNG' and image.getexif().get(274,1)==1,'Unexpected image encoding/orientation')
            native=np.array(image);rgb=np.array(image.convert('RGB'),dtype=np.uint8);mode=image.mode
        require(native.dtype==np.uint8 and mode in ('L','RGB'),'Only lossless uint8 native pixels supported')
        height,width=rgb.shape[:2];raw=digest(path);pixels=pixel_hash(rgb)
        for label,value,seen in [('raw',raw,raw_seen),('decoded_rgb',pixels,pixel_seen)]:
            if value in seen:duplicates.append({'kind':label,'first':seen[value],'second':name})
            else:seen[value]=name
        mask=None
        if anomaly:
            mask_name='grid/ground_truth/'+parts[2]+'/'+PurePosixPath(name).stem+'_mask.png';mask_path=folder/mask_name
            require(mask_path.exists(),'Missing official anomaly mask')
            with Image.open(mask_path) as image:array=np.array(image)
            require(array.shape==(height,width) and set(np.unique(array)).issubset({0,255}) and np.any(array),'Invalid anomaly mask')
            mask={'path':'files/'+mask_name,'sha256':digest(mask_path),'pixel_sha256':pixel_hash(array),
                  'dtype':str(array.dtype),'positive_pixels':int(np.count_nonzero(array))}
        records.append({'id':'mvtec_grid/'+str(PurePosixPath(name).relative_to('grid')).removesuffix('.png'),
            'category':'grid','official_split':parts[1],'defect_type':parts[2],'role':role,'is_anomaly':anomaly,
            'rgb_path':'files/'+name,'sha256':raw,'pixel_sha256':pixels,'native_pixel_sha256':pixel_hash(native),
            'native_mode':mode,'native_shape':list(native.shape),'width':width,'height':height,'mask':mask})
    require(not duplicates,'Exact duplicate images in archive; preserve evidence and investigate before evaluation')
    manifest={'format':'hypernet.mvtec_grid.manifest.v15','completed':True,'created_utc':now(),
        'dataset':'MVTec AD','category':'grid','license':read(DATA/'acquisition_plan.json')['license'],
        'archive_sha256':roster['archive_sha256'],'roster_freeze_sha256':digest(DATA/'roster_freeze.json'),
        'counts':dict(Counter(r['role'] for r in records)),'records':records,'data_budget':roster['data_budget'],
        'pixel_hash_convention':'SHA256(str(dtype)+str(shape)+contiguous decoded bytes); RGB converts grayscale only by channel repetition',
        'pixels_modified':False,'test_pixel_access':'Mechanical integrity/mask audit only until protocol admission; no model/UI inspection',
        'normal_mask_convention':'All-zero original-size boolean mask synthesized only by evaluator; no normal mask file expected'}
    write_new(DATA/'manifest.json',manifest)
    dev={**manifest,'completed_partition':'reference_and_normal_calibration_only',
         'records':[r for r in records if r['role'] in ('reference','dev_normal')]}
    dev['counts']=dict(Counter(r['role'] for r in dev['records']))
    write_new(DATA/'development_manifest.json',dev)
    print(json.dumps({'manifest_sha256':digest(DATA/'manifest.json'),'counts':manifest['counts'],
        'test_normal':sum(r['role']=='test' and not r['is_anomaly'] for r in records),
        'test_anomaly':sum(r['role']=='test' and r['is_anomaly'] for r in records)}),flush=True)

def audit():
    roster=verify_roster();manifest=read(DATA/'manifest.json');require(manifest['completed'],'Incomplete manifest')
    require(manifest['roster_freeze_sha256']==digest(DATA/'roster_freeze.json'),'Manifest/roster binding changed')
    records=manifest['records'];require(len(records)==len(roster['roles'])==len({r['id'] for r in records}),'Roster mismatch')
    raw=set();pixels=set();masks=0;test_normal=0;test_anomaly=0
    for record in records:
        path=DATA/record['rgb_path'];name=record['rgb_path'].removeprefix('files/')
        require(roster['roles'][name]==record['role'] and digest(path)==record['sha256'],'Raw source or role mismatch')
        with Image.open(path) as image:
            native=np.array(image);rgb=np.array(image.convert('RGB'),dtype=np.uint8)
        require(pixel_hash(native)==record['native_pixel_sha256'] and pixel_hash(rgb)==record['pixel_sha256'],'Decoded pixels differ')
        require(rgb.shape==(record['height'],record['width'],3),'Dimensions differ')
        require(record['sha256'] not in raw and record['pixel_sha256'] not in pixels,'Image duplication/leakage')
        raw.add(record['sha256']);pixels.add(record['pixel_sha256'])
        if record['mask']:
            m=record['mask'];require(digest(DATA/m['path'])==m['sha256'],'Mask raw hash changed')
            with Image.open(DATA/m['path']) as image:array=np.array(image)
            require(pixel_hash(array)==m['pixel_sha256'] and array.shape==rgb.shape[:2],'Mask pixels/geometry changed');masks+=1
        if record['role']=='test':
            test_anomaly+=int(record['is_anomaly']);test_normal+=int(not record['is_anomaly'])
        else:require(record['official_split']=='train' and not record['is_anomaly'],'Non-normal development input')
    dev=read(DATA/'development_manifest.json')
    require(dev['records']==[r for r in records if r['role'] in ('reference','dev_normal')],'DEV projection leaks or differs')
    require(masks==test_anomaly,'Anomaly/mask mismatch')
    result={'format':'hypernet.mvtec_grid.data_audit.v15','valid':True,'created_utc':now(),
        'archive_sha256':roster['archive_sha256'],'roster_freeze_sha256':digest(DATA/'roster_freeze.json'),
        'manifest_sha256':digest(DATA/'manifest.json'),'development_manifest_sha256':digest(DATA/'development_manifest.json'),
        'all_raw_and_native_rgb_pixel_hashes_verified':True,'no_exact_image_duplicates':True,'official_labels_and_splits_preserved':True,
        'images':len(records),'masks':masks,'counts':manifest['counts'],'test_normal_images':test_normal,'test_anomaly_images':test_anomaly,
        'model_forward_calls':0,'test_ui_displayed':False,'natural_weight_training_images':0,
        'normal_false_positive_resolution_per_10000':10000/test_normal,
        'zero_false_positives_one_sided95_upper_per_10000':10000*(1-.05**(1/test_normal)),
        'small_N_warning':'FP/10,000 is a normalized estimate from the actual normal TEST count, not a 10,000-image deployment test.',
        'data_budget':manifest['data_budget'],'no_publisher_checksum':'Local digest records identity; HTTPS official link establishes source, not a publisher signed hash.'}
    write_new(ROOT/'experiments/mvtec_grid_data_audit.json',result)
    print(json.dumps(result),flush=True)

def admit(protocol_path,expected_sha):
    protocol_path=Path(protocol_path).resolve();require(digest(protocol_path)==expected_sha,'Protocol digest differs')
    protocol=read(protocol_path);audit=read(ROOT/'experiments/mvtec_grid_data_audit.json')
    require(audit['valid'] and audit['manifest_sha256']==digest(DATA/'manifest.json'),'Dataset audit differs')
    require(protocol.get('model_selection_complete') is True and protocol.get('test_scores_computed_before_freeze') is False,'Method not frozen before TEST')
    require(protocol.get('grid_manifest_sha256')==digest(DATA/'manifest.json'),'Model/data manifest binding differs')
    require(protocol.get('grid_roster_freeze_sha256')==digest(DATA/'roster_freeze.json'),'Model/roster binding differs')
    require(protocol.get('behavior_files') and protocol.get('checkpoint_files'),'Require behavior/checkpoint bindings')
    for group in ('behavior_files','checkpoint_files'):
        for name,value in protocol[group].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            require(digest(path)==value,'Frozen artifact changed: '+name)
    write_new(DATA/'test_admission.json',{'created_utc':now(),'protocol_path':str(protocol_path),'protocol_sha256':expected_sha,
        'manifest_sha256':digest(DATA/'manifest.json'),'data_audit_sha256':digest(ROOT/'experiments/mvtec_grid_data_audit.json'),
        'test_pixels_authorized_for_frozen_inference':True,'test_mask_access_evaluator_only':True})
    print(json.dumps({'test_admitted':True,'admission_sha256':digest(DATA/'test_admission.json')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['acquire','inventory','prepare','audit','admit'],required=True)
    parser.add_argument('--protocol',type=Path);parser.add_argument('--expected-protocol-sha');args=parser.parse_args()
    if args.stage=='admit':admit(args.protocol,args.expected_protocol_sha)
    else:globals()[args.stage]()
