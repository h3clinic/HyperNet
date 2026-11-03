"""Replay a completed public v19 benchmark in a new checkout/cache/output folder.

This helper never reads the original workstation's paths or private files.
Use --print-manifest-template for the publication contract and --self-test for
offline gate/hash/crop checks. See --help for a typical replay command.
"""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import argparse, hashlib, io, json, os, shutil, sys, tempfile, time, urllib.request
import numpy as np
from PIL import Image

FORMAT='hypernet.v19.published_benchmark.v1'
METHODS=('v19','v17','owlvit_base_patch32','raw_template')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path,data):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def pixel_sha(rgb):
    a=np.ascontiguousarray(rgb)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def relative_file(root,name):
    # Only repository-relative POSIX paths belong in the public manifest.
    if not isinstance(name,str) or not name or '\\' in name or ':' in name:
        raise ValueError('Expected a portable relative path: '+str(name))
    posix=PurePosixPath(name)
    if posix.is_absolute() or '..' in posix.parts:raise ValueError('Path leaves public bundle: '+name)
    path=(root/Path(*posix.parts)).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Symlink leaves public bundle: '+name)
    return path

def checked_descriptor(root,descriptor):
    path=relative_file(root,descriptor['path'])
    if sha(path)!=descriptor['sha256']:raise ValueError('Published file hash differs: '+descriptor['path'])
    return path

def rgb_from_bytes(payload):
    with Image.open(io.BytesIO(payload)) as image:
        if image.getexif().get(274,1)!=1:raise ValueError('Unexpected source EXIF orientation')
        return np.asarray(image.convert('RGB')).copy()

def validate_image(payload,row,record):
    if hashlib.sha256(payload).hexdigest()!=record['sha256']:
        raise ValueError('COCO source bytes differ: '+str(row['image_id']))
    rgb=rgb_from_bytes(payload)
    if rgb.shape!=(row['height'],row['width'],3) or pixel_sha(rgb)!=record['pixel_sha256']:
        raise ValueError('Decoded COCO pixels differ: '+str(row['image_id'])+'; check Pillow/JPEG decoder version')
    return rgb

def crop_reference(rgb,row):
    x1,y1,x2,y2=row['bbox_xyxy']
    left,top=max(0,int(np.floor(x1))),max(0,int(np.floor(y1)))
    right,bottom=min(rgb.shape[1],int(np.ceil(x2))),min(rgb.shape[0],int(np.ceil(y2)))
    if right<=left or bottom<=top:raise ValueError('Empty reference crop')
    return rgb[top:bottom,left:right].copy()

def official_url(row):
    source=row['source']; iid=int(row['image_id']); name=f'{iid:012d}.jpg'
    if iid<0 or source['file_name']!=name or source['official_source_split']!='train2017':
        raise ValueError('Unexpected pinned COCO source identity')
    parsed=urlparse(source['coco_url'])
    if parsed.hostname!='images.cocodataset.org' or parsed.path!='/train2017/'+name:
        raise ValueError('Source URL is not the published COCO train2017 object')
    # Same official COCO S3 object, reached over HTTPS; no Flickr substitutions.
    return 'https://s3.amazonaws.com/images.cocodataset.org/train2017/'+name

def open_bundle(manifest_path,method,split):
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent;manifest=read(manifest_path)
    if manifest.get('format')!=FORMAT or manifest.get('completed')is not True:
        raise ValueError('A completed published benchmark manifest is required; development plans cannot fetch TEST')
    entry=manifest['methods'][method]
    plan=read(checked_descriptor(root,manifest['data_plan']))
    target_ids=[int(t['category_id']) for t in plan['targets']]
    if len(target_ids)!=12 or len(set(target_ids))!=12:raise ValueError('Expected the fixed 12 target categories')
    refs=plan['references']
    if sorted(r['category_id']for r in refs)!=sorted(target_ids):raise ValueError('Exactly one reference per category required')
    if len({r['image_id']for r in refs})!=12:raise ValueError('Reference photos must be distinct')
    # Hash only portable published files. Original absolute input_bindings and
    # workspace_path fields, if retained in the metadata, are never accessed.
    common=['evaluation/vendor_v17/detection_metrics.py']
    if method=='owlvit_base_patch32':common+=['evaluation/vendor_v17/owl_baseline.py']
    elif method=='raw_template':common+=['evaluation/vendor_v17/raw_template.py']
    elif method=='v17':common+=['evaluation/vendor_v17/local_matcher.py','evaluation/vendor_v17/local_relation_model.py']
    else:common+=['hypernet_v19/model.py','hypernet_v19/matcher.py','hypernet_v19/proposals.py']
    for name in common:
        expected=manifest['code_files'].get(name)
        if expected is None or sha(relative_file(root,name))!=expected:raise ValueError('Published inference source changed: '+name)
    calibration=read(checked_descriptor(root,entry['calibration']))
    for key,budget in [('primary_5pct',.05),('secondary_1pct',.01)]:
        if calibration[key]['nominal_dev_negative_fpr']!=budget or not np.isfinite(calibration[key]['cutoff']):
            raise ValueError('Missing fixed finite DEV threshold')
    expected=entry['expected'][split]
    arrays_path=checked_descriptor(root,expected['arrays']);result_path=checked_descriptor(root,expected['result'])
    expected_result=read(result_path)
    if expected_result.get('method')!=method or expected_result.get('split')!=split:
        raise ValueError('Published result belongs to a different method or split')
    if sha(arrays_path)!=expected_result['score_file_sha256']:
        raise ValueError('Published score array is not bound by the completed result')
    records={str(r['image_id']):{'sha256':r['sha256'],'pixel_sha256':r['pixel_sha256']}for r in refs}
    if split=='test':
        acquisition=read(checked_descriptor(root,manifest['test_acquisition']))
        if acquisition.get('completed')is not True:raise ValueError('TEST acquisition has not completed')
        eligible=acquisition['eligible_image_ids']
        if not eligible or len(eligible)!=len(set(eligible)):raise ValueError('Invalid completed TEST roster')
        if not set(eligible)<=set(r['image_id']for r in plan['test']):raise ValueError('TEST roster is outside the published plan')
        rows=[r for r in plan['test']if r['image_id']in set(eligible)]
        for row in rows:
            record=acquisition['records'][str(row['image_id'])]
            if record['image_id']!=row['image_id']:raise ValueError('TEST source record identity differs')
            records[str(row['image_id'])]=record
    else:
        rows=plan['dev']
        for row in rows:records[str(row['image_id'])]={'sha256':row['sha256'],'pixel_sha256':row['pixel_sha256']}
    if set(r['image_id']for r in refs)&set(r['image_id']for r in rows):raise ValueError('Support/query source overlap')
    if len(set(r['image_id']for r in rows))!=len(rows):raise ValueError('Query photos are not unique')
    expected_arrays=dict(np.load(arrays_path,allow_pickle=False))
    if expected_arrays['image_ids'].tolist()!=[r['image_id']for r in rows] or expected_arrays['category_ids'].tolist()!=target_ids:
        raise ValueError('Completed result does not cover this exact photo/target roster')
    for row in refs+rows:
        official_url(row)
        record=records[str(row['image_id'])]
        if any(not isinstance(record.get(k),str)or len(record[k])!=64 for k in ('sha256','pixel_sha256')):
            raise ValueError('Missing measured raw/pixel hashes; unfinished TEST data cannot be fetched')
    return {'root':root,'manifest':manifest,'manifest_sha256':sha(manifest_path),'entry':entry,'plan':plan,
            'references':refs,'rows':rows,'records':records,'calibration':calibration,
            'expected_arrays':expected_arrays,'expected_result':expected_result}

def acquire(bundle,data_root,workers=4):
    data_root=Path(data_root).resolve();images=data_root/'images';images.mkdir(parents=True,exist_ok=True)
    rows=bundle['references']+bundle['rows']
    def one(row):
        record=bundle['records'][str(row['image_id'])];path=images/f"{row['image_id']:012d}.jpg"
        if path.exists():
            validate_image(path.read_bytes(),row,record);return 'verified_cache'
        url=official_url(row)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=45)as response:payload=response.read()
                break
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
        validate_image(payload,row,record)
        # Validate in memory before the exclusive write; never overwrite a cache.
        with path.open('xb')as f:f.write(payload)
        return 'downloaded'
    counts={'downloaded':0,'verified_cache':0};failures=[]
    with ThreadPoolExecutor(max_workers=workers)as pool:
        futures={pool.submit(one,row):row['image_id']for row in rows}
        for future in as_completed(futures):
            try:counts[future.result()]+=1
            except Exception as error:failures.append({'image_id':futures[future],'error':str(error)})
    if failures:raise RuntimeError('No substitutions or replacements allowed: '+json.dumps(failures))
    return {'photos':len(rows),**counts,'all_raw_and_decoded_pixel_hashes_verified':True}

def load_photo(bundle,data_root,row):
    path=Path(data_root)/'images'/f"{row['image_id']:012d}.jpg"
    return validate_image(path.read_bytes(),row,bundle['records'][str(row['image_id'])])

def stage_owl_checkpoint(bundle,checkpoint):
    """Own a temporary pinned checkpoint view without modifying supplied files.

    The unchanged baseline expects acquisition.json, which is project metadata
    rather than an official model file. Its metadata comes from the public bundle.
    Caller must keep the returned TemporaryDirectory alive while using the model.
    """
    entry=bundle['entry'];checkpoint=Path(checkpoint).resolve()
    metadata=read(checked_descriptor(bundle['root'],entry['owl_acquisition']))
    expected=entry['owl_checkpoint_files']
    if metadata.get('format')!='hypernet.published_owl_acquisition.v1' or metadata.get('repo_id')!=entry['owl_repo_id'] or metadata.get('revision')!=entry['owl_revision']:
        raise ValueError('Bundled OWL repository/revision metadata differs')
    if {name:record['sha256']for name,record in metadata['files'].items()}!=expected or 'acquisition.json'in expected:
        raise ValueError('Bundled OWL file manifest differs')
    sources={name:relative_file(checkpoint,name)for name in expected}
    for name,path in sources.items():
        if sha(path)!=expected[name]:raise ValueError('Google checkpoint file differs: '+name)
    temporary=tempfile.TemporaryDirectory(prefix='hypernet-owl-replay-')
    try:
        root=Path(temporary.name)
        for name,source in sources.items():
            destination=relative_file(root,name);destination.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(source,destination)
            except OSError:shutil.copyfile(source,destination)
            if sha(destination)!=expected[name]:raise ValueError('Staged Google checkpoint differs: '+name)
        write_new(root/'acquisition.json',metadata)
    except BaseException:
        temporary.cleanup();raise
    return temporary


def model_for(bundle,args):
    root=bundle['root'];entry=bundle['entry'];sys.path.insert(0,str(root));sys.path.insert(0,str(root/'evaluation'))
    if args.method=='owlvit_base_patch32':
        if args.owl_checkpoint is None:raise ValueError('--owl-checkpoint is required; Google weights are not redistributed')
        from vendor_v17.owl_baseline import OwlBaseline
        temporary=stage_owl_checkpoint(bundle,args.owl_checkpoint)
        try:model=OwlBaseline(temporary.name,args.device)
        except BaseException:
            temporary.cleanup();raise
        model._published_owl_checkpoint_stage=temporary
    elif args.method=='raw_template':
        from vendor_v17.raw_template import RawTemplate
        model=RawTemplate()
    else:
        if args.checkpoint is None:raise ValueError('--checkpoint is required for own HyperNet weights')
        if sha(args.checkpoint)!=entry['checkpoint_sha256']:raise ValueError('HyperNet checkpoint is not the published checkpoint')
        config=entry['inference']
        if args.method=='v17':
            from vendor_v17.local_matcher import FrozenLocalMatcher
            model=FrozenLocalMatcher(args.checkpoint,device=args.device,**config)
        else:
            from hypernet_v19.matcher import FrozenDenseMatcher
            if config.get('max_side') not in(384,640):raise ValueError('Unpublished v19 image scale')
            model=FrozenDenseMatcher(args.checkpoint,device=args.device,**config)
    module=sys.modules[model.__class__.__module__]
    if not Path(module.__file__).resolve().is_relative_to(root):raise ValueError('Imported a model outside the public bundle')
    return model

def verify_replay(scores,boxes,overlaps,expected,calibration):
    score_close=np.isclose(scores,expected['scores'],rtol=1e-4,atol=1e-5)
    box_close=np.isclose(boxes,expected['boxes'],rtol=1e-4,atol=.01)
    iou_close=np.isclose(overlaps,expected['overlaps'],rtol=1e-4,atol=1e-5)
    points={}
    for name,point in calibration.items():
        cutoff=point['cutoff'];actual=scores>=cutoff;published=expected['scores']>=cutoff
        points[name]={'presence_threshold_crossings':int((actual!=published).sum()),
                      'localization_decision_changes':int(((actual&(overlaps>=.5))!=(published&(expected['overlaps']>=.5))).sum())}
    verified=bool(score_close.all()and box_close.all()and iou_close.all()and all(p['presence_threshold_crossings']==0 and p['localization_decision_changes']==0 for p in points.values()))
    return {'verified_with_declared_numeric_tolerances':verified,'scores_allclose':bool(score_close.all()),
            'boxes_allclose':bool(box_close.all()),'ious_allclose':bool(iou_close.all()),'score_max_abs_difference':float(np.abs(scores-expected['scores']).max()),
            'box_max_abs_difference_pixels':float(np.abs(boxes-expected['boxes']).max()),
            'score_rtol':1e-4,'score_atol':1e-5,'box_rtol':1e-4,'box_atol_pixels':.01,'iou_rtol':1e-4,'iou_atol':1e-5,
            'operating_points':points,'note':'Cross-device/library differences may change threshold decisions. Original results are never overwritten or silently recalibrated.'}

def run(args):
    bundle=open_bundle(args.manifest,args.method,args.split)
    if args.download_only:
        print(json.dumps({'manifest_sha256':bundle['manifest_sha256'],'download':acquire(bundle,args.data_root,args.workers)}));return 0
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('--output must name a new directory; original results remain intact')
    if output==bundle['root']or bundle['root'].is_relative_to(output):raise ValueError('Output must not contain the published bundle')
    import torch
    torch.set_num_threads(args.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_for(bundle,args)
    from vendor_v17.detection_metrics import clip_box,iou,summarize,truth_arrays,COMMON_BOX_RULE
    source=acquire(bundle,args.data_root,args.workers)
    refs={row['category_id']:crop_reference(load_photo(bundle,args.data_root,row),row)for row in bundle['references']}
    targets=bundle['plan']['targets'];rows=bundle['rows'];banks=[model.enroll(refs[t['category_id']])for t in targets]
    scores=np.zeros((len(rows),len(targets)));boxes=np.zeros((*scores.shape,4));overlaps=np.zeros_like(scores)
    output.mkdir(parents=True);start=time.monotonic()
    with(output/'predictions.jsonl').open('x',encoding='utf-8')as f:
        for i,row in enumerate(rows):
            rgb=load_photo(bundle,args.data_root,row);encoded=model.encode_query(rgb)if hasattr(model,'encode_query')else rgb
            grouped=model.predict_many(banks,encoded)if hasattr(model,'predict_many')else[model.predict(bank,encoded)for bank in banks]
            if len(grouped)!=len(banks):raise ValueError('Reference batch size changed')
            for j,(target,proposals)in enumerate(zip(targets,grouped)):
                if not proposals:raise ValueError('Missing finite top-one prediction')
                top=dict(max(proposals,key=lambda item:item['score']));top['raw_box']=list(top['box'])
                top['box']=clip_box(top['raw_box'],rgb.shape[1],rgb.shape[0]);scores[i,j]=float(top['score']);boxes[i,j]=top['box']
                if not np.isfinite([scores[i,j],*boxes[i,j]]).all():raise ValueError('Nonfinite prediction')
                overlaps[i,j]=iou(boxes[i,j],row['boxes_by_target'][str(target['category_id'])])
                f.write(json.dumps({'image_id':row['image_id'],'category_id':target['category_id'],**top,'iou':overlaps[i,j]},allow_nan=False)+'\n')
            f.flush()
            if(i+1)%24==0 or i+1==len(rows):print(json.dumps({'completed_photos':i+1,'total':len(rows),'seconds':round(time.monotonic()-start,2)}),flush=True)
    model.assert_unchanged();present,valid=truth_arrays(rows,targets)
    summary={name:summarize(scores,overlaps,present,valid,point['cutoff'],targets)for name,point in bundle['calibration'].items()}
    verification=verify_replay(scores,boxes,overlaps,bundle['expected_arrays'],bundle['calibration'])
    np.savez_compressed(output/'arrays.npz',scores=scores,boxes=boxes,overlaps=overlaps,image_ids=[r['image_id']for r in rows],category_ids=[t['category_id']for t in targets])
    result={'method':args.method,'split':args.split,'published_manifest_sha256':bundle['manifest_sha256'],
            'download':source,'provenance':model.provenance(),'common_box_rule':COMMON_BOX_RULE,'summary':summary,
            'expected_result_verification':verification,'runtime_seconds':time.monotonic()-start,
            'scores_sha256':sha(output/'arrays.npz'),'predictions_sha256':sha(output/'predictions.jsonl'),
            'reference_images_per_target':1,'thresholds_refitted':False,'original_results_modified':False}
    write_new(output/'replay_result.json',result);print(json.dumps(verification),flush=True)
    return 0 if verification['verified_with_declared_numeric_tolerances']else 2

def template():
    descriptor=lambda path:{'path':path,'sha256':'REPLACE_WITH_MEASURED_SHA256'}
    entry={'checkpoint_sha256':'REPLACE_WITH_OWN_CHECKPOINT_SHA256',
           'inference':{'max_side':384,'proposal_limit':100,'nms_iou':.5,'region_batch':128},
           'calibration':descriptor('evaluation/results/v19_dev/calibration.json'),
           'expected':{split:{'arrays':descriptor(f'evaluation/results/v19_{split}/arrays.npz'),
                              'result':descriptor(f'evaluation/results/v19_{split}/result.json')}for split in('dev','test')}}
    return {'format':FORMAT,'completed':False,'data_plan':descriptor('evaluation/data/plan.json'),
            'test_acquisition':descriptor('evaluation/data/acquisition_public.json'),
            'code_files':{name:'REPLACE_WITH_MEASURED_SHA256'for name in ['hypernet_v19/model.py','hypernet_v19/matcher.py','hypernet_v19/proposals.py','evaluation/vendor_v17/detection_metrics.py']},
            'methods':{'v19':entry},
            'publication_notes':['Only completed measured artifacts may set completed=true.',
                'The v19 inference kwargs must be the exact frozen model_config values, including chosen384or640max_side.',
                'DEV cutoffs are replayed unchanged; no recalibration or method selection on TEST.',
                'No photos, Google weights, private workstation paths, or development-only candidates belong in this manifest.']}

def self_test():
    import unittest
    from unittest.mock import patch
    class ReplayTests(unittest.TestCase):
        def test_incomplete_manifest_cannot_reach_network(self):
            with tempfile.TemporaryDirectory()as d:
                path=Path(d)/'published_benchmark.json';write_new(path,{'format':FORMAT,'completed':False})
                with patch('urllib.request.urlopen')as network:
                    with self.assertRaisesRegex(ValueError,'completed'):open_bundle(path,'v17','test')
                    network.assert_not_called()
        def test_relative_paths_and_hashes(self):
            with tempfile.TemporaryDirectory()as d:
                root=Path(d);path=root/'source.py';path.write_text('one')
                descriptor={'path':'source.py','sha256':sha(path)};self.assertEqual(checked_descriptor(root,descriptor),path)
                path.write_text('two')
                with self.assertRaisesRegex(ValueError,'hash'):checked_descriptor(root,descriptor)
                for name in ('../secret','/secret','C:/private','..\\secret'):
                    with self.assertRaises(ValueError):relative_file(root,name)
        def test_byte_and_pixel_hashes_are_both_required(self):
            a=np.arange(6*7*3,dtype=np.uint8).reshape(6,7,3);buffer=io.BytesIO();Image.fromarray(a).save(buffer,format='PNG');payload=buffer.getvalue()
            row={'image_id':1,'height':6,'width':7};record={'sha256':hashlib.sha256(payload).hexdigest(),'pixel_sha256':pixel_sha(a)}
            self.assertTrue(np.array_equal(validate_image(payload,row,record),a))
            with self.assertRaisesRegex(ValueError,'bytes'):validate_image(payload+b'x',row,record)
            with self.assertRaisesRegex(ValueError,'pixels'):validate_image(payload,row,{**record,'pixel_sha256':'0'*64})
        def test_crop_uses_same_floor_ceil_clipping(self):
            a=np.arange(6*7*3,dtype=np.uint8).reshape(6,7,3)
            self.assertTrue(np.array_equal(crop_reference(a,{'bbox_xyxy':[-.3,1.2,4.1,5.7]}),a[1:6,0:5]))
        def test_replay_reports_threshold_crossing(self):
            scores=np.array([[.5]]);boxes=np.array([[[0,0,1,1]]]);overlaps=np.array([[1.]])
            expected={'scores':scores,'boxes':boxes,'overlaps':overlaps};cal={'primary_5pct':{'cutoff':.5}}
            self.assertTrue(verify_replay(scores,boxes,overlaps,expected,cal)['verified_with_declared_numeric_tolerances'])
            result=verify_replay(scores-1e-8,boxes,overlaps,expected,cal)
            self.assertTrue(result['scores_allclose']);self.assertFalse(result['verified_with_declared_numeric_tolerances'])
        def test_owl_staging_links_or_copies_without_mutating_supplied_weights(self):
            with tempfile.TemporaryDirectory()as directory:
                root=Path(directory);source=root/'supplied';source.mkdir();public=root/'bundle';public.mkdir()
                payloads={'config.json':b'{"tiny":true}', 'model.safetensors':b'offline fake weight bytes'}
                for name,payload in payloads.items():(source/name).write_bytes(payload)
                hashes={name:sha(source/name)for name in payloads}
                metadata={'format':'hypernet.published_owl_acquisition.v1','repo_id':'google/fake-offline-test','revision':'fixed-test-revision','files':{name:{'sha256':value}for name,value in hashes.items()}}
                write_new(public/'owl_acquisition.json',metadata)
                entry={'owl_checkpoint_files':hashes,'owl_repo_id':metadata['repo_id'],'owl_revision':metadata['revision'],'owl_acquisition':{'path':'owl_acquisition.json','sha256':sha(public/'owl_acquisition.json')}}
                bundle={'root':public,'entry':entry}
                for force_copy in (False,True):
                    if force_copy:
                        with patch('os.link',side_effect=OSError('simulated unsupported links')):temporary=stage_owl_checkpoint(bundle,source)
                    else:temporary=stage_owl_checkpoint(bundle,source)
                    staged=Path(temporary.name)
                    self.assertEqual(read(staged/'acquisition.json'),metadata)
                    self.assertEqual({name:sha(staged/name)for name in hashes},hashes)
                    temporary.cleanup();self.assertFalse(staged.exists())
                    self.assertEqual({p.name:p.read_bytes()for p in source.iterdir()},payloads)
                    self.assertFalse((source/'acquisition.json').exists())
                (source/'config.json').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint file differs'):stage_owl_checkpoint(bundle,source)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReplayTests))
    return 0 if result.wasSuccessful()else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,epilog='Example: python scripts/replay_published.py --manifest published_benchmark.json --method v19 --split test --data-root /path/to/coco-cache --checkpoint checkpoints/primary_final.pt --device cuda --output /path/to/new-replay')
    p.add_argument('--manifest',type=Path);p.add_argument('--method',choices=METHODS,default='v19');p.add_argument('--split',choices=['dev','test'],default='test')
    p.add_argument('--data-root',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--owl-checkpoint',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--download-only',action='store_true');p.add_argument('--print-manifest-template',action='store_true');p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.print_manifest_template:print(json.dumps(template(),indent=2));sys.exit(0)
    if args.self_test:sys.exit(self_test())
    if not args.manifest or not args.data_root or(not args.download_only and not args.output):p.error('--manifest, --data-root, and --output (unless --download-only) are required')
    if not 1<=args.workers<=8 or args.threads<1:p.error('--workers must be1..8 and --threads positive')
    sys.exit(run(args))
