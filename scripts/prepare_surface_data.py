"""Freeze, render and audit a wholly procedural native-resolution curriculum.

No natural images, pretrained models, or resizing enter this dataset builder.
Object seeds, camera seeds and every frame are fixed before rendering begins.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/surfaces_v10'
FREEZE=ROOT/'experiments/surface_data_plan_freeze.json'
RENDERER=ROOT/'hypernet/mathematical_surfaces.py'
EXPECTED_RENDERER_SHA='c4438068ca75eb222115339d69d093366e873effffbbb17dcbb360fd67a68157'
SIZE=128
STRENGTHS=(0.,.03,.07)
TARGET_CONVENTIONS={
    'mask':'bool[H,W]; foreground is True',
    'normal':'float16[H,W,3]; camera-space unit normal facing camera, not intrinsic outward orientation; zero outside mask',
    'depth':'float16[H,W]; positive orthographic camera distance 3-z in normalized object coordinates; zero outside mask',
    'component':'uint8[H,W]; renderer surface-component index, 255 background',
    'normal_training_note':'Convert to float32 and optionally renormalize after float16 decoding; use foreground mask for losses',
    'surface_strength':'Displacement bound before fixed base-object bounding-sphere normalization; not image blur',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def array_sha(array):
    array=np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def require(condition,message):
    if not condition:raise ValueError(message)


def make_plan():
    from hypernet.mathematical_surfaces import FAMILY_NAMES,FAMILY_TO_INDEX,FAMILY_LABELS,REVISION
    require(len(FAMILY_NAMES)==8,'Expected exactly eight preregistered families')
    records=[];objects=[]
    for split,count,seed_base in [('train',32,1009000),('heldout',8,2009000)]:
        for family in FAMILY_NAMES:
            family_index=FAMILY_TO_INDEX[family]
            for index in range(count):
                object_seed=seed_base+1000*family_index+index
                object_id=f'{split}_{family}_{index:03d}'
                view_seeds=[int.from_bytes(hashlib.sha256(f'v10:{object_id}:view:{v}'.encode()).digest()[:8],'little')%(2**63)
                            for v in (0,1)]
                require(view_seeds[0]!=view_seeds[1],'View seed collision')
                object_record={'object_id':object_id,'split':split,'family':family,'family_index':family_index,
                               'object_seed':object_seed,'view_seeds':view_seeds,'record_ids':[]}
                for view_index,view_seed in enumerate(view_seeds):
                    for strength in STRENGTHS:
                        frame_id=f'{object_id}_v{view_index}_s{int(round(1000*strength)):03d}'
                        stem=f'{split}/{family}/{frame_id}'
                        records.append({'id':frame_id,'split':split,'family':family,'family_index':family_index,
                            'object_id':object_id,'object_seed':object_seed,'view_index':view_index,'view_seed':view_seed,
                            'surface_strength':strength,'rgb_path':stem+'.rgb.png','target_path':stem+'.targets.npz',
                            'metadata_path':stem+'.metadata.json'})
                        object_record['record_ids'].append(frame_id)
                objects.append(object_record)
    seeds={s:{o['object_seed'] for o in objects if o['split']==s} for s in ('train','heldout')}
    require(len(seeds['train'])==256 and len(seeds['heldout'])==64 and not seeds['train']&seeds['heldout'],
            'Object seeds must be disjoint across splits')
    require(len({v for o in objects for v in o['view_seeds']})==640,'Expected distinct view seeds for every object/view')
    require(len(records)==1920 and len({r['id'] for r in records})==1920,'Wrong render roster')
    return {'format':'hypernet.surface_dataset_plan.v10','renderer_revision':REVISION,'size':SIZE,
            'families':list(FAMILY_NAMES),'family_to_index':FAMILY_TO_INDEX,'family_labels':FAMILY_LABELS,
            'surface_strengths':list(STRENGTHS),'objects':objects,'records':records,
            'counts':{'train_objects':256,'heldout_objects':64,'train_frames':1536,'heldout_frames':384,'total_frames':1920},
            'split_rule':'Object seed is disjoint; all views and strength controls of one object stay in one split',
            'view_seed_rule':'Low 63 bits of SHA-256(v10:<object_id>:view:<index>), little endian; disjoint across object/view pairs',
            'target_conventions':TARGET_CONVENTIONS,'supervision':'Known mathematical family labels and exact renderer geometry targets',
            'natural_images':0,'external_assets':0,'pretrained_models':False,
            'scope':'Synthetic family/detail and same-object different-view evaluation; not fresh-natural-image accuracy'}


def plan_stage():
    require(sha(RENDERER)==EXPECTED_RENDERER_SHA,'Renderer differs from reviewed stable API/hash')
    require(not (DATA/'manifest.json').exists(),'Preserve existing dataset')
    plan=make_plan();plan_path=DATA/'plan.json';write_new(plan_path,plan)
    freeze={'created_utc':datetime.now(timezone.utc).isoformat(),
        'plan':{'path':str(plan_path),'sha256':sha(plan_path)},
        'behavior_files':{str(RENDERER):sha(RENDERER),str(Path(__file__).resolve()):sha(__file__)},
        'expected_renderer_sha256':EXPECTED_RENDERER_SHA,'workers':4,'native_resolution':128,
        'generation_started_before_freeze':False,'training_or_scoring_used_for_selection':False,
        'natural_images':0,'external_assets':0,'pretrained_models':False,
        'python':platform.python_version(),'numpy':np.__version__,'target_storage':TARGET_CONVENTIONS,
        'limits':['Family names are explicitly supervised labels, not automatically discovered concepts.',
                  'Held-out object seeds are new programs from the same eight designed families.']}
    write_new(FREEZE,freeze)
    print(json.dumps({'plan':str(plan_path),'freeze':str(FREEZE),'counts':plan['counts']},indent=2),flush=True)


def check_freeze():
    freeze=read(FREEZE)
    require(sha(freeze['plan']['path'])==freeze['plan']['sha256'],'Plan changed after freeze')
    for path,digest in freeze['behavior_files'].items():require(sha(path)==digest,'Frozen generation code changed: '+path)
    return freeze


def validate_arrays(render,size=SIZE):
    rgb,mask,normal,depth,component=(render[k] for k in ('rgb','mask','normal','depth','component'))
    require(rgb.shape==(size,size,3) and rgb.dtype==np.uint8,'RGB must be native uint8 HWC')
    require(mask.shape==(size,size) and mask.dtype==np.bool_ and 0<int(mask.sum())<size*size,'Invalid foreground mask')
    require(normal.shape==(size,size,3) and normal.dtype==np.float32 and np.isfinite(normal).all(),'Invalid normal target')
    require(depth.shape==(size,size) and depth.dtype==np.float32 and np.isfinite(depth).all(),'Invalid depth target')
    require(component.shape==(size,size) and component.dtype==np.uint8,'Invalid component target')
    require(not np.any(normal[~mask]) and not np.any(depth[~mask]) and np.all(component[~mask]==255),
            'Targets must retain explicit background conventions')
    require(np.all(depth[mask]>0) and np.all(component[mask]!=255),'Invalid foreground depth/component')
    norm_error=float(np.max(np.abs(np.linalg.norm(normal[mask],axis=-1)-1)))
    require(norm_error<1e-4 and float(normal[mask][:,2].min())>=-1e-6,'Normals must be unit and camera-facing')
    return {'foreground_pixels':int(mask.sum()),'float32_max_normal_norm_error':norm_error,
            'depth_min':float(depth[mask].min()),'depth_max':float(depth[mask].max())}


def worker_init():
    check_freeze()


def render_one(record):
    from hypernet.mathematical_surfaces import render_surface
    started=time.perf_counter()
    rendered=render_surface(record['family'],record['object_seed'],record['view_seed'],record['surface_strength'],SIZE)
    qa=validate_arrays(rendered)
    require(rendered['family']==record['family'] and rendered['family_index']==record['family_index'],'Renderer family mismatch')
    require(rendered['parameters']['surface_strength']==record['surface_strength'],'Renderer strength mismatch')
    require(rendered['view']['view_seed']==record['view_seed'],'Renderer view seed mismatch')
    require(rendered['render']['native_resolution']==SIZE and rendered['render']['image_resizing'] is False
            and rendered['render']['gaussian_image_blur'] is False,'Native sharp render required')
    metadata={k:v for k,v in rendered.items() if k not in ('rgb','mask','normal','depth','component')}
    targets={'mask':rendered['mask'],'normal':rendered['normal'].astype(np.float16),
             'depth':rendered['depth'].astype(np.float16),'component':rendered['component']}
    qa['float16_max_normal_component_error']=float(np.abs(targets['normal'].astype(np.float32)-rendered['normal']).max())
    qa['float16_max_depth_error']=float(np.abs(targets['depth'].astype(np.float32)-rendered['depth']).max())
    qa['float16_max_normal_norm_error']=float(np.abs(np.linalg.norm(targets['normal'][targets['mask']].astype(np.float32),axis=-1)-1).max())
    rgb_path,target_path,metadata_path=(DATA/record[k] for k in ('rgb_path','target_path','metadata_path'))
    require(not any(p.exists() for p in (rgb_path,target_path,metadata_path)),'Preserve existing render files')
    rgb_path.parent.mkdir(parents=True,exist_ok=True)
    with rgb_path.open('xb') as stream:Image.fromarray(rendered['rgb']).save(stream,format='PNG')
    with target_path.open('xb') as stream:np.savez_compressed(stream,**targets)
    write_new(metadata_path,metadata)
    return {**record,'rgb_sha256':sha(rgb_path),'target_sha256':sha(target_path),'metadata_sha256':sha(metadata_path),
            'rgb_array_sha256':array_sha(rendered['rgb']),'target_array_sha256':{k:array_sha(v) for k,v in targets.items()},
            'base_program_id':rendered['base_program_id'],'program_id':rendered['program_id'],
            'geometry_sha256':rendered['geometry_sha256'],'view':rendered['view'],
            'maximum_actual_displacement':rendered['parameters']['maximum_actual_displacement'],
            'qa':qa,'elapsed_seconds':time.perf_counter()-started}


def render_stage():
    freeze=check_freeze();plan=read(freeze['plan']['path'])
    require(not (DATA/'manifest.json').exists(),'Preserve existing manifest')
    started=time.perf_counter();results={}
    with ProcessPoolExecutor(max_workers=4,initializer=worker_init) as pool:
        futures={pool.submit(render_one,row):row['id'] for row in plan['records']}
        for future in as_completed(futures):
            row=future.result();require(row['id'] not in results,'Duplicate rendered ID');results[row['id']]=row
            if len(results)%128==0:print(f'Rendered {len(results)}/1920 frames',flush=True)
    check_freeze();require(len(results)==1920,'Incomplete render job')
    records=[results[row['id']] for row in plan['records']]
    manifest={**plan,'format':'hypernet.surface_dataset.v10','completed':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),'records':records,
        'plan_freeze':{'path':str(FREEZE),'sha256':sha(FREEZE)},
        'elapsed_seconds':time.perf_counter()-started}
    write_new(DATA/'manifest.json',manifest)
    print(json.dumps({'manifest':str(DATA/'manifest.json'),'frames':len(records),
                      'elapsed_seconds':manifest['elapsed_seconds']},indent=2),flush=True)


def audit_stage():
    freeze=check_freeze();plan=read(freeze['plan']['path']);manifest=read(DATA/'manifest.json')
    require(manifest['completed'] is True,'Incomplete dataset')
    require(manifest['plan_freeze']['sha256']==sha(FREEZE),'Manifest freeze mismatch')
    require([r['id'] for r in manifest['records']]==[r['id'] for r in plan['records']],'Manifest plan roster differs')
    require(manifest['objects']==plan['objects'],'Object identity plan changed')
    groups=defaultdict(list);seen_paths=set();rgb_hashes=set();geometry_ids=defaultdict(set);metadata_by_id={}
    counters=Counter();max_norm_error=max_depth_quantization=0.
    for row,planned in zip(manifest['records'],plan['records']):
        require(all(row[k]==v for k,v in planned.items()),'Rendered record changed a planned field')
        for key,digest_key in [('rgb_path','rgb_sha256'),('target_path','target_sha256'),('metadata_path','metadata_sha256')]:
            path=(DATA/row[key]).resolve();require(path.is_relative_to(DATA.resolve()) and path not in seen_paths,'Unsafe/duplicate output path')
            seen_paths.add(path);require(sha(path)==row[digest_key],'Stored file hash mismatch')
        with Image.open(DATA/row['rgb_path']) as image:
            require(image.mode=='RGB' and image.size==(SIZE,SIZE),'Stored RGB resolution/mode changed')
            rgb=np.array(image,dtype=np.uint8)
        require(array_sha(rgb)==row['rgb_array_sha256'],'Stored RGB pixel hash mismatch');rgb_hashes.add(row['rgb_array_sha256'])
        with np.load(DATA/row['target_path'],allow_pickle=False) as archive:
            require(set(archive.files)=={'mask','normal','depth','component'},'Target channel roster mismatch')
            targets={k:archive[k] for k in archive.files}
        for k,v in targets.items():require(array_sha(v)==row['target_array_sha256'][k],'Stored decoded target mismatch')
        require(targets['normal'].dtype==np.float16 and targets['depth'].dtype==np.float16,'Target storage dtype mismatch')
        mask=targets['mask'];normals=targets['normal'].astype(np.float32);depth=targets['depth'].astype(np.float32)
        require(mask.dtype==np.bool_ and mask.shape==(SIZE,SIZE) and bool(mask.any()),'Stored mask mismatch')
        require(normals.shape==(SIZE,SIZE,3) and depth.shape==(SIZE,SIZE) and
                targets['component'].shape==(SIZE,SIZE) and targets['component'].dtype==np.uint8,
                'Stored target shape/dtype mismatch')
        require(np.isfinite(normals).all() and np.isfinite(depth).all() and np.all(depth[mask]>0),'Stored geometry is invalid')
        require(not np.any(normals[~mask]) and not np.any(depth[~mask]) and np.all(targets['component'][~mask]==255),
                'Stored target background mismatch')
        norm_error=float(np.abs(np.linalg.norm(normals[mask],axis=-1)-1).max())
        require(norm_error<.001,'Float16 normals exceed declared unit-normal tolerance')
        max_norm_error=max(max_norm_error,norm_error);max_depth_quantization=max(max_depth_quantization,row['qa']['float16_max_depth_error'])
        metadata=read(DATA/row['metadata_path']);metadata_by_id[row['id']]=metadata
        for key in ('family','family_index','base_program_id','program_id','geometry_sha256','view'):
            require(metadata[key]==row[key],'Exact renderer metadata mismatch: '+key)
        require(metadata['render']['native_resolution']==128 and metadata['render']['image_resizing'] is False
                and metadata['render']['gaussian_image_blur'] is False and metadata['render']['natural_images']==0
                and metadata['render']['external_assets']==0 and metadata['render']['pretrained_models'] is False,
                'Renderer budget/native-resolution claim failed')
        require(metadata['parameters']['maximum_actual_displacement']<=row['surface_strength']+1e-12,
                'Geometry displacement exceeds declared bound')
        if row['surface_strength']==0.:
            require(metadata['parameters']['maximum_actual_displacement']==0.,'Zero-strength geometry control changed')
        groups[row['object_id']].append(row);geometry_ids[row['split']].add(row['base_program_id'])
        counters[(row['split'],row['family'])]+=1
    require(len(groups)==320 and not geometry_ids['train']&geometry_ids['heldout'],'Object geometry split leakage')
    for split,frames_per_family in [('train',192),('heldout',48)]:
        require(all(counters[(split,f)]==frames_per_family for f in plan['families']),'Unbalanced family/split roster')
    view_pairs=zero_controls=0
    for object_id,rows in groups.items():
        require(len(rows)==6 and len({r['base_program_id'] for r in rows})==1,'Object must contain exactly six related frames')
        require(len({r['object_seed'] for r in rows})==1,'Object seed changed across views/strengths')
        for strength in STRENGTHS:
            pair=sorted([r for r in rows if r['surface_strength']==strength],key=lambda r:r['view_index'])
            require(len(pair)==2 and [r['view_index'] for r in pair]==[0,1],'Missing paired view')
            require(pair[0]['program_id']==pair[1]['program_id'] and pair[0]['geometry_sha256']==pair[1]['geometry_sha256'],
                    'Object geometry changed with view seed')
            require(pair[0]['view']['azimuth']!=pair[1]['view']['azimuth'] and
                    pair[0]['view']['material_rgb']!=pair[1]['view']['material_rgb'] and
                    pair[0]['rgb_array_sha256']!=pair[1]['rgb_array_sha256'],'Two views must change camera, appearance and pixels')
            view_pairs+=1
            if strength==0.:zero_controls+=2
        require(len({r['geometry_sha256'] for r in rows})==3,'Three strengths must produce distinct geometry')
        for view_index in (0,1):
            controlled=[r for r in rows if r['view_index']==view_index]
            require(all(r['view']==controlled[0]['view'] for r in controlled),'View/light/material changed across strength controls')
    require(view_pairs==960 and zero_controls==640,'Wrong paired-view/control counts')
    result={'valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(DATA/'manifest.json'),
        'plan_sha256':sha(DATA/'plan.json'),'plan_freeze_sha256':sha(FREEZE),'renderer_sha256':sha(RENDERER),
        'builder_sha256':sha(__file__),'records_verified':1920,'files_verified':len(seen_paths),
        'objects_by_split':{s:len(geometry_ids[s]) for s in ('train','heldout')},
        'frames_by_split':{'train':1536,'heldout':384},'paired_view_checks':view_pairs,'zero_strength_frames':zero_controls,
        'unique_rgb_arrays':len(rgb_hashes),'no_object_seed_or_base_program_overlap':True,
        'all_actual_camera_and_material_pairs_differ':True,'all_strength_controls_share_view_parameters':True,
        'all_strengths_have_distinct_geometry':True,'native_resolution':128,'image_blur_or_resizing':False,
        'natural_images':0,'external_assets':0,'pretrained_models':False,
        'max_float16_normal_norm_error':max_norm_error,'max_float16_depth_quantization_error':max_depth_quantization,
        'target_conventions':TARGET_CONVENTIONS,'scope':plan['scope']}
    write_new(ROOT/'experiments/surface_data_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('plan','render','audit','all'),required=True)
    args=parser.parse_args()
    for stage,fn in [('plan',plan_stage),('render',render_stage),('audit',audit_stage)]:
        if args.stage in (stage,'all'):fn()


if __name__=='__main__':main()
