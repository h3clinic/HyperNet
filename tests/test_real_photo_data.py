import importlib.util
from pathlib import Path
import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/prepare_real_photos.py'
spec=importlib.util.spec_from_file_location('v13data',SOURCE)
data=importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def test_matching_recovers_collision_without_duplicate_photos():
    assignment=data.match_unique({'a':[1,2],'b':[1],'c':[2,3]})
    assert assignment=={'b':1,'a':2,'c':3}


def test_matching_rejects_infeasible_unique_roster():
    with pytest.raises(ValueError,match='Insufficient'):
        data.match_unique({'a':[1],'b':[1]})


def test_eligibility_has_crowd_size_and_area_limits():
    image={'width':100,'height':100}
    assert data.eligible({'bbox':[0,0,20,20],'iscrowd':0},image)
    assert not data.eligible({'bbox':[0,0,20,20],'iscrowd':1},image)
    assert not data.eligible({'bbox':[0,0,15,90],'iscrowd':0},image)
    assert not data.eligible({'bbox':[0,0,20,20],'iscrowd':0},image,reference=True)


def test_hash_binds_dtype_shape_and_pixels():
    x=np.arange(12,dtype=np.uint8).reshape(2,2,3)
    assert data.pixels(x)!=data.pixels(x.reshape(1,4,3))
    assert data.pixels(x)!=data.pixels(x.astype(np.int8))
    assert data.pixels(x)==data.pixels(x.copy())


def test_write_new_preserves_evidence(tmp_path):
    p=tmp_path/'proof.json';data.write_new(p,{'valid':True})
    with pytest.raises(FileExistsError):data.write_new(p,{'valid':False})
    assert data.read(p)=={'valid':True}


def test_plan_preserves_all_targets_and_global_photo_independence():
    source={'images':[],'annotations':[],'categories':[],'licenses':[{'id':4,'name':'BY','url':'license'}]}
    for category in range(1,13):
        source['categories'].append({'id':category,'name':f'type{category}'})
        for n in range(26):
            iid=category*100+n
            source['images'].append({'id':iid,'width':100,'height':100,'license':4,'file_name':f'{iid}.jpg','coco_url':f'https://example/{iid}.jpg'})
            source['annotations'] += [{'id':iid*2,'image_id':iid,'category_id':category,'bbox':[5,5,40,40],'iscrowd':0},
                                      {'id':iid*2+1,'image_id':iid,'category_id':category,'bbox':[70,70,5,5],'iscrowd':0}]
    plan=data.select_plan(source,{'excluded_coco_image_ids':[]})
    rows=plan['references']+plan['queries']
    assert len(rows)==len({r['image_id'] for r in rows})==300
    assert len(plan['references'])==12
    assert len([r for r in plan['queries'] if r['split']=='dev'])==96
    for row in plan['queries']:
        assert len(row['gt_boxes_xyxy'])==(2 if row['present'] else 0)
    repeated=data.select_plan(source,{'excluded_coco_image_ids':[]})
    assert plan['references']==repeated['references'] and plan['queries']==repeated['queries']
