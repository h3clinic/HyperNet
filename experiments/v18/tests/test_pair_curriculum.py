from pathlib import Path
import sys,hashlib
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet_v18 import pair_curriculum as pc
from hypernet_v18.vendor_v17 import scene_curriculum as sc
from hypernet_v18.vendor_v17.local_matcher import prepare_image


def test_frozen_source_and_deterministic_rgb_contract():
    assert pc.verify_frozen_source()==pc.V17_SOURCE_SHA256
    a=pc.pair(5);b=pc.pair(5)
    for key in pc.IMAGE_KEYS:
        assert a[key].shape==(96,96,3)and a[key].dtype==np.uint8
        assert np.array_equal(a[key],b[key])
        assert hashlib.sha256(a[key].tobytes()).hexdigest()==a['metadata']['image_sha256'][key]
    assert a['metadata']==b['metadata']
    assert not np.array_equal(a['support'],a['positive'])
    assert a['metadata']['support_instance_seed']!=a['metadata']['query_instance_seed']


@pytest.mark.parametrize('index',range(12))
def test_counterfactual_is_one_edit_and_visibly_distinct_after_resize(index):
    row=pc.pair(index);meta=row['metadata'];target=meta['program'];negative=meta['near_negative_program']
    assert any(sc.program_id(p)==sc.program_id(negative)for p,_ in sc._single_edits(target))
    assert sc.program_split(target)==sc.program_split(negative)=='train'
    assert meta['positive_crop_xyxy']==meta['near_negative_crop_xyxy']
    changed=np.max(np.abs(row['positive'].astype(np.int16)-row['near_negative'].astype(np.int16)),axis=2)>=3
    assert changed.sum()==meta['final_changed_rgb_pixels']and changed.sum()>=8
    assert meta['positive_target_visibility']>=.65
    assert meta['negative_slot_visibility']>=.65
    assert meta['wrong_program']['family']!=target['family']
    assert meta['wrong_program_id']not in meta['scene_complete_and_contained_program_ids']
    assert meta['natural_images_used']==0 and meta['borrowed_pretrained_weights']is False


def test_whole_program_splits_and_compositional_exception():
    ids={}
    for split in ('train','development','test','compositional'):
        row=pc.pair(14,split=split);m=row['metadata']
        programs=[m['program'],m['near_negative_program'],m['wrong_program']]
        assert all(sc.program_split(p)==split for p in programs)
        ids[split]={sc.program_id(p)for p in programs}
        if split=='compositional':
            assert m['wrong_family_rule']=='different_constituent_family_pair'
            assert tuple(p['family']for p in m['program']['parts'])!=tuple(p['family']for p in m['wrong_program']['parts'])
    for split,values in ids.items():
        assert all(not values&other for name,other in ids.items()if name!=split)


def test_shared_letterbox_does_not_create_differences_in_padding():
    a=np.zeros((30,60,3),np.uint8)+35;b=a.copy();b[10:15,20:25]=100
    images,frame=pc._fit_pair(a,b,[0,0,60,30],96)
    x,y,w,h=frame['letterbox_xywh'];outside=np.ones((96,96),bool);outside[y:y+h,x:x+w]=False
    assert np.array_equal(images[0][outside],images[1][outside])


def test_patch_preprocessing_exactly_matches_frozen_inference():
    row=pc.pair(12);m=row['metadata'];attempt=m['nuisance_attempt']
    positive=sc._episode_once(12,'train',1801,96,192,1.,.65,'cluttered',nuisance_attempt=attempt)
    negative=sc._episode_once(12,'train',1801,96,192,0.,.65,'cluttered',nuisance_attempt=attempt)
    x1,y1,x2,y2=m['shared_crop']['crop_xyxy']
    for key,scene in (('positive',positive),('near_negative',negative)):
        expected,_=prepare_image(scene['query'][y1:y2,x1:x2],support=True)
        assert np.array_equal(row[key],expected)


def test_invalid_arguments_and_lazy_batch():
    with pytest.raises(ValueError):pc.pair(-1)
    with pytest.raises(ValueError):pc.pair(0,size=64)
    with pytest.raises(ValueError):pc.pair(0,split='dev')
    with pytest.raises(ValueError):pc.batch([])
    data=pc.batch([0,1]);assert data['support'].shape==(2,3,96,96)
    assert all(float(data[key].min())>=0 and float(data[key].max())<=1 for key in pc.IMAGE_KEYS)
