import numpy as np
import pytest
from scipy.ndimage import label
from hypernet.mathematical_shapes2d import (
    FAMILY_COUNT,decode_family,family_spec,render_episode,raster_shape,canonical_mask,
)


def test_grammar_is_discrete_composition_not_seed_count():
    assert FAMILY_COUNT == 8**4
    digits={tuple(decode_family(index)) for index in range(FAMILY_COUNT)}
    assert len(digits)==4096 and min(digits)==(0,0,0,0) and max(digits)==(7,7,7,7)
    for index in (0,1,8,64,512,4095):
        spec=family_spec(index)
        assert spec['family_digits']==decode_family(index)
        assert len(spec['grammar_names'])==4 and len(spec['canonical_program_id'])==64


@pytest.mark.parametrize('family',[0,136,511,887,1054,1733,2476,2643,3386,4065,4095])
def test_native_shapes_are_connected_and_microstates_change_equal_area_features(family):
    first=raster_shape(family,73,0)
    second=raster_shape(family,73,1)
    for item in (first,second):
        assert item['mask'].shape==(96,96) and item['mask'].dtype==np.bool_
        assert label(item['mask'],structure=np.ones((3,3)))[1]==1
        assert item['micro_statistics']['primitive_count']==2
        assert item['micro_statistics']['continuous_removed_area']==pytest.approx(.0456)
    assert not np.array_equal(first['mask'],second['mask'])
    assert np.array_equal(first['macro_mask'],second['macro_mask'])
    assert abs(int(first['mask'].sum())-int(second['mask'].sum()))/first['macro_mask'].sum() < .02


@pytest.mark.parametrize('scene_seed',[0,17,991])
def test_query_pixels_cannot_predict_counterbalanced_presence(scene_seed):
    arguments=dict(family_id=1234,object_seed=14,scene_seed=scene_seed,distractor_family_ids=[1,19,522,2711,4000])
    first=render_episode(enrolled_variant=0,**arguments)
    second=render_episode(enrolled_variant=1,**arguments)
    assert first['support_rgb'].shape==(96,96,3)
    assert not np.array_equal(first['support_rgb'],second['support_rgb'])
    assert first['parameters']['support_parameters']==second['parameters']['support_parameters']
    for variant in ('0','1'):
        a,b=first['queries'][variant],second['queries'][variant]
        assert a['query_rgb'].shape==(256,256,3) and a['query_rgb'].dtype==np.uint8
        for key in ('query_rgb','union_mask','candidate_mask'):
            np.testing.assert_array_equal(a[key],b[key])
        assert a['matches_reference'] != b['matches_reference']
        matching=a if a['matches_reference'] else b
        missing=b if a['matches_reference'] else a
        np.testing.assert_array_equal(matching['target_mask'],matching['candidate_mask'])
        assert not missing['target_mask'].any() and missing['target_bbox_xyxy'] is None
        assert label(a['candidate_mask'],structure=np.ones((3,3)))[1]==1
        assert label(a['union_mask'],structure=np.ones((3,3)))[1]==6
        candidate=next(record for record in a['object_records'] if record['role']=='candidate')
        assert candidate['instance_seed'] != arguments['object_seed']
        assert candidate['parameters'] != first['parameters']['support_parameters']
        assert a['micro_statistics']['minimum_nominal_feature_width_pixels'] >= 4
        assert a['micro_statistics']['maximum_nominal_feature_length_pixels'] <= 8
        assert all(record['family_id']!=arguments['family_id'] for record in a['object_records'] if record['role']=='distractor')
    zero,one=first['queries']['0'],first['queries']['1']
    changed=np.any(zero['query_rgb']!=one['query_rgb'],axis=-1)
    assert changed.sum()>10
    assert not np.any(changed&~(zero['candidate_mask']|one['candidate_mask']))
    assert zero['candidate_bbox_xyxy']==one['candidate_bbox_xyxy']
    for key in ('micro_control','primitive_count','continuous_removed_area'):
        assert zero['micro_statistics'][key]==one['micro_statistics'][key]


def test_six_object_bboxes_are_tight_and_nonoverlapping():
    result=render_episode(4095,19,1,44,[0,5,91,512,2000])
    for query in result['queries'].values():
        boxes=query['object_boxes']
        assert len(boxes)==6
        for i,(x0,y0,x1,y1) in enumerate(boxes):
            assert 0<=x0<x1<=256 and 0<=y0<y1<=256
            for bx0,by0,bx1,by1 in boxes[i+1:]:
                assert min(x1,bx1)<=max(x0,bx0) or min(y1,by1)<=max(y0,by0)
        rows=np.argwhere(query['candidate_mask'])
        y0,x0=rows.min(0);y1,x1=rows.max(0)+1
        assert query['candidate_bbox_xyxy']==[x0,y0,x1,y1]


def test_caller_cannot_mutate_cached_queries():
    args=(114,13,0,8,[15,90,1045,2300,3100])
    first=render_episode(*args)
    expected=first['queries']['0']['query_rgb'].copy()
    first['queries']['0']['query_rgb'][:]=0
    first['queries']['0']['object_records'][0]['bbox_xyxy'][0]=999
    second=render_episode(*args)
    np.testing.assert_array_equal(second['queries']['0']['query_rgb'],expected)
    assert second['queries']['0']['object_records'][0]['bbox_xyxy'][0]!=999


@pytest.mark.parametrize('change',[{'family_id':4096},{'family_id':True},{'enrolled_variant':2},
    {'object_seed':-1},{'support_size':64},{'query_size':320},{'distractor_family_ids':[0,2,3,4]},
    {'distractor_family_ids':[1,1,2,3]},{'distractor_family_ids':[1,2,3]}])
def test_invalid_episode_requests_rejected(change):
    arguments=dict(family_id=0,object_seed=1,enrolled_variant=0,scene_seed=2,distractor_family_ids=[1,2,3,4,5])
    with pytest.raises(ValueError):render_episode(**(arguments|change))
