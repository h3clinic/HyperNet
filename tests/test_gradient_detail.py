import numpy as np
from hypernet.gradient_detail import (describe_gradient_region,reference_patches,
    compare_patches,match_gradient,make_reference,compare,unit,detail_rgb)

def example(variant=0):
    y,x=np.mgrid[:96,:96]
    outer=((x-48)**2+(y-48)**2<28**2)|((x>=65)&(x<88)&(y>=42)&(y<56))
    rgb=np.full((96,96,3),20,np.uint8);rgb[outer]=120
    if variant==0:
        rgb[35:39,37:45]=20;rgb[55:63,53:57]=20
    else:
        rgb[33:41,39:43]=20;rgb[57:61,51:59]=20
    yy,xx=np.nonzero(outer)
    return rgb,[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]

def test_same_input_and_local_change():
    rgb,box=example(0);other,_=example(1)
    positive=match_gradient(rgb,box,rgb,[box])
    negative=match_gradient(rgb,box,other,[box])
    assert positive['score']>.99999
    assert positive['score']>negative['score']+.1
    assert positive['proposals']==[box] and positive['bbox_xyxy']==box

def test_exact_quarter_turn_and_affine_brightness():
    rgb,box=example(0)
    query=np.rot90(rgb*2+10).copy()
    x1,y1,x2,y2=box;qbox=[y1,96-x2,y2,96-x1]
    result=match_gradient(rgb,box,query,[qbox])
    assert result['score']>.99999

def test_no_hole_feature_extractor_is_used(monkeypatch):
    import hypernet.native_detail as native
    def prohibited(*args,**kwargs):raise AssertionError('Interior component descriptor was called')
    monkeypatch.setattr(native,'_features',prohibited)
    rgb,box=example(0)
    description=describe_gradient_region(rgb,box)
    assert 'features' not in description
    patches=reference_patches(rgb,description)
    assert 1<=len(patches)<=12
    assert compare_patches(description,description,patches,rgb)['score']>.99999

def test_empty_proposals_and_blank_images():
    rgb,box=example(0)
    empty=match_gradient(rgb,box,rgb,[])
    assert empty['score']==0 and empty['bbox_xyxy'] is None
    flat=np.full_like(rgb,20)
    result=match_gradient(flat,box,flat,[box])
    assert result['score']==0 and np.isfinite(result['score'])

def test_lowpass_changes_only_patch_vectors_not_locations_or_alignment():
    rgb,box=example(0);query,_=example(1)
    reference=make_reference(rgb,box)
    description=describe_gradient_region(query,box)
    native=compare(reference,description,query,1)
    lowpass=compare(reference,description,query,4)
    assert native['angle']==lowpass['angle']
    assert native['outline']==lowpass['outline']
    assert native['patch_count']==lowpass['patch_count']
    for first,second in zip(reference['patches_by_stride'][1],reference['patches_by_stride'][4]):
        for key in ('position','u','v'):assert np.array_equal(first[key],second[key])
    assert any(not np.array_equal(a['vector'],b['vector']) for a,b in
        zip(reference['patches_by_stride'][1],reference['patches_by_stride'][4]))

def test_constant_noise_floor_and_stride_validation():
    import pytest
    values=np.ones((7,7))*123+np.arange(49).reshape(7,7)*1e-5
    assert not unit(values).any()
    rgb,_=example()
    for stride in (0,-1,1.5,True):
        with pytest.raises(ValueError):detail_rgb(rgb,stride)
