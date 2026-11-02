"""Geometry and no-update controls use generated arrays, never real test photos."""
import numpy as np
import pytest
import torch
from hypernet.detail_pipeline import DetailPipeline, state_hash
from hypernet.one_shot_detector import OneReferenceDetector
from hypernet.real_photo import (RealPhotoDetector, reference_image, grid_proposals,
    resize_long_edge, map_boxes, validate_rgb, validate_box, ALIASES)


def test_rounded_resize_uses_independent_actual_scales():
    image=np.zeros((403,617,3),np.uint8)
    small,(sx,sy)=resize_long_edge(image,512)
    assert small.shape==(334,512,3)
    assert sx==512/617 and sy==334/403 and sx!=sy
    assert map_boxes([[0,0,512,334]],sx,sy,617,403)==[[0.,0.,617.,403.]]


def test_reference_letterbox_preserves_box_geometry_and_clips_context():
    image=np.zeros((125,301,3),np.uint8);image[:]=[20,40,60]
    canvas,box,meta=reference_image(image,[0,10,120,70])
    assert canvas.shape==(96,96,3)
    assert meta['expanded_crop'][0]==0
    sx,sy=meta['actual_resize_scales'];ox,oy=meta['padding_xy'];x1,y1,_,_=meta['expanded_crop']
    recovered=[(box[0]-ox)/sx+x1,(box[1]-oy)/sy+y1,(box[2]-ox)/sx+x1,(box[3]-oy)/sy+y1]
    assert np.allclose(recovered,[0,10,120,70])
    assert 0<=box[0]<box[2]<=96 and 0<=box[1]<box[3]<=96


def test_fixed_grid_is_dimension_only_unique_and_bounded():
    boxes=grid_proposals(617,403)
    assert len(boxes)==51 and len({tuple(b) for b in boxes})==51
    assert boxes[0]==[0.,0.,617.,403.]
    assert all(0<=b[0]<b[2]<=617 and 0<=b[1]<b[3]<=403 for b in boxes)


@pytest.mark.parametrize('box',[[0,0,0,10],[-1,0,10,10],[0,0,101,10],[0,0,float('nan'),10],[0,0,1,10]])
def test_reference_box_validation(box):
    with pytest.raises(ValueError):validate_box(box,100,100)


def test_input_contract_and_no_upsampling():
    with pytest.raises(ValueError):validate_rgb(np.zeros((96,96),np.uint8))
    with pytest.raises(ValueError):validate_rgb(np.zeros((96,96,3),np.float32))
    image=np.zeros((40,80,3),np.uint8)
    actual,scale=resize_long_edge(image,512)
    assert scale==(1.,1.) and np.array_equal(actual,image) and actual is not image


@pytest.fixture(scope='module')
def detector():
    torch.set_num_threads(2);torch.manual_seed(1313)
    item=RealPhotoDetector.__new__(RealPhotoDetector)
    item.base=DetailPipeline.__new__(DetailPipeline)
    item.model=OneReferenceDetector().eval().requires_grad_(False)
    item.base.model=item.model
    return item


def generated_example():
    reference=np.full((96,96,3),25,np.uint8)
    reference[15:80,20:75]=[150,190,160]
    reference[34:39,32:44]=25;reference[53:65,51:56]=25
    query=np.full((256,256,3),25,np.uint8)
    query[20:116,30:126]=reference
    query[135:231,140:236]=np.rot90(reference)
    return reference,query


def test_main_methods_equal_frozen_v12_after_explicit_reference_adapter(detector,monkeypatch):
    import hypernet.real_photo as adapter
    import hypernet.detail_pipeline as legacy
    proposals=[[50,35,105,100],[155,156,220,211]]
    monkeypatch.setattr(adapter,'proposals_from_foreground',lambda *a,**k:proposals)
    monkeypatch.setattr(legacy,'proposals_from_foreground',lambda *a,**k:proposals)
    reference,query=generated_example();ref_before=reference.copy();query_before=query.copy()
    before=state_hash(detector.model)
    normalized,box,_=reference_image(reference,[20,15,75,80])
    old=detector.base.detect(normalized,box,query)
    new=detector.detect(reference,[20,15,75,80],query)
    assert tuple(new)==ALIASES
    for alias in ('old_cnn','native_patch'):
        assert new[alias]['score']==old[alias]['score']
        assert new[alias]['bbox_xyxy']==old[alias]['bbox_xyxy']
        assert new[alias]['proposal_scores']==old[alias]['proposal_scores']
        assert new[alias]['evidence']==old[alias]['evidence']
    assert state_hash(detector.model)==before and all(p.grad is None for p in detector.model.parameters())
    assert np.array_equal(reference,ref_before) and np.array_equal(query,query_before)


def test_padding_does_not_leak_boxes_outside_original_image(detector,monkeypatch):
    import hypernet.real_photo as adapter
    monkeypatch.setattr(adapter,'proposals_from_foreground',lambda *a,**k:[[10,10,112,84],[109,10,112,84]])
    reference,_=generated_example()
    query=np.full((83,109,3),30,np.uint8)
    result=detector.detect(reference,[20,15,75,80],query)
    assert result['old_cnn']['proposals']==[[10.,10.,109.,83.]]
    for record in result.values():
        assert all(0<=b[0]<b[2]<=109 and 0<=b[1]<b[3]<=83 for b in record['proposals'])
        assert np.isfinite(record['score'])
