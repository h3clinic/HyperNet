from pathlib import Path
import sys
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypernet.local_relation_model import (LocalRelationNet,FORMAT,state_hash,objective,decoded_boxes,
                                           integral_box_means,anchor_sizes)
from hypernet.local_matcher import FrozenLocalMatcher,prepare_image,nms
torch.set_num_threads(2)


def example_batch(constant=False):
    torch.manual_seed(3)
    support=torch.full((2,3,64,64),.5)if constant else torch.rand(2,3,64,64)
    query=torch.full((2,3,64,64),.5)if constant else torch.rand(2,3,64,64)
    sm=torch.zeros(2,1,64,64);sm[:,:,16:48,16:48]=1
    tm=torch.zeros(2,1,64,64);tm[0,:,20:50,18:48]=1
    return {'support':support,'query':query,'support_mask':sm,'target_mask':tm,
            'bbox':torch.tensor([[18.,20.,48.,50.],[0.,0.,0.,0.]]),'present':torch.tensor([1.,0.])}


@pytest.mark.parametrize('geometry',[True,False])
def test_constant_images_have_finite_gradients(geometry):
    torch.manual_seed(1717);model=LocalRelationNet(geometry);batch=example_batch(True)
    out=model(batch['support'],batch['query']);loss,_=objective(out,batch);loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None)
    assert model.stem[0].weight.grad.abs().sum()>0
    assert model.descriptor.weight.grad.abs().sum()>0
    assert model.match_head[-1].weight.grad.abs().sum()>0


def test_reference_changes_predictions_and_receives_gradient():
    torch.manual_seed(1717);model=LocalRelationNet();batch=example_batch()
    support=batch['support'].clone().requires_grad_(True)
    query=batch['query'].clone().requires_grad_(True)
    out=model(support,query);loss,_=objective(out,batch);loss.backward()
    assert support.grad.abs().sum()>0 and query.grad.abs().sum()>0
    with torch.no_grad():
        changed=model(1-support.detach(),query.detach())
    assert (out['center_logits']-changed['center_logits']).abs().max()>1e-6


def test_predicted_reference_extent_removes_letterbox_coordinates():
    # Oracle encoded masks are used ONLY to unit-test coordinate bookkeeping;
    # real inference always obtains logits from the model's own foreground head.
    torch.manual_seed(7);patch=torch.randn(1,32,8,12);model=LocalRelationNet()
    def encoded(h,w,y,x):
        raw=torch.randn(1,32,h,w);raw[:,:,y:y+8,x:x+12]=patch
        logits=torch.full((1,1,h,w),-20.);logits[:,:,y:y+8,x:x+12]=20.
        return {'raw':raw.requires_grad_(True),'foreground_logits':logits.requires_grad_(True)}
    a=encoded(16,18,3,2);b=encoded(24,32,10,17)
    first=model.support_bank(a);second=model.support_bank(b)
    for key in ('parts','weights','prototype'):assert torch.equal(first[key],second[key])
    assert first['predicted_extent'].tolist()==[[2,3,14,11]]
    first['parts'].square().sum().backward()
    assert a['raw'].grad is not None and torch.isfinite(a['raw'].grad).all()
    assert a['foreground_logits'].grad is not None and torch.isfinite(a['foreground_logits'].grad).all()


def test_predicted_reference_extent_empty_and_tiny_cases():
    model=LocalRelationNet();raw=torch.randn(2,32,16,16);logits=torch.full((2,1,16,16),-20.)
    logits[1,0,15,15]=20.
    result=model.support_bank({'raw':raw,'foreground_logits':logits})
    assert result['predicted_extent'].tolist()==[[0,0,16,16],[12,12,16,16]]
    assert torch.isfinite(result['parts']).all()


def test_part_order_affects_geometry_but_not_bag():
    torch.manual_seed(24);parts=torch.nn.functional.normalize(torch.randn(1,32,16),dim=1)
    bank={'parts':parts,'weights':torch.ones(1,16),'prototype':torch.nn.functional.normalize(parts.mean(2),dim=1)}
    changed={**bank,'parts':parts.flip(-1)}
    q={'dense':torch.nn.functional.normalize(torch.randn(1,32,16,16),dim=1),
       'foreground_logits':torch.zeros(1,1,16,16),'height':32,'width':32}
    geometric=LocalRelationNet(True);bag=LocalRelationNet(False)
    a=geometric.relation_evidence(bank,q)[0];b=geometric.relation_evidence(changed,q)[0]
    c=bag.relation_evidence(bank,q)[0];d=bag.relation_evidence(changed,q)[0]
    assert (a-b).abs().max()>.01
    assert torch.allclose(c,d,atol=2e-6,rtol=2e-6)


def test_other_batch_members_do_not_change_prediction():
    torch.manual_seed(1717);model=LocalRelationNet().eval();batch=example_batch()
    with torch.no_grad():
        single=model(batch['support'][:1],batch['query'][:1])['center_logits']
        together=model(batch['support'],batch['query'])['center_logits'][:1]
    assert torch.allclose(single,together,atol=2e-5,rtol=1e-5)


def test_rectangle_means_equal_direct_valid_crop():
    x=torch.arange(30,dtype=torch.float32).reshape(1,1,5,6)
    sizes=[(1,1),(3,5),(7,9)];results=integral_box_means(x,sizes)
    for (kh,kw),actual in zip(sizes,results):
        for y in range(5):
            for z in range(6):
                expected=x[...,max(0,y-kh//2):min(5,y+kh//2+1),max(0,z-kw//2):min(6,z+kw//2+1)].mean()
                assert actual[0,0,y,z]==expected


def test_rectangle_means_preserve_small_values_after_large_prefixes():
    # Cosine correlations at opposite extremes make the ablation exponentials
    # span almost seven orders of magnitude. Prefix cancellation must not turn
    # a valid low-similarity window into zero/negative or inflate its match.
    correlations=torch.ones(1,1,96,96);correlations[:,:,48:,48:]=-1
    x=(8*correlations).exp().requires_grad_(True)
    actual=integral_box_means(x,[(15,15)])[0]
    for y,z in ((75,75),(60,80),(5,5),(47,60),(90,90)):
        expected=x[...,max(0,y-7):min(96,y+8),max(0,z-7):min(96,z+8)].double().mean().float()
        assert torch.isclose(actual[0,0,y,z],expected,rtol=1e-4,atol=1e-8)
    assert (actual>0).all()
    actual.sum().backward();assert torch.isfinite(x.grad).all()


def test_box_decode_zero_delta_is_declared_anchor():
    sizes=anchor_sizes(64,64)
    out={'center_logits':torch.zeros(1,15,32,32),'box_deltas':torch.zeros(1,15,4,32,32),'anchor_sizes':sizes}
    box=decoded_boxes(out)[0,0]
    assert torch.allclose(box,torch.tensor([1.,1.,1.,1.])+torch.cat([-sizes[0]/2,sizes[0]/2]))


def checkpoint(path,geometry=True):
    torch.manual_seed(1717);model=LocalRelationNet(geometry)
    torch.save({'format':FORMAT,'state_dict':model.state_dict(),'state_sha256':state_hash(model),
        'optimizer_steps':0,'training_episodes':0,'training_views':0,'natural_training_images':0,
        'borrowed_pretrained_weights':False,'config':{'geometry':geometry}},path)


def test_frozen_api_uses_only_rgb_and_returns_absolute_boxes(tmp_path):
    p=tmp_path/'own.pt';checkpoint(p)
    model=FrozenLocalMatcher(p,max_side=64,proposal_limit=4,pre_nms=20)
    rng=np.random.default_rng(4);reference=rng.integers(0,256,(35,50,3),dtype=np.uint8)
    query=rng.integers(0,256,(51,83,3),dtype=np.uint8)
    bank=model.enroll(reference);encoded=model.encode_query(query)
    first=model.predict(bank,encoded);second=model.predict(bank,encoded)
    assert first==second and len(first)>=1
    for row in first:
        x1,y1,x2,y2=row['box'];assert 0<=x1<x2<=83 and 0<=y1<y2<=51
        assert np.isfinite(row['score'])
    assert bank['reference_images']==1 and bank['real_weight_updates']==0
    model.assert_unchanged()
    bank['state_sha256']='different'
    with pytest.raises(ValueError,match='different model'):model.predict(bank,encoded)


def test_resize_and_stable_nms():
    rgb=np.zeros((50,100,3),np.uint8);prepared,meta=prepare_image(rgb,max_side=64)
    assert prepared.shape==(32,64,3)and meta['scale_x']==.64
    assert nms(np.array([[0,0,10,10],[0,0,10,10],[20,20,30,30]]),[1,1,.5])==[0,2]
