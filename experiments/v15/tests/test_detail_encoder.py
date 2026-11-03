import sys
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hypernet.detail_encoder import DetailEncoder,detail_loss,sample_at_grid,correspondence_direction,state_hash,conservative_validity


def identity_grid(batch,size):
    y,x=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    return (2*(torch.stack([x,y],-1).float()+.5)/size-1)[None].expand(batch,-1,-1,-1).clone()


def test_pixel_center_sampling_has_no_half_pixel_drift():
    field=torch.arange(64.).reshape(1,1,8,8)
    grid=identity_grid(1,8).reshape(1,64,2)
    assert torch.allclose(sample_at_grid(field,grid).flatten(),field.flatten(),atol=1e-6)


def test_explicit_bilinear_sampling_matches_framework_forward_and_backward():
    torch.manual_seed(4);a=torch.randn(2,3,16,20,requires_grad=True);b=a.detach().clone().requires_grad_()
    grid=torch.rand(2,37,2)*2.4-1.2
    actual=sample_at_grid(a,grid)
    expected=F.grid_sample(b,grid[:,None],mode='bilinear',padding_mode='zeros',align_corners=False).squeeze(2).transpose(1,2)
    assert torch.allclose(actual,expected,atol=2e-6)
    actual.square().sum().backward();expected.square().sum().backward()
    assert torch.allclose(a.grad,b.grad,atol=5e-6)


def test_one_pixel_phase_survives_stride_one_features():
    torch.set_num_threads(2);torch.manual_seed(1515);model=DetailEncoder().eval()
    a=torch.zeros(1,3,32,32);b=a.clone();a[:,:,16,16]=1;b[:,:,16,17]=1
    assert a.mean()==b.mean()
    with torch.no_grad():first=model(a);second=model(b)
    assert first['dense'].shape==(1,24,32,32)
    assert first['fine'].shape[-2:]==(32,32)
    assert (first['fine']-second['fine']).abs().max()>.01
    assert (first['dense']-second['dense']).abs().max()>.01
    fine=first['fine']
    assert torch.equal(F.pixel_shuffle(F.pixel_unshuffle(fine,2),2),fine)


def test_correct_translation_correspondence_beats_wrong_map():
    torch.set_num_threads(2);torch.manual_seed(2);size=32
    rgb=torch.rand(2,3,size,size)
    source=F.normalize(torch.randn(2,12,size,size),dim=1)
    target=torch.roll(source,3,-1)
    first={'dense':source,'raw_dense':source};second={'dense':target,'raw_dense':target}
    grid=identity_grid(2,size);grid[...,0]+=6/size
    valid=torch.ones(2,size,size,dtype=torch.bool);valid[:,:,-4:]=False;valid[:,:,:3]=False
    good,_,stats=correspondence_direction(first,second,rgb,grid,valid,valid,torch.Generator().manual_seed(5))
    bad,_,wrong=correspondence_direction(first,second,rgb,identity_grid(2,size),valid,valid,torch.Generator().manual_seed(5))
    assert stats['correspondence_accuracy']>.95
    assert good<bad
    assert stats['positive_cosine']>.999
    assert wrong['positive_cosine']<.5


def test_losses_reach_dense_detail_and_context_parameters():
    torch.set_num_threads(2);torch.manual_seed(3);model=DetailEncoder();before=state_hash(model)
    a=torch.rand(2,3,32,32);b=torch.roll(a,1,-1);grid=identity_grid(2,32)
    forward=grid.clone();forward[...,0]+=2/32
    backward=grid.clone();backward[...,0]-=2/32
    valid=torch.ones(2,32,32,dtype=torch.bool);valid[:,:,:2]=False;valid[:,:,-2:]=False
    batch={'rgb_a':a,'rgb_b':b,'grid_a_to_b':forward,'grid_b_to_a':backward,
        'valid_a_to_b':valid,'valid_b_to_a':valid,'detail_a':valid,'detail_b':valid,'mask_a':valid,'mask_b':valid}
    loss,stats=detail_loss(model(a),model(b),batch,torch.Generator().manual_seed(7))
    loss.backward()
    for prefix in ('stem','fine','mid','coarse','project','detail_head','mask_head'):
        gradients=[p.grad for name,p in model.named_parameters() if name.startswith(prefix)]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0
    assert state_hash(model)==before
    assert stats['informative_anchor_fraction']>.8


def test_invalid_correspondence_support_is_rejected():
    import pytest
    valid=torch.zeros(1,16,16,dtype=torch.bool)
    fake={'dense':torch.zeros(1,24,16,16),'raw_dense':torch.zeros(1,24,16,16)}
    with pytest.raises(ValueError,match='visible corresponding'):
        correspondence_direction(fake,fake,torch.zeros(1,3,16,16),identity_grid(1,16),valid,valid,torch.Generator())


def test_one_pixel_ridge_is_not_given_background_mixed_correspondence_labels():
    size=16;mask=torch.zeros(1,size,size,dtype=torch.bool);mask[:,:,8]=True
    grid=identity_grid(1,size);grid[...,0]+=1/size
    batch={'mask_a':mask,'mask_b':mask,'valid_a_to_b':mask,'valid_b_to_a':mask,
        'grid_a_to_b':grid,'grid_b_to_a':grid}
    a,b,stats=conservative_validity(batch)
    assert not a.any() and not b.any()
    assert stats['correspondence_support_retained']==0
    rgb=mask[:,None].expand(-1,3,-1,-1).float()
    batch.update(rgb_a=rgb,rgb_b=rgb,detail_a=mask,detail_b=mask)
    model=DetailEncoder();loss,values=detail_loss(model(rgb),model(rgb),batch,torch.Generator().manual_seed(3))
    loss.backward()
    assert values['correspondence_loss']==0
    assert values['detail_reconstruction']>0
    assert model.detail_head.weight.grad.abs().sum()>0


def test_cuda_training_step_is_repeatable_when_available():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA runtime not installed in CPU audit environment')
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def run():
        torch.manual_seed(1515);model=DetailEncoder().cuda();optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        rgb=torch.rand(2,3,32,32,device='cuda');mask=torch.ones(2,32,32,dtype=torch.bool,device='cuda')
        grid=identity_grid(2,32).cuda()
        batch={'rgb_a':rgb,'rgb_b':rgb,'grid_a_to_b':grid,'grid_b_to_a':grid,
            'valid_a_to_b':mask,'valid_b_to_a':mask,'detail_a':mask,'detail_b':mask,'mask_a':mask,'mask_b':mask}
        loss,_=detail_loss(model(rgb),model(rgb),batch,torch.Generator(device='cuda').manual_seed(3))
        loss.backward();optimizer.step();return state_hash(model)
    assert run()==run()
