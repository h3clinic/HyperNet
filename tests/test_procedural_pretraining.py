import numpy as np
import pytest
import torch
from hypernet.procedural_shapes import procedural_triplets,render,Program
from hypernet.procedural_pretraining import fine_target,transformed_triplets,train_procedural,FORMAT

def test_generator_deterministic_visible_nuisance_matched_and_no_assets():
    x=procedural_triplets(8,17);y=procedural_triplets(8,17)
    assert torch.equal(x['images'],y['images']) and torch.equal(x['masks'],y['masks'])
    assert x['images'].shape==(24,3,32,32) and x['masks'].shape==(24,1,32,32)
    assert x['images'].min()>=0 and x['images'].max()<=1
    assert x['program_sha256']==y['program_sha256']
    a,p,n=x['masks'].chunk(3);assert torch.equal(a,p)
    for row in x['records']:assert row['matched_nuisance_pixel_difference']>.003
    # Same program/nuisance gives identical pixels; no hidden reference photo input.
    row=x['records'][0];rgb,mask=render(Program(**row['program']),row['query_nuisance_seed'])
    np.testing.assert_array_equal(rgb,x['images'][8].permute(1,2,0).mul(255).round().byte().numpy())

def test_common_deformation_preserves_positive_masks_and_keeps_inputs():
    torch.set_num_threads(1);data=procedural_triplets(3,19);before=data['images'].clone()
    images,masks=transformed_triplets(data,73)
    assert torch.equal(masks[:3],masks[3:6])
    assert torch.equal(data['images'],before)
    target=fine_target(images,masks)
    assert target.shape==(9,512) and torch.isfinite(target).all()
    torch.testing.assert_close(target.norm(dim=1),torch.ones(9),atol=1e-5,rtol=0)
    assert torch.isfinite(fine_target(torch.zeros_like(images),torch.zeros_like(masks))).all()

def test_ring_width_mutation_keeps_valid_inner_ellipse():
    # A thick ring remains drawable when a mutation reduces its vertical radius.
    p=Program(4,48.,48.,20.,15.,10.,17.,0.,4.,0.,.1)
    rgb,mask=render(p,10)
    assert rgb.shape==(32,32,3) and 0<mask.mean()<1

def test_pretraining_checkpoint_is_procedural_and_modes_share_inputs(tmp_path):
    torch.set_num_threads(1)
    a=train_procedural(tmp_path/'a.pt','restoration',steps=2,triplets_per_step=2)
    b=train_procedural(tmp_path/'b.pt','detail_metric',steps=2,triplets_per_step=2)
    assert a['initial_state_sha256']==b['initial_state_sha256']
    assert a['program_sequence_sha256']==b['program_sequence_sha256']
    assert a['rendered_image_exposures']==12 and a['corrupted_view_exposures']==24
    assert b['natural_training_images']==b['natural_pretraining_images']==0
    assert not b['externally_provided_pretrained_weights']
    ckpt=torch.load(tmp_path/'b.pt',weights_only=True);assert ckpt['format']==FORMAT
    assert all(torch.isfinite(v).all() for v in ckpt['state_dict'].values())
    with pytest.raises(FileExistsError):train_procedural(tmp_path/'b.pt',steps=1)
