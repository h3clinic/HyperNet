import hashlib
import numpy as np
import pytest
import torch
from hypernet import math_curriculum as cpu
from hypernet import math_curriculum_torch as tensor


def test_namespace_and_structural_modes():
    assert cpu.FAMILY_COUNT==1_048_576 and len(cpu.GEOMETRY_MODES)==8
    ids=np.r_[np.arange(32),cpu.FAMILY_COUNT+np.arange(32)]
    batch=cpu.sample_batch(ids,seed=7,size=64)
    assert batch['view_a'].shape==(64,64,64,3)
    assert batch['view_a'].dtype==np.uint8
    assert batch['mask_a'].dtype==bool and batch['grid_a_to_b'].dtype==np.float32
    assert np.isfinite(batch['grid_a_to_b']).all()
    assert np.all(batch['mask_a'].sum((1,2))>=16)
    assert np.all((batch['detail_a']&batch['valid_a_to_b']).sum((1,2))>=8)
    assert np.array_equal(batch['parameters']['geometry_mode'],ids%8)
    assert len({hashlib.sha256(x.tobytes()).digest() for x in batch['canonical_raster']})==64


@pytest.mark.parametrize('seed',[0,19,1515])
def test_partition_invariance_and_no_global_rng_changes(seed):
    ids=np.array([0,7,17,1024,1_048_576,2**31-1],np.int64)
    before=np.random.get_state();all_rows=cpu.sample_batch(ids,seed,32)
    for i in range(len(ids)):
        single=cpu.render_pair(int(ids[i]),seed,32)
        for name in ('view_a','view_b','canonical_raster','mask_a','grid_a_to_b'):
            np.testing.assert_array_equal(all_rows[name][i],single[name])
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_seed_changes_only_view_and_appearance_not_canonical_geometry():
    ids=np.arange(16)
    a,b=cpu.sample_batch(ids,0,64),cpu.sample_batch(ids,999,64)
    np.testing.assert_array_equal(a['canonical_raster'],b['canonical_raster'])
    np.testing.assert_array_equal(a['canonical_mask'],b['canonical_mask'])
    assert not np.array_equal(a['view_a'],b['view_a'])
    assert not np.array_equal(a['canonical_to_view_a'],b['canonical_to_view_a'])
    for key in a['parameters']:np.testing.assert_array_equal(a['parameters'][key],b['parameters'][key])


def test_exact_analytic_coordinate_map_both_directions():
    batch=cpu.sample_batch(np.arange(24),37,64)
    coord=2*(np.arange(64,dtype=np.float32)+.5)/64-1
    yy,xx=np.meshgrid(coord,coord,indexing='ij')
    for first,second in [('a','b'),('b','a')]:
        source=batch['canonical_to_view_'+first];target=batch['canonical_to_view_'+second]
        cx,cy=cpu._apply(cpu._invert(source),xx[None],yy[None])
        expected=cpu._apply(target,cx,cy)
        np.testing.assert_array_equal(batch['grid_'+first+'_to_'+second],np.stack(expected,-1))
        actual_canonical=cpu._apply(cpu._invert(target),*expected)
        np.testing.assert_allclose(actual_canonical[0],cx,atol=6e-7,rtol=0)
        np.testing.assert_allclose(actual_canonical[1],cy,atol=6e-7,rtol=0)
        valid=batch['valid_'+first+'_to_'+second]
        assert not (valid&~batch['mask_'+first]).any()
        assert (np.abs(batch['grid_'+first+'_to_'+second][valid])<=1-1/64).all()


def test_crisp_material_is_discrete_not_smoothed():
    batch=cpu.sample_batch(np.arange(128),1515,64)
    allowed=np.rint(np.array([0]+[.28+.36*a+.28*b+.08*c for a in (0,1) for b in (0,1) for c in (0,1)])*255).astype(np.uint8)
    assert set(np.unique(batch['canonical_raster']))<=set(allowed)
    assert all(batch['canonical_mask'][i].any() for i in range(128))


def test_torch_cpu_parity_and_training_gradient_compatibility():
    torch.set_num_threads(4);ids=np.r_[np.arange(32),1_048_576+np.arange(32)]
    a=cpu.sample_batch(ids,1515,64);b=tensor.sample_batch(ids,1515,64,device='cpu')
    for key in ('mask_a','mask_b','canonical_mask'):
        assert np.mean(a[key]!=b[key].numpy())<1e-5
    np.testing.assert_allclose(a['grid_a_to_b'],b['grid_a_to_b'].numpy(),atol=1e-6,rtol=0)
    differences=np.abs(a['view_a'].astype(int)-b['view_a'].numpy().astype(int))
    assert np.mean(differences>1)<1e-5
    features=torch.rand(64,3,64,64,requires_grad=True)
    sampled=torch.nn.functional.grid_sample(features,b['grid_a_to_b'],align_corners=False)
    sampled.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


@pytest.mark.parametrize('ids',[[],[-1],[2**31],[1.5],[True]])
def test_invalid_ids_rejected(ids):
    with pytest.raises(ValueError):cpu.sample_batch(ids,0,64)


@pytest.mark.parametrize('size',[0,31,257,64.5,True])
def test_invalid_sizes_rejected(size):
    with pytest.raises(ValueError):cpu.sample_batch([0],0,size)


@pytest.mark.parametrize('seed',[-1,2**32,True,1.5])
def test_invalid_seeds_rejected(seed):
    with pytest.raises(ValueError):cpu.sample_batch([0],seed,64)
    with pytest.raises(ValueError):tensor.sample_batch([0],seed,64)
