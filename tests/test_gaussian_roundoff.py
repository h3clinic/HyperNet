import pytest
import torch
from hypernet.gaussian_variation import gaussian_variation

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('kind',['none','shape','shape_pattern'])
def test_small_interpolation_roundoff_matches_explicit_clipping(dtype,kind):
    torch.set_num_threads(1)
    eps=torch.finfo(dtype).eps
    patches=torch.full((2,3,32,32),.5,dtype=dtype)
    patches[0,0,0,0]=-2*eps;patches[1,2,31,31]=1+2*eps
    original=patches.clone()
    ga=torch.Generator().manual_seed(73);gb=torch.Generator().manual_seed(73)
    actual=gaussian_variation(patches,ga,kind=kind)
    expected=gaussian_variation(patches.clamp(0,1),gb,kind=kind)
    assert torch.equal(actual,expected)
    assert torch.equal(ga.get_state(),gb.get_state())
    assert torch.equal(patches,original)
    assert actual.min()>=0 and actual.max()<=1

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('value',['lower','upper','nan','inf','negative_inf'])
def test_real_input_errors_still_rejected(dtype,value):
    eps=torch.finfo(dtype).eps
    values={'lower':-8*eps,'upper':1+8*eps,'nan':float('nan'),'inf':float('inf'),'negative_inf':-float('inf')}
    patches=torch.full((1,3,32,32),.5,dtype=dtype);patches[0,0,0,0]=values[value]
    with pytest.raises(ValueError,match='finite RGB'):
        gaussian_variation(patches,torch.Generator().manual_seed(1))

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_valid_rgb_identity_and_rng_unchanged(dtype):
    patches=torch.rand((2,3,32,32),dtype=dtype,generator=torch.Generator().manual_seed(17))
    patches[0,0,0,0]=0.;patches[1,2,31,31]=1.
    generator=torch.Generator().manual_seed(5);state=generator.get_state().clone()
    assert torch.equal(gaussian_variation(patches,generator,kind='none'),patches)
    assert torch.equal(generator.get_state(),state)
