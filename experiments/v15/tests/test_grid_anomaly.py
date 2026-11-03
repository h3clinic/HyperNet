import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'hypernet/grid_anomaly.py'
spec=importlib.util.spec_from_file_location('grid_anomaly_test',path);grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)

def features(values):
    values=np.asarray(values,dtype=np.float32);gh,gw=values.shape[:2];h,w=2*gh,2*gw
    yy,xx=np.meshgrid(np.arange(gh)*2,np.arange(gw)*2,indexing='ij')
    return {'descriptors':values,'detail':values,'xy':np.stack([(xx+.5)/w,(yy+.5)/h],-1).astype(np.float32),
            'source_shape':[h,w],'source_pixels_resized':False,'sampling_stride':2}

def test_bank_selection_deterministic_and_uniform_without_linear_aliasing():
    a=grid.bank_indices(10000);b=grid.bank_indices(10000)
    assert len(a)==4096 and len(set(a))==4096 and np.array_equal(a,b)
    assert a.min()>=0 and a.max()<10000 and len(set(np.diff(a)))>1
    assert np.array_equal(grid.bank_indices(5),np.arange(5))

def test_exact_reference_is_normal_orthogonal_patch_is_anomalous_without_adaptation():
    reference=features(np.tile([1.,0.],(4,4,1)));bank=grid.ReferenceBank(reference)
    before=bank.provenance();normal=bank.score(reference)
    query=features(np.tile([1.,0.],(4,4,1)));query['descriptors'][1,2]=[0,1]
    anomaly=bank.score(query)
    assert normal['score']==0 and anomaly['score']==1 and anomaly['patch_scores'][1,2]==1
    assert bank.provenance()==before

def test_zero_analytic_features_match_constant_reference_but_not_nonzero_query():
    bank=grid.ReferenceBank(features(np.zeros((2,2,4),np.float32)),key='detail')
    assert bank.score(features(np.zeros((2,2,4),np.float32)))['score']==0
    assert bank.score(features(np.ones((2,2,4),np.float32)))['score']==1

def test_image_score_uses_fixed_upper_half_percent_mean():
    a=np.zeros(1000,np.float32);a[-5:]=[.1,.2,.3,.4,1.]
    assert grid.image_score(a)==pytest.approx(.4)

def test_pixel_map_uses_actual_native_sample_centers_without_mask_resize():
    f=features(np.zeros((2,2,2),np.float32));score=np.array([[0.,1.],[2.,3.]],np.float32)
    result=grid.original_pixel_map(score,f['xy'],f['source_shape'])
    assert result.shape==(4,4)
    assert np.array_equal(result[::2,::2],score)
    assert result[1,1]==1.5 and result[-1,-1]==3.

def test_invalid_coordinates_resized_features_and_mutated_bank_rejected():
    f=features(np.ones((2,2,2),np.float32))
    with pytest.raises(ValueError):grid.ReferenceBank({**f,'source_pixels_resized':True})
    bad={**f,'xy':f['xy']+.1}
    with pytest.raises(ValueError):grid.ReferenceBank(bad)
    bank=grid.ReferenceBank(f);bank.bank_np[0,0]=8
    with pytest.raises(RuntimeError):bank.provenance()
