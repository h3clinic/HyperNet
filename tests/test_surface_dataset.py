from pathlib import Path
import importlib.util
from collections import Counter

import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/prepare_surface_data.py'
spec=importlib.util.spec_from_file_location('surface_builder',path)
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_complete_plan_has_disjoint_object_and_view_seeds_and_balanced_controls():
    plan=builder.make_plan();again=builder.make_plan()
    assert plan==again and len(plan['records'])==1920 and len(plan['objects'])==320
    by_split={s:{o['object_seed'] for o in plan['objects'] if o['split']==s} for s in ('train','heldout')}
    assert len(by_split['train'])==256 and len(by_split['heldout'])==64
    assert not by_split['train']&by_split['heldout']
    assert len({v for o in plan['objects'] for v in o['view_seeds']})==640
    counts=Counter((r['split'],r['family'],r['surface_strength']) for r in plan['records'])
    for family in plan['families']:
        for strength in (0.,.03,.07):
            assert counts['train',family,strength]==64
            assert counts['heldout',family,strength]==16
    for obj in plan['objects']:
        rows=[r for r in plan['records'] if r['object_id']==obj['object_id']]
        assert len(rows)==6 and len({r['object_seed'] for r in rows})==1
        assert all(len({r['view_seed'] for r in rows if r['view_index']==v})==1 for v in (0,1))


def fake_frame():
    mask=np.zeros((128,128),dtype=bool);mask[30:90,40:80]=True
    normal=np.zeros((128,128,3),dtype=np.float32);normal[mask,2]=1
    depth=np.zeros((128,128),dtype=np.float32);depth[mask]=3
    component=np.full((128,128),255,dtype=np.uint8);component[mask]=0
    return {'rgb':np.zeros((128,128,3),dtype=np.uint8),'mask':mask,'normal':normal,
            'depth':depth,'component':component}


def test_targets_enforce_foreground_depth_normal_and_background_conventions():
    frame=fake_frame();assert builder.validate_arrays(frame)['foreground_pixels']==2400
    frame['depth'][frame['mask']]=-1
    with pytest.raises(ValueError,match='foreground depth'):builder.validate_arrays(frame)
    frame=fake_frame();frame['normal'][0,0,0]=.1
    with pytest.raises(ValueError,match='background conventions'):builder.validate_arrays(frame)
    frame=fake_frame();frame['normal'][frame['mask'],2]=.5
    with pytest.raises(ValueError,match='unit and camera-facing'):builder.validate_arrays(frame)


def test_output_evidence_cannot_be_overwritten(tmp_path):
    path=tmp_path/'evidence.json';builder.write_new(path,{'value':1})
    with pytest.raises(FileExistsError):builder.write_new(path,{'value':2})
    assert builder.read(path)=={'value':1}


def test_array_hash_binds_dtype_shape_and_pixels():
    first=np.array([1,2,3,4],dtype=np.uint8)
    assert builder.array_sha(first)!=builder.array_sha(first.reshape(2,2))
    assert builder.array_sha(first)!=builder.array_sha(first.astype(np.int16))
    altered=first.copy();altered[0]=2
    assert builder.array_sha(first)!=builder.array_sha(altered)
