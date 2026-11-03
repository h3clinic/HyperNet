from pathlib import Path
import sys
import numpy as np
import cv2
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypernet.scene_curriculum import (sample_program,program_id,program_split,hard_negative,
    render_instance,episode,batch,bbox_from_mask,FAMILIES,TRAIN_PAIRS,HELDOUT_PAIRS,contains_program)


def test_disjoint_program_partitions_and_reserved_compositions():
    seen={}
    for split in ('train','development','test','compositional'):
        programs=[sample_program(i,split) for i in range(30)]
        assert all(program_split(p)==split for p in programs)
        for p in programs:
            if p['family']=='composite':
                pair=tuple(q['family'] for q in p['parts'])
                assert pair in (HELDOUT_PAIRS if split=='compositional' else TRAIN_PAIRS)
        seen[split]={program_id(p) for p in programs}
    for a in seen:
        for b in seen:
            if a!=b: assert not seen[a]&seen[b]


def test_families_balanced_and_cached_program_immutable():
    programs=[sample_program(i) for i in range(60)]
    assert all(sum(p['family']==f for p in programs)==10 for f in FAMILIES)
    first=sample_program(0); first['family']='corrupted'
    assert sample_program(0)['family']!='corrupted'


def test_new_instances_change_geometry_without_changing_program():
    for family in FAMILIES:
        p=sample_program(42,family=family)
        a=render_instance(p,41,96); b=render_instance(p,42,96)
        assert a['program_id']==b['program_id']
        assert not np.array_equal(a['mask'],b['mask'])
        assert np.array_equal(a['mask']>0,a['parts']>0)
        negative,reason=hard_negative(p)
        assert reason and program_id(negative)!=program_id(p)
        assert program_split(negative)==program_split(p)


def test_explicit_holes_survive_nuisance_geometry():
    p={'family':'contour','spec':[5,3,1,3,0]}
    for seed in range(8):
        mask=render_instance(p,seed,96)['mask']
        # Three enclosed holes plus the outside are four background components.
        count,_=cv2.connectedComponents((1-mask).astype(np.uint8),8)
        assert count-1==4


def test_presence_does_not_change_scene_count_or_outside_slot_pixels():
    for index in range(18):
        yes=episode(index,positive_probability=1); no=episode(index,positive_probability=0)
        assert np.array_equal(yes['support'],no['support'])
        assert len(yes['metadata']['instances'])==len(no['metadata']['instances'])
        slots=[r for r in yes['metadata']['instances'] if r['role']=='target']
        x,y,w,h=slots[0]['placement_xywh']
        allowed=np.zeros((192,192),bool);allowed[y:y+h,x:x+w]=True
        assert np.array_equal(yes['query'][~allowed],no['query'][~allowed])
        assert np.any(yes['query'][allowed]!=no['query'][allowed])
        rgb_changed=np.max(np.abs(yes['query'].astype(np.int16)-no['query'].astype(np.int16)),axis=2)>=3
        assert rgb_changed.sum()>=8
        assert rgb_changed.sum()>=yes['metadata']['visible_counterfactual_rgb_difference_pixels_lower_bound']
        assert not no['target_mask'].any() and not no['query_parts'].any()
        assert all(r['program_id']!=no['program_id'] for r in no['metadata']['instances'])


def test_exact_visible_masks_boxes_and_resolution_guard():
    for i in range(24):
        r=episode(i,positive_probability=1)
        assert .65<=r['visibility']<=1
        assert np.array_equal(r['target_mask']>0,r['query_parts']>0)
        assert np.all(r['target_mask']<=r['amodal_target_mask'])
        assert np.array_equal(r['bbox'],bbox_from_mask(r['target_mask']))
        assert r['metadata']['hard_negative_changed_pixels']>=8
        assert r['metadata']['absence_replacement_changed_pixels']>=8
        assert r['metadata']['visible_hard_negative_detail_pixels']>=8
        assert r['metadata']['visible_absence_detail_pixels']>=8
        x1,y1,x2,y2=r['bbox'].astype(int)
        assert r['target_mask'][y1:y2,x1:x2].sum()==r['target_mask'].sum()


def test_reproducible_rgb_and_no_reference_copy():
    a=episode(829); b=episode(829)
    for key in ('support','query','support_mask','target_mask','bbox','query_parts'):
        assert np.array_equal(a[key],b[key])
    assert a['metadata']==b['metadata']
    assert a['metadata']['support_instance_seed']!=a['metadata']['query_instance_seed']
    assert a['metadata']['same_raster_copy'] is False


def test_subprogram_target_exclusion():
    part=sample_program(25,family='branch')
    composite={'family':'composite','layout':0,'parts':[part,sample_program(9,family='ridge')]}
    assert contains_program(composite,part)
    assert not contains_program(part,composite)


def test_torch_adapter_contract_and_absent_empty_masks():
    r=batch([0,1],positive_probability=0)
    assert tuple(r['support'].shape)==(2,3,96,96)
    assert tuple(r['query'].shape)==(2,3,192,192)
    assert tuple(r['target_mask'].shape)==(2,1,192,192)
    assert tuple(r['query_foreground_mask'].shape)==(2,1,192,192)
    assert (r['query_foreground_mask'].sum(dim=(1,2,3))>0).all()
    assert r['query'].min()>=0 and r['query'].max()<=1
    assert r['present'].sum()==0 and r['bbox'].sum()==0 and r['target_mask'].sum()==0


def test_query_foreground_union_keeps_absent_objects_and_occluders():
    for present_probability in (0,1):
        for index in range(12):
            row=episode(index,positive_probability=present_probability)
            expected=np.zeros((192,192),np.uint8)
            # Independently redraw each recorded program to reproduce the scene union.
            program=row['program'];negative,_=hard_negative(program)
            from hypernet.scene_curriculum import _second_negative
            replacement,_=_second_negative(program,negative)
            candidates={program_id(p):p for p in (program,negative,replacement)}
            for j in range(5):
                p=sample_program(index*13+j+104729);candidates[program_id(p)]=p
            for j in range(2):
                p=sample_program(index*17+j+982451653);candidates[program_id(p)]=p
            for record in row['metadata']['instances']:
                x,y,w,h=record['placement_xywh']
                mask=render_instance(candidates[record['program_id']],record['seed'],w)['mask']
                expected[y:y+h,x:x+w]|=mask
            assert expected.any() and np.array_equal(expected,row['query_foreground_mask'])
            assert np.all(row['target_mask']<=expected)


def test_support_letterbox_uses_crop_border_median_and_empty_label_padding():
    from hypernet.scene_curriculum import _frame_support
    rgb=np.full((96,96,3),[17,68,129],np.uint8);rgb[30:60,20:80]=[240,110,35]
    parts=np.zeros((96,96),np.int16);parts[31:59,21:79]=2
    framed,labels,meta=_frame_support(rgb,parts,np.random.default_rng(8),96)
    x1,y1,x2,y2=meta['crop_xyxy'];source=rgb[y1:y2,x1:x2]
    expected=np.median(np.concatenate([source[0],source[-1],source[:,0],source[:,-1]]),axis=0).astype(np.uint8)
    assert meta['pad_rgb']==expected.tolist()
    x,y,w,h=meta['letterbox_xywh'];padding=np.ones((96,96),bool);padding[y:y+h,x:x+w]=False
    assert np.all(framed[padding]==expected) and not labels[padding].any()


def test_invalid_requests_fail():
    with pytest.raises(ValueError): sample_program(0,'unknown')
    with pytest.raises(ValueError): episode(0,query_size=32)
    with pytest.raises(ValueError): episode(0,positive_probability=2)


def test_three_ridge_spacing_aliases_are_excluded_from_generated_programs_and_edits():
    from hypernet.scene_curriculum import _single_edits
    for index in range(80):
        program=sample_program(index,family='ridge')
        for p in [program]+[p for p,_ in _single_edits(program)]:
            n,_,spacing,_=p['spec']
            assert n!=2 or spacing==0
            assert n!=3 or spacing!=1
    for index in (498,534):
        row=episode(index)
        assert row['metadata']['visible_counterfactual_rgb_difference_pixels_lower_bound']>=8


def test_all_split_nuisance_seed_streams_differ_at_same_index():
    rows=[episode(5,split=split) for split in ('train','development','test','compositional')]
    assert len({r['metadata']['support_instance_seed'] for r in rows})==4
    assert len({r['metadata']['query_instance_seed'] for r in rows})==4


def test_tiny_ridge_strokes_survive_final_pixel_centre_sampling():
    for seed in range(24):
        row=render_instance({'family':'ridge','spec':[2,0,0,1]},seed,24)
        assert row['mask'].any() and (row['parts']>0).any()
    for index in (3468,14604,18738,26988):
        row=episode(index)
        assert row['metadata']['visible_counterfactual_rgb_difference_pixels_lower_bound']>=8


def test_ambiguous_background_rejection_keeps_type_and_presence_counterfactuals_matched():
    for index in (782,3350,3530,9699,13795):
        yes=episode(index,positive_probability=1);no=episode(index,positive_probability=0)
        assert yes['program_id']==program_id(sample_program(index))
        assert yes['program_id']==no['program_id']
        assert yes['metadata']['ambiguous_nuisance_draws_rejected']==no['metadata']['ambiguous_nuisance_draws_rejected']
        assert np.array_equal(yes['support'],no['support'])
        difference=np.max(np.abs(yes['query'].astype(np.int16)-no['query'].astype(np.int16)),axis=2)>=3
        assert difference.sum()>=8


def test_texture_contrast_survives_dark_and_bright_foreground():
    from hypernet.scene_curriculum import _paint
    instance=render_instance({'family':'texture','spec':[0,6,0,2]},42,96)
    for foreground in (.02,.98):
        canvas=np.full((96,96,3),.5,np.float32)
        _paint(canvas,instance,0,0,np.full(3,foreground,np.float32))
        core=instance['alpha']>.99
        assert np.ptp(canvas[core,0])>.20


def test_hidden_composite_edit_retries_nuisance_without_changing_program():
    row=episode(451,split='compositional',seed=1777)
    assert row['program_id']==program_id(sample_program(451,split='compositional',seed=1777))
    assert row['metadata']['ambiguous_nuisance_draws_rejected']==2
    assert len(row['metadata']['ambiguous_nuisance_rejection_reasons'])==2


def test_larger_query_scene_keeps_pixel_coordinates_and_label_shapes():
    row=episode(3,query_size=256,positive_probability=1)
    assert row['support'].shape==(96,96,3) and row['query'].shape==(256,256,3)
    assert row['query_foreground_mask'].shape==row['target_mask'].shape==(256,256)
    assert np.array_equal(row['bbox'],bbox_from_mask(row['target_mask']))
    assert row['bbox'].min()>=0 and row['bbox'].max()<=256
