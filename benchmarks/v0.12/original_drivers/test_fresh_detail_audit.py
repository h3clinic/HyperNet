import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/audit_fresh_detail_v12.py'
spec=importlib.util.spec_from_file_location('fresh_independent_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def constant_family_fixture():
    values={'old_cnn':0.,'outline_only':.5,'centroid_only':1.,'local_moments':.5,'lowres_patch':.5,'native_patch':1.}
    rows={};cutoffs={alias:.5 for alias in values}
    for alias,value in values.items():
        rows[alias]=[{'family_id':family,'split':'test','present':True,'score':.9 if enrolled<int(2*value) else .1,'iou':1.}
                     for family in range(128) for enrolled in (0,1)]
    reported={'unit':'family','families':128,'samples':10000,'seed':1812,'metric':'localized_recall_at_iou50',
              'methods':{alias:{'estimate':value,'ci95':[value,value]} for alias,value in values.items()},
              'differences':{'native_patch_minus_old_cnn':{'estimate':1.,'ci95':[1.,1.]},
                             'native_patch_minus_lowres_patch':{'estimate':.5,'ci95':[.5,.5]}}}
    return rows,cutoffs,reported


def test_manual_percentiles_use_linear_order_statistic_interpolation():
    assert audit.bounds(np.array([30.,0.,20.,10.]))==pytest.approx([.75,29.25])


def test_family_cluster_bootstrap_preserves_known_constant_paired_effects():
    rows,cutoffs,reported=constant_family_fixture()
    result=audit.verify_bootstrap(rows,cutoffs,reported)
    assert result['verified_values']==24 and result['families']==128
    assert result['differences']['native_patch_minus_old_cnn']['ci95']==[1.,1.]


def test_wrong_cluster_or_randomization_protocol_is_rejected():
    rows,cutoffs,reported=constant_family_fixture();reported['seed']=99
    with pytest.raises(ValueError,match='cluster roster/protocol'):
        audit.verify_bootstrap(rows,cutoffs,reported)


def test_wrong_paired_interval_is_rejected():
    rows,cutoffs,reported=constant_family_fixture()
    reported['differences']['native_patch_minus_lowres_patch']['ci95']=[.49,.5]
    with pytest.raises(ValueError,match='Paired interval'):
        audit.verify_bootstrap(rows,cutoffs,reported)
