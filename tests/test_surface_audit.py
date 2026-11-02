from pathlib import Path
import importlib.util
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/audit_surface_encoder.py'
spec=importlib.util.spec_from_file_location('surface_auditor',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def records():
    return [{'family_index':f,'object_id':f'object_{f}_{o}','view_index':v,'surface_strength':s}
            for f in range(8) for o in range(8) for v in (0,1) for s in (0.,.03,.07)]


def test_identity_only_cannot_win_different_object_family_enrollment():
    rows=records();features=np.eye(64,dtype=np.float32).repeat(6,axis=0)
    metrics=audit.representation_metrics(features,rows)
    assert metrics['cross_view_instance_retrieval']==1.
    assert metrics['one_support_per_family_accuracy']==.125
    assert metrics['balanced_cross_view_surface_match']==.5
    assert metrics['family_decisions']==1344 and metrics['instance_decisions']==192
    assert metrics['surface_triplets']==128


def test_family_representation_and_tie_credit_are_separate_from_identity():
    rows=records();features=np.eye(8,dtype=np.float32)[[r['family_index'] for r in rows]]
    metrics=audit.representation_metrics(features,rows)
    assert metrics['one_support_per_family_accuracy']==1.
    assert metrics['cross_view_instance_retrieval']==.125
    assert metrics['balanced_cross_view_surface_match']==.5
    assert audit.fractional_accuracy(np.ones((4,8)),[0,1,2,3])==.125


def test_independent_fft_targets_match_float32_reference_within_roundoff():
    import torch
    from hypernet.surface_encoder import surface_spectrum
    rng=np.random.default_rng(2301)
    images=rng.integers(0,256,(2,128,128,3),dtype=np.uint8)
    mask=np.zeros((2,128,128),dtype=bool);mask[:,10:110,15:100]=True
    independent=audit.spectrum_targets(images,mask)
    reference=surface_spectrum(torch.from_numpy(images).permute(0,3,1,2).float()/255,
                               torch.from_numpy(mask).float()).numpy()
    np.testing.assert_allclose(independent,reference,rtol=1e-5,atol=1e-6)


def test_duplicate_object_view_strength_is_rejected():
    rows=records();rows[1]=rows[0].copy()
    with pytest.raises(ValueError,match='Duplicate object'):
        audit.representation_metrics(np.zeros((384,8)),rows)
