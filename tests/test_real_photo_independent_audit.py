import importlib.util
import math
from pathlib import Path
import pytest

source=Path(__file__).resolve().parents[1]/'scripts/audit_real_photo.py'
spec=importlib.util.spec_from_file_location('photo_audit',source)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_iou_matches_any_same_category_instance():
    assert audit.iou([20,20,30,30],[[0,0,5,5],[20,20,30,30]])==1
    assert audit.iou([0,0,10,10],[[5,0,15,10]])==pytest.approx(1/3)
    assert audit.iou(None,[[0,0,10,10]])==0


def test_rank_auc_and_localization_ap_handle_ties_independently():
    rows=[{'present':True,'score':.8,'iou':.7,'proposal_best_iou':.7},
          {'present':True,'score':.4,'iou':.1,'proposal_best_iou':.8},
          {'present':False,'score':.8,'iou':0,'proposal_best_iou':0},
          {'present':False,'score':.1,'iou':0,'proposal_best_iou':0}]
    result=audit.metric(rows,.5)
    assert result['auroc']==.625 and result['single_box_ap50']==.25
    assert result['localized_true_positive']==1 and result['false_positive']==1
    assert result['localized_recall_at_iou50']==.5 and result['proposal_recall_at_iou50']==1


def test_calibration_uses_only_dev_negatives_and_respects_ties():
    rows=[{'score':i/100,'split':'dev','present':False} for i in range(48)]
    rows += [{'score':100,'split':'test','present':False},{'score':200,'split':'dev','present':True}]
    assert audit.dev_cutoff(rows)==math.nextafter(.45,math.inf)
    tied=[{'score':0.,'split':'dev','present':False} for _ in range(48)]
    assert audit.dev_cutoff(tied)>0


def test_percentile_linear_interpolation_and_numeric_verification():
    assert audit.percentile([0,1,2,3],.25)==.75
    assert audit.percentile([4,4],.975)==4
    assert audit.compare_numeric({'a':[.5,1]},{'a':[.5,1]})==2
    with pytest.raises(AssertionError):audit.compare_numeric([1.],[1.001])
