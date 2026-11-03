import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from hypernet.gravity_spy_quality import classify_plot,is_constant_plot,RULE_ID
from hypernet.gravity_spy_images import GravitySpyCohort,digest
from test_gravity_spy_images import fixture


def test_exact_constant_only_no_low_contrast_threshold():
    blank=np.full((20,30,3),[68,1,84],dtype=np.uint8)
    assert not classify_plot(blank)['eligible']
    blank[5,5,1]+=1
    assert classify_plot(blank)['eligible']
    with pytest.raises(ValueError):is_constant_plot(blank.astype(np.float32))


def test_quality_amendment_filters_development_without_rewriting_roster(tmp_path):
    sha=fixture(tmp_path);source=Path(__file__).resolve().parents[1]/'hypernet/gravity_spy_quality.py'
    quality={'roster_sha256':sha,'rule':RULE_ID,'quality_source_sha256':digest(source),
        'eligible_target_labels':[],'excluded_event_ids':['ref','dev']}
    (tmp_path/'quality_admission.json').write_text(json.dumps(quality))
    cohort=GravitySpyCohort(tmp_path)
    assert cohort.target_labels==()
    assert cohort.raw_plan['target_labels']==['pattern']
    assert cohort.plan['reference_by_label']=={}
    assert cohort.query_events('calibration')==()
    assert digest(tmp_path/'plan.json')==sha
    with pytest.raises(ValueError,match='constant'):cohort.load_rgb('dev')


def test_quality_source_and_admission_bound_before_test(tmp_path):
    sha=fixture(tmp_path);source=Path(__file__).resolve().parents[1]/'hypernet/gravity_spy_quality.py'
    q={'roster_sha256':sha,'rule':RULE_ID,'quality_source_sha256':digest(source),
        'eligible_target_labels':['pattern'],'excluded_event_ids':[]}
    qp=tmp_path/'quality_admission.json';qp.write_text(json.dumps(q))
    a={'roster_sha256':sha,'calibration_complete':True,'thresholds':{'pattern':.5},
        'model_checkpoint_sha256':'a'*64,'inference_source_sha256':'b'*64,'calibration_scores_sha256':'c'*64}
    ap=tmp_path/'admission.json';ap.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='quality'):GravitySpyCohort(tmp_path,ap)
    a['quality_admission_sha256']=digest(qp);ap.write_text(json.dumps(a))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):GravitySpyCohort(tmp_path,ap)
