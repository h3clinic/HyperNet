import importlib.util
from pathlib import Path
import math
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/evaluate_grid_v15.py'
spec=importlib.util.spec_from_file_location('grid_eval_test',path);evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)

def test_threshold_is_strictmax_of_exact64_normal_calibration_scores():
    scores=np.arange(64,dtype=float)/100
    result=evaluation.operating_cutoff(scores)
    assert result['threshold']==math.nextafter(.63,math.inf)
    assert result['observed_calibration_false_positives']==0 and result['normal_calibration_images']==64
    assert result['calibration_zero_alarm_one_sided95_upper_per_10000']==pytest.approx(10000*(1-.05**(1/64)))
    with pytest.raises(ValueError):evaluation.operating_cutoff(scores[:21])
    with pytest.raises(ValueError):evaluation.operating_cutoff([float('nan')]*64)

def test_pixel_metrics_use_all_original_pixels_and_remove_only_exact_temporary_files(tmp_path):
    xy=np.array([[[.125,.125],[.625,.125]],[[.125,.625],[.625,.625]]],np.float32)
    class Dataset:
        records={'normal':{'width':4,'height':4},'anomaly':{'width':4,'height':4}}
        def load_evaluation_target(self,key):return {'mask':np.full((4,4),key=='anomaly',bool)}
    rows=[]
    for key,value in [('normal',0.),('anomaly',1.)]:
        record={'id':key,'role':'test','official_split':'test','is_anomaly':key=='anomaly'}
        prediction={'score':value,'patch_scores':np.full((2,2),value,np.float32),'xy':xy,'source_shape':[4,4]}
        rows.append(evaluation.save_prediction(tmp_path,'method',key,prediction,record))
    result=evaluation.pixel_metrics(tmp_path,'method',rows,Dataset())
    assert result['auroc']==1 and result['average_precision']==1
    assert result['positive_pixels']==16 and result['negative_pixels']==16
    assert all((tmp_path/row['map_path']).exists() for row in rows)
    assert not list((tmp_path/'metric_work').iterdir())

def test_map_save_has_no_target_mask_and_preserves_source_shape(tmp_path):
    prediction={'score':.5,'patch_scores':np.ones((2,2),np.float32),'xy':np.ones((2,2,2),np.float32),'source_shape':[4,4]}
    record={'id':'query','role':'test','official_split':'test','is_anomaly':True,'defect_type':'broken','mask':'secret'}
    row=evaluation.save_prediction(tmp_path,'method','query',prediction,record)
    assert 'mask' not in row and 'is_anomaly' not in row and row['source_shape']==[4,4]
    with np.load(tmp_path/row['map_path']) as saved:assert set(saved.files)=={'patch_scores','xy'}

def test_prototype_results_cannot_be_frozen(tmp_path):
    result=tmp_path/'grid_results.json'
    evaluation.write_new(result,{'completed':True,'stage':'dev','prototype':True})
    with pytest.raises(ValueError,match='final complete'):evaluation.freeze(result)

def test_final_freeze_rejects_missing_arms_in_each_required_mapping():
    good={'method_aliases':evaluation.ALIASES,'primary_method':'learned_final','query_images':64,
        **{key:{alias:0 for alias in evaluation.ALIASES} for key in ('score_files','calibration','thresholds')},
        **{key:{alias:0 for alias in ('learned_final','random')} for key in ('checkpoints','checkpoint_provenance')}}
    evaluation.validate_final_dev_rosters(good)
    for key in ('score_files','calibration','thresholds','checkpoints','checkpoint_provenance'):
        with pytest.raises(ValueError):evaluation.validate_final_dev_rosters({**good,key:{}})

def test_alternative_protocol_cannot_use_an_existing_test_admission(tmp_path):
    first=tmp_path/'first.json';second=tmp_path/'second.json';admission=tmp_path/'admission.json'
    evaluation.write_new(first,{'frozen':'one'});evaluation.write_new(second,{'frozen':'two'})
    evaluation.write_new(admission,{'protocol_sha256':evaluation.digest(first)})
    evaluation.verify_test_admission_matches(first,admission)
    with pytest.raises(ValueError,match='different CLI protocol'):evaluation.verify_test_admission_matches(second,admission)
