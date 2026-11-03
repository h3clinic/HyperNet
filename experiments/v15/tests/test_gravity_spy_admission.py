from pathlib import Path
import importlib.util,json
import pytest

path=Path(__file__).resolve().parents[1]/'hypernet/gravity_spy_admission.py'
spec=importlib.util.spec_from_file_location('gs_admission_test',path);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')

@pytest.fixture
def frozen(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    labels=[f'pattern{i}' for i in range(19)];data=tmp_path/'data';data.mkdir()
    write(data/'plan.json',{'target_labels':labels,'events':[{'event_id':'fresh','split':'test','group_id':'new'},
        {'event_id':'old','split':'test','group_id':'used'}]});roster=gate.digest(data/'plan.json')
    write(data/'roster_freeze.json',{'roster_sha256':roster})
    sources={}
    for name in ('gravity_spy_admission.py','gravity_spy_images.py','gravity_spy_quality.py'):
        target=tmp_path/'hypernet'/name;target.parent.mkdir(exist_ok=True);target.write_text('frozen source')
        sources['hypernet/'+name]=gate.digest(target)
    quality=dict(roster_sha256=roster,model_scores_used=False,replacement_sampling=False,
        quality_source_sha256=sources['hypernet/gravity_spy_quality.py'],eligible_target_labels=labels)
    write(data/'quality_admission.json',quality);quality_sha=gate.digest(data/'quality_admission.json')
    prior=tmp_path/'prior.json';write(prior,{'event':'old'})
    exposure=dict(roster_sha256=roster,quality_admission_sha256=quality_sha,scoring_policy='fresh_only',
        test_ids_excluded_before_any_v15_test_inference=True,no_replacement_sampling=True,
        secondary_reused_test_download_or_scoring_authorized=False,prior_source_artifacts={'prior.json':gate.digest(prior)},
        excluded_group_ids=['used'],primary_test_event_ids=['fresh'])
    write(data/'prior_exposure_admission.json',exposure);exposure_sha=gate.digest(data/'prior_exposure_admission.json')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'own weights')
    scores=tmp_path/'scores.jsonl';scores.write_text('{"fixed":"normal calibration"}\n')
    sources['scores.jsonl']=gate.digest(scores);thresholds={label:.5 for label in labels}
    protocol=dict(model_selection_complete=True,test_scores_computed_before_freeze=False,roster_sha256=roster,
        quality_admission_sha256=quality_sha,prior_exposure_admission_sha256=exposure_sha,thresholds=thresholds,behavior_files=sources,
        checkpoint_files={'model.pt':gate.digest(checkpoint)},model_checkpoint_path='model.pt',
        calibration_scores_path='scores.jsonl',calibration_scores_sha256=gate.digest(scores))
    write(tmp_path/'protocol.json',protocol)
    admission=dict(protocol_path='protocol.json',protocol_sha256=gate.digest(tmp_path/'protocol.json'),
        calibration_complete=True,roster_sha256=roster,quality_admission_sha256=quality_sha,prior_exposure_admission_sha256=exposure_sha,thresholds=thresholds,
        inference_source_sha256=gate.mapping_digest(sources),model_checkpoint_sha256=gate.digest(checkpoint),calibration_scores_sha256=gate.digest(scores))
    admission_path=tmp_path/'admission.json';write(admission_path,admission)
    return tmp_path,data,admission_path

def test_actual_protocol_sources_checkpoint_and_calibration_verified(frozen):
    root,data,path=frozen;result=gate.verify_test_admission(data,path)
    assert result['verified_actual_artifacts'] and result['verified_behavior_files']==4 and result['verified_checkpoint_files']==1

@pytest.mark.parametrize('name',['protocol.json','model.pt','scores.jsonl','hypernet/gravity_spy_images.py'])
def test_unchanged_admission_cannot_authorize_mutated_actual_artifacts(frozen,name):
    root,data,path=frozen;(root/name).write_bytes(b'changed')
    with pytest.raises((ValueError,json.JSONDecodeError)):gate.verify_test_admission(data,path)

def test_well_formed_dummy_hashes_no_longer_unseal_test(frozen):
    root,data,path=frozen;a=gate.read(path);a['model_checkpoint_sha256']='a'*64;write(path,a)
    with pytest.raises(ValueError,match='checkpoint'):gate.verify_test_admission(data,path)

def test_modified_cutoffs_quality_or_missing_reference_target_rejected(frozen):
    root,data,path=frozen;a=gate.read(path);a['thresholds'].pop('pattern0');write(path,a)
    with pytest.raises(ValueError,match='19 frozen'):gate.verify_test_admission(data,path)

def test_relative_artifact_paths_cannot_escape_root(frozen):
    root,data,path=frozen
    with pytest.raises(ValueError,match='escapes'):gate.resolve('../outside')

def test_prior_exposure_mask_and_its_actual_sources_are_bound(frozen):
    root,data,path=frozen;(root/'prior.json').write_text('changed')
    with pytest.raises(ValueError,match='prior exposure source'):gate.verify_test_admission(data,path)
