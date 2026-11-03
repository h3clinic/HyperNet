"""Verify actual frozen artifacts before Gravity Spy TEST pixel acquisition."""
from hashlib import sha256
from pathlib import Path
import json,math

ROOT=Path(__file__).resolve().parents[1]

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def mapping_digest(mapping):return sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def resolve(path):
    path=Path(path)
    if path.is_absolute():return path.resolve()
    result=(ROOT/path).resolve()
    if not result.is_relative_to(ROOT.resolve()):raise ValueError('Relative artifact path escapes v15 root')
    return result
def require(value,message):
    if not value:raise ValueError(message)
def file_bindings(mapping,kind):
    require(isinstance(mapping,dict) and mapping,'Missing frozen '+kind+' files')
    result={}
    for name,expected in mapping.items():
        path=resolve(name)
        require(path.is_file() and digest(path)==expected,'Actual frozen '+kind+' file differs: '+name)
        require(path not in result,'Duplicate normalized frozen path')
        result[path]=expected
    return result

def verify_test_admission(data_root,admission_path):
    data=Path(data_root).resolve();admission=read(admission_path)
    plan_path=data/'plan.json';plan=read(plan_path);roster_sha=digest(plan_path)
    require(read(data/'roster_freeze.json')['roster_sha256']==roster_sha,'Roster freeze changed')
    quality_path=data/'quality_admission.json';quality=read(quality_path);quality_sha=digest(quality_path)
    require(quality['roster_sha256']==roster_sha and quality.get('model_scores_used') is False and quality.get('replacement_sampling') is False,'Invalid frozen quality policy')
    require(quality['quality_source_sha256']==digest(ROOT/'hypernet/gravity_spy_quality.py'),'Quality rule source changed')
    labels=quality['eligible_target_labels']
    require(len(labels)==len(set(labels))==19 and set(labels).issubset(plan['target_labels']),'Expected19 eligible frozen targets')
    require(admission.get('calibration_complete') is True,'Calibration is incomplete')
    require(admission.get('roster_sha256')==roster_sha and admission.get('quality_admission_sha256')==quality_sha,'Admission roster/quality binding differs')
    exposure_path=data/'prior_exposure_admission.json';exposure=read(exposure_path);exposure_sha=digest(exposure_path)
    require(exposure.get('roster_sha256')==roster_sha and exposure.get('quality_admission_sha256')==quality_sha,'Exposure mask roster/quality binding differs')
    require(exposure.get('scoring_policy')=='fresh_only' and exposure.get('test_ids_excluded_before_any_v15_test_inference') is True,'TEST freshness exclusion was not frozen')
    require(exposure.get('no_replacement_sampling') is True and exposure.get('secondary_reused_test_download_or_scoring_authorized') is False,'Unexpected exposure sampling policy')
    file_bindings(exposure.get('prior_source_artifacts'),'prior exposure source')
    excluded_groups=set(exposure['excluded_group_ids'])
    expected_fresh=sorted(row['event_id'] for row in plan['events'] if row['split']=='test' and row['group_id'] not in excluded_groups)
    require(expected_fresh and exposure['primary_test_event_ids']==expected_fresh,'Fresh TEST IDs differ from whole-group exclusion')
    require(admission.get('prior_exposure_admission_sha256')==exposure_sha,'Admission omits/changes exposure amendment')
    protocol_path=resolve(admission['protocol_path'])
    require(protocol_path.is_file() and digest(protocol_path)==admission['protocol_sha256'],'Actual frozen protocol differs')
    protocol=read(protocol_path)
    require(protocol.get('model_selection_complete') is True and protocol.get('test_scores_computed_before_freeze') is False,'Protocol was not frozen beforeTEST')
    require(protocol.get('roster_sha256')==roster_sha and protocol.get('quality_admission_sha256')==quality_sha,'Protocol roster/quality binding differs')
    require(protocol.get('prior_exposure_admission_sha256')==exposure_sha,'Protocol omits/changes exposure amendment')
    thresholds=admission.get('thresholds')
    require(isinstance(thresholds,dict) and set(thresholds)==set(labels) and thresholds==protocol.get('thresholds'),'Exactly19 frozen decision thresholds required')
    require(all(type(value) in (int,float) and math.isfinite(value) for value in thresholds.values()),'Thresholds must be finite numbers')
    behavior=file_bindings(protocol.get('behavior_files'),'behavior')
    checkpoints=file_bindings(protocol.get('checkpoint_files'),'checkpoint')
    # At minimum the gate itself, pixel loader, and source-quality rule must be
    # frozen. The protocol lists the actual scorer/extractor/evaluator as well.
    for name in ('hypernet/gravity_spy_admission.py','hypernet/gravity_spy_images.py','hypernet/gravity_spy_quality.py'):
        require(resolve(name) in behavior,'Protocol omits required inference boundary: '+name)
    inference_sha=mapping_digest(protocol['behavior_files'])
    require(admission.get('inference_source_sha256')==inference_sha,'Inference binding digest differs from actual verified sources')
    checkpoint_path=resolve(protocol['model_checkpoint_path'])
    require(checkpoint_path in checkpoints and admission.get('model_checkpoint_sha256')==checkpoints[checkpoint_path],'Primary model checkpoint is not bound')
    scores_path=resolve(protocol['calibration_scores_path'])
    require(scores_path.is_file() and scores_path.stat().st_size>0,'Actual calibration score artifact is absent/empty')
    scores_sha=digest(scores_path)
    require(scores_sha==protocol['calibration_scores_sha256']==admission.get('calibration_scores_sha256'),'Actual calibration scores differ')
    require(scores_path in behavior and behavior[scores_path]==scores_sha,'Calibration scores not included in frozen artifact set')
    return dict(admission,verified_actual_artifacts=True,verified_behavior_files=len(behavior),
                verified_checkpoint_files=len(checkpoints),verified_protocol_path=str(protocol_path))
