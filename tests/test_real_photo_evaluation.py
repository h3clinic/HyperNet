"""Independent runner/metric controls using synthetic arrays and mocked inference."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from hypernet.real_photo_metrics import max_box_iou,cutoff_from_real_dev,summarize,category_bootstrap
from hypernet.real_photo import ALIASES,TRANSFERRED_CUTOFFS

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('real_photo_runner_tests',ROOT/'scripts/evaluate_real_photo.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def row(identity,present,score,iou=1.,category=1,split='test'):
    return dict(id=str(identity),category_id=category,present=present,score=score,iou=iou,
                proposal_best_iou=1. if present else 0.,split=split)

def test_iou_uses_best_of_all_instances_and_empty_predictions():
    assert max_box_iou([10,10,20,20],[[0,0,3,3],[10,10,20,20]])==1.
    assert max_box_iou(None,[[0,0,3,3]])==0.
    assert max_box_iou([0,0,3,3],[])==0.
    assert max_box_iou([0,0,10,10],[[5,0,15,10]])==pytest.approx(1/3)
    with pytest.raises(ValueError):max_box_iou([0,0,0,3],[[0,0,3,3]])

def test_calibration_is_dev_only_and_strict_ties_allow_two_of48():
    records=[row(i,False,1-i/100,split='dev') for i in range(48)]
    cutoff=cutoff_from_real_dev(records)
    assert sum(r['score']>=cutoff for r in records)==2
    records[0]['score']=records[1]['score']=records[2]['score']=.99
    assert sum(r['score']>=cutoff_from_real_dev(records) for r in records)==0
    with pytest.raises(ValueError):cutoff_from_real_dev(records+[row('test',False,1)])
    with pytest.raises(ValueError):cutoff_from_real_dev([row('none',True,1,split='dev')])

def test_metrics_distinguish_presence_localization_and_tied_ap():
    records=[row('p1',True,.9),row('p2',True,.8,.2),row('n1',False,.8,0),row('n2',False,.1,0)]
    metric=summarize(records,.85)
    assert metric['localized_true_positive']==1 and metric['false_positive']==0
    assert metric['recall']==.5 and metric['localized_recall_at_iou50']==.5
    assert metric['top1_iou_recall_at_iou50']==.5 and metric['proposal_recall_at_iou50']==1
    assert metric['balanced_accuracy']==.75 and metric['auroc']==.875
    assert metric['single_box_ap50']==.5
    assert summarize([row('p',True,.4),row('n',False,.4,0)],.4)['single_box_ap50']==.5
    assert summarize([row('p',True,.4),row('n',False,.4,0)],.4)['auroc']==.5
    with pytest.raises(ValueError):summarize(records+[records[0]],.5)
    json.dumps(summarize([row('p',True,np.float32(.9)),row('n',False,np.float32(.1),0)],.5),allow_nan=False)

def test_bootstrap_clusters_categories_with_shared_draws_and_no_refit():
    records=[row('a+',True,.9,category=1),row('a-',False,.1,category=1),
        row('b1+',True,.9,category=2),row('b2+',True,.9,category=2),row('b-',False,.9,category=2)]
    arms={'native_patch':records,'old_cnn':copy.deepcopy(records)}
    result=category_bootstrap(arms,dict(native_patch=.5,old_cnn=.5),samples=100,seed=7)
    assert result['categories']==2 and result['unit']=='category'
    assert result['methods']['native_patch']['localized_recall_at_iou50']['estimate']==1.
    assert result['methods']['native_patch']['false_positive_rate']['estimate']==.5
    assert result['localized_recall_differences']['native_patch_minus_old_cnn']['ci95']==[0.,0.]
    assert result==category_bootstrap(arms,dict(native_patch=.5,old_cnn=.5),samples=100,seed=7)
    arms['old_cnn'][0]['present']=False
    with pytest.raises(ValueError):category_bootstrap(arms,dict(native_patch=.5,old_cnn=.5))

def test_load_rgb_checks_raw_pixels_dimensions_and_containment(tmp_path):
    rgb=np.arange(16*20*3,dtype=np.uint8).reshape(16,20,3);path=tmp_path/'rgb.png';Image.fromarray(rgb).save(path)
    digest=hashlib.sha256(str(rgb.dtype).encode()+str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    record=dict(rgb_path='rgb.png',sha256=runner.sha(path),pixel_sha256=digest,height=16,width=20)
    assert np.array_equal(runner.load_rgb(tmp_path,record),rgb)
    for altered in ({**record,'pixel_sha256':'bad'},{**record,'sha256':'bad'},{**record,'width':21},
                    {**record,'rgb_path':'../outside.png'}):
        with pytest.raises(ValueError):runner.load_rgb(tmp_path,altered)

def prediction():
    return dict(score=.7,bbox_xyxy=[1,1,9,9],proposals=[[1,1,9,9]],proposal_scores=[.7],evidence={},diagnostics={})

def test_prediction_contract_rejects_bad_coordinate_space_or_selection():
    runner.validate_prediction(prediction(),10,10)
    for changes in (dict(bbox_xyxy=[0,0,9,9]),dict(score=.8),dict(proposals=[[1,1,11,9]]),
                    dict(proposal_scores=[float('nan')]),dict(proposal_scores=[])):
        with pytest.raises(ValueError):runner.validate_prediction({**prediction(),**changes},10,10)
    runner.validate_prediction(dict(score=0.,bbox_xyxy=None,proposals=[],proposal_scores=[],evidence={},diagnostics={}),10,10)

def fake_manifest():
    refs=[];queries=[]
    for category in range(12):
        refs.append(dict(category_id=category,image_id=category,name=str(category),rgb_path=f'r{category}',bbox_xyxy=[1,1,9,9]))
        for split,count in [('dev',4),('test',8)]:
            for present in [False,True]:
                for i in range(count):
                    identity=f'{category}-{split}-{present}-{i}'
                    queries.append(dict(id=identity,category_id=category,name=str(category),split=split,present=present,
                        image_id=100+len(queries),rgb_path=identity,width=10,height=10,
                        gt_boxes_xyxy=[[0,0,1,1],[1,1,9,9]] if present else []))
    return dict(schema_version='hypernet-real-photo-v13-v1',references=refs,queries=queries)

def test_runner_enrolls_once_and_queries_without_labels_then_uses_all_gt(monkeypatch):
    manifest=fake_manifest();predicted=set();active={};calls=[]
    class Guard(dict):
        def __getitem__(self,key):
            if key in ('present','gt_boxes_xyxy') and dict.__getitem__(self,'id') not in predicted:
                raise AssertionError('Query annotations read before inference')
            return super().__getitem__(key)
    manifest['queries']=[Guard(q) for q in manifest['queries']]
    def load(data,record):
        active['record']=record
        return np.zeros((10,10,3),np.uint8)
    class Engine:
        def enroll(self,rgb,box):calls.append(box);return {'pixels_only':True}
        def predict(self,enrolled,rgb):
            assert enrolled=={'pixels_only':True} and isinstance(rgb,np.ndarray)
            predicted.add(active['record']['id'])
            return {alias:prediction() for alias in ALIASES}
    monkeypatch.setattr(runner,'load_rgb',load)
    result=runner.evaluate_queries(Engine(),manifest,Path('.'))
    assert len(calls)==12 and len(predicted)==288 and len(result)==4
    for alias,records in result.items():
        assert len(records)==288 and all(r['iou']==int(r['present']) for r in records)
        assert all(r['method']==alias for r in records)

def test_modes_keep_transferred_cutoffs_and_fit_only_dev(monkeypatch):
    records=[]
    for split in ['dev','test']:
        for category in range(12):
            for present in [False,True]:
                records.extend(row(f'{split}-{category}-{present}-{i}',present,.8 if present else .2,
                                   category=category,split=split) for i in range(4 if split=='dev' else 8))
    arms={alias:copy.deepcopy(records) for alias in ALIASES}
    monkeypatch.setattr(runner,'category_bootstrap',lambda *args,**kwargs:{'test':'stub; function tested separately'})
    result=runner.evaluate_modes(arms)
    assert result['transferred']['thresholds']==TRANSFERRED_CUTOFFS
    assert result['real_dev_calibrated']['thresholds']==dict.fromkeys(ALIASES,np.nextafter(.2,np.inf))
    for rows in arms.values():
        for r in rows:
            if r['split']=='test':r['score']=.999
    assert runner.evaluate_modes(arms)['real_dev_calibrated']['thresholds']==result['real_dev_calibrated']['thresholds']

@pytest.fixture
def protocol_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path);(tmp_path/'experiments').mkdir();(tmp_path/'scripts').mkdir();(tmp_path/'data').mkdir()
    def write(relative,value):
        path=tmp_path/relative;runner.write(path,value);return runner.sha(path)
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'not loaded by protocol test')
    script=tmp_path/'scripts/recover_real_photo_transport.py';script.write_text('# transport fixture')
    freeze=write(runner.DATA_BINDINGS['data_freeze_sha256'],{'frozen':True})
    fetch=write(runner.DATA_BINDINGS['fetch_admission_sha256'],{'start':True})
    recovery=write(runner.DATA_BINDINGS['transport_recovery_admission_sha256'],dict(data_freeze_sha256=freeze,
        original_fetch_admission_sha256=fetch,recovery_script_sha256=runner.sha(script),
        selection_annotations_and_criteria_unchanged=True,model_runs=0,original_dataset_images_written=0))
    manifest=fake_manifest();manifest['transport_recovery_admission_sha256']=recovery
    manifest_hash=write('data/manifest.json',manifest)
    audit=write(runner.DATA_BINDINGS['data_audit_sha256'],dict(valid=True,manifest_sha256=manifest_hash,
        data_freeze_sha256=freeze,fetch_admission_sha256=fetch))
    protocol=dict(model_selection_complete=True,photo_scores_computed_before_freeze=False,
        primary_mode='transferred',primary_method='native_patch',transferred_thresholds=dict(TRANSFERRED_CUTOFFS),
        behavior_files={'scripts/recover_real_photo_transport.py':runner.sha(script)},data_manifest_sha256=manifest_hash,
        data_audit_sha256=audit,data_freeze_sha256=freeze,fetch_admission_sha256=fetch,
        transport_recovery_admission_sha256=recovery,checkpoint_sha256=runner.sha(checkpoint))
    write('protocol.json',protocol)
    return tmp_path/'protocol.json',tmp_path/'data',checkpoint,protocol

def test_protocol_binds_actual_schema_audit_recovery_and_checkpoint(protocol_fixture):
    path,data,checkpoint,protocol=protocol_fixture
    _,manifest,tracked=runner.verify_protocol(path,data,checkpoint)
    assert manifest['schema_version']=='hypernet-real-photo-v13-v1' and 'completed' not in manifest
    assert checkpoint.resolve() in tracked and (data/'manifest.json').resolve() in tracked
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError):runner.verify_protocol(path,data,checkpoint)

def test_protocol_rejects_unfrozen_recovery_script_and_changed_cutoffs(protocol_fixture):
    path,data,checkpoint,protocol=protocol_fixture
    protocol['behavior_files']={};runner.write(path,protocol)
    with pytest.raises(ValueError,match='Recovery script'):runner.verify_protocol(path,data,checkpoint)
    protocol['transferred_thresholds']['native_patch']=.7;runner.write(path,protocol)
    with pytest.raises(ValueError,match='thresholds changed'):runner.verify_protocol(path,data,checkpoint)
