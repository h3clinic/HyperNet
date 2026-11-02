"""Independent engine/provenance controls, using synthetic pixels only."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from hypernet.detail_pipeline import DetailPipeline,METHODS,state_hash,tensor
from hypernet.one_shot_detector import OneReferenceDetector
from hypernet.detail_metrics import cutoff_from_dev,summarize,bootstrap

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reviewed_detail_runner',ROOT/'scripts/evaluate_detail.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def example(size=96):
    rgb=np.full((size,size,3),20,np.uint8)
    rgb[16:80,16:80]=160
    rgb[33:38,34:43]=20;rgb[53:62,56:61]=20
    return rgb

@pytest.fixture(scope='module')
def engine():
    torch.set_num_threads(2);torch.manual_seed(147)
    item=DetailPipeline.__new__(DetailPipeline)
    item.model=OneReferenceDetector().eval().requires_grad_(False)
    return item

def test_tensor_rejects_wrong_channel_or_dtype():
    for wrong in (np.zeros((96,96,3),np.float32),np.zeros((96,96),np.uint8),np.zeros((3,96,96),np.uint8)):
        with pytest.raises(ValueError):tensor(wrong)

def test_all_arms_share_proposals_and_native_alignment(engine,monkeypatch):
    import hypernet.detail_pipeline as module
    boxes=[[16,16,80,80],[140,144,204,208]]
    monkeypatch.setattr(module,'proposals_from_foreground',lambda *a,**k:boxes)
    support=example();query=np.full((256,256,3),20,np.uint8)
    query[:96,:96]=support;query[128:224,124:220]=support
    before=state_hash(engine.model)
    reference=engine.prepare_reference(support,[16,16,80,80])
    prepared=engine.prepare_query(query)
    result=engine.score(reference,prepared)
    assert tuple(result)==METHODS
    for method in METHODS:
        assert result[method]['proposals']==boxes
        assert len(result[method]['proposal_scores'])==len(boxes)
        assert np.isfinite(result[method]['score'])
    for fine,coarse in zip(result['native_patch']['all_evidence'],result['lowres_patch']['all_evidence']):
        assert fine['outline']==coarse['outline'] and fine.get('angle')==coarse.get('angle')
        assert fine['patch_count']==coarse['patch_count']
    assert state_hash(engine.model)==before
    assert all(p.grad is None and not p.requires_grad for p in engine.model.parameters())

def test_empty_proposals_preserve_rejection(engine,monkeypatch):
    import hypernet.detail_pipeline as module
    monkeypatch.setattr(module,'proposals_from_foreground',lambda *a,**k:[])
    result=engine.detect(example(),[16,16,80,80],example(256))
    for method,row in result.items():
        assert row['proposals']==row['proposal_scores']==[]
        assert row['bbox_xyxy'] is None
        assert row['score']==(-1 if method=='old_cnn' else 0)

def test_query_annotations_are_not_read(engine,monkeypatch):
    import hypernet.detail_pipeline as module
    monkeypatch.setattr(module,'proposals_from_foreground',lambda *a,**k:[[16,16,80,80]])
    reference=engine.prepare_reference(example(),[16,16,80,80])
    prepared=engine.prepare_query(example(256))
    class Guard(dict):
        def __getitem__(self,key):
            assert key not in ('family_id','target_bbox_xyxy','target_mask','present')
            return super().__getitem__(key)
    guarded=Guard(prepared)
    guarded.update(family_id=999,target_bbox_xyxy=[1,1,2,2],target_mask='forbidden',present=True)
    assert engine.score(reference,guarded)==engine.score(reference,prepared)

@pytest.mark.parametrize('values',[[.2]*256,[.9]*13+[.1]*243,np.linspace(-1,1,256).tolist()])
def test_dev_threshold_enforces_budget_with_ties(values):
    rows=[dict(split='dev',score=float(v),present=False) for v in values]
    threshold=cutoff_from_dev(rows)
    assert sum(v>=threshold for v in values)<=12
    with pytest.raises(ValueError):cutoff_from_dev([dict(rows[0],split='test')])

def test_tied_metric_and_family_bootstrap_controls():
    rows=[dict(id=str(i),episode_id=str(i//2),family_id=i//4,split='test',present=bool(i%2),
        score=.5,iou=1. if i%2 else 0.,proposal_best_iou=1. if i%2 else 0.) for i in range(16)]
    metrics=summarize(rows,np.nextafter(.5,np.inf))
    assert metrics['auroc']==metrics['paired_scene_ranking_accuracy']==metrics['single_box_ap50']==.5
    assert metrics['localized_recall_at_iou50']==metrics['false_positive_rate']==0
    arms={method:rows for method in METHODS};cuts={method:.6 for method in METHODS}
    intervals=bootstrap(arms,cuts,samples=100,seed=3)
    assert intervals['families']==4 and intervals['unit']=='family'
    assert all(r['estimate']==0 and r['ci95']==[0.,0.] for r in intervals['differences'].values())

def test_rgb_validation_checks_file_and_decoded_hashes(tmp_path):
    rgb=example();path=tmp_path/'sample.png';Image.fromarray(rgb).save(path)
    h=hashlib.sha256();h.update(str(rgb.dtype).encode());h.update(str(rgb.shape).encode());h.update(rgb.tobytes())
    record=dict(rgb_path='sample.png',rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rgb_array_sha256=h.hexdigest())
    assert np.array_equal(runner.load_rgb(tmp_path,record),rgb)
    with pytest.raises(ValueError):runner.load_rgb(tmp_path,dict(record,rgb_sha256='0'*64))
    with pytest.raises(ValueError):runner.load_rgb(tmp_path,dict(record,rgb_array_sha256='0'*64))
    nested=tmp_path/'nested';nested.mkdir()
    with pytest.raises(ValueError):runner.load_rgb(nested,dict(record,rgb_path='../sample.png'))

def test_protocol_rejects_post_freeze_behavior_change(tmp_path):
    source=tmp_path/'behavior.py';source.write_text('version = 1')
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(model_selection_complete=True,fresh_test_generated_before_freeze=False,
        behavior_files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()})))
    assert runner.verify_protocol(protocol)['model_selection_complete']
    source.write_text('version = 2')
    with pytest.raises(ValueError):runner.verify_protocol(protocol)
