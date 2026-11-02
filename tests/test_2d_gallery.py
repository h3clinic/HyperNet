from pathlib import Path
import importlib.util
import json
from io import BytesIO
import numpy as np
from PIL import Image
import pytest

spec=importlib.util.spec_from_file_location('gallery_2d',Path(__file__).resolve().parents[1]/'scripts/serve_2d_gallery.py')
gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    Image.new('RGB',(96,96)).save(data/'support.png')
    Image.new('RGB',(256,256)).save(data/'query.png')
    mask=np.zeros((256,256),dtype=bool);mask[30:80,40:100]=True
    np.savez(data/'mask.npz',union_mask=mask,target_mask=mask,candidate_mask=mask)
    episodes=[]
    for family in range(30):
        for enrollment in (0,1):
            episode=dict(episode_id=f'f{family}_e{enrollment}',split='test',family_id=family,
                grammar_digits=[0,0,0,family],enrolled_variant=enrollment,
                support={'rgb_path':'support.png','bbox_xyxy':[0,0,96,96]},queries=[])
            for variant in (0,1):
                episode['queries'].append(dict(id=f'f{family}_e{enrollment}_q{variant}',
                    present=variant==enrollment,candidate_variant=variant,rgb_path='query.png',
                    target_path='mask.npz',candidate_bbox_xyxy=[40,30,100,80],
                    target_bbox_xyxy=[40,30,100,80] if variant==enrollment else None,all_boxes=[]))
            episodes.append(episode)
    path=data/'manifest.json';path.write_text(json.dumps({'completed':True,'episodes':episodes}))
    monkeypatch.setattr(gallery,'ROOT',tmp_path)
    monkeypatch.setattr(gallery,'MANIFESTS',(path,tmp_path/'missing.json'))
    gallery._CACHE.clear()
    gallery._PREDICTIONS.clear()
    return tmp_path


def test_pagination_covers_every_row_once_and_caps_response(fixture):
    ids=[]
    for number in range(5):
        result=gallery.page({'page':[str(number)]})
        assert len(result['rows'])==24 and result['pages']==5
        ids += [row['id'] for row in result['rows']]
    assert len(ids)==len(set(ids))==120
    assert len(gallery.page({'limit':['100000']})['rows'])==48
    assert 'target_path' not in result['rows'][0]
    assert str(fixture) not in json.dumps(result)


def test_filters_preserve_counterbalanced_identity_labels(fixture):
    rows=gallery.page({'family':['4']})['rows']
    assert len(rows)==4 and sum(r['present'] for r in rows)==2
    for variant in (0,1):
        same=[r for r in rows if r['candidate_variant']==variant]
        assert {r['present'] for r in same}=={True,False}
        assert same[0]['query_url']==same[1]['query_url']
    assert len(gallery.page({'family':['4'],'present':['yes'],'micro':['1']})['rows'])==1


def test_target_previews_are_lossless_binary_and_unknown_fields_rejected(fixture):
    png=gallery.mask_png('f0_e0_q0','union_mask')
    decoded=np.array(Image.open(BytesIO(png)))
    assert decoded.shape==(256,256) and set(np.unique(decoded))=={0,255}
    assert (decoded>0).sum()==3000
    with pytest.raises(KeyError):gallery.mask_png('f0_e0_q0','other')


def test_manifest_paths_cannot_escape_dataset_root(fixture):
    with pytest.raises(ValueError):gallery.safe_path(fixture/'data','../private.txt')


def test_completed_prediction_overlay_binds_log_hash_and_dev_cutoff(fixture):
    from hashlib import sha256
    results=fixture/'results';results.mkdir()
    log=results/'micro_aware_predictions.jsonl'
    log.write_text(json.dumps({'id':'f0_e0_q0','split':'test','score':.7,'bbox_xyxy':[1,2,3,4],'iou':.1})+'\n')
    result=results/'detector_results.json'
    result.write_text(json.dumps({'completed':True,'sources_unchanged':True,'arms':{'micro_aware':{
        'predictions_sha256':sha256(log.read_bytes()).hexdigest(),'metrics':{'test':{'threshold':.8}}}}}))
    row=gallery.page({'family':['0']})['rows'][0]
    assert row['predictions']['micro_aware']['accepted'] is False
    assert row['predictions']['micro_aware']['bbox_xyxy']==[1,2,3,4]
    assert 'target_path' not in row
    log.write_text(log.read_text()+' ')
    gallery._PREDICTIONS.clear()
    with pytest.raises(ValueError,match='hash'):gallery.prediction_index()
