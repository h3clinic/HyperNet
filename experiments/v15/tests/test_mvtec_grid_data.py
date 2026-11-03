import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
api=load('grid_data_api',ROOT/'hypernet/mvtec_grid.py')
builder=load('grid_data_builder',ROOT/'scripts/prepare_mvtec_grid.py')

def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')

@pytest.fixture
def cohort(tmp_path):
    records=[]
    for i,role in enumerate(('reference','dev_normal','unused_normal','test','test')):
        image=np.arange(64,dtype=np.uint8).reshape(8,8)+i
        rgb=np.repeat(image[:,:,None],3,2);path=tmp_path/f'{i}.png';Image.fromarray(image).save(path)
        anomaly=i==4;mask=None
        if anomaly:
            array=np.zeros((8,8),dtype=np.uint8);array[2:4,3:5]=255
            target=tmp_path/'mask.png';Image.fromarray(array).save(target)
            mask={'path':target.name,'sha256':api.digest(target),'pixel_sha256':api.pixels(array)}
        records.append(dict(id=str(i),role=role,official_split='test' if role=='test' else 'train',
            defect_type='broken' if anomaly else 'good',is_anomaly=anomaly,rgb_path=path.name,
            sha256=api.digest(path),pixel_sha256=api.pixels(rgb),native_pixel_sha256=api.pixels(image),width=8,height=8,mask=mask))
    manifest={'format':'hypernet.mvtec_grid.manifest.v15','completed':True,'records':records}
    write(tmp_path/'manifest.json',manifest)
    write(tmp_path/'development_manifest.json',{**manifest,'records':records[:2]})
    return tmp_path,records

def authorize(root):
    protocol=root/'protocol.json'
    write(protocol,{'model_selection_complete':True,'test_scores_computed_before_freeze':False,
                    'grid_manifest_sha256':api.digest(root/'manifest.json')})
    write(root/'test_admission.json',{'test_pixels_authorized_for_frozen_inference':True,
        'manifest_sha256':api.digest(root/'manifest.json'),'protocol_path':str(protocol),'protocol_sha256':api.digest(protocol)})

def test_native_grayscale_values_preserved_without_resize_or_annotation_output(cohort):
    root,records=cohort;dataset=api.GridDataset(root)
    rgb=dataset.load_rgb('0')
    assert rgb.dtype==np.uint8 and rgb.shape==(8,8,3)
    assert np.array_equal(rgb[:,:,0],np.arange(64,dtype=np.uint8).reshape(8,8))
    assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2])
    assert dataset.ids('reference')==['0'] and dataset.ids('dev_normal')==['1']
    with pytest.raises(KeyError):dataset.load_rgb('3')

def test_test_pixels_are_blocked_before_any_decoder_call(cohort,monkeypatch):
    root,_=cohort;dataset=api.GridDataset(root,development_only=False)
    monkeypatch.setattr(api.Image,'open',lambda *a,**k:pytest.fail('Pixel decoder must not be reached'))
    with pytest.raises(ValueError,match='frozen protocol admission'):dataset.load_rgb('3')
    with pytest.raises(ValueError,match='frozen protocol admission'):dataset.load_evaluation_target('4')
    with pytest.raises(ValueError,match='Unused normals'):dataset.load_rgb('2')

def test_admitted_masks_are_separate_original_size_boolean_targets(cohort):
    root,_=cohort;authorize(root);dataset=api.GridDataset(root,development_only=False)
    assert isinstance(dataset.load_rgb('4'),np.ndarray)
    target=dataset.load_evaluation_target('4')
    assert target['is_anomaly'] and target['mask'].dtype==bool and target['mask'].shape==(8,8) and target['mask'].sum()==4
    assert dataset.load_evaluation_target('3')['mask'].sum()==0
    with pytest.raises(ValueError):dataset.load_evaluation_target('0')

def test_gate_rechecks_protocol_and_input_integrity(cohort):
    root,_=cohort;authorize(root);dataset=api.GridDataset(root,development_only=False)
    dataset.load_rgb('3')
    protocol=json.loads((root/'protocol.json').read_text());protocol['test_scores_computed_before_freeze']=True
    write(root/'protocol.json',protocol)
    with pytest.raises(ValueError,match='Protocol admission changed'):dataset.load_rgb('3')
    (root/'0.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw hash'):api.GridDataset(root).load_rgb('0')

@pytest.mark.parametrize('name',['../escape','/absolute','grid/../../escape','C:/escape','grid\\bad.png'])
def test_archive_paths_reject_escape_and_nonportable_syntax(name):
    with pytest.raises(ValueError):builder.safe_member(name)

def test_normal_archive_paths_and_evidence_are_preserved(tmp_path):
    assert builder.safe_member('grid/train/good/001.png')=='grid/train/good/001.png'
    path=tmp_path/'proof.json';builder.write_new(path,{'fixed':True})
    with pytest.raises(FileExistsError):builder.write_new(path,{'fixed':False})
    assert builder.read(path)=={'fixed':True}

def test_dataset_path_escape_rejected_before_read(cohort):
    root,records=cohort
    path=root/'development_manifest.json';manifest=json.loads(path.read_text());manifest['records'][0]['rgb_path']='../other.png';write(path,manifest)
    with pytest.raises(ValueError,match='escapes root'):api.GridDataset(root).load_rgb('0')
