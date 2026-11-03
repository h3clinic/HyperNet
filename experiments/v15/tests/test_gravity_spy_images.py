import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hypernet.gravity_spy_images import GravitySpyCohort
from PIL import Image
import numpy as np
import pytest


def fixture(root):
    rows=[]
    for event_id,split in [('ref','reference'),('dev','calibration'),('hidden','test'),('excluded','excluded_pre_exposed_group')]:
        rows.append({'event_id':event_id,'image_url':'https://example.invalid/'+event_id,'image_path':f'images/{event_id}.png','split':split,'group_id':event_id})
    plan={'events':rows,'target_labels':['pattern'],'reference_by_label':{'pattern':'ref'},'plot_crop_xyxy':[105,66,671,532]}
    (root/'plan.json').write_text(json.dumps(plan));sha=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
    (root/'roster_freeze.json').write_text(json.dumps({'roster_sha256':sha}))
    (root/'images').mkdir();(root/'image_records').mkdir()
    for row in rows:
        rgb=np.zeros((466,566,3),dtype=np.uint8);path=root/row['image_path'];Image.fromarray(rgb).save(path)
        record={**row,'crop_xyxy':plan['plot_crop_xyxy'],'crop_dimensions':[566,466],
            'crop_png_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'crop_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
        (root/'image_records'/f'{row["event_id"]}.json').write_text(json.dumps(record))
    return sha


def test_rgb_api_does_not_unseal_test_pixels(tmp_path):
    fixture(tmp_path);cohort=GravitySpyCohort(tmp_path)
    assert cohort.reference('pattern').shape==(466,566,3)
    assert len(cohort.query_events('calibration'))==1
    with pytest.raises(PermissionError):cohort.load_rgb('hidden')
    with pytest.raises(PermissionError):cohort.load_rgb('excluded')


def test_image_mutation_detected(tmp_path):
    fixture(tmp_path);cohort=GravitySpyCohort(tmp_path)
    (tmp_path/'images/ref.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='PNG changed'):cohort.reference('pattern')


def test_calibration_admission_is_roster_bound(tmp_path):
    sha=fixture(tmp_path)
    admission={'roster_sha256':sha,'calibration_complete':True,'thresholds':{'pattern':.5},
        'model_checkpoint_sha256':'a'*64,'inference_source_sha256':'b'*64,'calibration_scores_sha256':'c'*64}
    path=tmp_path/'admission.json';path.write_text(json.dumps(admission))
    # Schema-shaped hashes alone can no longer authorize TEST pixels.
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):GravitySpyCohort(tmp_path,path)
    admission['roster_sha256']='d'*64;path.write_text(json.dumps(admission))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):GravitySpyCohort(tmp_path,path)
