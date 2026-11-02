import base64,hashlib,importlib.util,io,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import http.client,threading
from functools import partial

path=Path(__file__).resolve().parents[1]/'scripts/serve_photo_lab.py'
spec=importlib.util.spec_from_file_location('photo_lab',path);lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)
def encoded(image=None):
    image=np.full((12,18,3),120,np.uint8) if image is None else image
    out=io.BytesIO();Image.fromarray(image).save(out,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
def test_upload_is_rgb_and_no_image_file_is_written(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);rgb=lab.decode_upload(encoded(np.arange(216,dtype=np.uint8).reshape(12,18)))
    assert rgb.shape==(12,18,3) and rgb.dtype==np.uint8
    assert np.array_equal(rgb[:,:,0],rgb[:,:,1]);assert list(tmp_path.iterdir())==[]
@pytest.mark.parametrize('value',['https://example.invalid/image.png','data:image/svg+xml;base64,PHN2Zy8+','data:image/png;base64,invalid!',None])
def test_upload_does_not_accept_urls_svg_or_invalid_encoding(value):
    with pytest.raises(ValueError):lab.decode_upload(value)
def test_decoded_pixel_limit_is_enforced_before_rgb_expansion(monkeypatch):
    monkeypatch.setattr(lab,'MAX_IMAGE_PIXELS',100)
    with pytest.raises(ValueError,match='16 million'):lab.decode_upload(encoded())
@pytest.mark.parametrize('box',[[0,0,float('nan'),12],[-1,0,8,8],[0,0,19,12],[0,0,1,8],[True,0,8,8],[0,0,8]])
def test_reference_box_rejects_invalid_coordinates(box):
    with pytest.raises(ValueError):lab.reference_box(box,np.zeros((12,18,3),np.uint8))
def test_whole_reference_default_is_native_coordinates():
    assert lab.reference_box(None,np.zeros((12,18,3),np.uint8))==[0,0,18,12]
def test_inference_receives_only_reference_box_and_query_rgb(monkeypatch,tmp_path):
    calls=[]
    def infer(reference,box,query):
        calls.append((reference,box,query));return {'native_patch':{'score':.25,'bbox_xyxy':[1,2,8,9],'proposals':[[1,2,8,9]],'evidence':{}}}
    monkeypatch.setattr(lab,'INFERENCE',infer);monkeypatch.setattr(lab,'completed_results',lambda:None);monkeypatch.chdir(tmp_path)
    result=lab.compare_request({'reference':encoded(),'query':encoded(),'reference_box':[1,1,17,11]})
    assert len(calls)==1 and calls[0][1]==[1,1,17,11]
    assert result['weight_updates']==0 and result['saved_uploads'] is False
    assert result['predictions']['native_patch']['threshold'] is None and result['predictions']['native_patch']['accepted'] is None
    assert list(tmp_path.iterdir())==[]
    with pytest.raises(ValueError,match='Only reference'):lab.compare_request({'reference':encoded(),'query':encoded(),'query_box':[1,1,2,2]})
def test_calibration_is_read_only_and_uses_dev_cutoff(monkeypatch):
    monkeypatch.setattr(lab,'INFERENCE',lambda *args:{'old_cnn':{'score':.8,'bbox_xyxy':None,'proposals':[]}})
    monkeypatch.setattr(lab,'completed_results',lambda:{'modes':{'real_dev_calibrated':{'methods':{'old_cnn':{'dev':{'threshold':.9},'test':{'threshold':.1}}}}}})
    r=lab.compare_request({'reference':encoded(),'query':encoded(),'operating_mode':'real_dev_calibrated'})['predictions']['old_cnn']
    assert r['threshold']==.9 and r['accepted'] is False
def test_path_and_json_inputs_reject_escape_and_nonfinite(tmp_path):
    with pytest.raises(ValueError):lab.safe_path(tmp_path,'../outside')
    with pytest.raises(ValueError):lab.finite_json('{"score":NaN}')
def test_gallery_original_dimensions_truth_and_crop_hashes(tmp_path,monkeypatch):
    source=np.zeros((12,18,3),np.uint8);source[2:10,3:15]=120
    for name in ['ref.png','query.png']:Image.fromarray(source).save(tmp_path/name)
    common={'category_id':8,'name':'example','width':18,'height':12}
    ref={**common,'rgb_path':'ref.png','sha256':lab.sha(tmp_path/'ref.png'),'image_id':1,'bbox_xyxy':[3,2,15,10]}
    query={**common,'rgb_path':'query.png','sha256':lab.sha(tmp_path/'query.png'),'image_id':2,'id':'test_example','split':'test','present':True,'gt_boxes_xyxy':[[3,2,15,10]]}
    (tmp_path/'manifest.json').write_text(json.dumps({'references':[ref],'queries':[query]}))
    monkeypatch.setattr(lab,'DATA',tmp_path);monkeypatch.setattr(lab,'prediction_index',lambda *a:{});lab._CACHE.clear()
    page=lab.scene_page({'limit':['999'],'page':['999']});assert page['page']==0 and page['total_count']==1
    assert page['rows'][0]['query_size']==[18,12] and page['rows'][0]['target_boxes']==[[3,2,15,10]]
    assert 'query_path' not in page['rows'][0]
    with Image.open(io.BytesIO(lab.cohort_image('test_example','reference.png'))) as image:assert image.size==(96,96)
    (tmp_path/'query.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):lab.cohort_image('test_example','query.png')

@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(lab,'completed_results',lambda:None)
    monkeypatch.setattr(lab,'INFERENCE',lambda *args:{'native_patch':{'score':.2,'bbox_xyxy':None,'proposals':[]}})
    instance=lab.ThreadingHTTPServer(('127.0.0.1',0),partial(lab.Handler,directory=str(lab.ROOT)))
    thread=threading.Thread(target=instance.serve_forever,daemon=True);thread.start()
    yield instance.server_port
    instance.shutdown();instance.server_close();thread.join(timeout=2)

def request(port,method,path,body=None,headers=None):
    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=3)
    connection.request(method,path,body=body,headers=headers or {});response=connection.getresponse()
    code=response.status;data=response.read();connection.close();return code,data

def test_http_rejects_nonlocal_host_and_cross_origin_upload(server):
    assert request(server,'GET','/api/config',headers={'Host':'external.example'})[0]==403
    headers={'Origin':'https://external.example','X-HyperNet-Token':lab.TOKEN,'Content-Type':'application/json'}
    assert request(server,'POST','/api/compare','{}',headers)[0]==403
    headers['Origin']=f'http://127.0.0.1:{server}';headers['X-HyperNet-Token']='incorrect'
    assert request(server,'POST','/api/compare','{}',headers)[0]==403

def test_http_local_upload_is_memory_only_and_cannot_request_files(server,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    headers={'Origin':f'http://127.0.0.1:{server}','X-HyperNet-Token':lab.TOKEN,'Content-Type':'application/json'}
    code,raw=request(server,'POST','/api/compare',json.dumps({'reference':encoded(),'query':encoded()}),headers)
    assert code==200 and json.loads(raw)['saved_uploads'] is False and list(tmp_path.iterdir())==[]
    assert request(server,'GET','/checkpoints/detector_micro_aware.pt')[0]==404
    assert request(server,'GET','/../outside')[0] in (400,404)

def test_http_request_size_limit_rejects_before_body_read(server):
    headers={'Origin':f'http://127.0.0.1:{server}','X-HyperNet-Token':lab.TOKEN,'Content-Type':'application/json',
             'Content-Length':str(lab.MAX_REQUEST_BYTES+1)}
    assert request(server,'POST','/api/compare','',headers)[0]==400

def test_public_benchmark_fallback_supplies_predictions_and_cutoffs(tmp_path,monkeypatch):
    benchmark=tmp_path/'benchmarks/v0.13';benchmark.mkdir(parents=True)
    log=benchmark/'native_patch_scores.jsonl';log.write_text(json.dumps({'id':'query','split':'test','score':.2,'bbox_xyxy':[1,2,3,4],'iou':0.})+'\n')
    aggregate=benchmark/'aggregate_metrics.json';aggregate.write_text(json.dumps({'completed':True,
        'modes':{'transferred':{'methods':{'native_patch':{'dev':{'threshold':.4},'test':{'threshold':.4}}}},
                 'real_dev_calibrated':{'methods':{'native_patch':{'dev':{'threshold':.1},'test':{'threshold':.1}}}}},
        'score_files':{'native_patch':{'path':log.name,'sha256':lab.sha(log)}}}))
    (benchmark/'public_export_provenance.json').write_text(json.dumps({'files':{'aggregate_metrics.json':{'public_sha256':lab.sha(aggregate)}}}))
    monkeypatch.setattr(lab,'ROOT',tmp_path);monkeypatch.setattr(lab,'RESULTS',tmp_path/'results/real_eval')
    assert lab.result_source()==aggregate
    assert lab.prediction_index('transferred')['native_patch']['query']['accepted'] is False
    assert lab.prediction_index('real_dev_calibrated')['native_patch']['query']['accepted'] is True
    log.write_text('{}')
    with pytest.raises(ValueError,match='hash'):lab.prediction_index()

def test_public_report_redirect_preserves_relative_asset_links(server,tmp_path,monkeypatch):
    benchmark=tmp_path/'benchmarks/v0.13';benchmark.mkdir(parents=True);(benchmark/'report.html').write_text('public report')
    monkeypatch.setattr(lab,'ROOT',tmp_path)
    connection=http.client.HTTPConnection('127.0.0.1',server,timeout=3);connection.request('GET','/index.html')
    response=connection.getresponse();assert response.status==302 and response.getheader('Location')=='/benchmarks/v0.13/report.html'
    assert response.read()==b'';connection.close()
