"""Scientific data invariants: independent human votes, overlap, and sealed TEST."""
from pathlib import Path
import importlib.util,json,sqlite3
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/prepare_fresh_gravity.py'
spec=importlib.util.spec_from_file_location('data_gravity_v16_builder',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_registered_user_counts_once_and_multilabel_or_changing_users_do_not_vote():
    c=sqlite3.connect(':memory:');c.execute('CREATE TABLE votes(subject TEXT,user TEXT,workflow INTEGER,version TEXT,ballot INTEGER,label TEXT)')
    rows=[('event',user,2360,'92.12',ballot,label)for user,ballot,label in
        [('repeat',1,'Blip'),('repeat',1,'Blip'),('repeat',2,'Blip'),('multi',3,'Blip'),('multi',3,'Helix'),('change',4,'Blip'),('change',5,'Helix'),('second',6,'Blip')]]
    rows += [('event','wrong_version',2360,'91.12',7,'Helix'),('event','wrong_workflow',7767,'92.12',8,'Helix')]
    c.executemany('INSERT INTO votes VALUES(?,?,?,?,?,?)',rows)
    assert c.execute(m.VERSION_VOTE_QUERY,(2360,'92.12')).fetchall()==[('event','Blip',2)]

def test_consensus_requires_five_users_and_eighty_percent_without_tie_break_labels():
    rows=[{'subject_id':'a','label':'Blip','votes':4},{'subject_id':'a','label':'Helix','votes':1},
        {'subject_id':'b','label':'Blip','votes':4},{'subject_id':'c','label':'Blip','votes':3},{'subject_id':'c','label':'Helix','votes':2},
        {'subject_id':'d','label':'Blip','votes':5},{'subject_id':'d','label':'Helix','votes':5}]
    assert set(m.consensus_counts(rows))=={'a'}
    assert m.consensus_counts(rows)['a']['agreement']==.8

def test_connected_time_groups_close_transitive_exposure_and_keep_detectors_separate():
    def row(name,t,detector='H1'):return {'event_id':name,'gps_ns':t*1_000_000_000,'detector':detector}
    groups=m.connected_time_groups([row('last',8),row('prior',0),row('middle',4),row('other_detector',4,'L1'),row('far',13)])
    assert [[r['event_id']for r in group]for group in groups]==[['prior','middle','last'],['far'],['other_detector']]

def test_data_paths_reject_traversal_before_any_network_request(tmp_path):
    with pytest.raises(ValueError,match='Noncanonical'):
        m.fetch_event({'event_id':'../outside','image_path':'../outside.png','image_url':'https://panoptes-uploads.zooniverse.org/x'},tmp_path,'test')

def test_source_url_rejects_nonofficial_host(tmp_path):
    with pytest.raises(ValueError,match='Official HTTPS'):
        m.fetch_event({'event_id':'0123456789','image_path':'images/0123456789_2.0.png','image_url':'https://example.com/x'},tmp_path,'test')

def test_protocol_checks_actual_model_and_source_bytes_not_just_hash_syntax(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'DATA',tmp_path/'data');m.DATA.mkdir()
    plan=m.DATA/'plan.json';plan.write_text('{}');source=tmp_path/'inference.py';source.write_text('frozen');checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'own model')
    protocol={'fresh_plan_sha256':m.digest(plan),'model_selection_complete':True,'test_scores_computed_before_freeze':False,
        'behavior_files':{'inference.py':m.digest(source)},'checkpoint_files':{'model.pt':m.digest(checkpoint)}}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol));expected=m.digest(path)
    assert m.verify_model_protocol(path,expected)==protocol
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='binding differs'):m.verify_model_protocol(path,expected)

def test_scoring_admission_does_not_accept_absent_authorization(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'DATA',tmp_path);monkeypatch.setattr(m,'verify_plan',lambda:{'events':[]})
    (tmp_path/'plan.json').write_text('{}');admission=tmp_path/'admission.json';admission.write_text(json.dumps({'fresh_plan_sha256':m.digest(tmp_path/'plan.json')}))
    with pytest.raises(PermissionError,match='admission required'):m.FreshGravityDataset(admission)
