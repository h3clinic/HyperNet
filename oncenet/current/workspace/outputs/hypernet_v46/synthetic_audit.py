"""Independent saved-margin/provenance audit; no rendering or model inference."""
from pathlib import Path
import argparse,hashlib,json,itertools
import numpy as np
R=Path(__file__).resolve().parent;S=R/'spectrogram';V45=R.parent/'hypernet_v45';M=V45/'multiseed';D=M/'diversity'
SEEDS=[45011,45029,45047];OFFSETS=[4600000,4610000,4620000];EDITS=['repetition','connectivity','curvature','orientation'];MODES=['base','spatial']
SETS={'spectrogram':('v46-spectral-power',4690000,'scene'),'retention':('v44-part-graphs',4691000,'graph')}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def check(ok,message):
 if not ok:raise AssertionError(message)
def verify_files(mapping):
 for p,h in mapping.items():check(sha(p)==h,'Bound file changed: '+p)
def seed_value(namespace,identity,split,component):return int.from_bytes(hashlib.sha256(f'{namespace}|{split}|{identity}|{component}'.encode()).digest()[:8],'little')
def order(seed):return np.random.default_rng(seed+7919).permutation(3200)+OFFSETS[SEEDS.index(seed)]
def fingerprint(entry):
 import torch
 check(sha(entry['path'])==entry['sha256'],'Checkpoint bytes changed: '+entry['id']);state=torch.load(entry['path'],map_location='cpu',weights_only=True);state=state.get('model',state);check(all(k.startswith('net.')for k in state),'Unexpected checkpoint key');check(sum(v.numel()for v in state.values())==42480==entry['parameters'],'Parameter count mismatch');h=hashlib.sha256()
 for k in sorted(state):
  value=state[k].detach().cpu().contiguous().numpy();check(np.isfinite(value).all(),'Nonfinite checkpoint');h.update(k.encode());h.update(str(value.shape).encode());h.update(str(value.dtype).encode());h.update(value.tobytes())
 return dict(id=entry['id'],seed=entry['seed'],arm=entry['arm'],checkpoint_sha256=entry['sha256'],tensor_sha256=h.hexdigest(),parameters=42480)
def namespace_check(plan):
 new_keys=['scene','reference_pose','query_pose','reference_appearance','query_appearance',*EDITS,'local_points'];old_keys=['graph','reference_pose','query_pose','reference_appearance','query_appearance',*EDITS,'v45_local_foreground_points'];records={};idsets={};seedsets={};existing=read(S/'checks.json')['namespace']
 for seed in SEEDS:records[f'train_{seed}']=('v46-spectral-power','train',order(seed),new_keys)
 records['spectrogram_test']=('v46-spectral-power','test',np.arange(4690000,4690256),new_keys);records['retention_test']=('v44-part-graphs','test',np.arange(4691000,4691256),old_keys)
 prior_ids=set(range(4500000,4503200))|set(range(4510000,4513200))|set(range(4520000,4523200))|set(range(4590000,4590256));prior_seeds={seed_value('v44-part-graphs',i,'test'if i>=4590000 else'train',k)for i in prior_ids for k in old_keys};details={}
 for name,(namespace,split,identities,keys)in records.items():
  ids=set(map(int,identities));values=[seed_value(namespace,int(i),split,k)for i in identities for k in keys];ss=set(values);check(len(ids)==len(identities)and len(ss)==len(values),'Within-set duplicate');check(not(ids&prior_ids)and not(ss&prior_seeds),'Overlap with prior v45 data');idsets[name]=ids;seedsets[name]=ss
  info=dict(identities=len(identities),minimum=int(min(identities)),maximum=int(max(identities)),component_keys=keys,seed_sha256=hashlib.sha256(np.asarray(values,dtype='<u8').tobytes()).hexdigest());check(info==existing[name],'Namespace audit mismatch: '+name);details[name]=info
  if name.startswith('train_'):check(hashlib.sha256(identities.astype('<i8').tobytes()).hexdigest()==plan['order_sha256'][name.split('_')[1]],'Frozen training order mismatch')
 for a,b in itertools.combinations(records,2):check(not(idsets[a]&idsets[b])and not(seedsets[a]&seedsets[b]),'Namespace overlap')
 check(len(set().union(*seedsets.values()))==101120,'Unexpected unique seed total')
 return dict(passed=True,groups=details,unique_component_seeds=101120,overlap_with_v45=0,heldout_images_rendered=0)
def audit(expected_protocol):
 audit_spec=read(R/'synthetic_audit_plan.json');check(audit_spec['audit_source_sha256']==sha(__file__)and audit_spec['protocol_sha256']==expected_protocol,'Audit source/protocol not prebound');check(audit_spec['no_synthetic_started_at_declaration']and audit_spec['no_synthetic_results_at_declaration'],'Audit was not declared before test entry')
 check(sha(S/'protocol.json')==expected_protocol,'Unexpected v46 protocol');p=read(S/'protocol.json');plan=p['plan'];oldp=read(M/'protocol.json');divp=read(D/'protocol.json');verify_files(p['files']);verify_files(oldp['source_hashes']);verify_files(divp['files']);checks=read(S/'checks.json');check(checks['passed']and sha(S/'checks.json')==p['checks_sha256'],'CPU checks not bound/passed');check(checks['files']==p['files'],'CPU-checked code differs');check(plan['seeds']==SEEDS and plan['parameters']==42480 and plan['steps']==800 and plan['batch']==4,'Training budget/schema differs');check(plan['arm']=='spectrogram'and plan['real_encoder_training_images']==0 and plan['borrowed_neural_weights']is False,'Training regime differs');check(plan['original_objective_protocol_sha256']==sha(D/'protocol.json')and plan['original_model_protocol_sha256']==sha(M/'protocol.json'),'Baseline protocol differs')
 check(plan['readouts']==MODES and plan['primary_readout']=='base','Readout differs')
 for domain,(namespace,offset,_)in SETS.items():check(plan['synthetic_sets'][domain]==dict(split='test',offset=offset,identities=256,generator_namespace=namespace),'Planned synthetic population differs')
 ns=namespace_check(plan);registry=read(S/'checkpoints.json');oldregistry=read(D/'checkpoints.json');originalregistry=read(M/'checkpoints.json');check(registry['training_complete']and registry['protocol_sha256']==expected_protocol,'New registry incomplete');check(oldregistry['training_complete']and oldregistry['protocol_sha256']==sha(D/'protocol.json'),'Old global registry incomplete');check(originalregistry['training_complete']and originalregistry['protocol_sha256']==sha(M/'protocol.json'),'Original random registry incomplete')
 randoms=[e for e in originalregistry['models']if e['arm']=='random'];globals_=oldregistry['models'];new=registry['models'];entries=randoms+globals_+new;check(len(entries)==9 and len({e['id']for e in entries})==9,'Nine-model membership differs');check({(e['seed'],e['arm'])for e in entries}=={(s,a)for s in SEEDS for a in ['random','global_diversity','spectrogram']},'Unexpected seed/arm pair')
 initial={e['seed']:e for e in plan['initial_controls']};check(set(initial)==set(SEEDS),'Initial controls incomplete')
 for seed in SEEDS:
  rand=next(e for e in randoms if e['seed']==seed);newrow=next(e for e in new if e['seed']==seed);oldrow=next(e for e in globals_ if e['seed']==seed);check(initial[seed]['sha256']==sha(initial[seed]['path'])==rand['sha256'],'Different initial checkpoint');check(next(e for e in oldp['initial_controls']if e['seed']==seed)['sha256']==rand['sha256'],'Random state not original frozen initialization')
  done=read(S/f'seed_{seed}/completed.json');check(done==read(S/f'seed_{seed}/status.json'),'New completion/status mismatch');check(done['stage']=='completed'and done['step']==800 and done['total']==800 and done['seed']==seed and done['protocol_sha256']==expected_protocol and done['no_checkpoint_selection'],'New training completion inconsistent');check(done['checkpoint_sha256']==newrow['sha256']and done['random_initialization_sha256']==rand['sha256'],'New weights/init not bound')
  previous=read(D/f'seed_{seed}/completed.json');check(previous['checkpoint_sha256']==oldrow['sha256']and previous['random_initialization_sha256']==rand['sha256'],'Old global model/init binding changed')
  history=read(S/f'seed_{seed}/history.json');check(len(history)==800 and[r['step']for r in history]==list(range(1,801)),'Incomplete step history');check([i for row in history for i in row['identities']]==order(seed).tolist(),'Incorrect episode sequence')
 details=[fingerprint(e)for e in entries];check(len({d['tensor_sha256']for d in details})==9,'Duplicate neural states')
 result=read(S/'synthetic_results.json');started=read(S/'synthetic_started.json');done=read(S/'synthetic_completed.json');rh=sha(S/'synthetic_results.json');registry_hash=sha(S/'checkpoints.json');check(result['protocol_sha256']==started['protocol_sha256']==done['protocol_sha256']==expected_protocol,'Synthetic protocol binding differs');check(result['registry_sha256']==started['registry_sha256']==done['registry_sha256']==registry_hash,'Synthetic registry binding differs');check(result['models']==started['models']==entries,'Evaluated models differ');check(started['time']>p['frozen_epoch']and started['time']>audit_spec['declared_epoch'],'Test began before protocol/audit freeze');check(done['passed']and done['synthetic_results_sha256']==rh and done['models']==9 and done['sets']==2 and done['identities_per_set']==256,'Synthetic completion differs');check(result['no_checkpoint_selection']and set(result['sets'])==set(SETS),'Unexpected tests/selection');rows=[]
 for domain,(namespace,offset,component)in SETS.items():
  data=result['sets'][domain];check(data['identities']==256 and data['paired_cases']==1024 and data['identity_offset']==offset,'Synthetic count/range differs');latent=[seed_value(namespace,i,'test',component)for i in range(offset,offset+256)];check(data['latent_seed_sha256']==hashlib.sha256(json.dumps(latent).encode()).hexdigest(),'Synthetic latent order mismatch');check(set(data['models'])=={e['id']for e in entries},'Model set mismatch')
  for model,modes in data['models'].items():
   check(set(modes)==set(MODES),'Readout mismatch')
   for mode,cell in modes.items():
    check(set(cell['edits'])==set(EDITS),'Edit types differ');margins=[]
    for edit in EDITS:
     saved=cell['edits'][edit];values=np.asarray(saved['margins'],np.float64);check(values.shape==(256,)and np.isfinite(values).all(),'Invalid saved margin array');check(saved['accuracy']==float(np.mean(values>0))and saved['mean_margin']==float(values.mean()),'Edit statistic differs');margins.append(values)
    a=np.stack(margins);accuracy=float(np.mean(a>0));all_four=float(np.mean(np.all(a>0,axis=0)));check(cell['overall_accuracy']==accuracy and cell['all_four_correct']==all_four,'Aggregate statistic differs');rows.append(dict(domain=domain,model=model,readout=mode,overall_accuracy=accuracy,all_four_correct=all_four,mean_margin=float(a.mean()),median_margin=float(np.median(a)),zero_margin_count=int((a==0).sum()),identities=256,dependent_edit_pairs=1024))
 check(len(rows)==36,'Expected9models×2sets×2readouts');paths=[Path(__file__),R/'synthetic_audit_plan.json',S/'protocol.json',S/'checks.json',S/'plan.json',S/'research.json',S/'checkpoints.json',S/'synthetic_started.json',S/'synthetic_results.json',S/'synthetic_completed.json',M/'protocol.json',M/'checkpoints.json',D/'protocol.json',D/'checkpoints.json'];return dict(passed=True,audit_code_sha256=sha(__file__),protocol_sha256=expected_protocol,results_sha256=rh,synthetic_results_sha256=rh,registry_sha256=registry_hash,model_count=9,set_count=2,readouts=MODES,model_readout_results_checked=36,margin_arrays_checked=144,total_saved_margins_checked=36864,namespace=ns,checkpoint_details=details,rows=rows,files={str(p):sha(p)for p in paths},encoder_forwards=0,images_rendered=0,tie_convention='Strictly positive margin is correct; ties count as incorrect.',limits='Two renderer/grammar-specific identity holdouts with dependent edits. This verifies recorded calculations and provenance, not real-image transfer or absence of undisclosed historical inspection.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();check(R in out.parents and not out.exists(),'Audit destination must be new and within v46');value=audit(a.protocol_sha256);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2,allow_nan=False));print('V46 SYNTHETIC AUDIT PASSED',out)
