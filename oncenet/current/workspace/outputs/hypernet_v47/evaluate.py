"""Two prespecified development rounds; cached frozen features, no GPU."""
from core import *
import argparse
from threadpoolctl import threadpool_limits
O=R/'evaluation';O.mkdir(exist_ok=True)

def freeze():
 assert not (R/'protocol.json').exists() and not (O/'results.json').exists()
 sp=read(R/'splits.json');assert sp['audits']['passed'];bindings(sp['source_hashes'])
 checks=read(R/'checks.json');assert checks['passed'] and checks['core_sha256']==sha(R/'core.py');bindings(checks['files'])
 prior=read(V46/'evaluation/audit.json');assert prior['passed'] and prior['results_sha256']==sha(V46/'evaluation/results.json')
 reg=read(V46/'evaluation/registry.json');assert len(reg['models'])==11
 files=[Path(__file__),R/'core.py',R/'checks.py',R/'checks.json',R/'split.py',R/'splits.json',R/'audit.py',R/'method_review.json',R/'protocol_review.json',R/'data_review.json',R/'live_research.json',V46/'common.py',V46/'evaluation/audit.json',V46/'evaluation/results.json',V46/'evaluation/registry.json',V46/'evaluation/fits.json',V46/'data/records.json']
 files.extend([Path(C.MATCH.__file__),Path(C.CAL.__file__),Path(C.METRIC.__file__)])
 source_files={str(p):sha(p) for p in files}
 source_files.update(sp['source_hashes'])
 for e in reg['models']:
  folder=V46/'evaluation/features'/e['id'];b=read(folder/'binding.json');bindings(b['files']);source_files.update(b['files']);source_files[str(folder/'binding.json')]=sha(folder/'binding.json')
  for suffix in ['scores','states']:p=V46/'evaluation'/(e['id']+'_'+suffix+'.npz');source_files[str(p)]=sha(p)
 save(R/'protocol.json',dict(frozen_epoch=time.time(),files=source_files,models=reg['models'],matchers=MATCHERS,calibration_pools=POOLS,groupings=GROUPINGS,alphas=ALPHAS,
  execution_rounds=['pooled','detector'],all_rounds_declared_before_scores=True,primary_alpha=.001,secondary_alpha=.0005,
  target=dict(macro_balanced_accuracy=.9,each_target_fa_per10000_max=10.,lfb_mean_recall_not_below_v46_single=True),
  primary_ranking_metric='pauc_std_01',secondary_ranking_metric='pauc_std_001',
  input='Frozen v46 four-duration full+low descriptors; no new neural inference, weights or preprocessing',
  fit='Same original target-negative512ELDA FIT examples and same sole positive reference event; fit each duration separately',
  tune='Unused original FIT groups hash-stratified1/3 TUNE and2/3 CAL_ADD; target positives excluded. TUNE estimates per-view median and1.4826MAD. MAD<=1e-12 uses sample SD(ddof=1); SD<=1e-12 uses scale1.',
  fusion='Fixed mean or max of four standardized raw ELDA scores. Final fused score receives outer negative CAL. No independent-view or p-merger claim.',
  calibration='All original CAL retained; expanded adds disjoint CAL_ADD. Pooled or perIFO final-score order statistics. Ties conservative, unsupported alpha returns no alarms.',
  self_evaluation='Run all16 fixed configurations for every encoder, inspect both rounds and report all. No thresholds chosen using DEV; no reserve release; no further grid expansion based on these results.',
  one_positive_event=True,borrowed_hypernet_weights=False,reserve_closed=True,development_only=True,
  limitations=['Adaptive historical development and current DEV comparisons are not confirmation.','CAL_ADD consists of historically consumed FIT-role images and changes calibration mixture.','False-alarm control requires exchangeability not established across observing eras.','Three seeds reuse the same events.','Four windows are four PNGs of one reference event.','One positive event plus negative FIT/TUNE/CAL is not one total real example.','DINO is a matched-input comparison, not its strongest previously selected task-specific setup.']))
 print('V47 FROZEN',sha(R/'protocol.json'),flush=True)

def build_scores(e,rows,sp):
 folder=O/e['id'];folder.mkdir(exist_ok=True)
 assert not (folder/'binding.json').exists(),'Do not rerun scored model'
 raw_arrays={};states={};fused={};normalization={};parity=[]
 previous=np.load(V46/'evaluation'/(e['id']+'_scores.npz'),allow_pickle=False)
 features={d:np.load(V46/'evaluation/features'/e['id']/(d+'.npy'),mmap_mode='r',allow_pickle=False) for d in DURATIONS}
 for target in TARGETS:
  parts=[];ts=sp['targets'][target];fi=ts['fit'];ref=ts['reference']
  for d in DURATIONS:
   f=features[d];state=C.fit(f[fi],f[ref],f.shape[1]);score=np.empty(len(rows),np.float64)
   for start in range(0,len(rows),512):score[start:start+512]=np.asarray(f[start:start+512],np.float64)@state['weight']+state['offset']
   parts.append(score)
   for k,v in state.items():states[target+'__'+d+'__'+k]=v
  raw=np.stack(parts,1);joint=previous['multi__'+target];baseline=previous['single__'+target]
  np.testing.assert_allclose(raw[:,1],baseline,atol=2e-7,rtol=2e-8)
  params=scale_parameters(raw[ts['tune']]);out=fuse(raw,joint,params)
  normalization[target]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in params.items()}
  raw_arrays[target]=raw
  for method,value in out.items():fused[method+'__'+target]=value
  for alpha in ALPHAS:
   cut=C.CAL.cutoff(baseline[ts['original_cal']],alpha);cut2=threshold(raw[ts['original_cal'],1],alpha)
   assert np.array_equal(baseline[sp['indices']['development']]>cut,raw[sp['indices']['development'],1]>cut2)
  parity.append(dict(target=target,max_single_error=float(np.max(np.abs(raw[:,1]-baseline))),original_single_decisions_identical=True,joint_exact=np.array_equal(out['joint4'],joint)))
 for name,value in [('raw',raw_arrays),('states',states),('fused',fused)]:np.savez(folder/(name+'.npz'),**value)
 save(folder/'normalization.json',normalization)
 binding=dict(protocol_sha256=sha(R/'protocol.json'),files={str(folder/name):sha(folder/name) for name in ['raw.npz','states.npz','fused.npz','normalization.json']},parity=parity)
 save(folder/'binding.json',binding);return binding

def run():
 threadpool_limits(2);p=read(R/'protocol.json');bindings(p['files']);sp=read(R/'splits.json');rows=read(V46/'data/records.json')
 assert not (O/'started.json').exists(),'Do not rerun a started evaluation without diagnosis'
 save(O/'started.json',dict(protocol_sha256=sha(R/'protocol.json'),epoch=time.time()))
 labels=np.array([r['label'] for r in rows]);dev=np.array(sp['indices']['development']);sources={};start=time.time()
 for e in p['models']:
  save(O/'status.json',dict(stage='per_duration_matching',model=e['id'],seconds=time.time()-start))
  b=build_scores(e,rows,sp);sources.update(b['files']);sources[str(O/e['id']/'binding.json')]=sha(O/e['id']/'binding.json')
  print('V47 MATCHED',e['id'],flush=True)
 result_rows=[];macros=[];detector_rows=[];negative_rows=[];unions=[];calibration_details={}
 for grouping in GROUPINGS:
  for e in p['models']:
   fid=e['id'];z=np.load(O/fid/'fused.npz',allow_pickle=False);pp={};dd={};calibration_details[fid+'__'+grouping]={}
   for pool in POOLS:
    for method in MATCHERS:
     target_preds={};rank_scores={}
     for target in TARGETS:
      key=method+'__'+target;score=z[key];ci=sp['targets'][target]['original_cal' if pool=='original' else 'expanded_cal']
      pv,preds,detail=calibration(score,rows,ci,grouping);ck=pool+'__'+key;pp[ck]=pv;calibration_details[fid+'__'+grouping][ck]=detail
      # Conditional p-values provide the deployment ranking when group cutoffs differ.
      ranking=score if grouping=='pooled' else -pv;rank_scores[target]=ranking
      for alpha in ALPHAS:
       pred=preds[alpha];dd[ck+'__'+str(alpha)]=pred;target_preds[(target,alpha)]=pred
       info=dict(model=fid,seed=e['seed'],training=e['training'],matcher=method,pool=pool,grouping=grouping,target=target,alpha=alpha)
       result_rows.append(dict(**info,**metrics(labels==target,ranking,pred,dev)))
       for ifo in ['H1','L1']:
        ix=np.array([i for i in dev if rows[i]['ifo']==ifo],int);detector_rows.append(dict(**info,detector=ifo,**metrics(labels==target,ranking,pred,ix)))
       for label in sorted(set(labels[dev])-{target}):
        ix=dev[labels[dev]==label];negative_rows.append(dict(**info,negative_label=label,fp=int(pred[ix].sum()),negative=len(ix)))
     for alpha in ALPHAS:
      group=[x for x in result_rows if x['model']==fid and x['matcher']==method and x['pool']==pool and x['grouping']==grouping and x['alpha']==alpha]
      assert len(group)==4
      info=dict(model=fid,seed=e['seed'],training=e['training'],matcher=method,pool=pool,grouping=grouping,alpha=alpha)
      macros.append(dict(**info,**summarize(group)))
      anypred=np.logical_or.reduce([target_preds[(t,alpha)] for t in TARGETS]);bg=dev[~np.isin(labels[dev],TARGETS)]
      unions.append(dict(**info,negative_label='ALL_NON_TARGET',fp=int(anypred[bg].sum()),negative=len(bg),fa_per10000=float(anypred[bg].mean()*10000)))
      for label in sorted(set(labels[bg])):
       ix=bg[labels[bg]==label];unions.append(dict(**info,negative_label=label,fp=int(anypred[ix].sum()),negative=len(ix),fa_per10000=float(anypred[ix].mean()*10000)))
   folder=O/fid
   for suffix,data in [('pvalues',pp),('decisions',dd)]:
    path=folder/(grouping+'_'+suffix+'.npz');np.savez(path,**data);sources[str(path)]=sha(path)
   save(O/'status.json',dict(stage='calibration',grouping=grouping,model=fid,seconds=time.time()-start))
  # Every subsequent round was frozen before this self-evaluation checkpoint.
  save(O/('round_'+grouping+'.json'),dict(complete=True,grouping=grouping,protocol_sha256=sha(R/'protocol.json'),macro=[r for r in macros if r['grouping']==grouping],no_setting_changes=True))
  print('V47 ROUND COMPLETE',grouping,flush=True)
 save(O/'calibration.json',calibration_details);sources[str(O/'calibration.json')]=sha(O/'calibration.json')
 save(O/'results.json',dict(complete=True,development_only=True,reserve_closed=True,no_selection=True,protocol_sha256=sha(R/'protocol.json'),rows=result_rows,macro=macros,detector_rows=detector_rows,false_alarm_by_label=negative_rows,any_target_false_alarm=unions,files=sources,seconds=time.time()-start))
 save(O/'status.json',dict(stage='complete',complete=True,seconds=time.time()-start));print('V47 COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args();globals()[args.action]()
