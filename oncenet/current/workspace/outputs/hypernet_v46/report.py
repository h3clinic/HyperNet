"""Publish the fixed v46 development ablations after every required audit."""
from common import *
import html,shutil
from datetime import datetime,timezone
WEB=P/'hypernet_v20/research';ASSETS=WEB/'v46_assets'
NAMES={'random':'Random weights','shapes':'Prior shape training','spectrogram':'Spectrogram training','historical':'Old HyperNet v37','external':'DINO-small · matched inputs'}
def esc(x):return html.escape(str(x),quote=True)
def n(x,d=3):return '—' if x is None else f'{x:.{d}f}'
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def mean(rows,key):return float(np.mean([x[key] for x in rows]))

def main():
 result=read(R/'evaluation/results.json');audit=read(R/'evaluation/audit.json');syn=read(R/'spectrogram/synthetic_results.json');syn_audit=read(R/'synthetic_audit.json');data=read(R/'data/audit.json');acquisition=read(R/'data/acquisition_audit.json')
 preflight=read(R/'evaluation_data_preflight.json');review=read(R/'data/collision_review.json')
 assert result['complete'] and result['no_new_real_holdout'] and result['no_selection'] and audit['passed'] and syn_audit['passed'] and preflight['passed']
 bindings(preflight['files'])
 assert audit['results_sha256']==sha(R/'evaluation/results.json')
 assert syn_audit['results_sha256']==sha(R/'spectrogram/synthetic_results.json')
 assert audit['protocol_sha256']==sha(R/'evaluation/protocol.json') and syn_audit['protocol_sha256']==sha(R/'spectrogram/protocol.json')
 assert audit['audit_code_sha256']==sha(R/'evaluation_audit.py')
 assert syn_audit['audit_code_sha256']==sha(R/'synthetic_audit.py')
 bindings(syn_audit['files'])
 protocol=read(R/'evaluation/protocol.json');bindings(protocol['files']);bindings(result['files'])
 assert result['protocol_sha256']==sha(R/'evaluation/protocol.json')
 assets={}
 for name,path in [
  ('intent.json',R/'intent.json'),('research.json',R/'research.json'),('protocol_review.json',R/'protocol_review.json'),
  ('evaluation_protocol.json',R/'evaluation/protocol.json'),('evaluation_results.json',R/'evaluation/results.json'),('evaluation_audit.json',R/'evaluation/audit.json'),
  ('data_plan.json',R/'data/plan.json'),('data_audit.json',R/'data/audit.json'),('acquisition_audit.json',R/'data/acquisition_audit.json'),
  ('training_protocol.json',R/'spectrogram/protocol.json'),('training_checks.json',R/'spectrogram/checks.json'),('training_registry.json',R/'spectrogram/checkpoints.json'),
  ('synthetic_results.json',R/'spectrogram/synthetic_results.json'),('synthetic_audit.json',R/'synthetic_audit.json'),('matcher_checks.json',R/'matcher_checks.json')]:assets[name]=path
 assets['evaluation_data_preflight.json']=R/'evaluation_data_preflight.json'
 assets['transfer_hypothesis.json']=R/'transfer_hypothesis.json'
 for name in ['download_failures.json','excluded_events.json','source_failure_probe.json','supplemental_data_provenance.json','collision_review.json']:
  if (R/'data'/name).exists():assets[name]=R/'data'/name
 for path in [R/'comparison.png',R/'comparison.svg',R/'reference_windows.png',R/'spectrogram/gallery.png']:
  if path.exists():assets[path.name]=path
 ASSETS.mkdir(parents=True,exist_ok=True)
 for name,path in assets.items():shutil.copy2(path,ASSETS/name);assert sha(path)==sha(ASSETS/name)
 primary=[];all_settings=[]
 for x in result['seed_summary']:
  m=x['metrics'];row=[esc(NAMES[x['training']]),esc(x['view']),x['alpha'],pct(m['balanced_accuracy']['mean']),n(m['mcc']['mean']),pct(m['recall']['mean']),n(m['fa_per10000']['mean'],1),n(m['pauc_std_01']['mean']),n(m['pauc_std_001']['mean'])]
  all_settings.append(row)
  if x['alpha']==.001:primary.append(row)
 for x in result['macro']:
  if x['seed'] is None:
   row=[esc(NAMES[x['training']]),esc(x['view']),x['alpha'],pct(x['balanced_accuracy']),n(x['mcc']),pct(x['recall']),n(x['fa_per10000'],1),n(x['pauc_std_01']),n(x['pauc_std_001'])];all_settings.append(row)
   if x['alpha']==.001:primary.append(row)
 duration=[];generator=[];interaction=[]
 for training in ['random','shapes','spectrogram','historical','external']:
  a=[x for x in result['duration_pairs'] if x['training']==training and x['alpha']==.001]
  duration.append([esc(NAMES[training]),n(np.mean([x['metrics']['pauc_std_01'] for x in a])),', '.join(n(x['metrics']['pauc_std_01']) for x in a),n(np.mean([x['metrics']['mcc'] for x in a])),n(100*np.mean([x['metrics']['recall'] for x in a]),2),n(np.mean([x['metrics']['fa_per10000'] for x in a]),1)])
 for left,right in [('spectrogram','shapes'),('shapes','random'),('spectrogram','random')]:
  for view in ['single','multi']:
   a=[x for x in result['training_pairs'] if x['left']==left and x['right']==right and x['view']==view and x['alpha']==.001]
   generator.append([esc(NAMES[left]+' minus '+NAMES[right]),view,n(np.mean([x['metrics']['pauc_std_01'] for x in a])),', '.join(n(x['metrics']['pauc_std_01']) for x in a),n(np.mean([x['metrics']['mcc'] for x in a])),n(100*np.mean([x['metrics']['recall'] for x in a]),2),n(np.mean([x['metrics']['fa_per10000'] for x in a]),1)])
 for trained in ['shapes','spectrogram']:
  a=[x for x in result['interactions'] if x['trained']==trained and x['alpha']==.001]
  interaction.append([esc(NAMES[trained]),n(np.mean([x['metrics']['pauc_std_01'] for x in a])),', '.join(n(x['metrics']['pauc_std_01']) for x in a)])
 synthetic=[]
 for domain,d in syn['sets'].items():
  for group,suffix in [('random','random'),('shapes','global_diversity'),('spectrogram','spectrogram')]:
   for readout in ['base','spatial']:
    values=[d['models'][f's{seed}_{suffix}'][readout] for seed in SEEDS]
    synthetic.append([domain,esc(NAMES[group]),readout,pct(np.mean([x['overall_accuracy'] for x in values])),', '.join(pct(x['overall_accuracy']) for x in values),pct(np.mean([x['all_four_correct'] for x in values]))])
 classrows=[];detectors=[]
 for group in ['random','shapes','spectrogram','historical','external']:
  for view in ['single','multi']:
   for target in TARGETS:
    a=[x for x in result['rows'] if x['training']==group and x['view']==view and x['target']==target and x['alpha']==.001]
    classrows.append([esc(NAMES[group]),view,esc(target),a[0]['positive'],a[0]['negative'],pct(mean(a,'recall')),n(mean(a,'fa_per10000'),1),n(mean(a,'mcc')),n(mean(a,'pauc_std_01'))])
    for detector in ['H1','L1']:
     a=[x for x in result['detector_rows'] if x['training']==group and x['view']==view and x['target']==target and x['alpha']==.001 and x['detector']==detector]
     if a:detectors.append([esc(NAMES[group]),view,esc(target),detector,a[0]['positive'],a[0]['negative'],pct(mean(a,'recall')),n(mean(a,'fa_per10000'),1),n(mean(a,'mcc'))])
 stress=[]
 for x in result['any_target_false_alarm_by_label']:
  if x['alpha']==.001:stress.append([esc(x['model']),x['view'],esc(x['negative_label']),x['fp'],x['negative'],n(x['fp']/x['negative']*10000,1)])
 gain=[x['metrics']['pauc_std_01'] for x in result['duration_pairs'] if x['training']=='spectrogram' and x['alpha']==.001]
 train_gain=[x['metrics']['pauc_std_01'] for x in result['training_pairs'] if x['left']=='spectrogram' and x['right']=='shapes' and x['view']=='single' and x['alpha']==.001]
 union_summary=[]
 for group in ['random','shapes','spectrogram','historical','external']:
  ids=sorted({x['model'] for x in result['macro'] if x['training']==group})
  for view in ['single','multi']:
   counts=[];denominators=[]
   for model in ids:
    a=[x for x in result['any_target_false_alarm_by_label'] if x['model']==model and x['view']==view and x['alpha']==.001]
    counts.append(sum(x['fp'] for x in a));denominators.append(sum(x['negative'] for x in a))
   assert len(set(denominators))==1
   union_summary.append([esc(NAMES[group]),view,', '.join(map(str,counts)),denominators[0],n(np.mean(counts)/denominators[0]*10000,1)])
 def group_metrics(group,view,alpha=.001):
  return next(x['metrics'] for x in result['seed_summary'] if x['training']==group and x['view']==view and x['alpha']==alpha)
 prior=group_metrics('shapes','single');new=group_metrics('spectrogram','single');multi_new=group_metrics('spectrogram','multi')
 strict_prior=group_metrics('shapes','multi',.0005);strict_new=group_metrics('spectrogram','multi',.0005)
 generated=datetime.now(timezone.utc).isoformat()
 css='''body{margin:0;background:#f3f5f8;color:#172334;font:16px/1.6 system-ui}main{max-width:1150px;margin:auto;padding:16px}header,section{padding:24px;border-radius:16px;margin:0 0 20px;background:white;border:1px solid #dce3ec}header{background:#142d48;color:white}h1{font-size:clamp(27px,4vw,40px);line-height:1.15}h2{line-height:1.3}a{color:#285f9b}header a{color:#c1e0ff}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:760px;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dde3eb;text-align:left}th{background:#eef2f7}.notice{border-left:4px solid #a87a18;background:#fff5de;padding:12px 16px;margin:16px 0}img{width:100%;height:auto}details{margin:20px 0}summary{font-weight:650;cursor:pointer}code{overflow-wrap:anywhere}footer{padding:20px;color:#5d6775}@media(max-width:650px){main{padding:10px}header,section{padding:17px}}'''
 doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v46 · temporal context and spectrogram training</title><style>'+css+'</style><main>'
 doc+='<header><p>CONTROLLED DEVELOPMENT STUDY · THREE PAIRED SEEDS</p><h1>Does more temporal context—and more relevant synthetic training—help?</h1><p>Two separately measured changes: one versus four duration views, and generic shapes versus analytic spectrogram training.</p><p>One labeled reference event per target · No borrowed HyperNet weights · Reserved real images remain closed</p><p><a href="#results">Real results</a> · <a href="#attribution">What caused the change?</a> · <a href="#shapes">Synthetic tests</a> · <a href="#limits">Limits and evidence</a></p></header>'
 doc+='<section><h2>What changed</h2><p>The original inputs were verified against official catalog URLs as 1-second views. The temporal arm uses genuine 0.5, 1, 2 and 4-second views of exactly the same events. These are more observed time samples, not image augmentations: the reference is still one labeled event, but now four PNGs.</p><p>The training ablation keeps the 42,480-parameter CNN, three saved random initializations, objective, optimizer and 800-step budget fixed. Only the mathematical generator changes: analytic chirps, lines, scattering-inspired arches and isolated/repeated bursts over colored noise. These are approximate time–frequency power fields, not measured strain or exact Q-transform simulations.</p><div class="notice">These are previously consumed development events. There is no fresh confirmatory real-image test in this report. All settings and every seed are reported; the primary setting is unchanged after seeing results.</div>'
 if 'reference_windows.png' in assets:doc+='<h3>The same reference events across duration</h3><img src="v46_assets/reference_windows.png" alt="Four durations of each of the four reference events">'
 if 'gallery.png' in assets:doc+='<h3>Mathematical training examples</h3><img src="v46_assets/gallery.png" alt="Procedural training reference, positive and structural hard negatives">'
 doc+='</section>'
 doc+='<section><h2>Common-event data audit</h2><p>'+str(acquisition['complete_events'])+' of the original 8,251 consumed events have all four valid native PNGs; '+str(acquisition['excluded_events'])+' incomplete events are excluded from every arm under the predeclared rule. All four active references and all original negative FIT members are retained. The 1-second arrays are bit-for-bit copies of the corresponding earlier inputs. No missing duration is stretched or synthesized.</p><p>Split counts: '+esc(json.dumps(acquisition['roles'],sort_keys=True))+'. Exact source and input-pixel duplicates across roles are checked before scoring. Acquisition failures and exclusions are linked below.</p></section>'
 doc+='<section><h2>Blank-subview exception, reviewed before scoring</h2><p>The initial automatic audit flagged 32 cross-role matches of all-zero low-frequency subviews. The affected nine No Glitch events include FIT, CAL and one DEV event; their native images, full-frequency views and complete inputs are distinct. The original failed audit is retained. A separate bound review and independent preflight admit only exactly spatially constant low-frequency subviews, with zero source, full-view or complete-input cross-role duplicates. No rows or pixels were changed, and the final audit recomputes these checks.</p></section>'
 headers=['Model / training','Duration input','CAL alpha','Balanced accuracy','MCC','Recall','FA / 10,000','pAUROC ≤1%','pAUROC ≤0.1%']
 doc+='<section id="results"><h2>Real-image development results</h2><p>Means across three paired seeds for random, prior-shape and spectrogram training; historical v37 and DINO-small are single frozen checkpoints. All models use the same events, one reference event per target, 512 negative FIT examples and separate negative CAL. Metrics are macro averages across four targets.</p>'+table(headers,primary)+'<p>Calibration alpha = 0.001 is the primary setting, not a guarantee of achieved 0.1% FPR. FA / 10,000 is the observed FP / negative-count rate rescaled to 10,000; it does not mean 10,000 independent negatives were tested. Partial AUROC is standardized (chance 0.5) and measures ranking in the low-alarm region; it is not ordinary accuracy. DINO uses matched energy inputs and ELDA here, rather than its previously selected task-specific matching pipeline.</p>'
 if 'comparison.png' in assets:doc+='<img src="v46_assets/comparison.png" alt="Paired duration and generator effects across all three seeds">'
 doc+='<details><summary>Both prespecified calibration settings</summary>'+table(headers,all_settings)+'</details></section>'
 doc+='<section id="attribution"><h2>Separate attribution checks</h2><p>Concatenation normally changes the old ELDA ridge because it changes feature dimension. Here, ridge scaling uses the single-duration dimension for both arms. Repeating the same 1-second descriptor four times preserves the kernel and is checked to preserve every calibrated development decision. The four-window arm therefore tests extra temporal information rather than that accidental regularization change.</p><h3>Duration: four windows minus one, same checkpoint</h3>'+table(['Training / model','Mean Δ pAUROC ≤1%','Each seed Δ pAUROC','Mean Δ MCC','Mean Δ recall (pp)','Mean Δ FA / 10k'],duration)+'<h3>Training: same input, paired initializations</h3>'+table(['Training contrast','Input','Mean Δ pAUROC ≤1%','Each seed Δ pAUROC','Mean Δ MCC','Mean Δ recall (pp)','Mean Δ FA / 10k'],generator)+'<h3>Does training add more duration benefit than random weights?</h3>'+table(['Trained arm','Mean difference of duration gains','Each seed difference'],interaction)+'<p>The single-window spectrogram-minus-shape contrast isolates the generator change. Combining spectrogram training and four windows changes both factors; it is reported as the factorial combination, not an unexplained architecture gain. No ensemble, reference augmentation, bigger encoder or detector-specific calibration was added.</p></section>'
 doc+='<section id="shapes"><h2>New synthetic identities and retention check</h2><p>All nine paired checkpoints were evaluated once on 256 new spectrogram identities and 256 new generic-shape identities. Each identity has four dependent structural edits. These tests measure same-instance/counterfactual ranking: the same latent shape under nuisance variation must outrank its structurally edited versions. They do not test recognition of a different instance from the same category. Both sets use familiar rendering grammars with new latent identities; neither establishes new semantic-family or real-domain generalization.</p>'+table(['Test grammar','Training','Readout','Mean pair ranking','All three seed accuracies','All four edits correct'],synthetic)+'</section>'
 doc+='<section><h2>Where targets are missed and alarms arise</h2>'+table(['Model / training','Input','Target','Positive n','Negative n','Recall','FA / 10k','MCC','pAUROC ≤1%'],classrows)+'<details><summary>H1 and L1 separately, with unchanged global calibration</summary>'+table(['Model / training','Input','Target','Detector','Positive n','Negative n','Recall','FA / 10k','MCC'],detectors)+'</details><details><summary>False alarms by negative label: any of four target matchers fires</summary>'+table(['Checkpoint','Input','Negative class','FP','n','FA / 10k'],stress)+'</details><p>Each seed scores the same events. Its counts must not be pooled as additional independent images. Per-target macro false-alarm rates are not the rate for an archive search that triggers on any target; the separate any-target table answers that question.</p></section>'
 doc+='<section id="limits"><h2>Scope and limitations</h2><p>One positive event is used for each fitted target, alongside labeled negative FIT/CAL and previously examined labeled DEV data. Four reference renders contain more context than one image. New HyperNet weights are trained only on mathematical images; DINO is an external pretrained comparison. This bounded pilot does not reproduce the historical 100× training budget.</p><p>Three seeds reveal variability but provide little evidence about its population distribution. The fixed reference may be atypical. Source-time dependence, detector/epoch distribution shifts, common rendering conventions and adaptive historical development remain limitations. Detector splits diagnose associations; they do not prove the model learned or removed a detector shortcut. No archive discovery or deployed false-alarm guarantee is claimed.</p><p>Primary sources: <a href="https://github.com/gwpy/gwpy/blob/c42f7281b1a2ca6098c1e2f7d6921a11b17f2e56/gwpy/table/gravityspy.py">GWpy duration URL mapping (pinned source)</a>; <a href="https://arxiv.org/abs/2401.12913">published multi-view Gravity Spy architecture</a>; <a href="https://arxiv.org/abs/1005.2876">chirplet methods</a>; <a href="https://arxiv.org/abs/2007.14876">scattered-light physics</a>. Temporal fusion, chirps and contrastive objectives are established ideas; this study measures their use in this constrained pipeline.</p></section>'
 doc+='<section><h2>Frozen protocols and audits</h2><p>'+' · '.join('<a href="v46_assets/'+esc(name)+'">'+esc(name)+'</a>' for name in assets)+'</p><p>Evaluation protocol SHA256<br><code>'+sha(R/'evaluation/protocol.json')+'</code></p></section><footer>Completed '+generated+' · Development only · No new real holdout consumed</footer></main></html>'
 findings='<section><h2>What the completed comparison shows</h2><p>The new mathematical generator improves real-event development matching at one-second input: recall '+pct(prior['recall']['mean'])+' → '+pct(new['recall']['mean'])+', MCC '+n(prior['mcc']['mean'])+' → '+n(new['mcc']['mean'])+', and standardized partial AUROC below 1% FPR '+n(prior['pauc_std_01']['mean'])+' → '+n(new['pauc_std_01']['mean'])+'. All three paired seeds improve on these endpoints over both prior shape training and random weights.</p><p>False alarms rise slightly versus prior shape training: '+n(prior['fa_per10000']['mean'],1)+' → '+n(new['fa_per10000']['mean'],1)+' per 10,000 negatives. Four windows add little overall to the new model: recall becomes '+pct(multi_new['recall']['mean'])+'. They help Scattered Light while hurting Low Frequency Bursts in every seed.</p><p>The stricter secondary setting still favors prior shape training with four windows: recall '+pct(strict_prior['recall']['mean'])+' at '+n(strict_prior['fa_per10000']['mean'],1)+' false alarms per 10,000, versus '+pct(strict_new['recall']['mean'])+' at '+n(strict_new['fa_per10000']['mean'],1)+' for spectrogram training. This is useful generator transfer on consumed development data, not universal low-tail superiority or a final model promotion.</p></section>'
 union_html='<section><h2>What if any target can trigger an alarm?</h2><p>The per-target macro rate above is not the alarm rate of a search running all four target matchers. This table counts each background event once if any matcher fires, using the 641 DEV events outside all four target classes. Repeating the same events across seeds does not increase the sample size.</p>'+table(['Model / training','Input','FP in each seed/checkpoint','Same background n','Mean any-target FA / 10k'],union_summary)+'</section>'
 hypothesis='<section><h2>Next hypothesis, not a tested explanation</h2><p>The generator trains exact-instance correspondence and discrimination under nuisance changes. Real category recognition must accept different instances. Some cross-instance contrastive negatives share a procedural family; the 1-second renderer also only approximates the frequency and timescale ranges of real glitches. The linked review proposes a future controlled family-aware objective test, while preserving defining relationships between parts. No such intervention was made in this study.</p></section>'
 assert doc.count('<section><h2>What changed')==1 and doc.count('<section id="limits">')==1
 doc=doc.replace('<section><h2>What changed',findings+'<section><h2>What changed',1)
 doc=doc.replace('<section id="limits">',union_html+hypothesis+'<section id="limits">',1)
 path=WEB/'v46_context.html';tmp=path.with_suffix('.tmp');tmp.write_text(doc,encoding='utf-8');tmp.replace(path)
 save(R/'report_summary.json',dict(complete=True,generated_utc=generated,report=str(path),report_sha256=sha(path),url='http://127.0.0.1:8800/research/v46_context.html',evaluation_results_sha256=sha(R/'evaluation/results.json'),synthetic_results_sha256=sha(R/'spectrogram/synthetic_results.json'),audits_passed=True,no_new_real_holdout=True,mean_duration_gain_spectrogram_pauc01=float(np.mean(gain)),mean_generator_gain_single_pauc01=float(np.mean(train_gain))))
 print('V46 REPORT COMPLETE',path,flush=True)
if __name__=='__main__':main()
