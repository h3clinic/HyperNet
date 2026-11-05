"""Render committed factorial results; never fit, infer, or select models."""
from pathlib import Path
import csv, hashlib, html, json, shutil
import numpy as np

R=Path(__file__).resolve().parent
DEST=R.parents[1]/'outputs/hypernet_v20/research/oncenet_factorial_20261105'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
esc=lambda s:html.escape(str(s))

def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def label(training,matcher):
 return ('Math-trained' if training=='trained' else 'Random weights')+' + '+('negative-aware' if matcher=='negative_aware' else 'cosine')

def main():
 results=read(R/'results.json');protocol=read(R/'protocol.json')
 assert results['complete'] and results['protocol_sha256']==sha(R/'protocol.json')
 commitment=read(R/'score_commitment.json')
 assert commitment['protocol_sha256']==sha(R/'protocol.json')
 assert results['score_commitment_sha256']==sha(R/'score_commitment.json')
 for f,h in commitment['files'].items():assert sha(f)==h,f
 DEST.mkdir(parents=True,exist_ok=True)
 sections=[]
 p=results['primary']
 conclusions=[]
 for row in p:
  name='Mathematical pretraining' if row['contrast']=='training_main' else 'Negative-aware matching'
  ci=row['paired_group_interval975'];estimate=row['estimate']
  if ci is None:
   conclusions.append(f'{name}: point difference {estimate:+.4f}; paired interval unavailable, so no interval-based conclusion is established.')
   continue
  direction='improved' if ci['low']>0 else 'reduced' if ci['high']<0 else 'did not establish a change in'
  conclusions.append(f'{name} {direction} low-false-alarm ranking averaged over the other factor: difference {estimate:+.4f}, 97.5% paired interval [{ci["low"]:+.4f}, {ci["high"]:+.4f}].')
 sections.append('<section class="finding"><h2>What the controlled comparison shows</h2>'+''.join('<p>'+esc(t)+'</p>' for t in conclusions)+'<p>These are main effects. The four conditions and their individual effects below show whether the benefit depends on the matcher or encoder.</p></section>')
 for suffix,alpha in [('001',.001),('01',.01)]:
  rows=[]
  for cell in results['cells']:
   m={k:v['estimate'] for k,v in cell['metrics'].items()}
   configs=[r for r in results['per_configuration'] if r['training']==cell['training'] and r['matcher']==cell['matcher']]
   ops=[next(o for o in c['operating_points'] if o['alpha']==alpha) for c in configs]
   tp=sum(o['tp'] for o in ops)/len(ops);fp=sum(o['fp'] for o in ops)/len(ops)
   rows.append([label(cell['training'],cell['matcher']),f"{100*m['balanced_accuracy_alpha'+suffix]:.2f}%",f"{m['mcc_alpha'+suffix]:.3f}",f"{100*m['recall_alpha'+suffix]:.2f}%",f"{m['fa_per10000_alpha'+suffix]:.1f}",f'{tp:.2f} / {fp:.2f}',f"{m['pauc_std_001']:.4f}",f"{m['pauc_std_01']:.4f}",f"{m['auroc']:.4f}"])
  sections.append(f'<section><h2>Frozen calibration target: {alpha*100:g}% false alarms</h2>'+table(['Condition','Balanced accuracy','MCC','Recall','FA / 10,000 negatives','Mean TP / FP per class/run','pAUC ≤0.1% FPR','pAUC ≤1% FPR','AUROC'],rows)+'<p>Entries equally average four targets, three seeds, and five single-reference episodes. Counts are means across these runs, not additional independent events. Actual false alarms may exceed the calibration target under distribution shift.</p></section>')
 rows=[]
 for r in results['secondary_contrasts']:
  m=r['metrics']['pauc_std_001'];ci=m['paired_group_interval95']
  rows.append([r['contrast'].replace('_',' '),f"{m['estimate']:+.4f}",f"[{ci['low']:+.4f}, {ci['high']:+.4f}]" if ci else 'Unavailable'])
 sections.append('<section><h2>Where the effect comes from</h2>'+table(['Contrast','pAUC difference','Descriptive 95% paired interval'],rows)+'<p>These secondary intervals are unadjusted. The two main contrasts above use 97.5% intervals to account for the two prespecified primary questions. Partial AUROC is standardized: 0.5 is the random diagonal and 1 is perfect ranking. It is not accuracy.</p></section>')
 rows=[]
 for t in protocol['axis_order']['targets']:
  for cell in results['cells']:
   configs=[r for r in results['per_configuration'] if r['target']==t and r['training']==cell['training'] and r['matcher']==cell['matcher']]
   rec=[r['metrics']['recall_alpha001'] for r in configs]
   fa=[r['metrics']['fa_per10000_alpha001'] for r in configs]
   rows.append([t.replace('_',' '),label(cell['training'],cell['matcher']),f'{100*np.mean(rec):.1f}%',f'{np.mean(fa):.1f}',f'{100*min(rec):.1f}–{100*max(rec):.1f}%'])
 sections.append('<section><h2>Target and reference sensitivity at the primary alarm threshold</h2>'+table(['Target','Condition','Mean recall','Mean FA / 10,000','Recall range across 15 seed/reference runs'],rows)+'<p>The range describes sensitivity to these particular choices; it is not a confidence interval. No winning seed or reference is selected.</p></section>')
 rows=[]
 for cell in results['cells']:
  for alpha in [.001,.01]:
   selected=[r for r in results['any_target_background'] if r['training']==cell['training'] and r['matcher']==cell['matcher'] and r['alpha']==alpha]
   if selected:rows.append([label(cell['training'],cell['matcher']),f'{alpha*100:g}%',selected[0]['negative'],f"{np.mean([r['fp'] for r in selected]):.2f}",f"{np.mean([r['fa_per10000'] for r in selected]):.1f}"])
 sections.append('<section><h2>False alarms when searching for any of the four targets</h2>'+table(['Condition','Per-target calibration budget','Outside-target events','Mean false alerts per seed/reference run','Mean FA / 10,000 outside-target events'],rows)+'<p>An event is a false alert here if any target detector fires on an event labeled outside all four target classes. This combined-search rate differs from the one-versus-rest macro rates above.</p></section>')
 if (R/'detector_descriptive.json').exists():
  detector=read(R/'detector_descriptive.json')
  if 'detector_strata' not in detector: detector=detector.get('summary',detector.get('results',detector))
  rows=[]
  for stratum in detector.get('detector_strata',[]):
   if not stratum.get('available'):continue
   for training in ['random','trained']:
    selected=[r for r in stratum['per_configuration'] if r['target']=='Low_Frequency_Burst' and r['matcher']=='negative_aware' and r['training']==training]
    rows.append([stratum['detector'],stratum['target_counts']['Low_Frequency_Burst'],label(training,'negative_aware'),f"{100*np.mean([r['metrics']['recall_alpha001'] for r in selected]):.2f}%",f"{np.mean([r['metrics']['fa_per10000_alpha001'] for r in selected]):.2f}"])
  sections.append('<section><h2>Low Frequency Bursts: the gain is not uniform</h2><p>The planned Low Frequency Burst cohort contains 90 Hanford and 10 Livingston events. Training does not improve this target under negative-aware matching in either detector. The pooled training benefit comes from other targets, especially Scattered Light. These descriptive comparisons reuse the same thresholds and references.</p>'+table(['Detector','LFB positives','Condition','LFB recall','FA / 10,000 detector non-targets'],rows)+'<p>Only ten Livingston positives make that stratum particularly uncertain. <a href="detector_descriptive.json">Complete detector breakdown</a>.</p></section>')
 cohort=results['cohort'];acq=results['acquisition'];n=cohort['unique_query_events']
 sections.append('<section><h2>What was held fixed</h2><p>'+esc(f'{n:,} real O3a Gravity Spy events in {cohort["recording_groups"]:,} distinct detector-hours; {cohort["target_counts"]}; {cohort["outside_target_events"]:,} outside-target events. {acq["intent_count"]-n} of the {acq["intent_count"]:,} prespecified events were excluded by the frozen acquisition/integrity rules, with no replacements.')+'</p><p>Every condition sees the same authentic 0.5, 1, 2, and 4 second views, full and low-frequency preprocessing, and spatial-statistic descriptors. Each run uses one positive reference event per target; the five reference episodes are separate tests. Encoders have 42,480 parameters. Trained encoders use existing mathematical-only pretraining and random controls use their exact saved initial weights. No encoder receives borrowed pretrained weights.</p><p>The negative-aware matcher fits background statistics using 512 historical negative events per target. Both matchers use the same separate historical calibration negatives. This is one positive reference with background data, not one total training image. The matcher contrast measures both the matching rule and its use of negative data.</p></section>')
 hist=protocol['sampling'];negative_counts=[n-c for c in cohort['target_counts'].values()]
 limits_intro=f'Source-event and detector-hour audits exclude {hist["historical_event_ids"]:,} known historical events and {hist["historical_detector_hours"]:,} hours; exact image-hash coverage of earlier work is incomplete. There are {min(negative_counts):,}–{max(negative_counts):,} non-target events per class, providing only a few empirical steps in the 0.1% false-alarm region.'
 sections.append('<section><h2>Limits on the conclusion</h2><p>Labels are consensus votes from registered Gravity Spy volunteers (at least five consistent voters, at least 80% agreement), not expert adjudication. Metadata labels informed sampling. The test is prospective, but it is not label-blind. '+esc(limits_intro)+'</p><p>Paired intervals resample detector-hour groups and condition on these three seeds, five references, and fixed FIT/CAL data. Residual detector/time dependence and sampling imbalance remain. MCC and precision depend on class prevalence; all metrics here are conditional on this sampled cohort.</p><p>This experiment isolates training and negative-aware matching under fixed preprocessing. It does not establish preprocessing novelty, performance on every scientific archive, or new discoveries. DINO was not rerun on this cohort; older DINO comparisons are separate evidence.</p></section>')
 names=['protocol.json','protocol.sha256','results.json','cell_summary.csv','score_commitment.json','fit_commitment.json','independent_data_review.json','independent_runner_review.json','independent_preflight.json','implementation_checks.json','reference_plan.json','model_registry.json','final_audit.json','independent_result_audit.json','detector_descriptive.json','recovery.json','calibration_file_lock_failure_20261105.json']
 links=[]
 for name in names:
  if (R/name).exists():shutil.copy2(R/name,DEST/name);links.append(f'<a href="{name}">{esc(name)}</a>')
 sections.append('<section><h2>Reproducible evidence</h2><p class="links">'+' · '.join(links)+'</p><p>Protocol SHA256: <code>'+sha(R/'protocol.json')+'</code><br>Scores committed: '+esc(commitment['committed_utc'])+'<br>Analysis completed: '+esc(results['completed_utc'])+'</p></section>')
 style='body{font:17px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#152536;margin:0}main{max-width:1250px;margin:0 auto;padding:40px 24px}h1{font-size:38px;line-height:1.15}h2{font-size:24px}section{background:white;border:1px solid #dbe2e8;border-radius:12px;padding:24px;margin:24px 0}.finding{border-left:6px solid #27716f}.eyebrow{color:#45647f;letter-spacing:.08em;text-transform:uppercase;font-size:13px}.scroll{overflow-x:auto}table{border-collapse:collapse;min-width:720px;width:100%;font-size:14px}th,td{padding:12px;text-align:left;border-bottom:1px solid #dbe2e8}th{background:#edf2f7}tr:nth-child(even){background:#fafbfd}code{overflow-wrap:anywhere;font-size:12px}a{color:#006c96}.links{font-size:14px}p{max-width:100ch}'
 page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OnceNet controlled training × matcher test</title><style>'+style+'</style><main><div class="eyebrow">Frozen real-image test · 5 November 2026</div><h1>Does learning help—and does the matcher?</h1><p>Four controlled conditions separate mathematical pretraining from background-aware matching, with identical real spectrograms and one positive reference per target per run.</p>'+''.join(sections)+'</main></html>'
 (DEST/'index.html').write_text(page,encoding='utf-8')
 print(DEST/'index.html')

if __name__=='__main__':main()
