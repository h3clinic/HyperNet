"""Publish every frozen development comparison after the independent audit."""
from core import *
import html,shutil
from datetime import datetime,timezone
from collections import defaultdict
WEB=R.parent/'hypernet_v20/research';ASSETS=WEB/'v47_assets'
NAMES={'random':'Random HyperNet','shapes':'Prior shape-trained HyperNet','spectrogram':'Spectrogram-trained HyperNet','historical':'Old HyperNet v37','external':'DINOv2-small · matched inputs'}
METHOD={'single1s':'1-second control','joint4':'Joint four-window control','late_mean4':'Separate views → mean','late_max4':'Separate views → max'}

def esc(v):return html.escape(str(v),quote=True)
def f(v,d=3):return '—' if v is None else f'{v:.{d}f}'
def pct(v):return f'{100*v:.2f}%'
def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def avg(rows,k):return float(np.mean([r[k] for r in rows]))
def aggregate(rows,keys,fields):
 groups=defaultdict(list)
 for row in rows:groups[tuple(row[k] for k in keys)].append(row)
 return [dict(zip(keys,key))|{k:avg(vals,k) for k in fields}|{'n_models':len(vals)} for key,vals in groups.items()]
def find(rows,**kw):return [r for r in rows if all(r[k]==v for k,v in kw.items())]

def main():
 p=read(R/'protocol.json');r=read(R/'evaluation/results.json');a=read(R/'evaluation/audit.json');s=read(R/'splits.json')
 assert a['passed'] and r['complete'] and r['reserve_closed'] and r['development_only'] and r['no_selection']
 assert a['protocol_sha256']==r['protocol_sha256']==sha(R/'protocol.json')
 assert a['results_sha256']==sha(R/'evaluation/results.json') and a['audit_code_sha256']==sha(R/'audit.py')
 bindings(p['files']);bindings(r['files'])
 uncertainty=read(R/'uncertainty.json');assert uncertainty['complete'] and uncertainty['post_hoc'] and uncertainty['protocol_sha256']==sha(R/'protocol.json');bindings(uncertainty['source_hashes'])
 keys=['training','matcher','pool','grouping','alpha']
 macros=aggregate(r['macro'],keys,FIELDS)
 targets=aggregate(r['rows'],keys+['target'],FIELDS+['tp','fp','positive','negative'])
 unions=aggregate(r['any_target_false_alarm'],keys+['negative_label'],['fp','negative','fa_per10000'])
 primary=find(macros,training='spectrogram',alpha=.001)
 best=max(primary,key=lambda x:x['balanced_accuracy'])
 single=find(primary,matcher='single1s',pool='original',grouping='pooled')[0]
 baseline_lfb=find(targets,training='spectrogram',alpha=.001,matcher='single1s',pool='original',grouping='pooled',target='Low_Frequency_Burst')[0]['recall']
 goals=[]
 for q in primary:
  setting={k:q[k] for k in ['training','matcher','pool','grouping','alpha']}
  rr=find(r['rows'],**setting)
  lfb=avg([v for v in rr if v['target']=='Low_Frequency_Burst'],'recall')
  goals.append(setting|dict(balanced_accuracy=q['balanced_accuracy'],lfb_recall=lfb,each_seed_target_fa_max=max(v['fa_per10000'] for v in rr),passed=q['balanced_accuracy']>=.9 and max(v['fa_per10000'] for v in rr)<=10 and lfb>=baseline_lfb))
 summary=dict(protocol_sha256=sha(R/'protocol.json'),results_sha256=sha(R/'evaluation/results.json'),audit_sha256=sha(R/'evaluation/audit.json'),uncertainty_sha256=sha(R/'uncertainty.json'),best_observed_primary_development_balanced_accuracy=best,baseline_single=single,target_met=any(x['passed'] for x in goals),goal_checks=goals,macro_seed_means=macros,target_seed_means=targets,union_seed_means=unions,seed_means_are_not_ensembles=True)
 save(R/'report_summary.json',summary)
 ASSETS.mkdir(parents=True,exist_ok=True)
 downloads=[]
 for name,path in [('protocol.json',R/'protocol.json'),('results.json',R/'evaluation/results.json'),('audit.json',R/'evaluation/audit.json'),('summary.json',R/'report_summary.json'),('checks.json',R/'checks.json'),('research.json',R/'live_research.json'),('method_review.json',R/'method_review.json'),('protocol_review.json',R/'protocol_review.json'),('splits.json',R/'splits.json'),('core.py',R/'core.py'),('evaluate.py',R/'evaluate.py'),('audit.py',R/'audit.py')]:
  shutil.copy2(path,ASSETS/name);downloads.append(f'<a href="v47_assets/{name}">{name}</a>')
 for name in ['uncertainty.py','uncertainty.json']:
  shutil.copy2(R/name,ASSETS/name);downloads.append(f'<a href="v47_assets/{name}">{name}</a>')
 uncertainty_rows=[]
 for contrast in uncertainty['contrasts']:
  for metric,value in contrast['metrics'].items():
   multiplier=100 if metric in ['balanced_accuracy','recall'] else 1
   unit='percentage points' if multiplier==100 else ('per 10,000' if metric=='fa_per10000' else 'MCC')
   uncertainty_rows.append([esc(contrast['id']),metric,f(value['difference']*multiplier,3),f(value['percentile_95'][0]*multiplier,3)+' to '+f(value['percentile_95'][1]*multiplier,3),unit])
 # Exportable scientific figure; fixed settings, not best-on-DEV selection.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
 arms=['random','shapes','spectrogram','historical','external'];colors=['#8a939d','#b18443','#007f78','#8f55a0','#326fc0']
 for ax,metric,title in zip(axes,['balanced_accuracy','recall','fa_per10000'],['Balanced accuracy','Recall','False alarms / 10,000']):
  for arm,c in zip(arms,colors):
   vals=[find(macros,training=arm,matcher=m,pool='expanded',grouping='pooled',alpha=.001)[0][metric] for m in MATCHERS]
   ax.plot(range(4),vals,'o-',color=c,label=NAMES[arm].replace(' · matched inputs',''),markersize=4)
  ax.set_xticks(range(4),['1 sec','Joint 4','Mean 4','Max 4']);ax.set_title(title);ax.grid(alpha=.15)
  if metric=='balanced_accuracy':ax.axhline(.9,color='black',ls=':',lw=1,label='BA target .90');ax.set_ylim(.45,1)
  elif metric=='recall':ax.set_ylim(0,1)
  else:ax.axhline(10,color='black',ls=':',lw=1)
 fig.suptitle('Fixed expanded / pooled calibration · α = .001 · development events only',fontsize=11)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
 fig.savefig(ASSETS/'comparison.png',dpi=170);plt.close(fig)
 fixed=[]
 for method,pool,group in [('single1s','original','pooled'),('joint4','original','pooled'),('late_mean4','original','pooled'),('late_mean4','expanded','pooled'),('late_mean4','expanded','detector'),('late_max4','expanded','pooled')]:
  v=find(primary,matcher=method,pool=pool,grouping=group)[0]
  lfb=find(targets,training='spectrogram',matcher=method,pool=pool,grouping=group,alpha=.001,target='Low_Frequency_Burst')[0]
  u=find(unions,training='spectrogram',matcher=method,pool=pool,grouping=group,alpha=.001,negative_label='ALL_NON_TARGET')[0]
  fixed.append([METHOD[method],pool,group,pct(v['balanced_accuracy']),f(v['mcc']),pct(v['recall']),f(v['fa_per10000'],1),pct(lfb['recall']),f(u['fp'],2)+' / 641'])
 grid=[]
 for group in GROUPINGS:
  for pool in POOLS:
   for method in MATCHERS:
    v=find(primary,matcher=method,pool=pool,grouping=group)[0];lfb=find(targets,training='spectrogram',matcher=method,pool=pool,grouping=group,alpha=.001,target='Low_Frequency_Burst')[0]
    grid.append([METHOD[method],pool,group,pct(v['balanced_accuracy']),f(v['mcc']),pct(v['recall']),f(v['fa_per10000'],1),pct(lfb['recall']),f(v['pauc_std_01'])])
 refs=read(R/'live_research.json')
 research='''<ul><li><a href="https://openreview.net/forum?id=b2ubx9iWQ-z">CrossTransformers</a>: preserve query–support spatial evidence. Our per-view ELDA fusion is inspired by separating evidence, not a reproduction of its attention architecture.</li><li><a href="https://openreview.net/forum?id=7uFbs68MSI">Adaptive Conformal Anomaly Detection</a>: calibrate an anomaly score and inspect shift assumptions. The OpenReview author correction changes a theorem-bound sign; its repository's test-label oracle threshold is not used here.</li><li><a href="https://openreview.net/forum?id=I614HtAuvJ">Calibration Selection Breaks Conformal FDR Control</a>: calibration selection changes the population being controlled. We retain all original CAL and disclose the added mixture.</li><li><a href="https://openreview.net/forum?id=33XGfHLtZg">Conformal Risk Control</a>: distinguish expectation guarantees from a deployment guarantee. Our cross-era sample does not establish exchangeability.</li></ul>'''
 details=''
 for alpha in ALPHAS:
  dr=[]
  for group in GROUPINGS:
   for pool in POOLS:
    for method in MATCHERS:
     for arm in arms:
      v=find(macros,training=arm,matcher=method,pool=pool,grouping=group,alpha=alpha)[0]
      dr.append([NAMES[arm],METHOD[method],pool,group,pct(v['balanced_accuracy']),f(v['mcc']),pct(v['recall']),f(v['fa_per10000'],1),f(v['auroc']),f(v['pauc_std_01'])])
  details+=f'<details><summary>All 16 configurations × 5 model families · α={alpha}</summary>'+table(['Encoder','Matching','CAL','Grouping','BA','MCC','Recall','FA/10k','AUROC','pAUC≤.01'],dr)+'</details>'
 payload={'rows':r['rows'],'models':p['models'],'unions':r['any_target_false_alarm']}
 (ASSETS/'interactive.json').write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
 headline=f'Best observed BA: {pct(best["balanced_accuracy"])}. '+('The fixed target was met.' if summary['target_met'] else 'The 90% low-alarm target was not met.')
 body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v47 · four-window matching</title>
 <style>body{{font:16px/1.55 system-ui;margin:0;color:#172b35;background:#f3f6f7}}main{{max-width:1220px;margin:auto;padding:36px 24px}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:38px}}p{{max-width:1000px}}a{{color:#006f6c}}.tag{{font-size:13px;color:#49626b;text-transform:uppercase;letter-spacing:.08em}}.banner{{padding:20px;background:#fff1d6;border-left:5px solid #b87700}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}.card{{padding:18px;background:white;border-radius:10px}}.card b{{font-size:26px;display:block}}.scroll{{overflow-x:auto;background:white;border:1px solid #dfe7ea;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 13px;text-align:left;border-bottom:1px solid #e4eaed;white-space:nowrap}}th{{background:#e5efef}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}details{{margin:20px 0}}summary{{cursor:pointer;padding:14px;background:#e5efef}}label{{display:inline-flex;gap:6px;align-items:center;margin:0 14px 14px 0}}select{{font:inherit;padding:5px}}.small{{font-size:13px;color:#52656d}}.downloads a{{display:inline-block;margin:4px 14px 4px 0}}.formula{{padding:20px;background:#e5efef;overflow-wrap:anywhere}}</style>
 <main><div class="tag">HyperNet v47 · frozen development experiment · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div><h1>Keep the short signal.<br>Calibrate the combined evidence.</h1>
 <p class="banner"><strong>{headline}</strong><br>This is real Gravity Spy development data, repeatedly used during research. The 195 reserved images remain closed. The best observed configuration is descriptive; it is not a fresh-test result or a promoted model.</p>
 <div class="cards"><div class="card"><b>1,558</b>development events</div><div class="card"><b>1 event</b>positive reference per target</div><div class="card"><b>11 × 16</b>encoders × fixed configurations</div><div class="card"><b>42,480</b>HyperNet parameters</div><div class="card"><b>195</b>reserved images unopened</div></div>
 <h2>What changed</h2><p>In the previous joint descriptor, longer-window scores often opposed a valid Low Frequency Burst match. We now fit the same negative-background ELDA separately for 0.5, 1, 2 and 4 seconds, normalize each score using a separate TUNE split, then combine by a fixed mean or maximum. A final calibration step learns the cutoff for the combined score, including the extra chance of a high maximum. We separately test a broader CAL pool and H1/L1-specific cutoffs.</p>
 <div class="formula">w<sub>d</sub> = (0.5Σ<sub>d</sub> + λ<sub>d</sub>I)<sup>−1</sup>(r<sub>d</sub> − μ<sub>d</sub>) &nbsp;→&nbsp; s<sub>d</sub>(x) = w<sub>d</sub><sup>T</sup>(x<sub>d</sub> − μ<sub>d</sub>)<br>z<sub>d</sub> = (s<sub>d</sub> − median<sub>TUNE</sub>)/scale<sub>TUNE</sub> &nbsp;→&nbsp; S = mean(z) or max(z)<br>p(x) = (1 + #{{negative CAL scores ≥ S(x)}})/(n + 1); alarm only when p ≤ α.</div>
 <p class="small">Scale = 1.4826 × MAD; zero-MAD fallback is sample standard deviation (ddof=1), then 1. TUNE does not include the current target. This is a fixed matcher change; no encoder was retrained and no pretrained weights entered HyperNet.</p>
 <h2>Key fixed comparisons · spectrogram-trained HyperNet</h2><p>α=.001 means a nominal 10 alarms per 10,000 negatives under the calibration assumptions. The achieved rate below is measured on development events. Three-seed means are averages of separate models, not an ensemble or three independent datasets.</p>
 {table(['Matching','CAL pool','Grouping','Balanced accuracy','MCC','Recall','FA/10k','LFB recall','Any-target FP'],fixed)}
 <p>The clearest alarm tradeoff is expanded-CAL mean fusion: detector-specific thresholds reduce macro false alarms from 21.92 to 14.38 per 10,000, while balanced accuracy changes from 66.94% to 66.71%. Low Frequency Burst recall improves from 34.71% to 38.41%, still short of the single-window 47.05%. Broader calibration alone often loses recall; it is not a universal improvement.</p>
 <p>At a false-positive rate of 0.1%, 90% balanced accuracy requires about <strong>80.1% recall</strong>. Our best observed setting has 36.84% recall. An AUROC near 95% does not close that gap: the remaining challenge is distinguishing real targets from the most similar negatives at the extreme score tail.</p>
 <img src="v47_assets/comparison.png" alt="Fixed expanded pooled comparisons for HyperNet controls and DINO on balanced accuracy, recall and false alarms">
 <h2>How certain is the change?</h2><p>The small mean-fusion gain is not clearly separated from zero: +0.44 percentage points in balanced accuracy, with a paired 95% interval from −0.36 to +1.24. Two of three seeds lose BA against their single-window control. Detector calibration gives a more consistent alarm reduction, but it is still a development finding.</p>
 <details><summary>Paired uncertainty · post hoc analysis</summary>{table(['Contrast (left minus right)','Metric','Difference','95% percentile interval','Unit'],uncertainty_rows)}<p>2,000 shared event-group bootstrap samples, fixed seed 47051; the same draws pair all three model seeds and all four targets. These unadjusted intervals were added after inspecting results. They condition on fixed FIT, CAL, references and trained seeds; they do not account for historical development selection, broader temporal dependence or multiple comparisons, and are not confirmatory evidence.</p></details>
 <h2>All matching and calibration changes</h2><p>No setting was changed between the two predeclared execution rounds. Both pooled and detector-specific rounds, including regressions, are reported.</p>{table(['Matching','CAL pool','Grouping','BA','MCC','Recall','FA/10k','LFB recall','pAUC≤.01'],grid)}
 <h2>Inspect every model and target</h2><p>Exact TP/FP counts are preferable to interpreting rounded percentages. Use the controls to compare identical settings for HyperNet, random weights, old models and DINO.</p>
 <div id="controls"></div><div id="interactive">Loading saved audited results…</div>
 <h2>Full model-family comparison</h2>{details}
 <h2>What these measurements mean</h2><ul><li>Balanced accuracy = (recall + specificity)/2. MCC uses all four confusion counts. AUROC is ranking quality, not accuracy at the chosen low-alarm cutoff.</li><li>FA/10,000 is the observed false-positive fraction ×10,000. We have only 1,315–1,351 negatives per target and 641 background events outside all four targets. One false alarm is already about 7.4–7.6 per 10,000 for a target. Zero false alarms would still allow roughly 22–23 per 10,000 at the one-sided 95% binomial upper bound, even under independent events.</li><li>Macro averages give equal weight to four targets. Any-target union counts an event once if any target alarms; it is not the mean of target false-alarm rates.</li><li>Detector-mode AUROC/pAUC ranks by −conditional p-value; pooled mode ranks by raw score. Differences therefore include calibration and discrete ties, and cannot be attributed solely to learned features.</li><li>DINOv2-small has 22,056,576 parameters and uses external pretraining. It receives the same reference, revised inputs and calibration here. This is a matched-input baseline, not a claim to have beaten every DINO configuration.</li></ul>
 <h2>Data boundaries and limits</h2><p>Each target uses the same 512 original negative FIT events and one positive reference event. The four-window method sees four PNGs of that event. There are 474 separate TUNE events, 1,049 CAL_ADD events, 4,608 original CAL events, and 1,558 development events. Target-positive examples are excluded from that target’s FIT/TUNE/CAL. This is one positive event plus labeled negative data, not learning from one total real image.</p>
 <p>Expanded calibration changes the label/detector/era mixture; it is not merely more precise estimation of the original population. CAL and development differ substantially by observing era and detector. Exchangeability is not established, so nominal α is not a certified deployment false-alarm rate. These historically consumed development images cannot establish generalization; the untouched reserve was not used. There are no confirmed discoveries in this report.</p>
 <h2>Live paper and implementation review</h2><p>OpenReview pages and public discussion were visited directly in the sidebrowser. Commit-pinned code review covered the Gravity Spy multi-view implementation, CrossTransformers, DeepEMD, SupContrast and conformal evaluation. The changes above use their relevant principles without claiming to reproduce architectures that were not implemented.</p>{research}
 <h2>Verification and reproducibility</h2><p>Outcome-free adversarial tests checked ties, finite-sample boundaries, detector isolation, view-axis fusion, batch invariance and a dense ELDA solve. A second implementation independently checked 176 fitted states, 1,408 target result rows, every saved score, thresholds, detector groups, union alarms and source/split hashes. Source IDs, event groups and same-detector time neighbors were checked against the closed reserve.</p>
 <p class="small">Protocol SHA256 <code>{sha(R/'protocol.json')}</code><br>Results SHA256 <code>{sha(R/'evaluation/results.json')}</code><br>Audit SHA256 <code>{sha(R/'evaluation/audit.json')}</code></p><div class="downloads">{''.join(downloads)}</div>
 <p><a href="v46_context.html">Previous four-window study</a></p></main>
 <script>
 fetch('v47_assets/interactive.json').then(r=>r.json()).then(data=>{{
 const fields={{model:data.models.map(x=>x.id),matcher:['single1s','joint4','late_mean4','late_max4'],pool:['original','expanded'],grouping:['pooled','detector'],alpha:[.001,.0005]}};
 const defaults={{model:'s45011_spectrogram',matcher:'late_mean4',pool:'expanded',grouping:'pooled',alpha:.001}};
 const controls=document.querySelector('#controls');for(const [key,values] of Object.entries(fields)){{let lab=document.createElement('label');lab.textContent=key+' ';let sel=document.createElement('select');sel.id='f_'+key;for(const value of values){{let op=document.createElement('option');op.value=value;op.textContent=value;op.selected=value===defaults[key];sel.append(op)}}sel.onchange=render;lab.append(sel);controls.append(lab)}}
 function render(){{let ok=r=>Object.keys(fields).every(k=>String(r[k])===document.querySelector('#f_'+k).value);let rows=data.rows.filter(ok);let head=['Target','TP / positives','FP / negatives','Balanced accuracy','MCC','Recall','FA/10k'];let html='<div class="scroll"><table><tr>'+head.map(x=>'<th>'+x+'</th>').join('')+'</tr>';for(let r of rows){{let vals=[r.target,r.tp+' / '+r.positive,r.fp+' / '+r.negative,(r.balanced_accuracy*100).toFixed(2)+'%',r.mcc.toFixed(3),(r.recall*100).toFixed(2)+'%',r.fa_per10000.toFixed(2)];html+='<tr>'+vals.map(x=>'<td>'+x+'</td>').join('')+'</tr>'}}html+='</table></div>';let u=data.unions.filter(x=>ok(x)&&x.negative_label==='ALL_NON_TARGET')[0];html+='<p>Any-target background alarms: <b>'+u.fp+' / '+u.negative+'</b> ('+u.fa_per10000.toFixed(1)+' per 10,000).</p>';document.querySelector('#interactive').innerHTML=html}}render();
 }}).catch(e=>document.querySelector('#interactive').textContent='Results could not load: '+e.message);
 </script></html>'''
 path=WEB/'v47_matching.html';path.write_text(body,encoding='utf-8')
 save(R/'publication.json',dict(complete=True,report=str(path),report_sha256=sha(path),summary_sha256=sha(R/'report_summary.json'),audit_sha256=sha(R/'evaluation/audit.json'),no_reserved_images_opened=True))
 print(json.dumps({'url':'http://127.0.0.1:8800/research/v47_matching.html','headline':headline,'best_observed':best},indent=2))

if __name__=='__main__':main()
