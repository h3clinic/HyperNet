"""Publish completed frozen result tables only; never infer, fit or open labels."""
from pathlib import Path
from datetime import datetime,timezone
import collections,csv,hashlib,html,json,shutil

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[1]/'outputs/hypernet_v20/research/oncenet_confirmation'
MODELS=['oncenet','random','dino']
NAMES={'oncenet':'OnceNet','random':'Random-weight control','dino':'DINOv2-small'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(x):return html.escape(str(x))
def num(x,d=3,scale=1):return 'Not estimable' if x is None else f'{x*scale:.{d}f}'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def main():
    path=ROOT/'results.json'
    if not path.exists():
        print('Waiting for completed results.json; no labels or scores opened.');return
    result=read(path)
    assert result.get('complete') and result.get('confirmation_consumed')
    assert result.get('no_tuning') and result.get('no_refit') and result.get('no_training')
    sources=[path,ROOT/'protocol.json',ROOT/'scores_commitment.json',ROOT/'frozen_thresholds.json']
    protocol=read(sources[1]);commitment=read(sources[2])
    assert result['protocol_sha256']==sha(sources[1])==commitment['protocol_sha256']
    assert result['score_commitment_sha256']==sha(sources[2])
    assert result['code_sha256']==protocol['code_sha256']
    assert commitment['complete'] and commitment['label_metrics_not_yet_computed']
    assert commitment['thresholds_sha256']==sha(sources[3])
    for p,digest in commitment['files'].items():assert sha(p)==digest
    assert commitment['committed_utc']<=result['completed_utc']
    source_hashes={str(p):sha(p) for p in sources}
    rows=result['per_target'];macros=result['macro'];unions=result['non_target_union']
    assert len({(r['model'],r['alpha']) for r in macros})==6
    assert all(r['tp']+r['fn']==r['positive'] and r['fp']+r['tn']==r['negative'] for r in rows)
    summary={'result_sha256':sha(path),'protocol_sha256':result['protocol_sha256'],
        'intent_count':result['intent_count'],'eligible_count':result['eligible_count'],
        'excluded_count':len(result['excluded']),'primary':[r for r in macros if r['alpha']==.001],
        'secondary':[r for r in macros if r['alpha']==.01],
        'primary_per_target':[r for r in rows if r['alpha']==.001],
        'primary_union':[r for r in unions if r['alpha']==.001],
        'paired_primary_intervals':result['paired_primary_intervals'],
        'bootstrap':result['bootstrap'],'limitations':result['limitations'],
        'scores_committed_before_label_metrics':True,'new_inference_by_publisher':False}
    h=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
       '<title>OnceNet frozen confirmation results</title>',
       '<style>body{font:17px/1.6 Georgia,serif;color:#172936;background:#f7f8fa;margin:0}main{max-width:1080px;margin:30px auto;padding:30px;background:white}h1,h2,h3{line-height:1.25}h2{margin-top:32px}.notice{border-left:4px solid #426f83;padding:12px 18px;background:#edf4f7}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font:15px/1.45 Arial,sans-serif}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #d6dde3;white-space:nowrap}th{background:#edf1f4}small{font-size:13px}a{color:#155c86}li{margin-bottom:8px}code{overflow-wrap:anywhere}@media(max-width:650px){main{margin:0;padding:18px}}</style><main>',
       '<h1>OnceNet: frozen confirmation results</h1>',
       '<p class="notice">One positive reference event per target; pre-existing negative fitting and calibration sets. Frozen models and calibration rules were applied once. All model scores were committed before labels were joined for test metrics. These are recognition results, not discoveries.</p>',
       '<p>'+esc(result['eligible_count'])+' eligible events were evaluated from '+esc(result['intent_count'])+' intended events. '+esc(len(result['excluded']))+' were excluded under the acquisition and input-validation rules. Development results are not pooled with this cohort.</p>']
    for alpha,title in [(.001,'Primary: nominal alpha 0.001'),(.01,'Secondary: nominal alpha 0.01')]:
        h.append('<h2>'+title+'</h2><p>Macro averages give each of four target searches equal weight. The observed false-alarm rate is measured; the nominal setting is not a guaranteed achieved rate.</p>')
        rr=[next(r for r in macros if r['model']==model and r['alpha']==alpha) for model in MODELS]
        h.append(table(['Model','Balanced accuracy %','MCC','Recall %','FA / 10,000 negatives','AUROC','pAUC ≤0.001','pAUC ≤0.01'],
            [[NAMES[r['model']],num(r['balanced_accuracy'],2,100),num(r['mcc']),num(r['recall'],2,100),num(r['fa_per10000'],2),num(r['auroc'],4),num(r['pauc_std_001'],4),num(r['pauc_std_01'],4)] for r in rr]))
    h.append('<h2>Primary counts for each target</h2><p>TP = recovered targets; FP = negative events that raised alarms. Confidence limits below concern each target’s false-positive proportion, without assuming it equals the nominal setting.</p>')
    rr=[r for model in MODELS for r in rows if r['model']==model and r['alpha']==.001]
    h.append(table(['Model','Target','Positive','Negative','TP','FP','Recall %','FA /10k','FPR 95% interval /10k'],
        [[NAMES[r['model']],r['target'].replace('_',' '),r['positive'],r['negative'],r['tp'],r['fp'],num(r['recall'],2,100),num(r['fa_per10000'],2),
          ' to '.join(num(v,2,10000) for v in r['fpr_clopper_pearson_95'])] for r in rr]))
    h.append('<h2>Searching all targets together</h2><p>These counts include only events outside all four target classes and count each event once. This is the combined archive-alarm burden, not the macro target false-alarm rate above.</p>')
    h.append(table(['Model','Alpha','Background events','Any-target false alarms','FA /10k'],
        [[NAMES[r['model']],r['alpha'],r['negative'],r['fp'],num(r['fa_per10000'],2)] for r in unions]))
    h.append('<h2>Paired primary uncertainty</h2><p>Differences are OnceNet minus the control, with paired 95% percentile intervals from detector-by-hour block resampling. A positive false-alarm difference means more false alarms. These intervals do not remove the limitations of the small cohort.</p>')
    primary={r['model']:r for r in macros if r['alpha']==.001};cirows=[]
    for control,metrics in result['paired_primary_intervals'].items():
        for metric,bounds in metrics.items():
            a=primary['oncenet'][metric];b=primary[control][metric];scale=100 if metric in ('balanced_accuracy','recall') else 1
            label=metric.replace('_',' ')+(' (percentage points)' if scale==100 else '')
            cirows.append([NAMES[control],label,num(None if a is None or b is None else a-b,3,scale),
                'Not estimable' if bounds is None else num(bounds['low'],3,scale)+' to '+num(bounds['high'],3,scale)])
    h.append(table(['Control','Metric','Difference','Paired95% interval'],cirows))
    boot=result['bootstrap'];h.append('<p>'+esc(boot['valid'])+' valid resamples of '+esc(boot['requested'])+' requested; '+esc(boot['groups'])+' temporal/detector groups. Resamples missing a target were skipped as declared, not replaced.</p>')
    h.append('<h2>Scope and limitations</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in result['limitations'])+'</ul>')
    h.append('<p>Standardized partial AUROC is a ranking measure, with a diagonal reference of0.5 and a perfect score of1. It is not recall or accuracy at the deployed threshold.</p>')
    h.append('<h2>Reproducibility</h2><p><a href="results.json">Full results JSON</a> · <a href="protocol.json">Frozen protocol</a> · <a href="scores_commitment.json">Score commitment</a> · <a href="confirmation_comparison.csv">Per-target CSV</a></p>')
    h.append('<p><small>Results SHA-256: <code>'+sha(path)+'</code><br>Completed '+esc(result['completed_utc'])+'. This publisher formats completed artifacts; it performs no model inference, refitting, training or threshold updates.</small></p></main></html>')
    (ROOT/'index.html').write_text(''.join(h),encoding='utf-8')
    (ROOT/'report_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    fields=['model','target','alpha','positive','negative','tp','fp','tn','fn','balanced_accuracy','mcc','recall','fa_per10000','auroc','pauc_std_001','pauc_std_01']
    with (ROOT/'confirmation_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    for p,digest in source_hashes.items():assert sha(p)==digest,'Completed source changed'
    DEST.mkdir(parents=True,exist_ok=True)
    names=['index.html','report_summary.json','confirmation_comparison.csv','results.json','protocol.json','scores_commitment.json']
    for name in names:shutil.copy2(ROOT/name,DEST/name)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes,'publisher_sha256':sha(__file__),
        'files':{name:{'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in names},'new_inference':False}
    (ROOT/'publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(ROOT/'publication_manifest.json',DEST/'publication_manifest.json')
    print('Published http://127.0.0.1:8800/research/oncenet_confirmation/')

if __name__=='__main__':main()
