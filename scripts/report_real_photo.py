"""Read-only real-photo transfer report; natural-photo figures stay local."""
from pathlib import Path
import hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ALIASES=['old_cnn','native_patch','old_cnn_grid','native_patch_grid']
LABELS={'old_cnn':'Previous CNN / learned proposals','native_patch':'Native patches / learned proposals (primary)',
 'old_cnn_grid':'Previous CNN / fixed-grid diagnostic','native_patch_grid':'Native patches / fixed-grid diagnostic'}
MODES={'transferred':'Primary: unchanged synthetic cutoff','real_dev_calibrated':'Secondary: real DEV calibration'}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pct(value):return f'{100*value:.2f}%'
def pp(value):return f'{100*value:+.2f} pp'
def verify_results(path):
    result=read(path)
    if not result['completed'] or result['gradients_created'] or not result['frozen_sources_and_inputs_unchanged'] or result['model_state_before']!=result['model_state_after']:
        raise ValueError('Completed read-only frozen evaluation required')
    if result['primary_mode']!='transferred' or result['primary_method']!='native_patch':raise ValueError('Unexpected primary')
    manifest_path=ROOT/'data/coco_real_v13/manifest.json'
    if sha(manifest_path)!=result['manifest_sha256']:raise ValueError('Manifest hash changed')
    protocols=[p for p in (ROOT/'experiments').glob('*.json') if sha(p)==result['protocol_sha256']]
    if len(protocols)!=1:raise ValueError('Expected one exact protocol')
    for name,digest in read(protocols[0])['behavior_files'].items():
        source=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(source)!=digest:raise ValueError('Frozen source changed: '+name)
    rows={}
    for alias,record in result['score_files'].items():
        source=path.parent/record['path']
        if sha(source)!=record['sha256']:raise ValueError('Score log changed')
        rows[alias]=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
    return result,read(manifest_path),rows,protocols[0]

def main():
    path=ROOT/'results/real_eval/real_results.json';r,manifest,rows,protocol=verify_results(path)
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#899db1','#167fa3','#bb9a63','#6e67b0'];x=np.arange(4)
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for row,(mode,title) in enumerate(MODES.items()):
        data=r['modes'][mode]['methods']
        for col,(metric,label) in enumerate([('localized_recall_at_iou50','Accepted localization at IoU ≥ 0.5'),('false_positive_rate','False-positive rate'),('balanced_accuracy','Presence balanced accuracy')]):
            values=[100*data[a]['test'][metric] for a in ALIASES];ax=axes[row,col]
            ax.bar(x,values,color=colors);ax.set_ylim(0,110);ax.set_yticks(range(0,101,20));ax.set_xticks(x,['CNN','Native\npatch','CNN\n+ grid','Patch\n+ grid']);ax.set_ylabel('TEST percent');ax.set_title(title+'\n'+label,fontsize=11)
            for index,value in enumerate(values):ax.text(index,value+2,f'{value:.2f}',ha='center',fontsize=9)
            if metric=='balanced_accuracy':ax.axhline(50,color='#888',linestyle=':',linewidth=1)
    fig.suptitle('Fresh natural photographs: four fixed methods, two declared operating points\nGrid methods are separate diagnostics; their proposals are not learned.',fontsize=15)
    fig.savefig(figures/'real_photo_operating_points.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,alias in zip(axes,ALIASES):
        edges=np.histogram_bin_edges([row['score'] for row in rows[alias] if row['split']=='test'],bins=20)
        for present,color,label in [(True,'#188a74','Type present'),(False,'#bf6e46','Type absent')]:
            values=[row['score'] for row in rows[alias] if row['split']=='test' and row['present']==present]
            ax.hist(values,bins=edges,alpha=.6,color=color,label=label)
        for mode,color,style in [('transferred','#173e59','--'),('real_dev_calibrated','#8c477f',':')]:
            ax.axvline(r['modes'][mode]['methods'][alias]['test']['threshold'],color=color,linestyle=style,label='Synthetic cutoff' if mode=='transferred' else 'Real DEV cutoff')
        ax.set_title(LABELS[alias].replace(' / ','\n').replace(' (primary)',''),fontsize=10);ax.set_xlabel('Score (not probability)');ax.set_ylabel('TEST scenes')
    axes[0].legend(fontsize=8);fig.savefig(figures/'real_photo_score_distributions.png',dpi=155);plt.close(fig)
    table=[]
    for mode in MODES:
        for split in ['dev','test']:
            for alias in ALIASES:
                m=r['modes'][mode]['methods'][alias][split]
                cells=[MODES[mode],split.upper(),LABELS[alias],f"{m['localized_true_positive']}/{m['positive_scenes']}",pct(m['localized_recall_at_iou50']),
                    f"{m['false_positive']}/{m['negative_scenes']} ({pct(m['false_positive_rate'])})",pct(m['recall']),pct(m['balanced_accuracy']),pct(m['proposal_recall_at_iou50']),pct(m['top1_iou_recall_at_iou50']),pct(m['auroc']),pct(m['single_box_ap50'])]
                table.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in cells)+'</tr>')
    primary=r['modes']['transferred'];p=primary['methods']['native_patch']['test'];b=primary['methods']['old_cnn']['test']
    contrast=primary['bootstrap']['localized_recall_differences']['native_patch_minus_old_cnn']
    secondary=r['modes']['real_dev_calibrated']['methods']['native_patch']['test']
    title='Real-photo category detection failed in this test.' if p['localized_true_positive']==0 else 'The real-photo transfer result.'
    failure_note=(f"<p class=\"note\"><strong>No positive TEST photograph was correctly localized and accepted.</strong> "
        f"The transferred primary rejects all {p['positive_scenes']} positive scenes. Real-photo calibration accepts "
        f"{secondary['accepted_positive']}/{secondary['positive_scenes']} positive scenes, but still yields "
        f"{secondary['localized_true_positive']} correct accepted boxes, with {secondary['false_positive']}/{secondary['negative_scenes']} false alarms. "
        "A slightly above-chance presence balanced accuracy does not establish successful object localization or recognition.</p>") if p['localized_true_positive']==0 else ''
    statement='The paired interval supports a localized-recall improvement in this cohort.' if contrast['ci95'][0]>0 else 'The paired interval does not establish a localized-recall improvement in this cohort.'
    contrast_text=(f"Both primary methods have zero accepted localizations; their observed paired difference is {pp(contrast['estimate'])}. "
        "Empirical resampling of these all-zero category outcomes is degenerate and cannot quantify an unseen-success rate or establish population equivalence.") if contrast['ci95']==[0.,0.] else (
        f"The paired primary difference is <strong>{pp(contrast['estimate'])}</strong>, 95% category-bootstrap interval "
        f"<strong>[{pp(contrast['ci95'][0])}, {pp(contrast['ci95'][1])}]</strong>. {statement}")
    ci_rows=[]
    for mode in MODES:
        for name,value in r['modes'][mode]['bootstrap']['localized_recall_differences'].items():
            ci_rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [MODES[mode],name,pp(value['estimate']),f"[{pp(value['ci95'][0])}, {pp(value['ci95'][1])}]"])+'</tr>')
    diagnostics=[]
    for ref in sorted(manifest['references'],key=lambda x:x['name']):
        records=[x for x in rows['native_patch'] if x['category_id']==ref['category_id']]
        meta=records[0]['diagnostics']['reference'];test=[x for x in records if x['split']=='test']
        metrics=primary['per_category'][str(ref['category_id'])]['native_patch']['test']
        cells=[ref['name'],str(meta['resolved_reference_foreground']),str(meta['reference_patch_count']),f"{np.mean([x['diagnostics']['cnn_proposal_count'] for x in test]):.1f}",pct(metrics['proposal_recall_at_iou50']),pct(metrics['top1_iou_recall_at_iou50']),pct(metrics['localized_recall_at_iou50']),pct(metrics['false_positive_rate'])]
        diagnostics.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    optional=[]
    for image,caption in [('assets/real_reference_contact_sheet.png','All twelve automatically selected references and their exact96px model inputs. Reference clarity and occlusion were not curated; no reference was swapped after TEST scoring. This natural-photo figure stays local.'),('assets/real_transfer_examples.png','Predetermined examples with reference and original-coordinate predictions. This local figure contains natural photographs and is not exported in the public release.'),('assets/real_photo_failure_examples.png','Descriptive error inspection after frozen scoring; this is not method-selection evidence.')]:
        if (ROOT/image).exists():optional.append(f'<figure><img src="{image}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    artifacts=[path,protocol,ROOT/'data/coco_real_v13/manifest.json']+[q for q in sorted((ROOT/'experiments').glob('*.json')) if any(word in q.name for word in ['audit','review','evidence','recovery'])]
    links=''.join(f'<li><a href="{q.relative_to(ROOT).as_posix()}">{q.relative_to(ROOT).as_posix()}</a></li>' for q in artifacts)
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v13 — real-photo transfer results</title><style>body{{font:17px/1.6 system-ui;background:#f1f6f9;color:#183546;margin:0}}main{{max-width:1380px;margin:auto;padding:36px 26px}}h1{{font-size:43px;line-height:1.15}}h2{{margin-top:34px}}a{{color:#086b92}}.lead{{font-size:23px}}.note{{background:#fff0cf;padding:20px;border-left:5px solid #bc8830}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #d7e3ed;text-align:left}}th{{background:#dde9f2}}img{{max-width:100%;height:auto}}.small,figcaption{{font-size:13px;color:#597386}}figure{{margin:24px 0}}</style><main>
<p class="small">HyperNet v13 / natural-photo category transfer / one boxed reference per category</p><h1>{title}</h1>{failure_note}
<p class="lead">At the unchanged synthetic cutoff, native-patch matching correctly localizes and accepts <strong>{p['localized_true_positive']}/{p['positive_scenes']} positives ({pct(p['localized_recall_at_iou50'])})</strong>, with <strong>{p['false_positive']}/{p['negative_scenes']} false positives ({pct(p['false_positive_rate'])})</strong>. The previous CNN reaches <strong>{pct(b['localized_recall_at_iou50'])}</strong> with <strong>{pct(b['false_positive_rate'])}</strong> false positives.</p>
<p>{contrast_text}</p>
<p class="note">The synthetic v12 result of 98.83% concerned separated 2D micro-pattern identity. This experiment asks whether another real photograph contains any instance of the reference's semantic category. Different pictures, object instances, backgrounds and appearance make this a substantially different task; the two percentages are not directly comparable benchmark scores. The model still uses its own synthetic-only CNN and analytic patch matching, with no natural-photo weight training or borrowed pretrained weights.</p>
<p><a href="photo_lab.html">Try local reference enrollment and inspect every benchmark photo</a> · <a href="https://github.com/h3clinic/HyperNet">Public code and earlier audited experiments</a></p>
<h2>Two operating points, with their data budgets kept separate</h2><p><strong>Primary:</strong> transfer the v12 synthetic DEV cutoff unchanged, with no real-photo calibration labels. <strong>Secondary:</strong> use 96 additional real DEV photos (48 negatives) to set a pooled cutoff per method, accepting at most two DEV negatives. This changes the decision threshold only. It is not a one-photo-total-data claim, and it does not train the CNN. TEST labels select neither methods nor thresholds.</p>
<img src="figures/real_photo_operating_points.png" alt="Natural-photo localization, false positives and presence balanced accuracy"><div class="tablewrap"><table><tr><th>Mode</th><th>Split</th><th>Method</th><th>Localized positives</th><th>Localized recall</th><th>False positives</th><th>Presence recall</th><th>Presence balanced accuracy</th><th>Proposal recall</th><th>Top-one IoU recall</th><th>AUROC</th><th>Single-box AP50</th></tr>{''.join(table)}</table></div>
<p>Each TEST method sees the same 96 positive and 96 negative photographs. Primary CNN and native-patch methods use exactly the same learned proposal roster. The two explicitly named grid diagnostics instead use 51 predeclared windows based only on image dimensions. They are not promoted primary methods or learned proposal improvements. A localized success must pass its cutoff and overlap any annotated same-category object at IoU ≥0.5. Presence metrics ignore box quality; proposal and top-one IoU recall ignore the cutoff. Single-box AP50 is not official COCO multi-detection, multi-IoU AP.</p>
<h2>What the scores and regions reveal</h2><p>Learned proposals cover {round(p['proposal_recall_at_iou50']*p['positive_scenes'])}/{p['positive_scenes']} positive TEST scenes at IoU ≥0.5. The native matcher selects a correct top box in {round(p['top1_iou_recall_at_iou50']*p['positive_scenes'])}/{p['positive_scenes']}, even when its cutoff is ignored. Fixed-grid proposals cover {round(primary['methods']['native_patch_grid']['test']['proposal_recall_at_iou50']*p['positive_scenes'])}/{p['positive_scenes']}, but their native matcher also needs accurate ranking; its top-one IoU recall is {pct(primary['methods']['native_patch_grid']['test']['top1_iou_recall_at_iou50'])}. Changing only the threshold cannot recover a correct box that the matcher never selects.</p><img src="figures/real_photo_score_distributions.png" alt="Positive and negative TEST scores with both predeclared cutoffs"><p>Score overlap diagnoses ranking and calibration; it does not prove a specific internal cause. The primary's foreground fitting and outline registration were built for separated high-contrast shapes. Natural photos introduce textured backgrounds, touching objects, shading, occlusion and category variation that the synthetic curriculum did not teach. These results support both proposal-coverage and matching-transfer failures; they do not isolate CNN compression as their sole cause.</p>
<div class="tablewrap"><table><tr><th>Category</th><th>Reference foreground resolved</th><th>Reference patches</th><th>Mean CNN proposals / TEST scene</th><th>Proposal recall</th><th>Top-one IoU recall</th><th>Primary localized recall</th><th>Primary false positives</th></tr>{''.join(diagnostics)}</table></div><p>The diagnostic foreground flag only means the analytic extraction returned a region; it does not certify that it isolated the correct subject. Category rows use one fixed reference each, so a poor reference can affect every corresponding query.</p>
<h2>Paired uncertainty</h2><table><tr><th>Mode</th><th>Localized-recall contrast</th><th>Difference</th><th>95% interval</th></tr>{''.join(ci_rows)}</table><p>Intervals use 10,000 paired category-cluster draws, seed 1313, keeping each category's reference-linked queries together. There are 12 category clusters. They condition on these selected references, this source dataset, fixed architecture and fixed operating cutoffs. Cutoffs are not refitted in bootstrap draws. Intervals are individual, not simultaneous multiple-comparison guarantees. With zero observed localized successes, resampling these observations gives degenerate [0,0] localized-recall intervals; that does not establish population equivalence or rule out rare successes on other images.</p>
<h2>What one reference means here</h2><p>The inherited CNN has 897,207 parameters and was trained on 8,192 distinct procedural RGB images. v13 adds zero optimizer updates. There are 12 reference photos, one per category, 96 DEV query photos and 192 TEST query photos; all 300 original sources are globally distinct. Both splits reuse the same 12 category labels and enrolled references, so this is new-photo transfer rather than evaluation on unseen categories. Enrollment extracts descriptors and local patterns; it does not perform SGD.</p>
<p>The supplied reference box is expanded 1.25× and converted to an aspect-preserving 96px letterbox. Query detail uses a maximum long edge of 1024px; CNN proposals use at most 512px with minimal padding. All reported boxes map back to original photo coordinates. Model inputs are the reference RGB, its box, and full query RGB. Query category IDs only choose the corresponding enrolled reference; query boxes, masks, target identities and truth labels do not enter prediction.</p><p><strong>Reference quality is a limitation:</strong> references were automatically selected from annotations with a minimum 32px box edge and 2% frame area, not curated for clear, frontal or unoccluded subjects. The first car reference is small and partly occluded; 96px normalization can discard more detail. Results condition on one chosen reference per category and do not establish performance for every possible clearer reference. No reference was replaced after viewing TEST results.</p>
<h2>Freshness and scope</h2><p>The roster was chosen deterministically from COCO annotations before scoring. It excludes 2,472 previously used COCO source IDs and checks raw/decoded overlap against 7,979 prior input files. Every selected image has normal EXIF orientation and its original dimensions. Prior category labels were reused in earlier project studies. Exact file and pixel disjointness does not prove different physical individuals, capture sessions or absence of near-duplicates. Annotation-absent negatives can contain missed objects. Positive and reference eligibility imposes minimum annotated object sizes, so this is a selected subset rather than the full COCO benchmark.</p>
<p>Original photograph licenses are preserved individually: 198 CC BY 2.0 and 102 CC BY-SA 2.0 sources. Source/Flickr and license links accompany each local gallery pair. Natural photographs and user uploads are excluded from public GitHub releases. Original-host TLS transport failure was resolved through the same official S3 object path with valid HTTPS; selection, roster and bytes were audited rather than silently replaced.</p>
{''.join(optional)}<h2>Local use and evidence</h2><p>The photo lab keeps uploads in request memory. It accepts one reference crop and one full query image, invokes the same fixed adapter, and reports each method separately. Its manual comparisons are not new benchmark evidence and do not update any weights. Selecting the secondary cutoff explicitly adds the previously collected real DEV calibration budget.</p><ul>{links}</ul><p class="small">Result SHA256: {sha(path)}. Report generation reads frozen outputs and creates presentation figures only. No parameters, thresholds or scored rows are changed.</p></main></html>'''
    (ROOT/'index.html').write_text(content,encoding='utf-8')
    print(json.dumps({'report':'index.html','primary':p,'paired_difference':contrast,'result_sha256':sha(path)},indent=2))
if __name__=='__main__':main()
