"""Post-evaluation paired analysis and honest 2D detection report; no model selection."""
from pathlib import Path
import hashlib
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ARMS=('coarse_match','micro_aware')
LABELS={'coarse_match':'Family-only match control','micro_aware':'Micro-aware match objective'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pct(value):return f'{100*value:.2f}%'


def secondary_section():
    path=ROOT/'results/analytic_detail_bypass_results.json'
    if not path.is_file():return ''
    result=read(path)
    if not result.get('completed') or not result.get('weights_and_frozen_sources_unchanged'):return ''
    if sha(ROOT/'results/analytic_detail_bypass_predictions.jsonl')!=result['predictions_sha256']:
        raise ValueError('Hybrid diagnostic predictions changed')
    rows=[]
    for split in ('dev','test'):
        m=result['metrics'][split]
        rows.append('<tr><td>'+split.upper()+'</td>'+''.join('<td>'+pct(m[k])+'</td>' for k in
            ('localized_recall_at_iou50','recall','false_positive_rate','paired_scene_ranking_accuracy','auroc'))+'</tr>')
    return '''<h2>Secondary diagnostic: CNN proposals + raw-pixel correlation</h2>
<p>The separately declared classical hybrid keeps the exact micro-aware CNN proposal boxes and replaces its learned descriptor scores with normalized grayscale pixel correlation. It searches 24 angles × 3 scales × 5 translations at 96px, using one support image and no extra weight updates. Its scoring function receives no query truth, mask, identity label or pose. The original two-CNN primary comparison above is unchanged.</p>
<table><tr><th>Split</th><th>Localized recall</th><th>Presence recall</th><th>False-positive rate</th><th>Paired ranking</th><th>AUROC</th></tr>'''+''.join(rows)+f'''</table>
<p>The hybrid orders the correct member above its matched negative in {pct(result['metrics']['test']['paired_scene_ranking_accuracy'])} of TEST episode pairs, but pooled AUROC is {pct(result['metrics']['test']['auroc'])} and calibrated localized recall is {pct(result['metrics']['test']['localized_recall_at_iou50'])}. Better within-pair ordering is not the same as reliable scene acceptance. This is a classical hybrid, not a newly learned CNN or a training-free detector: proposals still come from the trained CNN. Any gain also changes alignment and rotation search, so it cannot uniquely establish where the CNN loses information. Full-patch correlation can favor macro-outline agreement over individual minute features. It remains a synthetic micro-pattern probe, with natural-category transfer unvalidated.</p>
<p><a href="results/analytic_detail_bypass_results.json">Complete diagnostic results</a> · <a href="experiments/analytic_detail_bypass_plan.json">Original predeclaration and fixed configuration</a></p>'''


def type_present_section():
    path=ROOT/'experiments/type_present_localization_diagnostic.json'
    if not path.is_file():return ''
    diagnostic=read(path)
    rows=[]
    for arm,m in diagnostic['methods'].items():
        if sha(ROOT/'results'/f'{arm}_predictions.jsonl')!=m['source_sha256']:
            raise ValueError('Type-present diagnostic score source changed')
        name=LABELS.get(arm,'CNN proposals + classical NCC')
        rows.append(f'<tr><td>{name}</td><td>{m["correct_candidate_box"]}/{m["queries"]}</td><td>{pct(m["fraction"])}</td></tr>')
    return '''<h2>Post hoc: can it locate the broad shape when that type is present?</h2>
<p>This descriptive check ignores both micro-pattern identity and the acceptance threshold. It asks whether the already selected top box overlaps the broad shape-family candidate at IoU ≥0.5. That candidate is present in every scene, including the primary hard negatives.</p>
<table><tr><th>Method</th><th>Correct candidate boxes</th><th>Forced-choice localization</th></tr>'''+''.join(rows)+'''</table>
<p>This is type-present forced-choice localization, not absence detection, calibrated recognition, a new primary outcome or natural-photo category accuracy. It shows that broad shape localization can work more often than the fine-pattern acceptance task. The reused scenes and enrollments are dependent. This analysis was selected after seeing the main results and does not replace the failed predeclared comparison.</p>
<p><a href="experiments/type_present_localization_diagnostic.json">Post hoc diagnostic and bound score hashes</a></p>'''


def cluster_values(rows,cutoff,families):
    results={key:[] for key in ('localized_recall','recall','false_positive_rate','balanced_accuracy','paired_ranking_accuracy')}
    for family in families:
        selected=[r for r in rows if r['family_id']==family]
        pos=[r for r in selected if r['present']];neg=[r for r in selected if not r['present']]
        if len(pos)!=2 or len(neg)!=2:raise ValueError('Expected two counterbalanced episodes per TEST family')
        recall=np.mean([r['score']>=cutoff for r in pos]);fpr=np.mean([r['score']>=cutoff for r in neg])
        results['recall'].append(recall);results['false_positive_rate'].append(fpr)
        results['localized_recall'].append(np.mean([r['score']>=cutoff and r['iou']>=.5 for r in pos]))
        results['balanced_accuracy'].append(.5*(recall+1-fpr))
        pairs={}
        for r in selected:pairs.setdefault(r['episode_id'],{})[r['present']]=r['score']
        results['paired_ranking_accuracy'].append(np.mean([float(v[True]>v[False])+.5*float(v[True]==v[False]) for v in pairs.values()]))
    return {k:np.asarray(v,dtype=float) for k,v in results.items()}


def main():
    result_path=ROOT/'results/detector_results.json';results=read(result_path)
    freeze_path=ROOT/'experiments/detector_training_freeze.json';freeze=read(freeze_path)
    if not results['completed'] or not results['sources_unchanged']:raise ValueError('Require completed frozen results')
    if sha(freeze_path)!=results['freeze_sha256']:raise ValueError('Wrong freeze')
    if any(sha(ROOT/p)!=h for p,h in freeze['sources'].items()):raise ValueError('Frozen model changed')
    predictions={}
    for arm in ARMS:
        path=ROOT/'results'/f'{arm}_predictions.jsonl'
        if sha(path)!=results['arms'][arm]['predictions_sha256']:raise ValueError('Predictions changed')
        predictions[arm]=[json.loads(line) for line in path.read_text().splitlines() if line]
    tests={arm:[r for r in rows if r['split']=='test'] for arm,rows in predictions.items()}
    families=sorted({r['family_id'] for r in tests[ARMS[0]]})
    if len(families)!=128:raise ValueError('Unexpected held-out family count')
    if {r['id'] for r in tests[ARMS[0]]}!={r['id'] for r in tests[ARMS[1]]}:raise ValueError('Pairing mismatch')
    values={arm:cluster_values(tests[arm],results['arms'][arm]['metrics']['test']['threshold'],families) for arm in ARMS}
    indices=np.random.default_rng(1811).integers(0,len(families),(10000,len(families)))
    contrasts={}
    for key in values[ARMS[0]]:
        difference=values[ARMS[1]][key]-values[ARMS[0]][key]
        samples=difference[indices].mean(1);low,high=np.quantile(samples,[.025,.975])
        contrasts[key]={'difference':float(difference.mean()),'low':float(low),'high':float(high)}
    analysis={'method':'Paired TEST-family bootstrap;10000 draws,seed1811,shared128-family index matrix',
        'test_families':families,'contrasts_micro_aware_minus_coarse_match':contrasts,
        'limitations':'Post-evaluation uncertainty conditional on these models, generator and fixed DEV cutoffs. One initialization; familiar building blocks. Query pixels recur under opposite supports. No multiplicity correction for secondary metrics.',
        'results_sha256':sha(result_path),'predictions_sha256':{a:sha(ROOT/'results'/f'{a}_predictions.jsonl') for a in ARMS}}
    (ROOT/'experiments/detector_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metric_keys=[('localized_recall_at_iou50','Localized recall IoU≥0.5'),('recall','Presence recall'),('false_positive_rate','False-positive rate')]
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),constrained_layout=True)
    for ax,(key,label) in zip(axes,metric_keys):
        numbers=[100*results['arms'][a]['metrics']['test'][key] for a in ARMS]
        ax.bar([0,1],numbers,color=['#8299ac','#12628a']);ax.set_ylim(0,105)
        ax.set_xticks([0,1],['Family-only','Micro-aware']);ax.set_ylabel('TEST scenes (%)');ax.set_title(label)
        for x,v in enumerate(numbers):ax.text(x,v+1,f'{v:.2f}%',ha='center')
    fig.suptitle('v11: 2D detection on 128 held-out grammar compositions\nThresholds fixed independently on DEV at ≤5% negative-scene false positives',fontsize=13)
    fig.savefig(figures/'detector_operating_point.png',dpi=160);plt.close(fig)
    # Fixed first three TEST families, first enrollment; selection ignores scores.
    example_families=families[:3];manifest=read(ROOT/'data/shapes2d_v11/manifest.json')
    episodes={e['episode_id']:e for e in manifest['episodes']}
    lookups={a:{r['id']:r for r in tests[a]} for a in ARMS}
    fig,axes=plt.subplots(3,5,figsize=(15,10),constrained_layout=True)
    selected=[]
    for row_index,family in enumerate(example_families):
        rows=sorted([r for r in tests[ARMS[0]] if r['family_id']==family and r['enrolled_variant']==0],key=lambda r:not r['present'])
        episode=episodes[rows[0]['episode_id']]
        axes[row_index,0].imshow(Image.open(ROOT/'data/shapes2d_v11'/episode['support']['rgb_path']))
        axes[row_index,0].set_title(f'Family {family:04d}\nOne reference, pattern 0')
        for query_index,row in enumerate(rows):
            target=next(q for q in episode['queries'] if q['id']==row['id'])
            for arm_index,arm in enumerate(ARMS):
                axis=axes[row_index,1+query_index*2+arm_index];prediction=lookups[arm][row['id']]
                axis.imshow(Image.open(ROOT/'data/shapes2d_v11'/target['rgb_path']))
                cutoff=results['arms'][arm]['metrics']['test']['threshold'];accepted=prediction['score']>=cutoff
                for box,color,style in [(target.get('target_bbox_xyxy'),'#2aff94','--'),(prediction['bbox_xyxy'],'#ff922c','-')]:
                    if box:
                        axis.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor=color,linewidth=1.8,linestyle=style))
                axis.set_title(('Family-only' if arm_index==0 else 'Micro-aware')+' / '+('positive' if row['present'] else 'hard negative')+'\n'+('accept' if accepted else 'reject')+f" · score {prediction['score']:.3f}",fontsize=9)
                selected.append({'id':row['id'],'arm':arm})
        for axis in axes[row_index]:axis.axis('off')
    fig.suptitle('Actual predictions: first three TEST families, enrollment 0\nGreen dashed=true matching box; orange=model-selected box (including rejected predictions)',fontsize=14)
    fig.savefig(figures/'detector_fixed_examples.png',dpi=150);plt.close(fig)
    metrics=results['arms'];primary=contrasts['localized_recall']
    point=100*primary['difference'];ci=f"[{100*primary['low']:+.2f},{100*primary['high']:+.2f}]"
    conclusion=('The added micro-pattern objective improved the primary metric in this controlled run.' if primary['low']>0 else
                'The added micro-pattern objective reduced the primary metric in this controlled run.' if primary['high']<0 else
                'The interval includes zero; this run does not establish a primary-metric improvement.')
    table=[]
    for split in ('dev','test'):
        for arm in ARMS:
            m=metrics[arm]['metrics'][split]
            table.append('<tr><td>'+split.upper()+'</td><td>'+LABELS[arm]+'</td>'+''.join('<td>'+pct(m[k])+'</td>' for k in ('localized_recall_at_iou50','recall','false_positive_rate','proposal_recall_at_iou50','paired_scene_ranking_accuracy','auroc','single_box_ap50'))+'</tr>')
    ci_rows=''.join(f'<tr><td>{html.escape(key.replace("_"," "))}</td><td>{100*v["difference"]:+.2f}pp</td><td>[{100*v["low"]:+.2f},{100*v["high"]:+.2f}]pp</td></tr>' for key,v in contrasts.items())
    artifacts=['results/detector_results.json','experiments/detector_training_freeze.json','experiments/detector_analysis.json',
               'experiments/shapes2d_data_audit.json','research/two_dimensional_primary_sources.json']
    for candidate in sorted((ROOT/'experiments').glob('*audit*.json')):
        relative=candidate.relative_to(ROOT).as_posix()
        if relative not in artifacts:artifacts.append(relative)
    links=''.join(f'<li><a href="{p}">{p}</a></li>' for p in artifacts if (ROOT/p).exists())
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v11 —2D detection results</title><style>body{{font:17px/1.6 system-ui,sans-serif;color:#213542;background:#f3f6f8;margin:0}}main{{max-width:1180px;margin:auto;padding:36px24px}}h1{{font-size:42px;line-height:1.15}}h2{{margin-top:38px}}a{{color:#075d88}}.lead{{font-size:21px}}.note{{padding:18px;background:#fff3d6;border-left:5px solid #c78d25}}.chart{{width:100%;height:auto;background:white;margin:22px0}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #d6e2e9;text-align:left}}th{{background:#e2edf3}}.formula{{background:#e5edf3;padding:18px;font:15px/1.7 ui-monospace,monospace;white-space:pre-wrap}}.small{{font-size:13px;color:#587081}}@media(max-width:650px){{h1{{font-size:30px}}main{{padding:20px}}}}</style><main>
<p class="small">HyperNet v11 • audited procedural2D experiment • one exemplar at inference</p>
<h1>Detect the enrolled2D pattern inside a larger scene.</h1>
<p class="lead">Micro-aware localized recall: <strong>{pct(metrics['micro_aware']['metrics']['test']['localized_recall_at_iou50'])}</strong>; family-only control: <strong>{pct(metrics['coarse_match']['metrics']['test']['localized_recall_at_iou50'])}</strong>. Paired difference <strong>{point:+.2f}percentage points</strong>,95% family-bootstrap interval <strong>{ci}</strong>. {conclusion}</p>
<p class="note">This is a controlled micro-pattern detection probe using unseen combinations of familiar mathematical drawing rules in flat 2D scenes. The original goal remains recognizing any example of the same type in natural photographs; this probe is not a replacement for that goal. It is not a natural-photograph result or proof of recognizing every example of an unknown semantic type. Training uses thousands of procedural images; “one reference” describes enrollment, not the total training budget.</p>
<p><a href="two_dimensional_gallery.html">Browse every reference and scene</a> · <a href="research/two_dimensional_research.html">Research and adopted ideas</a> · <a href="https://github.com/h3clinic/HyperNet">Public source and previous experiments</a></p>
<h2>The task</h2><p>A96px reference identifies a shape composition and its internal micro-pattern. The CNN searches a256px scene containing six separated objects. The positive contains the enrolled pattern in a separately jittered macro-instance (approximately ±3% geometry variation), rather than an identical physical template. The hard negative retains the same outline, placement, colors and clutter while exchanging the orientations of two equal-area internal holes. Thus this probe deliberately rejects a near-identical object from the same broad shape family when its fine characteristic differs; that is a narrower target than natural semantic category recognition. Opposite enrollments reuse the same scene pixels with reversed labels, forcing the decision to depend on the reference.</p>
<p>There are2,048 training compositions,128 DEV and128 TEST compositions, drawn from a4,096-program universe of outline × attached-part layout × stroke profile × microprogram. The32 individual factor choices are designed priors. Whole composition keys are separated across splits; the component operators remain familiar. The dataset contains4,608 references and4,608 unique query images, forming9,216 reference/query rows through counterbalanced reuse. Every pair is inspectable in the gallery.</p>
<h2>Complete operating-point results</h2><img class="chart" src="figures/detector_operating_point.png" alt="Both controls on TEST localized recall, presence recall and false-positive rate."><div class="tablewrap"><table><tr><th>Split</th><th>Training objective</th><th>Localized recall</th><th>Presence recall</th><th>False-positive rate</th><th>Proposal recall</th><th>Paired ranking</th><th>AUROC</th><th>Single-box AP50</th></tr>{''.join(table)}</table></div>
<p>The correct target is available among learned proposals in {pct(metrics['coarse_match']['metrics']['test']['proposal_recall_at_iou50'])} of positive TEST rows for the family-only arm and {pct(metrics['micro_aware']['metrics']['test']['proposal_recall_at_iou50'])} for the micro-aware arm. The micro-aware pooled AUROC is {pct(metrics['micro_aware']['metrics']['test']['auroc'])}. These separate measures distinguish candidate availability from reference-based scoring and acceptance.</p><p>Each split has256 positive and256 negative rows. Each arm chooses its threshold using DEV negatives only, with at most5% accepted there; the TEST false-positive rate is measured, not assumed to be5%. Primary localized recall requires both acceptance and IoU≥0.5. Proposal recall is measured before support matching and shows whether the correct object was even available to the descriptor. The single-box AP50 metric ranks one selected box per scene and is not the full COCO multi-detection/multi-IoU AP.</p>
<h2>Paired uncertainty</h2><table><tr><th>Micro-aware minus family-only</th><th>Difference</th><th>95% paired interval</th></tr>{ci_rows}</table><p>Post-evaluation bootstrap uses10,000 shared draws of128 TEST families, seed1811. All four counterbalanced rows from a family stay together. Intervals are conditional on this generator, one model initialization and fixed DEV cutoffs. They do not include uncertainty from new training seeds or natural-image domains. Secondary intervals are not multiplicity-corrected.</p>
<h2>Actual detections, without selecting easy examples</h2><img class="chart" src="figures/detector_fixed_examples.png" alt="Actual predictions for the first three TEST families, with true and selected boxes shown."><p>Examples are the first three TEST family IDs with enrollment0, chosen without looking at accuracy. Orange boxes show the top model proposal even when the score is rejected; the caption explicitly states acceptance. Green dashed boxes are ground truth on positive scenes. Negative scenes have no matching target.</p>
<h2>What the CNN learns</h2><p>Both arms share the same randomly initialized CNN, scene batches and proposal training roster. A native-resolution feature branch is retained before stride2; fine and coarse maps also contribute. Region features are sampled on a24×24 grid and compressed to128 dimensions. A learned foreground map yields connected-component proposals. The single reference is encoded in four exact90° pixel rotations, without interpolation; each predicted region is scored by its maximum cosine similarity to those reference descriptors. These are derived views of the same image, not extra enrollment sources. Queries can have arbitrary rotations, so this does not provide exact rotation invariance. Query annotations are read only after inference for scoring.</p>
<div class="formula">L =0.4 L_foreground + L_balanced_match
micro-aware additionally includes candidate hard-negative match labels
and0.5 · max(0,0.08 + cosine(reference,hard negative) − cosine(reference,positive))</div>
<p>The family-only arm omits that candidate hard-negative BCE contribution and paired margin; it still receives identical all-object foreground labels, other negatives, input pixels and optimizer updates. This tests the added micro-pattern supervision as a bundle. It does not isolate BCE and ranking separately. The compact128D descriptor still discards information; no claim that every minute detail survives is warranted.</p>
<p>Per arm: {freeze['steps_per_arm']:,} optimizer updates, {freeze['reference_exposures_per_arm']:,} reference enrollment exposures, {freeze['reference_cnn_views_per_arm']:,} derived reference CNN views, {freeze['query_exposures_per_arm']:,} query exposures, {freeze['parameter_count']:,} parameters. The TRAIN pool contains4,096 unique supports plus4,096 unique query RGB images; repeated queries and rotated reference views are counted separately from unique images. Natural training images:0. Externally pretrained weights:0. Enrollment uses one support image and support box; no query label, box or mask is supplied to the detector. Final checkpoints were fixed by the training schedule, without held-out early stopping.</p>
<h2>Limitations that remain</h2><p>Objects are separated against simple backgrounds, so connected components are a favorable proposal mechanism. Touching objects, occlusion and natural texture could reduce proposal recall. The internal holes provide a controlled identity cue, not a general model of category variation. Canonical program uniqueness does not make every pair perceptually distant. The minimum nominal query micro-feature width is 4.32 pixels. The paired raster ink difference averages 0.1473% and reaches 1.8878% (23 pixels), despite equal continuous hole area and count. All 4,608 query candidates are connected; two TRAIN support images retain an isolated one-pixel rendering fragment. Finite pixels and ROI compression still constrain recognition. A2D procedural gain must be retested on fresh natural photographs before claiming progress on the original category-recognition goal.</p>
<p>Siamese matching, margin ranking, dense descriptors and compositional splits are established ideas. The engineering contribution under examination is this constrained training curriculum and controlled evaluation, not a newly discovered mathematical operation or demonstrated human-like mechanism.</p>
<h2>Evidence</h2><ul>{links}</ul><p class="small">Result SHA256:{sha(result_path)}. Frozen sources verified unchanged when generating this report. Presentation and paired analysis do not retrain or select a replacement model.</p></main></html>'''
    document=document.replace('padding:36px24px','padding:36px 24px').replace('margin:22px0','margin:22px 0')
    document=document.replace('<h2>Limitations that remain</h2>',secondary_section()+'<h2>Limitations that remain</h2>')
    document=document.replace('<h2>Limitations that remain</h2>',type_present_section()+'<h2>Limitations that remain</h2>')
    for old,new in [('flat2D','flat 2D'),('procedural2D','procedural 2D'),('enrolled2D','enrolled 2D'),('A96px','A 96px'),('a256px','a 256px'),('are2,048','are 2,048'),('compositions,128','compositions, 128'),('and128','and 128'),('a4,096','a 4,096'),('The32','The 32'),('contains4,608','contains 4,608'),('and4,608','and 4,608'),('forming9,216','forming 9,216'),('has256','has 256'),('and256','and 256'),('most5%','most 5%'),('be5%','be 5%'),('uses10,000','uses 10,000'),('of128','of 128'),('seed1811','seed 1811'),('enrollment0','enrollment 0'),('stride2','stride 2'),('a24×24','a 24×24'),('to128','to 128'),('exact90°','exact 90°'),('compact128D','compact 128D'),('contains4,096','contains 4,096'),('plus4,096','plus 4,096'),('images:0','images: 0'),('weights:0','weights: 0'),('A2D','A 2D'),('percentage points</strong>,95%',' percentage points</strong>, 95%')]:
        document=document.replace(old,new)
    (ROOT/'index.html').write_text(document,encoding='utf-8')
    print(json.dumps({'report':'index.html','primary_difference':primary,'result_sha256':sha(result_path)},indent=2))


if __name__=='__main__':main()
