"""Render the frozen fresh v12 comparison; never fits models or changes cutoffs."""
from pathlib import Path
import hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
ALIASES=['old_cnn','outline_only','centroid_only','local_moments','lowres_patch','native_patch']
LABELS={'old_cnn':'Previous CNN','outline_only':'Filled outline only','centroid_only':'Local positions + sizes','local_moments':'Local shape moments','lowres_patch':'Compressed fine patches','native_patch':'Native fine patches (primary)'}
SHORT=['Old CNN','Outline','Positions','Moments','Low-res','Native']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f} pp'

def main():
    result_path=ROOT/'results/fresh_eval/fresh_results.json';r=read(result_path)
    if not r['completed'] or r['scope']!='FROZEN FRESH 2D SYNTHETIC EVALUATION' or r['gradients_created']:
        raise ValueError('Require completed fresh, read-only evaluation')
    if r['model_state_before']!=r['model_state_after']:raise ValueError('Model state changed')
    protocols=[p for p in (ROOT/'experiments').glob('*.json') if sha(p)==r['protocol_sha256']]
    if len(protocols)!=1:raise ValueError('Cannot bind exactly one frozen protocol')
    protocol_path=protocols[0];protocol=read(protocol_path)
    for name,expected in protocol['behavior_files'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(p)!=expected:raise ValueError('Frozen source changed: '+name)
    manifest_path=ROOT/'data/shapes2d_fresh_v12/manifest.json'
    if sha(manifest_path)!=r['manifest_sha256']:raise ValueError('Fresh manifest changed')
    manifest=read(manifest_path);episodes={e['episode_id']:e for e in manifest['episodes']}
    predictions={}
    for alias in ALIASES:
        info=r['score_files'][alias];path=result_path.parent/info['path']
        if sha(path)!=info['sha256']:raise ValueError('Score log changed')
        predictions[alias]=[json.loads(line) for line in path.read_text().splitlines() if line]
    test={a:r['methods'][a]['test'] for a in ALIASES};boot=r['bootstrap']
    primary=boot['differences']['native_patch_minus_old_cnn'];detail=boot['differences']['native_patch_minus_lowres_patch']
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),constrained_layout=True)
    for axis,key,label in zip(axes,['localized_recall_at_iou50','false_positive_rate','auroc'],['Accepted localization (IoU ≥0.5)','Measured false-positive rate','Pooled AUROC']):
        values=[test[a][key]*100 for a in ALIASES]
        axis.bar(range(6),values,color=['#8c9eaa']*4+['#d49b3e','#087c91'])
        axis.set_ylim(0,110);axis.set_xticks(range(6),SHORT,rotation=35,ha='right');axis.set_title(label);axis.set_ylabel('TEST (%)')
        for i,value in enumerate(values):axis.text(i,value+1,f'{value:.1f}',ha='center',fontsize=9)
    fig.suptitle('Fresh v12 compositions: identical scenes and CNN proposal boxes for every method\nEach threshold fixed on fresh DEV negatives; previous 4.30% is historical, not this comparison',fontsize=13)
    fig.savefig(figures/'fresh_operating_points.png',dpi=160);plt.close(fig)
    # Fixed roster, selected without reference to scores or correctness.
    families=sorted({x['family_id'] for x in predictions['old_cnn'] if x['split']=='test'})[:3]
    lookups={a:{x['id']:x for x in rows} for a,rows in predictions.items()}
    fig,axes=plt.subplots(3,5,figsize=(15,10),constrained_layout=True)
    for i,family in enumerate(families):
        rows=sorted([x for x in predictions['old_cnn'] if x['split']=='test' and x['family_id']==family and x['enrolled_variant']==0],key=lambda x:not x['present'])
        ep=episodes[rows[0]['episode_id']];data=manifest_path.parent
        axes[i,0].imshow(Image.open(data/ep['support']['rgb_path']));axes[i,0].set_title(f'Family {family:04d}\nOne reference, pattern 0')
        for j,row in enumerate(rows):
            q=next(q for q in ep['queries'] if q['id']==row['id'])
            for k,alias in enumerate(['old_cnn','native_patch']):
                axis=axes[i,1+2*j+k];pred=lookups[alias][row['id']]
                axis.imshow(Image.open(data/q['rgb_path']))
                for box,color,style in [(q['target_bbox_xyxy'],'#20fa95','--'),(pred['bbox_xyxy'],'#ff982e','-')]:
                    if box is not None:axis.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor=color,linestyle=style,linewidth=1.8))
                accepted=pred['score']>=test[alias]['threshold']
                axis.set_title(('Old CNN' if k==0 else 'Native patch hybrid')+' / '+('positive' if row['present'] else 'hard negative')+'\n'+('ACCEPT' if accepted else 'REJECT')+f" · score {pred['score']:.3f}",fontsize=9)
        for axis in axes[i]:axis.axis('off')
    fig.suptitle('First three fresh TEST families, enrollment 0 — selection ignores correctness\nGreen dashed = true matching box; orange = selected model box, including rejected predictions',fontsize=13)
    fig.savefig(figures/'fresh_fixed_examples.png',dpi=150);plt.close(fig)
    table=[]
    for split in ['dev','test']:
        for alias in ALIASES:
            m=r['methods'][alias][split]
            cells=[split.upper(),LABELS[alias],f"{m['localized_true_positive']}/{m['positive_scenes']}"]+[pct(m[k]) for k in ['localized_recall_at_iou50','false_positive_rate','proposal_recall_at_iou50','paired_scene_ranking_accuracy','auroc','single_box_ap50']]
            table.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
    claim='The paired interval supports an improvement in this controlled synthetic test.' if primary['ci95'][0]>0 else 'The paired interval does not establish an improvement in this controlled synthetic test.'
    ci_rows=''.join('<tr><td>'+label+'</td><td>'+pp(d['estimate'])+'</td><td>['+pp(d['ci95'][0])+', '+pp(d['ci95'][1])+']</td></tr>' for label,d in [('Primary minus previous CNN',primary),('Primary minus compressed fine evidence',detail)])
    optional=[]
    for filename,caption in [('assets/train_detail_layer_diagnosis.png','TRAIN-only stage diagnosis. Candidate boxes and known changed pixels are diagnostic controls, not inference inputs or fresh accuracy.'),('assets/fresh_patch_comparison_labeled.png','Fixed fresh example selected before results: reference-selected local evidence and actual selected query regions. Scores have different scales and are not probabilities. Both old CNN scores fall below its own frozen cutoff.'),('assets/fresh_all_errors.png','Every primary TEST error: three rejected positives and thirteen accepted negatives. This is post hoc descriptive analysis; no parameter or threshold was changed.')]:
        if (ROOT/filename).exists():optional.append(f'<figure><img src="{filename}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    artifacts=[result_path,protocol_path,manifest_path,ROOT/'research/detail_matching_primary_sources.json']
    artifacts+=sorted(p for p in (ROOT/'experiments').glob('*.json') if any(word in p.name for word in ['audit','review','evidence','interpretation']))
    links=''.join(f'<li><a href="{p.relative_to(ROOT).as_posix()}">{p.relative_to(ROOT).as_posix()}</a></li>' for p in artifacts)
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v12 — fresh 2D detail results</title><style>body{{font:17px/1.6 system-ui;color:#183546;background:#f1f6f9;margin:0}}main{{max-width:1240px;margin:auto;padding:36px 24px}}h1{{font-size:42px;line-height:1.15}}h2{{margin-top:36px}}a{{color:#086b92}}.lead{{font-size:22px}}.note{{background:#fff0cb;padding:18px;border-left:5px solid #c18b25}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #cedeeb;text-align:left}}th{{background:#dfeaf1}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption,.small{{color:#527081;font-size:13px}}.formula{{white-space:pre-wrap;padding:18px;background:#e2edf3;font:15px/1.7 ui-monospace,monospace}}</style><main>
<p class="small">HyperNet v12 • fresh held-out 2D micro-pattern detection • one support per enrollment</p>
<h1>Keep the small evidence separate from the broad outline.</h1>
<p class="lead">On the same fresh TEST scenes, native fine-patch matching reaches <strong>{pct(test['native_patch']['localized_recall_at_iou50'])}</strong> accepted localization versus <strong>{pct(test['old_cnn']['localized_recall_at_iou50'])}</strong> for the previous CNN. The paired difference is <strong>{pp(primary['estimate'])}</strong>, 95% family interval <strong>[{pp(primary['ci95'][0])}, {pp(primary['ci95'][1])}]</strong>. {claim}</p>
<p>Measured TEST false-positive rates: <strong>{pct(test['native_patch']['false_positive_rate'])}</strong> for native patches and <strong>{pct(test['old_cnn']['false_positive_rate'])}</strong> for the old CNN. The historical 4.30% result came from a different cohort; the comparison above reruns both methods on identical fresh images.</p>
<p class="note">This improvement concerns a controlled 2D micro-pattern task. The primary is a hybrid: our existing synthetic-trained CNN proposes object boxes, and an analytic local-image matcher scores them. No new fine-matching CNN weights were learned. The original goal—recognizing any natural example of a category from one reference—remains unvalidated.</p>
<p><a href="fresh_gallery.html">Browse all fresh scenes and predictions</a> · <a href="research/detail_matching_research.html">Mathematics, primary sources and implementation</a> · <a href="https://github.com/h3clinic/HyperNet">Public code and prior experiments</a></p>
<h2>Six methods, identical fresh scenes</h2><img src="figures/fresh_operating_points.png" alt="All six methods on fresh localized recall, false-positive rate and AUROC"><div class="tablewrap"><table><tr><th>Split</th><th>Method</th><th>Localized positives</th><th>Localized recall</th><th>False positives</th><th>Proposal recall</th><th>Paired ranking</th><th>AUROC</th><th>Single-box AP50</th></tr>{''.join(table)}</table></div>
<p>Each split contains 256 positive and 256 negative query rows from 128 composition families. Every method uses the exact same CNN-predicted proposal roster. A positive is localized correctly only if its top score is accepted and its selected box has IoU ≥0.5. Each threshold is fixed independently on fresh DEV negatives at no more than 5% false positives, then applied unchanged to TEST. TEST false positives are measured rather than forced to 5%. The AP50 column is a one-box-per-scene ranking metric, not full COCO multi-box/multi-IoU AP.</p>
<h2>What the controls isolate</h2><p>Filled-outline matching discards internal evidence. The position-and-size control retains matched interior-feature locations and sizes but omits covariance. The moment control additionally compares local shape covariances; it uses a more specific connected-interior-feature prior. The primary patch matcher selects interior gradient peaks from the reference alone and compares the actual local image patterns. Its compressed control uses the same native alignment, patch centers, patch count and equal weights, but downsamples/re-enlarges only fine RGB evidence by a factor of four.</p>
<table><tr><th>Paired localized-recall comparison</th><th>Difference</th><th>95% interval</th></tr>{ci_rows}</table><p>Intervals use 10,000 paired bootstrap draws of 128 TEST families, seed 1812. All four query rows and both opposite enrollments stay together. They are conditional on the selected algorithms, this generator, the existing CNN and the fixed DEV thresholds. No cutoff refitting occurs inside the bootstrap. These are individual intervals, not simultaneous familywise guarantees.</p>
<h2>The actual matching rule</h2><p>The matcher estimates foreground from RGB contrast against a perimeter background fit. Filled silhouettes determine broad rotation/scale alignment. Only then does it compare at most twelve reference-selected 7×7 interior gradient patches, each normalized independently. Five fixed local one-pixel shifts tolerate small registration errors. This keeps a few informative pixels from being overwhelmed by the area of the unchanged silhouette.</p>
<div class="formula">cⱼ = best normalized local correlation over five fixed shifts
D = meanⱼ((1 − cⱼ) / 2)
score = filled-outline overlap⁴ × exp(−8D)</div>
<p>Near-constant patches have a fixed RMS floor. The primary does not consult the connected-hole/covariance descriptors used by the separate moment controls. The shared evaluation computes those descriptors for the controls. Native evidence still undergoes bilinear affine sampling; this is not exact preservation of every pixel. The alignment and patch rules are established analytic ingredients, not a newly discovered theorem or proof of human-like reasoning.</p>
<h2>Freshness and data budget</h2><p>The new cohort contains 128 DEV and 128 TEST compositions chosen from previously unused grammar IDs. These are new combinations of familiar operators, not new natural semantic classes. Its 512 support images and 512 distinct query scene images yield 1,024 query rows through opposite enrollment reuse. The same candidate scene is positive for one support pattern and negative for the other. Fresh pixels are checked against the previous dataset for overlap.</p>
<p>The proposal CNN retains its prior synthetic-only training: 2,048 training compositions, 4,096 unique support images plus 4,096 unique query images, 1,024 optimizer updates, and 897,207 parameters. v12 adds zero optimizer updates. Algorithm development used a declared subset of 64 old TRAIN families and one old DEV pass; those development scores are not fresh validation. The algorithm, controls and replay roster were frozen before fresh scenes were generated. Fresh DEV is used for threshold calibration only. Enrollment receives one RGB reference and its support box; no query truth, mask, pose or grammar identity enters scoring. Natural training images and borrowed pretrained weights are zero.</p>
<h2>Actual examples, selected without looking at correctness</h2><img src="figures/fresh_fixed_examples.png" alt="First three TEST families: previous CNN and native patch predictions"><p>These are the first three TEST family IDs, enrollment 0. Green dashed boxes show the true matching target; orange boxes show each method's selected proposal, including rejected decisions. Negative scenes contain no matching micro-pattern.</p>{''.join(optional)}
<h2>Remaining limits</h2><p>The images have clean, separated, high-contrast objects, making foreground extraction and outline alignment favorable. The new path can fail with touching objects, low contrast, occlusion, changing texture, symmetric outlines or missing local structure. Per-patch translation search can sometimes align unrelated local evidence. The task rejects same-outline objects with a different internal pattern; that is narrower than recognizing all natural examples of a type. A result here must be retested on genuinely new natural photos before claiming progress on semantic category recognition.</p>
<h2>Evidence</h2><p>The independent audit reproduced all 168 metric fields, 6,144 localization rows, 24 interval values, 96 full predictions and 16 direct original-CNN predictions. A documentation filename collision temporarily replaced an offline review after freezing; its exact original bytes were restored to the frozen hash, the replacement was preserved separately, and the recovery is disclosed below. No model, scoring rule or result changed.</p><ul>{links}</ul><p class="small">Results SHA256: {sha(result_path)}. Source and score hashes verified during report generation. Report generation does not alter model weights, scoring rules or thresholds.</p></main></html>'''
    (ROOT/'index.html').write_text(report,encoding='utf-8')
    print(json.dumps({'report':'index.html','primary_difference':primary,'matched_detail_difference':detail,'result_sha256':sha(result_path)},indent=2))
if __name__=='__main__':main()
