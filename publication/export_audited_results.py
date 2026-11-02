"""Export allowlisted public results after the experiment's independent audits pass.

No images, weights, local paths or full model provenance are copied. Invoke only
after the experiment owner confirms the final audits. This script never commits.
"""
from collections import defaultdict
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re

from verify_benchmark import verify

ALIASES=('v07_cnn','v08_assignment','v08_foreground','v08_tolerance',
         'v08_counterfactual','v08_full','fixed_v06','fixed_v08')
OWL='owlvit_base_patch32'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def assert_public(value):
    text=json.dumps(value,ensure_ascii=False)
    forbidden=(r'(?i)[a-z]:[/\\]+Users[/\\]+','/'+'home/','/'+'Users/',
               r'gh[pousr]_[A-Za-z0-9]{20,}',r'-----BEGIN [A-Z ]*PRIVATE KEY-----')
    assert not any(re.search(p,text) for p in forbidden),'Private path or credential-like string in export'


def coco_id(value):
    match=re.fullmatch(r'coco_image_(\d+)',str(value))
    if not match:
        raise ValueError('Expected a COCO image identifier')
    return int(match.group(1))


def partition(value):
    return {key:value[key] for key in ('overall','category_macro','by_category','category_cluster_ci')
            if key in value}


def public_cohort(root,manifest_hash):
    path=root/'data/coco_fresh_v08/manifest.json'
    assert sha(path)==manifest_hash
    source=read(path)
    license_path=path.parent/'source_licenses.json'
    licenses=read(license_path)
    image_fields=('image_id','coco_image_id','category_id','category_ids','source_category_ids',
                  'path','sha256','pixel_sha256','source_relative_path','source_sha256',
                  'source_pixel_sha256','source_record_id','original_width','original_height',
                  'stored_width','stored_height','support_crop_xyxy_original','support_annotation_id')
    pair_fields=('id','category_id','episode_id','support_id','reference','query',
                 'support_image_id','query_image_id','present','split','condition','evaluation_annotations')
    images=[{k:r[k] for k in image_fields if k in r} for r in source['images']]
    for image in images:
        for key in ('path','source_relative_path'):
            logical=Path(image[key])
            assert not logical.is_absolute() and '..' not in logical.parts
    attribution=[{k:r[k] for k in ('id','file_name','license','coco_url','flickr_url','height','width') if k in r}
                 for r in licenses['images']]
    cohort={'source_manifest_sha256':manifest_hash,'source_attribution_sha256':sha(license_path),
            'dataset':source['dataset'],'protocol':source['protocol'],'seed':source['seed'],
            'categories':source['categories'],'settings':source['settings'],
            'annotation_budget':source['annotation_budget'],
            'original_annotation_sha256':licenses['original_annotation_sha256'],
            'images':images,'pairs':[{k:r[k] for k in pair_fields if k in r} for r in source['pairs']],
            'attribution':{'dataset_url':licenses['dataset_url'],'licenses':licenses['licenses'],
                           'source_images':attribution},
            'reconstruction':'Read original source RGB with EXIF transpose; apply recorded support crop when present; use Pillow Image.thumbnail((384,384), LANCZOS); save RGB PNG. Verify stored dimensions, file hashes and decoded-pixel hashes. Exact PNG bytes can depend on library versions.',
            'implementation_reference':'hypernet/category_coco.py and hypernet/data.py',
            'limitations':['Metadata only: no photographs are redistributed.',
                           'Public source availability and original image license terms remain external to this repository.',
                           'Distinct COCO IDs and hashes do not establish distinct physical instances or capture sessions.']}
    assert len(images)==318 and len(cohort['pairs'])==960
    assert_public(cohort)
    return cohort


def write_report(out,aggregate,analysis):
    names={'v07_cnn':'Previous scratch CNN','v08_assignment':'Assignment',
           'v08_foreground':'Assignment + foreground','v08_tolerance':'Assignment + tolerance',
           'v08_counterfactual':'Assignment + counterfactual','v08_full':'Full v0.8.1 (predeclared primary)',
           'fixed_v06':'Fixed features / original matching','fixed_v08':'Fixed features / full matching',
           OWL:'OWL-ViT Base/32 (pretrained comparator)'}
    baseline=aggregate['models']['v07_cnn']['test']['overall']
    primary=aggregate['models']['v08_full']['test']['overall']
    interval=analysis['contrasts_against_v07']['v08_full']['balanced_accuracy']
    interpretation=('The paired balanced-accuracy interval includes zero; this experiment does not establish a clear improvement over the previous CNN.'
        if interval['low']<=0<=interval['high'] else
        'The paired balanced-accuracy interval favors the full method on this cohort; eight category clusters and recurring photos limit generalization.'
        if interval['low']>0 else
        'The paired balanced-accuracy interval favors the previous CNN on this cohort; the predeclared full method did not improve it.')
    lines=['# HyperNet v0.8.1: audited fresh-photo comparison','',
        f"The predeclared full method found **{primary['tp']}/240 positive TEST pairs**, versus **{baseline['tp']}/240** for the previous scratch CNN. False alarms were **{primary['fp']}/240 negatives**, versus **{baseline['fp']}/240**. Recall must be read together with false-positive rate.",'',
        interpretation,'',
        'This is a direct comparison on the same category-presence pairs. It is not a state-of-the-art claim, a detector-AP benchmark or an equal-pretraining-budget comparison with OWL-ViT.','',
        '## Complete TEST results','',
        'Each method uses its own global threshold selected only on DEV to allow at most 12 false positives among 240 DEV negatives. Thresholds are then frozen. TEST false-positive rates are measured, not forced to 5%. All eight HyperNet variants are retained; the best-looking ablation is not substituted for the predeclared primary.','',
        '| Method | Recall | False-positive rate | Balanced accuracy | AUROC | Average precision |',
        '|---|---:|---:|---:|---:|---:|']
    for name in (*ALIASES,OWL):
        m=aggregate['models'][name]['test']['overall']
        lines.append(f"| {names[name]} | {100*m['tpr']:.2f}% | {100*m['fpr']:.2f}% | {100*m['balanced_accuracy']:.2f}% | {m['auroc']:.4f} | {m['average_precision']:.4f} |")
    lines += ['', '## Paired change from the previous CNN','',
        'Full-method differences are percentage points. Intervals use the paired category-cluster percentile bootstrap: 10,000 resamples, seed 1807, and fixed DEV thresholds. They are approximate because there are only eight TEST category clusters and query photos can recur across categories or support episodes.','',
        '| Metric | Difference | 95% interval |','|---|---:|---:|']
    for key,label in [('tpr','Recall'),('fpr','False-positive rate'),('balanced_accuracy','Balanced accuracy')]:
        item=analysis['contrasts_against_v07']['v08_full'][key]
        lines.append(f"| {label} | {100*item['difference']:+.2f} | [{100*item['low']:+.2f}, {100*item['high']:+.2f}] |")
    lines += ['', 'All ablation and OWL contrasts, plus localization diagnostics, appear in [paired_analysis.json](paired_analysis.json). Localization counts use the top selected box and are not detection AP.','',
        '## Protocol and data budgets','',
        '- 16 categories: eight DEV and eight TEST. Three support episodes per category, 48 independent reference models, 960 pairs total.',
        '- DEV contains 480 pairs (240 present, 240 absent); TEST contains 480 pairs with the same balance.',
        '- Each learned HyperNet model trains from one source photograph and one annotated support box for 120 steps. Six learned scoring variants share that reference-specific CNN. No external image training, pretrained weights, learned teacher, query labels or query-time updates are used.',
        '- Fixed controls have no learned parameters. Image-guided OWL-ViT Base/32 uses the official pretrained checkpoint, one support image and no text prompt or target-time updates. Its external pretraining is substantial and not counted as one-image learning.',
        '- Support sources are disjoint from queries. Photos and category annotations can be imperfect; distinct file identities do not verify distinct physical objects or capture sessions.',
        '- Repeated query images make pair rows dependent. Development photographs used for method design are a separate budget from reference-model training.',
        '', '## Numerical repair and preserved evidence','',
        'The first attempt stopped after 47 reference checkpoints and 15 category outputs when interpolation produced a float32 value just above one. That incomplete evidence was preserved. Version 0.8.1 changes only a four-dtype-epsilon boundary tolerance and nonmutating clipping; valid-input outputs and RNG states were independently checked for bitwise agreement. All 48 CNNs were retrained and all 960 pairs rescored with the corrected code. The same cohort had already been partly scored, so this rerun is disclosed rather than described as untouched after the repair. No model, scoring, support-selection or threshold-selection parameter was chosen from those scores.',
        '', '## Verify the published arithmetic','',
        'From the repository root, after installing the development dependencies:', '',
        '```bash','python publication/verify_snapshot.py',
        'python publication/verify_benchmark.py benchmarks/v0.8.1','```','',
        '[scores.jsonl](scores.jsonl) contains all 8,640 model/pair scores, COCO image IDs, labels, split/category IDs and selected boxes. No photographs, weights, local paths or full private provenance are included. Boxes use processed-query coordinates. The verifier independently reconstructs each DEV cutoff and TEST confusion, AUROC and average-precision metrics using scikit-learn; it also checks identical pair identities across all methods. It does not rerun image inference or prove provenance.','',
        '[aggregate_metrics.json](aggregate_metrics.json) retains DEV/TEST results, per-category metrics, category-cluster intervals, data budgets and source evidence hashes. [arithmetic_verification.json](arithmetic_verification.json) records the public-data check. Audited source-artifact hashes are retained without machine-specific paths.','',
        '[public_cohort_manifest.json](public_cohort_manifest.json) supplies all pair identities, COCO source IDs, original and stored image hashes, image dimensions, exact support crops, evaluation boxes and public source/license metadata. This supports independent reconstruction without redistributing the photographs. Read original RGB with EXIF transpose, crop supports to the recorded integer coordinates, and apply Pillow LANCZOS thumbnail resizing with maximum side 384. The published package contains the original preprocessing implementation; file/pixel hashes expose differences caused by source files or library versions. Source image licenses remain those recorded by COCO.','',
        'No claim of general semantic invariance, biological equivalence or superiority to established vision systems follows from this small benchmark.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def export(args):
    root,owl_root,out=args.corrected_root.resolve(),args.owl_root.resolve(),args.output.resolve()
    summary_path=root/'results/fresh_v08/summary.json'
    raw_path=root/'results/fresh_v08/predictions.jsonl'
    owl_summary_path=owl_root/'results/owlvit_fresh_v08/summary.json'
    owl_raw_path=owl_root/'results/owlvit_fresh_v08/predictions.jsonl'
    summary,owl_summary=read(summary_path),read(owl_summary_path)
    freeze_path=root/'experiments/v08_freeze.json'
    freeze=read(freeze_path)
    analysis_path=root/'experiments/v08_analysis.json'
    analysis=read(analysis_path)
    assert summary['completed'] is True and owl_summary['completed'] is True
    assert set(summary['models'])==set(ALIASES) and set(owl_summary['models'])=={OWL}
    assert summary['manifest_sha256']==owl_summary['manifest_sha256']==freeze['manifest']['sha256']
    assert freeze['primary_model']=='v08_full' and freeze['baseline_model']=='v07_cnn'
    assert freeze['expected_checkpoints']==48 and freeze['images_per_model']==1
    assert freeze['external_training_images']==freeze['query_training_images']==0
    assert freeze['numerical_repair_not_model_selection'] is True
    audits={}
    for path in args.audit:
        evidence=read(path)
        repair_pass=(evidence.get('completed') is True and
                     evidence.get('all_previously_completed_scores_boxes_and_states_exact') is True)
        if evidence.get('passed') is not True and evidence.get('valid') is not True and not repair_pass:
            raise ValueError('Every supplied final audit must explicitly pass')
        audits[path.name]={'passed':True,'sha256':sha(path),
            'checks':{k:evidence[k] for k in ('predictions_checked','final_metric_predictions_checked',
                'scratch_checkpoints_checked','exact_inference_replay_predictions',
                'exact_original_repair_comparison_predictions','exact_original_state_dicts',
                'old_prediction_rows_compared','old_checkpoints_compared','exact_state_dicts',
                'all_scores_exact','all_boxes_exact','all_full_predictions_exact',
                'all_previously_completed_scores_boxes_and_states_exact','same_960_pairs',
                'independent_metrics_recomputed') if k in evidence}}
    if not audits:
        raise ValueError('At least one final audit artifact is required')
    if out.exists():
        raise FileExistsError('Public result destination must be new')
    models={}
    for name in (*ALIASES,OWL):
        item=(owl_summary if name==OWL else summary)['models'][name]
        budget={'external_pretraining':name==OWL,'support_photos_per_episode':1,
                'support_boxes_per_episode':1,'query_weight_updates':0,
                'query_boxes_as_model_input':0}
        if name==OWL:
            provenance=item['provenance']
            assert provenance['image_guided'] is True and provenance['text_prompts'] is False
            budget.update(external_training_images=None,
                          checkpoint_id=provenance['checkpoint_id'],
                          checkpoint_revision=provenance['checkpoint_revision'],
                          checkpoint_files=provenance['checkpoint_files'],
                          external_training_source=provenance['external_training_source'],
                          pretraining_exposure_note=provenance['pretraining_exposure_note'],
                          target_weight_updates=0)
        else:
            assert item['provenance']['external_pretraining'] is False
            budget.update(external_training_images=0,
                          trained_parameters=0 if name.startswith('fixed') else 40976,
                          target_training_steps=0 if name.startswith('fixed') else 120,
                          training_seed=None if name.startswith('fixed') else 73)
        models[name]={'budget':budget,'calibration':item['calibration'],
                      'dev':partition(item['development_diagnostics']),
                      'test':partition(item['test'])}
    assert analysis['input_summary_sha256']==sha(summary_path)
    assert set(analysis['test_categories'])==set(summary['category_splits']['test'])
    assert set(analysis['contrasts_against_v07'])==set((*ALIASES,OWL))
    assert set(analysis['localization'])==set((*ALIASES,OWL))
    public_analysis={key:analysis[key] for key in ('method','test_categories','limitations',
                        'contrasts_against_v07','localization','input_summary_sha256')}
    public_analysis['source_analysis_sha256']=sha(analysis_path)
    public_analysis['bootstrap']={'resamples':10000,'seed':1807,'confidence_level':.95,
                                  'cluster':'TEST target category','thresholds':'fixed from DEV'}
    assert_public(public_analysis)
    aggregate={'created_utc':datetime.now(timezone.utc).isoformat(),'release':'0.8.1',
        'primary_model':'v08_full','predeclared_baseline':'v07_cnn','models':models,
        'protocol':{'task':'Independent-photograph category presence; not instance identity or detector AP',
            'support_photos_per_episode':1,'support_boxes_per_episode':1,
            'learned_reference_models':48,'categories':16,'episodes_per_category':3,
            'dev_pairs':480,'dev_positives':240,'dev_negatives':240,
            'test_pairs':480,'test_positives':240,'test_negatives':240,
            'dev_categories':summary['category_splits']['dev'],
            'test_categories':summary['category_splits']['test'],
            'target_dev_fpr':.05,'threshold_rule':'score >= nextafter(13th-highest DEV-negative score, +infinity)',
            'threshold_scope':'Separate global threshold per method, frozen from DEV; TEST FPR is unconstrained',
            'uncertainty':'Category-cluster intervals; only eight TEST category clusters, repeated query photos across support episodes',
            'annotation_budget':summary['annotation_budget'],
            'physical_instance_ids_verified':False,
            'same_cohort_all_models':True,
            'box_coordinates':'Pixels in the processed query image, not necessarily original COCO image coordinates'},
        'numerical_repair':{'original_attempt':'Stopped on interpolation roundoff; 47 checkpoints and 15 category outputs preserved as incomplete evidence',
            'corrected_run':'All 48 CNNs retrained and all 960 pairs rescored; no score-based hyperparameter or method selection',
            'same_fresh_cohort_already_partly_scored':True,
            'repair':'Only four-dtype-epsilon boundary tolerance and nonmutating RGB clipping'},
        'source_hashes':{'manifest':summary['manifest_sha256'],'corrected_freeze':sha(freeze_path),
            'corrected_summary':sha(summary_path),'corrected_predictions':sha(raw_path),
            'owl_summary':sha(owl_summary_path),'owl_predictions':sha(owl_raw_path),
            'analysis':sha(analysis_path)},
        'independent_audits':audits,
        'limitations':summary['dataset_limitations']+[
            'Repeated query photographs and only eight TEST categories limit uncertainty estimates.',
            'Novel-category status relative to OWL-ViT pretraining is not established.',
            'These results do not establish state-of-the-art performance or biological equivalence.'],
        'public_artifacts':{}}
    compact=[]
    for path,expected in ((raw_path,set(ALIASES)),(owl_raw_path,{OWL})):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                assert row['model'] in expected and math.isfinite(row['score'])
                box=row['prediction'].get('bbox_xyxy')
                if box is not None:
                    assert len(box)==4 and all(math.isfinite(v) for v in box)
                compact.append({'pair_id':row['id'],'model':row['model'],
                    'episode_id':row['episode_id'],'support_coco_id':coco_id(row['support_image_id']),
                    'query_coco_id':coco_id(row['query_image_id']),'category':row['category_id'],
                    'split':row['split'],'present':row['present'],'score':row['score'],'bbox_xyxy':box})
    assert len(compact)==8640
    assert_public(aggregate)
    assert_public(compact)
    out.mkdir(parents=True)
    with (out/'scores.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for row in sorted(compact,key=lambda r:(r['model'],r['pair_id'])):
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    aggregate['public_artifacts']['scores_sha256']=sha(out/'scores.jsonl')
    aggregate['public_artifacts']['scores_rows']=len(compact)
    write(out/'public_cohort_manifest.json',public_cohort(root,summary['manifest_sha256']))
    aggregate['public_artifacts']['cohort_manifest_sha256']=sha(out/'public_cohort_manifest.json')
    write(out/'aggregate_metrics.json',aggregate)
    write(out/'paired_analysis.json',public_analysis)
    verification=verify(out)
    write(out/'arithmetic_verification.json',verification)
    write_report(out,aggregate,public_analysis)
    print(json.dumps({'output':str(out),'verification':verification},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrected-root',type=Path,required=True)
    parser.add_argument('--owl-root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/v0.8.1'))
    export(parser.parse_args())
