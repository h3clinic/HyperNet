"""Allowlisted v09 export, permitted only after the independent evidence bundle passes."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from export_audited_results import sha, read, write, assert_public, coco_id, partition
from verify_v09_results import verify

NAMES = {
    'scratch_reference':'Scratch, one reference',
    'procedural_restoration_frozen':'Procedural restoration, frozen',
    'procedural_restoration_adapted':'Procedural restoration, adapted',
    'procedural_detail_frozen':'Procedural detail, frozen (predeclared primary)',
    'procedural_detail_adapted':'Procedural detail, adapted',
    'random_frozen':'Random CNN, frozen',
    'fixed_shape':'Fixed shape descriptor',
    'owlvit_base_patch32':'OWL-ViT B/32 (external pretraining)',
}
PRIMARY='procedural_detail_frozen'
BASELINE='scratch_reference'
OWL='owlvit_base_patch32'


def cohort(root, manifest_hash):
    path=root/'data/coco_fresh_v09/manifest.json'
    assert sha(path)==manifest_hash
    source=read(path)
    license_path=path.parent/'source_licenses.json'
    licenses=read(license_path)
    fields=('image_id','coco_image_id','category_id','category_ids','source_category_ids',
            'path','sha256','pixel_sha256','source_relative_path','source_sha256',
            'source_pixel_sha256','source_record_id','original_width','original_height',
            'stored_width','stored_height','support_crop_xyxy_original','support_annotation_id')
    images=[{k:r[k] for k in fields if k in r} for r in source['images']]
    for image in images:
        for key in ('path','source_relative_path'):
            p=Path(image[key]);assert not p.is_absolute() and '..' not in p.parts
    pair_fields=('id','category_id','episode_id','support_id','reference','query',
                 'support_image_id','query_image_id','present','split','condition','evaluation_annotations')
    result={'source_manifest_sha256':manifest_hash,'source_attribution_sha256':sha(license_path),
        'dataset':source['dataset'],'protocol':source['protocol'],'seed':source['seed'],
        'categories':source['categories'],'settings':source['settings'],
        'annotation_budget':source['annotation_budget'],
        'original_annotation_sha256':licenses['original_annotation_sha256'],
        'images':images,'pairs':[{k:r[k] for k in pair_fields if k in r} for r in source['pairs']],
        'attribution':{'dataset_url':licenses['dataset_url'],'licenses':licenses['licenses'],
            'source_images':[{k:r[k] for k in ('id','file_name','license','coco_url','flickr_url','height','width') if k in r}
                             for r in licenses['images']]},
        'reconstruction':'Read original RGB with EXIF transpose; apply recorded support crop; Pillow Image.thumbnail((384,384), LANCZOS); save RGB PNG. Verify dimensions and file/pixel hashes. Exact PNG bytes can depend on library versions.',
        'implementation_reference':'hypernet/category_coco.py and hypernet/data.py',
        'limitations':['Metadata only; no photographs are redistributed.',
                       'Source availability and image licenses remain external to this repository.',
                       'Different file identities do not prove different physical instances or capture sessions.']}
    assert len(images)==218 and len(result['pairs'])==720
    assert_public(result)
    return result


def write_report(out, aggregate, analysis):
    ci=analysis['contrasts_against_scratch'][PRIMARY]
    lines=['# HyperNet v0.9: audited primitive-shape control','',
        '**The primary remains near chance.** The frozen procedural-detail CNN detected **10/180 positive TEST pairs** and falsely accepted **10/180 negatives**: recall **5.56%**, FPR **5.56%**, balanced accuracy **50.00%**. The scratch reference CNN detected **4/180** with **9/180** false alarms: balanced accuracy **48.61%**.','',
        f"The paired balanced-accuracy change is **{100*ci['balanced_accuracy']['difference']:+.2f} percentage points**, 95% category-bootstrap interval **[{100*ci['balanced_accuracy']['low']:+.2f}, {100*ci['balanced_accuracy']['high']:+.2f}]**. This does not establish an improvement. The primary's pooled AUROC (0.5427) and category-macro AUROC (0.5417) also trail scratch (0.5521 and 0.5504).",'',
        'This is a control using primitive generated shapes, not the final richer named-shape curriculum. It is not a human-vision, state-of-the-art or equal-pretraining-budget claim. All seven HyperNet controls and the separate OWL-ViT comparator are retained.','',
        '## Complete TEST comparison','',
        'Each method has its own global cutoff, chosen only on DEV to allow at most9 false positives among180 DEV negatives. TEST false-positive rates are measured, not forced to5%. Pooled ranking metrics appear below.','',
        '| Method | TP /180 | Recall | FP /180 | FPR | Balanced accuracy | AUROC | Average precision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        m=aggregate['models'][name]['test']['overall']
        lines.append(f"| {NAMES[name]} | {m['tp']} | {100*m['tpr']:.2f}% | {m['fp']} | {100*m['fpr']:.2f}% | {100*m['balanced_accuracy']:.2f}% | {m['auroc']:.4f} | {m['average_precision']:.4f} |")
    lines += ['', '## Paired primary change','',
        'Percentage-point differences use10,000 paired category-bootstrap draws, seed1809, with the same sampled category indices for every method and fixed DEV cutoffs. There are only six TEST category clusters, queries recur, and only one shared-pretraining seed was evaluated. The intervals are conditional on these models/cutoffs and exclude training/calibration uncertainty. This post-evaluation paired analysis does not replace the original marginal intervals retained in the aggregate file.','',
        '| Metric | Difference | 95% interval |','|---|---:|---:|']
    for key,label in [('tpr','Recall'),('fpr','FPR'),('balanced_accuracy','Balanced accuracy'),('auroc_category_macro','Category-macro AUROC')]:
        c=ci[key];lines.append(f"| {label} | {100*c['difference']:+.2f} | [{100*c['low']:+.2f}, {100*c['high']:+.2f}] |")
    loc=analysis['localization'][PRIMARY]
    lines += ['', 'All control contrasts are in [paired_analysis.json](paired_analysis.json); they are descriptive and not multiplicity-adjusted. No best-looking control replaces the predeclared primary.','',
        f"Localization remains weak: the primary's selected box reaches IoU≥0.5 on **{loc['all_positive_iou50']}/180** positive pairs, and **{loc['accepted_iou50']}/{loc['accepted_positive_pairs']}** accepted positives. **{loc['accepted_zero_overlap']}/{loc['accepted_positive_pairs']}** accepted positives have zero box overlap. This top-box diagnostic is not detector AP.",'',
        '## Training and evaluation budget','',
        '- Two own synthetic backbones, initialized randomly: restoration and restoration plus0.25 foreground mask/detail prediction plus0.2 matched-nuisance triplet loss. Same CNN initialization and procedural sequence; no borrowed weights, natural-image pretraining, learned teacher or downloaded assets.',
        '- Per backbone:2,000 steps ×16 triplets;32,000 base-program draws +32,000 mutant-program draws;96,000 raw RGB exposures and192,000 corrupted training-view exposures. Unique image counts were not measured. The CNN has40,976 parameters; the auxiliary training-only head has33,280.',
        '- Primary frozen-detail model: one real reference and one support box for enrollment; zero real-image weight updates. Adapted controls:120 updates on297 patches from that one reference, with independent copies of the same synthetic checkpoint. Scratch control starts randomly. No query weight updates.',
        '- 12 categories, six DEV and six TEST; three support episodes per category;36 real supports and218 new source photographs;720 pairs total. DEV and TEST each contain360 pairs,180 present and180 absent.',
        '- 108 separate reference CNNs across scratch and two adapted controls. Fixed/random controls have no weight training. One shared synthetic CNN per frozen arm is reused across episodes.',
        '- OWL-ViT B/32 uses its official external pretrained checkpoint, image-guided inference and no text prompts. Its natural pretraining budget and possible dataset overlap differ from HyperNet and are not established as absent.',
        '- Support/query sources are separated. Previously studied target types can recur; freshness means new photo sources with ID/hash exclusions, not categories unseen by every model. Query boxes are evaluation-only.',
        '', '## Independent fine-detail diagnostic','',
        'On96 generated native32 comparisons, the combined detail arm wins78 (81.25%) versus77 (80.21%) for restoration: one net comparison. Downsampling96px inputs to32px reduces both. This diagnostic is not natural-photo recognition accuracy or a demonstrated detail breakthrough. See [detail_diagnostic.json](detail_diagnostic.json). Because the combined objective changes two terms, it does not isolate either term alone.',
        '', '## Audits and reproducibility','',
        'The final independent audit checks5,040 HyperNet scores,720 OWL scores,108 reference checkpoints and two synthetic backbones. Twenty-eight read-only inference replays match exactly. A preliminary procedural-training attempt failed on thick-ring geometry; that generator defect was repaired before the completed synthetic run and before fresh-photo evaluation. No fresh score selected the repair, hyperparameters or primary model.',
        '', '```bash','python publication/verify_snapshot.py','python publication/verify_v09_results.py benchmarks/v0.9','```','',
        '[scores.jsonl](scores.jsonl) contains all5,760 scores, pair/category/split identities, COCO IDs and selected boxes. [public_cohort_manifest.json](public_cohort_manifest.json) supplies exact support crops, source/stored dimensions and hashes, evaluation boxes and original license/source URLs. Photographs and weights are not redistributed. Coordinates refer to processed query images; preprocessing is in the published package.',
        '', '[aggregate_metrics.json](aggregate_metrics.json) includes DEV/TEST, per-category and marginal interval results, budgets and evidence hashes. [arithmetic_verification.json](arithmetic_verification.json) independently verifies confusion/ranking metrics with scikit-learn, DEV cutoffs, pair identities, paired intervals and localization. It does not rerun inference or prove image provenance.',
        '', 'COCO labels can be incomplete; distinct files do not verify physical-instance/capture independence. Repeated photos, six TEST categories and a small hand-designed shape vocabulary limit interpretation. This result is evidence for further work, not a successful category-recognition system.']
    text='\n'.join(lines)+'\n'
    for a,b in [('most9','most 9'),('among180','among 180'),('to5%','to 5%'),('use10,000','use 10,000'),('seed1809','seed 1809'),('plus0.','plus 0.'),('backbone:2,000','backbone: 2,000'),('×16','× 16'),(';32,000','; 32,000'),(';96,000','; 96,000'),('and192,000','and 192,000'),('has40,976','has 40,976'),('has33,280','has 33,280'),('controls:120','controls: 120'),('on297','on 297'),(';36','; 36'),('and218','and 218'),(';720','; 720'),('contain360','contain 360'),('pairs,180','pairs, 180'),('and180','and 180'),('On96','On 96'),('wins78','wins 78'),('versus77','versus 77'),('Downsampling96','Downsampling 96'),('to32','to 32'),('checks5,040','checks 5,040'),('scores,720','scores, 720'),('scores,108','scores, 108'),('all5,760','all 5,760')]:text=text.replace(a,b)
    (out/'README.md').write_text(text,encoding='utf-8',newline='\n')


def export(args):
    root,out=args.root.resolve(),args.output.resolve()
    if out.exists():raise FileExistsError('Choose a new publication directory')
    evidence_path=root/'experiments/v09_evidence_bundle.json'
    evidence=read(evidence_path);assert evidence['valid'] is True
    artifacts={}
    for path,digest in evidence['artifacts'].items():
        path=Path(path);assert sha(path)==digest,'Audited artifact changed: '+path.name
        artifacts[path.relative_to(root).as_posix()]=digest
    summary_path=root/'results/fresh_v09/summary.json';owl_path=root/'results/owlvit_fresh_v09/summary.json'
    summary,owl=read(summary_path),read(owl_path)
    assert summary['completed'] and owl['completed']
    assert summary['manifest_sha256']==owl['manifest_sha256']
    assert set(summary['models'])==set(NAMES)-{OWL} and set(owl['models'])=={OWL}
    freeze=read(root/'experiments/v09_freeze.json')
    assert freeze['primary_model']==PRIMARY and freeze['baseline_model']==BASELINE
    assert freeze['natural_pretraining_images']==0 and freeze['externally_provided_pretrained_weights'] is False
    analysis_path=root/'experiments/v09_analysis.json';analysis=read(analysis_path)
    assert analysis['input_summary_sha256']==sha(summary_path)
    assert analysis['input_owl_summary_sha256']==sha(owl_path)
    assert set(analysis['contrasts_against_scratch'])==set(NAMES)
    analysis['source_analysis_sha256']=sha(analysis_path)
    models={}
    all_models={**summary['models'],**owl['models']}
    for alias in NAMES:
        item=all_models[alias]
        budget={'support_photos_per_episode':1,'support_boxes_per_episode':1,
                'query_weight_updates':0,'query_boxes_as_model_input':0}
        if alias==OWL:
            p=item['provenance'];assert p['image_guided'] is True and p['text_prompts'] is False
            budget.update(pretrained=True,externally_provided_pretrained_weights=True,
                natural_pretraining_images=None,target_weight_updates=0,
                **{k:p[k] for k in ('checkpoint_id','checkpoint_revision','checkpoint_files',
                                   'external_training_source','pretraining_exposure_note')})
        else:
            own=alias.startswith('procedural_');adapt=alias.endswith('_adapted') or alias==BASELINE
            budget.update(pretrained=own,externally_provided_pretrained_weights=False,
                natural_pretraining_images=0,additional_real_training_images=0,
                real_training_source_images_per_reference_model=1 if adapt else 0,
                real_reference_enrollment_images=1,real_reference_weight_updates=120 if adapt else 0,
                cnn_parameters=0 if alias=='fixed_shape' else 40976,
                synthetic_pretraining_updates=2000 if own else 0,
                base_program_draws=32000 if own else 0,mutant_program_draws=32000 if own else 0,
                synthetic_rendered_image_exposures=96000 if own else 0,
                synthetic_corrupted_view_exposures=192000 if own else 0,
                auxiliary_training_head_parameters=33280 if own else 0,
                unique_synthetic_images_measured=False,
                reference_derived_training_view_exposures=7680 if adapt else 0,
                training_seed=73,inter_reference_adapted_weight_transfer=False)
        models[alias]={'budget':budget,'calibration':item['calibration'],
                       'dev':partition(item['development_diagnostics']),'test':partition(item['test'])}
    aggregate={'created_utc':datetime.now(timezone.utc).isoformat(),'release':'0.9.0',
        'status':'Audited primitive-shape control; primary remains near chance',
        'primary_model':PRIMARY,'predeclared_baseline':BASELINE,'models':models,
        'protocol':{'task':'Independent-photo category presence, not instance identification or detector AP',
            'categories':12,'episodes_per_category':3,'real_supports':36,'source_photographs':218,
            'per_reference_checkpoints':108,'own_synthetic_checkpoints':2,
            'dev_pairs':360,'dev_positives':180,'dev_negatives':180,
            'test_pairs':360,'test_positives':180,'test_negatives':180,
            'dev_categories':summary['category_splits']['dev'],'test_categories':summary['category_splits']['test'],
            'target_dev_fpr':.05,'threshold_rule':'score >= nextafter(10th-highest DEV-negative score, +infinity)',
            'threshold_scope':'Method-specific global DEV cutoff; fixed for TEST',
            'annotation_budget':summary['annotation_budget'],'same_cohort_all_models':True,
            'physical_instance_ids_verified':False,'box_coordinates':'Processed query image pixels'},
        'independent_audits':{'evidence_bundle_sha256':sha(evidence_path),
            'artifacts':artifacts,'hypernet_predictions_checked':5040,'owl_predictions_checked':720,
            'reference_checkpoints_checked':108,'synthetic_backbones_checked':2,'exact_inference_replays':28},
        'source_hashes':{'manifest':summary['manifest_sha256'],'summary':sha(summary_path),
            'owl_summary':sha(owl_path),'analysis':sha(analysis_path),
            'freeze':sha(root/'experiments/v09_freeze.json')},
        'limitations':summary['dataset_limitations']+[
            'Only six TEST category clusters, recurring query photographs and one procedural pretraining seed.',
            'Own synthetic pretraining is additional training data; not one image total.',
            'Combined detail/triplet objective does not isolate either component.',
            'Primitive-shape curriculum is a control, not the final richer named-shape direction.',
            'OWL external pretraining and potential dataset overlap are not matched to HyperNet.'],
        'public_artifacts':{}}
    rows=[]
    for result_name in ('fresh_v09','owlvit_fresh_v09'):
        for line in (root/'results'/result_name/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert math.isfinite(r['score'])
            rows.append({'pair_id':r['id'],'model':r['model'],'episode_id':r['episode_id'],
                'support_coco_id':coco_id(r['support_image_id']),'query_coco_id':coco_id(r['query_image_id']),
                'category':r['category_id'],'split':r['split'],'present':r['present'],
                'score':r['score'],'bbox_xyxy':r['prediction'].get('bbox_xyxy')})
    assert len(rows)==5760
    raw_diagnostic=read(root/'experiments/detail_mechanism_comparison.json')
    fields=('method','mode','n','triplet_win','nuisance_distance','detail_only_distance','negative_distance',
        'bounded_margin','detail_to_nuisance_ratio','tie_fraction','visible_query_detail_fraction',
        'normalized_feature_variance_sum','zero_norm_fraction','per_family_triplet_win')
    diagnostic={'scope':raw_diagnostic['scope'],'configuration':raw_diagnostic['configuration'],
        'rows':[{k:r[k] for k in fields} for r in raw_diagnostic['rows']],
        'source_sha256':sha(root/'experiments/detail_mechanism_comparison.json')}
    for value in (aggregate,analysis,rows,diagnostic):assert_public(value)
    out.mkdir(parents=True)
    with (out/'scores.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for row in sorted(rows,key=lambda r:(r['model'],r['pair_id'])):
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(out/'public_cohort_manifest.json',cohort(root,summary['manifest_sha256']))
    aggregate['public_artifacts']={'scores_sha256':sha(out/'scores.jsonl'),'scores_rows':5760,
        'cohort_manifest_sha256':sha(out/'public_cohort_manifest.json')}
    write(out/'aggregate_metrics.json',aggregate)
    write(out/'paired_analysis.json',analysis)
    write(out/'detail_diagnostic.json',diagnostic)
    verification=verify(out);write(out/'arithmetic_verification.json',verification)
    write_report(out,aggregate,analysis)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/v0.9'))
    export(parser.parse_args())
