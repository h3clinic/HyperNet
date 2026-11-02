"""Freeze the transferred model and all controls before any real-photo scoring."""
from pathlib import Path
import datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.real_photo import TRANSFERRED_CUTOFFS,ALIASES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination=ROOT/'experiments/real_photo_protocol_freeze.json'
    if destination.exists():raise FileExistsError('Preserve the original protocol')
    if any((ROOT/'results').rglob('*scores.jsonl')):
        raise RuntimeError('Cannot preregister after real-photo prediction logs exist')
    data=ROOT/'data/coco_real_v13';manifest=json.loads((data/'manifest.json').read_text())
    audit=ROOT/'experiments/real_photo_data_audit.json';review=ROOT/'experiments/real_photo_runner_review.json'
    audit_data=json.loads(audit.read_text());review_data=json.loads(review.read_text())
    if audit_data.get('valid') is not True or audit_data['manifest_sha256']!=sha(data/'manifest.json'):
        raise ValueError('Require passed, exact data audit')
    if review_data.get('valid') is not True:raise ValueError('Require passed runner review/tests')
    checkpoint=ROOT.parent/'hypernet_v11/checkpoints/detector_micro_aware.pt'
    threshold_source=ROOT.parent/'hypernet_v12/results/fresh_eval/fresh_results.json'
    files=sorted((ROOT/'hypernet').rglob('*.py'))+[
        ROOT/'scripts/evaluate_real_photo.py',Path(__file__).resolve(),
        ROOT/'scripts/prepare_real_photos.py',ROOT/'scripts/recover_real_photo_transport.py',
        ROOT/'experiments/real_photo_replay_plan.json',
        ROOT/'experiments/real_photo_positive_replay_supplement.json']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    protocol=dict(format='hypernet.real_photo_transfer_protocol.v13',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_selection_complete=True,photo_scores_computed_before_freeze=False,
        primary_mode='transferred',primary_method='native_patch',methods=list(ALIASES),
        primary_metric='localized_recall_at_iou50',transferred_thresholds=TRANSFERRED_CUTOFFS,
        threshold_source=dict(path=str(threshold_source),sha256=sha(threshold_source)),
        behavior_files=bindings,checkpoint_sha256=sha(checkpoint),
        data_manifest_sha256=sha(data/'manifest.json'),data_audit_sha256=sha(audit),
        data_freeze_sha256=sha(ROOT/'experiments/real_photo_data_freeze.json'),
        fetch_admission_sha256=sha(ROOT/'experiments/real_photo_fetch_admission.json'),
        transport_recovery_admission_sha256=sha(ROOT/'experiments/real_photo_transport_recovery_admission.json'),
        runner_review_sha256=sha(review),
        data_counts=manifest['counts'],categories=manifest['categories'],
        enrollment='One reference photograph with one subject box per category. The existing model '
            'extracts descriptors and local patches; zero natural-photo gradient updates. A categoryID '
            'selects the enrolled reference in the evaluator but is never passed into inference.',
        preprocessing='Reference box expanded1.25x, cropped and aspect-preserving letterboxed to96px. '
            'Querydetail cap1024px, CNN cap512px, no query enlargement, pad bottom/right to multiple4 '
            'and at least32px. All boxes mapped with actual rounded x/y scales to original photo coordinates.',
        primary='Native-patch matching using only CNN proposals, unchanged own weights and v12 syntheticDEV cutoff. '
            'Old128D CNN descriptor is compared on identical CNN proposals.',
        diagnostic_grid='51 fixed overlapping rectangles from image dimensions only, separately named '
            'old_cnn_grid/native_patch_grid; no learned weights or query annotations construct these boxes.',
        secondary_calibration='Each method additionally gets one pooled cutoff from48 negative photos among96 realDEV '
            'photos, accepting at mostfloor(.05*48)=2. This uses extra validation data and is not the primary '
            'one-reference-only transferred operating point. TEST never fits or selects a cutoff.',
        metric='One selected box per query. Positive success requires score>=cutoff and IoU>=.5 with any '
            'annotated noncrowd instance of the enrolled category. FPR uses annotation-absent negative scenes. '
            'This is not official COCO AP(.50:.95).',
        uncertainty=dict(unit='category',clusters=12,bootstrap_draws=10000,seed=1313,
            conditional_on='Selected support photos, this finite category roster and fixed mode thresholds. '
                'No threshold refitting; both modes reuse identical predictions.'),
        budgets=dict(real_support_images_per_category=1,real_support_boxes_per_category=1,
            total_real_supports=12,primary_real_calibration_photos=0,secondary_real_calibration_photos=96,
            test_photos=192,real_weight_training_images=0,real_gradient_steps=0,
            inherited_unique_synthetic_training_rgb=8192,inherited_training_compositions=2048,
            inherited_optimizer_steps=1024,borrowed_pretrained_weights=0),
        limitations=['A natural-category transfer test differs from v12 synthetic micro-pattern identity detection.',
            'Same12categories and same one-reference enrollment appear in DEV/TEST, but all300 photos are distinct.',
            'Exact source IDs and pixel hashes do not rule out near-duplicates, shared capture sessions or same physical objects.',
            'COCO annotations may miss objects, and annotation-based reference cropping supplies one subject box.',
            'Support96px preprocessing resamples fine detail; native query evidence stays original only below1024px.',
            'Clean synthetic segmentation/alignment and local texture correspondences may fail on photographs.',
            'No parameter, method, photo inclusion or threshold will be changed based on TEST outcomes.'])
    with destination.open('x',encoding='utf-8') as stream:json.dump(protocol,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(destination),sha256=sha(destination),behavior_files=len(bindings)),indent=2))


if __name__=='__main__':main()
