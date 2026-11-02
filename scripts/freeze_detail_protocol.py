"""One-way preregistration after old-TRAIN/DEV development, before new images."""
import datetime, hashlib, json, platform
from pathlib import Path
import numpy, PIL, torch, scipy, cv2
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = ROOT/'data/shapes2d_fresh_v12'
    if any(data.rglob('*.png')) or (data/'manifest.json').exists():
        raise RuntimeError('Cannot freeze after generation')
    path = ROOT/'experiments/detail_protocol_freeze.json'
    if path.exists():
        raise FileExistsError(path)
    # Bind the complete local Python package, covering transitive dependencies.
    files = sorted((ROOT/'hypernet').glob('*.py')) + [
        ROOT/'scripts/evaluate_detail.py', ROOT/'scripts/prepare_fresh_2d_v12.py',
        Path(__file__).resolve(), ROOT/'data/SHAPES2D_FRESH_CONTRACT.json',
        ROOT/'experiments/fresh_replay_plan.json']
    checkpoint = ROOT.parent/'hypernet_v11/checkpoints/detector_micro_aware.pt'
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    sources[str(checkpoint.resolve())] = digest(checkpoint)
    evidence = ['experiments/gradient_patch_prototype_plan.json',
        'experiments/gradient_patch_prototype_results.json',
        'experiments/packaged_gradient_train_replay.json',
        'experiments/train_detail_diagnosis.json',
        'experiments/old_dev_stability/fresh_results.json',
        'experiments/pipeline_independent_review.json']
    for item in evidence:
        if not (ROOT/item).exists():
            raise FileNotFoundError('Missing development/review evidence: '+item)
    review = json.loads((ROOT/evidence[-1]).read_text())
    if review.get('valid') is not True:
        raise ValueError('Independent pipeline review must pass')
    protocol = dict(format='hypernet.native_detail_protocol.v12',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_selection_complete=True, fresh_test_generated_before_freeze=False,
        fresh_plan_sha256=digest(data/'plan.json'),
        fresh_data_freeze_sha256=digest(ROOT/'experiments/fresh_data_freeze.json'),
        behavior_files=sources, development_evidence={item:digest(ROOT/item) for item in evidence},
        primary_method='native_patch',
        methods=['old_cnn','outline_only','centroid_only','local_moments','lowres_patch','native_patch'],
        selection='Native gradient patch matching selected using64 balanced old TRAIN families; '
            'one full old DEV stability pass confirmed the choice. Fresh images were not generated or scored.',
        architecture='Own frozen synthetic-only CNN proposes separated 2D regions; native filled-silhouette '
            'alignment and analytic local patch matching bypass its128D descriptor. No v12 learned weight updates.',
        one_reference_contract='One96px RGB support with enrollment box per enrollment; '
            'query input is the full256px RGB scene. No query annotations or category IDs enter inference. '
            'Four exact quarter-turn copies of the same reference are used by the old CNN baseline.',
        primary_metric='localized_recall_at_iou50',
        calibration='Each method independently sets its threshold from fresh DEV negatives only: '
            'nextafter(descending_scores[floor(.05*N)],+inf). TEST thresholds remain fixed; actual TEST FPR reported.',
        primary_comparison='native_patch minus old_cnn on the same fresh TEST rows',
        mechanism_control='native_patch minus lowres_patch: identical CNN proposals, native outline alignment, '
            'native reference patch locations and equal patch weights; only fine RGB evidence is area-downsampled4x '
            'then bilinearly upsampled in the control.',
        bootstrap=dict(unit='family', samples=10000, seed=1812, confidence=.95,
            type='paired percentile, conditional on fixed DEV thresholds'),
        data_budget=dict(inherited_train_families=2048, inherited_unique_synthetic_training_rgb=8192,
            inherited_training_enrollments=4096, inherited_query_exposures=8192,
            inherited_reference_quarter_turn_views=16384, inherited_optimizer_steps=1024,
            natural_training_images=0, borrowed_pretrained_weights=0,
            learned_weight_updates_v12=0, reference_images_per_enrollment=1,
            development_train_families=64, development_old_dev_families=128,
            fresh_dev_families=128, fresh_test_families=128),
        limitations=['Controlled high-contrast separated synthetic2D patterns, not natural semantic categories.',
            'Existing shape primitives are recombined into held-out family compositions; this is not a new visual world.',
            'Foreground contrast, shape registration, resolved interior gradients and limited occlusion are assumptions.',
            'The primary is a CNN-plus-analytic matcher; no claim that the CNN alone learned this performance.',
            'Aligned patches still undergo bilinear sampling; native evidence does not mean every pixel is exact.',
            'Nominally equal-area microprimitives have small rasterized-area differences; outline/centroid controls reported.',
            'Conditional family CIs exclude uncertainty from DEV threshold calibration; no multiple-comparison adjustment.',
            'No fresh score may select a method, alter coefficients, thresholds beyond declared DEV calibration, or exclude cases.'],
        runtime=dict(python=platform.python_version(),numpy=numpy.__version__,pillow=PIL.__version__,
            torch=torch.__version__,scipy=scipy.__version__,opencv=cv2.__version__,torch_threads=4))
    path.write_text(json.dumps(protocol,indent=2,allow_nan=False))
    print(json.dumps(dict(path=str(path),sha256=digest(path),behavior_files=len(sources)),indent=2))


if __name__=='__main__':
    main()
