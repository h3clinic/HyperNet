"""One fixed comparison: own CNN baseline and five native-detail controls."""
import argparse, datetime, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from hypernet.detail_pipeline import DetailPipeline, METHODS, PRIMARY, sha256, state_hash
from hypernet.detail_metrics import cutoff_from_dev, summarize, bootstrap
from hypernet.one_shot_detector import box_iou


def verify_protocol(path):
    protocol = json.loads(path.read_text())
    if not protocol['model_selection_complete'] or protocol['fresh_test_generated_before_freeze']:
        raise ValueError('Protocol must be frozen before fresh generation')
    for file, expected in protocol['behavior_files'].items():
        source = Path(file) if Path(file).is_absolute() else ROOT/file
        if sha256(source) != expected:
            raise ValueError(f'Changed behavior file: {file}')
    return protocol


def load_rgb(data, record):
    path = (data/record['rgb_path']).resolve()
    if not path.is_relative_to(data.resolve()) or sha256(path) != record['rgb_sha256']:
        raise ValueError('RGB file does not match manifest')
    rgb = np.array(Image.open(path).convert('RGB'))
    # Same decoded-array digest convention as the frozen dataset builder.
    h = hashlib.sha256()
    h.update(str(rgb.dtype).encode()); h.update(str(rgb.shape).encode()); h.update(rgb.tobytes())
    if h.hexdigest() != record['rgb_array_sha256']:
        raise ValueError('Decoded RGB does not match manifest')
    return rgb


def evaluate_episodes(engine, episodes, data):
    rows = {method: [] for method in METHODS}
    cache = {}
    for index, ep in enumerate(episodes):
        reference_rgb = load_rgb(data, ep['support'])
        reference = engine.prepare_reference(reference_rgb, ep['support']['bbox_xyxy'])
        for query in ep['queries']:
            key = query['rgb_path']
            if key not in cache:
                rgb = load_rgb(data, query)
                cache[key] = engine.prepare_query(rgb)
            predictions = engine.score(reference, cache[key])
            # Ground truth is read only after all methods have scored the image.
            target = query['target_bbox_xyxy'] if query['present'] else None
            for method, prediction in predictions.items():
                box = prediction['bbox_xyxy']
                row = dict(id=query['id'], episode_id=ep['episode_id'], split=ep['split'],
                    family_id=ep['family_id'], enrolled_variant=ep['enrolled_variant'],
                    present=bool(query['present']), method=method, **prediction,
                    iou=box_iou(box, target) if box and target else 0.,
                    proposal_best_iou=max([box_iou(b, target) for b in prediction['proposals']], default=0.) if target else 0.,
                    support_path=ep['support']['rgb_path'], query_path=key)
                rows[method].append(row)
        if (index+1) % 64 == 0:
            print(f'Evaluated {index+1}/{len(episodes)} enrollments', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT.parent/'hypernet_v11/checkpoints/detector_micro_aware.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-dev-only', action='store_true')
    parser.add_argument('--protocol-freeze', type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite evaluation evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data/'manifest.json').read_text())
    protocol_sha = None
    if args.old_dev_only:
        if sha256(args.data/'manifest.json') != '515a46197650689e922dd00a63e8f08ce14a285f43142c1c0c93733b661b2d27':
            raise ValueError('Development accepts the declared old v11 manifest only')
        episodes = [e for e in manifest['episodes'] if e['split'] == 'dev']
    else:
        if args.protocol_freeze is None:
            raise ValueError('Fresh evaluation requires protocol freeze')
        protocol = verify_protocol(args.protocol_freeze)
        protocol_sha = sha256(args.protocol_freeze)
        if sha256(args.data/'plan.json') != protocol['fresh_plan_sha256']:
            raise ValueError('Unexpected fresh data plan')
        admission_path = ROOT/'experiments/fresh_generation_admission.json'
        admission = json.loads(admission_path.read_text())
        # The generation gate independently verifies the supplied protocol hash.
        if admission['protocol_sha256'] != protocol_sha:
            raise ValueError('Fresh images were not admitted under this protocol')
        audit = json.loads((ROOT/'experiments/fresh_data_audit.json').read_text())
        if (manifest.get('completed') is not True or audit.get('valid') is not True
                or audit['manifest_sha256'] != sha256(args.data/'manifest.json')
                or audit['model_protocol_sha256'] != protocol_sha
                or manifest['generation_admission_sha256'] != sha256(admission_path)
                or audit['generation_admission_sha256'] != sha256(admission_path)):
            raise ValueError('Requires completed fresh data and audit bound to this manifest and admission')
        episodes = [e for e in manifest['episodes'] if e['split'] in ('dev', 'test')]
    torch.set_num_threads(4)
    engine = DetailPipeline(args.checkpoint)
    before = state_hash(engine.model); start = time.perf_counter()
    rows = evaluate_episodes(engine, episodes, args.data)
    after = state_hash(engine.model)
    if before != after or any(p.grad is not None for p in engine.model.parameters()):
        raise RuntimeError('Inference modified weights or created gradients')
    cuts = {m: cutoff_from_dev([r for r in rows[m] if r['split'] == 'dev']) for m in METHODS}
    splits = ('dev',) if args.old_dev_only else ('dev', 'test')
    metrics = {m: {s: summarize([r for r in rows[m] if r['split'] == s], cuts[m]) for s in splits} for m in METHODS}
    for method in METHODS:
        with (args.output/f'{method}_scores.jsonl').open('x', encoding='utf-8') as file:
            for row in rows[method]:
                file.write(json.dumps(row, allow_nan=False)+'\n')
    result = dict(format='hypernet.native_detail_evaluation.v12', completed=True,
        scope='OLD DEV DEVELOPMENT ONLY' if args.old_dev_only else 'FROZEN FRESH 2D SYNTHETIC EVALUATION',
        primary_method=PRIMARY, primary_metric='localized_recall_at_iou50', methods=metrics,
        protocol_sha256=protocol_sha, manifest_sha256=sha256(args.data/'manifest.json'),
        checkpoint_sha256=sha256(args.checkpoint), model_state_before=before, model_state_after=after,
        gradients_created=False, seconds=time.perf_counter()-start,
        natural_training_images=0, externally_pretrained_weights=False,
        learned_weight_updates_this_version=0, reference_images_per_enrollment=1,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        score_files={m:dict(path=f'{m}_scores.jsonl', sha256=sha256(args.output/f'{m}_scores.jsonl')) for m in METHODS})
    if not args.old_dev_only:
        result['bootstrap'] = bootstrap(rows, cuts)
    (args.output/'fresh_results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
