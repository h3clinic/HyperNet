"""Frozen synthetic diagnosis: full search versus privileged ROI diagnostics.

No optimization or natural-image access. Each protocol gets its own DEV-only
cutoff; full-scene search never uses ground-truth proposals or masks. Oracle
boxes, training-style sampled boxes and correspondence labels are separate
diagnostics, not replacements for the actual detector result.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from datetime import datetime, timezone
import sys, argparse, hashlib, json, time, math
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'evaluation'))
import numpy as np
import torch
from hypernet_v19.curriculum import episode, FORMAT as DATA_FORMAT
from hypernet_v19.model import file_sha
from hypernet_v19.matcher import FrozenDenseMatcher
from hypernet_v19.training import collate, correspondence_loss, ARRAY_KEYS
from hypernet_v19.proposals import box_iou
from vendor_v17.detection_metrics import global_cutoff, auc

PLAN = {'format': 'hypernet.v19.synthetic_diagnosis.v1', 'seed': 1901, 'difficulty': 1.,
        'split_counts': {'dev': 128, 'test': 128, 'compositional': 64},
        'index_rule': 'Indices0..count-1 within each disjoint procedural split',
        'primary': 'Top1 full generic-search localization IoU>=0.5 at global5%DEV-negative scene-max FPR',
        'calibration': 'Each model/protocol uses only its own DEV negative maxima; cutoff fixed for TEST and composition',
        'diagnostics': ['GT-object-box-only search', 'Training-style64sampled-ROI search',
            'Oracle correct-target versus near-negative ranking', 'Final-descriptor correspondence candidate accuracy',
            'Generic proposal coverage before scoring and after top100NMS'],
        'ground_truth_policy': 'Masks/points/boxes enter privileged diagnostics only; full search receives RGB alone',
        'small_sample_limit': 'Only64TEST and32compositional negatives; false-alarms/10000 is a rate conversion, not10000observations',
        'selection': 'Frozen final and matching random checkpoint only; no selection or weight updates'}
SOURCES = ('hypernet_v19/model.py', 'hypernet_v19/matcher.py', 'hypernet_v19/proposals.py',
           'hypernet_v19/training.py', 'hypernet_v19/curriculum.py', 'scripts/diagnose_synthetic.py',
           'evaluation/vendor_v17/detection_metrics.py')


def jsonable(value): return json.loads(json.dumps(value, allow_nan=False))


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix('.writing.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def input_digest(row, batch):
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        value = batch[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    digest.update(row['program_id'].encode())
    return digest.hexdigest()


def wilson(hits, count):
    if not count: return None
    z = 1.959963984540054; p = hits/count; den = 1+z*z/count
    middle = (p+z*z/(2*count))/den
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [max(0., middle-half), min(1., middle+half)]


def summarize(records, score_key, overlap_key, cutoff):
    present = np.asarray([r['positive'] for r in records], bool)
    score = np.asarray([r[score_key] for r in records], float)
    overlap = np.asarray([r[overlap_key] for r in records], float)
    accepted = score >= cutoff; p = int(present.sum()); n = int((~present).sum())
    true_presence = int((accepted & present).sum())
    localized = int((accepted & present & (overlap >= .5)).sum()); false = int((accepted & ~present).sum())
    unthresholded = int((present & (overlap >= .5)).sum())
    return {'episodes': len(records), 'positive_episodes': p, 'negative_episodes': n,
            'cutoff': float(cutoff), 'presence_hits': true_presence, 'localized_hits': localized,
            'false_alarms': false, 'presence_recall': true_presence/p if p else None,
            'localized_recall': localized/p if p else None, 'false_positive_rate': false/n if n else None,
            'false_alarms_per10000_rate_conversion': 10000*false/n if n else None,
            'fpr_wilson95': wilson(false, n), 'localized_recall_wilson95': wilson(localized, p),
            'presence_auroc': auc(present, score), 'unthresholded_localized_hits': unthresholded,
            'unthresholded_localized_recall': unthresholded/p if p else None}


def diagnostic_summary(records):
    positives = [r for r in records if r['positive']]
    target = np.asarray([r['oracle_target_logit'] for r in positives], float)
    near = np.asarray([r['oracle_near_logit'] for r in positives], float)
    paired = int((target > near).sum()); tied = int((target == near).sum())
    point_count = sum(r['point_candidate_count'] for r in positives)
    point_correct = sum(r['point_candidate_accuracy']*r['point_candidate_count'] for r in positives)
    sampled_scores = np.asarray([x for r in records for x in r['sampled_roi_logits']], float)
    sampled_labels = np.asarray([x > 0 for r in records for x in r['sampled_roi_quality_targets']], bool)
    return {'oracle_positive_pairs': len(positives), 'correct_above_near_count': paired, 'correct_near_ties': tied,
            'correct_above_near_fraction': paired/len(positives) if positives else None,
            'correct_vs_near_auroc': auc(np.r_[np.ones(len(target), bool), np.zeros(len(near), bool)], np.r_[target, near]),
            'correct_minus_near_mean_logit': float((target-near).mean()) if positives else None,
            'oracle_target_top1_of_all_gt_objects': sum(r['oracle_top_index'] == 0 for r in positives)/len(positives) if positives else None,
            'generic_proposal_coverage_iou50': sum(r['generic_proposal_best_iou'] >= .5 for r in positives)/len(positives) if positives else None,
            'scored_top100_proposal_coverage_iou50': sum(r['top100_best_iou'] >= .5 for r in positives)/len(positives) if positives else None,
            'point_candidate_count': point_count,
            'point_candidate_accuracy': point_correct/point_count if point_count else None,
            'point_loss_positive_episode_mean': float(np.mean([r['point_loss'] for r in positives])) if positives else None,
            'sampled_roi_count': len(sampled_scores), 'sampled_positive_roi_count': int(sampled_labels.sum()),
            'sampled_roi_quality_binary_auroc': auc(sampled_labels, sampled_scores),
            'sampled_roi_binary_accuracy_logit0': float(((sampled_scores >= 0) == sampled_labels).mean()),
            'mean_generic_proposals': float(np.mean([r['generic_proposal_count'] for r in records])),
            'comparison_caution': 'Sampled ROIs and GT boxes are privileged diagnostics. Their accuracy and cutoffs do not describe full image search.'}


def result_summary(by_split):
    result = {'protocols': {}, 'diagnostics': {}}
    protocols = {'full_search': ('full_score', 'full_top1_iou'),
                 'oracle_gt_boxes_only': ('oracle_score', 'oracle_top1_iou'),
                 'training_style_sampled64_only': ('sampled_score', 'sampled_top1_iou')}
    for name, (score_key, overlap_key) in protocols.items():
        cutoff = global_cutoff([r[score_key] for r in by_split['dev'] if not r['positive']], .05)
        result['protocols'][name] = {'calibration_source': 'DEV negative scene maxima only', 'cutoff': cutoff,
            'splits': {split: summarize(rows, score_key, overlap_key, cutoff) for split, rows in by_split.items()}}
    for split, rows in by_split.items():
        result['diagnostics'][split] = diagnostic_summary(rows)
        result['diagnostics'][split]['per_family'] = {str(family): diagnostic_summary([r for r in rows if r['family'] == family])
                                                     for family in sorted({r['family'] for r in rows})}
        # This deliberate diagnostic misuse quantifies the multiple-comparisons
        # problem; it is never the method's primary acceptance rule.
        sampled_cutoff = result['protocols']['training_style_sampled64_only']['cutoff']
        result['diagnostics'][split]['full_search_at_sampled64_dev_cutoff'] = summarize(rows, 'full_score', 'full_top1_iou', sampled_cutoff)
    return result


@torch.inference_mode()
def evaluate_episode(matcher, row):
    cpu = collate([row]); digest = input_digest(row, cpu)
    bank = matcher.enroll(row['support']); encoded = matcher.encode_query(row['query'])
    detections = matcher.predict(bank, encoded); top = detections[0]
    truth = np.asarray(row['target_box'], np.float32).reshape(1, 4)
    full_overlap = box_iou(np.asarray([d['box'] for d in detections]), truth)[:, 0]
    generic_overlap = box_iou(encoded['boxes'], truth)[:, 0]
    # Full-search results are already computed and fixed. From this point on,
    # ground-truth information is used only for the named privileged diagnostics.
    batch = {key: value.to(matcher.device) if torch.is_tensor(value) else value for key, value in cpu.items()}
    output = matcher.model(batch['support'], batch['query'], batch['rois'])
    oracle = matcher.model.sample_query_regions(output['query_encoded'], batch['boxes'])
    oracle_scores = matcher.model.score_regions(output['support_bank'], oracle)[0].cpu().numpy()
    oracle_index = int(oracle_scores.argmax())
    oracle_overlap = box_iou(row['boxes'][oracle_index:oracle_index+1], truth)[0, 0]
    sampled_scores = output['scores'][0].cpu().numpy(); sampled_index = int(sampled_scores.argmax())
    sampled_overlap = box_iou(cpu['rois'][0, sampled_index:sampled_index+1].numpy(), truth)[0, 0]
    point_loss, point_accuracy = correspondence_loss(matcher.model, output, batch)
    point_count = int(row['support_point_valid'].sum()+row['query_point_valid'].sum()) if row['positive'] else 0
    return {'index': row['metadata']['index'], 'split': row['metadata']['split'], 'seed': row['metadata']['seed'],
        'family': int(row['metadata']['program']['family']), 'program_id': row['program_id'],
        'positive': bool(row['positive']), 'input_sha256': digest, 'rgb_sha256': row['metadata']['image_sha256'],
        'full_score': float(top['score']), 'full_top1_box': top['box'], 'full_top1_iou': float(full_overlap[0]),
        'generic_proposal_count': encoded['proposal_count'], 'generic_proposal_best_iou': float(generic_overlap.max()),
        'top100_best_iou': float(full_overlap.max()), 'oracle_score': float(oracle_scores[oracle_index]),
        'oracle_top_index': oracle_index, 'oracle_top1_iou': float(oracle_overlap),
        'oracle_target_logit': float(oracle_scores[0]) if row['positive'] else None,
        'oracle_near_logit': float(oracle_scores[1]) if row['positive'] else None,
        'oracle_all_gt_logits': oracle_scores.astype(float).tolist(), 'oracle_roles': row['roles'],
        'sampled_score': float(sampled_scores[sampled_index]), 'sampled_top1_iou': float(sampled_overlap),
        'sampled_roi_logits': sampled_scores.astype(float).tolist(),
        'sampled_roi_quality_targets': cpu['quality_targets'][0].numpy().astype(float).tolist(),
        'point_loss': float(point_loss), 'point_candidate_accuracy': float(point_accuracy),
        'point_candidate_count': point_count, 'valid_homologous_pairs': int(row['point_valid'].sum())}


def main(args):
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    destination = ROOT/args.output
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {'learned': Path(args.checkpoint).resolve()}
    if args.random_checkpoint: paths['random'] = Path(args.random_checkpoint).resolve()
    sources = {name: file_sha(ROOT/name) for name in SOURCES}
    plan = jsonable(PLAN); plan.update({'device': args.device, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_format': DATA_FORMAT, 'checkpoint_sha256': {name: file_sha(path) for name, path in paths.items()},
        'source_sha256': sources, 'natural_images_opened': 0, 'optimizer_updates': 0})
    atomic_json(destination/'plan.json', plan)
    started = time.perf_counter(); results = {}; stream_hashes = {}
    for method, path in paths.items():
        matcher = FrozenDenseMatcher(path, device=args.device, max_side=384, proposal_limit=100, nms_iou=.5, region_batch=128)
        method_dir = destination/method; method_dir.mkdir(); by_split = {}; stream = hashlib.sha256()
        with (method_dir/'records.jsonl').open('x') as ledger:
            for split, count in PLAN['split_counts'].items():
                records = []
                for index in range(count):
                    row = episode(index, split=split, seed=1901, difficulty=1.)
                    record = evaluate_episode(matcher, row); records.append(record)
                    stream.update(split.encode()); stream.update(index.to_bytes(8, 'little')); stream.update(bytes.fromhex(record['input_sha256']))
                    ledger.write(json.dumps(record, allow_nan=False)+'\n')
                    if (index+1)%16 == 0 or index+1 == count:
                        ledger.flush(); print(json.dumps({'method': method, 'split': split, 'completed': index+1,
                            'planned': count, 'elapsed_seconds': time.perf_counter()-started}), flush=True)
                by_split[split] = records
        matcher.assert_unchanged()
        if any(file_sha(ROOT/name) != expected for name, expected in sources.items()): raise RuntimeError('Diagnosis source changed')
        result = {'scope': 'Frozen synthetic diagnosis, no real-world performance claim', 'provenance': matcher.provenance(),
            'input_stream_sha256': stream.hexdigest(), 'records_sha256': file_sha(method_dir/'records.jsonl'),
            **result_summary(by_split)}
        atomic_json(method_dir/'results.json', result); results[method] = result
        stream_hashes[method] = stream.hexdigest(); del matcher
        if args.device == 'cuda': torch.cuda.empty_cache()
    if len(set(stream_hashes.values())) != 1: raise RuntimeError('Random/learned diagnostic inputs differ')
    atomic_json(destination/'comparison.json', {'plan_sha256': file_sha(destination/'plan.json'),
        'same_input_stream': True, 'input_stream_sha256': stream_hashes, 'seconds': time.perf_counter()-started,
        'model_results': results, 'natural_images_opened': 0, 'optimizer_updates': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--random-checkpoint'); parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', default='experiments/synthetic_diagnosis')
    main(parser.parse_args())
