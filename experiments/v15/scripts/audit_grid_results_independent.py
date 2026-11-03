"""Recompute saved Grid image metrics and replay two predetermined TEST images.

The scorer, protocol and checkpoint remain frozen. This independent audit reads
the first sorted normal and defect RGB exactly once each, after completed TEST.
It neither selects thresholds nor fits/tunes any parameters.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet.grid_anomaly import GridAnomaly
from hypernet.mvtec_grid import GridDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(path):
    return Path(path).resolve() if Path(path).is_absolute() else (ROOT / path).resolve()


def jsonl(path, expected):
    need(sha(path) == expected, 'Saved score-file digest changed')
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def verify_bindings(protocol, protocol_path, test, dev):
    need(test['completed'] and test['stage'] == 'test', 'Completed TEST required')
    need(dev['completed'] and dev['stage'] == 'dev' and not dev['prototype'], 'Completed final DEV required')
    need(protocol['model_selection_complete'] and protocol['test_scores_computed_before_freeze'] is False,
         'Invalid pre-TEST freeze')
    need(test['protocol_sha256'] == sha(protocol_path), 'TEST protocol binding differs')
    admission = read(ROOT / 'data/mvtec_grid_v15/test_admission.json')
    need(admission['protocol_sha256'] == sha(protocol_path), 'Admission binding differs')
    for group in ('behavior_files', 'checkpoint_files'):
        for name, digest in protocol[group].items():
            need(sha(resolve(name)) == digest, 'Frozen file differs: ' + name)
    need(test['device'] == protocol['device'] == dev['device'], 'Device differs')
    need(test['batch_size'] == protocol['batch_size'] == dev['batch_size'] == 32, 'Native tile batch differs')
    need(test['method_aliases'] == dev['method_aliases'] == ['learned_final', 'random', 'analytic_detail'],
         'Declared method roster differs')
    need(test['query_images'] == 78 and dev['query_images'] == 64, 'Declared data budget differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/grid_final_test/grid_results.json')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'experiments/grid_protocol_freeze.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier independent audit')
    test = read(args.results)
    protocol = read(args.protocol)
    dev_path = resolve(protocol['dev_results_path'])
    dev = read(dev_path)
    verify_bindings(protocol, args.protocol, test, dev)
    need(sha(dev_path) == protocol['dev_results_sha256'], 'DEV results changed')
    original_result_sha = sha(args.results)
    methods = {}
    saved_rows = {}
    for alias in test['method_aliases']:
        info = test['score_files'][alias]
        rows = jsonl(args.results.parent / info['path'], info['sha256'])
        dev_info = dev['score_files'][alias]
        calibration_rows = jsonl(dev_path.parent / dev_info['path'], dev_info['sha256'])
        need(len(rows) == 78 and len({row['id'] for row in rows}) == 78, 'TEST IDs differ')
        need(len(calibration_rows) == 64 and len({row['id'] for row in calibration_rows}) == 64,
             'DEV IDs differ')
        maximum = max(row['score'] for row in calibration_rows)
        cutoff = float(np.nextafter(np.float64(maximum), np.float64(math.inf)))
        need(cutoff == protocol['thresholds'][alias] == test['thresholds'][alias], 'DEV threshold differs')
        scores = []
        for row in rows:
            path = args.results.parent / row['map_path']
            need(sha(path) == row['map_sha256'], 'Saved TEST map digest changed')
            with np.load(path, allow_pickle=False) as data:
                values = data['patch_scores']
                need(values.dtype == np.float32 and values.ndim == 2 and np.isfinite(values).all(),
                     'Unexpected saved patch map')
                count = max(1, math.ceil(.005 * values.size))
                # Independent full sort, not the scorer's partition implementation.
                score = float(np.sort(values.ravel())[-count:].mean(dtype=np.float64))
                need(data['xy'].shape == (*values.shape, 2), 'Saved coordinates differ')
            need(score == row['score'], 'Top-0.5-percent image score differs from saved map')
            scores.append(score)
        values = np.asarray(scores, dtype=np.float64)
        labels = np.asarray([row['is_anomaly'] for row in rows], dtype=bool)
        decisions = values >= cutoff
        calculated = {'positive_events': int(labels.sum()), 'negative_events': int((~labels).sum()),
                      'true_positive': int((decisions & labels).sum()),
                      'false_positive': int((decisions & ~labels).sum()),
                      'recall': float(np.mean(decisions[labels])),
                      'false_positive_rate': float(np.mean(decisions[~labels])),
                      'false_alarms_per_10000_negatives': 10000 * float(np.mean(decisions[~labels])),
                      'auroc': float(roc_auc_score(labels, values)),
                      'average_precision': float(average_precision_score(labels, values)),
                      'threshold': cutoff}
        reported = test['methods'][alias]['image']
        for key, value in calculated.items():
            need(math.isclose(value, reported[key], rel_tol=0, abs_tol=1e-12), 'Image metric differs: ' + alias + '/' + key)
        methods[alias] = {'image_maps_reaggregated': len(rows), 'calibration_scores_checked': len(calibration_rows),
                          'metrics': calculated, 'all_map_scores_exact': True,
                          'auroc_ap_independent_library': 'sklearn.metrics', 'saved_score_sha256': info['sha256']}
        saved_rows[alias] = {row['id']: row for row in rows}
    roster = saved_rows['learned_final']
    for alias, rows in saved_rows.items():
        need(set(rows) == set(roster), 'Method TEST roster differs')
        need(all(rows[key]['is_anomaly'] == roster[key]['is_anomaly'] for key in roster), 'Labels differ across methods')
    replay_ids = [sorted(key for key, row in roster.items() if not row['is_anomaly'])[0],
                  sorted(key for key, row in roster.items() if row['is_anomaly'])[0]]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = GridDataset(ROOT / 'data/mvtec_grid_v15', development_only=False)
    dataset._test_gate()
    engine = GridAnomaly(protocol['checkpoints'], device=protocol['device'], batch_size=32)
    engine.enroll(dataset.load_rgb(protocol['reference_id']))
    before = engine.provenance()
    need(before == dev['provenance'] == test['provenance'], 'Reference bank/model provenance differs')
    replay = []
    for image_id in replay_ids:
        predicted = engine.predict(dataset.load_rgb(image_id))
        for alias, prediction in predicted.items():
            saved = saved_rows[alias][image_id]
            with np.load(args.results.parent / saved['map_path'], allow_pickle=False) as data:
                exact_map = np.array_equal(prediction['patch_scores'], data['patch_scores'])
                exact_xy = np.array_equal(prediction['xy'], data['xy'])
            need(exact_map and exact_xy and prediction['score'] == saved['score'], 'Replay differs: ' + image_id + '/' + alias)
            replay.append({'image_id': image_id, 'method': alias, 'score': prediction['score'],
                           'patch_map_bit_exact': exact_map, 'coordinates_bit_exact': exact_xy,
                           'saved_map_sha256': saved['map_sha256']})
    need(engine.provenance() == before, 'Replay changed model or reference bank')
    verify_bindings(protocol, args.protocol, test, dev)
    need(sha(args.results) == original_result_sha, 'TEST results changed during audit')
    report = {'format': 'hypernet.grid_results_independent_audit.v15', 'valid': True,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
              'test_results_sha256': original_result_sha, 'protocol_sha256': sha(args.protocol),
              'dev_results_sha256': sha(dev_path), 'methods': methods, 'replays': replay,
              'test_images_replayed': 2, 'reference_images_reloaded': 1,
              'replay_selection': 'First lexicographically sorted normal and first defect ID; fixed before this audit.',
              'real_weight_updates': 0, 'threshold_or_model_selection': False,
              'state_and_banks_unchanged': True,
              'scope': 'All 234 saved image scores independently reaggregated, 192 DEV score rows checked, '
              'image metrics recomputed with sklearn, and six method maps/scores replayed exactly. '
              'Pixel AUROC/AP are outside this independent audit scope.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'valid': True, 'output': str(args.output), 'sha256': sha(args.output),
                      'test_images_replayed': 2, 'methods': methods}))


if __name__ == '__main__':
    main()
