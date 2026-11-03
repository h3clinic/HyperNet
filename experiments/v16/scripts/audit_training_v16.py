"""Independent full-ledger census and predeclared sampled generator replay.

This never repeats optimization. The CPU stage can audit a completed arm while
the other arm trains. CUDA replay must be explicitly requested after GPU work.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet.variation_curriculum import stream, render, partition

KEYS = ('type_codes', 'instance_seeds', 'rgb_sha256', 'mask_sha256')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check(condition, message):
    if not condition: raise AssertionError(message)


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def hash_rows(values):
    return {row.tobytes() for row in values.reshape(-1, 32)}


def audit_arm(run, plan, replay=False):
    run = Path(run).resolve(); start = time.perf_counter()
    admission_path = run / 'admission.json'; completion_path = run / 'completion.json'
    admission = read(admission_path); completion = read(completion_path); config = admission['config']
    check(completion['status'] == 'complete', 'Only a completed arm can be audited')
    check(completion['config'] == config, 'Completion/admission configuration mismatch')
    check(completion['admission_sha256'] == sha(admission_path), 'Admission hash mismatch')
    check(completion['sources'] == admission['sources'], 'Admission/completion source maps differ')
    check(config['natural_training_images'] == 0 and config['borrowed_pretrained_weights'] is False, 'Unexpected natural/external training')
    check(config['smoke'] is False and config['deterministic_algorithms'] is True, 'Expected actual deterministic training')
    check(config['steps'] == 4096 and config['batch_types'] == 64 and config['image_side'] == 96, 'Budget differs from audited experiment')
    check(config['relations'] == (config['variant'] == 'primary'), 'Objective flag inconsistent with arm')
    bound = {str(admission_path): sha(admission_path), str(completion_path): sha(completion_path)}
    for relative, expected in admission['sources'].items():
        current = ROOT / relative; snapshot = run / 'source_snapshot' / relative
        check(sha(current) == expected == sha(snapshot), 'Frozen source or snapshot differs: ' + relative)
        bound[str(current)] = expected; bound[str(snapshot)] = expected
    history = run / 'history.jsonl'
    check(sha(history) == completion['history_sha256'], 'History hash differs')
    with history.open(encoding='utf-8') as handle: records = [json.loads(line) for line in handle]
    check(len(records) == config['steps'], 'History does not contain one row per optimizer step')
    for index, row in enumerate(records):
        check(row['step'] == index + 1, 'History step order differs')
        check(all(np.isfinite(value) for value in row.values() if isinstance(value, (int, float))), 'Nonfinite history metric')
    bound[str(history)] = sha(history)
    batch = config['batch_types']; steps = config['steps']; expected_views = steps * batch * 2
    check(len(completion['shards']) == 64, 'Expected64 ledger shards')
    known_shards = {str((run / item['path']).resolve()) for item in completion['shards']}
    actual_shards = {str(path.resolve()) for path in (run / 'ledger').glob('*.npz')}
    check(known_shards == actual_shards, 'Unlisted/missing ledger shard')
    stream_hash = hashlib.sha256(); types = set(); instances = set(); rgb = set(); masks = set()
    negative_kinds = Counter(); consumed_steps = 0; samples = []; shard_rows = []
    sample_steps = set(plan['zero_based_steps']); positions = plan['type_positions']
    for item in completion['shards']:
        path = (run / item['path']).resolve()
        check(path.is_relative_to(run), 'Ledger path escapes run')
        check(sha(path) == item['sha256'], 'Ledger hash mismatch: ' + path.name)
        with np.load(path, allow_pickle=False) as archive:
            check(set(archive.files) == set(KEYS), 'Ledger fields differ')
            data = {key: archive[key].copy() for key in KEYS}
        n = len(data['type_codes']); count_steps = n // batch
        check(n == 64 * batch and n == item['type_visits'] and item['views'] == 2 * n, 'Shard budget/count mismatch')
        expected_name = f'steps_{consumed_steps+1:06d}_{consumed_steps+count_steps:06d}.npz'
        check(path.name == expected_name, 'Shard step ranges not sequential')
        check(data['type_codes'].shape == (n,) and data['type_codes'].dtype == np.uint32, 'Wrong code representation')
        check(data['instance_seeds'].shape == (n, 2) and data['instance_seeds'].dtype == np.uint64, 'Wrong seed representation')
        for key in ('rgb_sha256', 'mask_sha256'):
            check(data[key].shape == (n, 2, 32) and data[key].dtype == np.uint8, 'Wrong image digest representation')
        check(np.all(partition(data['type_codes']) != 0), 'Held-out structural signature entered optimizer ledger')
        for offset in range(count_steps):
            step = consumed_steps + offset; section = slice(offset * batch, (offset + 1) * batch)
            codes, seed_a, seed_b, kinds = stream(step, batch, config['stream_seed'])
            check(np.array_equal(data['type_codes'][section], codes), 'Actual type codes differ from frozen deterministic stream')
            check(np.array_equal(data['instance_seeds'][section], np.stack([seed_a, seed_b], axis=1)), 'Instance seed stream differs')
            negative_kinds.update(kinds)
            for key in KEYS: stream_hash.update(data[key][section].tobytes())
            if step in sample_steps:
                for position in positions:
                    i = offset * batch + position
                    for instance in range(2):
                        samples.append({'zero_based_step': step, 'type_position': position,
                            'instance': ('a', 'b')[instance], 'type_code': int(data['type_codes'][i]),
                            'seed': int(data['instance_seeds'][i, instance]),
                            'rgb_sha256': data['rgb_sha256'][i, instance].tobytes().hex(),
                            'mask_sha256': data['mask_sha256'][i, instance].tobytes().hex()})
        types.update(map(int, data['type_codes']))
        instances.update((int(code), int(seed)) for code, pair in zip(data['type_codes'], data['instance_seeds']) for seed in pair)
        rgb.update(hash_rows(data['rgb_sha256'])); masks.update(hash_rows(data['mask_sha256']))
        consumed_steps += count_steps; bound[str(path)] = item['sha256']
        shard_rows.append({'path': str(path.relative_to(run)).replace('\\', '/'), 'sha256': item['sha256'], 'type_visits': n})
    census = {'optimizer_steps': consumed_steps, 'training_views': consumed_steps * batch * 2,
              'type_visits': consumed_steps * batch, 'unique_structural_signatures': len(types),
              'unique_instance_keys': len(instances), 'unique_rgb_views': len(rgb), 'unique_raster_masks': len(masks)}
    for key, actual in census.items(): check(actual == completion[key], 'Recomputed census differs: ' + key)
    check(census['training_views'] == expected_views == admission['views_planned'], 'Planned/actual view count differs')
    check(dict(negative_kinds) == completion['negative_kind_counts'], 'Negative generation accounting differs')
    check(stream_hash.hexdigest() == completion['input_stream_sha256'], 'Step-wise reconstructed input stream hash differs')
    check(len(samples) == plan['sample_count_per_arm'], 'Predeclared replay sample count differs')
    checkpoint_rows = []; reference_shapes = None
    for item in completion['checkpoints']:
        path = (ROOT / item['path']).resolve(); check(path.is_relative_to(run), 'Checkpoint path outside this run')
        check(sha(path) == item['sha256'], 'Checkpoint file hash differs')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        check(saved['format'] == config['model_format'] and saved['config'] == config, 'Checkpoint format/config differs')
        check(saved['natural_training_images'] == 0 and saved['borrowed_pretrained_weights'] is False, 'Checkpoint budget provenance differs')
        check(saved['optimizer_steps'] == item['optimizer_steps'], 'Checkpoint update count differs')
        check(saved['training_views'] == saved['consumed_instances'] == saved['optimizer_steps'] * batch * 2, 'Checkpoint exposure counts differ')
        check(saved['type_visits'] == saved['optimizer_steps'] * batch, 'Checkpoint type visit count differs')
        actual = tensor_digest(saved['state_dict'])
        check(actual == saved['state_sha256'] == item['state_sha256'], 'Checkpoint tensor state hash differs')
        shapes = {key: (list(tensor.shape), str(tensor.dtype)) for key, tensor in saved['state_dict'].items()}
        if reference_shapes is None: reference_shapes = shapes
        check(shapes == reference_shapes, 'Architecture/state shapes changed during training')
        check(all(torch.isfinite(tensor).all() for tensor in saved['state_dict'].values()), 'Nonfinite checkpoint tensor')
        bound[str(path)] = item['sha256']
        checkpoint_rows.append({'path': item['path'], 'sha256': item['sha256'], 'state_sha256': actual, 'optimizer_steps': item['optimizer_steps']})
    check([row['optimizer_steps'] for row in checkpoint_rows] == [0, 1024, 2048, 3072, 4096], 'Unexpected checkpoint roster')
    replay_result = {'performed': False, 'reason': 'CPU census only; CUDA pixel replay is separately gated.'}
    if replay:
        check(config['device'] == 'cuda' and torch.cuda.is_available(), 'Original CUDA renderer is required')
        check(str(torch.__version__) == admission['torch_version'], 'Replay PyTorch runtime differs')
        check(str(np.__version__) == admission['numpy_version'], 'Replay NumPy runtime differs')
        check(torch.cuda.get_device_name(0) == admission['cuda_device'], 'Replay GPU differs')
        codes = np.array([row['type_code'] for row in samples], np.uint32)
        seeds = np.array([row['seed'] for row in samples], np.uint64)
        actual = render(codes, seeds, config['image_side'], config['device'])
        actual_rgb = actual['rgb'].cpu().numpy(); actual_mask = actual['mask'].cpu().numpy().astype(np.uint8)
        for index, sample in enumerate(samples):
            check(hashlib.sha256(actual_rgb[index].tobytes()).hexdigest() == sample['rgb_sha256'], 'Sampled RGB replay differs')
            check(hashlib.sha256(actual_mask[index].tobytes()).hexdigest() == sample['mask_sha256'], 'Sampled mask replay differs')
        replay_result = {'performed': True, 'passed': True, 'instances': len(samples), 'rgb_and_mask_hashes_checked': len(samples) * 2,
                         'device': config['device'], 'torch': str(torch.__version__), 'batch_size': len(samples),
                         'scope': 'Renderer output replay only; no optimizer steps or model predictions.'}
    for path, expected in bound.items(): check(sha(path) == expected, 'Audited file changed during audit: ' + path)
    return {'passed': True, 'run': str(run.relative_to(ROOT)).replace('\\', '/'), 'config': config,
            'census': census, 'negative_kind_counts': dict(negative_kinds), 'input_stream_sha256': stream_hash.hexdigest(),
            'shards': shard_rows, 'checkpoints': checkpoint_rows, 'sources': admission['sources'],
            'sampled_replay': replay_result, 'declared_samples': samples, 'bound_files': bound,
            'seconds': time.perf_counter() - start}


def audit(primary, ablation, plan_path, replay=False, replay_evidence=None):
    plan = read(plan_path); expected = [str((ROOT / path).resolve()) for path in plan['arms']]
    actual = [str(Path(primary).resolve())] + ([str(Path(ablation).resolve())] if ablation else [])
    check(actual == expected[:len(actual)], 'Requested arms differ from replay plan')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if replay and replay_evidence: raise ValueError('Choose direct replay or previously completed replay evidence')
    arms = [audit_arm(path, plan, replay) for path in actual]
    replay_binding = None
    if replay_evidence:
        evidence = read(replay_evidence)
        check(evidence['passed'] is True and evidence['plan_sha256'] == sha(plan_path), 'Replay evidence/plan mismatch')
        check(evidence['script_sha256'] == sha(ROOT / 'scripts/replay_training_samples_v16.py'), 'Replay source changed')
        check(len(evidence['arms']) == len(arms), 'Replay arm roster differs')
        for row, recorded in zip(arms, evidence['arms']):
            check(recorded['passed'] is True and recorded['run'] == row['run'], 'Replay arm identity differs')
            check(recorded['samples'] == row['declared_samples'], 'Replayed samples do not equal independently rebuilt ledger samples')
            check(recorded['instances'] == len(row['declared_samples']), 'Replay instance count differs')
            row['sampled_replay'] = {'performed': True, 'passed': True, 'instances': recorded['instances'],
                                    'rgb_and_mask_hashes_checked': recorded['rgb_and_mask_hashes_checked'],
                                    'evidence_sha256': sha(replay_evidence), 'scope': evidence['scope']}
        for path, expected in evidence['bound_files'].items():
            check(sha(path) == expected, 'Replay-bound artifact changed: ' + path)
        replay_binding = {'path': str(Path(replay_evidence).resolve()), 'sha256': sha(replay_evidence)}
    comparison = {'available': False, 'reason': 'Only the completed primary arm was requested'}
    if len(arms) == 2:
        p, a = arms; differences = {key: [p['config'][key], a['config'][key]] for key in p['config'] if p['config'][key] != a['config'][key]}
        check(differences == {'variant': ['primary', 'no_relation_aux'], 'relations': [True, False]}, 'Unexpected between-arm configuration differences')
        check(p['sources'] == a['sources'], 'Between-arm frozen source differences')
        check(p['census'] == a['census'] and p['negative_kind_counts'] == a['negative_kind_counts'], 'Between-arm data census differs')
        check(p['input_stream_sha256'] == a['input_stream_sha256'], 'Actual RGB/mask/type/seed input streams differ')
        check(p['checkpoints'][0]['state_sha256'] == a['checkpoints'][0]['state_sha256'], 'Initial tensor states differ')
        check(p['checkpoints'][-1]['state_sha256'] != a['checkpoints'][-1]['state_sha256'], 'Final arm states unexpectedly identical')
        comparison = {'available': True, 'passed': True, 'config_differences': differences,
                      'identical_initial_state_sha256': p['checkpoints'][0]['state_sha256'],
                      'identical_actual_input_stream_sha256': p['input_stream_sha256'],
                      'distinct_unique_rendered_images_across_arms': p['census']['unique_rgb_views'],
                      'optimizer_image_exposures_across_arms': p['census']['training_views'] + a['census']['training_views']}
    return {'format': 'hypernet.v16.independent_training_audit.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'passed': True, 'scope': 'All saved ledger/source/checkpoint identity and census; optional predeclared small renderer replay. No optimizer-trajectory replay.',
            'arms': arms, 'comparison': comparison, 'plan_sha256': sha(plan_path), 'audit_script_sha256': sha(__file__),
            'separate_renderer_replay_evidence': replay_binding,
            'no_natural_image_access': True, 'model_inference_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, default=ROOT / 'training/primary_4096')
    parser.add_argument('--ablation', type=Path)
    parser.add_argument('--plan', type=Path, default=ROOT / 'experiments/training_audit_replay_plan.json')
    parser.add_argument('--replay', action='store_true', help='Only after training/diagnostic GPU owners release GPU')
    parser.add_argument('--replay-evidence', type=Path, help='Bind previously completed exact renderer replay; no GPU calls')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous audits; choose a new output')
    result = audit(args.primary, args.ablation, args.plan, args.replay, args.replay_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle: json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'passed': True, 'output': str(args.output), 'sha256': sha(args.output), 'arms': len(result['arms']), 'replay': args.replay}))


if __name__ == '__main__': main()
