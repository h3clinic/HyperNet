"""Fixed-budget mathematical-only v16 training with an actual-consumption ledger."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import torch
from hypernet.variation_curriculum import render, stream, EDGES, FORMAT as CURRICULUM_FORMAT
from hypernet.relational_encoder import RelationalEncoder, FORMAT, objective, state_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def setup(seed=1615, threads=4):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def joined(a, b):
    return {key: torch.cat([a[key], b[key]]) for key in a if isinstance(a[key], torch.Tensor)}


def byte_hashes(values):
    return np.stack([np.frombuffer(hashlib.sha256(value.tobytes()).digest(), np.uint8) for value in values])


def ledger_record(codes, seed_a, seed_b, a, b):
    rgb_a = a['rgb'].cpu().numpy()
    rgb_b = b['rgb'].cpu().numpy()
    masks_a = a['mask'].cpu().numpy().astype(np.uint8)
    masks_b = b['mask'].cpu().numpy().astype(np.uint8)
    return {'type_codes': np.asarray(codes, np.uint32),
            'instance_seeds': np.stack([seed_a, seed_b], 1),
            'rgb_sha256': np.stack([byte_hashes(rgb_a), byte_hashes(rgb_b)], 1),
            'mask_sha256': np.stack([byte_hashes(masks_a), byte_hashes(masks_b)], 1)}


def write_checkpoint(path, model, step, config):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {'format': FORMAT, 'state_dict': state, 'state_sha256': state_hash(model),
               'optimizer_steps': int(step), 'training_views': int(step * config['batch_types'] * 2),
               'consumed_instances': int(step * config['batch_types'] * 2),
               'type_visits': int(step * config['batch_types']),
               'natural_training_images': 0, 'borrowed_pretrained_weights': False,
               'config': config, 'torch_version': str(torch.__version__)}
    torch.save(payload, path)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    if loaded['state_sha256'] != payload['state_sha256']:
        raise RuntimeError('Checkpoint metadata round trip failed')
    for key in state:
        if not torch.equal(state[key], loaded['state_dict'][key]):
            raise RuntimeError('Checkpoint tensor round trip failed')
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': digest(path),
            'state_sha256': payload['state_sha256'], 'optimizer_steps': int(step)}


def preview(a, b, path, step, total_views):
    first = a['rgb'].cpu().numpy()
    second = b['rgb'].cpu().numpy()
    half = len(first) // 2
    count = min(6, half)
    side = first.shape[1]
    canvas = Image.new('RGB', (count * (side + 8), 3 * (side + 28) + 30), '#111827')
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 6), f'ACTUALLY CONSUMED: step {step}, {total_views:,} views', fill='white')
    for row, title in enumerate(('Instance A', 'Same type, independent B', 'Changed relation, same nuisance')):
        for col in range(count):
            source = first[col] if row == 0 else second[col] if row == 1 else first[col + half]
            y = 30 + row * (side + 28)
            draw.text((col * (side + 8), y), title, fill='white')
            canvas.paste(Image.fromarray(source), (col * (side + 8), y + 20))
    temporary = Path(path).with_suffix('.tmp.png')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(temporary)
    temporary.replace(path)


def train(args):
    setup(args.seed, args.threads)
    run = ROOT / args.run
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f'Run directory is nonempty: {run}')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'ledger').mkdir()
    (run / 'source_snapshot').mkdir()
    config = {'steps': args.steps, 'batch_types': args.batch_types, 'image_side': args.size,
              'seed': args.seed, 'stream_seed': args.stream_seed, 'threads': args.threads,
              'device': args.device, 'variant': args.variant, 'learning_rate': args.lr,
              'weight_decay': 0.0001, 'final_lr': 0.00003, 'gradient_clip': 5.0,
              'dtype': 'float32', 'amp': False, 'deterministic_algorithms': True,
              'curriculum_format': CURRICULUM_FORMAT, 'model_format': FORMAT,
              'relations': args.variant == 'primary', 'smoke': bool(args.smoke),
              'natural_training_images': 0, 'borrowed_pretrained_weights': False}
    sources = {}
    for relative in ('hypernet/variation_curriculum.py', 'hypernet/relational_encoder.py',
                     'scripts/train_relations.py'):
        source = ROOT / relative
        target = run / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[relative] = digest(source)
    admission = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                 'sources': sources, 'primary_checkpoint_rule': 'Final fixed-budget step; no TEST selection',
                 'structural_scope': 'Axis-relative 24-edge planar lattice arrangements with independent curved instances; not semantic categories',
                 'ablation': 'Same initialization/stream/architecture/budget, removes only .35 adjacency and .15 degree losses',
                 'views_planned': args.steps * args.batch_types * 2,
                 'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
                 'cuda_device': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None}
    atomic_json(run / 'admission.json', admission)
    model = RelationalEncoder().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    edges = torch.as_tensor(EDGES, device=args.device)
    checkpoints = [write_checkpoint(run / 'random.pt', model, 0, config)]
    history = (run / 'history.jsonl').open('w', encoding='utf-8')
    start = time.perf_counter()
    pending = []
    shards = []
    unique_types = set()
    unique_instances = set()
    unique_rgb = set()
    unique_masks = set()
    kinds_count = {}
    stream_hash = hashlib.sha256()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(args.steps):
            step_start = time.perf_counter()
            codes, seed_a, seed_b, kinds = stream(step, args.batch_types, args.stream_seed)
            a = render(codes, seed_a, args.size, args.device)
            b = render(codes, seed_b, args.size, args.device)
            batch = joined(a, b)
            inputs = batch['rgb'].permute(0, 3, 1, 2).float() / 255
            learning_rate = config['final_lr'] + (args.lr - config['final_lr']) * .5 * (1 + math.cos(math.pi * step / max(1, args.steps - 1)))
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            out = model(inputs)
            loss, stats = objective(model, out, batch, edges, relations=config['relations'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            # Record only successfully consumed optimizer batches, never previews.
            record = ledger_record(codes, seed_a, seed_b, a, b)
            pending.append(record)
            for key in ('type_codes', 'instance_seeds', 'rgb_sha256', 'mask_sha256'):
                stream_hash.update(record[key].tobytes())
            unique_types.update(map(int, codes))
            unique_instances.update((int(code), int(seed)) for code, seeds in zip(codes, record['instance_seeds']) for seed in seeds)
            unique_rgb.update(row.tobytes() for row in record['rgb_sha256'].reshape(-1, 32))
            unique_masks.update(row.tobytes() for row in record['mask_sha256'].reshape(-1, 32))
            for kind in kinds:
                kinds_count[kind] = kinds_count.get(kind, 0) + 1
            completed = step + 1
            if completed % 64 == 0 or completed == args.steps:
                data = {key: np.concatenate([record[key] for record in pending]) for key in pending[0]}
                shard_path = run / 'ledger' / f'steps_{completed-len(pending)+1:06d}_{completed:06d}.npz'
                np.savez_compressed(shard_path, **data)
                shards.append({'path': str(shard_path.relative_to(run)).replace('\\', '/'),
                               'sha256': digest(shard_path), 'type_visits': len(data['type_codes']),
                               'views': 2 * len(data['type_codes'])})
                pending.clear()
            elapsed = time.perf_counter() - start
            row = {'step': completed, 'seconds': elapsed, 'step_seconds': time.perf_counter() - step_start,
                   'learning_rate': learning_rate, 'gradient_norm': float(grad_norm), **stats}
            history.write(json.dumps(row, allow_nan=False) + '\n')
            if completed == 1 or completed % 16 == 0 or completed == args.steps:
                history.flush()
                progress = {'status': 'running', 'run': str(run.relative_to(ROOT)).replace('\\', '/'),
                            'variant': args.variant, 'smoke': bool(args.smoke), 'step': completed,
                            'steps_planned': args.steps, 'consumed_training_views': completed * args.batch_types * 2,
                            'unique_structural_signatures_seen': len(unique_types),
                            'unique_instance_keys_seen': len(unique_instances), 'unique_rgb_seen': len(unique_rgb),
                            'unique_masks_seen': len(unique_masks), 'seconds': elapsed,
                            'estimated_remaining_seconds': elapsed / completed * (args.steps - completed),
                            'last_metrics': row, 'source_hashes': sources}
                atomic_json(run / 'progress.json', progress)
                atomic_json(ROOT / 'training_progress.json', progress)
            if completed == 1 or completed % 128 == 0 or completed == args.steps:
                preview(a, b, ROOT / 'assets' / 'training_latest.png', completed, completed * args.batch_types * 2)
                print(json.dumps({'run': args.run, 'step': completed, 'seconds': round(elapsed, 2),
                                  'loss': stats['loss'], 'type_accuracy': stats['type_retrieval_accuracy']}), flush=True)
            if completed % 1024 == 0 or completed == args.steps:
                filename = 'final.pt' if completed == args.steps else f'recovery_{completed:06d}.pt'
                checkpoints.append(write_checkpoint(run / filename, model, completed, config))
        for relative, expected in sources.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f'Source changed during training: {relative}')
        completion = {'status': 'complete', 'config': config, 'optimizer_steps': args.steps,
                      'training_views': args.steps * args.batch_types * 2,
                      'type_visits': args.steps * args.batch_types,
                      'unique_structural_signatures': len(unique_types),
                      'unique_instance_keys': len(unique_instances), 'unique_rgb_views': len(unique_rgb),
                      'unique_raster_masks': len(unique_masks), 'negative_kind_counts': kinds_count,
                      'input_stream_sha256': stream_hash.hexdigest(), 'shards': shards,
                      'checkpoints': checkpoints, 'sources': sources, 'seconds': time.perf_counter() - start,
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
                      'admission_sha256': digest(run / 'admission.json'), 'history_sha256': digest(run / 'history.jsonl')}
        atomic_json(run / 'completion.json', completion)
        progress.update(status='complete', estimated_remaining_seconds=0, final_checkpoint=checkpoints[-1])
        atomic_json(run / 'progress.json', progress)
        atomic_json(ROOT / 'training_progress.json', progress)
        print(json.dumps(completion, indent=2), flush=True)
    except Exception as error:
        atomic_json(run / 'failure.json', {'error': repr(error), 'seconds': time.perf_counter() - start})
        raise
    finally:
        history.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--variant', choices=('primary', 'no_relation_aux'), default='primary')
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--batch-types', type=int, default=64)
    parser.add_argument('--size', type=int, default=96)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1615)
    parser.add_argument('--stream-seed', type=int, default=1627)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_types < 4 or args.batch_types % 2:
        parser.error('Positive steps and an even batch-types>=4 are required')
    train(args)


if __name__ == '__main__':
    main()
