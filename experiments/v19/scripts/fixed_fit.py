"""Disposable fixed-four-episode optimization sanity check, never a benchmark."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys, time, json, math
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hypernet_v19.model import DenseMatchNet, state_hash, file_sha
from hypernet_v19.curriculum import episode
from hypernet_v19.training import collate, objective
from hypernet_v19.proposals import box_iou


def main():
    torch.set_num_threads(2); torch.manual_seed(1919); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sources = {name: file_sha(ROOT/name) for name in ('hypernet_v19/model.py', 'hypernet_v19/training.py', 'hypernet_v19/curriculum.py')}
    rows = [episode(i, split='train', seed=1901, difficulty=.25) for i in range(4)]
    cpu = collate(rows); batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in cpu.items()}
    model = DenseMatchNet().cuda(); initial = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    logs = []; checks = []; start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def check(step):
        model.eval()
        scores = model(batch['support'], batch['query'], batch['rois'])['scores'].cpu().numpy()
        wrong = model(batch['support'].roll(1, 0), batch['query'], batch['rois'])['scores'].cpu().numpy()
        values = []
        for i, row in enumerate(rows):
            rois = cpu['rois'][i].numpy(); overlap = box_iou(rois, row['target_box'][None])[:, 0]
            quality = cpu['quality_targets'][i].numpy(); index = int(scores[i].argmax())
            values.append({'index': i, 'positive': row['positive'], 'target_or_replacement_logit': float(scores[i, 0]),
                'near_negative_logit': float(scores[i, 16]), 'other_logits': [float(scores[i, 20]), float(scores[i, 24])],
                'target_gt_quality': float(quality[0]), 'max_logit': float(scores[i, index]),
                'top1_training_roi_iou': float(overlap[index]), 'top1_training_roi_quality': float(quality[index]),
                'wrong_reference_target_slot_logit': float(wrong[i, 0]),
                'positive_mean_logit': float(scores[i, quality > 0].mean()) if (quality > 0).any() else None,
                'negative_mean_logit': float(scores[i, quality == 0].mean()),
                'sampled_roi_binary_accuracy_at_logit0': float(((scores[i] > 0) == (quality > 0)).mean())})
        check = {'step': step, 'cases': values}; checks.append(check); model.train()
        print(json.dumps(check), flush=True)

    check(0)
    for step in range(1, 129):
        optimizer.zero_grad(set_to_none=True); loss, stats = objective(model, batch, batch['rois'])
        if not math.isfinite(stats['loss']): raise RuntimeError('Nonfinite loss')
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        logs.append({'step': step, 'gradient_norm': float(gradient), **stats})
        if step in (32, 64, 128): check(step)
    assert all(file_sha(ROOT/name) == sha for name, sha in sources.items() if 'curriculum' not in name)
    out = ROOT/'experiments/fixed_fit'; out.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'Fixed-set optimization sanity only, not heldout accuracy; learned tensors discarded',
        'optimizer_updates': 128, 'batch_size': 4, 'unique_episodes': 4, 'rgb_exposures': 1024,
        'episode_indices': [0, 1, 2, 3], 'seed': 1901, 'initialization_seed': 1919, 'difficulty': .25,
        'initial_state_sha256': initial, 'discarded_final_state_sha256': state_hash(model),
        'sources_at_generation': sources, 'curriculum_changed_after_generation': file_sha(ROOT/'hypernet_v19/curriculum.py') != sources['hypernet_v19/curriculum.py'],
        'rgb_sha256': [r['metadata']['image_sha256'] for r in rows],
        'seconds': time.perf_counter()-start, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'history': logs, 'checks': checks, 'checkpoint_saved': False, 'natural_training_images': 0,
        'borrowed_pretrained_weights': False}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'done': True, 'seconds': result['seconds'], 'first': logs[0], 'last': logs[-1]}), flush=True)


if __name__ == '__main__': main()
