"""Bounded forward/backward plumbing profile; no optimizer or learned artifact."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys, time, json, argparse
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from hypernet_v19.model import DenseMatchNet, parameter_summary, state_hash
from hypernet_v19.curriculum import episode
from hypernet_v19.training import collate, objective


def main(args):
    torch.set_num_threads(2); torch.manual_seed(1919); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    results = []
    for count in args.batches:
        model = DenseMatchNet().to(args.device); before = state_hash(model)
        rows = [episode(i, split='dev', seed=1991, difficulty=.5) for i in range(count)]
        cpu = collate(rows); batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in cpu.items()}
        if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
        elapsed = []; stats = None
        for step in range(4):
            model.zero_grad(set_to_none=True)
            if args.device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter(); loss, stats = objective(model, batch, batch['rois']); loss.backward()
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not finite: raise RuntimeError('Nonfinite gradients')
            if args.device == 'cuda': torch.cuda.synchronize()
            if step >= 2: elapsed.append(time.perf_counter()-start)
        assert state_hash(model) == before
        record = {'batch_size': count, 'warmed_seconds': elapsed, 'mean_step_seconds': sum(elapsed)/len(elapsed),
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                  'parameters': parameter_summary(model), 'stats': stats, 'state_unchanged': True,
                  'optimizer_updates': 0, 'deterministic_algorithms': True}
        results.append(record); print(json.dumps(record), flush=True)
        del model, batch, cpu, rows, loss
        if args.device == 'cuda': torch.cuda.empty_cache()
    path = ROOT/args.output; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'scope': 'No-optimizer development profile, not trained performance', 'device': args.device,
                               'results': results}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cuda')
    parser.add_argument('--batches', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--output', default='experiments/profile_model.json')
    main(parser.parse_args())
