"""One synthetic scene timing, random model, no performance claim or updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys, time, json, argparse
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cv2
import torch
from hypernet_v19.model import DenseMatchNet, state_hash, FORMAT
from hypernet_v19.matcher import FrozenDenseMatcher
from hypernet_v19.curriculum import episode


def main(args):
    torch.set_num_threads(2); torch.manual_seed(1919); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    out = ROOT/args.run; out.mkdir(parents=True, exist_ok=True)
    path = out/'random.pt'
    if path.exists(): raise FileExistsError(path)
    model = DenseMatchNet()
    torch.save({'format': FORMAT, 'state_dict': model.state_dict(), 'state_sha256': state_hash(model),
                'natural_training_images': 0, 'borrowed_pretrained_weights': False,
                'training': {'optimizer_updates': 0, 'scope': 'Random initialized timing only'}}, path)
    matcher = FrozenDenseMatcher(path, device='cuda')
    examples = [episode(i, split='dev', seed=1993, difficulty=.5) for i in range(12)]
    # Deliberately a rectangular scene to measure the real evaluator's shape,
    # not a test of detection or scientific transfer.
    query = cv2.resize(examples[0]['query'], (384, 256), interpolation=cv2.INTER_LINEAR)
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    banks = [matcher.enroll(row['support']) for row in examples]
    torch.cuda.synchronize(); enroll_seconds = time.perf_counter()-start
    start = time.perf_counter(); encoded = matcher.encode_query(query); torch.cuda.synchronize()
    encode_seconds = time.perf_counter()-start
    start = time.perf_counter(); detections = matcher.predict_many(banks, encoded); torch.cuda.synchronize()
    rank_seconds = time.perf_counter()-start
    matcher.assert_unchanged()
    result = {'scope': 'Random model timing on one procedural scene; not accuracy',
        'optimizer_updates': 0, 'reference_count': 12, 'query_shape': list(query.shape),
        'proposal_count_before_nms': encoded['proposal_count'], 'enroll_seconds': enroll_seconds,
        'encode_query_seconds': encode_seconds, 'rank_all12_seconds': rank_seconds,
        'seconds_per_query12refs_excluding_enrollment': encode_seconds+rank_seconds,
        'detections_per_reference': [len(x) for x in detections],
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'provenance': matcher.provenance()}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', default='experiments/profile_inference')
    main(parser.parse_args())
