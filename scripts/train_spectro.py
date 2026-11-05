"""Train the v0.14 part encoder on the synthetic time-frequency curriculum.

Two objectives, same data, same architecture, same random start:

  --objective family    Positives are INDEPENDENT members of one family. Loss =
                        cross-entropy over family ids + supervised contrastive
                        on the global embedding (all same-family members are
                        positives) + 0.25 x foreground mask loss.
  --objective instance  The v0.14 run02 loss unchanged (matching_loss_v2:
                        InfoNCE on two augmented views of ONE member, local
                        Sinkhorn matching, mask, grammar heads, variance and
                        covariance). This arm isolates the data change.

Members are rendered on the fly, so no two steps see the same pixels. No real
spectrogram, photograph or borrowed weight is used. Checkpoints load with
PartDetector (architecture_revision v2_moderate); the family head is dropped
from the saved state so the public encoder class matches.

    python scripts/train_spectro.py --objective family --output <new folder>
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys, time
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet import spectro_curriculum as sc
from hypernet.part_encoder import FORMAT, crop_image, state_hash
from hypernet.part_encoder_v2 import PartEncoderV2, matching_loss_v2, ARCHITECTURE_REVISION

TRAIN_FAMILIES = sc.FAMILY_COUNT - sc.HELDOUT_FAMILIES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SpectroEncoder(PartEncoderV2):
    def __init__(self, families):
        super().__init__()
        self.family_head = nn.Linear(128, families)


def member_tensor(images, masks):
    """Render path identical to inference: scalar field -> uint8 RGB -> crop_image -> float."""
    rgb, mm = [], []
    for image, mask in zip(images, masks):
        crop, mapped, _ = crop_image(sc.to_rgb_uint8(image), mask=mask)
        rgb.append(crop); mm.append(mapped)
    x = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float() / 255
    return x, torch.from_numpy(np.stack(mm)).float()


def mask_loss(output, truth):
    truth = F.interpolate(truth[:, None], output['mask_logits'].shape[-2:], mode='area')
    logits = output['mask_logits']; prob = logits.sigmoid()
    dice = 1 - (2 * (prob * truth).sum((1, 2, 3)) + 1) / (prob.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, truth) + dice.mean()


def supcon(embedding, labels, temperature=.1):
    """Supervised contrastive loss: every other member of the same family is a positive."""
    sim = embedding @ embedding.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * positive).sum(1).div(positive.sum(1).clamp_min(1)).mean()


def family_loss(model, x, masks, labels):
    out = model(x)
    ce = F.cross_entropy(model.family_head(out['raw']), labels)
    con = supcon(out['global'], labels)
    mask = mask_loss(out, masks)
    loss = ce + con + .25 * mask
    stats = {'loss': float(loss), 'family_ce': float(ce), 'supcon': float(con), 'mask_loss': float(mask),
             'family_accuracy': float((model.family_head(out['raw']).argmax(1) == labels).float().mean())}
    return loss, stats


def instance_loss(model, images, masks, families, rng):
    """run02 objective: two physics-safe augmentations of each single member."""
    a = [sc.augment(im, rng) for im in images]; b = [sc.augment(im, rng) for im in images]
    xa, ma = member_tensor(a, masks); xb, mb = member_tensor(b, masks)
    out = model(torch.cat([xa, xb])); n = len(xa)
    first = {k: (v[:n] if torch.is_tensor(v) else [t[:n] for t in v]) for k, v in out.items()}
    second = {k: (v[n:] if torch.is_tensor(v) else [t[n:] for t in v]) for k, v in out.items()}
    factors = torch.tensor([[(int(f) // power) % 8 for power in (512, 64, 8, 1)] for f in families])
    return matching_loss_v2(first, second, ma, mb, factors)


@torch.inference_mode()
def heldout_retrieval(model, step_seed, ways=16, members=2):
    """Synthetic check: nearest-neighbour family retrieval among held-out families."""
    rng = np.random.default_rng(step_seed); hits = 0; total = 0
    for rep in range(8):
        fams = rng.choice(np.arange(TRAIN_FAMILIES, sc.FAMILY_COUNT), ways, replace=False)
        images, masks, labels = [], [], []
        for f in fams:
            recipe = sc.family_recipe(f)
            for m in range(members):
                im, mk = sc.render_member(recipe, int(rng.integers(1 << 31))); images.append(im); masks.append(mk); labels.append(int(f))
        x, _ = member_tensor(images, masks); emb = model(x)['global']; labels = np.array(labels)
        sim = (emb @ emb.T).numpy(); np.fill_diagonal(sim, -9)
        hits += int((labels[sim.argmax(1)] == labels).sum()); total += len(labels)
    return hits / total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--objective', choices=('family', 'instance'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2688); parser.add_argument('--checkpoint-every', type=int, default=224)
    parser.add_argument('--batch-families', type=int, default=32); parser.add_argument('--members', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4); parser.add_argument('--seed', type=int, default=1403)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh run directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
    model = SpectroEncoder(sc.FAMILY_COUNT); initial = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=.00005)
    plan = {'format': FORMAT, 'architecture_revision': ARCHITECTURE_REVISION, 'curriculum': sc.FORMAT,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'objective': args.objective, 'seed': args.seed,
            'steps': args.steps, 'batch_families': args.batch_families, 'members_per_family': args.members,
            'images_per_step': args.batch_families * args.members * (2 if args.objective == 'instance' else 1),
            'train_families': TRAIN_FAMILIES, 'heldout_families': sc.HELDOUT_FAMILIES,
            'curriculum_sha256': sha(ROOT / 'hypernet/spectro_curriculum.py'), 'encoder_sha256': sha(ROOT / 'hypernet/part_encoder.py'),
            'encoder_v2_sha256': sha(ROOT / 'hypernet/part_encoder_v2.py'), 'training_script_sha256': sha(__file__),
            'initial_state_sha256': initial, 'initialization': 'torch random initialization, no checkpoint loaded',
            'parameters': sum(p.numel() for p in model.parameters()), 'borrowed_weights': False,
            'natural_training_images': 0, 'spectrograms_used_in_training': 0,
            'positives': 'independent members of one family' if args.objective == 'family' else 'two augmented views of one member',
            'loss': {'family_ce': 1, 'supcon': 1, 'mask': .25} if args.objective == 'family' else 'matching_loss_v2 unchanged',
            'augmentation': 'time shift, time mask, frequency-band gain, gain; no flips, rotations or colour changes'}
    (args.output / 'training_plan.json').write_text(json.dumps(plan, indent=2))
    rng = np.random.default_rng(args.seed + 1); started = time.time(); history = []; seen = 0
    with (args.output / 'training_log.jsonl').open('x') as log:
        for step in range(1, args.steps + 1):
            model.train()
            fams = rng.choice(TRAIN_FAMILIES, args.batch_families, replace=False)
            images, masks, labels = [], [], []
            for f in fams:
                recipe = sc.family_recipe(f)
                for m in range(args.members):
                    im, mk = sc.render_member(recipe, int(rng.integers(1 << 31)))
                    images.append(sc.augment(im, rng) if args.objective == 'family' else im); masks.append(mk); labels.append(int(f))
            seen += len(images)
            optimizer.zero_grad(set_to_none=True)
            if args.objective == 'family':
                x, mm = member_tensor(images, masks); loss, stats = family_loss(model, x, mm, torch.tensor(labels))
            else:
                loss, stats = instance_loss(model, images, masks, labels, rng)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step(); schedule.step()
            if step % 25 == 0:
                entry = {'step': step, 'seconds': round(time.time() - started, 1), **stats}
                log.write(json.dumps(entry) + '\n'); log.flush(); print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                model.eval(); retrieval = heldout_retrieval(model, 777 + step)
                epoch = step // args.checkpoint_every
                history.append({'step': step, 'epoch': epoch, 'images_seen': seen, 'seconds': time.time() - started,
                                'heldout_16way_retrieval': retrieval, **stats})
                (args.output / 'history.json').write_text(json.dumps(history, indent=2))
                state = {k: v for k, v in model.state_dict().items() if not k.startswith('family_head')}
                probe = PartEncoderV2(); probe.load_state_dict(state, strict=True)
                torch.save({'format': FORMAT, 'architecture_revision': ARCHITECTURE_REVISION, 'state_dict': state,
                            'state_sha256': state_hash(probe), 'epoch': epoch, 'training_steps': step, 'image_exposures': seen,
                            'curriculum': sc.FORMAT, 'objective': args.objective, 'natural_training_images': 0, 'borrowed_weights': False,
                            'training_plan_sha256': sha(args.output / 'training_plan.json')}, args.output / f'epoch_{epoch:02d}.pt')
                print(json.dumps(history[-1]), flush=True)
    (args.output / 'completed.json').write_text(json.dumps({'completed': True, 'steps': args.steps, 'images_seen': seen,
                                                            'final_state_sha256': state_hash(model), 'seconds': time.time() - started}, indent=2))


if __name__ == '__main__':
    main()
