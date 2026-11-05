"""HyperNet-W v2: call-aware self-supervised pretraining on unlabeled whale spectrograms.

v1 learned the background texture: its transformer layers scored worse than its
first layer. v2 teaches the network what a sound event is, without labels:

1. Event bank (code/build_events.py): the largest blob that stands clearly above
   the whitened background in each unlabeled window.
2. Two views: the same event pasted (pixel max, random gain 0.5 to 1.2) into two
   different random unlabeled backgrounds, at random times, same frequency rows
   (a frequency shift changes the call type).
3. Event loss (InfoNCE across the batch): pooled tokens inside the event box,
   minus the image mean token, must identify the same event in the other view.
   This is the block-cosine matcher's own quantity, so training matches scoring.
4. Dense loss (when both pastes share the same sub-token phase): each event token
   must pick its partner among all 224 tokens of the other view, so event tokens
   stay distinct from background tokens.
5. Masked-patch reconstruction (from v1) keeps low-level detail.

No labels are read. Checkpoints are saved every 500 steps; select_v2.py picks the
checkpoint and depth on 28 March only.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from model import SpectroEncoder, two_channel, PATCH

ROOT = Path(__file__).resolve().parent
W = ROOT.parent
ROWS, COLS = 140 // PATCH, 160 // PATCH


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=1))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def load_events():
    crops = np.load(W / 'events_crops.npy', allow_pickle=True)
    boxes = np.load(W / 'events_boxes.npy')
    keep = (boxes[:, 2] > 0) & (boxes[:, 3] < 160)  # drop blobs cut off at the window edge
    bottom = boxes[:, 1] >= 139
    p = np.where(bottom, .3 / max(1, (keep & bottom).sum()), .7 / max(1, (keep & ~bottom).sum())) * keep
    return crops, boxes, p / p.sum()


def paste(bg, crop, r0, c, gain):
    x = bg.copy(); h, w = crop.shape
    x[r0:r0 + h, c:c + w] = np.maximum(x[r0:r0 + h, c:c + w], crop * gain)
    return x


def make_batch(data, crops, boxes, p, b, rng):
    ev = rng.choice(len(p), b, p=p)
    bgs = np.asarray(data[np.sort(rng.integers(0, len(data), 2 * b))], np.float32) / 255
    rng.shuffle(bgs)
    x = np.zeros((2, b, 140, 160), np.float32); box = np.zeros((2, b, 4), np.int64)
    for i, e in enumerate(ev):
        crop = crops[e].astype(np.float32) / 255; r0 = int(boxes[e][0]); h, w = crop.shape
        c1 = int(rng.integers(0, 160 - w + 1))
        if rng.random() < .5:  # same sub-token phase: dense correspondence is exact
            c2 = int(rng.choice(np.arange(c1 % PATCH, 160 - w + 1, PATCH)))
        else:
            c2 = int(rng.integers(0, 160 - w + 1))
        for v, c in enumerate((c1, c2)):
            x[v, i] = paste(bgs[2 * i + v], crop, r0, c, rng.uniform(.5, 1.2))
            box[v, i] = (r0, r0 + h, c, c + w)
    x = torch.from_numpy(x)[:, :, None]
    gamma = torch.tensor(rng.uniform(.8, 1.25, (2, b, 1, 1, 1)), dtype=torch.float32)
    x = x.clamp_min(0) ** gamma + .02 * torch.randn_like(x)
    return x[0], x[1], box


def coverage(box):
    """(B, ROWS, COLS) fraction of each token's pixels inside the event box."""
    m = torch.zeros(len(box), 140, 160)
    for i, (r0, r1, c0, c1) in enumerate(box):
        m[i, r0:r1, c0:c1] = 1
    return F.avg_pool2d(m[:, None], PATCH)[:, 0]


def info_nce(a, b, tau):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    lg = a @ b.T / tau; t = torch.arange(len(a))
    return (F.cross_entropy(lg, t) + F.cross_entropy(lg.T, t)) / 2, float((lg.argmax(1) == t).float().mean())


def objective(model, x1, x2, box, rng, a):
    b = x1.shape[0]
    r1 = two_channel(x1)
    cov1, cov2 = coverage(box[0]), coverage(box[1])
    mask = torch.tensor(rng.random((b, ROWS, COLS)) < a.mask_ratio) & (cov1 < .05)  # never hide the event
    t1 = model.tokens(r1, mask, transformed=True); t2 = model.tokens(x2)
    u1 = t1 - t1.mean((1, 2), keepdim=True); u2 = t2 - t2.mean((1, 2), keepdim=True)
    e1 = (u1 * cov1[..., None]).sum((1, 2)) / cov1.sum((1, 2))[:, None]
    e2 = (u2 * cov2[..., None]).sum((1, 2)) / cov2.sum((1, 2))[:, None]
    ev, ev_acc = info_nce(model.global_proj(e1), model.global_proj(e2), a.tau)
    # Dense: same-phase pairs only; token (r, c) in view 1 <-> (r, c + shift) in view 2.
    dl, dn, dacc = 0., 0, []
    z1, z2 = F.normalize(model.dense_proj(u1), dim=-1), F.normalize(model.dense_proj(u2), dim=-1)
    for i in range(b):
        d = int(box[1, i, 2] - box[0, i, 2])
        if d % PATCH:
            continue
        s = d // PATCH
        rr, cc = torch.nonzero(cov1[i] > .2, as_tuple=True)
        ok = (cc + s >= 0) & (cc + s < COLS); rr, cc = rr[ok], cc[ok]
        if len(rr) == 0:
            continue
        lg = z1[i, rr, cc] @ z2[i].reshape(-1, z2.shape[-1]).T / a.tau
        tgt = rr * COLS + cc + s
        dl = dl + F.cross_entropy(lg, tgt, reduction='sum'); dn += len(rr)
        dacc.append(float((lg.argmax(1) == tgt).float().mean()))
    dl = dl / max(dn, 1)
    target = F.unfold(r1[:, :1], PATCH, stride=PATCH).transpose(1, 2).reshape(b, ROWS, COLS, PATCH * PATCH)
    recon = F.mse_loss(model.reconstruct(t1[mask]), target[mask]) if mask.any() else torch.zeros(())
    loss = ev + a.dense_weight * dl + a.recon_weight * recon
    return loss, {'event': float(ev.detach()), 'event_acc': ev_acc, 'dense': float(dl.detach()) if dn else 0.,
                  'dense_acc': float(np.mean(dacc)) if dacc else 0., 'recon': float(recon.detach())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--width', type=int, default=144)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--tau', type=float, default=.1)
    ap.add_argument('--mask_ratio', type=float, default=.3)
    ap.add_argument('--dense_weight', type=float, default=1.)
    ap.add_argument('--recon_weight', type=float, default=5.)
    ap.add_argument('--seed', type=int, default=2403)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = np.load(W / 'corpus.npy', mmap_mode='r')
    crops, boxes, p = load_events()
    model = SpectroEncoder(width=a.width, depth=a.depth)
    params = sum(q.numel() for q in model.parameters())
    if not a.smoke:
        write('protocol.json', {'format': 'hypernet_w.whale_ssl.v2', 'parameters': params, 'borrowed_weights': False,
                                'init': 'random', 'labels_read_for_training': 0, 'corpus_windows': int(len(data)),
                                'corpus_sha256': sha(W / 'corpus.npy'), 'events': int((p > 0).sum()),
                                'objective': 'event InfoNCE across backgrounds + dense same-phase InfoNCE + masked recon',
                                'selection': 'select_v2.py: checkpoint x depth by mean AP, phase_block, 28 March only',
                                **{k: v for k, v in vars(a).items() if k != 'smoke'}})
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.05)
    sched = lambda s: min(1, s / 300) * .5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    start = time.time(); log = []
    for step in range(1, (3 if a.smoke else a.steps) + 1):
        for g in opt.param_groups:
            g['lr'] = a.lr * sched(step)
        x1, x2, box = make_batch(data, crops, boxes, p, a.batch, rng)
        loss, stats = objective(model, x1, x2, box, rng, a)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.); opt.step()
        if a.smoke:
            print(step, stats, f'{(time.time() - start) / step:.2f}s/step', flush=True); continue
        if step % 50 == 0:
            log.append({'step': step, 'loss': float(loss.detach()), **stats}); write('history.json', log)
            write('progress.json', {'step': step, 'total': a.steps, 'elapsed_seconds': time.time() - start, **log[-1]})
        if step % a.save_every == 0 or step == a.steps:
            torch.save(model.state_dict(), ROOT / f'ckpt_{step:05d}.pt')
    if a.smoke:
        print('params', params, 'events', int((p > 0).sum()))


if __name__ == '__main__':
    main()
