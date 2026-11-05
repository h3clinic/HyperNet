"""In-domain self-supervised pretraining of HyperNet-W v1 on unlabeled whale spectrograms.

Same encoder and objective as HyperNet-G v2 (gravity_ssl_v2). Training data:
whale/corpus.npy (random Bering windows plus DCLDE 2013 28 March windows). No
labels are read for training. Selection: one-reference upcall retrieval on
28 March only (10 references, seed 7; all definite-upcall windows plus 6,000
negatives of that day). Level 0 test days 29 March to 3 April are never used
here; eval_level0.py scores the selected checkpoint afterwards.
"""
import argparse, hashlib, json, math, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from model import SpectroEncoder, token_nn_score, two_channel, PATCH

ROOT = Path(__file__).resolve().parent
W = ROOT.parent
sys.path.insert(0, str(W / 'code'))
import spectro as S, labels as L


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=1))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def selection_set():
    idx = pd.read_csv(W / 'dclde2013' / 'index.csv')
    X = np.load(W / 'dclde2013' / 'windows.npy', mmap_mode='r')
    day = idx.file.str.contains('20090328').values & (idx.riwh.values >= 0)
    pos = np.flatnonzero(day & (idx.riwh.values == 1))
    neg = np.flatnonzero(day & (idx.riwh.values == 0))
    rng = np.random.default_rng(7)
    neg = rng.choice(neg, 6000, replace=False)
    refs = rng.choice(pos, 10, replace=False)
    q = np.sort(np.concatenate([np.setdiff1d(pos, refs), neg]))
    y = idx.riwh.values[q]
    return (np.asarray(X[refs], np.float32) / 255, np.asarray(X[q], np.float32) / 255, y)


@torch.no_grad()
def describe_all(model, images):
    model.eval()
    x = torch.tensor(images)[:, None]
    d = torch.cat([model.describe(x[i:i + 256])[0] for i in range(0, len(x), 256)])
    model.train()
    return d


@torch.no_grad()
def select_score(model, sel):
    refs, q, y = sel
    rd, qd = describe_all(model, refs), describe_all(model, q)
    ap = [average_precision_score(y, token_nn_score(qd, rd[j]).numpy()) for j in range(len(rd))]
    au = [roc_auc_score(y, token_nn_score(qd, rd[j]).numpy()) for j in range(len(rd))]
    return {'map': float(np.mean(ap)), 'macro_auroc': float(np.mean(au))}


def views(batch, rng):
    """Two time-shifted crops (whole-token offsets) with photometric and masking noise.
    No frequency shift and no rotation: those change the glitch class."""
    b = batch.shape[0]
    width = batch.shape[-1] - 2 * PATCH
    o = rng.integers(0, 3, size=(2, b)) * PATCH  # offsets 0, 10, 20 px
    out = []
    for v in range(2):
        x = torch.stack([batch[i, :, o[v, i]:o[v, i] + width] for i in range(b)])[:, None]
        gain = torch.tensor(rng.uniform(.7, 1.4, (b, 1, 1, 1)), dtype=torch.float32)
        gamma = torch.tensor(rng.uniform(.6, 1.6, (b, 1, 1, 1)), dtype=torch.float32)
        x = (x.clamp_min(0) ** gamma) * gain + .03 * torch.randn_like(x)
        cols = torch.tensor(rng.random((b, 1, 1, width // PATCH)) < .12)
        rows = torch.tensor(rng.random((b, 1, 140 // PATCH, 1)) < .06)
        mask = (cols | rows).repeat_interleave(PATCH, -1).repeat_interleave(PATCH, -2)
        x = torch.where(mask, x.mean((-1, -2), keepdim=True), x)
        out.append(x)
    return out[0], out[1], (o[1] - o[0]) // PATCH


def vicreg(a, b, inv=25., var=25., cov=1.):
    n, d = a.shape
    loss_inv = F.mse_loss(a, b)

    def vc(z):
        z = z - z.mean(0)
        std = (z.var(0) + 1e-4).sqrt()
        c = (z.T @ z) / (n - 1)
        off = c - torch.diag(torch.diag(c))
        return F.relu(1 - std).mean(), off.square().sum() / d

    va, ca = vc(a)
    vb, cb = vc(b)
    return inv * loss_inv + var * (va + vb) / 2 + cov * (ca + cb) / 2, float(loss_inv.detach())


def objective(model, x1, x2, shift, rng, mask_ratio, recon_weight):
    r1 = two_channel(x1)
    b = x1.shape[0]
    rows, cols = 140 // PATCH, x1.shape[-1] // PATCH
    mask = torch.tensor(rng.random((b, rows, cols)) < mask_ratio)
    t1 = model.tokens(r1, mask, transformed=True)
    t2 = model.tokens(x2)  # (B,14,15,C)
    g, gi = vicreg(model.global_proj(t1.mean((1, 2))), model.global_proj(t2.mean((1, 2))))
    # View 1 crop starts at o1, view 2 at o2, shift = (o2 - o1) / PATCH.
    # Column c in view 1 covers the same pixels as column c - shift in view 2.
    _, h, w, c = t1.shape
    a, z = [], []
    for i in range(b):
        s = int(shift[i])
        if s >= 0:
            a.append(t1[i, :, s:]); z.append(t2[i, :, :w - s])
        else:
            a.append(t1[i, :, :w + s]); z.append(t2[i, :, -s:])
    a = torch.cat([u.reshape(-1, c) for u in a])
    z = torch.cat([u.reshape(-1, c) for u in z])
    pick = torch.randperm(len(a))[:4096]
    d, di = vicreg(model.dense_proj(a[pick]), model.dense_proj(z[pick]))
    target = F.unfold(r1[:, :1], PATCH, stride=PATCH).transpose(1, 2).reshape(b, rows, cols, PATCH * PATCH)
    recon = F.mse_loss(model.reconstruct(t1[mask]), target[mask])
    loss = g + d + recon_weight * recon
    return loss, {'global': float(g.detach()), 'dense': float(d.detach()), 'recon': float(recon.detach()),
                  'global_inv': gi, 'dense_inv': di}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--width', type=int, default=144)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--mask_ratio', type=float, default=.3)
    ap.add_argument('--recon_weight', type=float, default=10.)
    ap.add_argument('--seed', type=int, default=2402)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = np.load(W / 'corpus.npy', mmap_mode='r')
    sel = selection_set()
    model = SpectroEncoder(width=a.width, depth=a.depth)
    params = sum(p.numel() for p in model.parameters())
    if not a.smoke:
        write('protocol.json', {'format': 'hypernet_w.whale_ssl.v1', 'parameters': params, 'borrowed_weights': False,
                                'labels_read_for_training': 0, 'corpus_windows': int(len(data)),
                                'corpus_sha256': sha(W / 'corpus.npy'),
                                'selection': 'max mean AP, token_nn, 28 March only', 'steps': a.steps,
                                'batch': a.batch, 'lr': a.lr, 'seed': a.seed, 'width': a.width, 'depth': a.depth})
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.05)
    sched = lambda s: min(1, s / 500) * .5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    best, history, start = -1, [], time.time()
    if not a.smoke:
        history.append({'step': 0, 'select': select_score(model, sel)}); write('history.json', history)
    for step in range(1, (3 if a.smoke else a.steps) + 1):
        for g in opt.param_groups:
            g['lr'] = a.lr * sched(step)
        idx = np.sort(rng.integers(0, len(data), a.batch))
        batch = torch.tensor(np.asarray(data[idx]), dtype=torch.float32) / 255.
        x1, x2, shift = views(batch, rng)
        loss, stats = objective(model, x1, x2, shift, rng, a.mask_ratio, a.recon_weight)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.); opt.step()
        if a.smoke:
            print(step, stats, f'{(time.time() - start) / step:.2f}s/step', flush=True); continue
        if step % 50 == 0:
            write('progress.json', {'step': step, 'total': a.steps, 'loss': float(loss.detach()), **stats,
                                    'elapsed_seconds': time.time() - start})
        if step % a.eval_every == 0 or step == a.steps:
            s = select_score(model, sel); history.append({'step': step, 'select': s}); write('history.json', history)
            if s['map'] > best:
                best = s['map']; torch.save(model.state_dict(), ROOT / 'selected.pt')
                write('selection.json', {'step': step, 'select': s})
    if a.smoke:
        print('params', params, 'corpus', len(data), 'select', select_score(model, sel))


if __name__ == '__main__':
    main()
