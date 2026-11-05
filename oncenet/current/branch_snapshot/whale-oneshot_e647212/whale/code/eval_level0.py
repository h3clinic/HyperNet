"""Level 0: one-reference upcall retrieval on DCLDE 2013 (protocol fixed in
reports/Whale campaign plan.md). Usage: python eval_level0.py [model_dir ...]
Each model_dir must contain model.py and selected.pt (a SpectroEncoder)."""
import os, sys, json, importlib.util, numpy as np, pandas as pd, torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import spectro as S, labels as L

R = '../dclde2013'; SEED = 2013; NREF = 20
torch.set_num_threads(int(os.environ.get("THREADS", 12)))


def references(seed=SEED, n=NREF):
    d = L.load(R)
    c = d[(d.species == 'RIWH') & (d.confidence == 'Detected') & (d.start.dt.day == 28)]
    c = c.sample(n, random_state=seed)
    out, boxes = [], []
    for _, r in c.iterrows():
        f0 = r.start.floor('15min'); name = f'NOPP6_EST_{f0:%Y%m%d_%H%M%S}_CH10.wav'
        x = S.load_mono(f'{R}/audio/{name}'); p, t = S.log_spectrogram(x); w = S.to_uint8(S.whiten(p))
        mid = ((r.start - f0).total_seconds() + (r.end - r.start).total_seconds() / 2)
        c0 = int(np.clip(np.searchsorted(t, mid) - S.WIN_COLS // 2, 0, w.shape[1] - S.WIN_COLS))
        out.append(w[:, c0:c0 + S.WIN_COLS])
        a = np.searchsorted(t, (r.start - f0).total_seconds()) - c0; b = np.searchsorted(t, (r.end - f0).total_seconds()) - c0
        # Time box (columns, 0.25 s margin) and frequency box (rows, 10 Hz margin; row 0 = highest).
        hz = lambda f: int(np.clip(round((f - S.ROW_LO * S.SR / S.NFFT) / (S.SR / S.NFFT)), 0, S.ROWS - 1))
        r0, r1 = S.ROWS - 1 - hz(r.high + 10), S.ROWS - 1 - hz(r.low - 10)
        boxes.append((max(0, a - 10), min(S.WIN_COLS, b + 10), r0, r1 + 1))
    return np.stack(out), boxes, c


def queries():
    idx = pd.read_csv(f'{R}/index.csv')
    keep = (~idx.file.str.contains('20090328')) & (idx.riwh >= 0)
    X = np.load(f'{R}/windows.npy', mmap_mode='r')
    return X, np.flatnonzero(keep.values), idx.riwh.values[keep.values], idx[keep].reset_index(drop=True)


def batches(X, ids, bs=4096):
    for i in range(0, len(ids), bs):
        yield torch.from_numpy(np.asarray(X[ids[i:i + bs]], dtype=np.float32) / 255)


def std_flat(x):
    x = x.flatten(1); x = x - x.mean(1, keepdim=True); return F.normalize(x, dim=1)


def rank_flat(x):
    x = x.flatten(1); r = x.argsort(1).argsort(1).float(); return std_flat(r)


def score_raw(refs, X, ids, fn):
    rf = fn(torch.from_numpy(refs.astype(np.float32) / 255))
    return torch.cat([fn(q) @ rf.T for q in batches(X, ids)]).numpy()


def score_xcorr(refs, boxes, X, ids, freq_box=False):
    """Normalized cross-correlation of the reference call crop slid along time (Ishmael-style).
    freq_box=True also restricts it to the call's frequency rows."""
    kern, rws = [], []
    for r, (a, b, r0, r1) in zip(refs, boxes):
        lo, hi = (r0, r1) if freq_box else (0, S.ROWS)
        k = torch.from_numpy(r[lo:hi, a:b].astype(np.float32) / 255); k = k - k.mean(); kern.append(k / k.norm()); rws.append((lo, hi))
    out = np.zeros((len(ids), len(refs)), np.float32); o = 0
    for q in batches(X, ids, 2048):
        for j, k in enumerate(kern):
            L_ = k.shape[1]; qq = q[:, rws[j][0]:rws[j][1]]
            num = F.conv1d(qq, k[None])[:, 0]
            s1 = F.conv1d(qq, torch.ones(1, qq.shape[1], L_))[:, 0]
            s2 = F.conv1d(qq * qq, torch.ones(1, qq.shape[1], L_))[:, 0]
            n = qq.shape[1] * L_
            den = (s2 - s1 * s1 / n).clamp_min(1e-6).sqrt()
            out[o:o + len(q), j] = (num / den).amax(1).numpy()
        o += len(q)
    return out


def load_model(d):
    spec = importlib.util.spec_from_file_location('m_' + os.path.basename(d), os.path.join(d, 'model.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ck = torch.load(os.path.join(d, 'selected.pt'), map_location='cpu', weights_only=False)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    width = sd['patch.weight'].shape[0]
    enc = m.SpectroEncoder(width=width); enc.load_state_dict(sd); enc.eval()
    return m, enc


def box_mask(box, rows=14, cols=S.WIN_COLS // 10):
    a, b, r0, r1 = box
    m = torch.zeros(rows, cols, dtype=torch.bool)
    m[r0 // 10:(r1 + 9) // 10, a // 10:(b + 9) // 10] = True
    return m


def aligned_score(qd, rd, mask, rows=14):
    """Slide the reference call's tokens (inside its time-frequency box) along time
    only; score = best mean cosine over whole-token shifts. qd, rd: (.., rows*cols, C)."""
    cols = qd.shape[1] // rows
    q = qd.reshape(len(qd), rows, cols, -1); r = rd.reshape(rows, cols, -1)
    cs = torch.nonzero(mask.any(0)).flatten(); c0, c1 = int(cs.min()), int(cs.max()) + 1
    kr = r[:, c0:c1]; km = mask[:, c0:c1].float(); L_ = c1 - c0
    best = None
    for s in range(cols - L_ + 1):
        v = (torch.einsum('nhwc,hwc->nhw', q[:, :, s:s + L_], kr) * km).sum((1, 2)) / km.sum()
        best = v if best is None else torch.maximum(best, v)
    return best


@torch.no_grad()
def layer_tokens(m, enc, x, layer, norm=True):
    """Tokens (B, rows*cols, C) from a given depth, unit-norm unless norm=False. x: (B,140,W) in [0,1]."""
    if not norm:
        return raw_layer_tokens(m, enc, x, layer)
    x = x[:, None]
    if layer == 'pixels':
        u = F.unfold(x, 10, stride=10).transpose(1, 2); u = u - u.mean(-1, keepdim=True)
        return F.normalize(u, dim=-1)
    if layer == 'desc':
        return enc.describe(x)[0]
    z = enc.patch(m.two_channel(x)) + enc.freq; z = z + enc.local(z)
    t = z.flatten(2).transpose(1, 2)
    if layer == 'local':
        return F.normalize(t, dim=-1)
    for k, blk in enumerate(enc.blocks.layers):
        t = blk(t)
        if layer == f'block{k + 1}':
            return F.normalize(t, dim=-1)
    raise ValueError(layer)


@torch.no_grad()
def raw_layer_tokens(m, enc, x, layer):
    """Un-normalized tokens. Pixels: raw patch values. Learned layers: features
    minus a per-image mean token, so the background contributes ~0 and a call stands out."""
    x = x[:, None]
    if layer == 'pixels':
        return F.unfold(x, 10, stride=10).transpose(1, 2)
    z = enc.patch(m.two_channel(x)) + enc.freq; z = z + enc.local(z)
    t = z.flatten(2).transpose(1, 2)
    if layer != 'local':
        for k, blk in enumerate(enc.blocks.layers):
            t = blk(t)
            if layer == f'block{k + 1}':
                break
        if layer == 'final':
            t = enc.norm(t)
    return t - t.mean(1, keepdim=True)


def block_cosine(qd, rd, mask, rows=14):
    """Slide the reference box block along time; one cosine over the whole block
    (all box tokens concatenated), so energetic tokens dominate, as in cross-correlation."""
    cols = qd.shape[1] // rows
    q = qd.reshape(len(qd), rows, cols, -1); r = rd.reshape(rows, cols, -1)
    cs = torch.nonzero(mask.any(0)).flatten(); c0, c1 = int(cs.min()), int(cs.max()) + 1
    km = mask[:, c0:c1, None].float(); kr = r[:, c0:c1] * km; kr = kr / kr.norm().clamp_min(1e-6)
    L_ = c1 - c0; best = None
    for s in range(cols - L_ + 1):
        qb = q[:, :, s:s + L_] * km
        v = torch.einsum('nhwc,hwc->n', qb, kr) / qb.flatten(1).norm(dim=1).clamp_min(1e-6)
        best = v if best is None else torch.maximum(best, v)
    return best


@torch.no_grad()
def phase_block(m, enc, refs, boxes, X, ids, layer, phases=10):
    rt = torch.from_numpy(refs.astype(np.float32) / 255)
    ref_sets = []
    for j, (a, b, r0, r1) in enumerate(boxes):
        reps = []
        for p in range(phases):
            if a - p < 0:
                continue
            rd = raw_layer_tokens(m, enc, torch.roll(rt[j:j + 1], -p, dims=-1), layer)[0]
            reps.append((rd, box_mask((a - p, b - p, r0, r1))))
        ref_sets.append(reps)
    out = []
    for q in batches(X, ids, 512):
        qd = raw_layer_tokens(m, enc, q, layer)
        out.append(torch.stack([torch.stack([block_cosine(qd, rd, mk) for rd, mk in reps]).amax(0)
                                for reps in ref_sets], 1))
    return torch.cat(out).numpy()


@torch.no_grad()
def phase_aligned(m, enc, refs, boxes, X, ids, layer, phases=10):
    """Aligned score with the reference encoded at every 25 ms sub-token offset,
    so the call can be matched at any column shift, not only whole 250 ms tokens."""
    rt = torch.from_numpy(refs.astype(np.float32) / 255)
    ref_sets = []
    for j, (a, b, r0, r1) in enumerate(boxes):
        reps = []
        for p in range(phases):
            if a - p < 0:
                continue
            rd = layer_tokens(m, enc, torch.roll(rt[j:j + 1], -p, dims=-1), layer)[0]
            reps.append((rd, box_mask((a - p, b - p, r0, r1))))
        ref_sets.append(reps)
    out = []
    for q in batches(X, ids, 512):
        qd = layer_tokens(m, enc, q, layer)
        out.append(torch.stack([torch.stack([aligned_score(qd, rd, mk) for rd, mk in reps]).amax(0)
                                for reps in ref_sets], 1))
    return torch.cat(out).numpy()


@torch.no_grad()
def score_model(d, refs, boxes, X, ids):
    m, enc = load_model(d)
    rd, _ = enc.describe(torch.from_numpy(refs.astype(np.float32) / 255)[:, None])
    # Salient weights: tokens whose time column falls inside the annotated call box.
    masks = [box_mask(b) for b in boxes]
    tn, box, al = [], [], []
    for q in batches(X, ids, 512):
        qd, _ = enc.describe(q[:, None])
        tn.append(torch.stack([m.token_nn_score(qd, rd[j]) for j in range(len(rd))], 1))
        box.append(torch.stack([m.token_nn_score(qd, rd[j], weights=masks[j].flatten().float())
                                for j in range(len(rd))], 1))
        al.append(torch.stack([aligned_score(qd, rd[j], masks[j]) for j in range(len(rd))], 1))
    return {'token_nn': torch.cat(tn).numpy(), 'box_token_nn': torch.cat(box).numpy(),
            'aligned': torch.cat(al).numpy()}


def metrics(scores, y):
    ap = [average_precision_score(y, scores[:, j]) for j in range(scores.shape[1])]
    au = [roc_auc_score(y, scores[:, j]) for j in range(scores.shape[1])]
    p100 = [y[np.argsort(-scores[:, j])[:100]].mean() for j in range(scores.shape[1])]
    p1k = [y[np.argsort(-scores[:, j])[:1000]].mean() for j in range(scores.shape[1])]
    return dict(ap=ap, auroc=au, p100=p100, p1000=p1k)


def boot_diff(a, b, n=5000, seed=0):
    r = np.random.default_rng(seed); a, b = np.asarray(a), np.asarray(b); k = len(a)
    s = [(a[i] - b[i]).mean() for i in r.integers(0, k, (n, k))]
    return [float((a - b).mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    refs, boxes, meta = references()
    X, ids, y, _ = queries()
    print('queries', len(ids), 'positives', int(y.sum()), 'prevalence', y.mean(), flush=True)
    res = {'raw_pixels': metrics(score_raw(refs, X, ids, std_flat), y)}
    print('raw done', flush=True)
    res['raw_rank'] = metrics(score_raw(refs, X, ids, rank_flat), y); print('rank done', flush=True)
    res['xcorr'] = metrics(score_xcorr(refs, boxes, X, ids), y); print('xcorr done', flush=True)
    res['xcorr_box'] = metrics(score_xcorr(refs, boxes, X, ids, freq_box=True), y); print('xcorr_box done', flush=True)
    bases = ['raw_pixels', 'raw_rank', 'xcorr', 'xcorr_box']
    chosen = os.environ.get('CHOSEN')  # e.g. phase_block:local, fixed on 28 March by select_phase.py
    for d in sys.argv[1:]:
        tag = os.path.basename(d.rstrip("/"))
        if chosen:
            m_, enc = load_model(d)
            fn, layer = chosen.split(':')
            fn = {'phase_block': phase_block, 'phase_aligned': phase_aligned}[fn]
            res['block_pixels'] = metrics(phase_block(m_, enc, refs, boxes, X, ids, 'pixels'), y); bases.append('block_pixels')
            print('block_pixels done', flush=True)
            res[f'{tag}:{chosen}'] = metrics(fn(m_, enc, refs, boxes, X, ids, layer), y)
        else:
            for k, v in score_model(d, refs, boxes, X, ids).items():
                res[f'{tag}:{k}'] = metrics(v, y)
        print(d, 'done', flush=True)
    summary = {k: {m: float(np.mean(v[m])) for m in v} for k, v in res.items()}
    best_base = max(bases, key=lambda k: summary[k]['ap'])
    diffs = {k: {m: boot_diff(res[k][m], res[best_base][m]) for m in ('ap', 'auroc')} for k in res if k != best_base}
    out = dict(n_queries=len(ids), positives=int(y.sum()), references=[str(s) for s in meta.start],
               summary=summary, best_baseline=best_base, ap_diff_vs_best_baseline=diffs, per_reference=res)
    tag = os.environ.get('TAG', 'level0')
    json.dump(out, open(f'../{tag}_results.json', 'w'), indent=1)
    print(pd.DataFrame(summary).T.round(4).to_string()); print('vs', best_base, json.dumps(diffs, indent=0))


if __name__ == '__main__':
    main()
