"""Pretrained-model baselines for one-reference upcall retrieval (DCLDE 2013).

Same protocol as HyperNet-W:
  python bench.py select   # 28 March only: 10 references (seed 7), all readouts
  python bench.py test KEY [KEY ...]  # test days, 20 references (seed 2013), chosen readouts only

Readout keys look like  family:variant, e.g.
  dinov2-large@224:L18:block   DINOv2 tokens from layer 18, sliding box cosine
  dinov2-large@224:L24:cls     CLS-token cosine
  owlv2-base:box:logit         OWLv2 image-guided detection, query = box best matching the call
Spectrogram windows are 140 x 160 uint8 (row 0 = highest frequency), shown to
the models as grey images.
"""
import sys, os, json, time, numpy as np, pandas as pd, torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score as aps, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
os.chdir(os.path.join(HERE, '..', 'code'))
import eval_level0 as E

DEV = 'cuda'
DT = torch.float16
IMNET = (torch.tensor([.485, .456, .406]), torch.tensor([.229, .224, .225]))
CLIP = (torch.tensor([.4815, .4578, .4082]), torch.tensor([.2686, .2613, .2758]))


def to_img(x, hw, norm):
    """x: (B,140,160) float in [0,1] -> (B,3,H,W) normalized."""
    x = F.interpolate(x[:, None].to(DEV, torch.float32), size=hw, mode='bilinear', align_corners=False)
    m, s = norm
    return ((x.repeat(1, 3, 1, 1) - m.to(DEV)[:, None, None]) / s.to(DEV)[:, None, None]).to(DT)


# ---------------------------------------------------------------- DINOv2
class Dino:
    def __init__(self, name, res):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(f'facebook/{name}', torch_dtype=DT).to(DEV).eval()
        self.hw = {224: (224, 252), 448: (448, 518)}[res]
        self.R, self.C = self.hw[0] // 14, self.hw[1] // 14
        n = self.m.config.num_hidden_layers
        self.layers = [n // 2, (3 * n) // 4, n]

    @torch.no_grad()
    def feats(self, x):
        o = self.m(pixel_values=to_img(x, self.hw, IMNET), output_hidden_states=True)
        out = {}
        for L in self.layers:
            h = o.last_hidden_state if L == self.layers[-1] else o.hidden_states[L]
            h = h.float()
            out[L] = (h[:, 0], h[:, 1:])  # register-free DINOv2: CLS + R*C patch tokens
        return out


def token_box(box, R, C):
    a, b, r0, r1 = box
    m = torch.zeros(R, C, dtype=torch.bool, device=DEV)
    m[int(r0 * R / 140):max(int(np.ceil(r1 * R / 140)), int(r0 * R / 140) + 1),
      int(a * C / 160):max(int(np.ceil(b * C / 160)), int(a * C / 160) + 1)] = True
    return m


def block_cos(q, r, mask, R):
    """q: (N, R*C, D) tokens minus image mean; r: (R*C, D); slide the masked reference block along time."""
    return E.block_cosine(q, r, mask, rows=R)


@torch.no_grad()
def dino_scores(model, refs, boxes, batches):
    rt = torch.from_numpy(refs.astype(np.float32) / 255)
    phases = max(1, round(160 / model.C))
    # Reference encodings at sub-token phases.
    ref = {L: [] for L in model.layers}
    for j, (a, b, r0, r1) in enumerate(boxes):
        reps = {L: [] for L in model.layers}
        for p in range(phases):
            if a - p < 0:
                continue
            f = model.feats(torch.roll(rt[j:j + 1], -p, dims=-1))
            for L in model.layers:
                cls, tok = f[L]
                reps[L].append((cls[0], tok[0] - tok[0].mean(0), token_box((a - p, b - p, r0, r1), model.R, model.C), tok[0].mean(0)))
        for L in model.layers:
            ref[L].append(reps[L])
    out = {}
    for q in batches:
        f = model.feats(q)
        for L in model.layers:
            cls, tok = f[L]
            u = tok - tok.mean(1, keepdim=True)
            c_cls = torch.stack([F.cosine_similarity(cls, reps[0][0][None], dim=-1) for reps in ref[L]], 1)
            mean_q = F.normalize(tok.mean(1), dim=-1)
            c_mean = torch.stack([mean_q @ F.normalize(reps[0][3], dim=-1) for reps in ref[L]], 1)
            c_blk = torch.stack([torch.stack([block_cos(u, r, mk, model.R) for _, r, mk, _ in reps]).amax(0)
                                 for reps in ref[L]], 1)
            for k, v in (('cls', c_cls), ('mean', c_mean), ('block', c_blk)):
                out.setdefault(f'L{L}:{k}', []).append(v.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


# ---------------------------------------------------------------- OWLv2
class Owl:
    def __init__(self, name):
        from transformers import Owlv2ForObjectDetection
        self.m = Owlv2ForObjectDetection.from_pretrained(f'google/{name}', torch_dtype=DT).to(DEV).eval()
        self.S = self.m.config.vision_config.image_size  # 960
        self.h = round(self.S * 140 / 160)  # keep aspect; pad the bottom with grey

    def pixels(self, x):
        img = to_img(x, (self.h, self.S), CLIP)
        pad = torch.zeros(len(x), 3, self.S - self.h, self.S, device=DEV, dtype=DT)
        pad += ((.5 - CLIP[0]) / CLIP[1]).to(DEV, DT)[:, None, None]
        return torch.cat([img, pad], 2)

    @torch.no_grad()
    def embed(self, x):
        fm = self.m.image_embedder(pixel_values=self.pixels(x))[0]
        b, h, w, d = fm.shape
        feats = fm.reshape(b, h * w, d)
        return feats, fm

    @torch.no_grad()
    def query(self, x, box):
        feats, fm = self.embed(x)
        _, ce = self.m.class_predictor(feats)
        boxes = self.m.box_predictor(feats, fm)[0]  # (n,4) cx,cy,w,h in [0,1] of the padded square
        obj = self.m.objectness_predictor(feats)[0]
        a, b, r0, r1 = box  # pixel box in the 140x160 window
        tgt = torch.tensor([(a + b) / 2 / 160, (r0 + r1) / 2 / 140 * self.h / self.S,
                            (b - a) / 160, (r1 - r0) / 140 * self.h / self.S], device=DEV)
        iou = box_iou(boxes.float(), tgt[None].float())[:, 0]
        return {'box': ce[0, iou.argmax()], 'objectness': ce[0, obj.argmax()]}

    @torch.no_grad()
    def scores(self, x, qe):
        """qe: (Q, D) query embeddings -> (B, Q) max logit and max cosine over boxes."""
        feats, fm = self.embed(x)
        logits, ce = self.m.class_predictor(feats, qe[None].expand(len(x), -1, -1),
                                            torch.ones(len(x), len(qe), dtype=torch.bool, device=DEV))
        cos = torch.einsum('bnd,qd->bnq', F.normalize(ce.float(), dim=-1), F.normalize(qe.float(), dim=-1))
        return logits.float().amax(1), cos.amax(1)


def box_iou(a, b):
    def xyxy(z):
        return torch.stack([z[:, 0] - z[:, 2] / 2, z[:, 1] - z[:, 3] / 2, z[:, 0] + z[:, 2] / 2, z[:, 1] + z[:, 3] / 2], 1)
    a, b = xyxy(a), xyxy(b)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2]); rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter)


@torch.no_grad()
def owl_scores(model, refs, boxes, batches):
    rt = torch.from_numpy(refs.astype(np.float32) / 255)
    qs = [model.query(rt[j:j + 1], boxes[j]) for j in range(len(refs))]
    out = {}
    for kind in ('box', 'objectness'):
        qe = torch.stack([q[kind] for q in qs])
        lg, cs = [], []
        for x in batches():
            l, c = model.scores(x, qe); lg.append(l.cpu()); cs.append(c.cpu())
        out[f'{kind}:logit'] = torch.cat(lg).numpy(); out[f'{kind}:cos'] = torch.cat(cs).numpy()
    return out


# ---------------------------------------------------------------- data
def selection_set():
    refs, boxes, meta = E.references(seed=7, n=10)
    idx = pd.read_csv(f'{E.R}/index.csv')
    day = idx.file.str.contains('20090328').values & (idx.riwh.values >= 0)
    t = pd.to_datetime(idx.file.str.extract(r'_(\d{8}_\d{6})_')[0], format='%Y%m%d_%H%M%S') + pd.to_timedelta(idx.start_s, 's')
    near = np.zeros(len(idx), bool)
    for s in meta.start:
        near |= (abs((t - s).dt.total_seconds()) < 8).values
    rng = np.random.default_rng(7)
    pos = np.flatnonzero(day & ~near & (idx.riwh.values == 1)); neg = np.flatnonzero(day & ~near & (idx.riwh.values == 0))
    ids = np.sort(np.concatenate([pos, rng.choice(neg, 6000, replace=False)]))
    X = np.load(f'{E.R}/windows.npy', mmap_mode='r')
    return refs, boxes, X, ids, idx.riwh.values[ids]


def batch_iter(X, ids, bs):
    def it():
        for i in range(0, len(ids), bs):
            yield torch.from_numpy(np.asarray(X[ids[i:i + bs]], dtype=np.float32) / 255)
    return it


def run_family(fam, refs, boxes, X, ids, keep=None):
    t0 = time.time()
    if fam.startswith('dinov2'):
        name, res = fam.split('@'); model = Dino(name, int(res))
        bs = 64 if 'giant' in name and res == '448' else 128
        sc = dino_scores(model, refs, boxes, batch_iter(X, ids, bs)())
    else:
        model = Owl(fam); sc = owl_scores(model, refs, boxes, batch_iter(X, ids, 16))
    del model; torch.cuda.empty_cache()
    print(fam, f'{time.time() - t0:.0f}s', flush=True)
    sc = {f'{fam}:{k}': v for k, v in sc.items()}
    return {k: v for k, v in sc.items() if keep is None or k in keep}


FAMILIES = ['dinov2-large@224', 'dinov2-large@448', 'dinov2-giant@224', 'dinov2-giant@448',
            'owlv2-base-patch16-ensemble', 'owlv2-large-patch14-ensemble']


def main():
    mode = sys.argv[1]
    out_dir = os.path.join(HERE, '..')
    if mode == 'select':
        refs, boxes, X, ids, y = selection_set()
        path = os.path.join(out_dir, 'pretrained_selection.json')
        res = json.load(open(path)) if os.path.exists(path) else {}
        fams = sys.argv[2:] or FAMILIES
        for fam in fams:
            if any(k.startswith(fam + ':') for k in res):
                continue
            for k, v in run_family(fam, refs, boxes, X, ids).items():
                res[k] = float(np.mean([aps(y, v[:, j]) for j in range(v.shape[1])]))
                print(k, round(res[k], 4), flush=True)
            json.dump(res, open(path, 'w'), indent=1)
        best = {}
        for k, v in res.items():
            fam = 'dinov2' if k.startswith('dinov2') else 'owlv2'
            if fam not in best or v > res[best[fam]]:
                best[fam] = k
        print('best per family', {f: (k, round(res[k], 4)) for f, k in best.items()})
    else:
        keys = sys.argv[2:]
        refs, boxes, meta = E.references()
        X, ids, y, _ = E.queries()
        res = {}
        for fam in sorted({k.split(':')[0] for k in keys}):
            for k, v in run_family(fam, refs, boxes, X, ids, keep=set(keys)).items():
                res[k] = E.metrics(v, y); np.save(os.path.join(out_dir, f'scores_{k.replace(":", "_").replace("@", "_")}.npy'), v)
                print(k, {m: round(float(np.mean(x)), 4) for m, x in res[k].items()}, flush=True)
        json.dump(res, open(os.path.join(out_dir, 'level0_pretrained_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
