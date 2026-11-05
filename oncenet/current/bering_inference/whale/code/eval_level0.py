import os, importlib.util, numpy as np, torch
from torch.nn import functional as F
import spectro as S

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
