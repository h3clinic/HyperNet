"""One-shot search on the GPU: one example call -> score every 4 s window.

Methods (all from the single example, no labels):
  v3      HyperNet-W v3 final layer, sliding block cosine (the Level 0 winner)
  pixels  same matcher on raw 10x10 pixel patches
  xcorr   normalized cross-correlation of the call's time-frequency box slid along time
The example is given as (window, box): a 140 x 160 uint8 spectrogram window and
the call's box (col_start, col_end, row_top, row_bottom) in that window.
"""
import os, numpy as np, torch
from torch.nn import functional as F
import spectro as S
import eval_level0 as E

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, '..', 'whale_ssl_v3')


def hz_row(f):
    """Row index (0 = highest) of frequency f in Hz."""
    return S.ROWS - 1 - int(np.clip(round((f - S.ROW_LO * S.SR / S.NFFT) / (S.SR / S.NFFT)), 0, S.ROWS - 1))


def example_from_audio(path, start_s, end_s, low_hz, high_hz, pad_s=0.25, pad_hz=10):
    """Window centred on a call in an audio file, plus its box."""
    x = S.load_mono(path); p, t = S.log_spectrogram(x); w = S.to_uint8(S.whiten(p))
    mid = (start_s + end_s) / 2
    c0 = int(np.clip(np.searchsorted(t, mid) - S.WIN_COLS // 2, 0, w.shape[1] - S.WIN_COLS))
    a = max(0, np.searchsorted(t, start_s - pad_s) - c0); b = min(S.WIN_COLS, np.searchsorted(t, end_s + pad_s) - c0)
    r0, r1 = hz_row(high_hz + pad_hz), hz_row(max(low_hz - pad_hz, 19)) + 1
    return w[:, c0:c0 + S.WIN_COLS], (int(a), int(b), int(r0), int(r1))


class OneShot:
    def __init__(self, window, box, methods=('v3', 'pixels', 'xcorr'), phases=10):
        self.methods = methods
        self.m, self.enc = E.load_model(V3)
        self.enc = self.enc.to(DEV).eval()
        rt = torch.from_numpy(window.astype(np.float32) / 255)[None].to(DEV)
        a, b, r0, r1 = box
        self.refs = {}
        for meth in methods:
            if meth == 'xcorr':
                k = rt[0, r0:r1, a:b]; k = k - k.mean(); self.kern = (k / k.norm(), r0, r1)
                continue
            layer = 'final' if meth == 'v3' else 'pixels'
            reps = []
            for p in range(phases):
                if a - p < 0:
                    continue
                with torch.no_grad():
                    rd = E.raw_layer_tokens(self.m, self.enc, torch.roll(rt, -p, dims=-1), layer)[0]
                reps.append((rd, E.box_mask((a - p, b - p, r0, r1)).to(DEV)))
            self.refs[meth] = (layer, reps)

    @torch.no_grad()
    def score(self, x):
        """x: (N,140,160) uint8 numpy or tensor -> dict method -> (N,) scores."""
        q = torch.as_tensor(np.asarray(x)).to(DEV).float() / 255
        out = {}
        for meth in self.methods:
            if meth == 'xcorr':
                k, r0, r1 = self.kern; L_ = k.shape[1]; qq = q[:, r0:r1]
                num = F.conv1d(qq, k[None])[:, 0]
                ones = torch.ones(1, qq.shape[1], L_, device=DEV)
                s1 = F.conv1d(qq, ones)[:, 0]; s2 = F.conv1d(qq * qq, ones)[:, 0]
                den = (s2 - s1 * s1 / (qq.shape[1] * L_)).clamp_min(1e-6).sqrt()
                out[meth] = (num / den).amax(1).cpu().numpy(); continue
            layer, reps = self.refs[meth]
            qd = E.raw_layer_tokens(self.m, self.enc, q, layer)
            out[meth] = torch.stack([E.block_cosine(qd, rd, mk) for rd, mk in reps]).amax(0).cpu().numpy()
        return out


def score_array(os_, X, ids=None, bs=1024):
    ids = np.arange(len(X)) if ids is None else ids
    res = {m: [] for m in os_.methods}
    for i in range(0, len(ids), bs):
        s = os_.score(X[ids[i:i + bs]])
        for m in res:
            res[m].append(s[m])
    return {m: np.concatenate(v) for m, v in res.items()}
