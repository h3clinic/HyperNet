"""Edited MP4 of the Bering Sea sei whale candidates for a teacher / YouTube.

Scenes: title, the problem, the single example (with sound), the search, each
Bering match (spectrogram + sound + moving playhead), all matches side by side,
why not a fin whale, what this is and isn't, credits.
Sound is played 4x faster (pitch x4) so 30-60 Hz calls are audible.

  python make_video.py   ->  ../video/bering_sei_candidates.mp4
"""
import os, subprocess, datetime as dt, numpy as np, soundfile as sf, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import resample_poly, stft, istft
from scipy.ndimage import uniform_filter
import imageio_ffmpeg
import spectro as S

NAME = 'OnceNet'
W, H, FPS, AR = 1920, 1080, 30, 44100
BG, FG, ACC, DIM = '#0b1320', '#f2f5f8', '#4fd1e8', '#93a1b3'
OUT = os.path.abspath('../video'); os.makedirs(OUT, exist_ok=True)
SPEED = 4
HITS = [('Bering_2_stM2_041025_074456', 3366, .785), ('Bering_2_stM2_041025_034841', 7710, .771),
        ('Bering_2_stM2_041025_074456', 3980, .768), ('Bering_2_stM2_041025_034841', 6092, .754),
        ('Bering_2_stM2_041025_034841', 7552, .752), ('Bering_2_stM2_041121_012045', 5624, .751),
        ('Bering_2_stM2_041025_192804', 8460, .750)]
A1K = '../bering/audio1k/mbarc_lmr_bering02-m2/'


def fig():
    f = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG); return f


def to_rgb(f):
    f.canvas.draw(); a = np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(f); return a


def text_card(lines):
    """lines: list of (text, size, color, y, weight)."""
    f = fig()
    for t, s, c, y, w in lines:
        f.text(.5, y, t, ha='center', va='center', fontsize=s, color=c, weight=w, wrap=True)
    return to_rgb(f)


def spec_img(x, a, b, rows=(88, 140)):
    """Whitened spectrogram of x[a:b] (seconds, 1 kHz), with 30 s context for whitening."""
    lo = max(0, int((a - 30) * 1000)); seg = x[lo:int((b + 30) * 1000)]
    p, t = S.log_spectrogram(seg); w = S.to_uint8(S.whiten(p)); t = t + lo / 1000
    keep = (t >= a) & (t < b)
    return w[rows[0]:rows[1], keep]


def hz(r): return float(np.interp(r, [0, 139], [456, 19]))


class Scene:
    def __init__(self, secs, audio=None):
        self.n = int(round(secs * FPS)); self.audio = np.zeros(self.n * AR // FPS, np.float32)
        if audio is not None:
            k = min(len(audio), len(self.audio)); self.audio[:k] = audio[:k]

    def frames(self):
        raise NotImplementedError


class Card(Scene):
    def __init__(self, lines, secs):
        super().__init__(secs); self.lines = lines

    def frames(self):
        img = text_card(self.lines)
        for i in range(self.n): yield img


class Picture(Scene):
    def __init__(self, maker, secs):
        super().__init__(secs); self.maker = maker

    def frames(self):
        img = self.maker()
        for i in range(self.n): yield img


def sped_audio(seg):
    """1 kHz real-time audio -> played SPEED x faster, at 44.1 kHz."""
    y = resample_poly(seg.astype(np.float64), AR, 1000 * SPEED).astype(np.float32)
    y = y - y.mean(); y = y / (np.abs(y).max() + 1e-9) * .8
    r = int(.02 * AR); y[:r] *= np.linspace(0, 1, r); y[-r:] *= np.linspace(1, 0, r)
    return y


def denoise(x, a, b, band=(20, 90), k=2.5):
    """Noise-reduced x[a:b] (seconds, 1 kHz): keep only the call band and remove the
    steady background (spectral subtraction against the median of 20 s either side)."""
    lo = max(0, int((a - 20) * 1000)); seg = x[lo:int((b + 20) * 1000)].astype(np.float64)
    f, t, Z = stft(seg, 1000, nperseg=256, noverlap=224)
    mag = np.abs(Z) + 1e-12; noise = np.median(mag, axis=1, keepdims=True)
    g = np.sqrt(np.clip(1 - (k * noise / mag) ** 2, 0, 1))
    g = uniform_filter(g, size=(3, 5))  # smooth the gain to avoid "watery" artifacts
    g[(f < band[0]) | (f > band[1])] = 0
    _, y = istft(Z * g, 1000, nperseg=256, noverlap=224)
    s0 = int(a * 1000) - lo; return y[s0:s0 + int((b - a) * 1000)].astype(np.float32)


class SpecScene(Scene):
    """Spectrogram of `secs_real` seconds; a playhead moves while the sped-up sound plays."""
    LEAD, TAIL = 1.2, 1.6

    def __init__(self, x, a, b, title, sub, box=None):
        self.play = (b - a) / SPEED
        aud = np.concatenate([np.zeros(int(self.LEAD * AR), np.float32), sped_audio(denoise(x, a, b))])
        super().__init__(self.LEAD + self.play + self.TAIL, aud)
        self.x, self.a, self.b, self.title, self.sub, self.box = x, a, b, title, sub, box

    def frames(self):
        img = spec_img(self.x, self.a, self.b)
        f = fig(); ax = f.add_axes([.08, .14, .86, .62], facecolor='black')
        r0, r1 = 88, 140
        ax.imshow(img, cmap='magma', aspect='auto', interpolation='bilinear',
                  extent=[0, self.b - self.a, hz(r1 - 1), hz(r0)])
        ax.set_xlabel('seconds (real time)', color=DIM, fontsize=20); ax.set_ylabel('frequency (Hz)', color=DIM, fontsize=20)
        ax.tick_params(colors=DIM, labelsize=16)
        for s in ax.spines.values(): s.set_color(DIM)
        if self.box:
            t0, t1, f0, f1 = self.box
            ax.add_patch(plt.Rectangle((t0, f0), t1 - t0, f1 - f0, fill=False, ec=ACC, lw=3, ls='--'))
        f.text(.5, .91, self.title, ha='center', fontsize=40, color=FG, weight='bold')
        f.text(.5, .84, self.sub, ha='center', fontsize=24, color=DIM)
        f.text(.94, .04, f'sound: noise-reduced (20-90 Hz kept), sped up {SPEED}x so the low calls are audible', ha='right', fontsize=16, color=DIM)
        f.canvas.draw()
        base = np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy()
        (x0, y0), (x1, y1) = ax.transData.transform([(0, hz(r0)), (self.b - self.a, hz(r1 - 1))])
        plt.close(f)
        top, bot = int(H - y0), int(H - y1)
        for i in range(self.n):
            t = i / FPS - self.LEAD
            fr = base
            if 0 <= t <= self.play:
                px = int(x0 + (x1 - x0) * t / self.play); fr = base.copy()
                fr[min(top, bot):max(top, bot), max(px - 2, 0):px + 2] = (255, 255, 255)
            yield fr


def montage():
    f = fig(); f.text(.5, .94, 'The example and the 7 best Bering Sea matches, side by side', ha='center', fontsize=36, color=FG, weight='bold')
    panels = [('EXAMPLE  ·  Atlantic, 2 Apr 2009', ex_x, 138.5, 142.5)] + \
             [(f'#{k}  ·  {when(f_, s)}', load(f_), s, s + 4) for k, (f_, s, _) in enumerate(HITS, 1)]
    for k, (title, x, a, b) in enumerate(panels):
        ax = f.add_axes([.04 + (k % 4) * .24, .50 - (k // 4) * .42, .21, .33], facecolor='black')
        ax.imshow(spec_img(x, a, b), cmap='magma', aspect='auto', interpolation='bilinear', extent=[0, 4, hz(139), hz(88)])
        ax.set_title(title, color=ACC if k == 0 else FG, fontsize=17); ax.tick_params(colors=DIM, labelsize=12)
        for s in ax.spines.values(): s.set_color(ACC if k == 0 else DIM)
    f.text(.5, .03, 'frequency 19-170 Hz (vertical) over 4 seconds (horizontal)', ha='center', fontsize=18, color=DIM)
    return to_rgb(f)


_cache = {}
def load(name):
    if name not in _cache:
        _cache[name] = sf.read(A1K + name + '.flac', dtype='float32')[0]
    return _cache[name]


def when(name, s):
    t = dt.datetime.strptime(name[14:27], '%y%m%d_%H%M%S') + dt.timedelta(seconds=s + 2)
    return t.strftime('%d %b %Y, %H:%M')


x, sr = sf.read('../dclde2013/audio/NOPP6_EST_20090402_194500_CH10.wav', dtype='float32')
if x.ndim > 1: x = x[:, 0]
ex_x = resample_poly(x, 1000, sr).astype(np.float32)

L = lambda t, s, c, y, w='normal': (t, s, c, y, w)
scenes = [
    Card([L(NAME, 90, FG, .62, 'bold'),
          L('Finding rare whale calls in months of ocean recordings', 34, ACC, .46),
          L('from ONE example', 34, ACC, .39, 'bold'),
          L('a student research project', 22, DIM, .22)], 5),
    Card([L('Underwater recorders capture months of sound.', 40, FG, .64),
          L('A rare whale may call only a handful of times.', 40, FG, .53),
          L('Nobody can listen to all of it, and there are no labels for what is rare.', 32, DIM, .40),
          L('Can a model learn from a single example and find the rest?', 36, ACC, .26, 'bold')], 7),
    SpecScene(ex_x, 136, 146, 'The only example the model gets',
              'a labeled sei whale call  ·  Atlantic (DCLDE 2013 dataset)  ·  2 April 2009', box=(3.8, 5.2, 20, 72)),
    Card([L('The search', 52, FG, .76, 'bold'),
          L('About 2,900 hours of Bering Sea recordings, autumn 2004', 34, FG, .61),
          L('Mooring M2, southeastern Bering Sea shelf  ·  56.9° N, 164.1° W  ·  70 m deep', 28, DIM, .51),
          L('Public NOAA archive  ·  earlier studies of these months looked only for right whales', 28, DIM, .43),
          L('The model never saw a single label from this data.', 32, ACC, .26, 'bold')], 7),
]
for k, (f_, s, sc) in enumerate(HITS, 1):
    scenes.append(SpecScene(load(f_), s - 2, s + 8, f'Bering Sea match {k} of 7',
                            f'{when(f_, s)}  ·  match score {sc:.2f}  ·  model\'s pick, not hand-picked'))
scenes += [
    Picture(montage, 8),
    Card([L('Could these be fin whales?', 50, FG, .80, 'bold'),
          L('Fin whales at this same mooring make a "40 Hz" call (Sirovic et al., 2013):', 30, DIM, .67),
          L('about 61 Hz down to 48 Hz', 34, FG, .60),
          L('Our matches sweep from about 58 Hz down to 25-35 Hz in about 1 second,', 30, DIM, .47),
          L('a wider sweep that ends lower, like published sei whale calls (end 34-44 Hz).', 30, DIM, .41),
          L('Control test: same search on 216 hours from June 2005, the fin whale peak.', 26, ACC, .29),
          L('Its best matches all end at 44-50 Hz and sweep only 6-16 Hz. Ours end at 19-35 Hz: no overlap.', 26, ACC, .23)], 12),
    Card([L('What this is, and what it is not', 46, FG, .80, 'bold'),
          L('These are CANDIDATE sei whale calls found by the model.', 32, FG, .65),
          L('We found no published report of sei whale calls in the Bering Sea,', 30, DIM, .54),
          L('but they have not yet been confirmed by a whale-acoustics expert.', 30, DIM, .48),
          L('Next: finish the fin whale control, then expert review.', 32, ACC, .30, 'bold')], 9),
    Card([L('Credits', 40, FG, .80, 'bold'),
          L('Bering Sea recordings: NOAA NCEI Passive Acoustic Data Archive', 26, DIM, .66),
          L('(Scripps Institution of Oceanography, Marine Bioacoustics Research Collaborative; funded by U.S. Navy LMR)', 22, DIM, .60),
          L('Example call: DCLDE 2013 workshop dataset', 26, DIM, .51),
          L('Fin whale call measurements: Sirovic et al. 2013, Marine Biology 160:47-57', 22, DIM, .44),
          L(f'Model and analysis: {NAME}', 30, ACC, .28, 'bold')], 7),
]

FADE = 10
audio = np.concatenate([s.audio for s in scenes])
wav = os.path.join(OUT, '_audio.wav'); sf.write(wav, audio, AR)
mp4 = os.path.join(OUT, 'bering_sei_candidates.mp4')
cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
       '-r', str(FPS), '-i', '-', '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
       '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', mp4]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
total = 0
for k, s in enumerate(scenes):
    for i, fr in enumerate(s.frames()):
        a = min(1., (i + 1) / FADE, (s.n - i) / FADE)
        if a < 1: fr = (fr.astype(np.float32) * a + np.array([11, 19, 32], np.float32) * (1 - a)).astype(np.uint8)
        p.stdin.write(fr.tobytes()); total += 1
    print('scene', k, s.n, flush=True)
p.stdin.close(); p.wait(); os.remove(wav)
print(mp4, total / FPS, 's')
