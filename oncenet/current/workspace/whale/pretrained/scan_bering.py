"""Stream-scan Bering Sea HARP deployments with Google's multispecies whale model.

For each file: download from the official NOAA bucket, score 5 s clips every
2.5 s at 24 kHz, keep per-clip logits (12 classes) and features (fp16), save a
20 s, 2 kHz audio snippet for every clip whose Eg, Upcall or Gunshot logit
exceeds 0 (probability 0.5), then delete the audio. Resumable: finished files
are skipped.

Times are file start + offset. The flac files were converted from HARP XWAV
files that can have gaps, so a time is approximate until checked against the
XWAV headers.

Usage: python scan_bering.py DEPLOYMENT [every_nth]
"""
import os, sys, json, time, socket, threading, queue, urllib.request, numpy as np, soundfile as sf
socket.setdefaulttimeout(120)  # a dropped connection (e.g. after sleep) raises instead of hanging forever
from scipy.signal import resample_poly
import audio_models as A

W = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
B = 'https://storage.googleapis.com/noaa-passive-bioacoustic/'
HOP, CLIP, SR = 2.5, 5.0, 24000
KEEP = [A.WHALE_CLASSES.index(c) for c in ('Eg', 'Upcall', 'Gunshot')]


def download(names, q, tmp):
    for n in names:
        dst = os.path.join(tmp, os.path.basename(n))
        for attempt in range(20):
            try:
                urllib.request.urlretrieve(B + n, dst + '.part'); os.replace(dst + '.part', dst); break
            except Exception as e:
                print('retry', attempt, n, repr(e), flush=True); time.sleep(min(300, 15 * (attempt + 1)))
        if os.path.exists(dst):
            q.put(dst)
        else:
            print('gave up', n, flush=True)
    q.put(None)


def scan_file(path, out_dir):
    info = sf.info(path); sr0 = int(info.samplerate)
    logits, feats, snips = [], [], []
    block = int(1800 * sr0)  # 30 min blocks
    for b0 in range(0, info.frames, block):
        x, _ = sf.read(path, start=max(0, b0 - int(HOP * sr0)), frames=block + int((CLIP + HOP) * sr0), dtype='float32', always_2d=True)
        x = x[:, 0]; off = (b0 - max(0, b0 - int(HOP * sr0))) / sr0
        g = np.gcd(sr0, SR); x24 = resample_poly(x, SR // g, sr0 // g).astype(np.float32)
        starts = np.arange(off, off + block / sr0, HOP)
        starts = starts[starts + CLIP <= len(x24) / SR + 1e-6]
        for i in range(0, len(starts), 256):
            c = A.clips(x24, SR, starts[i:i + 256], dur=CLIP, lead=0.)
            lg, ft = A.whale(c); logits.append(lg.astype(np.float32)); feats.append(ft.astype(np.float16))
            hit = np.flatnonzero((lg[:, KEEP] > 0).any(1))
            for h in hit:
                t = starts[i + h]; a = int(max(0, (t - 7.5)) * SR); s = x24[a:a + 20 * SR]
                g2 = resample_poly(s, 1, 12).astype(np.float32)  # 2 kHz
                snips.append((float(b0 / sr0 + t - off), g2))
    lg = np.concatenate(logits); ft = np.concatenate(feats)
    name = os.path.basename(path).replace('.x.flac', '')
    np.savez(os.path.join(out_dir, name + '.npz'), logits=lg, feats=ft, hop=HOP, sr0=sr0,
             duration=info.duration)
    for t, s in snips:
        sf.write(os.path.join(out_dir, 'snips', f'{name}_{t:09.1f}.wav'), s, 2000)
    return len(lg), len(snips)


def main():
    dep = sys.argv[1]; nth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    names = json.load(open(f'{W}/bering/inventory.json'))[dep][::nth]
    out_dir = f'{W}/bering/scan/{dep}'; os.makedirs(f'{out_dir}/snips', exist_ok=True)
    tmp = f'{W}/bering/stream'; os.makedirs(tmp, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(out_dir, os.path.basename(n).replace('.x.flac', '.npz')))]
    print(dep, len(names), 'files,', len(todo), 'to do', flush=True)
    q = queue.Queue(maxsize=2)
    threading.Thread(target=download, args=(todo, q, tmp), daemon=True).start()
    t0, n = time.time(), 0
    while (p := q.get()) is not None:
        try:
            k, s = scan_file(p, out_dir); n += 1
            print(n, len(todo), os.path.basename(p), k, 'clips', s, 'hits', f'{time.time() - t0:.0f}s', flush=True)
        except Exception as e:
            print('fail', p, repr(e), flush=True)
        os.remove(p)


if __name__ == '__main__':
    main()
