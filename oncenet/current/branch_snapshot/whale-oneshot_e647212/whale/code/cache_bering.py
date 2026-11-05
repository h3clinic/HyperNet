"""Stream Bering HARP deployments from the official NOAA bucket and keep a 1 kHz
mono copy of every file (16-bit flac, ~12 MB per 4 h file). The full-rate
download is deleted after conversion. 1 kHz is the spectrogram rate of the
one-shot models (19-456 Hz rows) and the rate Scripps used for the right whale
search. Resumable: finished files are skipped.

Usage: python cache_bering.py DEPLOYMENT [DEPLOYMENT ...]
"""
import os, sys, json, time, socket, threading, queue, urllib.request, numpy as np, soundfile as sf
from scipy.signal import resample_poly

socket.setdefaulttimeout(120)
W = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
B = 'https://storage.googleapis.com/noaa-passive-bioacoustic/'
SR = 1000


def fetch(names, q, tmp):
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


def convert(src, dst):
    info = sf.info(src); sr0 = int(info.samplerate); g = np.gcd(sr0, SR)
    block = sr0 * 1800; out = []
    with sf.SoundFile(src) as f:
        while True:
            x = f.read(block, dtype='float32', always_2d=True)
            if len(x) == 0:
                break
            out.append(resample_poly(x[:, 0], SR // g, sr0 // g).astype(np.float32))
    y = np.concatenate(out)
    peak = float(np.abs(y).max()) or 1.
    sf.write(dst + '.part.flac', np.clip(y / peak * .9, -1, 1), SR, subtype='PCM_16')
    os.replace(dst + '.part.flac', dst)
    return len(y) / SR, peak, info.duration


def main():
    meta_path = f'{W}/bering/audio1k/meta.json'
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    inv = json.load(open(f'{W}/bering/inventory.json'))
    tmp = f'{W}/bering/stream'; os.makedirs(tmp, exist_ok=True)
    for dep in sys.argv[1:]:
        out_dir = f'{W}/bering/audio1k/{dep}'; os.makedirs(out_dir, exist_ok=True)
        todo = [n for n in inv[dep] if not os.path.exists(os.path.join(out_dir, os.path.basename(n).replace('.x.flac', '.flac')))]
        print(dep, len(inv[dep]), 'files,', len(todo), 'to do', flush=True)
        q = queue.Queue(maxsize=3); threading.Thread(target=fetch, args=(todo, q, tmp), daemon=True).start()
        t0, k = time.time(), 0
        while (p := q.get()) is not None:
            name = os.path.basename(p).replace('.x.flac', '')
            try:
                dur, peak, d0 = convert(p, os.path.join(out_dir, name + '.flac'))
                meta[f'{dep}/{name}'] = {'seconds': dur, 'peak': peak, 'orig_seconds': d0}; k += 1
                if k % 10 == 0:
                    json.dump(meta, open(meta_path, 'w'))
                    print(dep, k, len(todo), f'{time.time() - t0:.0f}s', flush=True)
            except Exception as e:
                print('fail', p, repr(e), flush=True)
            try:
                os.remove(p)
            except OSError:
                pass
        json.dump(meta, open(meta_path, 'w'))
    print('done')


if __name__ == '__main__':
    main()
