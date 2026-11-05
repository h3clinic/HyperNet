"""Fin whale control: fetch M2 recordings from 22-30 June 2005 (peak of fin whale
40 Hz calling at this site, Sirovic et al. 2013) at 1 kHz, then run the same
one-shot sei search on them. Uses its own meta file so it can run beside
cache_bering.py."""
import os, re, json, queue, threading, subprocess, sys, time
import cache_bering as C
dep = 'mbarc_lmr_bering03-m2'
inv = json.load(open(f'{C.W}/bering/inventory.json'))
names = [n for n in inv[dep] if re.search(r'_0506(2[2-9]|30)_', n)]
out = f'{C.W}/bering/audio1k/{dep}'; os.makedirs(out, exist_ok=True)
tmp = f'{C.W}/bering/stream_june'; os.makedirs(tmp, exist_ok=True)
todo = [n for n in names if not os.path.exists(os.path.join(out, os.path.basename(n).replace('.x.flac', '.flac')))]
print(len(names), 'files,', len(todo), 'to do', flush=True)
meta = {}
q = queue.Queue(maxsize=3); threading.Thread(target=C.fetch, args=(todo, q, tmp), daemon=True).start()
t0, k = time.time(), 0
while (p := q.get()) is not None:
    name = os.path.basename(p).replace('.x.flac', '')
    dur, peak, d0 = C.convert(p, os.path.join(out, name + '.flac')); meta[name] = {'seconds': dur, 'peak': peak}
    os.remove(p); k += 1
    if k % 5 == 0: print('cached', k, len(todo), f'{time.time() - t0:.0f}s', flush=True)
json.dump(meta, open(f'{C.W}/bering/june_control_meta.json', 'w'))
print('cache done', flush=True)
