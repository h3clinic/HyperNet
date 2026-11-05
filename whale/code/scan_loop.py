"""Keep the one-shot Bering searches up to date while the 1 kHz cache fills:
rerun every target over every deployment (finished files are skipped) until the
cache job has finished and a final pass finds nothing new."""
import os, subprocess, sys, time
W = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = sys.executable
TARGETS = ['sei', 'narw', 'bloop']
DEPS = ['mbarc_lmr_bering02-m2', 'mbarc_lmr_bering02-m4', 'mbarc_lmr_bering03-m2', 'mbarc_lmr_bering04-m2']


def cache_done():
    try:
        return open(f'{W}/bering/cache.log').read().rstrip().endswith('done')
    except OSError:
        return False


while True:
    finished = cache_done()
    for t in TARGETS:
        deps = [d for d in DEPS if os.path.isdir(f'{W}/bering/audio1k/{d}')]
        if deps:
            subprocess.run([PY, 'scan_oneshot.py', t, *deps], cwd=os.path.dirname(__file__))
    if finished:
        break
    time.sleep(120)
print('all searches done', flush=True)
