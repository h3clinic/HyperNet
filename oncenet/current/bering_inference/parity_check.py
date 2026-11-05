"""CPU-only check: shared query tokens versus original OneShot.score.

Uses only the three existing supports and deterministic shifts, not held-out
audio or labels. This does not establish T4/local-GPU numerical parity.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from scanner import SharedScorer,HERE,verify_bundle,atomic_json,manifest,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='cpu_parity.json');args=p.parse_args()
    assets=verify_bundle();manifest();t=time.monotonic();torch.set_num_threads(2)
    engine=SharedScorer('cpu');torch.set_num_threads(2)
    x=np.stack([np.load(HERE/f'whale/examples/{target}.npz')['window'] for target in ('sei','narw','bloop')])
    x=np.concatenate([x,np.roll(x[:1],3,axis=2)],axis=0)
    shared={**engine.score(x,['sei','narw']),**engine.score(x,['bloop'])};rows=[]
    for target in ('sei','narw','bloop'):
        legacy=engine.matchers[target].score(x)
        for method in ('v3','pixels','xcorr'):
            error=float(np.max(np.abs(shared[target][method]-legacy[method])))
            rows.append({'target':target,'method':method,'max_abs_error':error,'bitwise_equal':bool(np.array_equal(shared[target][method],legacy[method]))})
    passed=all(r['bitwise_equal'] for r in rows)
    output={'status':'passed' if passed else 'failed','scope':'CPU logical reuse parity on 4 already-used support/shift windows; not cross-device acceptance or accuracy',
            'rows':rows,'asset_hashes_sha256':assets,'manifest_sha256':digest(HERE/'manifest.json'),'seconds':time.monotonic()-t,'no_gpu':True,'no_reserved_images':True}
    atomic_json(Path(args.output),output);print(json.dumps(output,indent=2));assert passed

if __name__=='__main__':main()
