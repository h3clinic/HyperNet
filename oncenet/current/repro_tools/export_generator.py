"""Export new illustration-only images from the exact OnceNet math generator.

Uses no real images, model weights, optimizer or evaluation set. No action is
performed unless --export is explicit. Outputs are new examples, not metrics.
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from PIL import Image
from train_gravity import workspace_default, sha, HERE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    p.add_argument('--workspace',type=Path,default=workspace_default())
    p.add_argument('--output',type=Path)
    p.add_argument('--count',type=int,default=5)
    p.add_argument('--start',type=int,default=99010000)
    args=p.parse_args()
    if not args.export:p.print_help();return
    if args.output is None:raise ValueError('--export requires a new --output directory')
    if not 1<=args.count<=1000:raise ValueError('Illustration export count must be between 1 and 1000')
    if args.start<99000000:raise ValueError('Use the illustration-only ID namespace >=99000000')
    workspace=args.workspace.resolve();path=workspace/'outputs/hypernet_v46/spectrogram/generator.py'
    bindings=json.loads((HERE/'gravity_training_bindings.json').read_text(encoding='utf-8'))
    row=next(x for x in bindings['training_source_files'] if x['relative_path'].endswith('hypernet_v46/spectrogram/generator.py'))
    if sha(path)!=row['sha256']:raise RuntimeError('Original generator hash mismatch')
    out=args.output.resolve()
    if out.exists():raise ValueError('Output already exists; refusing overwrite')
    out.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('original_math_generator',path)
    gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    items=[]
    for identity in range(args.start,args.start+args.count):
        e=gen.episode(identity,'train');files=[]
        for view in gen.VIEWS:
            for suffix in ('','_low'):
                a=e[view+suffix]
                if a.shape!=(128,128,3) or not np.isfinite(a).all():raise RuntimeError('Invalid generated array')
                name=f'{identity}_{view}{suffix}.png'
                Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(out/name)
                files.append({'path':name,'sha256':sha(out/name),'view':view+suffix})
        items.append({'identity':identity,'family':gen.FAMILIES[identity%len(gen.FAMILIES)],'files':files})
    manifest={'illustration_only':True,'benchmark_run':False,'model_inference':False,'real_images_read':0,
        'generator_sha256':sha(path),'split_namespace':'train','identity_range':[args.start,args.start+args.count-1],
        'explanation':'The train namespace reproduces the exact generator distribution; these new IDs are not the historical training or evaluation identities.',
        'views':list(gen.VIEWS),'items':items}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'identities':args.count,'images':args.count*len(gen.VIEWS)*2,'output':str(out)},indent=2))

if __name__=='__main__':main()
