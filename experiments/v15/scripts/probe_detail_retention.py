"""Deterministic native-pixel probes; sensitivity is not task accuracy."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.detail_encoder import DetailEncoder,FORMAT,state_hash


def probes(size=64):
    y,x=np.mgrid[:size,:size];values=[]
    disk=(x-size/2)**2+(y-size/2)**2<(size/3)**2
    for name,pattern in [('one_pixel_vertical_ridges',x%2==0),('one_pixel_checkerboard',(x+y)%2==0),
        ('two_pixel_diagonal_ridges',(x+y)%4<2),('three_pixel_cross',((abs(x-size/2)<2)|(abs(y-size/2)<2)))]:
        gray=np.full((size,size),.25,np.float32);gray[disk]=.35+.4*pattern[disk]
        values.append((name,np.repeat(gray[...,None],3,2)))
    return values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError('Preserve earlier probe evidence')
    samples=probes();array=np.stack([rgb for _,rgb in samples]);x=torch.from_numpy(array).permute(0,3,1,2).to(args.device)
    changed=x.clone();changed[:,:,32,32]=1-changed[:,:,32,32]
    yy,xx=torch.meshgrid(torch.arange(64,device=args.device),torch.arange(64,device=args.device),indexing='ij')
    local=(xx-32).square()+(yy-32).square()<=4;far=(xx-32).square()+(yy-32).square()>16**2
    results=[]
    with torch.inference_mode():
        for path in args.checkpoint:
            saved=torch.load(path,map_location=args.device,weights_only=True)
            if saved.get('format')!=FORMAT or saved.get('natural_training_images')!=0 or saved.get('borrowed_pretrained_weights') is not False:raise ValueError('Expected own mathematical-only checkpoint')
            model=DetailEncoder().to(args.device);model.load_state_dict(saved['state_dict']);model.eval().requires_grad_(False)
            before=state_hash(model);base=model(x);perturbed=model(changed)
            cosine_change=1-(base['dense']*perturbed['dense']).sum(1)
            metrics=[]
            for index,(name,_) in enumerate(samples):
                metrics.append({'probe':name,'changed_input_pixels':1,
                    'detail_reconstruction_mae':float((base['detail_prediction'][index]-base['detail_target'][index]).abs().mean()),
                    'descriptor_change_at_changed_pixel':float(cosine_change[index,32,32]),
                    'mean_local_descriptor_change':float(cosine_change[index][local].mean()),
                    'mean_far_descriptor_change':float(cosine_change[index][far].mean()),
                    'maximum_stride1_feature_change':float((base['fine'][index]-perturbed['fine'][index]).abs().max())})
            if state_hash(model)!=before:raise RuntimeError('Probe changed weights')
            results.append({'checkpoint_name':path.name,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'state_sha256':before,'state_unchanged':True,'metrics':metrics})
    result={'completed':True,'scope':'Native1-3px mathematical probes and exactly one changed RGB pixel; no real images or gradient updates',
        'interpretation':'A nonzero local feature response demonstrates retained sensitivity. Larger sensitivity is not necessarily better recognition or anomaly detection, and noise robustness is a separate tradeoff.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'encoder_sha256':hashlib.sha256((ROOT/'hypernet/detail_encoder.py').read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(array.tobytes()).hexdigest(),'device':args.device,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
