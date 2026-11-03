"""Only the predeclared renderer samples; no optimizer/model calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, hashlib, json, sys
from datetime import datetime, timezone
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypernet.variation_curriculum import render,stream

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def run(plan_path,output):
    if Path(output).exists():raise FileExistsError(output)
    plan=read(plan_path);bound={str(Path(plan_path).resolve()):sha(plan_path)};arms=[]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    for relative in plan['arms']:
        root=ROOT/relative;admission=read(root/'admission.json');complete=read(root/'completion.json');config=admission['config']
        assert complete['status']=='complete' and complete['admission_sha256']==sha(root/'admission.json')
        assert str(torch.__version__)==admission['torch_version'] and str(np.__version__)==admission['numpy_version']
        assert torch.cuda.get_device_name(0)==admission['cuda_device'] and config['device']=='cuda'
        for source,expected in admission['sources'].items():
            assert sha(ROOT/source)==sha(root/'source_snapshot'/source)==expected
            bound[str(ROOT/source)]=expected;bound[str(root/'source_snapshot'/source)]=expected
        for name in ['admission.json','completion.json']:bound[str(root/name)]=sha(root/name)
        samples=[]
        for step in plan['zero_based_steps']:
            item=complete['shards'][step//64];path=root/item['path'];assert sha(path)==item['sha256']
            bound[str(path)]=item['sha256']
            with np.load(path,allow_pickle=False)as archive:
                codes,sa,sb,_=stream(step,config['batch_types'],config['stream_seed'])
                for pos in plan['type_positions']:
                    i=(step%64)*config['batch_types']+pos
                    assert archive['type_codes'][i]==codes[pos]
                    assert np.array_equal(archive['instance_seeds'][i],[sa[pos],sb[pos]])
                    for instance in range(2):
                        samples.append({'zero_based_step':step,'type_position':pos,'instance':('a','b')[instance],
                            'type_code':int(archive['type_codes'][i]),'seed':int(archive['instance_seeds'][i,instance]),
                            'rgb_sha256':archive['rgb_sha256'][i,instance].tobytes().hex(),
                            'mask_sha256':archive['mask_sha256'][i,instance].tobytes().hex()})
        assert len(samples)==plan['sample_count_per_arm']
        batch=render(np.array([s['type_code']for s in samples],np.uint32),np.array([s['seed']for s in samples],np.uint64),config['image_side'],'cuda')
        rgb=batch['rgb'].cpu().numpy();masks=batch['mask'].cpu().numpy().astype(np.uint8)
        for i,sample in enumerate(samples):
            assert hashlib.sha256(rgb[i].tobytes()).hexdigest()==sample['rgb_sha256']
            assert hashlib.sha256(masks[i].tobytes()).hexdigest()==sample['mask_sha256']
        arms.append({'run':relative,'passed':True,'samples':samples,'instances':len(samples),'rgb_and_mask_hashes_checked':2*len(samples)})
    torch.cuda.synchronize()
    for path,expected in bound.items():assert sha(path)==expected
    result={'format':'hypernet.v16.sampled_training_render_replay.v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'passed':True,'plan_sha256':sha(plan_path),'script_sha256':sha(__file__),'arms':arms,'bound_files':bound,
        'torch':str(torch.__version__),'numpy':str(np.__version__),'cuda_device':torch.cuda.get_device_name(0),
        'scope':'96 predeclared consumed training instances replayed;192RGB/mask SHA256values exact. No model or optimizer replay.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with Path(output).open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'output':str(output),'sha256':sha(output),'gpu_released':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=ROOT/'experiments/training_audit_replay_plan.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.plan,a.output)
