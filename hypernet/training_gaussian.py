"""Controlled Gaussian-derived augmentation of exactly one reference image."""
from pathlib import Path
from hashlib import sha256
import json,time
import numpy as np
import torch
from .types import rgb_uint8
from .training_v3 import TinyShapeCNN,analytic_shape_target
from .training_v4 import shared_geometry,corrupt_same_geometry,restoration_loss
from .training_v5 import reference_patch_bank,BalancedPatchCycle

FORMAT='hypernet.single_reference_gaussian_cnn.v07'
MODES=('baseline','pixel_noise','shape','shape_pattern')

def train_reference_gaussian(reference,checkpoint_path,mode='shape_pattern',steps=120,seed=73,device='cpu'):
    from .models.cnn import image_digest
    from .gaussian_variation import gaussian_variation,gaussian_variation_recipe
    if mode not in MODES:raise ValueError('Unknown Gaussian ablation')
    if isinstance(steps,bool) or int(steps)!=steps or steps<1:raise ValueError('steps must be positive integer')
    reference=rgb_uint8(reference);path=Path(checkpoint_path);sidecar=path.with_suffix('.pt.json')
    if path.exists() or sidecar.exists():raise FileExistsError('Preserve existing checkpoint/provenance')
    started=time.perf_counter();dev=torch.device(device)
    patches,groups,bank=reference_patch_bank(reference,device=dev)
    sampler_rng=torch.Generator(device=dev).manual_seed(seed+1)
    geometry_rng=torch.Generator(device=dev).manual_seed(seed+2)
    corruption_rng=torch.Generator(device=dev).manual_seed(seed+3)
    noise_rng=torch.Generator(device=dev).manual_seed(seed+4)
    sampler=BalancedPatchCycle(groups,sampler_rng)
    counts=torch.zeros(len(patches),dtype=torch.long,device=dev)
    patch_sequence=sha256();losses=[]
    insufficient=not np.any(reference!=reference[0,0])
    rng_devices=[] if dev.type=='cpu' else [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        model=TinyShapeCNN(64).to(dev)
        initial=sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
        model.train()
        if not insufficient:
            for step in range(int(steps)):
                ids=sampler.take(32);counts+=torch.bincount(ids,minlength=len(patches))
                patch_sequence.update(ids.cpu().numpy().tobytes())
                clean=shared_geometry(patches[ids],geometry_rng)
                if mode in ('shape','shape_pattern'):
                    # Matching per-step seed means pattern draws never alter the
                    # next batch's shape field or any legacy augmentation stream.
                    augmentation_rng=torch.Generator(device=dev).manual_seed(seed+10000+step)
                    clean=gaussian_variation(clean,augmentation_rng,kind=mode,strength=1.)
                first=corrupt_same_geometry(clean,corruption_rng)
                second=corrupt_same_geometry(clean,corruption_rng)
                if mode=='pixel_noise':
                    first=(first+.025*torch.randn(first.shape,device=dev,generator=noise_rng)).clamp(0,1)
                    second=(second+.025*torch.randn(second.shape,device=dev,generator=noise_rng)).clamp(0,1)
                with torch.no_grad():target=analytic_shape_target(clean)
                z0,z1=model(torch.cat([first,second])).chunk(2)
                loss,_=restoration_loss(z0,z1,target)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            predicted=torch.cat([model(patches[i:i+128]) for i in range(0,len(patches),128)])
            target=analytic_shape_target(patches)
            fidelity=float((predicted*target).sum(1).mean())
        metadata={
            'method':'one_reference_gaussian_ablation','training_revision':7,'mode':mode,
            'external_pretraining':False,'external_training_images':0,'target_reference_images':1,
            'query_weight_updates':0,'learned_teacher':False,'reference_rgb_sha256':image_digest(reference),
            'reference_shape':list(reference.shape),'target_weight_updates':len(losses),'requested_steps':int(steps),
            'seed':seed,'batch_size':32,'learning_rate':.002,'image_size':256,'patch_size':32,
            'parameters':sum(p.numel() for p in model.parameters()),'embedding_dim':64,
            'num_reference_patches':len(patches),**bank,'patch_training_draw_counts':counts.cpu().tolist(),
            'all_reference_patches_trained':bool((counts>0).all()),'insufficient_informative_patches':bool(insufficient),
            'initial_state_sha256':initial,'patch_sequence_sha256':patch_sequence.hexdigest(),
            'geometry_rng_state_sha256':sha256(geometry_rng.get_state().cpu().numpy().tobytes()).hexdigest(),
            'corruption_rng_state_sha256':sha256(corruption_rng.get_state().cpu().numpy().tobytes()).hexdigest(),
            'rng_policy':'Separate sampler/affine/corruption/intervention streams. Shape and shape_pattern share per-step displacement seeds.',
            'objective':'Unchanged v4 cosine restoration +0.2 relation loss; target recomputed on same transformed clean geometry',
            'extra_pixel_noise_std':.025 if mode=='pixel_noise' else 0.,
            'variation_strength':1. if mode in ('shape','shape_pattern') else 0.,
            'gaussian_variation_recipe':gaussian_variation_recipe(),
            'training_loss':losses,'clean_reference_target_cosine':fidelity,
            'synthetic_patch_views_seen':len(losses)*32*2,'elapsed_seconds':time.perf_counter()-started,
            'torch_version':str(torch.__version__),'device':str(dev),
            'limitation':'Generated variants share one source; no unseen class instances or human visual experience are supplied.',
            'human_mechanism_claim':False,'novel_augmentation_claim':False}
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'format':FORMAT,'metadata':metadata,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)
    sidecar.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata
