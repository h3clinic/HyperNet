"""Train a reusable CNN from generated pixels, never from natural photographs."""
from pathlib import Path
import hashlib,json,time
import torch
from torch import nn
from torch.nn import functional as F
from .training_v3 import TinyShapeCNN,analytic_shape_target
from .training_v4 import corrupt_same_geometry,restoration_loss
from .gaussian_variation import gaussian_variation
from .procedural_shapes import procedural_triplets,RECIPE

FORMAT='hypernet.procedural_pretraining.v09'
MODES=('restoration','detail_metric')

def fine_target(images,masks):
    """Known procedural masks and within-object detail, never real-photo masks."""
    gray=(images*images.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    low=F.avg_pool2d(F.pad(gray,(1,1,1,1),mode='reflect'),3,stride=1)
    high=(gray-low)*masks
    shape=F.adaptive_avg_pool2d(masks,(16,16)).flatten(1);shape=shape-shape.mean(1,keepdim=True)
    detail=F.adaptive_avg_pool2d(high,(16,16)).flatten(1)
    # Tiny residuals cannot receive arbitrarily large gain.
    detail=detail/detail.norm(dim=1,keepdim=True).clamp_min(.05)
    shape=F.normalize(shape,dim=1,eps=1e-6)
    return F.normalize(torch.cat((shape,detail),dim=1),dim=1,eps=1e-6)

def transformed_triplets(data,seed):
    chunks=data['images'].chunk(3);masks=data['masks'].chunk(3)
    transformed=[];warped=[]
    for rgb,mask in zip(chunks,masks):
        transformed.append(gaussian_variation(rgb,torch.Generator().manual_seed(seed),kind='shape_pattern'))
        warped.append(gaussian_variation(mask.expand(-1,3,-1,-1),torch.Generator().manual_seed(seed),kind='shape')[:,:1])
    return torch.cat(transformed),torch.cat(warped)

def train_procedural(path,mode='detail_metric',steps=2000,triplets_per_step=16,seed=73):
    if mode not in MODES:raise ValueError('Unknown pretraining objective')
    if steps<1 or triplets_per_step<2:raise ValueError('Positive steps and at least2triplets required')
    path=Path(path);sidecar=path.with_suffix('.pt.json')
    if path.exists() or sidecar.exists():raise FileExistsError('Preserve existing synthetic model')
    path.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=TinyShapeCNN(64);head=nn.Linear(64,512)
        initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=.001,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(seed+2);data_digest=hashlib.sha256();losses=[]
        model.train();head.train()
        for step in range(int(steps)):
            data=procedural_triplets(triplets_per_step,seed+100000+step)
            data_digest.update(data['program_sha256'].encode())
            clean,masks=transformed_triplets(data,seed+200000+step)
            first=corrupt_same_geometry(clean,generator);second=corrupt_same_geometry(clean,generator)
            z0,z1=model(torch.cat((first,second))).chunk(2)
            with torch.no_grad():target=analytic_shape_target(clean);fine=fine_target(clean,masks)
            base,_=restoration_loss(z0,z1,target)
            detail=1-(F.normalize(head(torch.cat((z0,z1))),dim=1)*torch.cat((fine,fine))).sum(1).mean()
            a,p,n=z0.chunk(3);aa,pp,nn_=z1.chunk(3)
            # True geometry is shared by positives; negative shares query nuisance.
            # This is an engineered triplet objective, not FDSML reproduction.
            triplet=.5*(F.relu(.1+(a*n).sum(1)-(a*p).sum(1)).mean()+
                         F.relu(.1+(aa*nn_).sum(1)-(aa*pp).sum(1)).mean())
            loss=base+(.25*detail+.2*triplet if mode=='detail_metric' else 0.)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite procedural loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5.)
            optimizer.step();losses.append({'total':float(loss.detach()),'restoration':float(base.detach()),
                'detail':float(detail.detach()),'triplet':float(triplet.detach())})
            if (step+1)%100==0:print(f'{mode}: {step+1}/{steps} loss={losses[-1]["total"]:.4f}',flush=True)
        model.eval().requires_grad_(False)
        metadata={'mode':mode,'seed':seed,'steps':int(steps),'triplets_per_step':int(triplets_per_step),
            'parameters':sum(p.numel() for p in model.parameters()),'auxiliary_training_parameters':sum(p.numel() for p in head.parameters()),
            'pretrained':True,'pretraining_origin':'our procedural generator from random initialization',
            'externally_provided_pretrained_weights':False,'natural_pretraining_images':0,'natural_training_images':0,
            'base_program_draws':steps*triplets_per_step,'mutant_program_draws':steps*triplets_per_step,
            'rendered_image_exposures':steps*triplets_per_step*3,'corrupted_view_exposures':steps*triplets_per_step*6,
            'unique_image_count_measured':False,'initial_state_sha256':initial,'program_sequence_sha256':data_digest.hexdigest(),
            'generator':RECIPE,'objective':'Old restoration' if mode=='restoration' else 'Restoration+.25foreground mask/detail prediction+.2matched-nuisance triplet',
            'query_images_seen':0,'real_reference_images_seen':0,'auxiliary_head_discarded_at_inference':True,
            'gaussian_fields_shared_across_triplet':True,'loss_history':losses,'elapsed_seconds':time.perf_counter()-start}
        torch.save({'format':FORMAT,'metadata':metadata,'state_dict':model.state_dict(),'auxiliary_head_state_dict':head.state_dict()},path)
        sidecar.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata
