"""Reference-only contrastive descriptor training with redundancy controls.

This is an original compact experiment inspired by contrastive learning and
variance/covariance regularization, not a reproduction of those full methods.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from .types import rgb_uint8
from .training import patch_grid

FORMAT = "hypernet.single_reference_descriptor_cnn.v2"

class TinyDescriptorCNN(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("At least four embedding dimensions are required")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, embedding_dim),
        )

    def embed(self, patches):
        return F.normalize(self.encoder(patches), dim=1)

    def forward(self, patches):
        return self.embed(patches)


def select_diverse_patches(patches, points, max_patches):
    """Texture-qualified, spatially seeded appearance diversity from the reference.

    Returns selected original indices and diagnostics. No semantic annotation or
    external image is used. Exact or nearly identical raw patches are not
    duplicated just to fill a target class count.
    """
    if len(patches) != len(points) or max_patches < 2:
        raise ValueError("Invalid patch selection inputs")
    x = patches.detach().float().cpu()
    gray = x.mean(1)
    texture = .5 * ((gray[:, 1:, :] - gray[:, :-1, :]).abs().mean((1, 2))
                     + (gray[:, :, 1:] - gray[:, :, :-1]).abs().mean((1, 2)))
    # A fixed image-space gradient cutoff excludes effectively flat patches.
    valid = torch.nonzero(texture >= .008,as_tuple=False).flatten()
    if len(valid) < 2:
        return valid.to(patches.device), {"candidate_patches": len(x), "texture_qualified_patches": len(valid),
                                         "selected_patches": len(valid), "texture_cutoff": .008}
    raw = F.adaptive_avg_pool2d(x, (8, 8)).flatten(1)
    # Preserve color and low-frequency shape; normalize distance by pixel count.
    distances = torch.cdist(raw[valid], raw[valid]).square() / raw.shape[1]
    coords = points.detach().float().cpu()[valid]
    coords = (coords - coords.min(0).values) / (coords.max(0).values-coords.min(0).values).clamp_min(1)
    spatial = torch.cdist(coords, coords).clamp(max=1.5) / 1.5
    ranked_texture = texture[valid] / texture[valid].max().clamp_min(1e-6)
    selected = [int(ranked_texture.argmax())]
    minimum = distances[selected[0]].clone()
    while len(selected) < min(max_patches, len(valid)):
        separation = minimum.clamp(min=0).sqrt()
        spatial_min = spatial[selected].min(0).values
        utility = separation * (.75 + .25 * ranked_texture) + .01 * spatial_min
        # Do not force surrogate differences between visually duplicate patches.
        utility[minimum < .0001] = -1
        utility[selected] = -1
        chosen = int(utility.argmax())
        if utility[chosen] < 0:
            break
        selected.append(chosen)
        minimum = torch.minimum(minimum, distances[chosen])
    result = valid[torch.tensor(selected)].to(patches.device)
    return result, {"candidate_patches": len(x), "texture_qualified_patches": len(valid),
                    "selected_patches": len(result), "texture_cutoff": .008,
                    "minimum_selected_raw_mse": .0001}


def _augment_v2(patches, generator):
    batch = len(patches)
    def random(shape):
        return torch.rand(shape, generator=generator, device=patches.device)
    angles = (random((batch,))-.5) * math.pi / 2  # +/-45 degrees
    scale = .7 + .65 * random((batch,))
    theta = torch.zeros(batch, 2, 3, device=patches.device)
    theta[:, 0, 0] = angles.cos()*scale
    theta[:, 1, 1] = angles.cos()*scale
    theta[:, 0, 1] = -angles.sin()*scale
    theta[:, 1, 0] = angles.sin()*scale
    theta[:, :, 2] = (random((batch,2))-.5)*.4
    grid = F.affine_grid(theta, patches.shape, align_corners=False)
    view = F.grid_sample(patches, grid, padding_mode="reflection", align_corners=False)
    mean = view.mean((2,3),keepdim=True)
    view = (view-mean)*(.6+.8*random((batch,1,1,1)))+mean
    view = view*(.55+.9*random((batch,1,1,1)))*(.8+.4*random((batch,3,1,1)))
    gray = view.mean(1,keepdim=True).expand_as(view)
    view = torch.where(random((batch,1,1,1))<.15,gray,view)
    blurred = F.avg_pool2d(F.pad(view,(1,1,1,1),mode="reflect"),3,stride=1)
    view = torch.where(random((batch,1,1,1))<.2,blurred,view)
    noise = torch.randn(view.shape,generator=generator,device=view.device)*.02
    return (view+noise).clamp(0,1)


def descriptor_loss(z0,z1,negative_allowed,temperature=.12):
    """Symmetric cross-view InfoNCE + small normalized-variance/decorrelation terms.

    Near-duplicate reference patches are excluded as negatives. This avoids
    claiming that similar reference regions must represent different semantics.
    """
    if z0.shape != z1.shape or len(z0)<2 or negative_allowed.shape!=(len(z0),len(z0)):
        raise ValueError("Invalid descriptor loss inputs")
    n,d=z0.shape
    targets=torch.arange(n,device=z0.device)
    logits=z0@z1.T/temperature
    allowed=negative_allowed.bool().clone()
    allowed.fill_diagonal_(True)
    logits=logits.masked_fill(~allowed,-1e4)
    contrastive=.5*(F.cross_entropy(logits,targets)+F.cross_entropy(logits.T,targets))
    invariance=(1-(z0*z1).sum(1)).mean()
    z=torch.cat([z0,z1])
    centered=z-z.mean(0,keepdim=True)
    std=torch.sqrt(centered.square().mean(0)+1e-5)
    target=.7/math.sqrt(d)
    variance=(F.relu(target-std)/target).mean()
    covariance=centered.T@centered/max(1,len(z)-1)
    off=covariance-torch.diag(torch.diagonal(covariance))
    decorrelation=off.square().sum()*d
    total=contrastive+.15*invariance+.2*variance+.05*decorrelation
    return total,{"contrastive":contrastive,"invariance":invariance,"variance":variance,
                  "decorrelation":decorrelation}


def train_reference_cnn_v2(reference,checkpoint_path,steps=120,seed=17,batch_size=32,
                           patch_size=32,max_patches=64,device="cpu",image_size=256,
                           embedding_dim=64,learning_rate=.002):
    from .models.cnn import image_digest
    reference=rgb_uint8(reference)
    if steps<1 or batch_size<2 or max_patches<2 or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("Positive steps/lr and at least two batch/selection patches required")
    if patch_size<8 or image_size<patch_size*2:
        raise ValueError("Image size must accommodate at least two patches per axis")
    checkpoint_path=Path(checkpoint_path)
    if checkpoint_path.exists() or checkpoint_path.with_suffix(checkpoint_path.suffix+".json").exists():
        raise FileExistsError("Choose a new v2 checkpoint path")
    device=str(device)
    if device!="cpu" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable")
    image=np.asarray(Image.fromarray(reference).resize((image_size,image_size),Image.Resampling.BILINEAR)).copy()
    tensor=torch.from_numpy(image).permute(2,0,1).float().div(255)[None].to(device)
    patches,points=patch_grid(tensor,patch_size,max(4,patch_size//2))
    indices,selection=select_diverse_patches(patches,points,max_patches)
    patches,points=patches[indices],points[indices]
    raw=F.adaptive_avg_pool2d(patches,(8,8)).flatten(1) if len(patches) else None
    negative_allowed=(torch.cdist(raw,raw).square()/raw.shape[1]>=.003) if len(patches)>1 else None
    cuda_devices=[] if device=="cpu" else [torch.device(device).index or 0]
    started=time.perf_counter()
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model=TinyDescriptorCNN(embedding_dim).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
        generator=torch.Generator(device=device).manual_seed(seed+1)
        history=[];components=[];batch_n=min(batch_size,len(patches));updates=0
        model.train()
        if batch_n>=2:
            for step in range(steps):
                # Without replacement: each patch identity has one paired positive.
                chosen=torch.randperm(len(patches),generator=generator,device=device)[:batch_n]
                selected=patches[chosen]
                views=torch.cat([_augment_v2(selected,generator),_augment_v2(selected,generator)])
                z=model(views)
                mask=negative_allowed[chosen[:,None],chosen[None,:]]
                loss,parts=descriptor_loss(z[:batch_n],z[batch_n:],mask)
                if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite v2 loss at step {step}")
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();updates+=1
                history.append(float(loss.detach().cpu()))
                components.append({key:float(value.detach().cpu()) for key,value in parts.items()})
        model.eval()
        with torch.inference_mode():
            clean=model(patches) if len(patches) else None
            offdiag=clean@clean.T if len(patches)>1 else None
            mean_similarity=(float(offdiag[~torch.eye(len(patches),dtype=torch.bool,device=device)].mean().cpu())
                             if offdiag is not None else None)
            mean_std=float(clean.std(0,unbiased=False).mean().cpu()) if clean is not None else None
        metadata={
            "method":"tiny_descriptor_cnn_v2","architecture":"3-convolution 64-dimensional descriptor; no patch classifier",
            "external_pretraining":False,"external_training_images":0,"target_reference_images":1,
            "requested_steps":steps,"target_weight_updates":updates,"reference_rgb_sha256":image_digest(reference),
            "reference_original_shape":list(reference.shape),"image_size":image_size,"patch_size":patch_size,
            "embedding_dim":embedding_dim,"num_patch_classes":len(patches),"selected_patch_centers":points.detach().cpu().tolist(),
            "selected_original_patch_indices":indices.cpu().tolist(),"selection":selection,
            "insufficient_informative_patches":len(patches)<2,
            "negative_exclusion_raw_mse":.003,"negative_pair_count":int(negative_allowed.sum()) if negative_allowed is not None else 0,
            "seed":seed,"batch_size":batch_size,"effective_batch_size":batch_n,"learning_rate":learning_rate,
            "synthetic_patch_views_seen":updates*batch_n*2,
            "augmentations":"rotation +/-45deg, normalized translation +/-0.2 (10% patch span), sampling scale .7-1.35, brightness/contrast/color, grayscale, blur/noise",
            "objective":"cross-view InfoNCE + .15 cosine invariance + .2 normalized variance hinge + .05 covariance penalty",
            "temperature":.12,"training_loss":history,"loss_components":components,
            "clean_offdiagonal_cosine_mean":mean_similarity,"clean_embedding_std_mean":mean_std,
            "elapsed_seconds":time.perf_counter()-started,"torch_version":str(torch.__version__),"device":device,
            "parameters":sum(p.numel() for p in model.parameters()),"score_is_probability":False,
            "sources":["https://arxiv.org/abs/2002.05709","https://arxiv.org/abs/2105.04906","https://arxiv.org/abs/1312.5242"],
            "paper_reproduction":False,
            "limitation":"Patch diversity and training separation are not semantic parts or independent-view object recognition. Negatives all come from the same reference.",
        }
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"format":FORMAT,"metadata":metadata,"state_dict":state},checkpoint_path)
    checkpoint_path.with_suffix(checkpoint_path.suffix+".json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata

