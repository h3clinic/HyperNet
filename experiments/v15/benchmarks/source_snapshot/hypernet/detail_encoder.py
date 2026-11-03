"""Randomly initialized, dense, phase-preserving mathematical-pattern CNN.

There is no semantic class head or global embedding bottleneck. The stride-one
path remains available at its native pixel resolution; pixel-unshuffle supplies
context branches without first averaging away the four pixel phases.
"""
from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

FORMAT='hypernet.mathematical_details.v15.1'


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SeparableBlock(nn.Module):
    def __init__(self,channels,groups):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False),
            nn.GroupNorm(groups,channels),nn.SiLU(),nn.Conv2d(channels,channels,1,bias=False),nn.GroupNorm(groups,channels))
    def forward(self,x):return F.silu(x+self.layers(x))


class DetailEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(7,12,1,bias=False),nn.GroupNorm(3,12),nn.SiLU())
        self.fine=SeparableBlock(12,3)
        self.mid=nn.Sequential(nn.Conv2d(12*4,20,1,bias=False),nn.GroupNorm(4,20),nn.SiLU(),SeparableBlock(20,4))
        self.coarse=nn.Sequential(nn.Conv2d(20*4,32,1,bias=False),nn.GroupNorm(4,32),nn.SiLU(),SeparableBlock(32,4))
        self.project=nn.Sequential(nn.Conv2d(12+20+32+4,32,1,bias=False),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,24,1))
        self.detail_head=nn.Conv2d(12,4,1)
        self.mask_head=nn.Conv2d(12+20+32,1,1)
        self.register_buffer('sobel_x',torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])[None,None]/8)
        self.register_buffer('laplacian',torch.tensor([[0.,1,0],[1,-4,1],[0,1,0]])[None,None])

    def preprocess(self,rgb):
        gray=rgb.mean(1,keepdim=True)
        high=gray-F.avg_pool2d(gray,5,1,2)
        contrast=high/(.06+F.avg_pool2d(high.square(),5,1,2).sqrt())
        gx=F.conv2d(gray,self.sobel_x,padding=1)*4
        gy=F.conv2d(gray,self.sobel_x.transpose(2,3),padding=1)*4
        lap=F.conv2d(gray,self.laplacian,padding=1)
        detail=torch.cat([contrast.clamp(-3,3)/3,gx,gy,lap],1)
        return torch.cat([2*rgb-1,detail],1),detail

    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1]!=3 or rgb.shape[-1]%4 or rgb.shape[-2]%4:
            raise ValueError('Use BCHW RGB with dimensions divisible by four')
        inputs,detail=self.preprocess(rgb)
        fine=self.fine(self.stem(inputs))
        mid=self.mid(F.pixel_unshuffle(fine,2))
        coarse=self.coarse(F.pixel_unshuffle(mid,2))
        combined=torch.cat([fine,F.interpolate(mid,fine.shape[-2:],mode='nearest'),
            F.interpolate(coarse,fine.shape[-2:],mode='nearest')],1)
        raw=self.project(torch.cat([combined,detail],1))
        return {'dense':F.normalize(raw,dim=1,eps=1e-6),'raw_dense':raw,'fine':fine,'mid':mid,'coarse':coarse,
            'detail_prediction':self.detail_head(fine),'detail_target':detail,
            'mask_logits':self.mask_head(combined)}


def sample_at_grid(features,grid):
    """Exact four-neighbor bilinear gather, align_corners=False pixel centers.

    The grid is label geometry, never optimized. Explicit gather avoids CUDA's
    nondeterministic grid_sample backward; deterministic gather is supported by
    torch.use_deterministic_algorithms. Gradients still reach all sampled CNN
    feature values. Outside-image contributions are zero.
    """
    b,c,h,w=features.shape
    xy=(grid+1)*torch.tensor([w,h],device=grid.device)/2-.5
    low=torch.floor(xy).to(torch.long);fraction=xy-low
    flat=features.flatten(2);result=torch.zeros(b,c,grid.shape[1],device=features.device,dtype=features.dtype)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        x=low[...,0]+dx;y=low[...,1]+dy
        valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        index=(y.clamp(0,h-1)*w+x.clamp(0,w-1))[:,None].expand(-1,c,-1)
        weight=(fraction[...,0] if dx else 1-fraction[...,0])*(fraction[...,1] if dy else 1-fraction[...,1])*valid
        result=result+flat.gather(2,index)*weight[:,None]
    return result.transpose(1,2)


def anchor_coordinates(valid,detail,count,generator):
    """Prefer informative pixels; never sample outside geometric validity."""
    b,h,w=valid.shape
    weights=(valid.float()*(.02+detail.float())).flatten(1)
    if torch.any(weights.sum(1)<=0):raise ValueError('Every generated pair needs visible corresponding support')
    indices=torch.multinomial(weights,count,replacement=True,generator=generator)
    xy=torch.stack([indices%w,indices//w],-1).float()
    grid=2*(xy+.5)/torch.tensor([w,h],device=xy.device)-1
    return grid,xy


def patch_appearance(rgb,grid):
    """Low-level ambiguity test, not a trained teacher or a target descriptor."""
    gray=rgb.mean(1,keepdim=True)
    patches=F.unfold(gray,5,padding=2).reshape(rgb.shape[0],25,*rgb.shape[-2:])
    values=sample_at_grid(patches,grid)
    values=values-values.mean(-1,keepdim=True)
    return F.normalize(values,dim=-1,eps=1e-6)


def correspondence_direction(first,second,rgb,warp,valid,detail,generator,count=32,temperature=.12):
    anchors,xy=anchor_coordinates(valid,detail,count,generator)
    target_grid=sample_at_grid(warp.permute(0,3,1,2),anchors)
    source=F.normalize(sample_at_grid(first['dense'],anchors),dim=-1,eps=1e-6)
    target=F.normalize(sample_at_grid(second['dense'],target_grid),dim=-1,eps=1e-6)
    raw=sample_at_grid(first['raw_dense'],anchors)
    b,k,d=source.shape
    logits=torch.bmm(source,target.transpose(1,2))/temperature
    # Nearby pixels and identical periodic motifs are not valid hard negatives.
    distance=torch.cdist(xy,xy)
    target_xy=(target_grid+1)*torch.tensor([rgb.shape[-1],rgb.shape[-2]],device=rgb.device)/2-.5
    target_distance=torch.cdist(target_xy,target_xy)
    with torch.no_grad():
        appearance=patch_appearance(rgb,anchors)
        ambiguous=torch.bmm(appearance,appearance.transpose(1,2))>.995
        flat=appearance.norm(dim=-1)<.1
        ambiguous=ambiguous|(flat[:,:,None]&flat[:,None,:])
        negative=(distance>=4)&(target_distance>=4)&~ambiguous
        identity=torch.eye(k,dtype=torch.bool,device=source.device)[None]
        allowed=negative|identity
    logits=logits.masked_fill(~allowed,-1e4)
    labels=torch.arange(k,device=logits.device)[None].expand(b,-1)
    per_anchor=F.cross_entropy(logits.reshape(b*k,k),labels.reshape(-1),reduction='none').reshape(b,k)
    informative=negative.any(-1)
    loss=(per_anchor*informative).sum()/informative.sum().clamp_min(1)
    accuracy=((logits.argmax(-1)==labels)&informative).sum()/informative.sum().clamp_min(1)
    positive=(source*target).sum(-1).mean()
    # Family context discrimination is deliberately absent: unrelated scenes
    # can legitimately contain the same local line, corner or periodic motif.
    return loss,raw.reshape(-1,d),{'correspondence_accuracy':accuracy,'positive_cosine':positive,
        'informative_anchor_fraction':informative.float().mean(),'negative_count':negative.sum(-1).float().mean()}


def detail_loss(first,second,batch,generator,anchors=32):
    valid_a,valid_b,coverage=conservative_validity(batch)
    def direction(source,target,rgb,warp,valid,detail):
        usable=valid.flatten(1).any(1)
        if usable.any():
            return correspondence_direction({k:v[usable] for k,v in source.items()},
                {k:v[usable] for k,v in target.items()},rgb[usable],warp[usable],valid[usable],detail[usable],generator,anchors)
        zero=source['dense'].sum()*0
        # One-pixel structures with no four-neighbor foreground footprint still
        # train reconstruction and foreground. They do not receive false warp
        # labels through a background-mixed descriptor sample.
        raw=source['raw_dense'].flatten(2).transpose(1,2)[:,::16].reshape(-1,24)
        return zero,raw,{key:zero for key in ('correspondence_accuracy','positive_cosine','informative_anchor_fraction','negative_count')}
    loss_a,raw_a,stats_a=direction(first,second,batch['rgb_a'],batch['grid_a_to_b'],valid_a,batch['detail_a'])
    loss_b,raw_b,stats_b=direction(second,first,batch['rgb_b'],batch['grid_b_to_a'],valid_b,batch['detail_b'])
    contrastive=(loss_a+loss_b)/2
    raw=torch.cat([raw_a,raw_b]);centered=raw-raw.mean(0,keepdim=True)
    variance=F.relu(.5-(centered.square().mean(0)+1e-4).sqrt()).mean()
    covariance=centered.T@centered/max(1,len(centered)-1)
    covariance=(covariance-torch.diag(torch.diagonal(covariance))).square().sum()/raw.shape[1]
    reconstruction=0.;foreground=0.
    for out,mask in [(first,batch['mask_a']),(second,batch['mask_b'])]:
        # Reconstruction must pass through the learned stride-one branch; its
        # head cannot read the analytic skip or the target tensor directly.
        reconstruction+=F.smooth_l1_loss(out['detail_prediction'],out['detail_target'])/2
        foreground+=F.binary_cross_entropy_with_logits(out['mask_logits'].squeeze(1),mask.float())/2
    loss=contrastive+.5*reconstruction+.15*foreground+variance+.05*covariance
    stats={'loss':loss,'correspondence_loss':contrastive,'detail_reconstruction':reconstruction,
        'foreground_loss':foreground,'variance_loss':variance,'covariance_loss':covariance}
    stats.update({key:(stats_a[key]+stats_b[key])/2 for key in stats_a})
    stats.update(coverage)
    keys=list(stats);values=torch.stack([stats[key].detach() for key in keys]).cpu().tolist()
    return loss,dict(zip(keys,values))


def conservative_validity(batch):
    """Require all target bilinear contributors to lie on visible foreground."""
    valid=[];retained=[];pairs=[]
    for source,target,key in [('a','b','a_to_b'),('b','a','b_to_a')]:
        declared=batch['valid_'+key]&batch['mask_'+source]
        support=F.grid_sample(batch['mask_'+target][:,None].float(),batch['grid_'+key],
            mode='bilinear',padding_mode='zeros',align_corners=False).squeeze(1)
        keep=declared&(support>=1-1e-6)
        valid.append(keep)
        retained.append(keep.float().sum()/declared.float().sum().clamp_min(1))
        pairs.append(keep.flatten(1).any(1).float().mean())
    return *valid,{'correspondence_support_retained':sum(retained)/2,'pairs_with_valid_correspondence':sum(pairs)/2}
