"""Random-init structural descriptors with local normalization and broad context.

RGB floatBCHW in[0,1], dimensions>=32; reflect-pad to8 internally. Dense32D
output is stride2 with centers at native index(2i+.5,2j+.5). The64D embedding
uses foreground-weighted1x1+2x2 pooling. No axes rotations/flips or tile pooling.
"""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

FORMAT='hypernet.relational_variation.v16.1'


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ChannelNorm(nn.Module):
    """Per-location channel normalization; no image/tile statistical coupling."""
    def __init__(self,c):
        super().__init__();self.weight=nn.Parameter(torch.ones(1,c,1,1));self.bias=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        centered=x-x.mean(1,keepdim=True)
        return centered*torch.rsqrt(centered.square().mean(1,keepdim=True)+1e-4)*self.weight+self.bias


class Block(nn.Module):
    def __init__(self,c,dilation=1):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(c,c,3,padding=dilation,dilation=dilation,groups=c,bias=False),
            ChannelNorm(c),nn.SiLU(),nn.Conv2d(c,c,1,bias=False),ChannelNorm(c))
    def forward(self,x):return F.silu(x+self.layers(x))


def stage(cin,cout,dilations):
    return nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),ChannelNorm(cout),nn.SiLU(),*[Block(cout,d)for d in dilations])


def sample_nodes(features,grid):
    """Deterministic bilinear gather at label node coordinates, align_corners=False."""
    b,c,h,w=features.shape;xy=(grid+1)*grid.new_tensor([w,h])/2-.5
    low=xy.floor().long();fraction=xy-low;flat=features.flatten(2)
    value=features.new_zeros((b,c,grid.shape[1]))
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        x=low[...,0]+dx;y=low[...,1]+dy;valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        index=(y.clamp(0,h-1)*w+x.clamp(0,w-1))[:,None].expand(-1,c,-1)
        weight=(fraction[...,0]if dx else 1-fraction[...,0])*(fraction[...,1]if dy else 1-fraction[...,1])*valid
        value=value+flat.gather(2,index)*weight[:,None]
    return value.transpose(1,2)


class RelationalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(7,16,3,padding=1,bias=False),ChannelNorm(16),nn.SiLU(),Block(16),Block(16))
        self.mid=stage(64,32,[1,2]);self.context=stage(128,48,[1,2,4]);self.deep=stage(192,64,[1,2])
        self.project=stage(64+32+48+64,48,[])
        self.dense_head=nn.Conv2d(48,32,1)
        self.mask_head=nn.Conv2d(32,1,1)
        self.embedding_head=nn.Sequential(nn.Linear(32*5,128),nn.SiLU(),nn.Linear(128,64))
        self.detail_head=nn.Conv2d(16,4,1)
        self.degree_head=nn.Linear(32,5)
        self.edge_head=nn.Sequential(nn.Linear(32*4+2,64),nn.SiLU(),nn.Linear(64,1))
        self.register_buffer('sobel',torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])[None,None]/8)
        self.register_buffer('laplace',torch.tensor([[0.,1,0],[1,-4,1],[0,1,0]])[None,None])

    def preprocess(self,rgb):
        gray=rgb.mean(1,keepdim=True);high=gray-F.avg_pool2d(gray,5,1,2)
        contrast=(high/(.06+F.avg_pool2d(high.square(),5,1,2).sqrt())).clamp(-3,3)/3
        detail=torch.cat([contrast,F.conv2d(gray,self.sobel,padding=1)*4,
            F.conv2d(gray,self.sobel.transpose(2,3),padding=1)*4,F.conv2d(gray,self.laplace,padding=1)],1)
        return torch.cat([2*rgb-1,detail],1),detail

    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1]!=3 or min(rgb.shape[-2:])<32:raise ValueError('Expected RGB BCHW, dimensions>=32, float[0,1]')
        h,w=rgb.shape[-2:];ph=(-h)%8;pw=(-w)%8
        if ph or pw:rgb=F.pad(rgb,(0,pw,0,ph),mode='reflect')
        inputs,detail=self.preprocess(rgb);fine=self.stem(inputs)
        phase=F.pixel_unshuffle(fine,2);mid=self.mid(phase)
        context=self.context(F.pixel_unshuffle(mid,2));deep=self.deep(F.pixel_unshuffle(context,2))
        fused=torch.cat([phase,mid,F.interpolate(context,mid.shape[-2:],mode='nearest'),F.interpolate(deep,mid.shape[-2:],mode='nearest')],1)
        raw=self.dense_head(self.project(fused))[:,:,:((h+1)//2),:((w+1)//2)]
        mask=self.mask_head(raw);weight=.05+.95*mask.sigmoid()
        summaries=[];dh,dw=raw.shape[-2:]
        regions=[(0,dh,0,dw),(0,dh//2,0,dw//2),(0,dh//2,dw//2,dw),
                 (dh//2,dh,0,dw//2),(dh//2,dh,dw//2,dw)]
        for y0,y1,x0,x1 in regions:
            v=raw[:,:,y0:y1,x0:x1];m=weight[:,:,y0:y1,x0:x1]
            summaries.append((v*m).sum((-2,-1))/m.sum((-2,-1)).clamp_min(1e-4))
        embedding_raw=self.embedding_head(torch.cat(summaries,1))
        return {'embedding':F.normalize(embedding_raw,dim=1,eps=1e-6),'embedding_raw':embedding_raw,
                'dense':F.normalize(raw,dim=1,eps=1e-6),'raw_dense':raw,'mask_logits':mask,
                'fine':fine[:,:,:h,:w],'detail_prediction':self.detail_head(fine)[:,:,:h,:w],
                'detail_target':detail[:,:,:h,:w]}

    def relations(self,out,node_xy,edge_indices):
        nodes=sample_nodes(out['raw_dense'],node_xy)
        a=nodes[:,edge_indices[:,0]];b=nodes[:,edge_indices[:,1]]
        delta=node_xy[:,edge_indices[:,1]]-node_xy[:,edge_indices[:,0]]
        pair=torch.cat([a,b,(a-b).abs(),a*b,delta],-1)
        return self.degree_head(nodes),self.edge_head(pair).squeeze(-1)


def supervised_contrast(embedding,codes,temperature=.12):
    similarity=embedding@embedding.T/temperature
    identity=torch.eye(len(codes),dtype=torch.bool,device=codes.device)
    positives=(codes[:,None]==codes[None])&~identity
    if not positives.any(1).all():raise ValueError('Every structural type needs another independent instance')
    log_probability=similarity-similarity.masked_fill(identity,-1e4).logsumexp(1,keepdim=True)
    loss=-(log_probability*positives).sum(1)/positives.sum(1)
    winner=similarity.masked_fill(identity,-1e4).argmax(1)
    return loss.mean(),(codes[winner]==codes).float().mean()


def objective(model,out,batch,edge_indices,relations=True):
    contrast,accuracy=supervised_contrast(out['embedding'],batch['type_codes'])
    degree_logits,edge_logits=model.relations(out,batch['node_xy'],edge_indices)
    degree=F.cross_entropy(degree_logits.flatten(0,1),batch['degrees'].flatten())
    adjacency=F.binary_cross_entropy_with_logits(edge_logits,batch['edges'])
    target=F.avg_pool2d(batch['mask'][:,None].float(),2,2)
    foreground=F.binary_cross_entropy_with_logits(out['mask_logits'],target)
    reconstruction=F.smooth_l1_loss(out['detail_prediction'],out['detail_target'])
    centered=out['embedding_raw']-out['embedding_raw'].mean(0,keepdim=True)
    variance=F.relu(.15-(centered.square().mean(0)+1e-4).sqrt()).mean()
    loss=contrast+.25*foreground+.25*reconstruction+.2*variance
    if relations:loss=loss+.35*adjacency+.15*degree
    stats={'loss':loss,'structural_contrast':contrast,'type_retrieval_accuracy':accuracy,
           'adjacency_loss':adjacency,'degree_loss':degree,'foreground_loss':foreground,
           'detail_reconstruction':reconstruction,'variance_penalty':variance,
           'edge_accuracy':((edge_logits>0)==batch['edges'].bool()).float().mean()}
    return loss,{key:float(value.detach())for key,value in stats.items()}
