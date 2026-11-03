"""Sharp independent instances of axis-preserving embedded mathematical graphs.

A type is a 24-bit edge arrangement on a 4x4 planar lattice, not a seed or a
semantic object category. Positive instances share only that arrangement.
Geometry, material and backgrounds are sampled independently. Paired hard
negatives share nuisance seeds, preventing palette/texture shortcuts.
"""
from functools import lru_cache
import numpy as np
import torch

FORMAT = 'hypernet.structural_variation.curriculum.v16.1'
EDGES = np.asarray([(4*y+x,4*y+x+1) for y in range(4) for x in range(3)] +
                   [(4*y+x,4*(y+1)+x) for y in range(3) for x in range(4)], np.int64)
CATALOG_SIZE = 32768
EDGE_BITS = np.arange(24,dtype=np.uint32)


def labels(codes):
    codes=np.asarray(codes,dtype=np.uint32).reshape(-1)
    edges=((codes[:,None]>>EDGE_BITS)&1).astype(np.float32)
    degree=np.zeros((len(codes),16),np.int64)
    for e,(a,b) in enumerate(EDGES):degree[:,a]+=edges[:,e].astype(np.int64);degree[:,b]+=edges[:,e].astype(np.int64)
    return edges,degree


def component_count(code):
    edges,degree=labels([code]);active=set(np.flatnonzero(degree[0]).tolist());count=0
    adjacency=[[]for _ in range(16)]
    for present,(a,b) in zip(edges[0],EDGES):
        if present:adjacency[a].append(int(b));adjacency[b].append(int(a))
    while active:
        count+=1;stack=[active.pop()]
        while stack:
            for node in adjacency[stack.pop()]:
                if node in active:active.remove(node);stack.append(node)
    return count


def partition(codes):
    x=np.asarray(codes,dtype=np.uint32).copy()
    x^=x>>16;x*=np.uint32(0x7feb352d);x^=x>>15;x*=np.uint32(0x846ca68b);x^=x>>16
    return x&7  # Zero is held out; all nonzero values are training signatures.


def eligible(codes):
    edge,degree=labels(codes);count=edge.sum(1);active=degree>0
    return ((count>=8)&(count<=17)&active[:,:4].any(1)&active[:,12:].any(1)&
            active[:,[0,4,8,12]].any(1)&active[:,[3,7,11,15]].any(1))


@lru_cache(maxsize=2)
def catalog(validation=False):
    # Odd multiplication is a permutation of the24-bit code space. Filtering
    # does not create duplicate signatures; hash partition prevents leakage.
    index=np.arange(1<<18,dtype=np.uint64)
    codes=((index*114067+12820163)&((1<<24)-1)).astype(np.uint32)
    allowed=partition(codes)==0 if validation else partition(codes)!=0
    selected=codes[allowed&eligible(codes)]
    count=512 if validation else CATALOG_SIZE
    if len(selected)<count:raise RuntimeError('Insufficient declared structural catalog')
    result=selected[:count].copy();result.flags.writeable=False
    return result


def reflected(code,horizontal=True):
    mapping={tuple(sorted(map(int,edge))):i for i,edge in enumerate(EDGES)}
    result=0
    def node(i):return 4*(i//4)+(3-i%4) if horizontal else 4*(3-i//4)+i%4
    for e,(a,b) in enumerate(EDGES):
        if code&(1<<e):result|=1<<mapping[tuple(sorted((node(int(a)),node(int(b)))))]
    return result


@lru_cache(maxsize=CATALOG_SIZE)
def hard_negative(code):
    """Prefer rewiring with exact edge/orientation/degree-hist/component counts."""
    edge,degree=labels([code]);original=np.sort(degree[0]);components=component_count(code)
    present=np.flatnonzero(edge[0]);absent=np.flatnonzero(1-edge[0])
    candidates=[int(code^(1<<int(a))^(1<<int(b))) for a in present for b in absent if (a<12)==(b<12)]
    rng=np.random.default_rng(int(code)^1631);rng.shuffle(candidates)
    if candidates:
        _,degrees=labels(candidates)
        keep=np.all(np.sort(degrees,axis=1)==original,axis=1)&(partition(candidates)!=0)&eligible(candidates)
        for candidate in np.asarray(candidates)[keep]:
            if component_count(int(candidate))==components:return int(candidate),'rewire_matched_statistics'
    # An axis reversal is a DIFFERENT type, never a positive augmentation.
    # It preserves all graph counts but changes the declared fixed-axis order.
    for candidate in (reflected(code),reflected(code,False),reflected(reflected(code),False)):
        if candidate!=code and int(partition([candidate])[0])!=0:return candidate,'axis_order_matched_statistics'
    for candidate in candidates:
        if int(partition([candidate])[0])!=0 and bool(eligible([candidate])[0]):
            return candidate,'edge_count_orientation_matched_only'
    raise RuntimeError('No distinct permitted hard-negative signature')


def stream(step,batch_types=64,seed=1627):
    if batch_types<4 or batch_types%2:raise ValueError('An even number of structural types is required')
    half=batch_types//2;indices=(np.arange(half)+step*half)%CATALOG_SIZE
    base=catalog()[indices];negative=[hard_negative(int(code)) for code in base]
    codes=np.concatenate([base,np.asarray([item[0] for item in negative],np.uint32)])
    visits=np.arange(half,dtype=np.uint64)+step*half
    # Shared nuisance for each base/negative, independent between positive views.
    seeds_a=np.concatenate([visits*2+seed,visits*2+seed])
    seeds_b=seeds_a+np.uint64(1<<40)
    return codes,seeds_a,seeds_b,[item[1] for item in negative]


def _random(seeds,columns=128):
    x=np.asarray(seeds,np.uint64)[:,None]+np.arange(columns,dtype=np.uint64)[None]*np.uint64(0x9e3779b97f4a7c15)
    x^=x>>30;x*=np.uint64(0xbf58476d1ce4e5b9);x^=x>>27;x*=np.uint64(0x94d049bb133111eb);x^=x>>31
    return ((x>>40).astype(np.float32)+.5)/16777216


@torch.no_grad()
def render(codes,seeds,size=96,device='cpu'):
    """Return uint8BHWC RGB, masks, normalized node centers and graph labels.

    Node locations are training-only supervision. Inference takes RGB alone.
    There is deliberately no cross-instance pixel-correspondence map.
    """
    codes=np.asarray(codes,np.uint32).reshape(-1);seeds=np.asarray(seeds,np.uint64).reshape(-1)
    if len(codes)!=len(seeds) or size<64 or size%8:raise ValueError('Use aligned code/seed arrays and side>=64 divisible8')
    edge_np,degree_np=labels(codes)
    edge=torch.as_tensor(edge_np,device=device);degree=torch.as_tensor(degree_np,device=device)
    u=torch.as_tensor(_random(seeds),device=device)
    b=len(codes);axis=torch.linspace(-1,1,4,device=device)
    columns=axis[None]*(.52+.16*u[:,0:1])+.035*(2*u[:,2:6]-1)
    rows=axis[None]*(.52+.16*u[:,1:2])+.035*(2*u[:,6:10]-1)
    x=columns[:,None,:].expand(-1,4,-1).reshape(b,16)
    y=rows[:,:,None].expand(-1,-1,4).reshape(b,16)
    # Composition of bounded shears (no rotation/flip); independent local
    # jitter stays small compared with lattice spacing and stroke width.
    x=x+.10*(2*u[:,10:11]-1)*torch.sin(torch.pi*y)+.02*(2*u[:,12:28]-1)
    y=y+.10*(2*u[:,11:12]-1)*torch.sin(torch.pi*x)+.02*(2*u[:,28:44]-1)
    x=x+.10*(2*u[:,44:45]-1);y=y+.10*(2*u[:,45:46]-1)
    nodes=torch.stack([x,y],-1)
    pixel=2*(torch.arange(size,device=device)+.5)/size-1
    yy,xx=torch.meshgrid(pixel,pixel,indexing='ij');xx=xx[None];yy=yy[None]
    width=.024+.014*u[:,46:47]
    bows=.045*(2*u[:,48:72]-1)
    mask=torch.zeros((b,size,size),dtype=torch.bool,device=device)
    for e,(a,c) in enumerate(EDGES):
        ax=x[:,a,None,None];ay=y[:,a,None,None]
        dx=x[:,c,None,None]-ax;dy=y[:,c,None,None]-ay
        length=(dx.square()+dy.square()).sqrt().clamp_min(.1)
        t=((xx-ax)*dx+(yy-ay)*dy)/length.square();t=t.clamp(0,1)
        bow=bows[:,e,None,None]*torch.sin(torch.pi*t)
        cx=ax+t*dx-bow*dy/length;cy=ay+t*dy+bow*dx/length
        distance=(xx-cx).square()+(yy-cy).square()
        mask|=(distance<=width[:,:,None].square())&(edge[:,e,None,None]>0)
    # Junction disks close the incident analytic strokes. Inactive lattice
    # nodes are not drawn, so hidden coordinates cannot identify the type.
    for n in range(16):
        mask|=((xx-x[:,n,None,None]).square()+(yy-y[:,n,None,None]).square()<=(1.12*width[:,:,None]).square())&(degree[:,n,None,None]>0)
    # Independent high-frequency materials/backgrounds, not shared instance
    # textures. Nuisance-matched negative pairs use exactly these same fields.
    angle=2*torch.pi*u[:,72,None,None];frequency=7+32*u[:,73,None,None]
    grating=(torch.sin(frequency*(xx*angle.cos()+yy*angle.sin())+6.28*u[:,74,None,None])>0).float()
    dots=(torch.sin((12+24*u[:,75,None,None])*xx+6.28*u[:,76,None,None])*
          torch.sin((12+24*u[:,77,None,None])*yy+6.28*u[:,78,None,None])>.55).float()
    material=.72+.18*grating+.10*dots
    base=.04+.18*u[:,79:82,None,None]
    target=.55+.43*u[:,82:85,None,None]
    background=base.expand(-1,-1,size,size).clone()
    smooth=.04*torch.sin((2+4*u[:,85,None,None])*xx+(2+4*u[:,86,None,None])*yy+6.28*u[:,87,None,None])
    background+=smooth[:,None]
    # Sparse mathematical distractor strokes with independently drawn material.
    for k in range(4):
        start=88+5*k;phi=torch.pi*u[:,start,None,None]
        offset=2*u[:,start+1,None,None]-1
        line=(xx*phi.cos()+yy*phi.sin()-offset).abs()<(.009+.012*u[:,start+2,None,None])
        gate=(xx*phi.sin()-yy*phi.cos()).abs()<(.12+.28*u[:,start+3,None,None])
        background+=((line&gate).float()*(.12+.25*u[:,start+4,None,None]))[:,None]
    foreground=target*material[:,None]
    rgb=torch.where(mask[:,None],foreground,background).clamp(0,1)
    rgb=(rgb*255).round().to(torch.uint8).permute(0,2,3,1).contiguous()
    return {'rgb':rgb,'mask':mask,'node_xy':nodes,'edges':edge,'degrees':degree,
            'active_nodes':degree>0,'type_codes':torch.as_tensor(codes.astype(np.int64),device=device),
            'stroke_radius':width[:,0],'edge_bows':bows,'instance_seeds':seeds.copy()}
