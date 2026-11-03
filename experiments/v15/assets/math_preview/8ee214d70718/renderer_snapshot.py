"""Stateless mathematical2D pairs with exact continuous correspondences.

No semantic objects, image assets, pretrained models or image blur. A Gaussian
changes shared function parameters or defines a thresholded anisotropic ridge;
it is never an image smoothing filter. Family count is a parameter-ID count,
not an asserted count of distinct finite-resolution rasters.
"""
from __future__ import annotations
import numpy as np

FAMILY_COUNT=1_048_576
PARAMETER_NAMESPACE_SIZE=2**31
VERSION='hypernet.math_contour_graph_field.v15.1'
GRID_CONVENTION='normalized[-1,1], pixel centers2*(i+.5)/size-1; torch.grid_sample align_corners=False'
GEOMETRY_MODES=('filled_fourier_domain','open_junction_graph','implicit_component_union',
 'nested_contour_loops','intersecting_elliptic_contours','oscillatory_level_sets',
 'anisotropic_ridge_network','contour_graph_component_union')


def _ids(values):
    a=np.asarray(values)
    if a.ndim!=1 or a.dtype.kind not in 'iu' or len(a)<1 or np.any(a<0) or np.any(a>=PARAMETER_NAMESPACE_SIZE):
        raise ValueError('Use integer family IDs0..2^31-1; training namespace is0..1048575')
    return a.astype(np.uint64)


def _validate_seed(seed):
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool) or not 0<=int(seed)<2**32:
        raise ValueError('Seed must be integer0..2^32-1')
    return int(seed)


def _uniform(ids,seed,count):
    """SplitMix64-derived uniforms; invariant to batch ordering/partitioning."""
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool) or not 0<=int(seed)<2**32:
        raise ValueError('Seed must be integer0..2^32-1')
    with np.errstate(over='ignore'):
        z=(ids[:,None]+np.uint64(0x9E3779B97F4A7C15)*(np.arange(count,dtype=np.uint64)[None]+1)
           +np.uint64(seed)*np.uint64(0xD1B54A32D192ED03))
        z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        z=z^(z>>np.uint64(31))
    return ((z>>np.uint64(40)).astype(np.float32)+.5)*(1/16777216)


def family_parameters(family_ids):
    """Fixed continuous family functions with bounded Gaussian parameter variation.

    Family identity is determined only by ID. The sample seed changes views and
    appearance, never canonical geometry or microstructure.
    """
    ids=_ids(family_ids);u=_uniform(ids,1571,52);n=_uniform(ids,15107,12)
    normal=np.clip(np.sqrt(-2*np.log(n[:,:6]))*np.cos(2*np.pi*n[:,6:]),-2,2)
    return dict(family_id=ids.astype(np.int64),
        axes=(.83+.32*u[:,:2]).astype(np.float32),
        harmonics=np.stack([2+(u[:,2]*4).astype(np.int16),6+(u[:,3]*4).astype(np.int16),11+(u[:,4]*7).astype(np.int16)],axis=1),
        phases=(2*np.pi*u[:,5:8]).astype(np.float32),
        amplitudes=(np.array([.075,.045,.025],np.float32)[None]*(.7+.6*u[:,8:11])+.004*normal[:,:3]).astype(np.float32),
        graph_angles=(2*np.pi*u[:,11:15]+.03*normal[:,:4]).astype(np.float32),
        graph_lengths=(.27+.40*u[:,15:19]).astype(np.float32),
        stroke_halfwidth=(.017+.025*u[:,19]).astype(np.float32),
        geometry_mode=(ids%8).astype(np.int64),topology=((ids>>np.uint64(5))%4).astype(np.int64),
        graph_mode=((ids>>np.uint64(3))%4).astype(np.int64),texture_mode=((ids>>np.uint64(7))%4).astype(np.int64),
        texture_frequency=(6+15*u[:,20]).astype(np.float32),
        texture_angle=(np.pi*u[:,21]).astype(np.float32),texture_phase=(2*np.pi*u[:,22]+.06*normal[:,4]).astype(np.float32),
        ridge_center=(.55*(u[:,23:31].reshape(-1,4,2)-.5)+.007*normal[:,:4,None]).astype(np.float32),
        ridge_angle=(np.pi*u[:,31:35]).astype(np.float32),
        ridge_long=(.06+.17*u[:,35:39]).astype(np.float32),
        ridge_short=(.012+.025*u[:,39:43]).astype(np.float32),
        hole_center=(.35*(u[:,43:45]-.5)).astype(np.float32),
        hole_axes=(.055+.10*u[:,45:47]).astype(np.float32),
        feature_phase=(2*np.pi*u[:,47]).astype(np.float32),node_radii=(.065+.11*u[:,48:52]).astype(np.float32),
        shared_gaussian=normal.astype(np.float32))


def _field(x,y,p):
    """Evaluate exact canonical functions at arbitrary per-family coordinates."""
    def scalar(name):return p[name][:,None,None]
    xx=x/p['axes'][:,0,None,None];yy=y/p['axes'][:,1,None,None]
    theta=np.arctan2(yy,xx);radius=np.sqrt(xx*xx+yy*yy)
    boundary=np.full_like(radius,.53)
    for k in range(3):
        boundary+=p['amplitudes'][:,k,None,None]*np.cos(p['harmonics'][:,k,None,None]*theta+p['phases'][:,k,None,None])
    disk=radius<boundary
    # Four segment primitives form stars, chains, crossings or bent junctions.
    angles=p['graph_angles'];lengths=p['graph_lengths'];gx=lengths*np.cos(angles);gy=lengths*np.sin(angles)
    mode=scalar('graph_mode');stroke=np.zeros_like(disk);components=np.zeros_like(disk)
    for k in range(4):
        previous=(k-1)%4
        startx=np.where(p['graph_mode']==1,gx[:,previous],0)[:,None,None]
        starty=np.where(p['graph_mode']==1,gy[:,previous],0)[:,None,None]
        endx=gx[:,k,None,None];endy=gy[:,k,None,None]
        dx=endx-startx;dy=endy-starty
        t=np.clip(((x-startx)*dx+(y-starty)*dy)/(dx*dx+dy*dy+1e-9),0,1)
        distance2=(x-startx-t*dx)**2+(y-starty-t*dy)**2
        active=(mode!=2)|(k%2==0)
        stroke|=(distance2<scalar('stroke_halfwidth')**2)&active
        rr=p['node_radii'][:,k,None,None]
        components|=((x-endx)/rr)**2+((y-endy)/(rr*.8))**2<1
    hx=x-p['hole_center'][:,0,None,None];hy=y-p['hole_center'][:,1,None,None]
    hole=(hx/p['hole_axes'][:,0,None,None])**2+(hy/p['hole_axes'][:,1,None,None])**2<1
    hole2=((x+.18)/(.045+scalar('stroke_halfwidth')))**2+((y-.12)/(.05+scalar('stroke_halfwidth')))**2<1
    topology=scalar('topology')
    angle=scalar('texture_angle');u=x*np.cos(angle)+y*np.sin(angle);v=-x*np.sin(angle)+y*np.cos(angle)
    frequency=scalar('texture_frequency');phase=scalar('texture_phase')
    wave=np.sin(np.pi*frequency*u+phase);cross=np.sin(np.pi*(frequency*.83)*v-phase)
    texture_mode=scalar('texture_mode')
    pattern=np.where(texture_mode==0,wave>.35,np.where(texture_mode==1,(wave>.4)&(cross>.4),
             np.where(texture_mode==2,wave*cross>.35,np.sin(np.pi*frequency*(u+.18*np.sin(5*v))+phase)>.45)))
    ridges=np.zeros_like(disk)
    for k in range(4):
        dx=x-p['ridge_center'][:,k,0,None,None];dy=y-p['ridge_center'][:,k,1,None,None]
        ca=np.cos(p['ridge_angle'][:,k,None,None]);sa=np.sin(p['ridge_angle'][:,k,None,None])
        along=(dx*ca+dy*sa)/p['ridge_long'][:,k,None,None]
        across=(-dx*sa+dy*ca)/p['ridge_short'][:,k,None,None]
        # exp(-.5*(along^2+across^2))>exp(-.5): sharp ellipse/ridge level set.
        ridges^=(along*along+across*across)<1
    halfwidth=scalar('stroke_halfwidth');geometry=scalar('geometry_mode')
    loops=(np.abs(radius-boundary)<halfwidth)|(np.abs(radius-boundary*.59)<halfwidth*.7)
    elliptic1=np.sqrt(((x+.16)/.50)**2+((y-.06)/.30)**2)
    elliptic2=np.sqrt(((x-.14)/.32)**2+((y+.08)/.55)**2)
    intersecting=(np.abs(elliptic1-1)<halfwidth*2)|(np.abs(elliptic2-1)<halfwidth*2)
    labyrinth=(np.sin(11*xx+phase)+np.cos(13*yy-phase)>.25)&(radius<.68)
    mask=np.where(geometry==0,disk|stroke,np.where(geometry==1,stroke,
         np.where(geometry==2,components,np.where(geometry==3,loops,
         np.where(geometry==4,intersecting,np.where(geometry==5,labyrinth,
         np.where(geometry==6,ridges|stroke,loops|stroke|components)))))))
    mask&=~((topology>=1)&hole);mask&=~((topology>=2)&hole2)
    # A slit is an actual topological shape change, not a rendered overlay.
    mask&=~((topology==3)&(np.abs(y+.21)<halfwidth*.60)&(x>.10)&(x<.48))
    # Four crisp material levels preserve localized ridge/texture interactions.
    material=(.28+.36*pattern+.28*ridges+.08*stroke).astype(np.float32)
    material=np.where(mask,material,0).astype(np.float32)
    return mask,material


def _transforms(ids,seed):
    u=_uniform(ids,int(seed)^23971,24).reshape(-1,2,12);out=[]
    for view in range(2):
        a=u[:,view];angle=(a[:,0]-.5)*.8;scale=.64+.48*a[:,1]
        aspect=.90+.20*a[:,2];shear=(a[:,3]-.5)*.18
        flip=np.where(a[:,4]>.5,-1.,1.).astype(np.float32)
        ca=np.cos(angle);sa=np.sin(angle)
        matrix=np.zeros((len(ids),2,3),np.float32)
        matrix[:,0,0]=ca*scale*aspect*flip;matrix[:,1,0]=sa*scale*aspect*flip
        matrix[:,0,1]=scale*(ca*shear-sa)/aspect;matrix[:,1,1]=scale*(sa*shear+ca)/aspect
        matrix[:,:,2]=(a[:,5:7]-.5)*.64
        out.append(matrix)
    return out


def _invert(matrix):
    a,b=matrix[:,0,0],matrix[:,0,1];c,d=matrix[:,1,0],matrix[:,1,1];det=a*d-b*c
    inverse=np.empty_like(matrix);inverse[:,0,0]=d/det;inverse[:,0,1]=-b/det
    inverse[:,1,0]=-c/det;inverse[:,1,1]=a/det
    inverse[:,:,2]=-np.einsum('bij,bj->bi',inverse[:,:,:2],matrix[:,:,2])
    return inverse


def _apply(matrix,x,y):
    return (matrix[:,0,0,None,None]*x+matrix[:,0,1,None,None]*y+matrix[:,0,2,None,None],
            matrix[:,1,0,None,None]*x+matrix[:,1,1,None,None]*y+matrix[:,1,2,None,None])


def _appearance(ids,seed,view,x,y,mask,material):
    u=_uniform(ids,((int(seed)^35119)+view)&0xFFFFFFFF,16)
    background=(.10+.75*u[:,:3])[:,None,None,:]
    foreground=(.12+.76*u[:,3:6])[:,None,None,:]
    # Force luminance contrast without tying any color to family identity.
    delta=np.abs(background.mean(-1)-foreground.mean(-1))<.18
    foreground=np.where(delta[...,None],1-background,foreground)
    phase=2*np.pi*u[:,6,None,None]
    bg_pattern=.09*np.sin(3.3*x+2.7*y+phase)+.06*np.cos(7.1*x-4.3*y-phase)
    bg=background+bg_pattern[...,None]
    clutter=_uniform(ids,((int(seed)^45159)+view)&0xFFFFFFFF,27).reshape(-1,3,9)
    for k in range(3):
        q=clutter[:,k];cx=(q[:,0]-.5)[:,None,None]*1.7;cy=(q[:,1]-.5)[:,None,None]*1.7
        dx=x-cx;dy=y-cy;ca=np.cos(q[:,2,None,None]*np.pi);sa=np.sin(q[:,2,None,None]*np.pi)
        a=(dx*ca+dy*sa)/(.09+.30*q[:,3,None,None]);b=(-dx*sa+dy*ca)/(.05+.19*q[:,4,None,None])
        distance=a*a+b*b;shape=(distance<1)&((q[:,5,None,None]>.5)|(distance>.40))
        fill=q[:,6:9,None,None].transpose(0,2,3,1)*.75+.10
        stripes=np.where(np.sin(27*(dx+dy)+phase)>0,.07,-.07)
        bg=np.where(shape[...,None],fill+stripes[...,None],bg)
    light=.90+.12*x+.08*y
    obj=(.16+.84*foreground)*(.30+.70*material[...,None])*light[...,None]
    rgb=np.where(mask[...,None],obj,bg)
    return np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)


def _detail(material,mask):
    differences=np.zeros_like(mask)
    differences[:,1:]|=np.abs(material[:,1:]-material[:,:-1])>.12
    differences[:,:-1]|=np.abs(material[:,1:]-material[:,:-1])>.12
    differences[:,:,1:]|=np.abs(material[:,:,1:]-material[:,:,:-1])>.12
    differences[:,:,:-1]|=np.abs(material[:,:,1:]-material[:,:,:-1])>.12
    return differences&mask


def sample_batch(family_ids,seed=0,size=64):
    """Native sharp pairs, both-direction normalized grids and foreground masks.

    Returned grids map a SOURCE pixel center to its TARGET sampling location.
    They are directly compatible with torch.grid_sample(align_corners=False).
    valid masks require source foreground and an in-frame target center. The
    exact continuous correspondence can lie between raster samples; boundary
    interpolation and subpixel disappearance remain measurable limitations.
    """
    ids=_ids(family_ids);seed=_validate_seed(seed)
    if isinstance(size,bool) or not isinstance(size,int) or not 32<=size<=256:raise ValueError('Native size must be integer32..256')
    p=family_parameters(ids);coords=(2*(np.arange(size,dtype=np.float32)+.5)/size-1)
    yy,xx=np.meshgrid(coords,coords,indexing='ij');x=xx[None];y=yy[None]
    transforms=_transforms(ids,seed);records=[]
    for view,matrix in enumerate(transforms):
        cx,cy=_apply(_invert(matrix),x,y);mask,material=_field(cx,cy,p)
        records.append((cx,cy,mask,material,_appearance(ids,seed,view,x,y,mask,material)))
    canonical_mask,canonical_material=_field(np.broadcast_to(x,(len(ids),size,size)),np.broadcast_to(y,(len(ids),size,size)),p)
    output=dict(view_a=records[0][4],view_b=records[1][4],mask_a=records[0][2],mask_b=records[1][2],
        detail_a=_detail(records[0][3],records[0][2]),detail_b=_detail(records[1][3],records[1][2]),
        canonical_raster=np.rint(canonical_material*255).astype(np.uint8),canonical_mask=canonical_mask,
        canonical_to_view_a=transforms[0],canonical_to_view_b=transforms[1],parameters=p,
        family_ids=ids.astype(np.int64),seed=int(seed),size=size,grid_convention=GRID_CONVENTION,version=VERSION)
    bound=1-1/size
    for source,target,name in [(0,1,'a_to_b'),(1,0,'b_to_a')]:
        tx,ty=_apply(transforms[target],records[source][0],records[source][1])
        output['grid_'+name]=np.stack([tx,ty],axis=-1).astype(np.float32)
        output['valid_'+name]=records[source][2]&(np.abs(tx)<=bound)&(np.abs(ty)<=bound)
    return output


def render_pair(family_id,pair_seed=0,size=64):
    """Convenience single-family view of the same batched implementation."""
    batch=sample_batch(np.array([family_id],np.int64),pair_seed,size)
    return {key:({k:v[0] for k,v in value.items()} if key=='parameters' else value[0] if isinstance(value,np.ndarray) else value)
            for key,value in batch.items()}
