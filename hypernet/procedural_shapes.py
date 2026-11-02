"""Procedural geometric programs: no images, fonts, weights, or external assets."""
from dataclasses import dataclass,asdict,replace
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter
import torch

FAMILIES=('ellipse','rectangle','triangle','star','ring','arc','cross','bar')
RECIPE={'revision':1,'families':list(FAMILIES),'render_size':96,'output_size':32,
        'positive':'same geometric program and texture; independent procedural background/color',
        'negative':'matched positive nuisance; visible contour, notch, gap, position or texture-frequency mutation',
        'negative_pixel_check':'resample mutation until structural-mask or foreground-pattern difference survives 32px rasterization',
        'assets':0,'natural_images':0,'pretrained_generator':False}

@dataclass(frozen=True)
class Program:
    family:int
    x:float
    y:float
    rx:float
    ry:float
    rotation:float
    width:float
    notch:float
    frequency:float
    phase:float
    texture:float

def program(rng):
    return Program(int(rng.integers(8)),float(rng.uniform(43,53)),float(rng.uniform(43,53)),
                   float(rng.uniform(20,35)),float(rng.uniform(20,35)),float(rng.uniform(-45,45)),
                   float(rng.uniform(5,12)),0.,float(rng.uniform(2,7)),float(rng.uniform(0,2*math.pi)),
                   float(rng.uniform(.03,.22)))

def mask_for(p):
    image=Image.new('L',(96,96));d=ImageDraw.Draw(image);box=(p.x-p.rx,p.y-p.ry,p.x+p.rx,p.y+p.ry)
    family=FAMILIES[p.family]
    if family=='ellipse':d.ellipse(box,fill=255)
    elif family=='rectangle':d.rounded_rectangle(box,radius=3,fill=255)
    elif family=='triangle':d.polygon([(p.x,p.y-p.ry),(p.x+p.rx,p.y+p.ry),(p.x-p.rx,p.y+p.ry)],fill=255)
    elif family=='star':
        points=[(p.x+p.rx*(1 if i%2==0 else .45)*math.sin(i*math.pi/5),
                 p.y-p.ry*(1 if i%2==0 else .45)*math.cos(i*math.pi/5)) for i in range(10)]
        d.polygon(points,fill=255)
    elif family in ('ring','arc'):
        width=min(p.width,.9*min(p.rx,p.ry))
        d.ellipse(box,fill=255);d.ellipse((box[0]+width,box[1]+width,box[2]-width,box[3]-width),fill=0)
        if family=='arc':d.rectangle((p.x,p.y-p.width/2,p.x+p.rx+1,p.y+p.width/2),fill=0)
    elif family=='cross':
        d.rectangle((p.x-p.width,p.y-p.ry,p.x+p.width,p.y+p.ry),fill=255)
        d.rectangle((p.x-p.rx,p.y-p.width,p.x+p.rx,p.y+p.width),fill=255)
    else:d.rectangle((p.x-p.rx,p.y-p.width/2,p.x+p.rx,p.y+p.width/2),fill=255)
    if p.notch>0:d.rectangle((p.x-p.notch/2,p.y-p.ry-2,p.x+p.notch/2,p.y),fill=0)
    image=image.rotate(p.rotation,resample=Image.Resampling.BILINEAR)
    return np.asarray(image,dtype=np.float32)/255.

def render(p,nuisance_seed):
    rng=np.random.default_rng(int(nuisance_seed));mask=mask_for(p);yy,xx=np.mgrid[:96,:96]
    bg=rng.uniform(.05,.45,3).astype(np.float32);fg=rng.uniform(.55,.95,3).astype(np.float32)
    if rng.random()<.5:bg,fg=1-bg,1-fg
    smooth=gaussian_filter(rng.standard_normal((96,96)),rng.uniform(1,8))
    smooth=smooth/max(float(smooth.std()),1e-5)
    background=bg[None,None]+rng.uniform(.01,.12)*smooth[:,:,None]
    texture=np.sin(2*math.pi*p.frequency*(xx/96+.35*yy/96)+p.phase)*p.texture
    foreground=fg[None,None]+texture[:,:,None]
    rgb=np.clip(background*(1-mask[:,:,None])+foreground*mask[:,:,None],0,1)
    rgb=np.asarray(Image.fromarray(np.uint8(np.rint(rgb*255))).resize((32,32),Image.Resampling.BILINEAR)).copy()
    small_mask=np.asarray(Image.fromarray(np.uint8(mask*255)).resize((32,32),Image.Resampling.BILINEAR)).copy()/255.
    return rgb,small_mask.astype(np.float32)

def mutate(p,rng,attempt=0):
    choice=int(rng.integers(5))
    if choice==0:return replace(p,notch=float(rng.uniform(4,12))), 'notch'
    if choice==1:return replace(p,frequency=p.frequency+float(rng.uniform(1.5,3))), 'texture_frequency'
    if choice==2:return replace(p,x=p.x+float(rng.choice([-1,1]))*rng.uniform(4,10)), 'part_position'
    if choice==3:return replace(p,width=max(3,p.width+float(rng.choice([-1,1]))*5),ry=p.ry*.75),'thickness'
    return replace(p,family=(p.family+int(rng.integers(1,8)))%8),'contour_family'

def procedural_triplets(batch_size=16,seed=73):
    if batch_size<1:raise ValueError('Positive batch size required')
    rng=np.random.default_rng(int(seed));a=[];p=[];n=[];ma=[];mp=[];mn=[];records=[]
    for i in range(batch_size):
        spec=program(rng);sa=int(rng.integers(2**31));sp=int(rng.integers(2**31))
        ar,am=render(spec,sa);pr,pm=render(spec,sp)
        for attempt in range(30):
            negative,mutation=mutate(spec,rng,attempt);nr,nm=render(negative,sp)
            difference=float(np.abs(nr.astype(float)-pr.astype(float)).mean()/255.)
            if difference>.003:break
        else:raise RuntimeError('Failed to render a visible structural negative')
        a.append(ar);p.append(pr);n.append(nr);ma.append(am);mp.append(pm);mn.append(nm)
        records.append({'program':asdict(spec),'negative':asdict(negative),'mutation':mutation,
                        'reference_nuisance_seed':sa,'query_nuisance_seed':sp,
                        'matched_nuisance_pixel_difference':difference})
    images=torch.from_numpy(np.stack(a+p+n)).permute(0,3,1,2).float()/255.
    masks=torch.from_numpy(np.stack(ma+mp+mn))[:,None]
    return {'images':images,'masks':masks,'records':records,
            'program_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            'rendered_images':3*batch_size,'base_programs':batch_size,'seed':int(seed)}
